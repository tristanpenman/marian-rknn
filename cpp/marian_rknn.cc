// Copyright (c) 2023 Rockchip Electronics Co., Ltd. All Rights Reserved.
// Copyright (c) 2025 Tristan Penman
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cmath>

// external
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <sentencepiece_processor.h>

// thirdparty
#include "easy_timer.h"
#include "rknn_api.h"

// internal
#include "file_utils.h"
#include "marian_rknn.h"
#include "rknn_utils.h"
#include "type_half.h"

#define EMBEDDING_DIM 512
#define DECODER_LAYER_NUM 6
#define MAX_SENTENCE_LEN 32
#define MAX_WORD_NUM_IN_SENTENCE 64
#define MAX_WORD_LEN 64
#define VERBOSE 0

// encoder input
#define ENC_IN_INPUT_IDS_IDX 0
#define ENC_IN_ATTENTION_MASK_IDX 1

// encoder output
#define ENC_OUT_ENCODER_HIDDEN_STATES 0

// decoder input
#define DEC_IN_INPUT_IDS_IDX 0
#define DEC_IN_ATTENTION_MASK_IDX 1
#define DEC_IN_ENCODER_HIDDEN_STATES 2

// decoder output
#define DEC_OUT_DECODER_OUTPUT 0

void rknn_marian_lm_head_t::operator()(const float* hidden, float* out_logits) const
{
    using RowMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    Eigen::Map<const Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>> h(hidden, D);
    Eigen::Map<const RowMat> W(Wt, D, V);
    Eigen::Map<Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>> y(out_logits, V);
    Eigen::Map<const Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>> bias(b, V);

    y.noalias() = h * W;
    y += bias;
}

int rknn_nmt_process(
    rknn_marian_rknn_context_t* app_ctx,
    int32_t* input_token,
    int32_t* output_token)
{
    int ret = 0;

    TIMER timer;
    TIMER timer_total;

    // encoder attention mask
    int32_t enc_mask[app_ctx->enc_len];
    memset(enc_mask, 0x00, sizeof(int32_t) * app_ctx->enc_len);

    // decoder attention mask
    int32_t dec_mask[app_ctx->dec_len];
    memset(dec_mask, 0x00, sizeof(int32_t) * app_ctx->dec_len);

    // count tokens
    int input_token_give = 0;
    for (int i=0; i<app_ctx->enc_len; i++) {
        if (input_token[i] <= 0 || input_token[i] == app_ctx->pad_token_id) {
            break;
        }
        input_token_give++;
    }

    // replace trailing tokens with eos, then pad tokens
    printf("--> tokens given (%d): ", input_token_give);
    int32_t input_token_sorted[app_ctx->enc_len];
    for (int i = 0; i < app_ctx->enc_len; i++) {
        if (i < input_token_give) {
            // copy original token
            input_token_sorted[i] = input_token[i];
        } else if (i == input_token_give) {
            // terminate with <eos>
            input_token_sorted[i] = app_ctx->eos_token_id;
        } else {
            // all other characters are <pad> tokens
            input_token_sorted[i] = app_ctx->pad_token_id;
        }
        printf(" %d", input_token_sorted[i]);
    }
    printf("\n");

    // attention mask includes 1s for kept tokens, 0s for masked tokens
    printf("--> generate encoder mask: ");
    bool padding = false;
    for (int i = 0; i < app_ctx->enc_len; i++) {
        if (padding) {
            enc_mask[i] = 0;
        } else {
            enc_mask[i] = 1;
            if (input_token_sorted[i] == app_ctx->eos_token_id) {
                padding = true;
            }
        }
        printf(" %d", enc_mask[i]);
    }
    printf("\n");

    printf("--> copy input ids to encoder\n");
    memcpy(
        app_ctx->enc.input_mem[ENC_IN_INPUT_IDS_IDX]->virt_addr,
        input_token_sorted,
        app_ctx->enc.in_attr[ENC_IN_INPUT_IDS_IDX].size
    );

    printf("--> copy mask to encoder\n");
    memcpy(
        app_ctx->enc.input_mem[ENC_IN_ATTENTION_MASK_IDX]->virt_addr,
        enc_mask,
        app_ctx->enc.in_attr[ENC_IN_ATTENTION_MASK_IDX].size
    );

    // Run encoder
    timer.tik();
    ret = rknn_run(app_ctx->enc.ctx, nullptr);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }
    timer.tok();
    timer.print_time("rknn encoder run");

    for (int i = 0; i < app_ctx->dec_len; i++) {
        output_token[i] = app_ctx->pad_token_id;
    }

    printf("--> reset decoder input and output mem\n");
    for (int input_index = 0; input_index < app_ctx->dec.n_input; input_index++) {
        memset(app_ctx->dec.input_mem[input_index]->virt_addr, 0, app_ctx->dec.in_attr[input_index].size);
    }
    for (int output_index = 0; output_index < app_ctx->dec.n_output; output_index++) {
        memset(app_ctx->dec.output_mem[output_index]->virt_addr, 0, app_ctx->dec.out_attr[output_index].size);
    }

    printf("--> copy output from encoder to decoder\n");
    memcpy(
        app_ctx->dec.input_mem[DEC_IN_ENCODER_HIDDEN_STATES]->virt_addr,
        app_ctx->enc.output_mem[ENC_OUT_ENCODER_HIDDEN_STATES]->virt_addr,
        app_ctx->enc.out_attr[ENC_OUT_ENCODER_HIDDEN_STATES].size
    );

    printf("--> setup decoder input state\n");
    int32_t decoder_input_ids[app_ctx->dec_len];
    memset(decoder_input_ids, 0, sizeof(int32_t) * app_ctx->dec_len);

    // decoder start token ID
    decoder_input_ids[0] = app_ctx->decoder_start_token_id;
    for (int i = 1; i < app_ctx->dec_len; i++) {
        decoder_input_ids[i] = app_ctx->pad_token_id;
    }

    timer_total.tik();
    for (int num_iter = 0; num_iter < app_ctx->dec_len - 1; num_iter++) {
        printf("--> decoder iteration %d\n", num_iter);
        memcpy(
            app_ctx->dec.input_mem[DEC_IN_INPUT_IDS_IDX]->virt_addr,
            decoder_input_ids,
            app_ctx->dec.in_attr[DEC_IN_INPUT_IDS_IDX].size
        );

        printf("--> generate decoder mask: ");
        for (int j = 0; j < app_ctx->dec_len; j++) {
            if (j > num_iter || decoder_input_ids[j] == app_ctx->pad_token_id) {
                dec_mask[j] = 0;
            } else {
                dec_mask[j] = 1;
            }
            printf(" %d", dec_mask[j]);
        }
        printf("\n");

        printf("--> copy mask to decoder\n");
        memcpy(
            app_ctx->dec.input_mem[DEC_IN_ATTENTION_MASK_IDX]->virt_addr,
            dec_mask,
            app_ctx->dec.in_attr[DEC_IN_ATTENTION_MASK_IDX].size
        );

        printf("--> rknn_run\n");
        timer.tik();
        ret = rknn_run(app_ctx->dec.ctx, nullptr);
        timer.tok();
        if (ret < 0) {
            printf("rknn_run fail! ret=%d\n", ret);
            return -1;
        }

        // convert fp16 to fp32
        half* ptr = (half*)(app_ctx->dec.output_mem[DEC_OUT_DECODER_OUTPUT]->virt_addr);
        std::vector<float> output_floats(app_ctx->lm_head.D, 0);
        for (int j = 0; j < app_ctx->lm_head.D; j++) {
            output_floats[j] = half_to_float(ptr[app_ctx->lm_head.D * num_iter + j]);
            printf("%f ", output_floats[j]);
        }
        printf("\n");

        printf("--> apply lm_head\n");
        std::vector<float> logits;
        logits.resize(app_ctx->lm_head.V);
        app_ctx->lm_head(
            output_floats.data(),
            logits.data()
        );

        printf("--> argmax: ");
        int max = 0;
        float value = -INFINITY;
        for (int i = 0; i < app_ctx->lm_head.V; i++) {
            if (logits[i] > value) {
                value = logits[i];
                max = i;
            }
        }

        printf("%d (%f)\n", max, value);

        output_token[num_iter] = max;

        if (num_iter < app_ctx->dec_len - 1) {
            decoder_input_ids[num_iter + 1] = max;
        }

        if (max == app_ctx->eos_token_id) {
            break;
        }
    }
    timer_total.tok();

    // for debug
    int output_len=0;
    printf("decoder output token: ");
    for (int i = 0; i < app_ctx->dec_len; i++) {
        if (output_token[i] == app_ctx->eos_token_id || output_token[i] == app_ctx->pad_token_id) {
            break;
        }
        printf("%d ", output_token[i]);
        output_len++;
    }
    printf("\n");

    timer.print_time("rknn decoder once run");
    printf("decoder run %d times. ", output_len-1);
    timer_total.print_time("cost");

    return output_len;
}

int init_marian_rknn_model(
    const char* encoder_path,
    const char* decoder_path,
    const char* source_spm_path,
    const char* target_spm_path,
    const char* vocab_path,
    const char* lm_weight_path,
    const char* lm_bias_path,
    rknn_marian_rknn_context_t* app_ctx)
{
    int ret = 0;

    printf("--> init rknn encoder %s\n", encoder_path);
    app_ctx->enc.m_path = encoder_path;
    app_ctx->enc.verbose_log = VERBOSE;
    ret = rknn_utils_init(&app_ctx->enc);
    if (ret != 0) {
        printf("rknn_utils_init ret=%d\n", ret);
        return -1;
    }

    printf("--> init rknn decoder %s\n", decoder_path);
    app_ctx->dec.m_path = decoder_path;
    app_ctx->dec.verbose_log = VERBOSE;
    ret = rknn_utils_init(&app_ctx->dec);
    if (ret != 0) {
        printf("rknn_utils_init ret=%d\n", ret);
        return -1;
    }

    app_ctx->enc_len = app_ctx->enc.in_attr[ENC_IN_INPUT_IDS_IDX].dims[1];
    printf("--> enc len: %d\n", app_ctx->enc_len);

    app_ctx->dec_len = app_ctx->dec.in_attr[DEC_IN_ATTENTION_MASK_IDX].dims[1];
    printf("--> dec len: %d\n", app_ctx->dec_len);

    printf("--> init encoder buffers\n");
    ret = rknn_utils_init_input_buffer_all(&app_ctx->enc, ZERO_COPY_API);
    if (ret != 0) {
        printf("rknn_utils_init_input_buffer_all ret=%d\n", ret);
        return -1;
    }

    ret = rknn_utils_init_output_buffer_all(&app_ctx->enc, ZERO_COPY_API);
    if (ret != 0) {
        printf("rknn_utils_init_output_buffer_all ret=%d\n", ret);
        return -1;
    }

    printf("--> init decoder buffers\n");
    ret = rknn_utils_init_input_buffer_all(&app_ctx->dec, ZERO_COPY_API);
    if (ret != 0) {
        printf("rknn_utils_init_input_buffer_all ret=%d\n", ret);
        return -1;
    }

    ret = rknn_utils_init_output_buffer_all(&app_ctx->dec, ZERO_COPY_API);
    if (ret != 0) {
        printf("rknn_utils_init_output_buffer_all ret=%d\n", ret);
        return -1;
    }

    printf("--> rknn_set_io_mem enc inputs; n_input=%u\n", app_ctx->enc.n_input);
    for (int input_index = 0; input_index < app_ctx->enc.n_input; input_index++) {
        ret = rknn_set_io_mem(app_ctx->enc.ctx, app_ctx->enc.input_mem[input_index], &(app_ctx->enc.in_attr[input_index]));
        if (ret < 0) {
            printf("rknn_set_io_mem fail! ret=%d\n", ret);
            return -1;
        }
    }

    printf("--> rknn_set_io_mem enc outputs; n_output=%u\n", app_ctx->enc.n_output);
    for (int output_index=0; output_index < app_ctx->enc.n_output; output_index++) {
        ret = rknn_set_io_mem(app_ctx->enc.ctx, app_ctx->enc.output_mem[output_index], &(app_ctx->enc.out_attr[output_index]));
        if (ret < 0) {
            printf("rknn_set_io_mem fail! ret=%d\n", ret);
            return -1;
        }
    }

    printf("--> rknn_set_io_mem dec inputs; n_input=%u\n", app_ctx->dec.n_input);
    for (int input_index=0; input_index< app_ctx->dec.n_input; input_index++) {
        if (app_ctx->dec.in_attr[input_index].fmt == RKNN_TENSOR_NHWC) {
            rknn_query(app_ctx->dec.ctx, RKNN_QUERY_NATIVE_NC1HWC2_INPUT_ATTR, &(app_ctx->dec.in_attr[input_index]), sizeof(app_ctx->dec.in_attr[input_index]));
            app_ctx->dec.input_mem[input_index] = rknn_create_mem(app_ctx->dec.ctx, app_ctx->dec.in_attr[input_index].n_elems * sizeof(float)*2);
            app_ctx->dec.in_attr[input_index].pass_through = 1;
        }
        ret = rknn_set_io_mem(app_ctx->dec.ctx, app_ctx->dec.input_mem[input_index], &(app_ctx->dec.in_attr[input_index]));
    }

    printf("--> rknn_set_io_mem dec outputs; n_output=%u\n", app_ctx->dec.n_output);
    for (int output_index=0; output_index< app_ctx->dec.n_output; output_index++) {
        if (app_ctx->dec.out_attr[output_index].fmt == RKNN_TENSOR_NCHW) {
            rknn_query(app_ctx->dec.ctx, RKNN_QUERY_NATIVE_NC1HWC2_OUTPUT_ATTR, &(app_ctx->dec.out_attr[output_index]), sizeof(app_ctx->dec.out_attr[output_index]));
            rknn_destroy_mem(app_ctx->dec.ctx, app_ctx->dec.output_mem[output_index]);
            app_ctx->dec.output_mem[output_index] = rknn_create_mem(app_ctx->dec.ctx, app_ctx->dec.out_attr[output_index].n_elems * sizeof(float)*2);
        }
        ret = rknn_set_io_mem(app_ctx->dec.ctx, app_ctx->dec.output_mem[output_index], &(app_ctx->dec.out_attr[output_index]));
    }

    printf("--> loading source spm\n");
    auto src_status = app_ctx->spm_src.Load(source_spm_path);
    if (!src_status.ok()) {
        printf("Failed to load source sentencepiece model: %s\n", src_status.ToString().c_str());
        return -1;
    }

    auto ps = app_ctx->spm_src.GetPieceSize();
    printf("--> source piece size: %d\n", ps);

    printf("--> loading target spm\n");
    auto tgt_status = app_ctx->spm_tgt.Load(target_spm_path);
    if (!tgt_status.ok()) {
        printf("Failed to load target sentencepiece model: %s\n", tgt_status.ToString().c_str());
        return -1;
    }

    ps = app_ctx->spm_tgt.GetPieceSize();
    printf("--> target piece size: %d\n", ps);

    int D = app_ctx->lm_head.D = 512;
    int V = app_ctx->lm_head.V = 59514;

    printf("--> load lm weight\n");
    app_ctx->lm_head.Wt = (float*)(malloc(sizeof(float) * V * D));
    read_fp32_from_file(lm_weight_path, V * D, app_ctx->lm_head.Wt);

    printf("--> load lm bias\n");
    app_ctx->lm_head.b = (float*)(malloc(sizeof(float) * V));
    read_fp32_from_file(lm_bias_path, V, app_ctx->lm_head.b);

    printf("--> load vocab\n");
    read_map_from_file(vocab_path, app_ctx->vocab);

    printf("--> invert vocab\n");
    app_ctx->vocab_inv.reserve(app_ctx->vocab.size());
    for (auto entry : app_ctx->vocab) {
        auto existing = app_ctx->vocab_inv.find(entry.second);
        if (existing != app_ctx->vocab_inv.end()) {
            printf("Vocab is not unique. Duplicate found on ID: %d\n", entry.second);
            return -1;
        }

        app_ctx->vocab_inv.emplace(entry.second, entry.first);
    }

    // TODO: Read these from config file

    app_ctx->decoder_start_token_id = 59513;
    printf("--> decoder start token id: %d\n", app_ctx->decoder_start_token_id);

    app_ctx->pad_token_id = 59513;
    printf("--> pad token id: %d\n", app_ctx->pad_token_id);

    app_ctx->eos_token_id = 0;
    printf("--> eos token id: %d\n", app_ctx->eos_token_id);

    app_ctx->bos_token_id = 0;
    printf("--> bos token id: %d\n", app_ctx->bos_token_id);

    return 0;
}

int release_marian_rknn_model(rknn_marian_rknn_context_t* app_ctx)
{
    rknn_utils_release(&app_ctx->enc);
    rknn_utils_release(&app_ctx->dec);

    free(app_ctx->lm_head.Wt);
    free(app_ctx->lm_head.b);

    return 0;
}

int inference_marian_rknn_model(
    rknn_marian_rknn_context_t* app_ctx,
    const char* input_sentence,
    char* output_sentence)
{
    // TODO: Remove magic number
    int token_list[100];
    int token_list_len = 0;
    memset(token_list, 0, sizeof(token_list));

    // encode tokens
    std::vector<int> encoded_tokens;
    auto pieces = app_ctx->spm_src.EncodeAsPieces(std::string(input_sentence));
    printf("--> sentence pieces:");
    for (auto piece : pieces) {
        printf(" %s", piece.c_str());
    }
    printf("\n");

    // apply vocab mapping
    printf("--> apply vocab mapping\n");
    encoded_tokens.reserve(pieces.size());
    for (auto piece : pieces) {
        auto itr = app_ctx->vocab.find(piece);
        if (itr == app_ctx->vocab.end()) {
            // unknown token
            encoded_tokens.push_back(0);
        } else {
            encoded_tokens.push_back(itr->second);
        }
    }

    // copy and truncate tokens
    token_list_len = encoded_tokens.size();
    if (token_list_len > (int)(sizeof(token_list) / sizeof(int))) {
        printf("WARNING: too many tokens (%d), truncating to %lu\n", token_list_len, sizeof(token_list)/sizeof(int));
        token_list_len = sizeof(token_list) / sizeof(int);
    }
    for (int i = 0; i < token_list_len; ++i) {
        token_list[i] = encoded_tokens[i];
    }

    // check input length
    int max_input_len = app_ctx->enc_len;
    if (token_list_len > max_input_len) {
        printf("\nWARNING: token_len(%d) > max_input_len(%d), only keep %d tokens!\n", token_list_len, max_input_len, max_input_len);
        printf("Tokens all     :");
        for (int i = 0; i < token_list_len; i++) {
            printf(" %d", token_list[i]);
        }
        printf("\n");
        token_list_len = max_input_len;
        printf("Tokens remains :");
        for (int i = 0; i < token_list_len; i++) {
            printf(" %d", token_list[i]);
        }
        printf("\n");
    }

    // run model
    std::vector<int32_t> output_token(app_ctx->dec_len, 0);
    int output_len = 0;
    output_len = rknn_nmt_process(app_ctx, token_list, output_token.data());

    // prepare tokens for decode
    printf("--> reverse vocab mapping\n");
    std::vector<std::string> decode_tokens;
    for (int i = 0; i < output_len; ++i) {
        if (output_token[i] == app_ctx->eos_token_id || output_token[i] == app_ctx->pad_token_id || output_token[i] <= 0) {
            break;
        }
        auto entry = app_ctx->vocab_inv.find(output_token[i]);
        if (entry == app_ctx->vocab_inv.end()) {
            printf("Warning: token not found: %d\n", output_token[i]);
        } else {
            decode_tokens.push_back(entry->second);
        }
    }

    // decode tokens
    std::string decoded;
    auto status = app_ctx->spm_tgt.Decode(decode_tokens, &decoded);
    if (!status.ok()) {
        printf("sentencepiece decode failed: %s\n", status.ToString().c_str());
        return -1;
    }

    // copy output sentence
    memset(output_sentence, 0, MAX_USER_INPUT_LEN);
    strncpy(output_sentence, decoded.c_str(), MAX_USER_INPUT_LEN-1);

    return 0;
}
