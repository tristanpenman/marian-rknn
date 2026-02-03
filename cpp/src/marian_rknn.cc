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
#include <fstream>
#include <sstream>
#include <vector>

// external
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <sentencepiece_processor.h>

// thirdparty
#include "rknn_api.h"

// internal
#include "easy_timer.h"
#include "file_utils.h"
#include "logger.h"
#include "marian_rknn.h"
#include "rknn_utils.h"
#include "type_half.h"

using json = nlohmann::json;

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

static int greedy_decode(
    rknn_marian_rknn_context_t* app_ctx,
    int32_t* output_token)
{
    int ret = 0;

    LOG(VERBOSE) << "Setup decoder input state";
    std::vector<int32_t> decoder_input_ids(static_cast<size_t>(app_ctx->dec_len), 0);
    decoder_input_ids[0] = app_ctx->decoder_start_token_id;
    for (int i = 1; i < app_ctx->dec_len; i++) {
        decoder_input_ids[i] = app_ctx->pad_token_id;
    }

    // output starts with pad token
    for (int i = 0; i < app_ctx->dec_len; i++) {
        output_token[i] = app_ctx->pad_token_id;
    }

    TIMER timer;
    TIMER timer_total;
    timer_total.tik();
    for (int num_iter = 0; num_iter < app_ctx->dec_len - 1; num_iter++) {
        LOG(VERBOSE) << "Decoder iteration " << num_iter;
        memcpy(
            app_ctx->dec.input_mem[DEC_IN_INPUT_IDS_IDX]->virt_addr,
            decoder_input_ids.data(),
            app_ctx->dec.in_attr[DEC_IN_INPUT_IDS_IDX].size
        );

        LOG(VERBOSE) << "rknn_run";
        timer.tik();
        ret = rknn_run(app_ctx->dec.ctx, nullptr);
        timer.tok();
        if (ret < 0) {
            LOG(ERROR) << "rknn_run failed. ret=" << ret;
            return -1;
        }

        LOG(VERBOSE) << "Convert fp16 to fp32";
        half* ptr = (half*)(app_ctx->dec.output_mem[DEC_OUT_DECODER_OUTPUT]->virt_addr);
        std::vector<float> output_floats(app_ctx->lm_head.D, 0);
        for (int j = 0; j < app_ctx->lm_head.D; j++) {
            output_floats[j] = half_to_float(ptr[app_ctx->lm_head.D * num_iter + j]);
        }

        LOG(VERBOSE) << "Apply LM head";
        std::vector<float> logits;
        logits.resize(app_ctx->lm_head.V);
        app_ctx->lm_head(
            output_floats.data(),
            logits.data()
        );

        LOG(VERBOSE) << "Argmax:";
        int max = 0;
        float value = -INFINITY;
        for (int i = 0; i < app_ctx->lm_head.V; i++) {
            if (logits[i] > value) {
                value = logits[i];
                max = i;
            }
        }

        LOG(VERBOSE) << max << " (" << value << ")";
        output_token[num_iter] = max;

        if (num_iter < app_ctx->dec_len - 1) {
            decoder_input_ids[num_iter + 1] = max;
        }

        if (max == app_ctx->eos_token_id) {
            break;
        }
    }
    timer_total.tok();

    int output_len = 0;
    std::ostringstream output_stream;
    output_stream << "Decoder output tokens:";
    for (int i = 0; i < app_ctx->dec_len; i++) {
        if (output_token[i] == app_ctx->eos_token_id || output_token[i] == app_ctx->pad_token_id) {
            break;
        }
        output_stream << " " << output_token[i];
        output_len++;
    }
    LOG(VERBOSE) << output_stream.str();

    timer.print_time("RKNN decoder once run");

    LOG(VERBOSE) << "Decoder run " << output_len - 1 << " times";
    timer_total.print_time("Total time");

    return output_len;
}

static int rknn_nmt_process(
    rknn_marian_rknn_context_t* app_ctx,
    const int32_t* input_token,
    int32_t* output_token)
{
    int ret = 0;

    // attention mask
    std::vector<int32_t> attention_mask;
    attention_mask.resize(app_ctx->enc_len, 0);

    // count tokens
    int input_token_give = 0;
    for (int i=0; i<app_ctx->enc_len; i++) {
        if (input_token[i] <= 0 || input_token[i] == app_ctx->pad_token_id) {
            break;
        }
        input_token_give++;
    }

    // replace trailing tokens with eos, then pad tokens
    LOG(VERBOSE) << "Tokens given: " << input_token_give;
    std::vector<int32_t> input_token_sorted;
    input_token_sorted.resize(app_ctx->enc_len, 0);
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
    }

    // attention mask includes 1s for kept tokens, 0s for masked tokens
    std::ostringstream mask_stream;
    mask_stream << "Generate attention mask:";
    bool padding = false;
    for (int i = 0; i < app_ctx->enc_len; i++) {
        if (padding) {
            attention_mask[i] = 0;
        } else {
            attention_mask[i] = 1;
            if (input_token_sorted[i] == app_ctx->eos_token_id) {
                padding = true;
            }
        }
        mask_stream << " " << attention_mask[i];
    }
    LOG(VERBOSE) << mask_stream.str();

    LOG(VERBOSE) << "Copy input ids to encoder";
    memcpy(
        app_ctx->enc.input_mem[ENC_IN_INPUT_IDS_IDX]->virt_addr,
        input_token_sorted.data(),
        app_ctx->enc.in_attr[ENC_IN_INPUT_IDS_IDX].size
    );

    LOG(VERBOSE) << "Copy mask to encoder";
    memcpy(
        app_ctx->enc.input_mem[ENC_IN_ATTENTION_MASK_IDX]->virt_addr,
        attention_mask.data(),
        app_ctx->enc.in_attr[ENC_IN_ATTENTION_MASK_IDX].size
    );

    LOG(VERBOSE) << "Run encoder";
    TIMER timer;
    timer.tik();
    ret = rknn_run(app_ctx->enc.ctx, nullptr);
    if (ret < 0) {
        LOG(ERROR) << "rknn_run failed. ret=" << ret;
        return -1;
    }
    timer.tok();
    timer.print_time("RKNN encoder run");

    LOG(VERBOSE) << "Copy output from encoder to decoder";
    memcpy(
        app_ctx->dec.input_mem[DEC_IN_ENCODER_HIDDEN_STATES]->virt_addr,
        app_ctx->enc.output_mem[ENC_OUT_ENCODER_HIDDEN_STATES]->virt_addr,
        app_ctx->enc.out_attr[ENC_OUT_ENCODER_HIDDEN_STATES].size
    );

    LOG(VERBOSE) << "Copy attention mask to decoder";
    memcpy(
        app_ctx->dec.input_mem[DEC_IN_ATTENTION_MASK_IDX]->virt_addr,
        attention_mask.data(),
        app_ctx->dec.in_attr[DEC_IN_ATTENTION_MASK_IDX].size
    );

    return greedy_decode(app_ctx, output_token);
}

int init_marian_rknn_model(
    const char* model_dir,
    bool verbose,
    rknn_marian_rknn_context_t* app_ctx)
{
    int ret = 0;

    std::string config_path = join_path(model_dir, "config.json");
    std::string encoder_path = join_path(model_dir, "encoder.rknn");
    std::string decoder_path = join_path(model_dir, "decoder.rknn");
    std::string source_spm_path = join_path(model_dir, "source.spm");
    std::string target_spm_path = join_path(model_dir, "target.spm");
    std::string vocab_path = join_path(model_dir, "vocab.json");
    std::string lm_weight_path = join_path(model_dir, "lm_weight.raw");
    std::string lm_bias_path = join_path(model_dir, "lm_bias.raw");

    LOG(INFO) << "load config " << config_path;
    std::ifstream config_file(config_path);
    if (!config_file) {
        LOG(ERROR) << "Failed to open config file: " << config_path;
        return -1;
    }
    json config;
    config_file >> config;
    if (!config.is_object()) {
        LOG(ERROR) << "Config is not a JSON object: " << config_path;
        return -1;
    }

    int d_model = config.value("d_model", 0);
    int vocab_size = config.value("vocab_size", 0);
    if (d_model <= 0 || vocab_size <= 0) {
        LOG(ERROR) << "Config missing required fields: d_model=" << d_model << " vocab_size=" << vocab_size;
        return -1;
    }
    app_ctx->decoder_start_token_id = config.value("decoder_start_token_id", 59513);
    app_ctx->pad_token_id = config.value("pad_token_id", app_ctx->decoder_start_token_id);
    app_ctx->eos_token_id = config.value("eos_token_id", 0);
    app_ctx->bos_token_id = config.value("bos_token_id", 0);
    app_ctx->unk_token_id = config.value("unk_token_id", 0);

    LOG(VERBOSE) << "d_model: " << d_model;
    LOG(VERBOSE) << "vocab size: " << vocab_size;
    LOG(VERBOSE) << "decoder start token id: " << app_ctx->decoder_start_token_id;
    LOG(VERBOSE) << "pad token id: " << app_ctx->pad_token_id;
    LOG(VERBOSE) << "eos token id: " << app_ctx->eos_token_id;
    LOG(VERBOSE) << "bos token id: " << app_ctx->bos_token_id;
    LOG(VERBOSE) << "unk token id: " << app_ctx->unk_token_id;

    LOG(INFO) << "Init RKNN encoder " << encoder_path;
    app_ctx->enc.m_path = encoder_path;
    ret = rknn_utils_init(&app_ctx->enc);
    if (ret != 0) {
        LOG(ERROR) << "rknn_utils_init failed. ret=" << ret;
        return -1;
    }

    LOG(INFO) << "Init RKNN decoder " << decoder_path;
    app_ctx->dec.m_path = decoder_path;
    ret = rknn_utils_init(&app_ctx->dec);
    if (ret != 0) {
        LOG(ERROR) << "rknn_utils_init failed. ret=" << ret;
        return -1;
    }

    app_ctx->enc_len = app_ctx->enc.in_attr[ENC_IN_INPUT_IDS_IDX].dims[1];
    LOG(INFO) << "Encoder length: " << app_ctx->enc_len;

    app_ctx->dec_len = app_ctx->dec.in_attr[DEC_IN_INPUT_IDS_IDX].dims[1];
    LOG(INFO) << "Decoder length: " << app_ctx->dec_len;

    LOG(VERBOSE) << "Init encoder buffers";
    ret = rknn_utils_init_input_buffer_all(&app_ctx->enc, ZERO_COPY_API);
    if (ret != 0) {
        LOG(ERROR) << "rknn_utils_init_input_buffer_all failed. ret=" << ret;
        return -1;
    }

    ret = rknn_utils_init_output_buffer_all(&app_ctx->enc, ZERO_COPY_API);
    if (ret != 0) {
        LOG(ERROR) << "rknn_utils_init_output_buffer_all failed. ret=" << ret;
        return -1;
    }

    LOG(VERBOSE) << "Init decoder buffers";
    ret = rknn_utils_init_input_buffer_all(&app_ctx->dec, ZERO_COPY_API);
    if (ret != 0) {
        LOG(ERROR) << "rknn_utils_init_input_buffer_all failed. ret=" << ret;
        return -1;
    }

    ret = rknn_utils_init_output_buffer_all(&app_ctx->dec, ZERO_COPY_API);
    if (ret != 0) {
        LOG(ERROR) << "rknn_utils_init_output_buffer_all failed. ret=" << ret;
        return -1;
    }

    LOG(VERBOSE) << "rknn_set_io_mem enc inputs; n_input=" << app_ctx->enc.n_input;
    for (int input_index = 0; input_index < app_ctx->enc.n_input; input_index++) {
        ret = rknn_set_io_mem(app_ctx->enc.ctx, app_ctx->enc.input_mem[input_index], &app_ctx->enc.in_attr[input_index]);
        if (ret < 0) {
            LOG(ERROR) << "rknn_set_io_mem failed. ret=" << ret;
            return -1;
        }
    }

    LOG(VERBOSE) << "rknn_set_io_mem enc outputs; n_output=" << app_ctx->enc.n_output;
    for (int output_index=0; output_index < app_ctx->enc.n_output; output_index++) {
        ret = rknn_set_io_mem(app_ctx->enc.ctx, app_ctx->enc.output_mem[output_index], &app_ctx->enc.out_attr[output_index]);
        if (ret < 0) {
            LOG(ERROR) << "rknn_set_io_mem failed. ret=" << ret;
            return -1;
        }
    }

    LOG(VERBOSE) << "rknn_set_io_mem dec inputs; n_input=" << app_ctx->dec.n_input;
    for (int input_index=0; input_index< app_ctx->dec.n_input; input_index++) {
        if (app_ctx->dec.in_attr[input_index].fmt == RKNN_TENSOR_NHWC) {
            rknn_query(app_ctx->dec.ctx, RKNN_QUERY_NATIVE_NC1HWC2_INPUT_ATTR, &app_ctx->dec.in_attr[input_index], sizeof(app_ctx->dec.in_attr[input_index]));
            app_ctx->dec.input_mem[input_index] = rknn_create_mem(app_ctx->dec.ctx, app_ctx->dec.in_attr[input_index].n_elems * sizeof(float)*2);
            app_ctx->dec.in_attr[input_index].pass_through = 1;
        }
        ret = rknn_set_io_mem(app_ctx->dec.ctx, app_ctx->dec.input_mem[input_index], &app_ctx->dec.in_attr[input_index]);
        if (ret < 0) {
            LOG(ERROR) << "rknn_set_io_mem failed. ret=" << ret;
            return -1;
        }
    }

    LOG(VERBOSE) << "rknn_set_io_mem dec outputs; n_output=" << app_ctx->dec.n_output;
    for (int output_index=0; output_index< app_ctx->dec.n_output; output_index++) {
        if (app_ctx->dec.out_attr[output_index].fmt == RKNN_TENSOR_NCHW) {
            rknn_query(app_ctx->dec.ctx, RKNN_QUERY_NATIVE_NC1HWC2_OUTPUT_ATTR, &app_ctx->dec.out_attr[output_index], sizeof(app_ctx->dec.out_attr[output_index]));
            rknn_destroy_mem(app_ctx->dec.ctx, app_ctx->dec.output_mem[output_index]);
            app_ctx->dec.output_mem[output_index] = rknn_create_mem(app_ctx->dec.ctx, app_ctx->dec.out_attr[output_index].n_elems * sizeof(float)*2);
        }
        ret = rknn_set_io_mem(app_ctx->dec.ctx, app_ctx->dec.output_mem[output_index], &app_ctx->dec.out_attr[output_index]);
        if (ret < 0) {
            LOG(ERROR) << "rknn_set_io_mem failed. ret=" << ret;
            return -1;
        }
    }

    LOG(INFO) << "loading source spm";
    auto src_status = app_ctx->spm_src.Load(source_spm_path);
    if (!src_status.ok()) {
        LOG(ERROR) << "Failed to load source sentencepiece model: " << src_status.ToString();
        return -1;
    }

    auto ps = app_ctx->spm_src.GetPieceSize();
    LOG(VERBOSE) << "source piece size: " << ps;

    LOG(INFO) << "loading target spm";
    auto tgt_status = app_ctx->spm_tgt.Load(target_spm_path);
    if (!tgt_status.ok()) {
        LOG(ERROR) << "Failed to load target sentencepiece model: " << tgt_status.ToString();
        return -1;
    }

    ps = app_ctx->spm_tgt.GetPieceSize();
    LOG(VERBOSE) << "Target piece size: " << ps;

    int D = app_ctx->lm_head.D = d_model;
    int V = app_ctx->lm_head.V = vocab_size;

    LOG(INFO) << "Load LM weight";
    app_ctx->lm_head.Wt = static_cast<float *>(malloc(sizeof(float) * V * D));
    read_fp32_from_file(lm_weight_path.c_str(), V * D, app_ctx->lm_head.Wt);

    LOG(INFO) << "Load LM bias";
    app_ctx->lm_head.b = static_cast<float *>(malloc(sizeof(float) * V));
    read_fp32_from_file(lm_bias_path.c_str(), V, app_ctx->lm_head.b);

    LOG(INFO) << "Load vocab";
    read_map_from_file(vocab_path, app_ctx->vocab);

    LOG(VERBOSE) << "Invert vocab";
    app_ctx->vocab_inv.reserve(app_ctx->vocab.size());
    for (const auto& entry : app_ctx->vocab) {
        auto existing = app_ctx->vocab_inv.find(entry.second);
        if (existing != app_ctx->vocab_inv.end()) {
            LOG(ERROR) << "Vocab is not unique. Duplicate found on ID: " << entry.second;
            return -1;
        }

        app_ctx->vocab_inv.emplace(entry.second, entry.first);
    }

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
    size_t token_list_len = 0;
    memset(token_list, 0, sizeof(token_list));

    // encode tokens
    std::vector<int> encoded_tokens;
    auto pieces = app_ctx->spm_src.EncodeAsPieces(std::string(input_sentence));
    std::ostringstream pieces_stream;
    pieces_stream << "sentence pieces:";
    for (const auto& piece : pieces) {
        pieces_stream << " " << piece;
    }
    LOG(VERBOSE) << pieces_stream.str();

    // apply vocab mapping
    LOG(VERBOSE) << "Apply vocab mapping";
    encoded_tokens.reserve(pieces.size());
    for (const auto& piece : pieces) {
        if (auto itr = app_ctx->vocab.find(piece); itr == app_ctx->vocab.end()) {
            // unknown token
            encoded_tokens.push_back(app_ctx->unk_token_id);
        } else {
            encoded_tokens.push_back(itr->second);
        }
    }

    // copy and truncate tokens
    token_list_len = encoded_tokens.size();
    if (token_list_len > (int)(sizeof(token_list) / sizeof(int))) {
        LOG(WARNING) << "Too many tokens (" << token_list_len << "), truncating to "
                     << sizeof(token_list) / sizeof(int);
        token_list_len = sizeof(token_list) / sizeof(int);
    }
    for (int i = 0; i < token_list_len; ++i) {
        token_list[i] = encoded_tokens[i];
    }

    // check input length
    uint32_t max_input_len = app_ctx->enc_len;
    if (token_list_len > max_input_len) {
        LOG(WARNING) << "token_len(" << token_list_len << ") > max_input_len(" << max_input_len
                     << "), only keep " << max_input_len << " tokens!";
        std::ostringstream tokens_all;
        tokens_all << "Tokens all     :";
        for (int i = 0; i < token_list_len; i++) {
            tokens_all << " " << token_list[i];
        }
        LOG(VERBOSE) << tokens_all.str();
        token_list_len = max_input_len;
        std::ostringstream tokens_remains;
        tokens_remains << "Tokens remains :";
        for (int i = 0; i < token_list_len; i++) {
            tokens_remains << " " << token_list[i];
        }
        LOG(VERBOSE) << tokens_remains.str();
    }

    // run model
    std::vector<int32_t> output_token;
    output_token.resize(app_ctx->dec_len, 0);
    int output_len = 0;
    output_len = rknn_nmt_process(app_ctx, token_list, output_token.data());

    // prepare tokens for decode
    LOG(VERBOSE) << "reverse vocab mapping";
    std::vector<std::string> decode_tokens;
    for (int i = 0; i < output_len; ++i) {
        if (output_token[i] == app_ctx->eos_token_id || output_token[i] == app_ctx->pad_token_id || output_token[i] <= 0) {
            break;
        }
        auto entry = app_ctx->vocab_inv.find(output_token[i]);
        if (entry == app_ctx->vocab_inv.end()) {
            LOG(WARNING) << "Token not found: " << output_token[i];
        } else {
            decode_tokens.push_back(entry->second);
        }
    }

    // decode tokens
    std::string decoded;
    if (auto status = app_ctx->spm_tgt.Decode(decode_tokens, &decoded); !status.ok()) {
        LOG(ERROR) << "Sentencepiece decode failed: " << status.ToString();
        return -1;
    }

    // copy output sentence
    memset(output_sentence, 0, MAX_USER_INPUT_LEN);
    strncpy(output_sentence, decoded.c_str(), MAX_USER_INPUT_LEN-1);

    return 0;
}
