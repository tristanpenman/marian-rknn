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

#include "rknn_api.h"
#include "sentencepiece_processor.h"

#include "easy_timer.h"
#include "marian_rknn.h"
#include "rknn_utils.h"
#include "type_half.h"

int token_embedding(float *token_embed, float *position_embed, int *tokens, int len, float *embedding, int pad_token_id)
{
    float scale = sqrt(EMBEDDING_DIM);
    int pad = 1;
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            embedding[i * EMBEDDING_DIM + j] = token_embed[tokens[i] * EMBEDDING_DIM + j] * scale;
        }
    }

    for (int i = 0; i < len; i++) {
        if (tokens[i] != pad_token_id) {
            pad++;
        } else {
            pad = 1;
        }

        for (int j = 0; j < EMBEDDING_DIM; j++) {
            embedding[i * EMBEDDING_DIM + j] += position_embed[EMBEDDING_DIM * pad + j];
        }
    }

    return 0;
}

// 1x4x16x64 -> 1x15x64x4, nchw -> nhwc
int preprocess_prev_key_value(float *prev_data, float *src_output_data, int decoder_len)
{
    float mid_data[decoder_len * EMBEDDING_DIM];

    // 1x4x16x64->1x16x64x4
    for (int s = 0; s < decoder_len * EMBEDDING_DIM / HEAD_NUM; s++) {
        for (int h = 0; h < HEAD_NUM; h++) {
            mid_data[s*HEAD_NUM+h] = src_output_data[h*decoder_len*EMBEDDING_DIM/HEAD_NUM+s];
        }
    }

    // 1x16x64x4->1x15x64x4
    memcpy(prev_data, mid_data+EMBEDDING_DIM, (decoder_len-1)*EMBEDDING_DIM*sizeof(float));
    return 0;
}

int load_bin_fp32(const char* filename, float* data, int len)
{
    FILE *fp_token_embed = fopen(filename, "rb");
    if (fp_token_embed != NULL) {
        fread(data, sizeof(float), len, fp_token_embed);
        fclose(fp_token_embed);
    } else {
        printf("Open %s fail!\n", filename);
        return -1;
    }
    return 0;
}

int rknn_nmt_process(
    rknn_marian_rknn_context_t* app_ctx,
    int* input_token,
    int* output_token)
{
    int ret = 0;

    TIMER timer;
    TIMER timer_total;

    // share max buffer
    float enc_embedding[app_ctx->enc_len * EMBEDDING_DIM];
    float dec_embedding[app_ctx->dec_len * EMBEDDING_DIM];
    float enc_mask[app_ctx->enc_len];
    float dec_mask[app_ctx->dec_len];
    int input_token_sorted[app_ctx->enc_len];
    memset(enc_embedding, 0x00, sizeof(enc_embedding));
    memset(dec_embedding, 0x00, sizeof(dec_embedding));
    memset(enc_mask, 0x00, sizeof(enc_mask));
    memset(dec_mask, 0x00, sizeof(dec_mask));

    // init prev key
    float prev_key[DECODER_LAYER_NUM][(app_ctx->dec_len-1) * EMBEDDING_DIM];
    float prev_value[DECODER_LAYER_NUM][(app_ctx->dec_len-1) * EMBEDDING_DIM];
    memset(prev_key, 0x00, sizeof(prev_key));
    memset(prev_value, 0x00, sizeof(prev_value));

    int input_token_give = 0;
    for (int i=0; i<app_ctx->enc_len; i++) {
        if (input_token[i] <= 0 || input_token[i] == app_ctx->pad_token_id) {
            break;
        }
        input_token_give++;
    }
#ifdef ENCODER_INPUT_TOKEN_RIGHTSIDE_ALIGN
    // working as [22,33,1,1,1,1] -> [1,1,1,22,33,2]
    memset(input_token_sorted, 0, app_ctx->enc_len*sizeof(int));
    input_token_sorted[app_ctx->enc_len-1] = app_ctx->eos_token_id;
    for (int i=0; i<input_token_give; i++) {
        input_token_sorted[app_ctx->enc_len-1 - input_token_give +i] = input_token[i];
    }
#else
    // working as [22,33,1,1,1,1] -> [22,33,2,1,1,1]
    input_token_sorted[token_list_len] = app_ctx->eos_token_id;
#endif

    // gen encoder mask
    printf("input tokens(all should > 0):\n");
    for (int i=0; i< app_ctx->enc_len; i++) {
        if (input_token_sorted[i] == 0) {
            input_token_sorted[i] = app_ctx->pad_token_id;
            enc_mask[i] = 1;
        } else if (input_token_sorted[i] == app_ctx->pad_token_id) {
            enc_mask[i] = 1;
        } else {
            enc_mask[i] = 0;
        }
        printf(" %d", input_token_sorted[i]);
    }
    printf("\n");

    // expand_encoder_mask
    float enc_mask_expand[app_ctx->enc_len * app_ctx->enc_len];
    memset(enc_mask_expand, 0x00, sizeof(enc_mask_expand));
    for (int i=0; i<app_ctx->enc_len; i++) {
        for (int j=0; j<app_ctx->enc_len; j++) {
            enc_mask_expand[i*app_ctx->enc_len+j] = enc_mask[j];
        }
    }

    token_embedding(app_ctx->nmt_tokens.enc_token_embed, app_ctx->nmt_tokens.enc_pos_embed, input_token_sorted, app_ctx->enc_len, enc_embedding, app_ctx->pad_token_id);
    float_to_half_array(enc_embedding, (half*)(app_ctx->enc.input_mem[0]->virt_addr), app_ctx->enc.in_attr[0].n_elems);
    float_to_half_array(enc_mask_expand, (half*)(app_ctx->enc.input_mem[1]->virt_addr), app_ctx->enc.in_attr[1].n_elems);

    // Run
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

    output_token[0] = app_ctx->bos_token_id;

    printf("reset decoder input and output mem\n");
    for (int input_index = 0; input_index < app_ctx->dec.n_input; input_index++) {
        memset(app_ctx->dec.input_mem[input_index]->virt_addr, 0, app_ctx->dec.in_attr[input_index].n_elems * sizeof(half));
    }
    for (int output_index = 0; output_index < app_ctx->dec.n_output; output_index++) {
        memset(app_ctx->dec.output_mem[output_index]->virt_addr, 0, app_ctx->dec.out_attr[output_index].n_elems * sizeof(half));
    }

    // copy output from encoder to decoder
    memcpy(app_ctx->dec.input_mem[1]->virt_addr, app_ctx->enc.output_mem[0]->virt_addr, app_ctx->enc.out_attr[0].n_elems * sizeof(half));
    memcpy(app_ctx->dec.input_mem[2]->virt_addr, app_ctx->enc.input_mem[1]->virt_addr, app_ctx->enc.in_attr[1].n_elems * sizeof(half));

    // decoder run
    timer_total.tik();
    for (int num_iter = 0; num_iter < app_ctx->dec_len; num_iter++) {
        token_embedding(app_ctx->nmt_tokens.dec_token_embed, app_ctx->nmt_tokens.dec_pos_embed, output_token, num_iter+1, dec_embedding, app_ctx->pad_token_id);
        float_to_half_array(dec_embedding + num_iter*EMBEDDING_DIM, (half*)(app_ctx->dec.input_mem[0]->virt_addr), app_ctx->dec.in_attr[0].n_elems);

        float mask;
        for (int j = 0; j < app_ctx->dec_len; j++) {
            if (j >= app_ctx->dec_len - 1 - num_iter) {
                mask = 0;
            } else {
                mask = 1;
            }
            dec_mask[j] = mask;
        }
        float_to_half_array(dec_mask, (half*)(app_ctx->dec.input_mem[3]->virt_addr), app_ctx->dec.in_attr[3].n_elems);

        // incremental copy
        if (num_iter != 0) {
            for (int i = 0; i < DECODER_LAYER_NUM*2; i++) {
                memset(app_ctx->dec.input_mem[4+i]->virt_addr, 0, app_ctx->dec.in_attr[4+i].n_elems * sizeof(half));
                int increment_input_index = 4+i;
                int increment_output_index = 1+i;
                for (int h=0; h < app_ctx->dec.in_attr[increment_input_index].dims[1]; h++) {
                    for (int w=0; w < app_ctx->dec.in_attr[increment_input_index].dims[2]; w++) {
                        int input_offset = 0;
                        int output_offset = 0;
                        int cpy_size = 0;
                        // input dims as nhwc
                        input_offset += h * app_ctx->dec.in_attr[increment_input_index].dims[2] * app_ctx->dec.in_attr[increment_input_index].dims[3];
                        input_offset += w * app_ctx->dec.in_attr[increment_input_index].dims[3];

                        cpy_size = app_ctx->dec.in_attr[increment_input_index].dims[3];

                        // output dims as nc1hwc2
                        output_offset += (h+1) * app_ctx->dec.out_attr[increment_output_index].dims[3] * app_ctx->dec.out_attr[increment_output_index].dims[4];
                        output_offset += w * app_ctx->dec.out_attr[increment_output_index].dims[4];

                        input_offset = input_offset * sizeof(half);
                        output_offset = output_offset * sizeof(half);
                        cpy_size = cpy_size * sizeof(half);
                        memcpy((char*)app_ctx->dec.input_mem[increment_input_index]->virt_addr + input_offset,
                            (char*)app_ctx->dec.output_mem[increment_output_index]->virt_addr + output_offset,
                            cpy_size);
                    }
                }
            }
        }

        // Run
        timer.tik();
        ret = rknn_run(app_ctx->dec.ctx, nullptr);
        timer.tok();
        if (ret < 0) {
            printf("rknn_run fail! ret=%d\n", ret);
            return -1;
        }

        // argmax
        int max = 0;
        half* decoder_result_array = (half*)app_ctx->dec.output_mem[0]->virt_addr;
        float value = half_to_float(decoder_result_array[0]);
        for (int index = 1; index < app_ctx->dec.out_attr[0].n_elems/ app_ctx->dec.out_attr[0].dims[0]; index++) {
            if (half_to_float(decoder_result_array[index]) > value) {
                value = half_to_float(decoder_result_array[index]);
                max = index;
            }
        }
        //debug
        // printf("argmax - index %d, value %f\n", max, value);
        output_token[num_iter + 1] = max;
        if (max == app_ctx->eos_token_id) {
            break;
        }
    }
    timer_total.tok();

    // for debug
    int output_len=0;
    printf("decoder output token: ");
    for (int i = 0; i < app_ctx->dec_len; i++) {
        if (output_token[i] == 1) {
            break;
        }
        printf("%d ", output_token[i]);
        output_len ++;
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
    const char* token_embed_path,
    const char* pos_embed_path,
    const char* source_spm_path,
    const char* target_spm_path,
    rknn_marian_rknn_context_t* app_ctx)
{
    int ret = 0;
    memset(app_ctx, 0x00, sizeof(rknn_marian_rknn_context_t));

    printf("--> init rknn encoder %s\n", encoder_path);
    app_ctx->enc.m_path = encoder_path;
    app_ctx->enc.verbose_log = 1;
    ret = rknn_utils_init(&app_ctx->enc);
    if (ret != 0) {
        printf("rknn_utils_init ret=%d\n", ret);
        return -1;
    }

    printf("--> init rknn decoder %s\n", decoder_path);
    app_ctx->dec.m_path = decoder_path;
    app_ctx->dec.verbose_log = 1;
    ret = rknn_utils_init(&app_ctx->dec);
    if (ret != 0) {
        printf("rknn_utils_init ret=%d\n", ret);
        return -1;
    }

    app_ctx->enc_len = app_ctx->enc.in_attr[0].dims[1];
    printf("--> enc len: %d\n", app_ctx->enc_len);

    app_ctx->dec_len = app_ctx->dec.in_attr[3].dims[1];
    printf("--> dec len: %d\n", app_ctx->dec_len);

    printf("--> init encoder buffers\n");
    ret = rknn_utils_init_input_buffer_all(&app_ctx->enc, ZERO_COPY_API, RKNN_TENSOR_FLOAT16);
    if (ret != 0) {
        printf("rknn_utils_init_input_buffer_all ret=%d\n", ret);
        return -1;
    }

    ret = rknn_utils_init_output_buffer_all(&app_ctx->enc, ZERO_COPY_API, 0);
    if (ret != 0) {
        printf("rknn_utils_init_output_buffer_all ret=%d\n", ret);
        return -1;
    }

    printf("--> init decoder buffers\n");
    ret = rknn_utils_init_input_buffer_all(&app_ctx->dec, ZERO_COPY_API, RKNN_TENSOR_FLOAT16);
    if (ret != 0) {
        printf("rknn_utils_init_input_buffer_all ret=%d\n", ret);
        return -1;
    }

    ret = rknn_utils_init_output_buffer_all(&app_ctx->dec, ZERO_COPY_API, 0);
    if (ret != 0) {
        printf("rknn_utils_init_output_buffer_all ret=%d\n", ret);
        return -1;
    }

    // encoder zero_copy_io_set
    printf("--> rknn_set_io_mem enc inputs; n_input=%u\n", app_ctx->enc.n_input);
    for (int input_index = 0; input_index < app_ctx->enc.n_input; input_index++) {
        printf("calling rknn_set_io_mem %d\n", input_index);

        auto &a = app_ctx->enc.in_attr[input_index];
        printf(" - enc input %d: attr.index=%d name=%s size=%u\n", input_index, a.index, a.name, a.size);

        ret = rknn_set_io_mem(app_ctx->enc.ctx, app_ctx->enc.input_mem[input_index], &(app_ctx->enc.in_attr[input_index]));
        if (ret < 0) {
            printf("rknn_set_io_mem fail! ret=%d\n", ret);
            return -1;
        }
    }

    printf("--> rknn_set_io_mem enc outputs\n");
    for (int output_index=0; output_index < app_ctx->enc.n_output; output_index++) {
        ret = rknn_set_io_mem(app_ctx->enc.ctx, app_ctx->enc.output_mem[output_index], &(app_ctx->enc.out_attr[output_index]));
        if (ret < 0) {
            printf("rknn_set_io_mem fail! ret=%d\n", ret);
            return -1;
        }
    }

    printf("--> rknn_set_io_mem dec outputs\n");
    for (int output_index=0; output_index< app_ctx->dec.n_output; output_index++) {
        if (app_ctx->dec.out_attr[output_index].fmt == RKNN_TENSOR_NCHW) {
            rknn_query(app_ctx->dec.ctx, RKNN_QUERY_NATIVE_NC1HWC2_OUTPUT_ATTR, &(app_ctx->dec.out_attr[output_index]), sizeof(app_ctx->dec.out_attr[output_index]));
            rknn_destroy_mem(app_ctx->dec.ctx, app_ctx->dec.output_mem[output_index]);
            app_ctx->dec.output_mem[output_index] = rknn_create_mem(app_ctx->dec.ctx, app_ctx->dec.out_attr[output_index].n_elems * sizeof(half)*2);
        }
        ret = rknn_set_io_mem(app_ctx->dec.ctx, app_ctx->dec.output_mem[output_index], &(app_ctx->dec.out_attr[output_index]));
    }

    printf("--> rknn_set_io_mem dec inputs\n");
    for (int input_index=0; input_index< app_ctx->dec.n_input; input_index++) {
        if (app_ctx->dec.in_attr[input_index].fmt == RKNN_TENSOR_NHWC) {
            rknn_query(app_ctx->dec.ctx, RKNN_QUERY_NATIVE_NC1HWC2_INPUT_ATTR, &(app_ctx->dec.in_attr[input_index]), sizeof(app_ctx->dec.in_attr[input_index]));
            // 1x4x16x64输出, nc1hwc2输出, 1x16x64x4
            // 1x4x15x64输入, nc1hwc2输入, 1x1x15x64x8
            // 这两块 buffer 无法对齐, 需要手动 memcpy, 如果 channel 改成 8, 则可以无需手动 memcpy
            // app_ctx->dec.input_mem[input_index] = rknn_create_mem_from_fd(app_ctx->dec.ctx,
            //                                                       app_ctx->dec.output_mem[input_index-3]->fd,
            //                                                       app_ctx->dec.output_mem[input_index-3]->virt_addr,
            //                                                       app_ctx->dec.in_attr[input_index].n_elems* sizeof(half),
            //                                                       EMBEDDING_DIM*sizeof(half));
            app_ctx->dec.input_mem[input_index] = rknn_create_mem(app_ctx->dec.ctx, app_ctx->dec.in_attr[input_index].n_elems * sizeof(half)*2);
            app_ctx->dec.in_attr[input_index].pass_through = 1;
        }
        ret = rknn_set_io_mem(app_ctx->dec.ctx, app_ctx->dec.input_mem[input_index], &(app_ctx->dec.in_attr[input_index]));
    }

    printf("--> malloc token embeddings\n");
    int nmt_word_dict_len = app_ctx->dec.out_attr[0].n_elems/ app_ctx->dec.out_attr[0].dims[0];
    app_ctx->nmt_tokens.enc_token_embed = (float*)malloc(nmt_word_dict_len* EMBEDDING_DIM * sizeof(float));
    app_ctx->nmt_tokens.enc_pos_embed = (float*)malloc(POS_LEN* EMBEDDING_DIM * sizeof(float));

    printf("--> load token embed: %s\n", token_embed_path);
    ret = load_bin_fp32(token_embed_path, app_ctx->nmt_tokens.enc_token_embed, nmt_word_dict_len* EMBEDDING_DIM);
    if (ret != 0) {
        return -1;
    }

    printf("--> load pos embed: %s\n", pos_embed_path);
    ret = load_bin_fp32(pos_embed_path, app_ctx->nmt_tokens.enc_pos_embed, POS_LEN* EMBEDDING_DIM);
    if (ret != 0) {
        return -1;
    }
    app_ctx->nmt_tokens.dec_token_embed = app_ctx->nmt_tokens.enc_token_embed;
    app_ctx->nmt_tokens.dec_pos_embed = app_ctx->nmt_tokens.enc_pos_embed;

    auto src_status = app_ctx->spm_src.Load(source_spm_path);
    if (!src_status.ok()) {
        printf("Failed to load source sentencepiece model: %s\n", src_status.ToString().c_str());
        return -1;
    }
    auto tgt_status = app_ctx->spm_tgt.Load(target_spm_path);
    if (!tgt_status.ok()) {
        printf("Failed to load target sentencepiece model: %s\n", tgt_status.ToString().c_str());
        return -1;
    }
    int src_pad_id = app_ctx->spm_src.PieceToId("<pad>");
    int tgt_pad_id = app_ctx->spm_tgt.PieceToId("<pad>");
    int pad_candidate = tgt_pad_id >= 0 ? tgt_pad_id : src_pad_id;
    app_ctx->pad_token_id = pad_candidate >= 0 ? pad_candidate : 0;

    int eos_id = app_ctx->spm_tgt.PieceToId("</s>");
    app_ctx->eos_token_id = eos_id >= 0 ? eos_id : app_ctx->pad_token_id + 1;

    int bos_id = app_ctx->spm_tgt.PieceToId("<s>");
    app_ctx->bos_token_id = bos_id >= 0 ? bos_id : app_ctx->pad_token_id;

    return 0;
}

int release_marian_rknn_model(rknn_marian_rknn_context_t* app_ctx)
{
    // Release
    rknn_utils_release(&app_ctx->enc);
    rknn_utils_release(&app_ctx->dec);
    free(app_ctx->nmt_tokens.enc_token_embed);
    free(app_ctx->nmt_tokens.enc_pos_embed);
    return 0;
}

int inference_marian_rknn_model(
    rknn_marian_rknn_context_t* app_ctx,
    const char* input_sentence,
    char* output_sentence)
{
    int token_list[100];
    int token_list_len=0;
    memset(token_list, 0, sizeof(token_list));

    std::vector<int> encoded_tokens;
    auto status = app_ctx->spm_src.Encode(std::string(input_sentence), &encoded_tokens);
    if (!status.ok()) {
        printf("sentencepiece encode failed: %s\n", status.ToString().c_str());
        return -1;
    }
    token_list_len = encoded_tokens.size();
    if (token_list_len > (int)(sizeof(token_list)/sizeof(int))) {
        printf("WARNING: too many tokens (%d), truncating to %lu\n", token_list_len, sizeof(token_list)/sizeof(int));
        token_list_len = sizeof(token_list)/sizeof(int);
    }
    for (int i = 0; i < token_list_len; ++i) {
        token_list[i] = encoded_tokens[i];
    }

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

    int output_token[max_input_len];
    memset(output_token, 0, sizeof(output_token));
    int output_len = 0;
    output_len = rknn_nmt_process(app_ctx, token_list, output_token);

    memset(output_sentence, 0, MAX_USER_INPUT_LEN);
    std::vector<int> decode_tokens;
    for (int i = 1; i < output_len; ++i) {
        if (output_token[i] == app_ctx->eos_token_id || output_token[i] == app_ctx->pad_token_id || output_token[i] <= 0) {
            break;
        }
        decode_tokens.push_back(output_token[i]);
    }
    std::string decoded;
    status = app_ctx->spm_tgt.Decode(decode_tokens, &decoded);
    if (!status.ok()) {
        printf("sentencepiece decode failed: %s\n", status.ToString().c_str());
        return -1;
    }
    strncpy(output_sentence, decoded.c_str(), MAX_USER_INPUT_LEN-1);
    return 0;
}
