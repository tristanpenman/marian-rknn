// Copyright (c) 2023 Rockchip Electronics Co., Ltd. All Rights Reserved.
// Copyright (c) 2026 Tristan Penman
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
#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

// external
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <sentencepiece_processor.h>

// third-party
#include "rknn_api.h"
#include "rknn_matmul_api.h"

// internal
#include "easy_timer.h"
#include "file_utils.h"
#include "logger.h"
#include "marian_rknn.h"
#include "rknn_utils.h"
#include "time_utils.h"
#include "type_half.h"

using json = nlohmann::json;

namespace {

enum class EncoderInput : int
{
    InputIds = 0,
    AttentionMask = 1
};

enum class DecoderInput : int
{
    InputIds = 0,
    AttentionMask = 1,
    EncoderHiddenStates = 2
};

enum class EncoderOutput : int
{
    EncoderHiddenStates = 0
};

enum class DecoderOutput : int
{
    DecoderOutput = 0
};

template<typename EnumType>
constexpr int to_index(const EnumType input)
{
    return static_cast<int>(input);
}

}  // namespace

void rknn_marian_lm_head_t::operator()(const float* hidden, float* out_logits) const
{
    // map inputs
    const Eigen::Map<const Eigen::Matrix<float, 1, Eigen::Dynamic>> h(hidden, D);
    const Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> W(Wt, D, V);
    const Eigen::Map<const Eigen::Matrix<float, 1, Eigen::Dynamic>> bias(b, V);

    // calculate result
    Eigen::Map<Eigen::Matrix<float, 1, Eigen::Dynamic>> y(out_logits, V);
    y.noalias() = h * W;
    y += bias;
}

int rknn_marian_lm_head_t::apply(const half* hidden, float* out_logits) const
{
    if (use_npu) {
        memcpy(matmul_A->virt_addr, hidden, sizeof(half) * D);
        rknn_mem_sync(matmul_ctx, matmul_A, RKNN_MEMORY_SYNC_TO_DEVICE);

        const int ret = rknn_matmul_run(matmul_ctx);
        if (ret < 0) {
            LOG(ERROR) << "rknn_matmul_run failed. ret=" << ret;
            return -1;
        }

        rknn_mem_sync(matmul_ctx, matmul_C, RKNN_MEMORY_SYNC_FROM_DEVICE);
        memcpy(out_logits, matmul_C->virt_addr, sizeof(float) * V);
        for (int i = 0; i < V; ++i) {
            out_logits[i] += b[i];
        }
        return 0;
    }

    std::vector<float> hidden_floats(D, 0.0f);
    half_to_float_array(hidden, hidden_floats.data(), D);
    (*this)(hidden_floats.data(), out_logits);
    return 0;
}


int init_lm_head_matmul(rknn_marian_lm_head_t* lm_head)
{
    rknn_matmul_info info{};
    info.M = 1;
    info.K = lm_head->D;
    info.N = lm_head->V;
    info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    info.B_layout = RKNN_MM_LAYOUT_NORM;
    info.AC_layout = RKNN_MM_LAYOUT_NORM;

    lm_head->matmul_io_attr = new rknn_matmul_io_attr{};
    int ret = rknn_matmul_create(&lm_head->matmul_ctx, &info, lm_head->matmul_io_attr);
    if (ret < 0) {
        LOG(WARNING) << "rknn_matmul_create failed for LM head. Falling back to Eigen. ret=" << ret;
        delete lm_head->matmul_io_attr;
        lm_head->matmul_io_attr = nullptr;
        return -1;
    }

    lm_head->matmul_A = rknn_create_mem(lm_head->matmul_ctx, lm_head->matmul_io_attr->A.size);
    lm_head->matmul_B = rknn_create_mem(lm_head->matmul_ctx, lm_head->matmul_io_attr->B.size);
    lm_head->matmul_C = rknn_create_mem(lm_head->matmul_ctx, lm_head->matmul_io_attr->C.size);
    if (!lm_head->matmul_A || !lm_head->matmul_B || !lm_head->matmul_C) {
        LOG(WARNING) << "Failed to allocate RKNN matmul memory for LM head. Falling back to Eigen.";
        return -1;
    }

    std::vector<half> Wt_fp16(static_cast<size_t>(lm_head->D) * lm_head->V);
    for (int d = 0; d < lm_head->D; ++d) {
        for (int v = 0; v < lm_head->V; ++v) {
            Wt_fp16[static_cast<size_t>(d) * lm_head->V + v] =
                float_to_half(lm_head->Wt[static_cast<size_t>(v) * lm_head->D + d]);
        }
    }
    memset(lm_head->matmul_B->virt_addr, 0, lm_head->matmul_io_attr->B.size);
    memcpy(
        lm_head->matmul_B->virt_addr,
        Wt_fp16.data(),
        std::min<size_t>(lm_head->matmul_io_attr->B.size, Wt_fp16.size() * sizeof(half)));
    rknn_mem_sync(lm_head->matmul_ctx, lm_head->matmul_B, RKNN_MEMORY_SYNC_TO_DEVICE);

    ret = rknn_matmul_set_io_mem(lm_head->matmul_ctx, lm_head->matmul_A, &lm_head->matmul_io_attr->A);
    if (ret < 0) {
        LOG(WARNING) << "rknn_matmul_set_io_mem(A) failed for LM head. Falling back to Eigen. ret=" << ret;
        return -1;
    }
    ret = rknn_matmul_set_io_mem(lm_head->matmul_ctx, lm_head->matmul_B, &lm_head->matmul_io_attr->B);
    if (ret < 0) {
        LOG(WARNING) << "rknn_matmul_set_io_mem(B) failed for LM head. Falling back to Eigen. ret=" << ret;
        return -1;
    }
    ret = rknn_matmul_set_io_mem(lm_head->matmul_ctx, lm_head->matmul_C, &lm_head->matmul_io_attr->C);
    if (ret < 0) {
        LOG(WARNING) << "rknn_matmul_set_io_mem(C) failed for LM head. Falling back to Eigen. ret=" << ret;
        return -1;
    }

    lm_head->use_npu = true;
    LOG(INFO) << "Using RKNN native matmul for LM head";
    return 0;
}

void release_lm_head_matmul(rknn_marian_lm_head_t* lm_head)
{
    if (lm_head->matmul_ctx) {
        if (lm_head->matmul_A) {
            rknn_destroy_mem(lm_head->matmul_ctx, lm_head->matmul_A);
            lm_head->matmul_A = nullptr;
        }
        if (lm_head->matmul_B) {
            rknn_destroy_mem(lm_head->matmul_ctx, lm_head->matmul_B);
            lm_head->matmul_B = nullptr;
        }
        if (lm_head->matmul_C) {
            rknn_destroy_mem(lm_head->matmul_ctx, lm_head->matmul_C);
            lm_head->matmul_C = nullptr;
        }
        rknn_matmul_destroy(lm_head->matmul_ctx);
        lm_head->matmul_ctx = 0;
    }
    delete lm_head->matmul_io_attr;
    lm_head->matmul_io_attr = nullptr;
    lm_head->use_npu = false;
}

int greedy_decode(
    rknn_marian_rknn_context_t* app_ctx,
    int32_t* output_token,
    rknn_marian_inference_stats_t* stats)
{
    int ret = 0;

    LOG(VERBOSE) << "Setup decoder input state";
    std::vector<int32_t> decoder_input_ids(app_ctx->dec_len, app_ctx->pad_token_id);
    decoder_input_ids[0] = app_ctx->decoder_start_token_id;

    // output starts with pad token
    std::fill_n(output_token, app_ctx->dec_len, app_ctx->pad_token_id);

    EasyTimer timer;
    EasyTimer timer_total;
    timer_total.tik();
    for (int num_iter = 0; num_iter < app_ctx->dec_len - 1; num_iter++) {
        LOG(VERBOSE) << "Decoder iteration " << num_iter;
        memcpy(
            app_ctx->dec.input_mem[to_index(DecoderInput::InputIds)]->virt_addr,
            decoder_input_ids.data(),
            app_ctx->dec.in_attr[to_index(DecoderInput::InputIds)].size
        );

        LOG(VERBOSE) << "rknn_run";
        auto run_start = std::chrono::steady_clock::now();
        timer.tik();
        ret = rknn_run(app_ctx->dec.ctx, nullptr);
        timer.tok();
        auto run_end = std::chrono::steady_clock::now();
        if (ret < 0) {
            LOG(ERROR) << "rknn_run failed. ret=" << ret;
            return -1;
        }
        if (stats) {
            stats->decoder_ms += elapsed_ms(run_start, run_end);
        }

        auto ptr = static_cast<half *>(app_ctx->dec.output_mem[to_index(DecoderOutput::DecoderOutput)]->virt_addr);
        const half* iter_ptr = ptr + app_ctx->lm_head.D * num_iter;

        LOG(VERBOSE) << "Apply LM head";
        auto lm_start = std::chrono::steady_clock::now();
        std::vector<float> logits(app_ctx->lm_head.V);
        ret = app_ctx->lm_head.apply(iter_ptr, logits.data());
        auto lm_end = std::chrono::steady_clock::now();
        if (ret < 0) {
            return -1;
        }
        if (stats) {
            stats->lm_head_ms += elapsed_ms(lm_start, lm_end);
        }

        // find argmax of logits
        const auto max_it = std::max_element(logits.begin(), logits.end());
        const int max = static_cast<int>(std::distance(logits.begin(), max_it));
        const float value = *max_it;
        LOG(VERBOSE) << "Argmax: " << max << " (" << value << ")";

        // write output
        output_token[num_iter] = max;

        // feed back into decoder
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

    if (stats) {
        stats->decoder_iterations += static_cast<size_t>(output_len > 0 ? output_len - 1 : 0);
        stats->output_tokens += static_cast<size_t>(output_len);
    }

    return output_len;
}

// Token flow: input tokens, then EOS, then PAD to fill the encoder length.
std::vector<int32_t> normalize_encoder_tokens(
    const rknn_marian_rknn_context_t* app_ctx,
    const int32_t* input_tokens,
    rknn_marian_inference_stats_t* stats)
{
    // find end of input tokens, which is marked by first occurrence of non-positive or pad token
    const auto input_end = std::find_if(
        input_tokens,
        input_tokens + app_ctx->enc_len,
        [app_ctx](const int32_t token) {
            return token <= 0 || token == app_ctx->pad_token_id;
        }
    );

    // report stats
    const size_t num_tokens = std::distance(input_tokens, input_end);
    LOG(VERBOSE) << "Tokens given: " << num_tokens;
    if (stats) {
        stats->input_tokens = num_tokens;
    }

    // replace trailing token with eos, pad if necessary
    std::vector<int32_t> normalized_tokens(app_ctx->enc_len, app_ctx->pad_token_id);
    std::copy_n(input_tokens, num_tokens, normalized_tokens.begin());
    if (num_tokens < app_ctx->enc_len) {
        normalized_tokens[num_tokens] = app_ctx->eos_token_id;
    }

    // log normalized token stream
    if (Logger::verbose()) {
        std::ostringstream token_stream;
        token_stream << "Token stream: ";
        for (const auto token : normalized_tokens) {
            token_stream << token << " ";
        }
        LOG(VERBOSE) << token_stream.str();
    }

    return normalized_tokens;
}

// Mask flow: 1s for input and EOS, then transitions to 0s for PAD tokens.
std::vector<int32_t> build_attention_mask(
    const rknn_marian_rknn_context_t *app_ctx,
    const std::vector<int32_t> &normalized_tokens)
{
    std::vector<int32_t> attention_mask(app_ctx->enc_len, 0);

    // input tokens are marked with 1 until the first occurrence of EOS or PAD,
    // after which all tokens are marked with 0
    std::transform(
        normalized_tokens.begin(),
        normalized_tokens.end(),
        attention_mask.begin(),
        [app_ctx, padding = false](const int32_t token) mutable {
            const int32_t mask = padding ? 0 : 1;
            if (!padding && token == app_ctx->eos_token_id) {
                padding = true;
            }
            return mask;
        }
    );

    // log attention mask
    if (Logger::verbose()) {
        std::ostringstream mask_stream;
        mask_stream << "Attention mask: ";
        for (const auto mask : attention_mask) {
            mask_stream << " " << mask;
        }
        LOG(VERBOSE) << mask_stream.str();
    }

    return attention_mask;
}

int rknn_nmt_process(
    rknn_marian_rknn_context_t* app_ctx,
    const int32_t* input_token,
    int32_t* output_token,
    rknn_marian_inference_stats_t* stats)
{
    std::vector<int32_t> normalized_tokens = normalize_encoder_tokens(app_ctx, input_token, stats);
    std::vector<int32_t> attention_mask = build_attention_mask(app_ctx, normalized_tokens);

    LOG(VERBOSE) << "Copy input ids to encoder";
    memcpy(
        app_ctx->enc.input_mem[to_index(EncoderInput::InputIds)]->virt_addr,
        normalized_tokens.data(),
        app_ctx->enc.in_attr[to_index(EncoderInput::InputIds)].size
    );

    LOG(VERBOSE) << "Copy mask to encoder";
    memcpy(
        app_ctx->enc.input_mem[to_index(EncoderInput::AttentionMask)]->virt_addr,
        attention_mask.data(),
        app_ctx->enc.in_attr[to_index(EncoderInput::AttentionMask)].size
    );

    LOG(VERBOSE) << "Run encoder";
    EasyTimer timer;
    auto enc_start = std::chrono::steady_clock::now();
    timer.tik();
    int ret = rknn_run(app_ctx->enc.ctx, nullptr);
    if (ret < 0) {
        LOG(ERROR) << "rknn_run failed. ret=" << ret;
        return -1;
    }
    timer.tok();
    auto enc_end = std::chrono::steady_clock::now();
    if (stats) {
        stats->encoder_ms += elapsed_ms(enc_start, enc_end);
    }
    timer.print_time("RKNN encoder run");

    LOG(VERBOSE) << "Copy output from encoder to decoder";
    memcpy(
        app_ctx->dec.input_mem[to_index(DecoderInput::EncoderHiddenStates)]->virt_addr,
        app_ctx->enc.output_mem[to_index(EncoderOutput::EncoderHiddenStates)]->virt_addr,
        app_ctx->enc.out_attr[to_index(EncoderOutput::EncoderHiddenStates)].size
    );

    LOG(VERBOSE) << "Copy attention mask to decoder";
    memcpy(
        app_ctx->dec.input_mem[to_index(DecoderInput::AttentionMask)]->virt_addr,
        attention_mask.data(),
        app_ctx->dec.in_attr[to_index(DecoderInput::AttentionMask)].size
    );

    return greedy_decode(app_ctx, output_token, stats);
}

size_t get_sequence_length(const rknn_tensor_attr& attr, const char* label)
{
    if (attr.n_dims < 2) {
        LOG(ERROR) << label << " has insufficient dims: " << attr.n_dims;
        return -1;
    }
    return attr.dims[1];
}

bool validate_equal_length(const size_t lhs, const size_t rhs, const char* lhs_label, const char* rhs_label)
{
    if (lhs != rhs) {
        LOG(ERROR) << lhs_label << " length (" << lhs << ") does not match "
                   << rhs_label << " length (" << rhs << ")";
        return false;
    }
    return true;
}

int validate_sequence_lengths(const rknn_marian_rknn_context_t* app_ctx)
{
    const auto enc_mask_len = get_sequence_length(app_ctx->enc.in_attr[to_index(EncoderInput::AttentionMask)], "encoder attention_mask");
    LOG(VERBOSE) << "Encoder mask len: " << enc_mask_len;
    if (!validate_equal_length(app_ctx->enc_len, enc_mask_len, "encoder input_ids", "encoder attention_mask")) {
        return -1;
    }

    const auto dec_mask_len = get_sequence_length(app_ctx->dec.in_attr[to_index(DecoderInput::AttentionMask)], "decoder attention_mask");
    LOG(VERBOSE) << "Decoder mask len: " << dec_mask_len;
    if (!validate_equal_length(app_ctx->dec_len, dec_mask_len, "decoder input_ids", "decoder attention_mask")) {
        return -1;
    }

    const size_t dec_hidden_len = get_sequence_length(app_ctx->dec.in_attr[to_index(DecoderInput::EncoderHiddenStates)], "decoder encoder_hidden_states");
    LOG(VERBOSE) << "Decoder hidden len: " << dec_hidden_len;
    if (dec_hidden_len <= 0) {
        return -1;
    }

    if (!validate_equal_length(dec_hidden_len, app_ctx->enc_len, "decoder encoder_hidden_states", "encoder input_ids")) {
        return -1;
    }

    if (app_ctx->dec.in_attr[to_index(DecoderInput::EncoderHiddenStates)].n_dims >= 3) {
        const size_t dec_hidden_dim = app_ctx->dec.in_attr[to_index(DecoderInput::EncoderHiddenStates)].dims[2];
        LOG(VERBOSE) << "Decoder hidden dim: " << dec_hidden_dim;
        if (dec_hidden_dim != app_ctx->lm_head.D) {
            LOG(ERROR) << "decoder encoder_hidden_states dim (" << dec_hidden_dim
                       << ") does not match d_model (" << app_ctx->lm_head.D << ")";
            return -1;
        }
    }

    return 0;
}

int init_marian_rknn_model(const std::string &model_dir, rknn_marian_rknn_context_t *app_ctx)
{
    int ret = 0;

    const auto config_path = join_path(model_dir, "config.json");
    const auto encoder_path = join_path(model_dir, "encoder.rknn");
    const auto decoder_path = join_path(model_dir, "decoder.rknn");
    const auto source_spm_path = join_path(model_dir, "source.spm");
    const auto target_spm_path = join_path(model_dir, "target.spm");
    const auto vocab_path = join_path(model_dir, "vocab.json");
    const auto lm_weight_path = join_path(model_dir, "lm_weight.raw");
    const auto lm_bias_path = join_path(model_dir, "lm_bias.raw");

    LOG(INFO) << "Load config " << config_path;
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

    app_ctx->lm_head.D = config.value("d_model", 0);
    app_ctx->lm_head.V = config.value("vocab_size", 0);
    if (app_ctx->lm_head.D <= 0 || app_ctx->lm_head.V <= 0) {
        LOG(ERROR) << "Config missing required fields: d_model=" << app_ctx->lm_head.D << " vocab_size=" << app_ctx->lm_head.V;
        return -1;
    }

    app_ctx->decoder_start_token_id = config.value("decoder_start_token_id", 59513);
    app_ctx->pad_token_id = config.value("pad_token_id", app_ctx->decoder_start_token_id);
    app_ctx->eos_token_id = config.value("eos_token_id", 0);
    app_ctx->bos_token_id = config.value("bos_token_id", 0);
    app_ctx->unk_token_id = config.value("unk_token_id", 0);

    LOG(VERBOSE) << "d_model: " << app_ctx->lm_head.D;
    LOG(VERBOSE) << "vocab_size: " << app_ctx->lm_head.V;
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

    app_ctx->enc_len = get_sequence_length(app_ctx->enc.in_attr[to_index(EncoderInput::InputIds)], "encoder input_ids");
    LOG(INFO) << "Encoder length: " << app_ctx->enc_len;

    app_ctx->dec_len = get_sequence_length(app_ctx->dec.in_attr[to_index(DecoderInput::InputIds)], "decoder input_ids");
    LOG(INFO) << "Decoder length: " << app_ctx->dec_len;

    if (validate_sequence_lengths(app_ctx) != 0) {
        return -1;
    }

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

    LOG(INFO) << "Loading source spm";
    if (auto src_status = app_ctx->spm_src.Load(source_spm_path); !src_status.ok()) {
        LOG(ERROR) << "Failed to load source sentencepiece model: " << src_status.ToString();
        return -1;
    }

    auto ps = app_ctx->spm_src.GetPieceSize();
    LOG(VERBOSE) << "Source pieces: " << ps;

    LOG(INFO) << "Loading target spm";
    if (auto tgt_status = app_ctx->spm_tgt.Load(target_spm_path); !tgt_status.ok()) {
        LOG(ERROR) << "Failed to load target sentencepiece model: " << tgt_status.ToString();
        return -1;
    }

    ps = app_ctx->spm_tgt.GetPieceSize();
    LOG(VERBOSE) << "Target pieces: " << ps;

    const int D = app_ctx->lm_head.D;
    const int V = app_ctx->lm_head.V;

    LOG(INFO) << "Load LM weight";
    app_ctx->lm_head.Wt = static_cast<float *>(malloc(sizeof(float) * V * D));
    read_fp32_from_file(lm_weight_path.c_str(), V * D, app_ctx->lm_head.Wt);

    LOG(INFO) << "Load LM bias";
    app_ctx->lm_head.b = static_cast<float *>(malloc(sizeof(float) * V));
    read_fp32_from_file(lm_bias_path.c_str(), V, app_ctx->lm_head.b);

    if (init_lm_head_matmul(&app_ctx->lm_head) != 0) {
        release_lm_head_matmul(&app_ctx->lm_head);
    }

    LOG(INFO) << "Load vocab";
    read_map_from_file(vocab_path, app_ctx->vocab);

    LOG(VERBOSE) << "Invert vocab";
    app_ctx->vocab_inv.reserve(app_ctx->vocab.size());
    for (const auto&[fst, snd] : app_ctx->vocab) {
        if (auto existing = app_ctx->vocab_inv.find(snd); existing != app_ctx->vocab_inv.end()) {
            LOG(ERROR) << "Vocab is not unique. Duplicate found on ID: " << snd;
            return -1;
        }

        app_ctx->vocab_inv.emplace(snd, fst);
    }

    return 0;
}

int release_marian_rknn_model(rknn_marian_rknn_context_t* app_ctx)
{
    rknn_utils_release(&app_ctx->enc);
    rknn_utils_release(&app_ctx->dec);
    release_lm_head_matmul(&app_ctx->lm_head);

    free(app_ctx->lm_head.Wt);
    free(app_ctx->lm_head.b);

    return 0;
}

int inference_marian_rknn_model(
    rknn_marian_rknn_context_t* app_ctx,
    const std::string &input_sentence,
    std::string &output_sentence)
{
    return inference_marian_rknn_model(app_ctx, input_sentence, output_sentence, nullptr);
}

int inference_marian_rknn_model(
    rknn_marian_rknn_context_t* app_ctx,
    const std::string &input_sentence,
    std::string &output_sentence,
    rknn_marian_inference_stats_t* stats)
{
    auto total_start = std::chrono::steady_clock::now();

    // encode tokens
    auto pieces = app_ctx->spm_src.EncodeAsPieces(input_sentence);
    std::ostringstream pieces_stream;
    pieces_stream << "sentence pieces:";
    for (const auto& piece : pieces) {
        pieces_stream << " " << piece;
    }
    LOG(VERBOSE) << pieces_stream.str();

    // apply vocab mapping
    LOG(VERBOSE) << "Apply vocab mapping";
    std::vector<int32_t> encoded_tokens;
    encoded_tokens.reserve(app_ctx->enc_len);
    for (const auto& piece : pieces) {
        if (auto itr = app_ctx->vocab.find(piece); itr == app_ctx->vocab.end()) {
            // unknown token
            encoded_tokens.push_back(app_ctx->unk_token_id);
        } else {
            encoded_tokens.push_back(itr->second);
        }
    }

    // check input length
    if (encoded_tokens.size() > app_ctx->enc_len) {
        LOG(INFO) << "Received " << encoded_tokens.size() << " tokens, truncating to " << app_ctx->enc_len;
    } else if (encoded_tokens.size() < app_ctx->enc_len) {
        LOG(INFO) << "Received " << encoded_tokens.size() << " tokens, padding to " << app_ctx->enc_len;
    }

    // resize and pad if necessary
    encoded_tokens.resize(app_ctx->enc_len, app_ctx->pad_token_id);

    // run model
    std::vector<int32_t> output_tokens;
    output_tokens.resize(app_ctx->dec_len, 0);
    int output_len = rknn_nmt_process(app_ctx, encoded_tokens.data(), output_tokens.data(), stats);

    // prepare tokens for decode
    LOG(VERBOSE) << "reverse vocab mapping";
    std::vector<std::string> decode_tokens;
    for (int i = 0; i < output_len; ++i) {
        if (output_tokens[i] == app_ctx->eos_token_id || output_tokens[i] == app_ctx->pad_token_id || output_tokens[i] <= 0) {
            break;
        }
        if (auto entry = app_ctx->vocab_inv.find(output_tokens[i]); entry == app_ctx->vocab_inv.end()) {
            LOG(WARNING) << "Token not found: " << output_tokens[i];
        } else {
            decode_tokens.push_back(entry->second);
        }
    }

    // decode tokens
    output_sentence.clear();
    if (auto status = app_ctx->spm_tgt.Decode(decode_tokens, &output_sentence); !status.ok()) {
        LOG(ERROR) << "Sentencepiece decode failed: " << status.ToString();
        return -1;
    }

    auto total_end = std::chrono::steady_clock::now();
    if (stats) {
        stats->total_ms += elapsed_ms(total_start, total_end);
    }

    return 0;
}
