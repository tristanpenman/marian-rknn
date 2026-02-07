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

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

#include <sentencepiece_processor.h>

#include "rknn_utils.h"

#define MAX_USER_INPUT_LEN 1024

struct rknn_marian_lm_head_t
{
    int D;
    int V;

    float* Wt; // DxV row-major
    float* b;  // V

    void operator()(const float* hidden, float* logits) const;
};

struct rknn_marian_rknn_context_t
{
    // read from spm files
    sentencepiece::SentencePieceProcessor spm_src;
    sentencepiece::SentencePieceProcessor spm_tgt;

    // read from vocab file
    std::unordered_map<std::string, int32_t> vocab;
    std::unordered_map<int32_t, std::string> vocab_inv;

    // rknn encoder and decoder
    MODEL_INFO enc;
    MODEL_INFO dec;

    // read from lm weight and bias files
    rknn_marian_lm_head_t lm_head;

    // read from config file
    int32_t bos_token_id;
    int32_t eos_token_id;
    int32_t decoder_start_token_id;
    int32_t pad_token_id;
    int32_t unk_token_id;

    // other constraints
    size_t enc_len;
    size_t dec_len;
};

struct rknn_marian_inference_stats_t
{
    double total_ms = 0.0;
    double encoder_ms = 0.0;
    double decoder_ms = 0.0;
    double lm_head_ms = 0.0;
    size_t decoder_iterations = 0;
    size_t input_tokens = 0;
    size_t output_tokens = 0;

    void reset()
    {
        total_ms = 0.0;
        encoder_ms = 0.0;
        decoder_ms = 0.0;
        lm_head_ms = 0.0;
        decoder_iterations = 0;
        input_tokens = 0;
        output_tokens = 0;
    }

    void accumulate(const rknn_marian_inference_stats_t& other)
    {
        total_ms += other.total_ms;
        encoder_ms += other.encoder_ms;
        decoder_ms += other.decoder_ms;
        lm_head_ms += other.lm_head_ms;
        decoder_iterations += other.decoder_iterations;
        input_tokens += other.input_tokens;
        output_tokens += other.output_tokens;
    }
};

int init_marian_rknn_model(
    const std::string &model_dir,
    rknn_marian_rknn_context_t *app_ctx);

int release_marian_rknn_model(
    rknn_marian_rknn_context_t* app_ctx);

int inference_marian_rknn_model(
    rknn_marian_rknn_context_t* app_ctx,
    const std::string &input_sentence,
    std::string &output_sentence);

int inference_marian_rknn_model(
    rknn_marian_rknn_context_t* app_ctx,
    const std::string &input_sentence,
    std::string &output_sentence,
    rknn_marian_inference_stats_t* stats);
