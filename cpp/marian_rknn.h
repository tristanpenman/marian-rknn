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

#pragma once

#include <string>
#include <unordered_map>

#include <rknn_api.h>
#include <sentencepiece_processor.h>

#include "rknn_utils.h"

#define MAX_USER_INPUT_LEN 1024

struct rknn_marian_lm_head_t
{
    int D;
    int V;

    float* Wt; // DxV row-major
    float* b;  // V

    void operator()(const float* hidden, float* out_logits) const;
};

struct rknn_marian_rknn_context_t
{
    MODEL_INFO enc;
    MODEL_INFO dec;

    sentencepiece::SentencePieceProcessor spm_src;
    sentencepiece::SentencePieceProcessor spm_tgt;

    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> vocab_inv;


    int bos_token_id; // 0
    int eos_token_id; // 0

    int decoder_start_token_id; // 59513
    int pad_token_id; // 59513

    int enc_len;
    int dec_len;

    rknn_marian_lm_head_t lm_head;
};

int init_marian_rknn_model(
    const char* encoder_path,
    const char* decoder_path,
    const char* source_spm_path,
    const char* target_spm_path,
    const char* vocab_path,
    const char* lm_weight_path,
    const char* lm_bias_path,
    rknn_marian_rknn_context_t* app_ctx);

int release_marian_rknn_model(
    rknn_marian_rknn_context_t* app_ctx);

int inference_marian_rknn_model(
    rknn_marian_rknn_context_t* app_ctx,
    const char* input_sentence,
    char* output_sentence);
