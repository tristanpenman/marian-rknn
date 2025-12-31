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

#include "sentencepiece_processor.h"

#include "rknn_api.h"
#include "common.h"
#include "rknn_utils.h"
#include "easy_timer.h"

#define HEAD_NUM 4
#define EMBEDDING_DIM 256
#define DECODER_LAYER_NUM 3
#define MAX_SENTENCE_LEN 16

#define POS_LEN 1026

#define ENCODER_INPUT_TOKEN_RIGHTSIDE_ALIGN

#define MAX_WORD_NUM_IN_SENTENCE 64
#define MAX_WORD_LEN 64

#define MAX_USER_INPUT_LEN 1024

struct rknn_marian_rknn_context_t {
    MODEL_INFO enc;
    MODEL_INFO dec;

    sentencepiece::SentencePieceProcessor spm_src;
    sentencepiece::SentencePieceProcessor spm_tgt;
    int pad_token_id;
    int bos_token_id;
    int eos_token_id;

    int enc_len;
    int dec_len;
};

int init_marian_rknn_model(
    const char* encoder_path,
    const char* decoder_path,
    const char* source_spm_path,
    const char* target_spm_path,
    rknn_marian_rknn_context_t* app_ctx);

int release_marian_rknn_model(
    rknn_marian_rknn_context_t* app_ctx);

int inference_marian_rknn_model(
    rknn_marian_rknn_context_t* app_ctx,
    const char* input_sentence,
    char* output_sentence);
