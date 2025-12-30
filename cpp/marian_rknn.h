// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
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

typedef struct _NMT_TOKENS{
    float *enc_token_embed;
    float *enc_pos_embed;
    float *dec_token_embed;
    float *dec_pos_embed;
} NMT_TOKENS;

typedef struct {
    MODEL_INFO enc;
    MODEL_INFO dec;
    NMT_TOKENS nmt_tokens;
    sentencepiece::SentencePieceProcessor spm_src;
    sentencepiece::SentencePieceProcessor spm_tgt;
    int pad_token_id;
    int bos_token_id;
    int eos_token_id;

    int enc_len;
    int dec_len;
} rknn_marian_rknn_context_t;


int init_marian_rknn_model(
    const char* encoder_path,
    const char* decoder_path,
    const char* token_embed_path,
    const char* pos_embed_path,
    const char* source_spm_path,
    const char* target_spm_path,
    rknn_marian_rknn_context_t* app_ctx);

int release_marian_rknn_model(
    rknn_marian_rknn_context_t* app_ctx);

int inference_marian_rknn_model(
    rknn_marian_rknn_context_t* app_ctx,
    const char* input_sentence,
    char* output_sentence);
