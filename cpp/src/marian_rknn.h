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
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <sentencepiece_processor.h>

#include "rknn_matmul_api.h"
#include "rknn_utils.h"
#include "type_half.h"

struct RknnMarianLmHeadMatmulChunk
{
    int vocabOffset = 0;
    int vocabSize = 0;
    int paddedVocabSize = 0;
    rknn_matmul_ctx ctx = 0;
    rknn_tensor_mem* input = nullptr;
    rknn_tensor_mem* weights = nullptr;
    rknn_tensor_mem* output = nullptr;
    rknn_matmul_io_attr ioAttr{};
};

struct RknnMarianLmHead
{
    int hiddenSize = 0;
    int vocabSize = 0;

    float* weights = nullptr;
    float* bias = nullptr;

    bool useNpu = false;
    std::vector<RknnMarianLmHeadMatmulChunk> matmulChunks;

    void operator()(const float* hidden, float* logits) const;
    int apply(const Half* hidden, float* logits) const;
};

struct RknnMarianContext
{
    // Read from SentencePiece model files
    sentencepiece::SentencePieceProcessor sourceTokenizer;
    sentencepiece::SentencePieceProcessor targetTokenizer;

    // Read from vocab file
    std::unordered_map<std::string, int32_t> vocab;
    std::unordered_map<int32_t, std::string> vocabInv;

    // RKNN encoder and decoder
    ModelInfo enc;
    ModelInfo dec;

    // Read from LM weight and bias files
    RknnMarianLmHead lmHead;

    // Read from config file
    int32_t bosTokenId = 0;
    int32_t eosTokenId = 0;
    int32_t decoderStartTokenId = 0;
    int32_t padTokenId = 0;
    int32_t unkTokenId = 0;

    // Model shape constraints
    size_t encoderLength = 0;
    size_t decoderLength = 0;
};

struct RknnMarianInferenceStats
{
    double totalMs = 0.0;
    double encoderMs = 0.0;
    double decoderMs = 0.0;
    double lmHeadMs = 0.0;
    size_t decoderIterations = 0;
    size_t inputTokens = 0;
    size_t outputTokens = 0;

    void reset()
    {
        totalMs = 0.0;
        encoderMs = 0.0;
        decoderMs = 0.0;
        lmHeadMs = 0.0;
        decoderIterations = 0;
        inputTokens = 0;
        outputTokens = 0;
    }

    void accumulate(const RknnMarianInferenceStats& other)
    {
        totalMs += other.totalMs;
        encoderMs += other.encoderMs;
        decoderMs += other.decoderMs;
        lmHeadMs += other.lmHeadMs;
        decoderIterations += other.decoderIterations;
        inputTokens += other.inputTokens;
        outputTokens += other.outputTokens;
    }
};

int initMarianRknnModel(
    const std::string& modelDir,
    RknnMarianContext* appCtx,
    bool eigen = false,
    std::optional<int> numCores = std::nullopt);

int releaseMarianRknnModel(
    RknnMarianContext* appCtx);

int inferenceMarianRknnModel(
    RknnMarianContext* appCtx,
    const std::string& inputSentence,
    std::string& outputSentence);

int inferenceMarianRknnModel(
    RknnMarianContext* appCtx,
    const std::string& inputSentence,
    std::string& outputSentence,
    RknnMarianInferenceStats* stats);

// Exposed for testing

std::vector<int32_t> buildAttentionMask(
    const RknnMarianContext* appCtx,
    const std::vector<int32_t>& normalizedTokens);

std::vector<int32_t> normalizeEncoderTokens(
    const RknnMarianContext* appCtx,
    const int32_t* inputTokens,
    RknnMarianInferenceStats* stats);
