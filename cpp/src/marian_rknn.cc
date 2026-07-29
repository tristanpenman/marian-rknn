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

#include "marian_rknn.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <sstream>
#include <vector>

#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <sentencepiece_processor.h>

#include "rknn_api.h"
#include "rknn_matmul_api.h"

#include "easy_timer.h"
#include "file_utils.h"
#include "logger.h"
#include "rknn_utils.h"
#include "time_utils.h"
#include "type_half.h"

using json = nlohmann::json;

namespace {

constexpr int kRk3588Fp16MatmulKAlignment = 32;
constexpr int kRk3588Fp16MatmulNAlignment = 32;
constexpr int kRk3588Fp16MatmulMaxN = 4096;

enum class EncoderInput : int
{
    kInputIds = 0,
    kAttentionMask = 1
};

enum class DecoderInput : int
{
    kInputIds = 0,
    kAttentionMask = 1,
    kEncoderHiddenStates = 2
};

enum class EncoderOutput : int
{
    kEncoderHiddenStates = 0
};

enum class DecoderOutput : int
{
    kDecoderOutput = 0
};

template<typename EnumType>
constexpr int toIndex(const EnumType input)
{
    return static_cast<int>(input);
}

int alignUp(const int value, const int alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

}  // namespace

void RknnMarianLmHead::operator()(const float* hidden, float* outLogits) const
{
    const Eigen::Map<const Eigen::Matrix<float, 1, Eigen::Dynamic>> h(hidden, hiddenSize);
    const Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> weightsMap(
        weights,
        hiddenSize,
        vocabSize);
    const Eigen::Map<const Eigen::Matrix<float, 1, Eigen::Dynamic>> biasMap(bias, vocabSize);

    Eigen::Map<Eigen::Matrix<float, 1, Eigen::Dynamic>> result(outLogits, vocabSize);
    result.noalias() = h * weightsMap;
    result += biasMap;
}

int RknnMarianLmHead::apply(const Half* hidden, float* outLogits) const
{
    if (useNpu) {
        for (const auto& chunk : matmulChunks) {
            memcpy(chunk.input->virt_addr, hidden, sizeof(Half) * hiddenSize);
            rknn_mem_sync(chunk.ctx, chunk.input, RKNN_MEMORY_SYNC_TO_DEVICE);

            int ret = rknn_matmul_set_io_mem(
                chunk.ctx,
                chunk.input,
                const_cast<rknn_matmul_tensor_attr*>(&chunk.ioAttr.A));
            if (ret < 0) {
                LOG(ERROR) << "rknn_matmul_set_io_mem(A) failed: " << rknnErrorMessage(ret);
                return -1;
            }

            ret = rknn_matmul_run(chunk.ctx);
            if (ret < 0) {
                LOG(ERROR) << "rknn_matmul_run failed: " << rknnErrorMessage(ret);
                return -1;
            }

            rknn_mem_sync(chunk.ctx, chunk.output, RKNN_MEMORY_SYNC_FROM_DEVICE);
            memcpy(
                outLogits + chunk.vocabOffset,
                chunk.output->virt_addr,
                sizeof(float) * chunk.vocabSize);
        }
        for (int i = 0; i < vocabSize; ++i) {
            outLogits[i] += bias[i];
        }
        return 0;
    }

    std::vector<float> hiddenFloats(hiddenSize, 0.0f);
    halfToFloatArray(hidden, hiddenFloats.data(), hiddenSize);
    (*this)(hiddenFloats.data(), outLogits);
    return 0;
}

void releaseLmHeadMatmul(RknnMarianLmHead* lmHead);

int initLmHeadMatmul(RknnMarianLmHead* lmHead)
{
    if (lmHead->hiddenSize % kRk3588Fp16MatmulKAlignment != 0) {
        LOG(WARNING) << "LM head hidden size " << lmHead->hiddenSize
                     << " is not " << kRk3588Fp16MatmulKAlignment
                     << "-aligned for RK3588 FP16 matmul. Falling back to Eigen.";
        return -1;
    }

    releaseLmHeadMatmul(lmHead);

    for (int offset = 0; offset < lmHead->vocabSize; offset += kRk3588Fp16MatmulMaxN) {
        RknnMarianLmHeadMatmulChunk chunk;
        chunk.vocabOffset = offset;
        chunk.vocabSize = std::min(kRk3588Fp16MatmulMaxN, lmHead->vocabSize - offset);
        chunk.paddedVocabSize = alignUp(chunk.vocabSize, kRk3588Fp16MatmulNAlignment);

        rknn_matmul_info info{};
        info.M = 1;
        info.K = lmHead->hiddenSize;
        info.N = chunk.paddedVocabSize;
        info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
        info.B_layout = RKNN_MM_LAYOUT_NORM;
        info.AC_layout = RKNN_MM_LAYOUT_NORM;

        int ret = rknn_matmul_create(&chunk.ctx, &info, &chunk.ioAttr);
        if (ret < 0) {
            LOG(WARNING) << "rknn_matmul_create failed for LM head chunk offset=" << offset
                         << " size=" << chunk.vocabSize << " padded_size=" << chunk.paddedVocabSize
                         << ". Falling back to Eigen: " << rknnErrorMessage(ret);
            releaseLmHeadMatmul(lmHead);
            return -1;
        }

        chunk.input = rknn_create_mem(chunk.ctx, chunk.ioAttr.A.size);
        chunk.weights = rknn_create_mem(chunk.ctx, chunk.ioAttr.B.size);
        chunk.output = rknn_create_mem(chunk.ctx, chunk.ioAttr.C.size);
        if (!chunk.input || !chunk.weights || !chunk.output) {
            LOG(WARNING) << "Failed to allocate RKNN matmul memory for LM head. Falling back to Eigen.";
            lmHead->matmulChunks.push_back(chunk);
            releaseLmHeadMatmul(lmHead);
            return -1;
        }

        std::vector<Half> fp16Weights(
            static_cast<size_t>(lmHead->hiddenSize) * chunk.paddedVocabSize,
            floatToHalf(0.0f));
        for (int d = 0; d < lmHead->hiddenSize; ++d) {
            for (int localVocabIndex = 0; localVocabIndex < chunk.vocabSize; ++localVocabIndex) {
                const int globalVocabIndex = offset + localVocabIndex;
                fp16Weights[static_cast<size_t>(d) * chunk.paddedVocabSize + localVocabIndex] =
                    floatToHalf(lmHead->weights[static_cast<size_t>(globalVocabIndex) * lmHead->hiddenSize + d]);
            }
        }
        memset(chunk.weights->virt_addr, 0, chunk.ioAttr.B.size);
        memcpy(
            chunk.weights->virt_addr,
            fp16Weights.data(),
            std::min<size_t>(chunk.ioAttr.B.size, fp16Weights.size() * sizeof(Half)));
        rknn_mem_sync(chunk.ctx, chunk.weights, RKNN_MEMORY_SYNC_TO_DEVICE);

        ret = rknn_matmul_set_io_mem(chunk.ctx, chunk.weights, &chunk.ioAttr.B);
        if (ret < 0) {
            LOG(WARNING) << "rknn_matmul_set_io_mem(B) failed for LM head. Falling back to Eigen: " << rknnErrorMessage(ret);
            lmHead->matmulChunks.push_back(chunk);
            releaseLmHeadMatmul(lmHead);
            return -1;
        }
        ret = rknn_matmul_set_io_mem(chunk.ctx, chunk.output, &chunk.ioAttr.C);
        if (ret < 0) {
            LOG(WARNING) << "rknn_matmul_set_io_mem(C) failed for LM head. Falling back to Eigen: " << rknnErrorMessage(ret);
            lmHead->matmulChunks.push_back(chunk);
            releaseLmHeadMatmul(lmHead);
            return -1;
        }

        lmHead->matmulChunks.push_back(chunk);
    }

    lmHead->useNpu = true;
    LOG(INFO) << "Using RKNN native matmul for LM head with " << lmHead->matmulChunks.size()
              << " RK3588-compatible vocab chunks";
    return 0;
}

void releaseLmHeadMatmul(RknnMarianLmHead* lmHead)
{
    for (auto& chunk : lmHead->matmulChunks) {
        if (chunk.ctx) {
            if (chunk.input) {
                rknn_destroy_mem(chunk.ctx, chunk.input);
                chunk.input = nullptr;
            }
            if (chunk.weights) {
                rknn_destroy_mem(chunk.ctx, chunk.weights);
                chunk.weights = nullptr;
            }
            if (chunk.output) {
                rknn_destroy_mem(chunk.ctx, chunk.output);
                chunk.output = nullptr;
            }
            rknn_matmul_destroy(chunk.ctx);
            chunk.ctx = 0;
        }
    }
    lmHead->matmulChunks.clear();
    lmHead->useNpu = false;
}

int greedyDecode(
    RknnMarianContext* appCtx,
    int32_t* outputToken,
    RknnMarianInferenceStats* stats)
{
    int ret = 0;

    LOG(VERBOSE) << "Setup decoder input state";
    std::vector<int32_t> decoderInputIds(appCtx->decoderLength, appCtx->padTokenId);
    decoderInputIds[0] = appCtx->decoderStartTokenId;

    // Output starts with pad token.
    std::fill_n(outputToken, appCtx->decoderLength, appCtx->padTokenId);

    EasyTimer timer;
    EasyTimer totalTimer;
    totalTimer.tik();
    for (int iteration = 0; iteration < appCtx->decoderLength - 1; iteration++) {
        LOG(VERBOSE) << "Decoder iteration " << iteration;
        memcpy(
            appCtx->dec.inputMem[toIndex(DecoderInput::kInputIds)]->virt_addr,
            decoderInputIds.data(),
            appCtx->dec.inputAttrs[toIndex(DecoderInput::kInputIds)].size
        );

        LOG(VERBOSE) << "rknn_run";
        auto runStart = std::chrono::steady_clock::now();
        timer.tik();
        ret = rknn_run(appCtx->dec.ctx, nullptr);
        timer.tok();
        auto runEnd = std::chrono::steady_clock::now();
        if (ret < 0) {
            LOG(ERROR) << "rknn_run failed: " << rknnErrorMessage(ret);
            return -1;
        }
        if (stats) {
            stats->decoderMs += elapsedMs(runStart, runEnd);
        }

        auto ptr = static_cast<Half*>(appCtx->dec.outputMem[toIndex(DecoderOutput::kDecoderOutput)]->virt_addr);
        const Half* iterationHidden = ptr + appCtx->lmHead.hiddenSize * iteration;

        LOG(VERBOSE) << "Apply LM head";
        auto lmStart = std::chrono::steady_clock::now();
        std::vector<float> logits(appCtx->lmHead.vocabSize);
        ret = appCtx->lmHead.apply(iterationHidden, logits.data());
        auto lmEnd = std::chrono::steady_clock::now();
        if (ret < 0) {
            return -1;
        }
        if (stats) {
            stats->lmHeadMs += elapsedMs(lmStart, lmEnd);
        }

        // Find argmax of logits.
        const auto maxIt = std::max_element(logits.begin(), logits.end());
        const int max = static_cast<int>(std::distance(logits.begin(), maxIt));
        const float value = *maxIt;
        LOG(VERBOSE) << "Argmax: " << max << " (" << value << ")";

        // Write output.
        outputToken[iteration] = max;

        // Feed back into decoder.
        if (iteration < appCtx->decoderLength - 1) {
            decoderInputIds[iteration + 1] = max;
        }

        if (max == appCtx->eosTokenId) {
            break;
        }
    }
    totalTimer.tok();

    int outputLength = 0;
    for (int i = 0; i < appCtx->decoderLength; i++) {
        if (outputToken[i] == appCtx->eosTokenId || outputToken[i] == appCtx->padTokenId) {
            break;
        }
        outputLength++;
    }

    if (Logger::verbose()) {
        std::ostringstream outputStream;
        outputStream << "Decoder output tokens:";
        for (int i = 0; i < outputLength; i++) {
            outputStream << " " << outputToken[i];
        }
        LOG(VERBOSE) << outputStream.str();
    }

    timer.printTime("RKNN decoder once run");

    LOG(VERBOSE) << "Decoder run " << outputLength - 1 << " times";
    totalTimer.printTime("Total time");

    if (stats) {
        stats->decoderIterations += static_cast<size_t>(outputLength > 0 ? outputLength - 1 : 0);
        stats->outputTokens += static_cast<size_t>(outputLength);
    }

    return outputLength;
}

// Token flow: input tokens, then EOS, then PAD to fill the encoder length.
std::vector<int32_t> normalizeEncoderTokens(
    const RknnMarianContext* appCtx,
    const int32_t* inputTokens,
    RknnMarianInferenceStats* stats)
{
    // Input tokens end at the first non-positive or pad token.
    const auto inputEnd = std::find_if(
        inputTokens,
        inputTokens + appCtx->encoderLength,
        [appCtx](const int32_t token) {
            return token <= 0 || token == appCtx->padTokenId;
        }
    );

    // Report stats.
    const size_t tokenCount = std::distance(inputTokens, inputEnd);
    LOG(VERBOSE) << "Tokens given: " << tokenCount;
    if (stats) {
        stats->inputTokens = tokenCount;
    }

    // Replace trailing token with EOS, then pad if necessary.
    std::vector<int32_t> normalizedTokens(appCtx->encoderLength, appCtx->padTokenId);
    std::copy_n(inputTokens, tokenCount, normalizedTokens.begin());
    if (tokenCount < appCtx->encoderLength) {
        normalizedTokens[tokenCount] = appCtx->eosTokenId;
    }

    // Log normalized token stream.
    if (Logger::verbose()) {
        std::ostringstream tokenStream;
        tokenStream << "Token stream: ";
        for (const auto token : normalizedTokens) {
            tokenStream << token << " ";
        }
        LOG(VERBOSE) << tokenStream.str();
    }

    return normalizedTokens;
}

// Mask flow: 1s for input and EOS, then transitions to 0s for PAD tokens.
std::vector<int32_t> buildAttentionMask(
    const RknnMarianContext* appCtx,
    const std::vector<int32_t>& normalizedTokens)
{
    std::vector<int32_t> attentionMask(appCtx->encoderLength, 0);

    // Input tokens are marked with 1 until the first EOS or PAD, then 0.
    std::transform(
        normalizedTokens.begin(),
        normalizedTokens.end(),
        attentionMask.begin(),
        [appCtx, padding = false](const int32_t token) mutable {
            const int32_t mask = padding ? 0 : 1;
            if (!padding && token == appCtx->eosTokenId) {
                padding = true;
            }
            return mask;
        }
    );

    // Log attention mask.
    if (Logger::verbose()) {
        std::ostringstream maskStream;
        maskStream << "Attention mask: ";
        for (const auto mask : attentionMask) {
            maskStream << " " << mask;
        }
        LOG(VERBOSE) << maskStream.str();
    }

    return attentionMask;
}

int rknnNmtProcess(
    RknnMarianContext* appCtx,
    const int32_t* inputToken,
    int32_t* outputToken,
    RknnMarianInferenceStats* stats)
{
    std::vector<int32_t> normalizedTokens = normalizeEncoderTokens(appCtx, inputToken, stats);
    std::vector<int32_t> attentionMask = buildAttentionMask(appCtx, normalizedTokens);

    LOG(VERBOSE) << "Copy input ids to encoder";
    memcpy(
        appCtx->enc.inputMem[toIndex(EncoderInput::kInputIds)]->virt_addr,
        normalizedTokens.data(),
        appCtx->enc.inputAttrs[toIndex(EncoderInput::kInputIds)].size
    );

    LOG(VERBOSE) << "Copy mask to encoder";
    memcpy(
        appCtx->enc.inputMem[toIndex(EncoderInput::kAttentionMask)]->virt_addr,
        attentionMask.data(),
        appCtx->enc.inputAttrs[toIndex(EncoderInput::kAttentionMask)].size
    );

    LOG(VERBOSE) << "Run encoder";
    EasyTimer timer;
    auto encoderStart = std::chrono::steady_clock::now();
    timer.tik();
    int ret = rknn_run(appCtx->enc.ctx, nullptr);
    if (ret < 0) {
        LOG(ERROR) << "rknn_run failed: " << rknnErrorMessage(ret);
        return -1;
    }
    timer.tok();
    auto encoderEnd = std::chrono::steady_clock::now();
    if (stats) {
        stats->encoderMs += elapsedMs(encoderStart, encoderEnd);
    }
    timer.printTime("RKNN encoder run");

    LOG(VERBOSE) << "Copy output from encoder to decoder";
    memcpy(
        appCtx->dec.inputMem[toIndex(DecoderInput::kEncoderHiddenStates)]->virt_addr,
        appCtx->enc.outputMem[toIndex(EncoderOutput::kEncoderHiddenStates)]->virt_addr,
        appCtx->enc.outputAttrs[toIndex(EncoderOutput::kEncoderHiddenStates)].size
    );

    LOG(VERBOSE) << "Copy attention mask to decoder";
    memcpy(
        appCtx->dec.inputMem[toIndex(DecoderInput::kAttentionMask)]->virt_addr,
        attentionMask.data(),
        appCtx->dec.inputAttrs[toIndex(DecoderInput::kAttentionMask)].size
    );

    return greedyDecode(appCtx, outputToken, stats);
}

std::optional<size_t> getSequenceLength(const rknn_tensor_attr& attr, const char* label)
{
    if (attr.n_dims < 2) {
        LOG(ERROR) << label << " has insufficient dims: " << attr.n_dims;
        return std::nullopt;
    }
    return attr.dims[1];
}

bool validateEqualLength(const size_t lhs, const size_t rhs, const char* lhsLabel, const char* rhsLabel)
{
    if (lhs != rhs) {
        LOG(ERROR) << lhsLabel << " length (" << lhs << ") does not match "
                   << rhsLabel << " length (" << rhs << ")";
        return false;
    }
    return true;
}

int validateSequenceLengths(const RknnMarianContext* appCtx)
{
    const auto encoderMaskLength = getSequenceLength(appCtx->enc.inputAttrs[toIndex(EncoderInput::kAttentionMask)], "encoder attentionMask");
    if (!encoderMaskLength) {
        return -1;
    }
    LOG(VERBOSE) << "Encoder mask len: " << *encoderMaskLength;
    if (!validateEqualLength(appCtx->encoderLength, *encoderMaskLength, "encoder input_ids", "encoder attentionMask")) {
        return -1;
    }

    const auto decoderMaskLength = getSequenceLength(appCtx->dec.inputAttrs[toIndex(DecoderInput::kAttentionMask)], "decoder attentionMask");
    if (!decoderMaskLength) {
        return -1;
    }
    LOG(VERBOSE) << "Decoder mask len: " << *decoderMaskLength;
    if (!validateEqualLength(appCtx->decoderLength, *decoderMaskLength, "decoder input_ids", "decoder attentionMask")) {
        return -1;
    }

    const auto decoderHiddenLength = getSequenceLength(appCtx->dec.inputAttrs[toIndex(DecoderInput::kEncoderHiddenStates)], "decoder encoder_hidden_states");
    if (!decoderHiddenLength) {
        return -1;
    }
    LOG(VERBOSE) << "Decoder hidden len: " << *decoderHiddenLength;
    if (*decoderHiddenLength <= 0) {
        return -1;
    }

    if (!validateEqualLength(*decoderHiddenLength, appCtx->encoderLength, "decoder encoder_hidden_states", "encoder input_ids")) {
        return -1;
    }

    if (appCtx->dec.inputAttrs[toIndex(DecoderInput::kEncoderHiddenStates)].n_dims >= 3) {
        const size_t decoderHiddenDim = appCtx->dec.inputAttrs[toIndex(DecoderInput::kEncoderHiddenStates)].dims[2];
        LOG(VERBOSE) << "Decoder hidden dim: " << decoderHiddenDim;
        if (decoderHiddenDim != appCtx->lmHead.hiddenSize) {
            LOG(ERROR) << "decoder encoder_hidden_states dim (" << decoderHiddenDim
                       << ") does not match d_model (" << appCtx->lmHead.hiddenSize << ")";
            return -1;
        }
    }

    return 0;
}

int initMarianRknnModel(
    const std::string& modelDir,
    RknnMarianContext* appCtx,
    bool eigen,
    std::optional<int> numCores)
{
    int ret = 0;

    const auto configPath = joinPath(modelDir, "config.json");
    const auto encoderPath = joinPath(modelDir, "encoder.rknn");
    const auto decoderPath = joinPath(modelDir, "decoder.rknn");
    const auto sourceSpmPath = joinPath(modelDir, "source.spm");
    const auto targetSpmPath = joinPath(modelDir, "target.spm");
    const auto vocabPath = joinPath(modelDir, "vocab.json");
    const auto lmWeightPath = joinPath(modelDir, "lm_weight.raw");
    const auto lmBiasPath = joinPath(modelDir, "lm_bias.raw");

    LOG(INFO) << "Load config " << configPath;
    std::ifstream configFile(configPath);
    if (!configFile) {
        LOG(ERROR) << "Failed to open config file: " << configPath;
        return -1;
    }
    json config;
    configFile >> config;
    if (!config.is_object()) {
        LOG(ERROR) << "Config is not a JSON object: " << configPath;
        return -1;
    }

    appCtx->lmHead.hiddenSize = config.value("d_model", 0);
    appCtx->lmHead.vocabSize = config.value("vocab_size", 0);
    if (appCtx->lmHead.hiddenSize <= 0 || appCtx->lmHead.vocabSize <= 0) {
        LOG(ERROR) << "Config missing required fields: d_model=" << appCtx->lmHead.hiddenSize
                   << " vocab_size=" << appCtx->lmHead.vocabSize;
        return -1;
    }

    appCtx->decoderStartTokenId = config.value("decoder_start_token_id", 59513);
    appCtx->padTokenId = config.value("pad_token_id", appCtx->decoderStartTokenId);
    appCtx->eosTokenId = config.value("eos_token_id", 0);
    appCtx->bosTokenId = config.value("bos_token_id", 0);
    appCtx->unkTokenId = config.value("unk_token_id", 0);

    LOG(VERBOSE) << "d_model: " << appCtx->lmHead.hiddenSize;
    LOG(VERBOSE) << "vocab_size: " << appCtx->lmHead.vocabSize;
    LOG(VERBOSE) << "decoder start token id: " << appCtx->decoderStartTokenId;
    LOG(VERBOSE) << "pad token id: " << appCtx->padTokenId;
    LOG(VERBOSE) << "eos token id: " << appCtx->eosTokenId;
    LOG(VERBOSE) << "bos token id: " << appCtx->bosTokenId;
    LOG(VERBOSE) << "unk token id: " << appCtx->unkTokenId;

    LOG(INFO) << "Init RKNN encoder " << encoderPath;
    appCtx->enc.path = encoderPath;
    ret = rknnUtilsInit(&appCtx->enc, numCores);
    if (ret != 0) {
        LOG(ERROR) << "rknnUtilsInit failed. ret=" << ret;
        return -1;
    }

    LOG(INFO) << "Init RKNN decoder " << decoderPath;
    appCtx->dec.path = decoderPath;
    ret = rknnUtilsInit(&appCtx->dec);
    if (ret != 0) {
        LOG(ERROR) << "rknnUtilsInit failed. ret=" << ret;
        return -1;
    }

    const auto encoderLength = getSequenceLength(appCtx->enc.inputAttrs[toIndex(EncoderInput::kInputIds)], "encoder input_ids");
    if (!encoderLength) {
        return -1;
    }
    appCtx->encoderLength = *encoderLength;
    LOG(INFO) << "Encoder length: " << appCtx->encoderLength;

    const auto decoderLength = getSequenceLength(appCtx->dec.inputAttrs[toIndex(DecoderInput::kInputIds)], "decoder input_ids");
    if (!decoderLength) {
        return -1;
    }
    appCtx->decoderLength = *decoderLength;
    LOG(INFO) << "Decoder length: " << appCtx->decoderLength;

    if (validateSequenceLengths(appCtx) != 0) {
        return -1;
    }

    LOG(VERBOSE) << "Init encoder buffers";
    ret = rknnUtilsInitInputBufferAll(&appCtx->enc, ApiType::kZeroCopy);
    if (ret != 0) {
        LOG(ERROR) << "rknnUtilsInitInputBufferAll failed. ret=" << ret;
        return -1;
    }

    ret = rknnUtilsInitOutputBufferAll(&appCtx->enc, ApiType::kZeroCopy);
    if (ret != 0) {
        LOG(ERROR) << "rknnUtilsInitOutputBufferAll failed. ret=" << ret;
        return -1;
    }

    LOG(VERBOSE) << "Init decoder buffers";
    ret = rknnUtilsInitInputBufferAll(&appCtx->dec, ApiType::kZeroCopy);
    if (ret != 0) {
        LOG(ERROR) << "rknnUtilsInitInputBufferAll failed. ret=" << ret;
        return -1;
    }

    ret = rknnUtilsInitOutputBufferAll(&appCtx->dec, ApiType::kZeroCopy);
    if (ret != 0) {
        LOG(ERROR) << "rknnUtilsInitOutputBufferAll failed. ret=" << ret;
        return -1;
    }

    LOG(VERBOSE) << "rknn_set_io_mem enc inputs; inputCount=" << appCtx->enc.inputCount;
    for (int inputIndex = 0; inputIndex < appCtx->enc.inputCount; inputIndex++) {
        ret = rknn_set_io_mem(appCtx->enc.ctx, appCtx->enc.inputMem[inputIndex], &appCtx->enc.inputAttrs[inputIndex]);
        if (ret < 0) {
            LOG(ERROR) << "rknn_set_io_mem failed: " << rknnErrorMessage(ret);
            return -1;
        }
    }

    LOG(VERBOSE) << "rknn_set_io_mem enc outputs; outputCount=" << appCtx->enc.outputCount;
    for (int outputIndex = 0; outputIndex < appCtx->enc.outputCount; outputIndex++) {
        ret = rknn_set_io_mem(appCtx->enc.ctx, appCtx->enc.outputMem[outputIndex], &appCtx->enc.outputAttrs[outputIndex]);
        if (ret < 0) {
            LOG(ERROR) << "rknn_set_io_mem failed: " << rknnErrorMessage(ret);
            return -1;
        }
    }

    LOG(VERBOSE) << "rknn_set_io_mem dec inputs; inputCount=" << appCtx->dec.inputCount;
    for (int inputIndex = 0; inputIndex < appCtx->dec.inputCount; inputIndex++) {
        if (appCtx->dec.inputAttrs[inputIndex].fmt == RKNN_TENSOR_NHWC) {
            rknn_query(appCtx->dec.ctx, RKNN_QUERY_NATIVE_NC1HWC2_INPUT_ATTR, &appCtx->dec.inputAttrs[inputIndex], sizeof(appCtx->dec.inputAttrs[inputIndex]));
            appCtx->dec.inputMem[inputIndex].reset(
                appCtx->dec.ctx,
                rknn_create_mem(appCtx->dec.ctx, appCtx->dec.inputAttrs[inputIndex].n_elems * sizeof(float) * 2));
            appCtx->dec.inputAttrs[inputIndex].pass_through = 1;
        }
        ret = rknn_set_io_mem(appCtx->dec.ctx, appCtx->dec.inputMem[inputIndex], &appCtx->dec.inputAttrs[inputIndex]);
        if (ret < 0) {
            LOG(ERROR) << "rknn_set_io_mem failed: " << rknnErrorMessage(ret);
            return -1;
        }
    }

    LOG(VERBOSE) << "rknn_set_io_mem dec outputs; outputCount=" << appCtx->dec.outputCount;
    for (int outputIndex = 0; outputIndex < appCtx->dec.outputCount; outputIndex++) {
        if (appCtx->dec.outputAttrs[outputIndex].fmt == RKNN_TENSOR_NCHW) {
            rknn_query(appCtx->dec.ctx, RKNN_QUERY_NATIVE_NC1HWC2_OUTPUT_ATTR, &appCtx->dec.outputAttrs[outputIndex], sizeof(appCtx->dec.outputAttrs[outputIndex]));
            appCtx->dec.outputMem[outputIndex].reset(
                appCtx->dec.ctx,
                rknn_create_mem(appCtx->dec.ctx, appCtx->dec.outputAttrs[outputIndex].n_elems * sizeof(float) * 2));
        }
        ret = rknn_set_io_mem(appCtx->dec.ctx, appCtx->dec.outputMem[outputIndex], &appCtx->dec.outputAttrs[outputIndex]);
        if (ret < 0) {
            LOG(ERROR) << "rknn_set_io_mem failed: " << rknnErrorMessage(ret);
            return -1;
        }
    }

    LOG(INFO) << "Loading source spm";
    if (auto sourceStatus = appCtx->sourceTokenizer.Load(sourceSpmPath); !sourceStatus.ok()) {
        LOG(ERROR) << "Failed to load source sentencepiece model: " << sourceStatus.ToString();
        return -1;
    }

    auto ps = appCtx->sourceTokenizer.GetPieceSize();
    LOG(VERBOSE) << "Source pieces: " << ps;

    LOG(INFO) << "Loading target spm";
    if (auto targetStatus = appCtx->targetTokenizer.Load(targetSpmPath); !targetStatus.ok()) {
        LOG(ERROR) << "Failed to load target sentencepiece model: " << targetStatus.ToString();
        return -1;
    }

    ps = appCtx->targetTokenizer.GetPieceSize();
    LOG(VERBOSE) << "Target pieces: " << ps;

    const int hiddenSize = appCtx->lmHead.hiddenSize;
    const int vocabSize = appCtx->lmHead.vocabSize;

    LOG(INFO) << "Load LM weight";
    appCtx->lmHead.weights = static_cast<float*>(malloc(sizeof(float) * vocabSize * hiddenSize));
    if (!appCtx->lmHead.weights) {
        LOG(ERROR) << "Failed to allocate LM weight buffer";
        return -1;
    }
    if (readFp32FromFile(lmWeightPath.c_str(), vocabSize * hiddenSize, appCtx->lmHead.weights) != 0) {
        return -1;
    }

    LOG(INFO) << "Load LM bias";
    appCtx->lmHead.bias = static_cast<float*>(malloc(sizeof(float) * vocabSize));
    if (!appCtx->lmHead.bias) {
        LOG(ERROR) << "Failed to allocate LM bias buffer";
        return -1;
    }
    if (readFp32FromFile(lmBiasPath.c_str(), vocabSize, appCtx->lmHead.bias) != 0) {
        return -1;
    }

    if (eigen) {
        LOG(INFO) << "Using Eigen for LM head";
    } else {
        LOG(INFO) << "Initialize LM head matmul";
        if (initLmHeadMatmul(&appCtx->lmHead) != 0) {
            releaseLmHeadMatmul(&appCtx->lmHead);
        }
    }

    LOG(INFO) << "Load vocab";
    readMapFromFile(vocabPath, appCtx->vocab);

    LOG(VERBOSE) << "Invert vocab";
    appCtx->vocabInv.reserve(appCtx->vocab.size());
    for (const auto& [fst, snd] : appCtx->vocab) {
        if (auto existing = appCtx->vocabInv.find(snd); existing != appCtx->vocabInv.end()) {
            LOG(ERROR) << "Vocab is not unique. Duplicate found on ID: " << snd;
            return -1;
        }

        appCtx->vocabInv.emplace(snd, fst);
    }

    return 0;
}

int releaseMarianRknnModel(RknnMarianContext* appCtx)
{
    rknnUtilsRelease(&appCtx->enc);
    rknnUtilsRelease(&appCtx->dec);
    releaseLmHeadMatmul(&appCtx->lmHead);

    free(appCtx->lmHead.weights);
    free(appCtx->lmHead.bias);

    return 0;
}

int inferenceMarianRknnModel(
    RknnMarianContext* appCtx,
    const std::string& inputSentence,
    std::string& outputSentence)
{
    return inferenceMarianRknnModel(appCtx, inputSentence, outputSentence, nullptr);
}

int inferenceMarianRknnModel(
    RknnMarianContext* appCtx,
    const std::string& inputSentence,
    std::string& outputSentence,
    RknnMarianInferenceStats* stats)
{
    auto totalStart = std::chrono::steady_clock::now();

    // Encode tokens.
    auto pieces = appCtx->sourceTokenizer.EncodeAsPieces(inputSentence);
    if (Logger::verbose()) {
        std::ostringstream piecesStream;
        piecesStream << "sentence pieces:";
        for (const auto& piece : pieces) {
            piecesStream << " " << piece;
        }
        LOG(VERBOSE) << piecesStream.str();
    }

    // Apply vocab mapping.
    LOG(VERBOSE) << "Apply vocab mapping";
    std::vector<int32_t> encodedTokens;
    encodedTokens.reserve(appCtx->encoderLength);
    for (const auto& piece : pieces) {
        if (auto itr = appCtx->vocab.find(piece); itr == appCtx->vocab.end()) {
            // Unknown token.
            encodedTokens.push_back(appCtx->unkTokenId);
        } else {
            encodedTokens.push_back(itr->second);
        }
    }

    // Check input length.
    if (encodedTokens.size() > appCtx->encoderLength) {
        LOG(INFO) << "Received " << encodedTokens.size() << " tokens, truncating to " << appCtx->encoderLength;
    } else if (encodedTokens.size() < appCtx->encoderLength) {
        LOG(INFO) << "Received " << encodedTokens.size() << " tokens, padding to " << appCtx->encoderLength;
    }

    // Resize and pad if necessary.
    encodedTokens.resize(appCtx->encoderLength, appCtx->padTokenId);

    // Run model.
    std::vector<int32_t> outputTokens;
    outputTokens.resize(appCtx->decoderLength, 0);
    int outputLength = rknnNmtProcess(appCtx, encodedTokens.data(), outputTokens.data(), stats);
    if (outputLength < 0) {
        return -1;
    }

    // Prepare tokens for decode.
    LOG(VERBOSE) << "reverse vocab mapping";
    std::vector<std::string> decodeTokens;
    for (int i = 0; i < outputLength; ++i) {
        if (outputTokens[i] == appCtx->eosTokenId || outputTokens[i] == appCtx->padTokenId || outputTokens[i] <= 0) {
            break;
        }
        if (auto entry = appCtx->vocabInv.find(outputTokens[i]); entry == appCtx->vocabInv.end()) {
            LOG(WARNING) << "Token not found: " << outputTokens[i];
        } else {
            decodeTokens.push_back(entry->second);
        }
    }

    // Decode tokens.
    outputSentence.clear();
    if (auto status = appCtx->targetTokenizer.Decode(decodeTokens, &outputSentence); !status.ok()) {
        LOG(ERROR) << "Sentencepiece decode failed: " << status.ToString();
        return -1;
    }

    auto totalEnd = std::chrono::steady_clock::now();
    if (stats) {
        stats->totalMs += elapsedMs(totalStart, totalEnd);
    }

    return 0;
}
