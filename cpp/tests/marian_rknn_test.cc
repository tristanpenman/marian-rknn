#include <algorithm>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "marian_rknn.h"

namespace {

TEST(MarianRknnTest, NormalizeStopsAtPadAndAppendsEos)
{
    const int32_t input[] = {10, 11, 0, 13, 14};
    RknnMarianInferenceStats dummyStats;
    RknnMarianContext dummyCtx;
    dummyCtx.encoderLength = 5;
    dummyCtx.padTokenId = 0;
    dummyCtx.eosTokenId = 2;

    const auto normalized = normalizeEncoderTokens(&dummyCtx, input, &dummyStats);

    EXPECT_EQ(dummyStats.inputTokens, 2u);
    EXPECT_EQ(normalized, (std::vector<int32_t>{10, 11, 2, 0, 0}));
}

TEST(MarianRknnTest, NormalizeStopsAtNonPositiveBoundary)
{
    const int32_t input[] = {7, -9, 4, 5};
    RknnMarianInferenceStats dummyStats;
    RknnMarianContext dummyCtx;
    dummyCtx.encoderLength = 4;
    dummyCtx.padTokenId = 99;
    dummyCtx.eosTokenId = 3;

    const auto normalized = normalizeEncoderTokens(&dummyCtx, input, &dummyStats);

    EXPECT_EQ(dummyStats.inputTokens, 1u);
    EXPECT_EQ(normalized, (std::vector<int32_t>{7, 3, 99, 99}));
}

TEST(MarianRknnTest, NormalizeWhenSequenceFullDoesNotInjectEos)
{
    const int32_t input[] = {1, 2, 3, 4};
    RknnMarianInferenceStats dummyStats;
    RknnMarianContext dummyCtx;
    dummyCtx.encoderLength = 4;
    dummyCtx.padTokenId = 0;
    dummyCtx.eosTokenId = 9;

    const auto normalized = normalizeEncoderTokens(&dummyCtx, input, &dummyStats);

    EXPECT_EQ(dummyStats.inputTokens, 4u);
    EXPECT_EQ(normalized, (std::vector<int32_t>{1, 2, 3, 4}));
}

TEST(MarianRknnTest, AttentionMaskKeepsEosVisibleThenPads)
{
    const std::vector<int32_t> normalized = {20, 21, 2, 0, 0};
    RknnMarianContext dummyCtx;
    dummyCtx.eosTokenId = 2;
    dummyCtx.encoderLength = 5;

    const auto mask = buildAttentionMask(&dummyCtx, normalized);

    EXPECT_EQ(mask, (std::vector<int32_t>{1, 1, 1, 0, 0}));
}

TEST(MarianRknnTest, AttentionMaskDropsAfterFirstEos)
{
    const std::vector<int32_t> normalized = {2, 2, 2};
    RknnMarianContext dummyCtx;
    dummyCtx.eosTokenId = 2;
    dummyCtx.encoderLength = 3;

    const auto mask = buildAttentionMask(&dummyCtx, normalized);

    EXPECT_EQ(mask, (std::vector<int32_t>{1, 0, 0}));
}

TEST(MarianRknnTest, LmHeadApplyFallsBackToEigen)
{
    const float weights[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float bias[] = {0.5f, -1.0f, 2.0f};
    const Half hidden[] = {floatToHalf(2.0f), floatToHalf(3.0f)};
    float logits[] = {0.0f, 0.0f, 0.0f};

    RknnMarianLmHead lmHead;
    lmHead.hiddenSize = 2;
    lmHead.vocabSize = 3;
    lmHead.weights = const_cast<float*>(weights);
    lmHead.bias = const_cast<float*>(bias);

    EXPECT_EQ(lmHead.apply(hidden, logits), 0);
    EXPECT_FLOAT_EQ(logits[0], 8.5f);
    EXPECT_FLOAT_EQ(logits[1], 17.0f);
    EXPECT_FLOAT_EQ(logits[2], 30.0f);
}

}  // namespace
