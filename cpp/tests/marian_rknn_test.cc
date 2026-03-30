#include <algorithm>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "marian_rknn.h"

namespace {

TEST(MarianRknnTest, NormalizeStopsAtPadAndAppendsEos)
{
    const int32_t input[] = {10, 11, 0, 13, 14};
    rknn_marian_inference_stats_t dummy_stats;
    rknn_marian_rknn_context_t dummy_ctx;
    dummy_ctx.enc_len = 5;
    dummy_ctx.pad_token_id = 0;
    dummy_ctx.eos_token_id = 2;

    const auto normalized = normalize_encoder_tokens(&dummy_ctx, input, &dummy_stats);

    EXPECT_EQ(dummy_stats.input_tokens, 2u);
    EXPECT_EQ(normalized, (std::vector<int32_t>{10, 11, 2, 0, 0}));
}

TEST(MarianRknnTest, NormalizeStopsAtNonPositiveBoundary)
{
    const int32_t input[] = {7, -9, 4, 5};
    rknn_marian_inference_stats_t dummy_stats;
    rknn_marian_rknn_context_t dummy_ctx;
    dummy_ctx.enc_len = 4;
    dummy_ctx.pad_token_id = 99;
    dummy_ctx.eos_token_id = 3;

    const auto normalized = normalize_encoder_tokens(&dummy_ctx, input, &dummy_stats);

    EXPECT_EQ(dummy_stats.input_tokens, 1u);
    EXPECT_EQ(normalized, (std::vector<int32_t>{7, 3, 99, 99}));
}

TEST(MarianRknnTest, NormalizeWhenSequenceFullDoesNotInjectEos)
{
    const int32_t input[] = {1, 2, 3, 4};
    rknn_marian_inference_stats_t dummy_stats;
    rknn_marian_rknn_context_t dummy_ctx;
    dummy_ctx.enc_len = 4;
    dummy_ctx.pad_token_id = 0;
    dummy_ctx.eos_token_id = 9;

    const auto normalized = normalize_encoder_tokens(&dummy_ctx, input, &dummy_stats);

    EXPECT_EQ(dummy_stats.input_tokens, 4u);
    EXPECT_EQ(normalized, (std::vector<int32_t>{1, 2, 3, 4}));
}

TEST(MarianRknnTest, AttentionMaskKeepsEosVisibleThenPads)
{
    const std::vector<int32_t> normalized = {20, 21, 2, 0, 0};
    rknn_marian_rknn_context_t dummy_ctx;
    dummy_ctx.eos_token_id = 2;
    dummy_ctx.enc_len = 5;

    const auto mask = build_attention_mask(&dummy_ctx, normalized);

    EXPECT_EQ(mask, (std::vector<int32_t>{1, 1, 1, 0, 0}));
}

TEST(MarianRknnTest, AttentionMaskDropsAfterFirstEos)
{
    const std::vector<int32_t> normalized = {2, 2, 2};
    rknn_marian_rknn_context_t dummy_ctx;
    dummy_ctx.eos_token_id = 2;
    dummy_ctx.enc_len = 3;

    const auto mask = build_attention_mask(&dummy_ctx, normalized);

    EXPECT_EQ(mask, (std::vector<int32_t>{1, 0, 0}));
}

}  // namespace
