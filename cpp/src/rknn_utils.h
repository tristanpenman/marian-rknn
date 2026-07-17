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
#include <utility>
#include <vector>

#include <rknn_api.h>

const char* rknnErrorMessage(int ret);
std::string tensorAttrToString(const rknn_tensor_attr& attr);

enum class ApiType
{
    kNormal,
    kZeroCopy
};

struct RknnUtilsInputParam
{
    /*
        RKNN_INPUT has follow param:
        index, buf, size, passThrough, fmt, type

        Here we keep:
            passThrough,
            'fmt' as 'layoutFormat',
            'type' as 'dtype'

        And add:
            apiType to record normal or zero-copy API usage
            enable to assign if this param was used
            alreadyInit to record if this param was already init
    */
    uint8_t passThrough{};
    rknn_tensor_format layoutFormat{};
    rknn_tensor_type dtype{};

    ApiType apiType = ApiType::kNormal;
    bool enable = false;
    bool alreadyInit = false;
};

struct RknnUtilsOutputParam
{
    ApiType apiType = ApiType::kNormal;
    bool enable = false;
    bool alreadyInit = false;
};

class RknnContext
{
public:
    RknnContext() = default;
    explicit RknnContext(rknn_context ctx)
        : ctx_(ctx)
    {
    }

    ~RknnContext()
    {
        reset();
    }

    RknnContext(const RknnContext&) = delete;
    RknnContext& operator=(const RknnContext&) = delete;

    RknnContext(RknnContext&& other) noexcept
        : ctx_(std::exchange(other.ctx_, 0))
    {
    }
    RknnContext& operator=(RknnContext&& other) noexcept
    {
        if (this != &other) {
            reset();
            ctx_ = std::exchange(other.ctx_, 0);
        }
        return *this;
    }

    operator rknn_context() const
    {
        return ctx_;
    }

    rknn_context get() const
    {
        return ctx_;
    }
    rknn_context* put()
    {
        reset();
        return &ctx_;
    }
    void reset(rknn_context ctx = 0)
    {
        if (ctx_ > 0) {
            rknn_destroy(ctx_);
        }
        ctx_ = ctx;
    }

private:
    rknn_context ctx_ = 0;
};

class RknnMemHandle
{
public:
    RknnMemHandle() = default;
    RknnMemHandle(rknn_context ctx, rknn_tensor_mem* mem)
        : ctx_(ctx)
        , mem_(mem)
    {
    }

    ~RknnMemHandle()
    {
        reset();
    }

    RknnMemHandle(const RknnMemHandle&) = delete;
    RknnMemHandle& operator=(const RknnMemHandle&) = delete;

    RknnMemHandle(RknnMemHandle&& other) noexcept
        : ctx_(std::exchange(other.ctx_, 0)), mem_(std::exchange(other.mem_, nullptr))
    {
    }
    RknnMemHandle& operator=(RknnMemHandle&& other) noexcept
    {
        if (this != &other) {
            reset();
            ctx_ = std::exchange(other.ctx_, 0);
            mem_ = std::exchange(other.mem_, nullptr);
        }
        return *this;
    }

    operator rknn_tensor_mem*() const
    {
        return mem_;
    }

    rknn_tensor_mem* operator->() const
    {
        return mem_;
    }

    rknn_tensor_mem* get() const
    {
        return mem_;
    }

    void reset(rknn_context ctx = 0, rknn_tensor_mem* mem = nullptr)
    {
        if (mem_ != nullptr) {
            rknn_destroy_mem(ctx_, mem_);
        }
        ctx_ = ctx;
        mem_ = mem;
    }

private:
    rknn_context ctx_ = 0;
    rknn_tensor_mem* mem_ = nullptr;
};

struct ModelInfo
{
    std::string path;
    RknnContext ctx;
    bool isDynamicShape = false;

    size_t inputCount = 0;
    std::vector<rknn_tensor_attr> inputAttrs;
    std::vector<rknn_tensor_attr> nativeInputAttrs;
    std::vector<rknn_input> inputs;
    std::vector<RknnMemHandle> inputMem;
    std::vector<RknnUtilsInputParam> inputParams;

    size_t outputCount = 0;
    std::vector<rknn_tensor_attr> outputAttrs;
    std::vector<rknn_tensor_attr> nativeOutputAttrs;
    std::vector<rknn_output> outputs;
    std::vector<RknnMemHandle> outputMem;
    std::vector<RknnUtilsOutputParam> outputParams;

    int diffInputIndex = -1;
    int initFlag = 0;

    std::vector<rknn_input_range> dynRange;
    rknn_mem_size memSize{};
    RknnMemHandle internalMemOutside;
    RknnMemHandle internalMemMax;
};

int rknnUtilsGetTypeSize(rknn_tensor_type type);

int rknnUtilsInit(ModelInfo* modelInfo, std::optional<int> numCores = std::nullopt);
int rknnUtilsQueryModelInfo(ModelInfo* modelInfo);

int rknnUtilsInitInputBuffer(ModelInfo* modelInfo, int nodeIndex, ApiType apiType, uint8_t passThrough, rknn_tensor_type dtype, rknn_tensor_format layoutFormat);
int rknnUtilsInitOutputBuffer(ModelInfo* modelInfo, int nodeIndex, ApiType apiType);

int rknnUtilsInitInputBufferAll(ModelInfo* modelInfo, ApiType defaultApiType);
int rknnUtilsInitOutputBufferAll(ModelInfo* modelInfo, ApiType defaultApiType);

int rknnUtilsRelease(ModelInfo* modelInfo);
