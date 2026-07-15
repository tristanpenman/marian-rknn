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

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <rknn_api.h>

const char* rknn_error_message(int ret);
std::string tensor_attr_to_string(const rknn_tensor_attr& attr);

enum API_TYPE
{
    NORMAL_API = 0,
    ZERO_COPY_API
};

struct RKNN_UTILS_INPUT_PARAM
{
    /*
        RKNN_INPUT has follow param:
        index, buf, size, pass_through, fmt, type

        Here we keep:
            pass_through,
            'fmt' as 'layout_fmt',
            'type' as 'dtype'

        And add:
            api_type to record normal_api/ zero_copy_api
            enable to assign if this param was used
            _already_init to record if this param was already init
    */
    uint8_t pass_through{};
    rknn_tensor_format layout_fmt{};
    rknn_tensor_type dtype{};

    API_TYPE api_type = NORMAL_API;
    bool enable = false;
    bool _already_init = false;
};

struct RKNN_UTILS_OUTPUT_PARAM
{
    API_TYPE api_type = NORMAL_API;
    bool enable = false;
    bool _already_init = false;
};

class RknnContext
{
public:
    RknnContext() = default;
    explicit RknnContext(rknn_context ctx) : ctx_(ctx) {}
    ~RknnContext() { reset(); }

    RknnContext(const RknnContext&) = delete;
    RknnContext& operator=(const RknnContext&) = delete;

    RknnContext(RknnContext&& other) noexcept : ctx_(std::exchange(other.ctx_, 0)) {}
    RknnContext& operator=(RknnContext&& other) noexcept
    {
        if (this != &other) {
            reset();
            ctx_ = std::exchange(other.ctx_, 0);
        }
        return *this;
    }

    operator rknn_context() const { return ctx_; }
    rknn_context get() const { return ctx_; }
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
    RknnMemHandle(rknn_context ctx, rknn_tensor_mem* mem) : ctx_(ctx), mem_(mem) {}
    ~RknnMemHandle() { reset(); }

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

    operator rknn_tensor_mem*() const { return mem_; }
    rknn_tensor_mem* operator->() const { return mem_; }
    rknn_tensor_mem* get() const { return mem_; }

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

struct MODEL_INFO
{
    std::string m_path;
    RknnContext ctx;
    bool is_dyn_shape = false;

    size_t n_input = 0;
    std::vector<rknn_tensor_attr> in_attr;
    std::vector<rknn_tensor_attr> in_attr_native;
    std::vector<rknn_input> inputs;
    std::vector<RknnMemHandle> input_mem;
    std::vector<RKNN_UTILS_INPUT_PARAM> rknn_input_param;

    size_t n_output = 0;
    std::vector<rknn_tensor_attr> out_attr;
    std::vector<rknn_tensor_attr> out_attr_native;
    std::vector<rknn_output> outputs;
    std::vector<RknnMemHandle> output_mem;
    std::vector<RKNN_UTILS_OUTPUT_PARAM> rknn_output_param;

    int diff_input_idx = -1;
    int init_flag = 0;

    std::vector<rknn_input_range> dyn_range;
    rknn_mem_size mem_size{};
    RknnMemHandle internal_mem_outside;
    RknnMemHandle internal_mem_max;
};

int rknn_utils_get_type_size(rknn_tensor_type type);

int rknn_utils_init(MODEL_INFO* model_info, std::optional<int> num_cores = std::nullopt);
int rknn_utils_query_model_info(MODEL_INFO* model_info);

int rknn_utils_init_input_buffer(MODEL_INFO* model_info, int node_index, API_TYPE api_type, uint8_t pass_through, rknn_tensor_type dtype, rknn_tensor_format layout_fmt);
int rknn_utils_init_output_buffer(MODEL_INFO* model_info, int node_index, API_TYPE api_type);

int rknn_utils_init_input_buffer_all(MODEL_INFO* model_info, API_TYPE default_api_type);
int rknn_utils_init_output_buffer_all(MODEL_INFO* model_info, API_TYPE default_api_type);

int rknn_utils_release(MODEL_INFO* model_info);
