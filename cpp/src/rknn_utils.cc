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

#include <cstring>
#include <sstream>
#include <string>

#include "rknn_api.h"
#include "rknn_utils.h"
#include "logger.h"

namespace {

void dump_tensor_attr(const rknn_tensor_attr *attr)
{
    std::ostringstream dims_stream;
    for (int i = 0; i < attr->n_dims; i++) {
        dims_stream << attr->dims[i] << ",";
    }

    LOG(VERBOSE) << "  index=" << attr->index
                 << ", name=" << attr->name
                 << ", n_dims=" << attr->n_dims
                 << ", dims=[" << dims_stream.str()
                 << "], n_elems=" << attr->n_elems
                 << ", size=" << attr->size
                 << ", size_with_stride=" << attr->size_with_stride
                 << ", fmt=" << get_format_string(attr->fmt)
                 << ", type=" << get_type_string(attr->type)
                 << ", qnt_type=" << get_qnt_type_string(attr->qnt_type)
                 << ", zp=" << attr->zp
                 << ", scale=" << attr->scale;
}

} // namespace

int rknn_utils_init(MODEL_INFO* model_info)
{
    if (model_info->m_path.empty()) {
        LOG(ERROR) << "Model path is null";
        return -1;
    }

    int ret = 0;
    ret = rknn_init(&model_info->ctx, model_info->m_path.data(), 0, model_info->init_flag, nullptr);
    if (ret < 0) {
        LOG(ERROR) << "rknn_init failed. ret=" << ret;
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(model_info->ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret != 0) {
        LOG(ERROR) << "Failed to query RKNN runtime information, error=" << ret;
        return -1;
    }

    LOG(VERBOSE) << "RKNN Runtime Information: librknnrt version: "
                 << version.drv_version << " (api version: " << version.api_version << ")";

    ret = rknn_utils_query_model_info(model_info);
    return ret;
}

int rknn_utils_query_model_info(MODEL_INFO* model_info)
{
    LOG(VERBOSE) << "rknn_utils_query_model_info";

    rknn_input_output_num io_num;
    int ret = rknn_query(model_info->ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        LOG(ERROR) << "rknn_query failed. ret=" << ret;
        return -1;
    }

    LOG(VERBOSE) << "model input num: " << io_num.n_input << ", output num: " << io_num.n_output;
    model_info->n_input = io_num.n_input;
    model_info->n_output = io_num.n_output;

    model_info->inputs = (rknn_input*)malloc(sizeof(rknn_input) * model_info->n_input);
    model_info->in_attr = (rknn_tensor_attr*)malloc(sizeof(rknn_tensor_attr) * model_info->n_input);
    model_info->in_attr_native = (rknn_tensor_attr*)malloc(sizeof(rknn_tensor_attr) * model_info->n_input);
    model_info->input_mem = (rknn_tensor_mem**)malloc(sizeof(rknn_tensor_mem*) * model_info->n_input);
    model_info->rknn_input_param = (RKNN_UTILS_INPUT_PARAM*)malloc(sizeof(RKNN_UTILS_INPUT_PARAM) * model_info->n_input);

    for (int i = 0; i < model_info->n_input; i++) {
        memset(&(model_info->inputs[i]), 0, sizeof(rknn_input));
        memset(&(model_info->rknn_input_param[i]), 0, sizeof(RKNN_UTILS_INPUT_PARAM));
    }

    model_info->outputs = (rknn_output*)malloc(sizeof(rknn_output) * model_info->n_output);
    model_info->out_attr = (rknn_tensor_attr*)malloc(sizeof(rknn_tensor_attr) * model_info->n_output);
    model_info->out_attr_native = (rknn_tensor_attr*)malloc(sizeof(rknn_tensor_attr) * model_info->n_output);
    model_info->output_mem = (rknn_tensor_mem**)malloc(sizeof(rknn_tensor_mem*) * model_info->n_output);
    model_info->rknn_output_param = (RKNN_UTILS_OUTPUT_PARAM*)malloc(sizeof(RKNN_UTILS_OUTPUT_PARAM) * model_info->n_output);

    for (int i = 0; i < model_info->n_output; i++) {
        memset(&(model_info->outputs[i]), 0, sizeof(rknn_output));
        memset(&(model_info->rknn_output_param[i]), 0, sizeof(RKNN_UTILS_OUTPUT_PARAM));
    }


    LOG(VERBOSE) << "INPUTS:";
    for (int i = 0; i < model_info->n_input; i++) {
        model_info->in_attr[i].index = i;
        ret = rknn_query(model_info->ctx, RKNN_QUERY_INPUT_ATTR, &model_info->in_attr[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            LOG(ERROR) << "rknn_query failed. ret=" << ret;
            return -1;
        }
        if (Logger::verbose()) {
            dump_tensor_attr(&model_info->in_attr[i]);
        }
    }

    LOG(VERBOSE) << "OUTPUTS:";
    for (int i = 0; i < model_info->n_output; i++) {
        model_info->out_attr[i].index = i;
        ret = rknn_query(model_info->ctx, RKNN_QUERY_OUTPUT_ATTR, &model_info->out_attr[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            LOG(ERROR) << "rknn_query failed. ret=" << ret;
            return -1;
        }
        if (Logger::verbose()) {
            dump_tensor_attr(&model_info->out_attr[i]);
        }
    }

    if (model_info->init_flag > 0) {
        ret = rknn_query(model_info->ctx, RKNN_QUERY_MEM_SIZE, &model_info->mem_size, sizeof(model_info->mem_size));
        if (ret != RKNN_SUCC) {
            LOG(ERROR) << "rknn_query failed. ret=" << ret;
            return -1;
        }
    }

    return 0;
}

int rknn_utils_get_type_size(const rknn_tensor_type type)
{
    switch (type)
    {
    case RKNN_TENSOR_FLOAT32:
        return 4;
    case RKNN_TENSOR_FLOAT16:
        return 2;
    case RKNN_TENSOR_UINT8:
    case RKNN_TENSOR_INT8:
        return 1;
    case RKNN_TENSOR_INT32:
        return 4;
    default:
        LOG(ERROR) << "Unsupported tensor type: " << get_type_string(type);
        return -1;
    }
}

int rknn_utils_init_input_buffer(
    const MODEL_INFO* model_info,
    const int node_index,
    const API_TYPE api_type,
    const uint8_t pass_through,
    const rknn_tensor_type dtype,
    const rknn_tensor_format layout_fmt)
{
    if (model_info->rknn_input_param[node_index]._already_init) {
        LOG(ERROR) << "Model input buffer already initialized";
        return -1;
    }
    model_info->rknn_input_param[node_index]._already_init = true;
    model_info->rknn_input_param[node_index].api_type = api_type;
    int elem_size = rknn_utils_get_type_size(dtype);

    if (api_type == NORMAL_API) {
        model_info->inputs[node_index].index = node_index;
        model_info->inputs[node_index].pass_through = pass_through;
        model_info->inputs[node_index].type = dtype;
        model_info->inputs[node_index].fmt = layout_fmt;
        model_info->inputs[node_index].size = model_info->in_attr[node_index].n_elems * elem_size;

        LOG(VERBOSE) << "rknn_utils_init_input_buffer: node_index=" << node_index
                        << ", size=" << model_info->inputs[node_index].size
                        << ", n_elems=" << model_info->in_attr[node_index].n_elems
                        << ", fmt=" << get_format_string(layout_fmt)
                        << ", type=" << get_type_string(dtype);
        return 0;
    }

    if (api_type == ZERO_COPY_API) {
        model_info->in_attr[node_index].fmt = layout_fmt;
        model_info->in_attr[node_index].type = dtype;

        if (layout_fmt == RKNN_TENSOR_UNDEFINED) {
            model_info->input_mem[node_index] = rknn_create_mem(model_info->ctx, model_info->in_attr[node_index].size);
        } else {
            model_info->input_mem[node_index] = rknn_create_mem(model_info->ctx, model_info->in_attr[node_index].size_with_stride);
        }

        LOG(VERBOSE) << "rknn_utils_init_input_buffer(zero copy): node_index=" << node_index
                     << ", size " << model_info->in_attr[node_index].size
                     << ", size with stride " << model_info->in_attr[node_index].size_with_stride
                     << ", fmt=" << get_format_string(layout_fmt)
                     << ", type=" << get_type_string(dtype);

        return 0;
    }

    LOG(ERROR) << "Unsupported api type: " << api_type;
    return -1;
}

int rknn_utils_init_output_buffer(const MODEL_INFO* model_info, const int node_index, const API_TYPE api_type)
{
    if (model_info->rknn_output_param[node_index]._already_init) {
        LOG(ERROR) << "Model output buffer already initialized";
        return -1;
    }
    model_info->rknn_output_param[node_index]._already_init = true;
    model_info->rknn_output_param[node_index].api_type = api_type;

    if (api_type == NORMAL_API) {
        model_info->outputs[node_index].index = node_index;
        LOG(VERBOSE) << "rknn_utils_init_output_buffer: node_index=" << node_index;
    } else if (api_type == ZERO_COPY_API) {
        int elem_size = rknn_utils_get_type_size(model_info->out_attr[node_index].type);
        model_info->output_mem[node_index] = rknn_create_mem(model_info->ctx, model_info->out_attr[node_index].n_elems * elem_size);
        LOG(VERBOSE) << "rknn_utils_init_output_buffer(zero copy): node_index="
                     << node_index << ", size with stride "
                     << model_info->out_attr[node_index].size;
    }
    return 0;
}

int rknn_utils_init_input_buffer_all(const MODEL_INFO* model_info, const API_TYPE default_api_type)
{
    rknn_tensor_format default_layout_fmt = RKNN_TENSOR_NHWC;

    for (int i = 0; i < model_info->n_input; i++) {
        if (model_info->rknn_input_param[i]._already_init) {
            LOG(WARNING) << "Model input buffer already init, ignore";
            continue;
        }
        int ret;
        if (model_info->rknn_input_param[i].enable) {
            ret = rknn_utils_init_input_buffer(model_info,
                                     i,
                                     model_info->rknn_input_param[i].api_type,
                                     model_info->rknn_input_param[i].pass_through,
                                     model_info->rknn_input_param[i].dtype,
                                     model_info->rknn_input_param[i].layout_fmt);
        } else {
            constexpr uint8_t default_pass_through = 0;
            if (model_info->in_attr[i].n_dims==4) {
                default_layout_fmt = model_info->in_attr[i].fmt;
            }

            ret = rknn_utils_init_input_buffer(
                model_info, i, default_api_type, default_pass_through,
                model_info->in_attr[i].type, default_layout_fmt);
        }
        if (ret != 0) {
            return ret;
        }
    }
    return 0;
}

int rknn_utils_init_output_buffer_all(const MODEL_INFO* model_info, const API_TYPE default_api_type)
{
    for (int i = 0; i < model_info->n_output; i++) {
        if (model_info->rknn_output_param[i]._already_init) {
            LOG(WARNING) << "Model output buffer already init, ignore";
            continue;
        }

        int ret;
        if (model_info->rknn_output_param[i].enable) {
            ret = rknn_utils_init_output_buffer(model_info, i, model_info->rknn_output_param[i].api_type);
        } else {
            ret = rknn_utils_init_output_buffer(model_info, i, default_api_type);
        }
        if (ret != 0) {
            return ret;
        }
    }
    return 0;
}

int rknn_utils_reset_all_buffer(const MODEL_INFO* model_info)
{
    for (int i = 0; i < model_info->n_input; i++) {
        if (model_info->input_mem[i] != nullptr) {
            rknn_destroy_mem(model_info->ctx, model_info->input_mem[i]);
        }
    }

    for (int i = 0; i < model_info->n_output; i++) {
        if (model_info->output_mem[i] != nullptr) {
            rknn_destroy_mem(model_info->ctx, model_info->output_mem[i]);
        }
    }

    memset(model_info->inputs, 0, sizeof(rknn_input) * model_info->n_input);
    memset(model_info->in_attr, 0, sizeof(rknn_tensor_attr) * model_info->n_input);
    memset(model_info->in_attr_native, 0, sizeof(rknn_tensor_attr) * model_info->n_input);
    memset(model_info->input_mem, 0, sizeof(rknn_tensor_mem*) * model_info->n_input);
    memset(model_info->rknn_input_param, 0, sizeof(RKNN_UTILS_INPUT_PARAM) * model_info->n_input);

    memset(model_info->outputs, 0, sizeof(rknn_output) * model_info->n_output);
    memset(model_info->out_attr, 0, sizeof(rknn_tensor_attr) * model_info->n_output);
    memset(model_info->out_attr_native, 0, sizeof(rknn_tensor_attr) * model_info->n_output);
    memset(model_info->output_mem, 0, sizeof(rknn_tensor_mem*) * model_info->n_output);
    memset(model_info->rknn_output_param, 0, sizeof(RKNN_UTILS_OUTPUT_PARAM) * model_info->n_output);

    return 0;
}

int rknn_utils_release(const MODEL_INFO* model_info)
{
    for (int i = 0; i < model_info->n_input; i++) {
        if (model_info->rknn_input_param[i].api_type == ZERO_COPY_API) {
            rknn_destroy_mem(model_info->ctx, model_info->input_mem[i]);
        }
    }

    for (int i = 0; i < model_info->n_output; i++) {
        if (model_info->rknn_output_param[i].api_type == ZERO_COPY_API) {
            rknn_destroy_mem(model_info->ctx, model_info->output_mem[i]);
        }
    }

    if (model_info->internal_mem_outside) {
        rknn_destroy_mem(model_info->ctx, model_info->internal_mem_outside);
    }
    if (model_info->internal_mem_max) {
        rknn_destroy_mem(model_info->ctx, model_info->internal_mem_max);
    }

    if (model_info->ctx>0) {
        rknn_destroy(model_info->ctx);
    }

    free(model_info->inputs);
    free(model_info->in_attr);
    free(model_info->in_attr_native);
    free(model_info->input_mem);
    free(model_info->rknn_input_param);

    free(model_info->outputs);
    free(model_info->out_attr);
    free(model_info->out_attr_native);
    free(model_info->output_mem);
    free(model_info->rknn_output_param);

    return 0;
}
