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

#include "rknn_utils.h"

#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>

#include "rknn_api.h"

#include "logger.h"

const char* rknnErrorMessage(int ret)
{
    switch (ret) {
    case RKNN_SUCC:
        return "RKNN_SUCC (0): execute succeeded";
    case RKNN_ERR_FAIL:
        return "RKNN_ERR_FAIL (-1): execute failed";
    case RKNN_ERR_TIMEOUT:
        return "RKNN_ERR_TIMEOUT (-2): execute timed out";
    case RKNN_ERR_DEVICE_UNAVAILABLE:
        return "RKNN_ERR_DEVICE_UNAVAILABLE (-3): device is unavailable";
    case RKNN_ERR_MALLOC_FAIL:
        return "RKNN_ERR_MALLOC_FAIL (-4): memory allocation failed";
    case RKNN_ERR_PARAM_INVALID:
        return "RKNN_ERR_PARAM_INVALID (-5): parameter is invalid";
    case RKNN_ERR_MODEL_INVALID:
        return "RKNN_ERR_MODEL_INVALID (-6): model is invalid";
    case RKNN_ERR_CTX_INVALID:
        return "RKNN_ERR_CTX_INVALID (-7): context is invalid";
    case RKNN_ERR_INPUT_INVALID:
        return "RKNN_ERR_INPUT_INVALID (-8): input is invalid";
    case RKNN_ERR_OUTPUT_INVALID:
        return "RKNN_ERR_OUTPUT_INVALID (-9): output is invalid";
    case RKNN_ERR_DEVICE_UNMATCH:
        return "RKNN_ERR_DEVICE_UNMATCH (-10): SDK and NPU driver or firmware do not match";
    case RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL:
        return "RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL (-11): pre-compiled model is incompatible with current driver";
    case RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION:
        return "RKNN_ERR_INCOMPATIBLE_OPTIMIZATION_LEVEL_VERSION (-12): model optimization level is incompatible with current driver";
    case RKNN_ERR_TARGET_PLATFORM_UNMATCH:
        return "RKNN_ERR_TARGET_PLATFORM_UNMATCH (-13): model target platform does not match current platform";
    default:
        return "Unknown RKNN error code";
    }
}

std::string tensorAttrToString(const rknn_tensor_attr& attr)
{
    std::ostringstream stream;
    stream << "index=" << attr.index
           << " name=" << attr.name
           << " n_dims=" << attr.n_dims
           << " dims=[";
    for (uint32_t i = 0; i < attr.n_dims && i < RKNN_MAX_DIMS; ++i) {
        if (i > 0) {
            stream << ", ";
        }
        stream << attr.dims[i];
    }
    stream << "] n_elems=" << attr.n_elems
           << " size=" << attr.size
           << " size_with_stride=" << attr.size_with_stride
           << " fmt=" << get_format_string(attr.fmt)
           << " type=" << get_type_string(attr.type)
           << " qnt_type=" << get_qnt_type_string(attr.qnt_type)
           << " zp=" << attr.zp
           << " scale=" << attr.scale;
    return stream.str();
}

int rknnUtilsInit(ModelInfo* modelInfo, std::optional<int> numCores)
{
    if (modelInfo->path.empty()) {
        LOG(ERROR) << "Model path is null";
        return -1;
    }

    int ret = 0;
    ret = rknn_init(modelInfo->ctx.put(), modelInfo->path.data(), 0, modelInfo->initFlag, nullptr);
    if (ret < 0) {
        LOG(ERROR) << "rknn_init failed: " << rknnErrorMessage(ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(modelInfo->ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret != 0) {
        LOG(ERROR) << "Failed to query RKNN runtime information: " << rknnErrorMessage(ret);
        return -1;
    }

    LOG(VERBOSE) << "RKNN Runtime Information: librknnrt version: "
                 << version.drv_version << " (api version: " << version.api_version << ")";

    ret = rknnUtilsQueryModelInfo(modelInfo);
    if (ret != 0) {
        LOG(ERROR) << "rknnUtilsQueryModelInfo failed. ret=" << ret;
        return -1;
    }

    if (numCores.has_value()) {
        int coreCount = numCores.value();
        if (coreCount == 2) {
            ret = rknn_set_core_mask(modelInfo->ctx, RKNN_NPU_CORE_0_1);
        } else if (coreCount == 3) {
            ret = rknn_set_core_mask(modelInfo->ctx, RKNN_NPU_CORE_0_1_2);
        } else {
            ret = rknn_set_core_mask(modelInfo->ctx, RKNN_NPU_CORE_AUTO);
        }
    }

    if (ret != 0) {
        LOG(ERROR) << "Failed to set RKNN core mask: " << rknnErrorMessage(ret);
        return -1;
    }

    return 0;
}

int rknnUtilsQueryModelInfo(ModelInfo* modelInfo)
{
    LOG(VERBOSE) << "rknnUtilsQueryModelInfo";

    rknn_input_output_num ioNum;
    int ret = rknn_query(modelInfo->ctx, RKNN_QUERY_IN_OUT_NUM, &ioNum, sizeof(ioNum));
    if (ret != RKNN_SUCC) {
        LOG(ERROR) << "rknn_query failed: " << rknnErrorMessage(ret);
        return -1;
    }

    LOG(VERBOSE) << "model input num: " << ioNum.n_input << ", output num: " << ioNum.n_output;
    modelInfo->inputCount = ioNum.n_input;
    modelInfo->outputCount = ioNum.n_output;

    modelInfo->isDynamicShape = false;
    for (uint32_t i = 0; i < ioNum.n_input; i++) {
        rknn_input_range inputRange = {};
        inputRange.index = i;

        ret = rknn_query(modelInfo->ctx, RKNN_QUERY_INPUT_DYNAMIC_RANGE, &inputRange, sizeof(inputRange));
        if (ret != RKNN_SUCC) {
            if (Logger::verbose()) {
                LOG(VERBOSE) << "No input dynamic range for input index " << i << ": " << rknnErrorMessage(ret);
            }
            continue;
        }

        if (inputRange.shape_number > 1) {
            modelInfo->isDynamicShape = true;
            LOG(ERROR) << "Dynamically shaped RKNN models are not supported. Input \"" << inputRange.name
                       << "\" has " << inputRange.shape_number << " possible input shapes.";
            return -1;
        }
    }

    modelInfo->inputs.assign(modelInfo->inputCount, {});
    modelInfo->inputAttrs.assign(modelInfo->inputCount, {});
    modelInfo->nativeInputAttrs.assign(modelInfo->inputCount, {});
    modelInfo->inputMem.clear();
    modelInfo->inputMem.resize(modelInfo->inputCount);
    modelInfo->inputParams.assign(modelInfo->inputCount, {});

    modelInfo->outputs.assign(modelInfo->outputCount, {});
    modelInfo->outputAttrs.assign(modelInfo->outputCount, {});
    modelInfo->nativeOutputAttrs.assign(modelInfo->outputCount, {});
    modelInfo->outputMem.clear();
    modelInfo->outputMem.resize(modelInfo->outputCount);
    modelInfo->outputParams.assign(modelInfo->outputCount, {});

    LOG(VERBOSE) << "INPUTS:";
    for (int i = 0; i < modelInfo->inputCount; i++) {
        modelInfo->inputAttrs[i].index = i;
        ret = rknn_query(modelInfo->ctx, RKNN_QUERY_INPUT_ATTR, &modelInfo->inputAttrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            LOG(ERROR) << "rknn_query failed: " << rknnErrorMessage(ret);
            return -1;
        }
        if (Logger::verbose()) {
            LOG(VERBOSE) << "  " << tensorAttrToString(modelInfo->inputAttrs[i]);
        }
    }

    LOG(VERBOSE) << "OUTPUTS:";
    for (int i = 0; i < modelInfo->outputCount; i++) {
        modelInfo->outputAttrs[i].index = i;
        ret = rknn_query(modelInfo->ctx, RKNN_QUERY_OUTPUT_ATTR, &modelInfo->outputAttrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            LOG(ERROR) << "rknn_query failed: " << rknnErrorMessage(ret);
            return -1;
        }
        if (Logger::verbose()) {
            LOG(VERBOSE) << "  " << tensorAttrToString(modelInfo->outputAttrs[i]);
        }
    }

    if (modelInfo->initFlag > 0) {
        ret = rknn_query(modelInfo->ctx, RKNN_QUERY_MEM_SIZE, &modelInfo->memSize, sizeof(modelInfo->memSize));
        if (ret != RKNN_SUCC) {
            LOG(ERROR) << "rknn_query failed: " << rknnErrorMessage(ret);
            return -1;
        }
    }

    return 0;
}

int rknnUtilsGetTypeSize(const rknn_tensor_type type)
{
    switch (type) {
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

int rknnUtilsInitInputBuffer(
    ModelInfo* modelInfo,
    const int nodeIndex,
    const ApiType apiType,
    const uint8_t passThrough,
    const rknn_tensor_type dtype,
    const rknn_tensor_format layoutFormat)
{
    if (modelInfo->inputParams[nodeIndex].alreadyInit) {
        LOG(ERROR) << "Model input buffer already initialized";
        return -1;
    }
    modelInfo->inputParams[nodeIndex].alreadyInit = true;
    modelInfo->inputParams[nodeIndex].apiType = apiType;
    int elemSize = rknnUtilsGetTypeSize(dtype);

    if (apiType == ApiType::kNormal) {
        modelInfo->inputs[nodeIndex].index = nodeIndex;
        modelInfo->inputs[nodeIndex].pass_through = passThrough;
        modelInfo->inputs[nodeIndex].type = dtype;
        modelInfo->inputs[nodeIndex].fmt = layoutFormat;
        modelInfo->inputs[nodeIndex].size = modelInfo->inputAttrs[nodeIndex].n_elems * elemSize;

        LOG(VERBOSE) << "rknnUtilsInitInputBuffer: nodeIndex=" << nodeIndex
                        << ", size=" << modelInfo->inputs[nodeIndex].size
                        << ", n_elems=" << modelInfo->inputAttrs[nodeIndex].n_elems
                        << ", fmt=" << get_format_string(layoutFormat)
                        << ", type=" << get_type_string(dtype);
        return 0;
    }

    if (apiType == ApiType::kZeroCopy) {
        modelInfo->inputAttrs[nodeIndex].fmt = layoutFormat;
        modelInfo->inputAttrs[nodeIndex].type = dtype;

        if (layoutFormat == RKNN_TENSOR_UNDEFINED) {
            modelInfo->inputMem[nodeIndex].reset(
                modelInfo->ctx,
                rknn_create_mem(modelInfo->ctx, modelInfo->inputAttrs[nodeIndex].size));
        } else {
            modelInfo->inputMem[nodeIndex].reset(
                modelInfo->ctx,
                rknn_create_mem(modelInfo->ctx, modelInfo->inputAttrs[nodeIndex].size_with_stride));
        }

        LOG(VERBOSE) << "rknnUtilsInitInputBuffer(zero copy): nodeIndex=" << nodeIndex
                     << ", size " << modelInfo->inputAttrs[nodeIndex].size
                     << ", size with stride " << modelInfo->inputAttrs[nodeIndex].size_with_stride
                     << ", fmt=" << get_format_string(layoutFormat)
                     << ", type=" << get_type_string(dtype);

        return 0;
    }

    LOG(ERROR) << "Unsupported api type: " << static_cast<int>(apiType);
    return -1;
}

int rknnUtilsInitOutputBuffer(ModelInfo* modelInfo, const int nodeIndex, const ApiType apiType)
{
    if (modelInfo->outputParams[nodeIndex].alreadyInit) {
        LOG(ERROR) << "Model output buffer already initialized";
        return -1;
    }
    modelInfo->outputParams[nodeIndex].alreadyInit = true;
    modelInfo->outputParams[nodeIndex].apiType = apiType;

    if (apiType == ApiType::kNormal) {
        modelInfo->outputs[nodeIndex].index = nodeIndex;
        LOG(VERBOSE) << "rknnUtilsInitOutputBuffer: nodeIndex=" << nodeIndex;
    } else if (apiType == ApiType::kZeroCopy) {
        int elemSize = rknnUtilsGetTypeSize(modelInfo->outputAttrs[nodeIndex].type);
        modelInfo->outputMem[nodeIndex].reset(
            modelInfo->ctx,
            rknn_create_mem(modelInfo->ctx, modelInfo->outputAttrs[nodeIndex].n_elems * elemSize));
        LOG(VERBOSE) << "rknnUtilsInitOutputBuffer(zero copy): nodeIndex="
                     << nodeIndex << ", size with stride "
                     << modelInfo->outputAttrs[nodeIndex].size;
    }
    return 0;
}

int rknnUtilsInitInputBufferAll(ModelInfo* modelInfo, const ApiType defaultApiType)
{
    rknn_tensor_format defaultLayoutFormat = RKNN_TENSOR_NHWC;

    for (int i = 0; i < modelInfo->inputCount; i++) {
        if (modelInfo->inputParams[i].alreadyInit) {
            LOG(WARNING) << "Model input buffer already init, ignore";
            continue;
        }
        int ret;
        if (modelInfo->inputParams[i].enable) {
            ret = rknnUtilsInitInputBuffer(
                modelInfo,
                i,
                modelInfo->inputParams[i].apiType,
                modelInfo->inputParams[i].passThrough,
                modelInfo->inputParams[i].dtype,
                modelInfo->inputParams[i].layoutFormat);
        } else {
            constexpr uint8_t defaultPassThrough = 0;
            if (modelInfo->inputAttrs[i].n_dims == 4) {
                defaultLayoutFormat = modelInfo->inputAttrs[i].fmt;
            }

            ret = rknnUtilsInitInputBuffer(
                modelInfo, i, defaultApiType, defaultPassThrough,
                modelInfo->inputAttrs[i].type, defaultLayoutFormat);
        }
        if (ret != 0) {
            return ret;
        }
    }
    return 0;
}

int rknnUtilsInitOutputBufferAll(ModelInfo* modelInfo, const ApiType defaultApiType)
{
    for (int i = 0; i < modelInfo->outputCount; i++) {
        if (modelInfo->outputParams[i].alreadyInit) {
            LOG(WARNING) << "Model output buffer already init, ignore";
            continue;
        }

        int ret;
        if (modelInfo->outputParams[i].enable) {
            ret = rknnUtilsInitOutputBuffer(modelInfo, i, modelInfo->outputParams[i].apiType);
        } else {
            ret = rknnUtilsInitOutputBuffer(modelInfo, i, defaultApiType);
        }
        if (ret != 0) {
            return ret;
        }
    }
    return 0;
}

int rknnUtilsResetAllBuffer(ModelInfo* modelInfo)
{
    for (auto& mem : modelInfo->inputMem) {
        mem.reset();
    }

    for (auto& mem : modelInfo->outputMem) {
        mem.reset();
    }

    modelInfo->inputs.assign(modelInfo->inputCount, {});
    modelInfo->inputAttrs.assign(modelInfo->inputCount, {});
    modelInfo->nativeInputAttrs.assign(modelInfo->inputCount, {});
    modelInfo->inputMem.clear();
    modelInfo->inputMem.resize(modelInfo->inputCount);
    modelInfo->inputParams.assign(modelInfo->inputCount, {});

    modelInfo->outputs.assign(modelInfo->outputCount, {});
    modelInfo->outputAttrs.assign(modelInfo->outputCount, {});
    modelInfo->nativeOutputAttrs.assign(modelInfo->outputCount, {});
    modelInfo->outputMem.clear();
    modelInfo->outputMem.resize(modelInfo->outputCount);
    modelInfo->outputParams.assign(modelInfo->outputCount, {});

    return 0;
}

int rknnUtilsRelease(ModelInfo* modelInfo)
{
    for (auto& mem : modelInfo->inputMem) {
        mem.reset();
    }

    for (auto& mem : modelInfo->outputMem) {
        mem.reset();
    }

    modelInfo->internalMemOutside.reset();
    modelInfo->internalMemMax.reset();
    modelInfo->ctx.reset();

    modelInfo->inputs.clear();
    modelInfo->inputAttrs.clear();
    modelInfo->nativeInputAttrs.clear();
    modelInfo->inputMem.clear();
    modelInfo->inputParams.clear();

    modelInfo->outputs.clear();
    modelInfo->outputAttrs.clear();
    modelInfo->nativeOutputAttrs.clear();
    modelInfo->outputMem.clear();
    modelInfo->outputParams.clear();
    modelInfo->dynRange.clear();

    modelInfo->inputCount = 0;
    modelInfo->outputCount = 0;

    return 0;
}
