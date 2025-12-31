#pragma once

#include <cstdlib>

#include <rknn_api.h>

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
    uint8_t pass_through;
    rknn_tensor_format layout_fmt;
    rknn_tensor_type dtype;

    API_TYPE api_type;
    bool enable = false;
    bool _already_init = false;
};

struct RKNN_UTILS_OUTPUT_PARAM
{
    uint8_t want_float;
    API_TYPE api_type;
    bool enable = false;
    bool _already_init = false;
};

struct MODEL_INFO
{
    const char* m_path = nullptr;
    rknn_context ctx;
    bool is_dyn_shape = false;

    int n_input;
    rknn_tensor_attr* in_attr = nullptr;
    rknn_tensor_attr* in_attr_native = nullptr;
    rknn_input *inputs;
    rknn_tensor_mem **input_mem;
    RKNN_UTILS_INPUT_PARAM *rkdmo_input_param;

    int n_output;
    rknn_tensor_attr* out_attr = nullptr;
    rknn_tensor_attr* out_attr_native = nullptr;
    rknn_output *outputs;
    rknn_tensor_mem **output_mem;
    RKNN_UTILS_OUTPUT_PARAM *rkdmo_output_param;

    bool verbose_log = false;
    int diff_input_idx = -1;

    // memory could be set ousider
    int init_flag = 0;

    rknn_input_range* dyn_range;
    rknn_mem_size mem_size;
    rknn_tensor_mem* internal_mem_outside = nullptr;
    rknn_tensor_mem* internal_mem_max = nullptr;
};

int rknn_utils_get_type_size(rknn_tensor_type type);

int rknn_utils_init(MODEL_INFO* model_info);
int rknn_utils_query_model_info(MODEL_INFO* model_info);

int rknn_utils_init_input_buffer(MODEL_INFO* model_info, int node_index, API_TYPE api_type, uint8_t pass_through, rknn_tensor_type dtype, rknn_tensor_format layout_fmt);
int rknn_utils_init_output_buffer(MODEL_INFO* model_info, int node_index, API_TYPE api_type, uint8_t want_float);

int rknn_utils_init_input_buffer_all(MODEL_INFO* model_info, API_TYPE default_api_type, rknn_tensor_type default_t_type);
int rknn_utils_init_output_buffer_all(MODEL_INFO* model_info, API_TYPE default_api_type, uint8_t default_want_float);

int rknn_utils_release(MODEL_INFO* model_info);

int rknn_utils_query_dynamic_input(MODEL_INFO* model_info);
int rknn_utils_reset_dynamic_input(MODEL_INFO* model_info, int dynamic_shape_group_index);
