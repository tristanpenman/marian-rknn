#pragma once

#include "rknn_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef rknn_context rknn_matmul_ctx;

typedef enum _rknn_matmul_type {
    RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32 = 1,
} rknn_matmul_type;

typedef enum _rknn_matmul_layout {
    RKNN_MM_LAYOUT_NORM = 0,
    RKNN_MM_LAYOUT_NATIVE = 1,
    RKNN_MM_LAYOUT_TP_NORM = 2,
} rknn_matmul_layout;

typedef struct _rknn_matmul_tensor_attr {
    char name[RKNN_MAX_NAME_LEN];
    uint32_t n_dims;
    uint32_t dims[RKNN_MAX_DIMS];
    uint32_t size;
    rknn_tensor_type type;
} rknn_matmul_tensor_attr;

typedef struct _rknn_matmul_io_attr {
    rknn_matmul_tensor_attr A;
    rknn_matmul_tensor_attr B;
    rknn_matmul_tensor_attr C;
} rknn_matmul_io_attr;

typedef struct rknn_matmul_info_t {
    int32_t M;
    int32_t K;
    int32_t N;
    rknn_matmul_type type;
    int16_t B_layout;
    int16_t B_quant_type;
    int16_t AC_layout;
    int16_t AC_quant_type;
    int32_t iommu_domain_id;
    int16_t group_size;
    int8_t reserved[34];
} rknn_matmul_info;

int rknn_matmul_create(rknn_matmul_ctx* ctx, rknn_matmul_info* info, rknn_matmul_io_attr* io_attr);
int rknn_matmul_set_io_mem(rknn_matmul_ctx ctx, rknn_tensor_mem* mem, rknn_matmul_tensor_attr* attr);
int rknn_matmul_run(rknn_matmul_ctx ctx);
int rknn_matmul_destroy(rknn_matmul_ctx ctx);

#ifdef __cplusplus
}
#endif
