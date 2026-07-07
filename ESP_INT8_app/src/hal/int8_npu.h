#ifndef ESP_INT8_NPU_H
#define ESP_INT8_NPU_H

#include "xespnet_encoder_int8_core.h"
#include "xil_types.h"

typedef struct {
    u32 magic;
    u32 version;
    u32 tensor_desc_count;
    u32 scale_desc_count;
    u32 conv_desc_count;
    u32 affine_desc_count;
    u32 add_desc_count;
    u32 pool_desc_count;
    u32 uop_count;
    u32 exec_plan_count;
    u32 tensor_desc_offset;
    u32 scale_desc_offset;
    u32 conv_desc_offset;
    u32 affine_desc_offset;
    u32 add_desc_offset;
    u32 pool_desc_offset;
    u32 uop_offset;
    u32 weight_data_offset;
    u32 conv_qparam_offset;
    u32 affine_qparam_offset;
    u32 add_qparam_offset;
    u32 pool_qparam_offset;
} Int8ParamBlobHeader;

typedef struct {
    XEspnet_encoder_int8_core ip;
} Int8NpuContext;

int int8_npu_init(Int8NpuContext *ctx);
int int8_npu_read_param_header(const u8 *param_blob, u32 param_bytes,
                               Int8ParamBlobHeader *header);
int int8_npu_validate_param_header(const Int8ParamBlobHeader *header,
                                   u32 param_bytes);
int int8_npu_run_init(Int8NpuContext *ctx, UINTPTR param_addr, u32 uop_count,
                      u32 timeout_polls);
int int8_npu_run_infer(Int8NpuContext *ctx, UINTPTR input_addr,
                       UINTPTR output_addr, UINTPTR param_addr, u32 uop_count,
                       u32 timeout_polls);
int int8_npu_run_debug(Int8NpuContext *ctx, UINTPTR input_addr,
                       UINTPTR output_addr, UINTPTR param_addr, u32 uop_count,
                       u32 stop_after_uop, u32 dump_tensor_id,
                       u32 dump_words, u32 timeout_polls, const char *tag);
int int8_npu_run_profile_prefix(Int8NpuContext *ctx, UINTPTR input_addr,
                                UINTPTR output_addr, UINTPTR param_addr,
                                u32 uop_count, u32 stop_after_pc,
                                u32 timeout_polls, const char *tag);
void int8_npu_dump_regs(const Int8NpuContext *ctx, const char *tag);
void int8_npu_dump_profile_regs(const Int8NpuContext *ctx, const char *tag);
u32 int8_npu_read_ap_ctrl(const Int8NpuContext *ctx);

#endif
