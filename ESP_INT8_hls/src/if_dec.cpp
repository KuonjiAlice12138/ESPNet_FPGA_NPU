#include "../include/npu_config.hpp"
#include "../include/npu_uop.hpp"

namespace esp_int8 {

bool param_dma_ready();
error_code_t param_dma_error();
bool param_dma_get_header(param_blob_header_t& header);
u32_t param_dma_uop_count();
bool param_dma_get_uop(u16_t uop_id, uop_t& uop);

static uop_t s_current_uop;
static u16_t s_current_uop_id = 0;
static error_code_t s_error = ERR_NONE;
static bool s_valid = false;

static bool opcode_supported(u8_t opcode) {
#pragma HLS INLINE
    const unsigned op = opcode.to_uint();
    return op == static_cast<unsigned>(UOP_NOP) ||
           op == static_cast<unsigned>(UOP_LOAD_FM) ||
           op == static_cast<unsigned>(UOP_CONV) ||
           op == static_cast<unsigned>(UOP_POOL) ||
           op == static_cast<unsigned>(UOP_ADD) ||
           op == static_cast<unsigned>(UOP_AFFINE) ||
           op == static_cast<unsigned>(UOP_STORE) ||
           op == static_cast<unsigned>(UOP_END);
}

static bool tensor_id_valid_or_none(u8_t tensor_id) {
#pragma HLS INLINE
    return tensor_id.to_uint() == static_cast<unsigned>(TID_INVALID) ||
           tensor_is_global(tensor_id) ||
           tensor_is_scratch(tensor_id);
}

static error_code_t validate_uop(const uop_t& uop, const param_blob_header_t& header) {
#pragma HLS INLINE
    if (!opcode_supported(uop.opcode)) {
        return ERR_UNSUPPORTED_OPCODE;
    }
    if (uop.reserved0 != 0 || uop.reserved1 != 0 || uop.reserved2 != 0) {
        return ERR_UOP_DECODE;
    }
    if (uop.qparam_id != 0) {
        return ERR_UOP_DECODE;
    }
    if (!tensor_id_valid_or_none(uop.src0_tensor) ||
        !tensor_id_valid_or_none(uop.src1_tensor) ||
        !tensor_id_valid_or_none(uop.dst_tensor)) {
        return ERR_TENSOR_DESC_RANGE;
    }

    switch (uop.opcode.to_uint()) {
        case static_cast<unsigned>(UOP_CONV):
            return (uop.param_id < header.conv_desc_count) ? ERR_NONE : ERR_PARAM_DESC_RANGE;
        case static_cast<unsigned>(UOP_POOL):
            return (uop.param_id < header.pool_desc_count) ? ERR_NONE : ERR_PARAM_DESC_RANGE;
        case static_cast<unsigned>(UOP_ADD):
            return (uop.param_id < header.add_desc_count) ? ERR_NONE : ERR_PARAM_DESC_RANGE;
        case static_cast<unsigned>(UOP_AFFINE):
            return (uop.param_id < header.affine_desc_count) ? ERR_NONE : ERR_PARAM_DESC_RANGE;
        default:
            return ERR_NONE;
    }
}

void instruction_fetch_decode(const axi_vec_t* gmem_param, u32_t uop_count) {
#pragma HLS INLINE off
    param_blob_header_t header;
    s_error = ERR_NONE;
    s_valid = false;
    s_current_uop_id = 0;
    s_current_uop = uop_t();

    if (!param_dma_ready()) {
        const error_code_t err = param_dma_error();
        s_error = (err == ERR_NONE) ? ERR_BAD_BLOB : err;
        return;
    }
    if (!param_dma_get_header(header)) {
        s_error = ERR_BAD_BLOB;
        return;
    }
    if (uop_count != param_dma_uop_count()) {
        s_error = ERR_UOP_DECODE;
        return;
    }

    for (int i = 0; i < MAX_UOP_COUNT; ++i) {
#pragma HLS PIPELINE II=1
        if (i >= static_cast<int>(uop_count.to_uint())) {
            break;
        }

        uop_t uop;
        if (!param_dma_get_uop(static_cast<u16_t>(i), uop)) {
            s_error = ERR_UOP_DECODE;
            s_current_uop_id = static_cast<u16_t>(i);
            return;
        }

        const error_code_t err = validate_uop(uop, header);
        s_current_uop = uop;
        s_current_uop_id = static_cast<u16_t>(i);
        s_valid = true;
        if (err != ERR_NONE) {
            s_error = err;
            return;
        }
        if (uop.opcode.to_uint() == static_cast<unsigned>(UOP_END)) {
            return;
        }
    }
}

error_code_t if_dec_error() {
#pragma HLS INLINE
    return s_error;
}

u16_t if_dec_current_uop_id() {
#pragma HLS INLINE
    return s_current_uop_id;
}

bool if_dec_get_current_uop(uop_t& uop) {
#pragma HLS INLINE
    uop = s_current_uop;
    return s_valid;
}

}  // namespace esp_int8
