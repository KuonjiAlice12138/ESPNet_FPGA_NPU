#pragma once

#include <ap_int.h>
#include <cstdint>
#include <hls_stream.h>

namespace esp_int8 {

using u8_t = ap_uint<8>;
using u16_t = ap_uint<16>;
using u32_t = ap_uint<32>;
using u64_t = ap_uint<64>;

using i8_t = ap_int<8>;
using i16_t = ap_int<16>;
using i32_t = ap_int<32>;
using i64_t = ap_int<64>;

using axi_vec_t = ap_uint<256>;
using act_vec_t = ap_uint<256>;
using wgt_vec_t = ap_uint<256>;
using psum_vec_t = ap_uint<32 * 32>;
using acc_t = ap_int<32>;

enum mode_t : std::uint32_t {
  MODE_IDLE = 0,
  MODE_INIT = 1,
  MODE_RUN = 2,
};

enum act_t : std::uint8_t {
  ACT_NONE = 0,
  ACT_RELU = 1,
};

enum post_mode_t : std::uint8_t {
  POST_CONV = 0,
  POST_ADD = 1,
  POST_AFFINE = 2,
};

enum bank_id_t : std::uint8_t {
  BANK_FMEM0 = 0,
  BANK_FMEM1 = 1,
  BANK_FMEM2 = 2,
  BANK_BRAM_SCR0 = 0x80,
  BANK_BRAM_SCR1 = 0x81,
};

enum error_code_t : std::uint16_t {
  ERR_NONE = 0,
  ERR_OK = 0,
  ERR_INVALID_MODE = 1,
  ERR_BAD_BLOB = 2,
  ERR_BAD_BLOB_MAGIC = 2,
  ERR_UNSUPPORTED_VERSION = 2,
  ERR_UOP_DECODE = 3,
  ERR_TENSOR_DESC_RANGE = 4,
  ERR_PARAM_DESC_RANGE = 5,
  ERR_BANK_OVERFLOW = 6,
  ERR_UNSUPPORTED_OPCODE = 7,
};

struct tensor_desc_t {
  u8_t bank_id;
  u8_t elem_bytes;
  u16_t reserved0;     // physical channel stride when non-zero
  u32_t base_offset;
  u16_t h;
  u16_t w;
  u16_t c;
  u16_t reserved1;     // channel offset inside physical row
};

struct scale_desc_t {
  i32_t mult;
  u8_t shift;
  u8_t reserved[3];
};

struct conv_param_desc_t {
  u32_t weight_offset;
  u32_t bias_offset;
  u32_t requant_offset;
  u32_t reserved;
};

struct affine_param_desc_t {
  u32_t affine_offset;
  u32_t reserved0;
  u32_t reserved1;
  u32_t reserved2;
};

struct add_param_desc_t {
  u32_t add_offset;
  u32_t reserved0;
  u32_t reserved1;
  u32_t reserved2;
};

struct pool_param_desc_t {
  u32_t pool_offset;
  u32_t reserved0;
  u32_t reserved1;
  u32_t reserved2;
};

struct conv_qparam_t {
  i32_t bias[32];
  i32_t mult[32];
  u8_t shift[32];
  u8_t reserved[32];
};

struct affine_qparam_t {
  i32_t mul[32];
  i32_t bias[32];
  u8_t shift[32];
  u8_t reserved[32];
};

struct add_qparam_t {
  i32_t mult;
  u8_t shift;
  u8_t act_type;
  u8_t requant_bypass;
  u8_t reserved0;
  u32_t src_scale_id_a;
  u32_t src_scale_id_b;
  u32_t dst_scale_id;
  u32_t reserved1;
};

struct pool_qparam_t {
  u8_t kernel;
  u8_t stride;
  u8_t same_scale;
  u8_t act_type;
  u32_t src_scale_id;
  u32_t dst_scale_id;
  i32_t mult;
  u8_t shift;
  u8_t reserved[11];
};

struct param_blob_header_t {
  u32_t magic;
  u32_t version;
  u32_t tensor_desc_count;
  u32_t scale_desc_count;
  u32_t conv_desc_count;       // v1: conv_param_desc_count; v3: conv_exec_desc_count
  u32_t affine_desc_count;
  u32_t add_desc_count;
  u32_t pool_desc_count;
  u32_t uop_count;
  u32_t reserved0;             // v3: exec_entry_count
  u32_t tensor_desc_offset;
  u32_t scale_desc_offset;
  u32_t conv_desc_offset;      // v1 only
  u32_t affine_desc_offset;    // v1 only
  u32_t add_desc_offset;       // v1 only
  u32_t pool_desc_offset;      // v1 only
  u32_t uop_offset;
  u32_t weight_data_offset;    // v1: logical weights; v3: active-OC packed weights
  u32_t conv_qparam_offset;
  u32_t affine_qparam_offset;
  u32_t add_qparam_offset;
  u32_t pool_qparam_offset;
  // v3 reserved1 mapping:
  // [0] conv_exec_desc_offset
  // [1] window_sched_desc_offset
  // [2] window_pack_cmd_offset
  // [3] row_consumer_desc_offset
  // [4] reserved/store/fusion offset
  // [5] exec_plan_offset
  // [6] window_sched_count
  // [7] window_pack_cmd_count
  // [8] reserved/store/fusion count
  // [9] row_consumer_count
  u32_t reserved1[10];
};

struct conv_cfg_t {
  u16_t in_h;
  u16_t in_w;
  u16_t in_c;
  u16_t out_c;
  ap_uint<2> kernel;
  ap_uint<2> stride;
  ap_uint<5> dilation;
  ap_uint<1> bias_en;
};

struct post_cfg_t {
  ap_uint<2> mode;
  ap_uint<2> act_type;
  ap_uint<1> requant_bypass;
  ap_uint<6> valid_tm;
};

using conv_q_t = conv_qparam_t;
using aff_q_t = affine_qparam_t;
using add_q_t = add_qparam_t;
using pool_q_t = pool_qparam_t;

static_assert(sizeof(tensor_desc_t) == 16, "tensor_desc_t must be 16 bytes");
static_assert(sizeof(scale_desc_t) == 8, "scale_desc_t must be 8 bytes");
static_assert(sizeof(conv_param_desc_t) == 16, "conv_param_desc_t must be 16 bytes");
static_assert(sizeof(affine_param_desc_t) == 16, "affine_param_desc_t must be 16 bytes");
static_assert(sizeof(add_param_desc_t) == 16, "add_param_desc_t must be 16 bytes");
static_assert(sizeof(pool_param_desc_t) == 16, "pool_param_desc_t must be 16 bytes");
static_assert(sizeof(param_blob_header_t) == 128, "param_blob_header_t must be 128 bytes");

}  // namespace esp_int8
