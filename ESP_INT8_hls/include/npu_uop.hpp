#pragma once

#include "npu_types.hpp"

namespace esp_int8 {

enum uop_opcode_t : std::uint8_t {
  UOP_NOP = 0,
  UOP_LOAD_FM = 1,
  UOP_CONV = 2,
  UOP_POOL = 3,
  UOP_ADD = 4,
  UOP_AFFINE = 5,
  UOP_STORE = 6,
  UOP_END = 15,
};

enum uop_flag_bit_t : std::uint8_t {
  UOP_FLAG_BIAS_EN = 0,
  UOP_FLAG_RELU_EN = 1,
  UOP_FLAG_REQUANT_BYPASS = 2,
  UOP_FLAG_CONCAT_MODE = 3,
  UOP_FLAG_ALIAS_ENABLE = 4,
  UOP_FLAG_POOL_SAME_SCALE = 5,
  UOP_FLAG_LAST_UOP_OF_STAGE = 6,
  UOP_FLAG_RESERVED = 7,
};

enum tensor_id_t : std::uint8_t {
  TID_INPUT = 0,
  TID_POOL1 = 1,
  TID_B1_CAT = 2,
  TID_B1_ACT = 3,
  TID_L20_CAT = 4,
  TID_L20_ACT = 5,
  TID_L2B0_CAT = 6,
  TID_L2B0_ACT = 7,
  TID_POOL2 = 8,
  TID_B2_CAT = 9,
  TID_B2_ACT = 10,
  TID_L30_CAT = 11,
  TID_L30_ACT = 12,
  TID_L3B0_CAT = 13,
  TID_L3B0_ACT = 14,
  TID_B3_CAT = 15,
  TID_B3_ACT = 16,
  TID_OUT = 17,
  TID_POOL_TMP = 18,
  TID_INVALID = 0xff,
};

enum local_scratch_id_t : std::uint8_t {
  LS_C1 = 0x80,
  LS_A = 0x81,
  LS_B = 0x82,
  LS_TMP = 0x83,
};

struct uop_t {
  u8_t opcode;
  u8_t flags;
  u8_t src0_tensor;
  u8_t src1_tensor;
  u8_t dst_tensor;
  u8_t param_id;
  u8_t act_type;
  u8_t reserved0;
  u16_t in_h;
  u16_t in_w;
  u16_t in_c;
  u16_t out_c;
  u8_t kernel;
  u8_t stride;
  u8_t dilation;
  u8_t padding;
  u16_t c_offset;
  u16_t valid_c;
  u16_t qparam_id;
  u16_t reserved1;
  u32_t reserved2;
};

inline bool uop_flag(const uop_t& uop, unsigned bit) {
#pragma HLS INLINE
  return ((uop.flags.to_uint() >> bit) & 0x1U) != 0;
}

inline bool uop_has_bias(const uop_t& uop) {
#pragma HLS INLINE
  return uop_flag(uop, UOP_FLAG_BIAS_EN);
}

inline bool uop_has_relu(const uop_t& uop) {
#pragma HLS INLINE
  return uop_flag(uop, UOP_FLAG_RELU_EN);
}

inline bool uop_requant_bypass(const uop_t& uop) {
#pragma HLS INLINE
  return uop_flag(uop, UOP_FLAG_REQUANT_BYPASS);
}

inline bool uop_concat_mode(const uop_t& uop) {
#pragma HLS INLINE
  return uop_flag(uop, UOP_FLAG_CONCAT_MODE);
}

inline bool uop_alias_enable(const uop_t& uop) {
#pragma HLS INLINE
  return uop_flag(uop, UOP_FLAG_ALIAS_ENABLE);
}

inline bool uop_pool_same_scale(const uop_t& uop) {
#pragma HLS INLINE
  return uop_flag(uop, UOP_FLAG_POOL_SAME_SCALE);
}

inline bool uop_last_of_stage(const uop_t& uop) {
#pragma HLS INLINE
  return uop_flag(uop, UOP_FLAG_LAST_UOP_OF_STAGE);
}

inline bool tensor_is_global(u8_t tensor_id) {
#pragma HLS INLINE
  return tensor_id <= 0x3f;
}

inline bool tensor_is_scratch(u8_t tensor_id) {
#pragma HLS INLINE
  return tensor_id >= 0x80 && tensor_id <= 0x8f;
}

static_assert(sizeof(uop_t) == 32, "uop_t must be 32 bytes");

}  // namespace esp_int8
