#pragma once

#include "npu_types.hpp"

namespace esp_int8 {

inline i32_t round_shift(i64_t x, u8_t shift) {
#pragma HLS INLINE
  if (shift == 0) {
    return static_cast<i32_t>(x);
  }

  const i64_t bias = i64_t(1) << (shift - 1);
  if (x >= 0) {
    return static_cast<i32_t>((x + bias) >> shift);
  }
  return static_cast<i32_t>((x - bias) >> shift);
}

inline i8_t clamp_i8(i32_t x) {
#pragma HLS INLINE
  if (x > 127) {
    return 127;
  }
  if (x < -128) {
    return -128;
  }
  return static_cast<i8_t>(x);
}

inline i8_t relu_i8(i8_t x) {
#pragma HLS INLINE
  return (x < 0) ? i8_t(0) : x;
}

inline i8_t apply_act(i8_t x, u8_t act_type) {
#pragma HLS INLINE
  return (act_type.to_uint() == static_cast<unsigned>(ACT_RELU)) ? relu_i8(x) : x;
}

inline i8_t apply_act(i8_t x, ap_uint<2> act_type) {
#pragma HLS INLINE
  return (act_type.to_uint() == static_cast<unsigned>(ACT_RELU)) ? relu_i8(x) : x;
}

inline i8_t requant_i32_to_i8(i32_t acc, i32_t bias, i32_t mult, u8_t shift, u8_t act_type) {
#pragma HLS INLINE
  const i64_t biased = static_cast<i64_t>(acc) + static_cast<i64_t>(bias);
  const i64_t scaled = biased * static_cast<i64_t>(mult);
  return apply_act(clamp_i8(round_shift(scaled, shift)), act_type);
}

inline i8_t affine_i8_to_i8(i8_t x, i32_t mul, i32_t bias, u8_t shift, u8_t act_type) {
#pragma HLS INLINE
  const i64_t scaled = static_cast<i64_t>(x) * static_cast<i64_t>(mul) + static_cast<i64_t>(bias);
  return apply_act(clamp_i8(round_shift(scaled, shift)), act_type);
}

inline i8_t add_i8(i8_t a, i8_t b, const add_qparam_t& q) {
#pragma HLS INLINE
  const i32_t sum = static_cast<i32_t>(a) + static_cast<i32_t>(b);
  if (q.requant_bypass != 0) {
    return apply_act(clamp_i8(sum), q.act_type);
  }

  const i64_t scaled = static_cast<i64_t>(sum) * static_cast<i64_t>(q.mult);
  return apply_act(clamp_i8(round_shift(scaled, q.shift)), q.act_type);
}

}  // namespace esp_int8
