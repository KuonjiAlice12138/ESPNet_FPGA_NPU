#include "../include/npu_config.hpp"
#include "../include/npu_types.hpp"
#include "../include/npu_uop.hpp"
#ifndef __SYNTHESIS__
#include <cassert>
#endif

namespace esp_int8 {

// Both builds use the same engine. These are exact half-pixel coordinates.
constexpr int MASK_FRAC_BITS = FULLRES_MASK_W == 512 ? 9 : 4;
constexpr int MASK_ONE = 1 << MASK_FRAC_BITS;
constexpr int MASK_STEP = FULLRES_MASK_W == 512 ? 30 : 2;
constexpr int MASK_INITIAL = FULLRES_MASK_W == 512 ? -241 : -7;
constexpr int MASK_BORDER = FULLRES_MASK_W == 512 ? 9 : 4;
constexpr int MASK_SEGMENT_MAX = FULLRES_MASK_W == 512 ? 18 : 8;
constexpr int MASK_EMIT_MAX = FULLRES_MASK_H == 512 ? 26 : 12;
using mask_weight_t = ap_uint<MASK_FRAC_BITS>;
static_assert(ENCODER_OUT_H == 30 && ENCODER_OUT_W == 30 && MAX_CLASS_C == 2,
              "Mask reconstruction expects 30x30 binary logits");
static_assert(MASK_SEGMENT_MAX < AXI_WORD_BYTES,
              "A segment may complete at most one AXI word");

static act_vec_t s_prev_logits_row[ENCODER_OUT_W];
static bool s_prev_logits_valid = false;
static u16_t s_next_output_y = 0;
static axi_vec_t s_mask_pack_word = 0;
static unsigned s_mask_pack_fill = 0;
static u32_t s_mask_word_index = 0;

static i16_t byte_to_i16_signed(u8_t value) {
#pragma HLS INLINE
  i8_t signed_value = 0;
  signed_value.range(7, 0) = value.range(7, 0);
  return static_cast<i16_t>(signed_value);
}

static void bilinear_axis_map(int out_idx, int& idx0, int& idx1, int& w1) {
#pragma HLS INLINE
  const int numer = out_idx * MASK_STEP + MASK_INITIAL;
  if (numer < 0) {
    idx0 = 0;
    idx1 = 0;
    w1 = 0;
  } else if (numer >= (ENCODER_OUT_H - 1) * MASK_ONE) {
    idx0 = ENCODER_OUT_H - 1;
    idx1 = idx0;
    w1 = 0;
  } else {
    idx0 = numer >> MASK_FRAC_BITS;
    idx1 = idx0 + 1;
    w1 = numer & (MASK_ONE - 1);
  }
}

static void copy_logits_row(const act_vec_t src[ENCODER_OUT_W],
                            act_vec_t dst[ENCODER_OUT_W]) {
#pragma HLS INLINE
  for (int col = 0; col < ENCODER_OUT_W; ++col) {
#pragma HLS PIPELINE off
    dst[col] = src[col];
  }
}

static i16_t binary_corner_difference(const act_vec_t& packed) {
#pragma HLS INLINE
  const i16_t class0 = byte_to_i16_signed(static_cast<u8_t>(packed.range(7, 0)));
  const i16_t class1 = byte_to_i16_signed(static_cast<u8_t>(packed.range(15, 8)));
  return static_cast<i16_t>(class1 - class0);
}

static void emit_mask_segment(axi_vec_t* gmem_frame_out,
                              i32_t left, i32_t right, mask_weight_t first_wx, ap_uint<5> pixels) {
#pragma HLS INLINE off
  const i32_t delta = right - left;
  i32_t difference = left * static_cast<i32_t>(MASK_ONE) +
                     delta * first_wx;
  const i32_t step = delta * static_cast<i32_t>(MASK_STEP);
  axi_vec_t completed_word = 0;
  bool completed = false;
  for (int pixel = 0; pixel < MASK_SEGMENT_MAX; ++pixel) {
#pragma HLS PIPELINE II=1
    if (pixel >= pixels) { break; }
    // No intermediate rounding; |difference| <= 255 * MASK_ONE^2. Ties select 0.
    const u8_t label = difference > 0 ? static_cast<u8_t>(1) : static_cast<u8_t>(0);
    difference += step;
    s_mask_pack_word >>= 8;
    s_mask_pack_word.range(255, 248) = label;
    ++s_mask_pack_fill;
    if (s_mask_pack_fill == AXI_WORD_BYTES) {
      completed_word = s_mask_pack_word;
      completed = true;
      s_mask_pack_word = 0;
      s_mask_pack_fill = 0;
    }
  }
  if (completed) {
    gmem_frame_out[s_mask_word_index++] = completed_word;
  }
}

static void emit_mask_row(axi_vec_t* gmem_frame_out,
                          const act_vec_t curr_row[ENCODER_OUT_W],
                          bool use_prev_row, mask_weight_t wy1) {
#pragma HLS INLINE off
  int cached_x0 = -1;
  int cached_x1 = -1;
  int next_wx = MASK_BORDER * MASK_STEP + MASK_INITIAL;
  i16_t top0 = 0, top1 = 0, bottom0 = 0, bottom1 = 0;
  for (int segment = 0; segment <= ENCODER_OUT_W; ++segment) {
#pragma HLS PIPELINE off
    const bool clipped = segment == 0 || segment == ENCODER_OUT_W;
    const int x0 = segment == 0 ? 0 : segment - 1;
    const int x1 = segment == ENCODER_OUT_W ? segment - 1 : segment;
    if (x0 != cached_x0 || x1 != cached_x1) {
      if (x0 == cached_x1) {
        top0 = top1;
        bottom0 = bottom1;
      } else {
        bottom0 = binary_corner_difference(curr_row[x0]);
        top0 = use_prev_row ? binary_corner_difference(s_prev_logits_row[x0]) : bottom0;
      }
      if (x1 == x0) {
        top1 = top0;
        bottom1 = bottom0;
      } else {
        bottom1 = binary_corner_difference(curr_row[x1]);
        top1 = use_prev_row ? binary_corner_difference(s_prev_logits_row[x1]) : bottom1;
      }
      cached_x0 = x0;
      cached_x1 = x1;
    }
    const i32_t left = static_cast<i32_t>(top0) * MASK_ONE +
                       (static_cast<i32_t>(bottom0) - static_cast<i32_t>(top0)) * wy1;
    const i32_t right = static_cast<i32_t>(top1) * MASK_ONE +
                        (static_cast<i32_t>(bottom1) - static_cast<i32_t>(top1)) * wy1;
    // Constant quotient/remainder; the 512 build emits 17/18 pixels per interior segment.
    const int pixels = clipped ? MASK_BORDER
        : MASK_ONE / MASK_STEP + (next_wx < MASK_ONE % MASK_STEP ? 1 : 0);
    const int first_wx = clipped ? 0 : next_wx;
    if (!clipped) { next_wx += pixels * MASK_STEP - MASK_ONE; }
    emit_mask_segment(gmem_frame_out, left, right, static_cast<mask_weight_t>(first_wx),
                       static_cast<ap_uint<5>>(pixels));
  }
}

void upsample_fused_begin() {
#pragma HLS INLINE off
  s_prev_logits_valid = false;
  s_next_output_y = 0;
  s_mask_pack_word = 0;
  s_mask_pack_fill = 0;
  s_mask_word_index = 0;
}

void upsample_fused_consume_logits_row(axi_vec_t* gmem_frame_out,
                                       u16_t encoder_row,
                                       u8_t valid_c,
                                       const act_vec_t row_buf[MAX_FM_W]) {
#pragma HLS INLINE off
#pragma HLS BIND_STORAGE variable=s_prev_logits_row type=ram_2p impl=lutram
#pragma HLS RESET variable=s_prev_logits_row off
  if (valid_c.to_uint() != 2U) { return; }
  const unsigned row = encoder_row.to_uint();
  for (int emit = 0; emit < MASK_EMIT_MAX; ++emit) {
#pragma HLS PIPELINE off
    if (s_next_output_y.to_uint() >= FULLRES_MASK_H) { break; }
    int y0 = 0, y1 = 0, wy1 = 0;
    bilinear_axis_map(s_next_output_y.to_uint(), y0, y1, wy1);
    if (static_cast<unsigned>(y1) != row) { break; }
    const bool use_prev_row = y0 != y1;
    if (use_prev_row && !s_prev_logits_valid) { break; }
    emit_mask_row(gmem_frame_out, row_buf, use_prev_row, static_cast<mask_weight_t>(wy1));
    ++s_next_output_y;
  }
#ifndef __SYNTHESIS__
  if (row == static_cast<unsigned>(ENCODER_OUT_H - 1)) {
    assert(s_next_output_y.to_uint() == FULLRES_MASK_H);
    assert(s_mask_pack_fill == 0U);
    assert(s_mask_word_index.to_uint() == OUTPUT_FRAME_AXI_WORDS);
  }
#endif
  copy_logits_row(row_buf, s_prev_logits_row);
  s_prev_logits_valid = true;
}

}  // namespace esp_int8
