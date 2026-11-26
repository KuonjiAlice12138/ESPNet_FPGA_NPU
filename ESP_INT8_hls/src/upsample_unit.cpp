#include "../include/npu_config.hpp"
#include "../include/npu_types.hpp"
#include "../include/npu_uop.hpp"
#ifndef __SYNTHESIS__
#include <cassert>
#endif

namespace esp_int8 {

static act_vec_t s_prev_logits_row[MAX_ENCODER_OUT_W];
static bool s_prev_logits_valid = false;
static axi_vec_t s_mask_pack_word = 0;
static unsigned s_mask_pack_fill = 0;
static u32_t s_mask_word_index = 0;
static unsigned s_next_mask_row = 0;
static unsigned s_next_logits_row = 0;
static unsigned s_y_index = 0;
static int s_y_phase = 0;

static i16_t byte_to_i16_signed(u8_t value) {
#pragma HLS INLINE
  i8_t signed_value = 0;
  signed_value.range(7, 0) = value.range(7, 0);
  return static_cast<i16_t>(signed_value);
}

static void output_axis_map(unsigned index, int phase, unsigned size,
                             unsigned& lo, unsigned& hi, unsigned& fraction) {
#pragma HLS INLINE
  if (phase < 0) {
    lo = hi = 0;
    fraction = 0;
  } else if (index >= size - 1U) {
    lo = hi = size - 1U;
    fraction = 0;
  } else {
    lo = index;
    hi = index + 1U;
    fraction = static_cast<unsigned>(phase);
  }
}

static void copy_logits_row(const act_vec_t src[MAX_FM_W],
                            act_vec_t dst[MAX_ENCODER_OUT_W], unsigned width) {
#pragma HLS INLINE
  for (int col = 0; col < MAX_ENCODER_OUT_W; ++col) {
#pragma HLS PIPELINE off
    if (static_cast<unsigned>(col) >= width) { break; }
    dst[col] = src[col];
  }
}

static i16_t binary_corner_difference(const act_vec_t& packed) {
#pragma HLS INLINE
  const i16_t class0 = byte_to_i16_signed(static_cast<u8_t>(packed.range(7, 0)));
  const i16_t class1 = byte_to_i16_signed(static_cast<u8_t>(packed.range(15, 8)));
  return static_cast<i16_t>(class1 - class0);
}

static ap_uint<64> interpolate_binary_segment(i32_t left,
                                              i32_t right,
                                              bool clipped,
                                              int pixel_count, int phase,
                                              int step, int denominator) {
#pragma HLS INLINE off
  u8_t labels[UPSAMPLE_SCALE];
#pragma HLS ARRAY_PARTITION variable=labels complete dim=1
  const i32_t delta = right - left;
  for (int pixel = 0; pixel < UPSAMPLE_SCALE; ++pixel) {
#pragma HLS PIPELINE II=1
    u8_t label = 0;
    if (pixel < pixel_count) {
      const int wx1 = clipped ? 0 : phase + pixel * step;
      // The original two scores are linear, with no intermediate rounding.
      // |difference| <= 255*512*512; there is no intermediate score rounding.
      const i32_t difference = left * denominator + delta * static_cast<i32_t>(wx1);
      label = difference > 0 ? static_cast<u8_t>(1) : static_cast<u8_t>(0);
    }
    labels[pixel] = label;
  }
  ap_uint<64> packed = 0;
  for (int pixel = 0; pixel < UPSAMPLE_SCALE; ++pixel) {
#pragma HLS UNROLL
    packed.range(pixel * 8 + 7, pixel * 8) = labels[pixel];
  }
  return packed;
}

static ap_uint<64> interpolate_class_segment(const act_vec_t& top0,
                                              const act_vec_t& top1,
                                              const act_vec_t& bottom0,
                                              const act_vec_t& bottom1,
                                              int wy0, int wy1, bool clipped,
                                              int pixel_count, unsigned classes,
                                              int phase, int step, int denominator) {
#pragma HLS INLINE off
  u8_t labels[UPSAMPLE_SCALE];
#pragma HLS ARRAY_PARTITION variable=labels complete dim=1
  for (int pixel = 0; pixel < UPSAMPLE_SCALE; ++pixel) {
#pragma HLS PIPELINE off
    i32_t best = static_cast<i32_t>(-0x7fffffff);
    u8_t label = 0;
    const int wx1 = clipped ? 0 : phase + pixel * step;
    for (int group = 0; group < MAX_CLASS_C / UPSAMPLE_CLASS_LANES; ++group) {
#pragma HLS PIPELINE off
      if (static_cast<unsigned>(group * UPSAMPLE_CLASS_LANES) >= classes) { break; }
      for (int lane = 0; lane < UPSAMPLE_CLASS_LANES; ++lane) {
#pragma HLS UNROLL
        const int c = group * UPSAMPLE_CLASS_LANES + lane;
        if (static_cast<unsigned>(c) < classes) {
          const i32_t left =
              static_cast<i32_t>(byte_to_i16_signed(top0.range(c * 8 + 7, c * 8))) * wy0 +
              static_cast<i32_t>(byte_to_i16_signed(bottom0.range(c * 8 + 7, c * 8))) * wy1;
          const i32_t right =
              static_cast<i32_t>(byte_to_i16_signed(top1.range(c * 8 + 7, c * 8))) * wy0 +
              static_cast<i32_t>(byte_to_i16_signed(bottom1.range(c * 8 + 7, c * 8))) * wy1;
          const i32_t score = left * (denominator - wx1) + right * wx1;
          if (score > best) { best = score; label = static_cast<u8_t>(c); }
        }
      }
    }
    labels[pixel] = pixel < pixel_count ? label : u8_t(0);
  }
  ap_uint<64> packed = 0;
  for (int pixel = 0; pixel < UPSAMPLE_SCALE; ++pixel) {
#pragma HLS UNROLL
    packed.range(pixel * 8 + 7, pixel * 8) = labels[pixel];
  }
  return packed;
}

static void emit_output_row(axi_vec_t* gmem_frame_out,
                             const act_vec_t curr_row[MAX_FM_W], bool use_prev_row,
                             unsigned in_w, unsigned out_w, unsigned classes,
                             int wy0, int wy1, int denominator, int step, int origin) {
#pragma HLS INLINE off
  int cached_x0 = -1;
  int cached_x1 = -1;
  unsigned out_col = 0;
  int phase = origin;
  act_vec_t top0 = 0, top1 = 0, bottom0 = 0, bottom1 = 0;
  for (int segment = 0; segment <= MAX_ENCODER_OUT_W; ++segment) {
#pragma HLS PIPELINE off
    if (static_cast<unsigned>(segment) > in_w) { break; }
    const bool last = static_cast<unsigned>(segment) == in_w;
    const bool clipped = segment == 0 || last;
    const int x0 = segment == 0 ? 0 : segment - 1;
    const int x1 = last ? segment - 1 : segment;
    if (x0 != cached_x0 || x1 != cached_x1) {
      if (x0 == cached_x1) {
        top0 = top1;
        bottom0 = bottom1;
      } else {
        bottom0 = curr_row[x0];
        top0 = use_prev_row ? s_prev_logits_row[x0] : bottom0;
      }
      if (x1 == x0) {
        top1 = top0;
        bottom1 = bottom0;
      } else {
        bottom1 = curr_row[x1];
        top1 = use_prev_row ? s_prev_logits_row[x1] : bottom1;
      }
      cached_x0 = x0;
      cached_x1 = x1;
    }
    // A 30->512 interval has at most 18 pixels, served in <=3 eight-pixel chunks.
    // Native 8x uses one chunk (four at the clipped edges), as in the fast path.
    for (int chunk = 0; chunk < 3; ++chunk) {
#pragma HLS PIPELINE off
      if (out_col >= out_w || (segment == 0 ? phase >= 0 : !last && phase >= denominator)) {
        break;
      }
      int pixels = 0;
      for (int pixel = 0; pixel < UPSAMPLE_SCALE; ++pixel) {
#pragma HLS UNROLL
        const int sample_phase = phase + pixel * step;
        if (out_col + pixel < out_w &&
            (segment == 0 ? sample_phase < 0 : last || sample_phase < denominator)) {
          ++pixels;
        }
      }
      ap_uint<64> labels;
      if (classes == 2U) {
        const i32_t left = static_cast<i32_t>(binary_corner_difference(top0)) * wy0 +
                          static_cast<i32_t>(binary_corner_difference(bottom0)) * wy1;
        const i32_t right = static_cast<i32_t>(binary_corner_difference(top1)) * wy0 +
                           static_cast<i32_t>(binary_corner_difference(bottom1)) * wy1;
        labels = interpolate_binary_segment(left, right, clipped, pixels, phase, step, denominator);
      } else {
        labels = interpolate_class_segment(top0, top1, bottom0, bottom1,
                                           wy0, wy1, clipped, pixels, classes, phase, step, denominator);
      }
      axi_vec_t completed_word = 0;
      bool completed = false;
      for (int pixel = 0; pixel < UPSAMPLE_SCALE; ++pixel) {
#pragma HLS PIPELINE II=1
        if (pixel >= pixels) { break; }
        const u8_t label = labels.range(7, 0);
        labels >>= 8;
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
      out_col += static_cast<unsigned>(pixels);
      phase += pixels * step;
    }
    if (segment != 0 && phase >= denominator) { phase -= denominator; }
  }
#ifndef __SYNTHESIS__
  assert(out_col == out_w);
#endif
}

void upsample_fused_begin() {
#pragma HLS INLINE off
  s_prev_logits_valid = false;
  s_mask_pack_word = 0;
  s_mask_pack_fill = 0;
  s_mask_word_index = 0;
  s_next_mask_row = 0;
  s_next_logits_row = 0;
  s_y_index = 0;
  runtime_io_t io;
  runtime_get_io(io);
  s_y_phase = io.axis_origin;
}

bool upsample_fused_consume_logits_row(axi_vec_t* gmem_frame_out,
                                       u16_t encoder_row,
                                       u8_t valid_c,
                                       const act_vec_t row_buf[MAX_FM_W]) {
#pragma HLS INLINE off
#pragma HLS BIND_STORAGE variable=s_prev_logits_row type=ram_2p impl=lutram
#pragma HLS RESET variable=s_prev_logits_row off
  static_assert(MAX_CLASS_C == 32 && UPSAMPLE_SCALE == 8, "Native INT8 output capacity");
  runtime_profile_t profile;
  runtime_io_t io;
  runtime_get_profile(profile);
  runtime_get_io(io);
  if (!runtime_profile_ready() || !runtime_io_ready() ||
      (valid_c.to_uint() != 2U && valid_c.to_uint() != 20U) ||
      encoder_row.to_uint() >= profile.logits_h || encoder_row.to_uint() != s_next_logits_row) {
    return false;
  }

  const unsigned row = encoder_row.to_uint();
  // The last source row can own an interior interval plus its clipped bottom edge.
  for (int emit = 0; emit < 32; ++emit) {
#pragma HLS PIPELINE off
    if (s_next_mask_row >= io.mask_h) { break; }
    unsigned y0, y1, fraction;
    output_axis_map(s_y_index, s_y_phase, profile.logits_h, y0, y1, fraction);
    if (y1 > row) { break; }
    const bool use_prev = y0 != y1;
    if (y1 != row || (use_prev && (!s_prev_logits_valid || y0 + 1U != row))) {
      return false;
    }
    emit_output_row(gmem_frame_out, row_buf, use_prev, profile.logits_w, io.mask_w,
        valid_c.to_uint(), io.axis_denominator - fraction, fraction,
        io.axis_denominator, io.axis_step, io.axis_origin);
    ++s_next_mask_row;
    s_y_phase += io.axis_step;
    if (s_y_phase >= static_cast<int>(io.axis_denominator)) {
      s_y_phase -= io.axis_denominator;
      ++s_y_index;
    }
  }

#ifndef __SYNTHESIS__
  if (row == profile.logits_h - 1U) {
    assert(s_mask_pack_fill == 0U);
    assert(s_mask_word_index.to_uint() == io.mask_bytes / AXI_WORD_BYTES);
  }
#endif
  copy_logits_row(row_buf, s_prev_logits_row, profile.logits_w);
  s_prev_logits_valid = true;
  ++s_next_logits_row;
  return true;
}

}  // namespace esp_int8
