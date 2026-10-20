#include "../include/npu_config.hpp"
#include "../include/npu_types.hpp"
#include "../include/npu_uop.hpp"
#ifndef __SYNTHESIS__
#include <cassert>
#endif

namespace esp_int8 {

static act_vec_t s_prev_logits_row[ENCODER_OUT_W];
static bool s_prev_logits_valid = false;
static axi_vec_t s_mask_pack_word = 0;
static unsigned s_mask_pack_fill = 0;
static u32_t s_mask_word_index = 0;

static i16_t byte_to_i16_signed(u8_t value) {
#pragma HLS INLINE
  i8_t signed_value = 0;
  signed_value.range(7, 0) = value.range(7, 0);
  return static_cast<i16_t>(signed_value);
}

static void bilinear_axis_map(int out_idx,
                              int in_size,
                              int& idx0,
                              int& idx1,
                              int& w0,
                              int& w1) {
#pragma HLS INLINE
  const int clip_high_start = in_size * UPSAMPLE_SCALE - (UPSAMPLE_SCALE / 2);
  if (out_idx < (UPSAMPLE_SCALE / 2)) {
    idx0 = 0;
    idx1 = 0;
    w0 = 16;
    w1 = 0;
    return;
  }
  if (out_idx >= clip_high_start) {
    idx0 = in_size - 1;
    idx1 = in_size - 1;
    w0 = 16;
    w1 = 0;
    return;
  }

  const int numer = 2 * out_idx - (UPSAMPLE_SCALE - 1);
  idx0 = numer >> 4;
  idx1 = idx0 + 1;
  w1 = numer & 0x0f;
  w0 = 16 - w1;
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

static ap_uint<64> interpolate_binary_segment(i32_t left,
                                              i32_t right,
                                              bool clipped,
                                              int pixel_count) {
#pragma HLS INLINE off
  u8_t labels[UPSAMPLE_SCALE];
#pragma HLS ARRAY_PARTITION variable=labels complete dim=1
  const i32_t delta = right - left;
  for (int pixel = 0; pixel < UPSAMPLE_SCALE; ++pixel) {
#pragma HLS PIPELINE II=1
    u8_t label = 0;
    if (pixel < pixel_count) {
      const int wx1 = clipped ? 0 : 1 + pixel * 2;
      // The original two scores are linear, with no intermediate rounding.
      // |difference| <= 255*256, and a tie must still select class 0.
      const i32_t difference = left * 16 + delta * static_cast<i32_t>(wx1);
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

static void emit_fullres_rows(axi_vec_t* gmem_frame_out,
                              const act_vec_t curr_row[ENCODER_OUT_W],
                              bool use_prev_row,
                              int out_row_begin,
                              int row_count) {
#pragma HLS INLINE off
  for (int rel_row = 0; rel_row < UPSAMPLE_SCALE; ++rel_row) {
    if (rel_row >= row_count) {
      break;
    }
    const int out_row = out_row_begin + rel_row;
    int y0 = 0;
    int y1 = 0;
    int wy0 = 16;
    int wy1 = 0;
    bilinear_axis_map(out_row, ENCODER_OUT_H, y0, y1, wy0, wy1);

    int cached_x0 = -1;
    int cached_x1 = -1;
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
      const i32_t left = static_cast<i32_t>(top0) * wy0 + static_cast<i32_t>(bottom0) * wy1;
      const i32_t right = static_cast<i32_t>(top1) * wy0 + static_cast<i32_t>(bottom1) * wy1;
      const int pixels = clipped ? UPSAMPLE_SCALE / 2 : UPSAMPLE_SCALE;
      ap_uint<64> labels = interpolate_binary_segment(left, right, clipped, pixels);
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
    }
  }
}

void upsample_fused_begin() {
#pragma HLS INLINE off
  s_prev_logits_valid = false;
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
  static_assert(MAX_CLASS_C == 2 && UPSAMPLE_SCALE == 8,
                "This branch implements exact binary 8x bilinear output only");
  if (valid_c.to_uint() != 2U) {
    return;
  }

  const unsigned row = encoder_row.to_uint();
  for (int emit_case = 0; emit_case < 2; ++emit_case) {
#pragma HLS PIPELINE off
    bool emit = false;
    bool use_prev_row = false;
    int out_row_begin = 0;
    int row_count = 0;

    if (emit_case == 0) {
      if (row == 0U) {
        emit = true;
        out_row_begin = 0;
        row_count = UPSAMPLE_SCALE / 2;
      } else if (s_prev_logits_valid) {
        emit = true;
        use_prev_row = true;
        out_row_begin = (UPSAMPLE_SCALE / 2) +
                        static_cast<int>((row - 1U) * static_cast<unsigned>(UPSAMPLE_SCALE));
        row_count = UPSAMPLE_SCALE;
      }
    } else if (row == static_cast<unsigned>(ENCODER_OUT_H - 1)) {
      emit = true;
      out_row_begin = FULLRES_MASK_H - (UPSAMPLE_SCALE / 2);
      row_count = UPSAMPLE_SCALE / 2;
    }

    if (emit) {
      emit_fullres_rows(
          gmem_frame_out, row_buf, use_prev_row, out_row_begin, row_count);
    }
  }

#ifndef __SYNTHESIS__
  if (row == static_cast<unsigned>(ENCODER_OUT_H - 1)) {
    assert(s_mask_pack_fill == 0U);
    assert(s_mask_word_index.to_uint() == OUTPUT_FRAME_AXI_WORDS);
  }
#endif
  copy_logits_row(row_buf, s_prev_logits_row);
  s_prev_logits_valid = true;
}

}  // namespace esp_int8
