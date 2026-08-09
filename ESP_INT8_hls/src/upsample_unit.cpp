#include "../include/npu_config.hpp"
#include "../include/npu_types.hpp"
#include "../include/npu_uop.hpp"

namespace esp_int8 {

static act_vec_t s_prev_logits_row[ENCODER_OUT_W];
static bool s_prev_logits_valid = false;

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

static i16_t packed_logit(const act_vec_t& packed, int class_id) {
#pragma HLS INLINE
  const int lo = class_id * 8;
  const u8_t raw = packed.range(lo + 7, lo);
  return byte_to_i16_signed(raw);
}

static u8_t interpolate_argmax_label(const act_vec_t curr_row[ENCODER_OUT_W],
                                     bool use_prev_row,
                                     int col,
                                     int wy0,
                                     int wy1,
                                     u8_t valid_c) {
#pragma HLS INLINE off
  int x0 = 0;
  int x1 = 0;
  int wx0 = 16;
  int wx1 = 0;
  bilinear_axis_map(col, ENCODER_OUT_W, x0, x1, wx0, wx1);

  const act_vec_t top_packed0 =
      use_prev_row ? s_prev_logits_row[x0] : curr_row[x0];
  const act_vec_t top_packed1 =
      use_prev_row ? s_prev_logits_row[x1] : curr_row[x1];
  const act_vec_t bottom_packed0 = curr_row[x0];
  const act_vec_t bottom_packed1 = curr_row[x1];

  i32_t best_value = static_cast<i32_t>(-0x7fffffff);
  u8_t best_class = 0;
  const unsigned class_count = valid_c.to_uint();
  for (int group = 0; group < MAX_CLASS_C / UPSAMPLE_CLASS_LANES; ++group) {
    if (static_cast<unsigned>(group * UPSAMPLE_CLASS_LANES) >= class_count) {
      break;
    }
    for (int lane = 0; lane < UPSAMPLE_CLASS_LANES; ++lane) {
#pragma HLS UNROLL
      const int class_id = group * UPSAMPLE_CLASS_LANES + lane;
      if (static_cast<unsigned>(class_id) < class_count) {
        const i32_t top =
            static_cast<i32_t>(packed_logit(top_packed0, class_id)) *
                static_cast<i32_t>(wx0) +
            static_cast<i32_t>(packed_logit(top_packed1, class_id)) *
                static_cast<i32_t>(wx1);
        const i32_t bottom =
            static_cast<i32_t>(packed_logit(bottom_packed0, class_id)) *
                static_cast<i32_t>(wx0) +
            static_cast<i32_t>(packed_logit(bottom_packed1, class_id)) *
                static_cast<i32_t>(wx1);
        const i32_t interp =
            top * static_cast<i32_t>(wy0) +
            bottom * static_cast<i32_t>(wy1);
        if (interp > best_value) {
          best_value = interp;
          best_class = static_cast<u8_t>(class_id);
        }
      }
    }
  }
  return best_class;
}

static axi_vec_t pack_mask_word(const act_vec_t curr_row[ENCODER_OUT_W],
                                bool use_prev_row,
                                int out_col_base,
                                int wy0,
                                int wy1,
                                u8_t valid_c) {
#pragma HLS INLINE off
  axi_vec_t word = 0;
  for (int lane = 0; lane < AXI_WORD_BYTES; ++lane) {
#pragma HLS PIPELINE II=1
    const int out_col = out_col_base + lane;
    const u8_t label = interpolate_argmax_label(
        curr_row, use_prev_row, out_col, wy0, wy1, valid_c);
    word.range(lane * 8 + 7, lane * 8) = label;
  }
  return word;
}

static void emit_fullres_rows(axi_vec_t* gmem_frame_out,
                              const act_vec_t curr_row[ENCODER_OUT_W],
                              bool use_prev_row,
                              int out_row_begin,
                              int row_count,
                              u8_t valid_c) {
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

    for (int word_col = 0; word_col < FULLRES_MASK_W / AXI_WORD_BYTES; ++word_col) {
#pragma HLS PIPELINE off
      const int out_col_base = word_col * AXI_WORD_BYTES;
      const u32_t word_idx =
          static_cast<u32_t>(out_row * (FULLRES_MASK_W / AXI_WORD_BYTES) + word_col);
      gmem_frame_out[word_idx] =
          pack_mask_word(curr_row, use_prev_row, out_col_base, wy0, wy1, valid_c);
    }
  }
}

void upsample_fused_begin() {
#pragma HLS INLINE off
  s_prev_logits_valid = false;
}

void upsample_fused_consume_logits_row(axi_vec_t* gmem_frame_out,
                                       u16_t encoder_row,
                                       u8_t valid_c,
                                       const act_vec_t row_buf[MAX_FM_W]) {
#pragma HLS INLINE off
#pragma HLS BIND_STORAGE variable=s_prev_logits_row type=ram_2p impl=lutram
#pragma HLS RESET variable=s_prev_logits_row off
  if (valid_c.to_uint() < 2U || valid_c.to_uint() > static_cast<unsigned>(MAX_CLASS_C)) {
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
          gmem_frame_out, row_buf, use_prev_row, out_row_begin, row_count, valid_c);
    }
  }

  copy_logits_row(row_buf, s_prev_logits_row);
  s_prev_logits_valid = true;
}

}  // namespace esp_int8
