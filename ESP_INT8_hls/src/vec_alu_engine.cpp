#include "../include/npu_config.hpp"
#include "../include/npu_ctrl.hpp"
#include "../include/npu_q.hpp"
#include "../include/npu_uop.hpp"

namespace esp_int8 {

bool param_dma_get_fixed_exec_desc(u8_t id, fixed_exec_desc_t& desc);
bool param_dma_get_affine_qparam(u8_t param_id, u8_t block_id, aff_q_t& qparam);
bool on_chip_memory_read_packed_tile(const tensor_desc_t& desc,
                                     i32_t h,
                                     i32_t w,
                                     u16_t c_begin,
                                     u8_t valid_c,
                                     act_vec_t& packed);
bool conv_store_write_aligned_tile(const tensor_desc_t& dst,
                                   u16_t h,
                                   u16_t w,
                                   u16_t c,
                                   const act_vec_t& word);
bool conv_store_write_row_contiguous_word(const tensor_desc_t& dst,
                                          u32_t abs_offset,
                                          const axi_vec_t& word);
bool conv_store_row_contiguous_plan_ok(const tensor_desc_t& dst,
                                       u16_t width,
                                       u16_t valid_c);
bool resolve_tensor_read(u8_t tensor_id, tensor_desc_t& desc);
bool resolve_tensor_write(u8_t tensor_id, u16_t h, u16_t w, u16_t c, tensor_desc_t& desc);
bool backup_b2_src1_rows_before_write(const tensor_desc_t& src1,
                                      int write_row,
                                      bool src1_saved[MAX_FM_H]);
bool read_b2_backup_src1_tile(u16_t h,
                              u16_t w,
                              u16_t c,
                              u8_t lanes,
                              act_vec_t& packed);

static axi_vec_t s_shared_row_contig_words[ROW_CONTIG_MAX_WORDS];
static bool s_shared_b2_src1_saved[MAX_FM_H];

static i8_t vec_get_act_i8_dynamic(const act_vec_t& word, int lane) {
#pragma HLS INLINE
  const u8_t raw = word.range(lane * 8 + 7, lane * 8);
  i8_t value = 0;
  value.range(7, 0) = raw;
  return value;
}

static void vec_pack_i8_lanes(const i8_t lanes[TM], act_vec_t& word) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
  word = 0;
  for (int lane = 0; lane < TM; ++lane) {
#pragma HLS UNROLL
    u8_t raw = 0;
    raw.range(7, 0) = lanes[lane].range(7, 0);
    word.range(lane * 8 + 7, lane * 8) = raw;
  }
}

constexpr int VEC_AFF_GROUP_LANES = 8;

static void vec_alu_apply_affine_group(const act_vec_t& in_word,
                                       unsigned valid,
                                       const aff_q_t& qparam,
                                       u8_t act_type,
                                       int lane_base,
                                       i8_t out_lanes[TM]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=out_lanes complete dim=1
  i32_t mul[VEC_AFF_GROUP_LANES];
  i32_t bias[VEC_AFF_GROUP_LANES];
  u8_t shift[VEC_AFF_GROUP_LANES];
#pragma HLS ARRAY_PARTITION variable=mul complete dim=1
#pragma HLS ARRAY_PARTITION variable=bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=shift complete dim=1

  for (int i = 0; i < VEC_AFF_GROUP_LANES; ++i) {
#pragma HLS UNROLL
    const int lane = lane_base + i;
    mul[i] = qparam.mul[lane];
    bias[i] = qparam.bias[lane];
    shift[i] = qparam.shift[lane];
  }

  for (int i = 0; i < VEC_AFF_GROUP_LANES; ++i) {
#pragma HLS UNROLL factor=4
    const int lane = lane_base + i;
    i8_t out_value = 0;
    if (static_cast<unsigned>(lane) < valid) {
      const i8_t in_value = vec_get_act_i8_dynamic(in_word, lane);
      out_value = affine_i8_to_i8(in_value, mul[i], bias[i], shift[i], act_type);
    }
    out_lanes[lane] = out_value;
  }
}

static void vec_alu_apply_affine_block(const act_vec_t& in_word,
                                       u8_t valid_c,
                                       const aff_q_t& qparam,
                                       u8_t act_type,
                                       act_vec_t& out_word) {
#pragma HLS INLINE off
  out_word = 0;
  const unsigned valid = valid_c.to_uint();
  i8_t out_lanes[TM];
#pragma HLS ARRAY_PARTITION variable=out_lanes complete dim=1
  for (int group = 0; group < TM / VEC_AFF_GROUP_LANES; ++group) {
#pragma HLS PIPELINE off
    vec_alu_apply_affine_group(in_word,
                               valid,
                               qparam,
                               act_type,
                               group * VEC_AFF_GROUP_LANES,
                               out_lanes);
  }
  vec_pack_i8_lanes(out_lanes, out_word);
}

static u8_t vec_tensor_lanes(u16_t remaining_c) {
#pragma HLS INLINE
  const unsigned rem = remaining_c.to_uint();
  return static_cast<u8_t>((rem > static_cast<unsigned>(TM)) ? TM : rem);
}

static u16_t vec_tensor_phys_c(const tensor_desc_t& desc) {
#pragma HLS INLINE
  return (desc.reserved0.to_uint() == 0U) ? desc.c : desc.reserved0;
}

static bool vec_can_use_aligned_full_tile(const tensor_desc_t& desc, u16_t c_begin) {
#pragma HLS INLINE
  const unsigned phys_c = vec_tensor_phys_c(desc).to_uint();
  const unsigned start_c = desc.reserved1.to_uint() + c_begin.to_uint();
  const unsigned base_start = desc.base_offset.to_uint() + start_c;
  return phys_c >= static_cast<unsigned>(AXI_WORD_BYTES) &&
         ((phys_c & static_cast<unsigned>(AXI_WORD_BYTES - 1)) == 0U) &&
         c_begin.to_uint() + static_cast<unsigned>(AXI_WORD_BYTES) <= desc.c.to_uint() &&
         ((base_start & static_cast<unsigned>(AXI_WORD_BYTES - 1)) == 0U);
}

static bool vec_full_aligned_tile_write_plan_ok(const tensor_desc_t& desc, u16_t total_c) {
#pragma HLS INLINE off
  const unsigned total = total_c.to_uint();
  if (total == 0U || (total & static_cast<unsigned>(TM - 1)) != 0U) {
    return false;
  }

  const int c_blocks = static_cast<int>(total / static_cast<unsigned>(TM));
  for (int c_blk = 0; c_blk < MAX_C_TILE_COUNT; ++c_blk) {
#pragma HLS PIPELINE off
    if (c_blk >= c_blocks) {
      break;
    }
    const u16_t c = static_cast<u16_t>(c_blk * TM);
    if (!vec_can_use_aligned_full_tile(desc, c)) {
      return false;
    }
  }
  return true;
}

static bool vec_row_contig_b2_needs_src1_backup(const fixed_exec_desc_t& desc,
                                                const tensor_desc_t& src0,
                                                const tensor_desc_t& src1,
                                                const tensor_desc_t& src2,
                                                bool has_src2,
                                                const tensor_desc_t& dst) {
#pragma HLS INLINE
  return has_src2 &&
         desc.in_h.to_uint() == 128U &&
         desc.in_w.to_uint() == static_cast<unsigned>(ROW_CONTIG_MAX_W) &&
         desc.valid_c.to_uint() == static_cast<unsigned>(ROW_CONTIG_MAX_C) &&
         src0.base_offset.to_uint() == 0U &&
         vec_tensor_phys_c(src0).to_uint() == 64U &&
         src1.base_offset.to_uint() == static_cast<unsigned>(FMBUF_L20_BASE) &&
         vec_tensor_phys_c(src1).to_uint() == 64U &&
         src2.bank_id.to_uint() == static_cast<unsigned>(BANK_BRAM_SCR1) &&
         dst.base_offset.to_uint() == 0U &&
         vec_tensor_phys_c(dst).to_uint() == static_cast<unsigned>(ROW_CONTIG_MAX_C);
}

static bool read_block_affine_source_tile(const tensor_desc_t& src0,
                                          const tensor_desc_t& src1,
                                          const tensor_desc_t& src2,
                                          bool has_src2,
                                          u16_t h,
                                          u16_t w,
                                          u16_t c,
                                          u8_t lanes,
                                          act_vec_t& packed) {
#pragma HLS INLINE
  const unsigned c_abs = c.to_uint();
  const unsigned c0 = src0.c.to_uint();
  const unsigned c1 = src1.c.to_uint();
  if (c_abs < c0) {
    return on_chip_memory_read_packed_tile(src0,
                                           static_cast<i32_t>(h),
                                           static_cast<i32_t>(w),
                                           c,
                                           lanes,
                                           packed);
  }
  if (c_abs < c0 + c1) {
    return on_chip_memory_read_packed_tile(src1,
                                           static_cast<i32_t>(h),
                                           static_cast<i32_t>(w),
                                           static_cast<u16_t>(c_abs - c0),
                                           lanes,
                                           packed);
  }
  if (has_src2) {
    return on_chip_memory_read_packed_tile(src2,
                                           static_cast<i32_t>(h),
                                           static_cast<i32_t>(w),
                                           static_cast<u16_t>(c_abs - c0 - c1),
                                           lanes,
                                           packed);
  }
  return false;
}

static bool read_fixed_affine_source_tile(const tensor_desc_t& src0,
                                          const tensor_desc_t& src1,
                                          const tensor_desc_t& src2,
                                          bool has_src2,
                                          bool use_b2_backup,
                                          const bool src1_saved[MAX_FM_H],
                                          u16_t h,
                                          u16_t w,
                                          u16_t c,
                                          u8_t lanes,
                                          act_vec_t& packed) {
#pragma HLS INLINE
  const unsigned c_abs = c.to_uint();
  const unsigned c0 = src0.c.to_uint();
  const unsigned c1 = src1.c.to_uint();
  if (use_b2_backup && c_abs >= c0 && c_abs < c0 + c1 && src1_saved[h.to_uint()]) {
    return read_b2_backup_src1_tile(h,
                                    w,
                                    static_cast<u16_t>(c_abs - c0),
                                    lanes,
                                    packed);
  }
  return read_block_affine_source_tile(src0, src1, src2, has_src2, h, w, c, lanes, packed);
}

static error_code_t resolve_fixed_affine_tensors(const fixed_exec_desc_t& desc,
                                                 tensor_desc_t& src0,
                                                 tensor_desc_t& src1,
                                                 tensor_desc_t& src2,
                                                 bool& has_src2,
                                                 tensor_desc_t& dst) {
#pragma HLS INLINE off
  if (!resolve_tensor_read(desc.src0_tensor, src0) ||
      !resolve_tensor_read(desc.src1_tensor, src1) ||
      !resolve_tensor_write(desc.dst_tensor, desc.in_h, desc.in_w, desc.valid_c, dst)) {
    return ERR_TENSOR_DESC_RANGE;
  }
  has_src2 = desc.reserved0.to_uint() != static_cast<unsigned>(TID_INVALID);
  if (has_src2 && !resolve_tensor_read(static_cast<u8_t>(desc.reserved0.to_uint()), src2)) {
    return ERR_TENSOR_DESC_RANGE;
  }
  if (src0.h.to_uint() != desc.in_h.to_uint() ||
      src0.w.to_uint() != desc.in_w.to_uint() ||
      src1.h.to_uint() != desc.in_h.to_uint() ||
      src1.w.to_uint() != desc.in_w.to_uint() ||
      (has_src2 && (src2.h.to_uint() != desc.in_h.to_uint() ||
                    src2.w.to_uint() != desc.in_w.to_uint()))) {
    return ERR_TENSOR_DESC_RANGE;
  }
  return ERR_NONE;
}

static error_code_t run_fixed_affine_common(const fixed_exec_desc_t& desc,
                                            const tensor_desc_t& src0,
                                            const tensor_desc_t& src1,
                                            const tensor_desc_t& src2,
                                            bool has_src2,
                                            const tensor_desc_t& dst,
                                            bool row_contiguous,
                                            bool use_b2_backup) {
#pragma HLS INLINE off
#pragma HLS BIND_STORAGE variable=s_shared_row_contig_words type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=s_shared_b2_src1_saved type=ram_2p impl=bram
  if (row_contiguous) {
    if (!conv_store_row_contiguous_plan_ok(dst, desc.in_w, desc.valid_c)) {
      return ERR_BANK_OVERFLOW;
    }
  } else if (!vec_full_aligned_tile_write_plan_ok(dst, desc.valid_c)) {
    return ERR_BANK_OVERFLOW;
  }
  if (use_b2_backup &&
      !vec_row_contig_b2_needs_src1_backup(desc, src0, src1, src2, has_src2, dst)) {
    return ERR_BANK_OVERFLOW;
  }

  const int h_count = static_cast<int>(desc.in_h.to_uint());
  const int w_count = static_cast<int>(desc.in_w.to_uint());
  const unsigned valid_c_u = desc.valid_c.to_uint();
  const unsigned row_bytes_u = static_cast<unsigned>(desc.in_w.to_uint()) * valid_c_u;
  const int row_word_count = static_cast<int>(row_bytes_u / static_cast<unsigned>(AXI_WORD_BYTES));
  const int c_blocks = static_cast<int>((valid_c_u + static_cast<unsigned>(TM) - 1U) /
                                        static_cast<unsigned>(TM));

  if (use_b2_backup) {
    for (int row = 0; row < MAX_FM_H; ++row) {
#pragma HLS PIPELINE off
      s_shared_b2_src1_saved[row] = false;
    }
  }

  for (int h_iter = 0; h_iter < MAX_FM_H; ++h_iter) {
    if (h_iter >= h_count) {
      break;
    }
    const int h_i = row_contiguous ? (h_count - 1 - h_iter) : h_iter;
    const u16_t h = static_cast<u16_t>(h_i);
    axi_vec_t packed_row_word = 0;
    unsigned packed_lane = 0U;
    unsigned packed_word_idx = 0U;

    for (int w_i = 0; w_i < MAX_FM_W; ++w_i) {
#pragma HLS PIPELINE off
      if (w_i >= w_count) {
        break;
      }
      const u16_t w = static_cast<u16_t>(w_i);
      for (int c_blk = 0; c_blk < MAX_C_TILE_COUNT; ++c_blk) {
#pragma HLS PIPELINE off
        if (c_blk >= c_blocks) {
          break;
        }
        aff_q_t aff_qparam;
#pragma HLS ARRAY_PARTITION variable=aff_qparam.mul complete dim=1
#pragma HLS ARRAY_PARTITION variable=aff_qparam.bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=aff_qparam.shift complete dim=1
        if (!param_dma_get_affine_qparam(desc.param_id,
                                         static_cast<u8_t>(c_blk),
                                         aff_qparam)) {
          return ERR_PARAM_DESC_RANGE;
        }
        const u16_t c = static_cast<u16_t>(c_blk * TM);
        const u16_t remaining = static_cast<u16_t>(desc.valid_c - c);
        const u8_t lanes = row_contiguous ? vec_tensor_lanes(remaining) : static_cast<u8_t>(TM);
        act_vec_t in_packed = 0;
        act_vec_t out_packed = 0;
        if (!read_fixed_affine_source_tile(src0,
                                           src1,
                                           src2,
                                           has_src2,
                                           use_b2_backup,
                                           s_shared_b2_src1_saved,
                                           h,
                                           w,
                                           c,
                                           lanes,
                                           in_packed)) {
          return ERR_BANK_OVERFLOW;
        }
        vec_alu_apply_affine_block(in_packed,
                                   lanes,
                                   aff_qparam,
                                   desc.act_type,
                                   out_packed);
        if (row_contiguous) {
          for (int lane = 0; lane < TM; ++lane) {
#pragma HLS PIPELINE II=1
            if (lane < static_cast<int>(lanes.to_uint())) {
              const i8_t out_value = vec_get_act_i8_dynamic(out_packed, lane);
              u8_t raw = 0;
              raw.range(7, 0) = out_value.range(7, 0);
              packed_row_word.range(static_cast<int>(packed_lane * 8U + 7U),
                                    static_cast<int>(packed_lane * 8U)) = raw;
              ++packed_lane;
              if (packed_lane == static_cast<unsigned>(AXI_WORD_BYTES)) {
                if (packed_word_idx >= static_cast<unsigned>(row_word_count)) {
                  return ERR_BANK_OVERFLOW;
                }
                s_shared_row_contig_words[packed_word_idx] = packed_row_word;
                ++packed_word_idx;
                packed_lane = 0U;
                packed_row_word = 0;
              }
            }
          }
        } else if (!conv_store_write_aligned_tile(dst, h, w, c, out_packed)) {
          return ERR_BANK_OVERFLOW;
        }
      }
    }

    if (row_contiguous) {
      if (packed_lane != 0U || packed_word_idx != static_cast<unsigned>(row_word_count)) {
        return ERR_BANK_OVERFLOW;
      }
      if (use_b2_backup &&
          !backup_b2_src1_rows_before_write(src1, h_i, s_shared_b2_src1_saved)) {
        return ERR_BANK_OVERFLOW;
      }
      const u32_t dst_row_base =
          dst.base_offset + static_cast<u32_t>(h_i) * static_cast<u32_t>(row_bytes_u);
      for (int word_idx = 0; word_idx < ROW_CONTIG_MAX_WORDS; ++word_idx) {
#pragma HLS PIPELINE off
        if (word_idx >= row_word_count) {
          break;
        }
        const u32_t abs_offset =
            dst_row_base + static_cast<u32_t>(word_idx * AXI_WORD_BYTES);
        if (!conv_store_write_row_contiguous_word(dst,
                                                  abs_offset,
                                                  s_shared_row_contig_words[word_idx])) {
          return ERR_BANK_OVERFLOW;
        }
      }
    }
  }
  return ERR_NONE;
}

error_code_t vec_alu_run_fixed_issue(const npu_issue_t& issue,
                                     axi_vec_t* gmem_frame_out) {
#pragma HLS INLINE off
  (void)gmem_frame_out;
  fixed_exec_desc_t desc;
  if (!param_dma_get_fixed_exec_desc(issue.fixed_desc_id, desc)) {
    return ERR_UOP_DECODE;
  }
  if ((desc.flags.to_uint() & static_cast<unsigned>(FIXED_FLAG_BLOCK5_ROW_GROUP)) != 0U) {
    return ERR_UNSUPPORTED_OPCODE;
  }

  const unsigned issue_kind = issue.kind.to_uint();
  if (issue_kind != static_cast<unsigned>(ISSUE_VEC_AFFINE)) {
    return ERR_UNSUPPORTED_OPCODE;
  }
  if (desc.kind.to_uint() != static_cast<unsigned>(EXEC_BLOCK_AFFINE)) {
    return ERR_UOP_DECODE;
  }

  tensor_desc_t src0;
  tensor_desc_t src1;
  tensor_desc_t src2;
  tensor_desc_t dst;
  bool has_src2 = false;
  const error_code_t resolve_err =
      resolve_fixed_affine_tensors(desc, src0, src1, src2, has_src2, dst);
  if (resolve_err != ERR_NONE) {
    return resolve_err;
  }
  const bool row_contiguous =
      (desc.flags.to_uint() & static_cast<unsigned>(FIXED_FLAG_ROW_CONTIGUOUS_STORE)) != 0U;
  const bool use_b2_backup =
      row_contiguous &&
      vec_row_contig_b2_needs_src1_backup(desc, src0, src1, src2, has_src2, dst);
  return run_fixed_affine_common(desc,
                                 src0,
                                 src1,
                                 src2,
                                 has_src2,
                                 dst,
                                 row_contiguous,
                                 use_b2_backup);
}

error_code_t vec_alu_engine_exec(const npu_issue_t& issue,
                                 axi_vec_t* gmem_frame_out,
                                 volatile u8_t& prof_stage_id) {
#pragma HLS INLINE off
  npu_profile_set_stage(prof_stage_id, PROF_STAGE_VEC_FIXED);
  const unsigned kind = issue.kind.to_uint();
  if (kind == static_cast<unsigned>(ISSUE_VEC_AFFINE)) {
    return vec_alu_run_fixed_issue(issue, gmem_frame_out);
  }
  return ERR_UNSUPPORTED_OPCODE;
}

} // namespace esp_int8
