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
void conv_store_row_contiguous_set_byte(axi_vec_t row_words[ROW_CONTIG_MAX_WORDS],
                                        u32_t byte_idx,
                                        i8_t value);
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
void profile_record_affine_fixed(const fixed_exec_desc_t& desc, const tensor_desc_t& src);

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

void vec_alu_apply_affine_block(const act_vec_t& in_word,
                                u8_t valid_c,
                                aff_q_t qparam,
                                u8_t act_type,
                                act_vec_t& out_word) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=qparam.mul complete dim=1
#pragma HLS ARRAY_PARTITION variable=qparam.bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=qparam.shift complete dim=1
  out_word = 0;
  const unsigned valid = valid_c.to_uint();
  i8_t out_lanes[TM];
#pragma HLS ARRAY_PARTITION variable=out_lanes complete dim=1
  for (int lane = 0; lane < TM; ++lane) {
#pragma HLS UNROLL factor=8
    i8_t out_value = 0;
    if (static_cast<unsigned>(lane) < valid) {
      const i8_t in_value = vec_get_act_i8_dynamic(in_word, lane);
      out_value = affine_i8_to_i8(in_value,
                                  qparam.mul[lane],
                                  qparam.bias[lane],
                                  qparam.shift[lane],
                                  act_type);
    }
    out_lanes[lane] = out_value;
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

static bool read_block_affine_source_tile_b2_backup(const tensor_desc_t& src0,
                                                    const tensor_desc_t& src1,
                                                    const tensor_desc_t& src2,
                                                    bool has_src2,
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
  if (c_abs >= c0 && c_abs < c0 + c1 && src1_saved[h.to_uint()]) {
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

static bool load_affine_qparam_blocks(u8_t param_id,
                                      int c_blocks,
                                      aff_q_t aff_qparams[MAX_C_TILE_COUNT]) {
#pragma HLS INLINE off
  for (int c_blk = 0; c_blk < MAX_C_TILE_COUNT; ++c_blk) {
#pragma HLS PIPELINE off
    if (c_blk >= c_blocks) {
      break;
    }
    if (!param_dma_get_affine_qparam(param_id, static_cast<u8_t>(c_blk), aff_qparams[c_blk])) {
      return false;
    }
  }
  return true;
}

static error_code_t run_fixed_row_contiguous_datapath(const fixed_exec_desc_t& desc,
                                                      const tensor_desc_t& src0,
                                                      const tensor_desc_t& src1,
                                                      const tensor_desc_t& src2,
                                                      bool has_src2,
                                                      const tensor_desc_t& dst) {
#pragma HLS INLINE off
#pragma HLS BIND_STORAGE variable=s_shared_row_contig_words type=ram_2p impl=bram
  if (!conv_store_row_contiguous_plan_ok(dst, desc.in_w, desc.valid_c)) {
    return ERR_BANK_OVERFLOW;
  }

  const int h_count = static_cast<int>(desc.in_h.to_uint());
  const int w_count = static_cast<int>(desc.in_w.to_uint());
  const unsigned valid_c_u = desc.valid_c.to_uint();
  const unsigned row_bytes_u = static_cast<unsigned>(desc.in_w.to_uint()) * valid_c_u;
  const int row_word_count = static_cast<int>(row_bytes_u / static_cast<unsigned>(AXI_WORD_BYTES));
  const int c_blocks = static_cast<int>((valid_c_u + static_cast<unsigned>(TM) - 1U) /
                                        static_cast<unsigned>(TM));

  aff_q_t aff_qparams[MAX_C_TILE_COUNT];
#pragma HLS ARRAY_PARTITION variable=aff_qparams complete dim=1
  if (!load_affine_qparam_blocks(desc.param_id, c_blocks, aff_qparams)) {
    return ERR_PARAM_DESC_RANGE;
  }

  for (int h_rev = 0; h_rev < MAX_FM_H; ++h_rev) {
    if (h_rev >= h_count) {
      break;
    }
    const int h_i = h_count - 1 - h_rev;
    const u16_t h = static_cast<u16_t>(h_i);

    for (int word_idx = 0; word_idx < ROW_CONTIG_MAX_WORDS; ++word_idx) {
#pragma HLS PIPELINE off
      if (word_idx >= row_word_count) {
        break;
      }
      s_shared_row_contig_words[word_idx] = 0;
    }

    for (int w_i = 0; w_i < MAX_FM_W; ++w_i) {
      if (w_i >= w_count) {
        break;
      }
      const u16_t w = static_cast<u16_t>(w_i);
      for (int c_blk = 0; c_blk < MAX_C_TILE_COUNT; ++c_blk) {
#pragma HLS PIPELINE off
        if (c_blk >= c_blocks) {
          break;
        }
        const u16_t c = static_cast<u16_t>(c_blk * TM);
        const u16_t remaining = static_cast<u16_t>(desc.valid_c - c);
        const u8_t lanes = vec_tensor_lanes(remaining);
        act_vec_t in_packed = 0;
        if (!read_block_affine_source_tile(src0, src1, src2, has_src2, h, w, c, lanes, in_packed)) {
          return ERR_BANK_OVERFLOW;
        }
        act_vec_t out_packed = 0;
        vec_alu_apply_affine_block(in_packed,
                                   lanes,
                                   aff_qparams[c_blk],
                                   desc.act_type,
                                   out_packed);
        for (int lane = 0; lane < TM; ++lane) {
#pragma HLS UNROLL
          if (static_cast<unsigned>(lane) < lanes.to_uint()) {
            const i8_t out_value = vec_get_act_i8_dynamic(out_packed, lane);
            const u32_t byte_idx = static_cast<u32_t>(static_cast<unsigned>(w_i) * valid_c_u +
                                                      c.to_uint() + static_cast<unsigned>(lane));
            conv_store_row_contiguous_set_byte(s_shared_row_contig_words, byte_idx, out_value);
          }
        }
      }
    }

    const u32_t dst_row_base = dst.base_offset + static_cast<u32_t>(h_i) * static_cast<u32_t>(row_bytes_u);
    for (int word_idx = 0; word_idx < ROW_CONTIG_MAX_WORDS; ++word_idx) {
#pragma HLS PIPELINE off
      if (word_idx >= row_word_count) {
        break;
      }
      const u32_t abs_offset = dst_row_base + static_cast<u32_t>(word_idx * AXI_WORD_BYTES);
      if (!conv_store_write_row_contiguous_word(dst, abs_offset, s_shared_row_contig_words[word_idx])) {
        return ERR_BANK_OVERFLOW;
      }
    }
  }
  return ERR_NONE;
}

static error_code_t run_b2_c131_row_contiguous_backup_op(const fixed_exec_desc_t& desc,
                                                         const tensor_desc_t& src0,
                                                         const tensor_desc_t& src1,
                                                         const tensor_desc_t& src2,
                                                         bool has_src2,
                                                         const tensor_desc_t& dst) {
#pragma HLS INLINE off
#pragma HLS BIND_STORAGE variable=s_shared_row_contig_words type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=s_shared_b2_src1_saved type=ram_2p impl=bram
  if (!conv_store_row_contiguous_plan_ok(dst, desc.in_w, desc.valid_c) ||
      !vec_row_contig_b2_needs_src1_backup(desc, src0, src1, src2, has_src2, dst)) {
    return ERR_BANK_OVERFLOW;
  }

  profile_record_affine_fixed(desc, dst);

  const int h_count = static_cast<int>(desc.in_h.to_uint());
  const int w_count = static_cast<int>(desc.in_w.to_uint());
  const unsigned valid_c_u = desc.valid_c.to_uint();
  const unsigned row_bytes_u = static_cast<unsigned>(desc.in_w.to_uint()) * valid_c_u;
  const int row_word_count = static_cast<int>(row_bytes_u / static_cast<unsigned>(AXI_WORD_BYTES));
  const int c_blocks = static_cast<int>((valid_c_u + static_cast<unsigned>(TM) - 1U) /
                                        static_cast<unsigned>(TM));

  aff_q_t aff_qparams[MAX_C_TILE_COUNT];
#pragma HLS ARRAY_PARTITION variable=aff_qparams complete dim=1
  if (!load_affine_qparam_blocks(desc.param_id, c_blocks, aff_qparams)) {
    return ERR_PARAM_DESC_RANGE;
  }

  for (int row = 0; row < MAX_FM_H; ++row) {
#pragma HLS PIPELINE off
    s_shared_b2_src1_saved[row] = false;
  }

  for (int h_rev = 0; h_rev < MAX_FM_H; ++h_rev) {
    if (h_rev >= h_count) {
      break;
    }
    const int h_i = h_count - 1 - h_rev;
    const u16_t h = static_cast<u16_t>(h_i);

    for (int word_idx = 0; word_idx < ROW_CONTIG_MAX_WORDS; ++word_idx) {
#pragma HLS PIPELINE off
      if (word_idx >= row_word_count) {
        break;
      }
      s_shared_row_contig_words[word_idx] = 0;
    }

    for (int w_i = 0; w_i < MAX_FM_W; ++w_i) {
      if (w_i >= w_count) {
        break;
      }
      const u16_t w = static_cast<u16_t>(w_i);
      for (int c_blk = 0; c_blk < MAX_C_TILE_COUNT; ++c_blk) {
#pragma HLS PIPELINE off
        if (c_blk >= c_blocks) {
          break;
        }
        const u16_t c = static_cast<u16_t>(c_blk * TM);
        const u16_t remaining = static_cast<u16_t>(desc.valid_c - c);
        const u8_t lanes = vec_tensor_lanes(remaining);
        act_vec_t in_packed = 0;
        if (!read_block_affine_source_tile_b2_backup(src0,
                                                     src1,
                                                     src2,
                                                     has_src2,
                                                     s_shared_b2_src1_saved,
                                                     h,
                                                     w,
                                                     c,
                                                     lanes,
                                                     in_packed)) {
          return ERR_BANK_OVERFLOW;
        }
        act_vec_t out_packed = 0;
        vec_alu_apply_affine_block(in_packed,
                                   lanes,
                                   aff_qparams[c_blk],
                                   desc.act_type,
                                   out_packed);
        for (int lane = 0; lane < TM; ++lane) {
#pragma HLS UNROLL
          if (static_cast<unsigned>(lane) < lanes.to_uint()) {
            const i8_t out_value = vec_get_act_i8_dynamic(out_packed, lane);
            const u32_t byte_idx = static_cast<u32_t>(static_cast<unsigned>(w_i) * valid_c_u +
                                                      c.to_uint() + static_cast<unsigned>(lane));
            conv_store_row_contiguous_set_byte(s_shared_row_contig_words, byte_idx, out_value);
          }
        }
      }
    }

    if (!backup_b2_src1_rows_before_write(src1, h_i, s_shared_b2_src1_saved)) {
      return ERR_BANK_OVERFLOW;
    }

    const u32_t dst_row_base = dst.base_offset + static_cast<u32_t>(h_i) * static_cast<u32_t>(row_bytes_u);
    for (int word_idx = 0; word_idx < ROW_CONTIG_MAX_WORDS; ++word_idx) {
#pragma HLS PIPELINE off
      if (word_idx >= row_word_count) {
        break;
      }
      const u32_t abs_offset = dst_row_base + static_cast<u32_t>(word_idx * AXI_WORD_BYTES);
      if (!conv_store_write_row_contiguous_word(dst, abs_offset, s_shared_row_contig_words[word_idx])) {
        return ERR_BANK_OVERFLOW;
      }
    }
  }
  return ERR_NONE;
}

static error_code_t run_fixed_aligned_affine_only_op(const fixed_exec_desc_t& desc) {
#pragma HLS INLINE off
  tensor_desc_t src0;
  tensor_desc_t src1;
  tensor_desc_t src2;
  tensor_desc_t dst;
  bool has_src2 = false;
  const error_code_t resolve_err = resolve_fixed_affine_tensors(desc, src0, src1, src2, has_src2, dst);
  if (resolve_err != ERR_NONE) {
    return resolve_err;
  }
  if (!vec_full_aligned_tile_write_plan_ok(dst, desc.valid_c)) {
    return ERR_BANK_OVERFLOW;
  }
  profile_record_affine_fixed(desc, dst);

  const int h_count = static_cast<int>(desc.in_h.to_uint());
  const int w_count = static_cast<int>(desc.in_w.to_uint());
  const int c_blocks = static_cast<int>((desc.valid_c.to_uint() + static_cast<unsigned>(TM) - 1U) /
                                        static_cast<unsigned>(TM));

  aff_q_t aff_qparams[MAX_C_TILE_COUNT];
#pragma HLS ARRAY_PARTITION variable=aff_qparams complete dim=1
  if (!load_affine_qparam_blocks(desc.param_id, c_blocks, aff_qparams)) {
    return ERR_PARAM_DESC_RANGE;
  }

  for (int h_i = 0; h_i < MAX_FM_H; ++h_i) {
    if (h_i >= h_count) {
      break;
    }
    const u16_t h = static_cast<u16_t>(h_i);
    for (int w_i = 0; w_i < MAX_FM_W; ++w_i) {
      if (w_i >= w_count) {
        break;
      }
      const u16_t w = static_cast<u16_t>(w_i);
      for (int c_blk = 0; c_blk < MAX_C_TILE_COUNT; ++c_blk) {
#pragma HLS PIPELINE off
        if (c_blk >= c_blocks) {
          break;
        }
        const u16_t c = static_cast<u16_t>(c_blk * TM);
        act_vec_t cat_packed = 0;
        act_vec_t out_packed = 0;
        if (!read_block_affine_source_tile(src0,
                                           src1,
                                           src2,
                                           has_src2,
                                           h,
                                           w,
                                           c,
                                           static_cast<u8_t>(TM),
                                           cat_packed)) {
          return ERR_BANK_OVERFLOW;
        }
        vec_alu_apply_affine_block(cat_packed,
                                   static_cast<u8_t>(TM),
                                   aff_qparams[c_blk],
                                   desc.act_type,
                                   out_packed);
        if (!conv_store_write_aligned_tile(dst, h, w, c, out_packed)) {
          return ERR_BANK_OVERFLOW;
        }
      }
    }
  }
  return ERR_NONE;
}

static error_code_t run_fixed_row_contiguous_op(const fixed_exec_desc_t& desc,
                                                const tensor_desc_t& src0,
                                                const tensor_desc_t& src1,
                                                const tensor_desc_t& src2,
                                                bool has_src2,
                                                const tensor_desc_t& dst) {
#pragma HLS INLINE off
  profile_record_affine_fixed(desc, dst);
  return run_fixed_row_contiguous_datapath(desc, src0, src1, src2, has_src2, dst);
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

  const bool row_contiguous =
      (desc.flags.to_uint() & static_cast<unsigned>(FIXED_FLAG_ROW_CONTIGUOUS_STORE)) != 0U;
  if (row_contiguous) {
    tensor_desc_t src0;
    tensor_desc_t src1;
    tensor_desc_t src2;
    tensor_desc_t dst;
    bool has_src2 = false;
    const error_code_t resolve_err = resolve_fixed_affine_tensors(desc, src0, src1, src2, has_src2, dst);
    if (resolve_err != ERR_NONE) {
      return resolve_err;
    }
    if (vec_row_contig_b2_needs_src1_backup(desc, src0, src1, src2, has_src2, dst)) {
      return run_b2_c131_row_contiguous_backup_op(desc, src0, src1, src2, has_src2, dst);
    }
    return run_fixed_row_contiguous_op(desc, src0, src1, src2, has_src2, dst);
  }
  return run_fixed_aligned_affine_only_op(desc);
}

error_code_t vec_alu_engine_exec(const npu_issue_t& issue,
                                 axi_vec_t* gmem_frame_out) {
#pragma HLS INLINE off
  const unsigned kind = issue.kind.to_uint();
  if (kind == static_cast<unsigned>(ISSUE_VEC_AFFINE)) {
    return vec_alu_run_fixed_issue(issue, gmem_frame_out);
  }
  return ERR_UNSUPPORTED_OPCODE;
}

} // namespace esp_int8
