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

static axi_vec_t s_shared_row_contig_words[ROW_CONTIG_MAX_WORDS];

constexpr int VEC_AFF_HALF_LANES = TM / 2;
constexpr int VEC_AFF_HALF_COUNT = TM / VEC_AFF_HALF_LANES;
constexpr int VEC_AFF_RESIDENT_BLOCKS = MAX_C_TILE_COUNT;

using vec_aff_acc_t = ap_int<40>;
using vec_aff_round_t = ap_int<41>;
using vec_aff_half_t = ap_uint<VEC_AFF_HALF_LANES * 8>;

static i32_t vec_round_shift_affine_narrow(vec_aff_acc_t value,
                                           u8_t shift) {
#pragma HLS INLINE
  if (shift == 0) {
    return static_cast<i32_t>(value);
  }
  // PARAM v4 constrains affine shifts to [0, 31]. One guard bit covers
  // the signed rounding correction without carrying a 64-bit shifter.
  const unsigned shift_u = shift.to_uint();
  const vec_aff_round_t extended = static_cast<vec_aff_round_t>(value);
  const vec_aff_round_t bias =
      static_cast<vec_aff_round_t>(1) << (shift_u - 1U);
  if (value >= 0) {
    return static_cast<i32_t>((extended + bias) >> shift_u);
  }
  const vec_aff_round_t scale =
      static_cast<vec_aff_round_t>(1) << shift_u;
  return static_cast<i32_t>(
      ((extended - bias) + scale - 1) >> shift_u);
}

static i8_t vec_affine_i8_to_i8_narrow(i8_t value,
                                        i32_t mul,
                                        i32_t bias,
                                        u8_t shift,
                                        u8_t act_type) {
#pragma HLS INLINE
  const vec_aff_acc_t product =
      static_cast<vec_aff_acc_t>(value * mul);
  const vec_aff_acc_t scaled =
      product + static_cast<vec_aff_acc_t>(bias);
  return apply_act(
      clamp_i8(vec_round_shift_affine_narrow(scaled, shift)),
      act_type);
}

static vec_aff_half_t vec_alu_apply_affine_half(
    const vec_aff_half_t& in_half,
    unsigned valid,
    const i32_t qmul[TM][VEC_AFF_RESIDENT_BLOCKS],
    const i32_t qbias[TM][VEC_AFF_RESIDENT_BLOCKS],
    const u8_t qshift[TM][VEC_AFF_RESIDENT_BLOCKS],
    int qblock,
    u8_t act_type,
    int lane_base) {
#pragma HLS INLINE
  vec_aff_half_t out_half = 0;
  for (int local_lane = 0; local_lane < VEC_AFF_HALF_LANES; ++local_lane) {
#pragma HLS UNROLL
    const int lane = lane_base + local_lane;
    i8_t out_value = 0;
    if (static_cast<unsigned>(lane) < valid) {
      i8_t in_value = 0;
      in_value.range(7, 0) =
          in_half.range(local_lane * 8 + 7, local_lane * 8);
      out_value = vec_affine_i8_to_i8_narrow(
          in_value,
          qmul[lane][qblock],
          qbias[lane][qblock],
          qshift[lane][qblock],
          act_type);
    }
    out_half.range(local_lane * 8 + 7, local_lane * 8) =
        out_value.range(7, 0);
  }
  return out_half;
}

static void vec_alu_apply_affine_block(const act_vec_t& in_word,
                                       u8_t valid_c,
                                       const i32_t qmul[TM][VEC_AFF_RESIDENT_BLOCKS],
                                       const i32_t qbias[TM][VEC_AFF_RESIDENT_BLOCKS],
                                       const u8_t qshift[TM][VEC_AFF_RESIDENT_BLOCKS],
                                       int qblock,
                                       u8_t act_type,
                                       act_vec_t& out_word) {
#pragma HLS INLINE
  out_word = 0;
  const unsigned valid = valid_c.to_uint();
  vec_aff_half_t out_halves[VEC_AFF_HALF_COUNT];
#pragma HLS ARRAY_PARTITION variable=out_halves complete dim=1
  for (int pair = 0; pair < VEC_AFF_HALF_COUNT; ++pair) {
#pragma HLS PIPELINE II=1
    const vec_aff_half_t in_half =
        (pair == 0)
            ? static_cast<vec_aff_half_t>(
                  in_word.range(VEC_AFF_HALF_LANES * 8 - 1, 0))
            : static_cast<vec_aff_half_t>(
                  in_word.range(TM * 8 - 1, VEC_AFF_HALF_LANES * 8));
    out_halves[pair] = vec_alu_apply_affine_half(
        in_half,
        valid,
        qmul,
        qbias,
        qshift,
        qblock,
        act_type,
        pair * VEC_AFF_HALF_LANES);
  }
  out_word.range(VEC_AFF_HALF_LANES * 8 - 1, 0) = out_halves[0];
  out_word.range(TM * 8 - 1, VEC_AFF_HALF_LANES * 8) = out_halves[1];
}

static bool vec_alu_load_affine_resident_bank(
    u8_t param_id,
    int c_blocks,
    i32_t qmul[TM][VEC_AFF_RESIDENT_BLOCKS],
    i32_t qbias[TM][VEC_AFF_RESIDENT_BLOCKS],
    u8_t qshift[TM][VEC_AFF_RESIDENT_BLOCKS]) {
#pragma HLS INLINE
  for (int block = 0; block < VEC_AFF_RESIDENT_BLOCKS; ++block) {
#pragma HLS PIPELINE off
    if (block >= c_blocks) {
      break;
    }
    aff_q_t loaded;
#pragma HLS ARRAY_PARTITION variable=loaded.mul complete dim=1
#pragma HLS ARRAY_PARTITION variable=loaded.bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=loaded.shift complete dim=1
    if (!param_dma_get_affine_qparam(
            param_id, static_cast<u8_t>(block), loaded)) {
      return false;
    }
    for (int lane = 0; lane < TM; ++lane) {
#pragma HLS UNROLL
      qmul[lane][block] = loaded.mul[lane];
      qbias[lane][block] = loaded.bias[lane];
      qshift[lane][block] = loaded.shift[lane];
    }
  }
  return true;
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

struct vec_affine_group_buffer_t {
  u32_t g0;
  u32_t g1;
  u32_t g2;
  u32_t g3;
  u32_t g4;
  u32_t g5;
  u32_t g6;
  u32_t g7;
};

static u32_t vec_extract_affine_group(const act_vec_t& word, int group) {
#pragma HLS INLINE
  switch (group) {
    case 0: return static_cast<u32_t>(word.range(31, 0));
    case 1: return static_cast<u32_t>(word.range(63, 32));
    case 2: return static_cast<u32_t>(word.range(95, 64));
    case 3: return static_cast<u32_t>(word.range(127, 96));
    case 4: return static_cast<u32_t>(word.range(159, 128));
    case 5: return static_cast<u32_t>(word.range(191, 160));
    case 6: return static_cast<u32_t>(word.range(223, 192));
    default: return static_cast<u32_t>(word.range(255, 224));
  }
}

static u64_t vec_extract_affine_group_pair(const act_vec_t& word, int pair) {
#pragma HLS INLINE
  switch (pair) {
    case 0: return static_cast<u64_t>(word.range(63, 0));
    case 1: return static_cast<u64_t>(word.range(127, 64));
    case 2: return static_cast<u64_t>(word.range(191, 128));
    default: return static_cast<u64_t>(word.range(255, 192));
  }
}

static void vec_set_compact_group(vec_affine_group_buffer_t& groups,
                                  unsigned slot,
                                  u32_t value) {
#pragma HLS INLINE
  switch (slot) {
    case 0: groups.g0 = value; break;
    case 1: groups.g1 = value; break;
    case 2: groups.g2 = value; break;
    case 3: groups.g3 = value; break;
    case 4: groups.g4 = value; break;
    case 5: groups.g5 = value; break;
    case 6: groups.g6 = value; break;
    default: groups.g7 = value; break;
  }
}

static axi_vec_t vec_pack_compact_groups(const vec_affine_group_buffer_t& groups) {
#pragma HLS INLINE
  axi_vec_t word = 0;
  word.range(31, 0) = groups.g0;
  word.range(63, 32) = groups.g1;
  word.range(95, 64) = groups.g2;
  word.range(127, 96) = groups.g3;
  word.range(159, 128) = groups.g4;
  word.range(191, 160) = groups.g5;
  word.range(223, 192) = groups.g6;
  word.range(255, 224) = groups.g7;
  return word;
}

static bool vec_append_compact_group(u32_t input_group,
                                     unsigned valid_bytes,
                                     u64_t& carry,
                                     unsigned& carry_bytes,
                                     vec_affine_group_buffer_t& groups,
                                     unsigned& group_slot,
                                     axi_vec_t& completed_word) {
#pragma HLS INLINE
  u32_t masked_group = input_group;
  if (valid_bytes == 1U) {
    masked_group &= static_cast<u32_t>(0x000000FFU);
  } else if (valid_bytes == 2U) {
    masked_group &= static_cast<u32_t>(0x0000FFFFU);
  } else if (valid_bytes == 3U) {
    masked_group &= static_cast<u32_t>(0x00FFFFFFU);
  }

  u64_t merged = carry;
  const u64_t incoming = static_cast<u64_t>(masked_group);
  switch (carry_bytes) {
    case 0: merged |= incoming; break;
    case 1: merged |= incoming << 8; break;
    case 2: merged |= incoming << 16; break;
    default: merged |= incoming << 24; break;
  }

  const unsigned total_bytes = carry_bytes + valid_bytes;
  if (total_bytes < 4U) {
    carry = merged;
    carry_bytes = total_bytes;
    return false;
  }

  vec_set_compact_group(groups, group_slot, static_cast<u32_t>(merged.range(31, 0)));
  carry = merged >> 32;
  carry_bytes = total_bytes - 4U;
  ++group_slot;
  if (group_slot != 8U) {
    return false;
  }
  completed_word = vec_pack_compact_groups(groups);
  group_slot = 0U;
  return true;
}

static bool vec_append_compact_group_pair(u64_t input_pair,
                                          u64_t& carry,
                                          unsigned carry_bytes,
                                          vec_affine_group_buffer_t& groups,
                                          unsigned& group_slot,
                                          axi_vec_t& completed_word) {
#pragma HLS INLINE
  ap_uint<96> merged = static_cast<ap_uint<96>>(carry);
  const ap_uint<96> incoming = static_cast<ap_uint<96>>(input_pair);
  switch (carry_bytes) {
    case 0: merged |= incoming; break;
    case 1: merged |= incoming << 8; break;
    case 2: merged |= incoming << 16; break;
    default: merged |= incoming << 24; break;
  }

  const u32_t out_group0 = static_cast<u32_t>(merged.range(31, 0));
  const u32_t out_group1 = static_cast<u32_t>(merged.range(63, 32));
  carry = static_cast<u64_t>(merged >> 64);

  bool completed = false;
  vec_set_compact_group(groups, group_slot, out_group0);
  ++group_slot;
  if (group_slot == 8U) {
    completed_word = vec_pack_compact_groups(groups);
    group_slot = 0U;
    completed = true;
  }

  vec_set_compact_group(groups, group_slot, out_group1);
  ++group_slot;
  if (group_slot == 8U) {
    completed_word = vec_pack_compact_groups(groups);
    group_slot = 0U;
    completed = true;
  }
  return completed;
}

static error_code_t run_fixed_affine_common(const fixed_exec_desc_t& desc,
                                            const tensor_desc_t& src0,
                                            const tensor_desc_t& src1,
                                            const tensor_desc_t& src2,
                                             bool has_src2,
                                             const tensor_desc_t& dst,
                                             bool row_contiguous) {
#pragma HLS INLINE off
#pragma HLS BIND_STORAGE variable=s_shared_row_contig_words type=ram_2p impl=bram
#pragma HLS RESET variable=s_shared_row_contig_words off
  if (row_contiguous) {
    if (!conv_store_row_contiguous_plan_ok(dst, desc.in_w, desc.valid_c)) {
      return ERR_BANK_OVERFLOW;
    }
  } else if (!vec_full_aligned_tile_write_plan_ok(dst, desc.valid_c)) {
    return ERR_BANK_OVERFLOW;
  }
  const int h_count = static_cast<int>(desc.in_h.to_uint());
  const int w_count = static_cast<int>(desc.in_w.to_uint());
  const unsigned valid_c_u = desc.valid_c.to_uint();
  const unsigned row_bytes_u = static_cast<unsigned>(desc.in_w.to_uint()) * valid_c_u;
  const int row_word_count = static_cast<int>(row_bytes_u / static_cast<unsigned>(AXI_WORD_BYTES));
  const int c_blocks = static_cast<int>((valid_c_u + static_cast<unsigned>(TM) - 1U) /
                                        static_cast<unsigned>(TM));
  const bool cblock_major =
      (desc.flags.to_uint() & static_cast<unsigned>(FIXED_FLAG_CBLOCK_MAJOR)) != 0U;
  if (row_contiguous == cblock_major) {
    return ERR_UOP_DECODE;
  }

  i32_t qmul[TM][VEC_AFF_RESIDENT_BLOCKS];
  i32_t qbias[TM][VEC_AFF_RESIDENT_BLOCKS];
  u8_t qshift[TM][VEC_AFF_RESIDENT_BLOCKS];
#pragma HLS ARRAY_PARTITION variable=qmul complete dim=1
#pragma HLS ARRAY_PARTITION variable=qbias complete dim=1
#pragma HLS ARRAY_PARTITION variable=qshift complete dim=1
#pragma HLS BIND_STORAGE variable=qmul type=ram_1p impl=lutram
#pragma HLS BIND_STORAGE variable=qbias type=ram_1p impl=lutram
#pragma HLS BIND_STORAGE variable=qshift type=ram_1p impl=lutram
  if (!vec_alu_load_affine_resident_bank(
          desc.param_id, c_blocks, qmul, qbias, qshift)) {
    return ERR_PARAM_DESC_RANGE;
  }

  const int outer_count = cblock_major ? c_blocks : h_count;
  const int middle_count = cblock_major ? h_count : w_count;
  const int inner_count = cblock_major ? w_count : c_blocks;
  for (int outer = 0; outer < MAX_FM_H; ++outer) {
    if (outer >= outer_count) {
      break;
    }
    const int row_h_i = h_count - 1 - outer;
    vec_affine_group_buffer_t packed_groups;
    u64_t packed_carry = 0;
    unsigned packed_carry_bytes = 0U;
    unsigned packed_group_slot = 0U;
    unsigned packed_word_idx = 0U;

    for (int middle = 0; middle < MAX_FM_W; ++middle) {
#pragma HLS PIPELINE off
      if (middle >= middle_count) {
        break;
      }
      for (int inner = 0; inner < MAX_FM_W; ++inner) {
#pragma HLS PIPELINE off
        if (inner >= inner_count) {
          break;
        }
        const int h_i = cblock_major ? middle : row_h_i;
        const int w_i = cblock_major ? inner : middle;
        const int c_blk = cblock_major ? outer : inner;
        const u16_t h = static_cast<u16_t>(h_i);
        const u16_t w = static_cast<u16_t>(w_i);
        const u16_t c = static_cast<u16_t>(c_blk * TM);
        const u16_t remaining = static_cast<u16_t>(desc.valid_c - c);
        const u8_t lanes = row_contiguous ? vec_tensor_lanes(remaining) : static_cast<u8_t>(TM);
        act_vec_t in_packed = 0;
        act_vec_t out_packed = 0;
        if (!read_block_affine_source_tile(src0,
                                           src1,
                                           src2,
                                           has_src2,
                                           h,
                                           w,
                                           c,
                                           lanes,
                                           in_packed)) {
          return ERR_BANK_OVERFLOW;
        }
        vec_alu_apply_affine_block(in_packed,
                                   lanes,
                                   qmul,
                                   qbias,
                                   qshift,
                                   c_blk,
                                   desc.act_type,
                                   out_packed);
        if (row_contiguous) {
          const unsigned lane_count = lanes.to_uint();
          const unsigned pair_count = lane_count / 8U;
          for (int pair = 0; pair < TM / 8; ++pair) {
#pragma HLS PIPELINE II=1
            if (pair >= static_cast<int>(pair_count)) {
              break;
            }
            axi_vec_t completed_word = 0;
            if (vec_append_compact_group_pair(
                    vec_extract_affine_group_pair(out_packed, pair),
                    packed_carry,
                    packed_carry_bytes,
                    packed_groups,
                    packed_group_slot,
                    completed_word)) {
              if (packed_word_idx >= static_cast<unsigned>(row_word_count)) {
                return ERR_BANK_OVERFLOW;
              }
              s_shared_row_contig_words[packed_word_idx] = completed_word;
              ++packed_word_idx;
            }
          }

          const unsigned paired_bytes = pair_count * 8U;
          const unsigned tail_bytes = lane_count - paired_bytes;
          const unsigned tail_group_count = (tail_bytes + 3U) / 4U;
          for (int tail_group = 0; tail_group < 2; ++tail_group) {
#pragma HLS PIPELINE II=1
            if (tail_group >= static_cast<int>(tail_group_count)) {
              break;
            }
            const unsigned consumed =
                paired_bytes + static_cast<unsigned>(tail_group) * 4U;
            const unsigned valid_group_bytes =
                ((lane_count - consumed) >= 4U) ? 4U : (lane_count - consumed);
            const int group = static_cast<int>(pair_count * 2U) + tail_group;
            axi_vec_t completed_word = 0;
            if (vec_append_compact_group(vec_extract_affine_group(out_packed, group),
                                         valid_group_bytes,
                                         packed_carry,
                                         packed_carry_bytes,
                                         packed_groups,
                                         packed_group_slot,
                                         completed_word)) {
              if (packed_word_idx >= static_cast<unsigned>(row_word_count)) {
                return ERR_BANK_OVERFLOW;
              }
              s_shared_row_contig_words[packed_word_idx] = completed_word;
              ++packed_word_idx;
            }
          }
        } else if (!conv_store_write_aligned_tile(dst, h, w, c, out_packed)) {
          return ERR_BANK_OVERFLOW;
        }
      }
    }

    if (row_contiguous) {
      if (packed_carry_bytes != 0U || packed_group_slot != 0U ||
          packed_word_idx != static_cast<unsigned>(row_word_count)) {
        return ERR_BANK_OVERFLOW;
      }
      const u32_t dst_row_base =
          dst.base_offset + static_cast<u32_t>(row_h_i) * static_cast<u32_t>(row_bytes_u);
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
  return run_fixed_affine_common(desc,
                                 src0,
                                 src1,
                                 src2,
                                 has_src2,
                                 dst,
                                 row_contiguous);
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
