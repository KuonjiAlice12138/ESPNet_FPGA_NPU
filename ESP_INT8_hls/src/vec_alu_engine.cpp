#include "../include/npu_config.hpp"
#include "../include/npu_ctrl.hpp"
#include "../include/npu_q.hpp"
#include "../include/npu_uop.hpp"

namespace esp_int8 {

bool param_dma_get_fixed_exec_desc(u8_t id, fixed_exec_desc_t& desc);
bool param_dma_get_affine_qparam(u8_t param_id, u8_t block_id, aff_q_t& qparam);
bool on_chip_memory_read_packed_span(u8_t bank_id,
                                      u32_t byte_offset,
                                      u8_t byte_count,
                                      act_vec_t& packed);
bool conv_store_write_row_contiguous_word(const tensor_desc_t& dst,
                                          u32_t abs_offset,
                                          const axi_vec_t& word);
bool conv_store_word_span_plan_ok(u8_t bank_id,
                                   u64_t first_offset,
                                   u32_t byte_step,
                                   u16_t word_count);
bool conv_store_row_contiguous_plan_ok(const tensor_desc_t& dst,
                                       u16_t width,
                                       u16_t valid_c);
bool resolve_tensor_read(u8_t tensor_id, tensor_desc_t& desc);
bool resolve_tensor_write(u8_t tensor_id, u16_t h, u16_t w, u16_t c, tensor_desc_t& desc);

static axi_vec_t s_shared_row_contig_words[ROW_CONTIG_MAX_WORDS];

constexpr int VEC_AFF_HALF_LANES = TM / 2;
constexpr int VEC_AFF_HALF_COUNT = TM / VEC_AFF_HALF_LANES;
constexpr int VEC_AFF_RESIDENT_BLOCKS = MAX_C_TILE_COUNT;
constexpr int VEC_MAX_ROW_TILES = MAX_FM_W * MAX_C_TILE_COUNT;
static_assert(VEC_MAX_ROW_TILES < 2048, "Vec tile counter is 11 bits");
static_assert(ROW_CONTIG_MAX_WORDS < 512, "Vec commit counter is 9 bits");
static_assert(MAX_FM_W <= ROW_CONTIG_MAX_WORDS, "Aligned Vec row fits the shared buffer");

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


static bool vec_full_aligned_tile_write_plan_ok(const tensor_desc_t& desc, u16_t total_c) {
#pragma HLS INLINE off
  const unsigned total = total_c.to_uint();
  const unsigned phys = tensor_phys_c(desc).to_uint();
  return total != 0U && total <= desc.c.to_uint() &&
         (total & static_cast<unsigned>(TM - 1)) == 0U &&
         phys >= static_cast<unsigned>(TM) &&
         (phys & static_cast<unsigned>(TM - 1)) == 0U &&
         ((desc.base_offset.to_uint() + desc.reserved1.to_uint()) &
          static_cast<unsigned>(AXI_WORD_BYTES - 1)) == 0U;
}

struct vec_source_span_t {
  u8_t bank;
  u8_t count;
  u32_t offset;
  u32_t row_stride;
  u16_t pixel_step;
};

static bool vec_select_source_span(const tensor_desc_t& src0,
                                    const tensor_desc_t& src1,
                                    const tensor_desc_t& src2,
                                    bool has_src2,
                                    u32_t source0,
                                    u32_t source1,
                                    u32_t source2,
                                    u16_t c,
                                    u8_t lanes,
                                    vec_source_span_t& span) {
#pragma HLS INLINE
  const unsigned c_abs = c.to_uint();
  const unsigned c0 = src0.c.to_uint();
  const unsigned c1 = src1.c.to_uint();
  tensor_desc_t selected = src0;
  u32_t start = source0;
  unsigned local_c = c_abs;
  if (c_abs < c0) {
    local_c = c_abs;
  } else if (c_abs < c0 + c1) {
    selected = src1;
    start = source1;
    local_c = c_abs - c0;
  } else if (has_src2) {
    selected = src2;
    start = source2;
    local_c = c_abs - c0 - c1;
  } else {
    return false;
  }
  const unsigned channels = selected.c.to_uint();
  span.bank = selected.bank_id;
  span.offset = start + static_cast<u32_t>(local_c);
  span.row_stride = tensor_row_stride_bytes(selected);
  span.pixel_step = tensor_phys_c(selected);
  span.count = 0;
  if (local_c >= channels) { return true; }
  const unsigned remaining = channels - local_c;
  const unsigned count = remaining < lanes.to_uint() ? remaining : lanes.to_uint();
  span.count = static_cast<u8_t>(count);
  return true;
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

static bool vec_stage_compact_word(const act_vec_t& out_packed,
                                    u8_t lanes,
                                    u64_t& carry, unsigned& carry_bytes,
                                    vec_affine_group_buffer_t& groups,
                                    unsigned& slot, unsigned& word_index,
                                    unsigned row_word_count) {
#pragma HLS INLINE
  const unsigned lane_count = lanes.to_uint();
  const unsigned pair_count = lane_count / 8U;
  for (int pair = 0; pair < TM / 8; ++pair) {
#pragma HLS PIPELINE II=1
    if (pair >= static_cast<int>(pair_count)) { break; }
    axi_vec_t completed = 0;
    if (vec_append_compact_group_pair(vec_extract_affine_group_pair(out_packed, pair),
                                      carry, carry_bytes, groups, slot, completed)) {
      if (word_index >= row_word_count) { return false; }
      s_shared_row_contig_words[word_index++] = completed;
    }
  }
  const unsigned paired_bytes = pair_count * 8U;
  const unsigned tail_count = (lane_count - paired_bytes + 3U) / 4U;
  for (int tail = 0; tail < 2; ++tail) {
#pragma HLS PIPELINE II=1
    if (tail >= static_cast<int>(tail_count)) { break; }
    const unsigned consumed = paired_bytes + static_cast<unsigned>(tail) * 4U;
    const unsigned bytes = lane_count - consumed < 4U ? lane_count - consumed : 4U;
    axi_vec_t completed = 0;
    if (vec_append_compact_group(
            vec_extract_affine_group(out_packed, static_cast<int>(pair_count * 2U) + tail),
            bytes, carry, carry_bytes, groups, slot, completed)) {
      if (word_index >= row_word_count) { return false; }
      s_shared_row_contig_words[word_index++] = completed;
    }
  }
  return true;
}

static bool vec_finish_compact_row(u64_t carry, unsigned carry_bytes,
                                    vec_affine_group_buffer_t& groups,
                                    unsigned slot, unsigned& word_index,
                                    unsigned row_word_count) {
#pragma HLS INLINE
  if (carry_bytes != 0U) {
    vec_set_compact_group(groups, slot++, static_cast<u32_t>(carry));
  }
  if (slot != 0U) {
    if (word_index >= row_word_count) { return false; }
    for (int index = 0; index < TM / 4; ++index) {
#pragma HLS UNROLL
      if (static_cast<unsigned>(index) >= slot) {
        vec_set_compact_group(groups, index, 0);
      }
    }
    s_shared_row_contig_words[word_index++] = vec_pack_compact_groups(groups);
  }
  return word_index == row_word_count;
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
  const unsigned row_bytes_u = tensor_row_stride_bytes(dst).to_uint();
  const int row_word_count = static_cast<int>(row_bytes_u / static_cast<unsigned>(AXI_WORD_BYTES));
  const int c_blocks = static_cast<int>((valid_c_u + static_cast<unsigned>(TM) - 1U) /
                                        static_cast<unsigned>(TM));
  if (h_count <= 0 || h_count > MAX_FM_H || w_count <= 0 || w_count > MAX_FM_W ||
      c_blocks <= 0 || c_blocks > VEC_AFF_RESIDENT_BLOCKS) {
    return ERR_TENSOR_DESC_RANGE;
  }
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
  const int tile_service_count = cblock_major ? w_count : w_count * c_blocks;
  const ap_uint<9> commit_word_count = static_cast<ap_uint<9>>(
      row_contiguous ? row_word_count : w_count);
  const int rows_per_outer = cblock_major ? h_count : 1;
  const u32_t stride0 = tensor_row_stride_bytes(src0);
  const u32_t stride1 = tensor_row_stride_bytes(src1);
  const u32_t stride2 = has_src2 ? tensor_row_stride_bytes(src2) : static_cast<u32_t>(0);
  const u32_t step0 = static_cast<u32_t>(tensor_phys_c(src0));
  const u32_t step1 = static_cast<u32_t>(tensor_phys_c(src1));
  const u32_t step2 = has_src2 ? static_cast<u32_t>(tensor_phys_c(src2)) : static_cast<u32_t>(0);
  const u32_t dst_step = static_cast<u32_t>(tensor_phys_c(dst));
  const u32_t commit_step = row_contiguous ? static_cast<u32_t>(AXI_WORD_BYTES) : dst_step;
  // All other rows/blocks end no later than this monotonically increasing span.
  const u64_t last_write_span = static_cast<u64_t>(dst.base_offset) +
      static_cast<u64_t>(dst.reserved1) +
      static_cast<u64_t>(h_count - 1) * static_cast<u64_t>(row_bytes_u) +
      static_cast<u64_t>(cblock_major ? (c_blocks - 1) * TM : 0);
  if (!conv_store_word_span_plan_ok(dst.bank_id, last_write_span, commit_step,
                                     static_cast<u16_t>(commit_word_count.to_uint()))) {
    return ERR_BANK_OVERFLOW;
  }
  const u32_t base0 = src0.base_offset + static_cast<u32_t>(src0.reserved1);
  const u32_t base1 = src1.base_offset + static_cast<u32_t>(src1.reserved1);
  const u32_t base2 = has_src2
      ? static_cast<u32_t>(src2.base_offset + static_cast<u32_t>(src2.reserved1))
      : static_cast<u32_t>(0);
  for (int outer = 0; outer < MAX_FM_H; ++outer) {
    if (outer >= outer_count) {
      break;
    }
    const int row_h_i = h_count - 1 - outer;
    const int first_h = cblock_major ? 0 : row_h_i;
    const u32_t source_row0 = base0 + static_cast<u32_t>(first_h) * stride0;
    const u32_t source_row1 = base1 + static_cast<u32_t>(first_h) * stride1;
    const u32_t source_row2 = base2 + static_cast<u32_t>(first_h) * stride2;
    vec_source_span_t aligned_source = {0, 0, 0, 0, 0};
    if (cblock_major &&
        !vec_select_source_span(src0, src1, src2, has_src2, base0, base1, base2,
                                  static_cast<u16_t>(outer * TM), static_cast<u8_t>(TM),
                                  aligned_source)) {
      return ERR_BANK_OVERFLOW;
    }
    u32_t aligned_row = aligned_source.offset;
    u32_t dst_row = dst.base_offset + static_cast<u32_t>(dst.reserved1) +
                    static_cast<u32_t>(first_h) * static_cast<u32_t>(row_bytes_u);
    for (int row = 0; row < MAX_FM_H; ++row) {
#pragma HLS PIPELINE off
      if (row >= rows_per_outer) { break; }
      u32_t source0 = source_row0;
      u32_t source1 = source_row1;
      u32_t source2 = source_row2;
      u32_t aligned_pixel = aligned_row;
      ap_uint<4> c_blk = cblock_major ? outer : 0;
      vec_affine_group_buffer_t packed_groups;
      u64_t packed_carry = 0;
      unsigned packed_carry_bytes = 0U;
      unsigned packed_group_slot = 0U;
      unsigned packed_word_idx = 0U;

      // Stage a compact row or one aligned channel-block row before any writeback.
      VEC_AFFINE_TILES:
      for (ap_uint<11> tile = 0; tile < VEC_MAX_ROW_TILES; ++tile) {
#pragma HLS PIPELINE off
        if (tile.to_uint() >= static_cast<unsigned>(tile_service_count)) { break; }
        const u16_t c = static_cast<u16_t>(c_blk.to_uint() * TM);
        const u16_t remaining = static_cast<u16_t>(desc.valid_c - c);
        const u8_t lanes = row_contiguous ? vec_tensor_lanes(remaining) : static_cast<u8_t>(TM);
        vec_source_span_t span = aligned_source;
        span.offset = aligned_pixel;
        if (row_contiguous &&
            !vec_select_source_span(src0, src1, src2, has_src2,
                                      source0, source1, source2, c, lanes, span)) {
          return ERR_BANK_OVERFLOW;
        }
        act_vec_t in_packed = 0;
        act_vec_t out_packed = 0;
        if (span.count != 0 &&
            !on_chip_memory_read_packed_span(span.bank, span.offset, span.count, in_packed)) {
          return ERR_BANK_OVERFLOW;
        }
        vec_alu_apply_affine_block(in_packed, lanes, qmul, qbias, qshift,
                                    static_cast<int>(c_blk.to_uint()), desc.act_type,
                                    out_packed);
        if (row_contiguous) {
          if (!vec_stage_compact_word(out_packed, lanes, packed_carry, packed_carry_bytes,
                                        packed_groups, packed_group_slot, packed_word_idx,
                                        static_cast<unsigned>(row_word_count))) {
            return ERR_BANK_OVERFLOW;
          }
          ++c_blk;
          if (c_blk.to_uint() == static_cast<unsigned>(c_blocks)) {
            c_blk = 0;
            source0 += step0;
            source1 += step1;
            source2 += step2;
          }
        } else {
          s_shared_row_contig_words[tile.to_uint()] = out_packed;
          aligned_pixel += static_cast<u32_t>(aligned_source.pixel_step);
        }
      }
      if (row_contiguous && !vec_finish_compact_row(
              packed_carry, packed_carry_bytes, packed_groups, packed_group_slot,
              packed_word_idx, static_cast<unsigned>(row_word_count))) {
        return ERR_BANK_OVERFLOW;
      }

      // One local BRAM read and one independent destination write per word.
      u32_t commit_offset = dst_row + static_cast<u32_t>(cblock_major ? outer * TM : 0);
      VEC_COMMIT_WORDS:
      for (ap_uint<9> word = 0; word < ROW_CONTIG_MAX_WORDS; ++word) {
#pragma HLS PIPELINE II=1
        if (word.to_uint() >= commit_word_count.to_uint()) { break; }
        const axi_vec_t packed = s_shared_row_contig_words[word.to_uint()];
        // The span proof makes this status redundant; do not feed it into the back edge.
        (void)conv_store_write_row_contiguous_word(dst, commit_offset, packed);
        commit_offset += commit_step;
      }
      aligned_row += aligned_source.row_stride;
      dst_row += static_cast<u32_t>(row_bytes_u);
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
