#include "../include/npu_config.hpp"
#include "../include/npu_ctrl.hpp"
#include "../include/npu_q.hpp"
#include "../include/npu_schedule.hpp"
#include "../include/npu_uop.hpp"

#ifndef __SYNTHESIS__
#include <cstdio>
#include <fstream>
#endif

namespace esp_int8 {

constexpr int PPU_BLOCK_LANES = 4;
constexpr int PPU_BLOCK_GROUPS = TM / PPU_BLOCK_LANES;

bool param_dma_get_affine_qparam(u8_t param_id, u8_t block_id, aff_q_t& qparam);
u32_t param_dma_generation();
bool on_chip_memory_read_packed_span(u8_t bank_id,
                                      u32_t byte_offset,
                                      u8_t byte_count,
                                      act_vec_t& packed);
bool on_chip_memory_read_packed_tile(const tensor_desc_t& desc,
                                     i32_t h,
                                     i32_t w,
                                     u16_t c_begin,
                                     u8_t valid_c,
                                     act_vec_t& packed);
bool on_chip_memory_read_fmbuf_abs_word(u8_t bank_id,
                                        u32_t byte_offset,
                                        axi_vec_t& packed);
bool on_chip_memory_write_fmbuf_abs_word(u8_t bank_id,
                                         u32_t byte_offset,
                                         act_vec_t packed);
bool on_chip_memory_write_pool2_abs_word(u32_t byte_offset,
                                         act_vec_t packed);
bool conv_store_write_row_contiguous_word(const tensor_desc_t& dst,
                                          u32_t byte_offset,
                                          const act_vec_t& word);
void upsample_fused_consume_logits_row(axi_vec_t* gmem_frame_out,
                                       u16_t encoder_row,
                                       u8_t valid_c,
                                       const act_vec_t row_buf[MAX_FM_W]);

static u16_t ppu_ceil_div_u16(u16_t a, u16_t b) {
#pragma HLS INLINE
  return static_cast<u16_t>((a + b - 1) / b);
}

static u16_t ppu_out_dim(u16_t in_size, u16_t stride) {
#pragma HLS INLINE
  return ppu_ceil_div_u16(in_size, stride);
}

static u16_t ppu_effective_stride(const conv_cfg_t& cfg) {
#pragma HLS INLINE
  return (cfg.stride == 0) ? static_cast<u16_t>(1) : static_cast<u16_t>(cfg.stride);
}


static bool ppu_full_aligned_tile_write_plan_ok(const tensor_desc_t& desc, u16_t total_c) {
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

static u32_t ppu_compact_row_base(const tensor_desc_t& dst, u16_t out_row) {
#pragma HLS INLINE
  return dst.base_offset +
         static_cast<u32_t>(out_row) * tensor_row_stride_bytes(dst) +
         static_cast<u32_t>(dst.reserved1);
}

static i8_t ppu_get_act_i8_dynamic(const act_vec_t& word, int lane) {
#pragma HLS INLINE
  const u8_t raw = word.range(lane * 8 + 7, lane * 8);
  i8_t value = 0;
  value.range(7, 0) = raw;
  return value;
}

static void ppu_unpack_lanes(const act_vec_t& word, i8_t lanes[TM]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
  for (int lane = 0; lane < TM; ++lane) {
#pragma HLS UNROLL
    const u8_t raw = word.range(lane * 8 + 7, lane * 8);
    i8_t value = 0;
    value.range(7, 0) = raw;
    lanes[lane] = value;
  }
}

static void ppu_pack_lanes(const i8_t lanes[TM], act_vec_t& word) {
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

static void ppu_apply_add_word(act_vec_t lhs_word,
                               act_vec_t rhs_word,
                               u8_t valid_c,
                               const add_q_t& add_qparam,
                               act_vec_t& out_word) {
#pragma HLS INLINE off
  i8_t out_lanes[TM];
#pragma HLS ARRAY_PARTITION variable=out_lanes complete dim=1
  const unsigned valid = valid_c.to_uint();
  for (int lane = 0; lane < TM; ++lane) {
#pragma HLS UNROLL factor=4
    i8_t out_value = 0;
    if (static_cast<unsigned>(lane) < valid) {
      const i8_t lhs_value = ppu_get_act_i8_dynamic(lhs_word, lane);
      const i8_t rhs_value = ppu_get_act_i8_dynamic(rhs_word, lane);
      out_value = add_i8(lhs_value, rhs_value, add_qparam);
    }
    out_lanes[lane] = out_value;
  }
  ppu_pack_lanes(out_lanes, out_word);
}

static void ppu_apply_compact_affine_word(const act_vec_t& in_word,
                                      u8_t valid_c,
                                      const aff_q_t& qparam0,
                                      const aff_q_t& qparam1,
                                      u8_t affine_lane0,
                                      u8_t act_type,
                                      act_vec_t& out_word) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=qparam0.mul complete dim=1
#pragma HLS ARRAY_PARTITION variable=qparam0.bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=qparam0.shift complete dim=1
#pragma HLS ARRAY_PARTITION variable=qparam1.mul complete dim=1
#pragma HLS ARRAY_PARTITION variable=qparam1.bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=qparam1.shift complete dim=1
  i8_t out_lanes[TM];
#pragma HLS ARRAY_PARTITION variable=out_lanes complete dim=1
  const unsigned valid = valid_c.to_uint();
  const unsigned lane0 = affine_lane0.to_uint();
  for (int lane = 0; lane < TM; ++lane) {
#pragma HLS UNROLL factor=4
    i8_t out_value = 0;
    if (static_cast<unsigned>(lane) < valid) {
      const unsigned abs_lane = lane0 + static_cast<unsigned>(lane);
      const bool use_second = abs_lane >= static_cast<unsigned>(TM);
      const unsigned qlane = use_second ? abs_lane - static_cast<unsigned>(TM) : abs_lane;
      const i8_t in_value = ppu_get_act_i8_dynamic(in_word, lane);
      out_value =
          use_second ? affine_i8_to_i8(in_value,
                                       qparam1.mul[qlane],
                                       qparam1.bias[qlane],
                                       qparam1.shift[qlane],
                                       act_type)
                     : affine_i8_to_i8(in_value,
                                       qparam0.mul[qlane],
                                       qparam0.bias[qlane],
                                       qparam0.shift[qlane],
                                       act_type);
    }
    out_lanes[lane] = out_value;
  }
  ppu_pack_lanes(out_lanes, out_word);
}

static void ppu_cat_tail_word(const act_vec_t& prefix_word,
                              const act_vec_t& tail_word,
                              u8_t prefix_c,
                              u8_t tail_c,
                              act_vec_t& out_word) {
#pragma HLS INLINE off
  i8_t prefix_lanes[TM];
  i8_t tail_lanes[TM];
  i8_t out_lanes[TM];
#pragma HLS ARRAY_PARTITION variable=prefix_lanes complete dim=1
#pragma HLS ARRAY_PARTITION variable=tail_lanes complete dim=1
#pragma HLS ARRAY_PARTITION variable=out_lanes complete dim=1
  ppu_unpack_lanes(prefix_word, prefix_lanes);
  ppu_unpack_lanes(tail_word, tail_lanes);

  const unsigned prefix = prefix_c.to_uint();
  const unsigned tail = tail_c.to_uint();
  for (int lane = 0; lane < TM; ++lane) {
#pragma HLS UNROLL
    const unsigned lane_u = static_cast<unsigned>(lane);
    i8_t value = prefix_lanes[lane];
    if (lane_u >= prefix && lane_u < prefix + tail) {
      value = tail_lanes[lane_u - prefix];
    }
    out_lanes[lane] = value;
  }
  ppu_pack_lanes(out_lanes, out_word);
}

bool ppu_preadd_segment(const tensor_desc_t& other,
                        u16_t out_row,
                        u16_t ow_begin,
                        u16_t ow_count,
                        u8_t valid_c,
                        const add_q_t& qparam,
                        act_vec_t row_buf[MAX_FM_W]) {
#pragma HLS INLINE off
  const int ow_begin_i = static_cast<int>(ow_begin.to_uint());
  const int ow_count_i = static_cast<int>(ow_count.to_uint());
  const unsigned lane_count = valid_c == 0 || valid_c > TM ? TM : valid_c.to_uint();
  const u8_t read_count = static_cast<u8_t>(
      other.c.to_uint() < lane_count ? other.c.to_uint() : lane_count);
  const bool row_valid = out_row < other.h;
  const u32_t source_step = static_cast<u32_t>(tensor_phys_c(other));
  u32_t source_offset = ppu_compact_row_base(other, out_row) +
                         static_cast<u32_t>(ow_begin) * source_step;
  for (int ow_i = 0; ow_i < MAX_FM_W; ++ow_i) {
#pragma HLS PIPELINE off
    if (ow_i >= ow_count_i) {
      break;
    }
    act_vec_t other_word = 0;
    if (row_valid && ow_begin_i + ow_i < static_cast<int>(other.w.to_uint()) &&
        read_count != 0 &&
        !on_chip_memory_read_packed_span(other.bank_id, source_offset, read_count, other_word)) {
      return false;
    }
    act_vec_t out_word = 0;
    ppu_apply_add_word(row_buf[ow_i], other_word, valid_c, qparam, out_word);
    row_buf[ow_i] = out_word;
    source_offset += source_step;
  }
  return true;
}

bool ppu_preadd_row(const tensor_desc_t& other,
                    u16_t out_row,
                    const conv_cfg_t& cfg,
                    u8_t valid_c,
                    const add_q_t& qparam,
                    act_vec_t row_buf[MAX_FM_W]) {
#pragma HLS INLINE off
  const u16_t out_w = ppu_out_dim(cfg.in_w, ppu_effective_stride(cfg));
  return ppu_preadd_segment(other,
                            out_row,
                            static_cast<u16_t>(0),
                            out_w,
                            valid_c,
                            qparam,
                            row_buf);
}

static bool ppu_layout_to_compact_shape(u16_t store_layout, u8_t& valid_c, u8_t& phys_c) {
#pragma HLS INLINE
  switch (store_layout.to_uint()) {
    case static_cast<unsigned>(STORE_LAYOUT_COMPACT_C12):
      valid_c = static_cast<u8_t>(12);
      phys_c = static_cast<u8_t>(12);
      return true;
    case static_cast<unsigned>(STORE_LAYOUT_COMPACT_C16):
      valid_c = static_cast<u8_t>(16);
      phys_c = static_cast<u8_t>(16);
      return true;
    case static_cast<unsigned>(STORE_LAYOUT_COMPACT_C17):
      valid_c = static_cast<u8_t>(17);
      phys_c = static_cast<u8_t>(17);
      return true;
    case static_cast<unsigned>(STORE_LAYOUT_COMPACT_C25):
      valid_c = static_cast<u8_t>(25);
      phys_c = static_cast<u8_t>(25);
      return true;
    case static_cast<unsigned>(STORE_LAYOUT_COMPACT_C28):
      valid_c = static_cast<u8_t>(28);
      phys_c = static_cast<u8_t>(28);
      return true;
    default:
      valid_c = 0;
      phys_c = 0;
      return false;
  }
}

static bool ppu_write_compact_word(const tensor_desc_t& dst,
                                   u32_t byte_offset,
                                   const act_vec_t& word) {
#pragma HLS INLINE off
  return (dst.bank_id.to_uint() == static_cast<unsigned>(BANK_BRAM_SCR1))
             ? on_chip_memory_write_pool2_abs_word(byte_offset, word)
             : on_chip_memory_write_fmbuf_abs_word(dst.bank_id, byte_offset, word);
}

static bool ppu_read_abs_word(u32_t byte_offset, axi_vec_t& word) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off
  return on_chip_memory_read_fmbuf_abs_word(
      static_cast<u8_t>(static_cast<unsigned>(BANK_FMEM0)),
      byte_offset,
      word);
}

static void ppu_clear_compact_groups(ap_uint<32> groups[8]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=groups complete dim=1
  for (int group = 0; group < 8; ++group) {
#pragma HLS UNROLL
    groups[group] = 0;
  }
}

static void ppu_set_compact_group(ap_uint<32> groups[8],
                                  ap_uint<3> group_index,
                                  ap_uint<32> value) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=groups complete dim=1
  switch (group_index.to_uint()) {
    case 0: groups[0] = value; break;
    case 1: groups[1] = value; break;
    case 2: groups[2] = value; break;
    case 3: groups[3] = value; break;
    case 4: groups[4] = value; break;
    case 5: groups[5] = value; break;
    case 6: groups[6] = value; break;
    default: groups[7] = value; break;
  }
}

static act_vec_t ppu_pack_compact_groups(const ap_uint<32> groups[8]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=groups complete dim=1
  act_vec_t packed = 0;
  for (int group = 0; group < 8; ++group) {
#pragma HLS UNROLL
    packed.range(group * 32 + 31, group * 32) = groups[group];
  }
  return packed;
}

static bool ppu_append_compact_pair(ap_uint<64> input, ap_uint<4> bytes,
                                      ap_uint<32>& carry, ap_uint<2>& carry_bytes,
                                      ap_uint<32> groups[8], ap_uint<3>& group_count,
                                      act_vec_t& completed_word) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=groups complete dim=1
  if (bytes == 1U) { input &= static_cast<ap_uint<64>>(0xFFU); }
  else if (bytes == 4U) { input &= static_cast<ap_uint<64>>(0xFFFFFFFFULL); }
  ap_uint<96> incoming = static_cast<ap_uint<96>>(input);
  switch (carry_bytes.to_uint()) {
    case 1: incoming <<= 8; break;
    case 2: incoming <<= 16; break;
    case 3: incoming <<= 24; break;
    default: break;
  }
  const ap_uint<96> merged = incoming | static_cast<ap_uint<96>>(carry);
  // Layout tails are 1 or 4 bytes; all other inputs are complete 8-byte pairs.
  const ap_uint<2> count = bytes == 8U ? static_cast<ap_uint<2>>(2)
      : ((bytes == 4U || carry_bytes == 3U) ? static_cast<ap_uint<2>>(1)
                                             : static_cast<ap_uint<2>>(0));
  const ap_uint<3> second_group = group_count + static_cast<ap_uint<3>>(1);
  const ap_uint<4> next_group = static_cast<ap_uint<4>>(group_count) +
                                  static_cast<ap_uint<4>>(count);
  const bool completed = next_group[3] != 0;
  const bool second_wraps = group_count == 7U;
  const ap_uint<32> first_value = merged.range(31, 0);
  const ap_uint<32> second_value = merged.range(63, 32);
  for (int group = 0; group < 8; ++group) {
#pragma HLS UNROLL
    const bool first_here = count != 0U && group_count.to_uint() == static_cast<unsigned>(group);
    const bool second_here = count == 2U && second_group.to_uint() == static_cast<unsigned>(group);
    const ap_uint<32> old_value = groups[group];
    const ap_uint<32> updated = first_here ? first_value : (second_here ? second_value : old_value);
    const ap_uint<32> committed = first_here ? first_value
        : ((second_here && !second_wraps) ? second_value : old_value);
    completed_word.range(group * 32 + 31, group * 32) = committed;
    // If the first group completes a word, only the second group belongs to the next word.
    groups[group] = completed
        ? ((second_wraps && second_here) ? second_value : static_cast<ap_uint<32>>(0))
        : updated;
  }
  group_count = static_cast<ap_uint<3>>(next_group);
  carry = count == 2U ? static_cast<ap_uint<32>>(merged.range(95, 64))
                       : (count == 1U ? static_cast<ap_uint<32>>(merged.range(63, 32))
                                      : static_cast<ap_uint<32>>(merged.range(31, 0)));
  if (bytes == 1U) { carry_bytes += static_cast<ap_uint<2>>(1); }
  return completed;
}

static bool ppu_transform_conv_word(const row_consumer_desc_t& consumer,
                                    const tensor_desc_t& add_other,
                                    bool has_add_other,
                                    const add_q_t& add_qparam,
                                    u16_t out_row,
                                    u16_t out_col,
                                    u8_t prefix_c,
                                    u8_t valid_c,
                                    u8_t affine_lane0,
                                    const aff_q_t& affine0,
                                    const aff_q_t& affine1,
                                    const act_vec_t& conv_word,
                                    act_vec_t& transformed) {
#pragma HLS INLINE off
  const unsigned mode = consumer.mode.to_uint();
  if (mode == static_cast<unsigned>(ROW_CONSUMER_NONE)) {
    if (!has_add_other) {
      transformed = conv_word;
      return true;
    }
    act_vec_t other_word = 0;
    if (!on_chip_memory_read_packed_tile(add_other,
                                         static_cast<i32_t>(out_row),
                                         static_cast<i32_t>(out_col),
                                         static_cast<u16_t>(0),
                                         valid_c,
                                         other_word)) {
      return false;
    }
    ppu_apply_add_word(conv_word, other_word, valid_c, add_qparam, transformed);
    return true;
  }

  if (mode != static_cast<unsigned>(ROW_CONSUMER_CAT_AFFINE_STORE) ||
      !has_add_other) {
    return false;
  }
  const unsigned prefix = prefix_c.to_uint();
  const unsigned final_c = valid_c.to_uint();
  if (prefix >= final_c || final_c > static_cast<unsigned>(TM)) {
    return false;
  }
  const u8_t tail_c = static_cast<u8_t>(final_c - prefix);
  act_vec_t tail_word = 0;
  if (!on_chip_memory_read_packed_tile(add_other,
                                       static_cast<i32_t>(out_row),
                                       static_cast<i32_t>(out_col),
                                       static_cast<u16_t>(0),
                                       tail_c,
                                       tail_word)) {
    return false;
  }
  act_vec_t merged_word = 0;
  ppu_cat_tail_word(conv_word, tail_word, prefix_c, tail_c, merged_word);
  ppu_apply_compact_affine_word(merged_word,
                            valid_c,
                            affine0,
                            affine1,
                            affine_lane0,
                            consumer.act_type,
                            transformed);
  return true;
}

void ppu_consume_conv_stream(const row_consumer_desc_t& consumer,
                             const tensor_desc_t& dst,
                             const tensor_desc_t& add_other,
                             bool has_add_other,
                             const add_q_t& add_qparam,
                             const conv_cfg_t& cfg,
                             u16_t out_row,
                             hls::stream<act_vec_t>& conv_stream,
                             u8_t& status) {
#pragma HLS INLINE off
  bool ok = true;
  const unsigned mode = consumer.mode.to_uint();
  u8_t layout_valid_c = 0;
  u8_t layout_phys_c = 0;
  const u8_t valid_c = (consumer.valid_c.to_uint() == 0U)
                           ? static_cast<u8_t>(cfg.out_c.to_uint())
                           : static_cast<u8_t>(consumer.valid_c.to_uint());
  if (!ppu_layout_to_compact_shape(
          consumer.reserved0, layout_valid_c, layout_phys_c) ||
      valid_c.to_uint() != layout_valid_c.to_uint() ||
      (mode != static_cast<unsigned>(ROW_CONSUMER_NONE) &&
       mode != static_cast<unsigned>(ROW_CONSUMER_CAT_AFFINE_STORE))) {
    ok = false;
  }

  aff_q_t affine0 = aff_q_t();
  aff_q_t affine1 = aff_q_t();
#pragma HLS ARRAY_PARTITION variable=affine0.mul complete dim=1
#pragma HLS ARRAY_PARTITION variable=affine0.bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=affine0.shift complete dim=1
#pragma HLS ARRAY_PARTITION variable=affine1.mul complete dim=1
#pragma HLS ARRAY_PARTITION variable=affine1.bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=affine1.shift complete dim=1
  const unsigned affine_c0 = consumer.reserved1.to_uint();
  const u8_t affine_lane0 =
      static_cast<u8_t>(affine_c0 & static_cast<unsigned>(TM - 1));
  if (mode == static_cast<unsigned>(ROW_CONSUMER_CAT_AFFINE_STORE)) {
    const u8_t block0 = static_cast<u8_t>(affine_c0 / static_cast<unsigned>(TM));
    if (!param_dma_get_affine_qparam(consumer.affine_param_id, block0, affine0)) {
      ok = false;
    }
    const bool need_block1 =
        affine_lane0.to_uint() + valid_c.to_uint() > static_cast<unsigned>(TM);
    if (need_block1) {
      if (!param_dma_get_affine_qparam(
              consumer.affine_param_id,
              static_cast<u8_t>(block0.to_uint() + 1U),
              affine1)) {
        ok = false;
      }
    } else {
      affine1 = affine0;
    }
  }

  const unsigned phys_c = layout_phys_c.to_uint();
  const unsigned pair_count = (phys_c + 7U) / 8U;
  const ap_uint<4> tail_bytes = static_cast<ap_uint<4>>(
      (phys_c & 7U) == 0U ? 8U : (phys_c & 7U));
  u32_t write_offset = ppu_compact_row_base(dst, out_row);
  if ((write_offset.to_uint() & static_cast<unsigned>(AXI_WORD_BYTES - 1)) != 0U) {
    ok = false;
  }

  ap_uint<32> packed_groups[8];
#pragma HLS ARRAY_PARTITION variable=packed_groups complete dim=1
  ppu_clear_compact_groups(packed_groups);
  ap_uint<32> carry = 0;
  ap_uint<2> carry_bytes = 0;
  ap_uint<3> packed_group_count = 0;
  const u16_t out_w = ppu_out_dim(cfg.in_w, ppu_effective_stride(cfg));
  const int out_w_i = static_cast<int>(out_w.to_uint());
  for (int ow_i = 0; ow_i <= MAX_FM_W; ++ow_i) {
#pragma HLS PIPELINE off
    if (ow_i > out_w_i) {
      break;
    }
    act_vec_t pending_word = 0;
    bool pending_valid = false;
    if (ow_i == out_w_i) {
      if (carry_bytes != 0U) {
        ppu_set_compact_group(packed_groups, packed_group_count, carry);
      }
      pending_word = ppu_pack_compact_groups(packed_groups);
      pending_valid = packed_group_count != 0U || carry_bytes != 0U;
    } else {
      const act_vec_t conv_word = conv_stream.read();
      if (!ok) {
        continue;
      }
      act_vec_t transformed = 0;
      if (!ppu_transform_conv_word(consumer,
                                 add_other,
                                 has_add_other,
                                 add_qparam,
                                 out_row,
                                 static_cast<u16_t>(ow_i),
                                 static_cast<u8_t>(cfg.out_c.to_uint()),
                                 valid_c,
                                 affine_lane0,
                                 affine0,
                                 affine1,
                                 conv_word,
                                 transformed)) {
        ok = false;
        continue;
      }

      act_vec_t source_cursor = transformed;
      for (int pair = 0; pair < AXI_WORD_BYTES / 8; ++pair) {
#pragma HLS PIPELINE II=1
        if (static_cast<unsigned>(pair) >= pair_count) { break; }
        const ap_uint<4> pair_bytes = static_cast<unsigned>(pair + 1) == pair_count
            ? tail_bytes : static_cast<ap_uint<4>>(8);
        const ap_uint<64> source_pair = source_cursor.range(63, 0);
        source_cursor >>= 64;
        act_vec_t packed_word = 0;
        if (ppu_append_compact_pair(source_pair, pair_bytes, carry, carry_bytes,
                                      packed_groups, packed_group_count, packed_word)) {
          pending_word = packed_word;
          pending_valid = true;
        }
      }
    }
    // At most 28 input bytes per pixel: one pending word covers every layout and the row tail.
    if (ok && pending_valid) {
      if (!ppu_write_compact_word(dst, write_offset, pending_word)) {
        ok = false;
      } else {
        write_offset += static_cast<u32_t>(AXI_WORD_BYTES);
      }
    }
  }
  status = ok ? static_cast<u8_t>(1) : static_cast<u8_t>(0);
}

struct ppu_block5_compact_cursor_t {
  u32_t word_offset;
  u8_t byte_index;
  axi_vec_t word;
  bool valid;
};

static bool ppu_init_block5_compact_cursor(const tensor_desc_t& src,
                                           u16_t local_row,
                                           unsigned bytes_per_pixel,
                                           ppu_block5_compact_cursor_t& cursor) {
#pragma HLS INLINE off
  if (src.bank_id.to_uint() != static_cast<unsigned>(BANK_FMEM0) ||
      tensor_phys_c(src).to_uint() != bytes_per_pixel) {
    return false;
  }
  const u32_t start_offset =
      src.base_offset +
      static_cast<u32_t>(local_row) * tensor_row_stride_bytes(src) +
      static_cast<u32_t>(src.reserved1);
  cursor.word_offset =
      start_offset & static_cast<u32_t>(~(AXI_WORD_BYTES - 1));
  cursor.byte_index = static_cast<u8_t>(
      start_offset.to_uint() & static_cast<unsigned>(AXI_WORD_BYTES - 1));
  cursor.word = 0;
  if (!ppu_read_abs_word(cursor.word_offset, cursor.word)) {
    cursor.valid = false;
    return false;
  }
  cursor.valid = true;
  return true;
}

static bool ppu_init_block5_source_cursors(
    const tensor_desc_t& s0,
    const tensor_desc_t& s1,
    const tensor_desc_t& s2,
    const tensor_desc_t& s3,
    u16_t local_row,
    unsigned source0_bytes,
    unsigned source1_bytes,
    unsigned source2_bytes,
    unsigned source3_bytes,
    ppu_block5_compact_cursor_t& cursor0,
    ppu_block5_compact_cursor_t& cursor1,
    ppu_block5_compact_cursor_t& cursor2,
    ppu_block5_compact_cursor_t& cursor3) {
#pragma HLS INLINE
  const bool ok0 =
      ppu_init_block5_compact_cursor(s0, local_row, source0_bytes, cursor0);
  const bool ok1 =
      ppu_init_block5_compact_cursor(s1, local_row, source1_bytes, cursor1);
  const bool ok2 =
      ppu_init_block5_compact_cursor(s2, local_row, source2_bytes, cursor2);
  const bool ok3 =
      ppu_init_block5_compact_cursor(s3, local_row, source3_bytes, cursor3);
  return ok0 && ok1 && ok2 && ok3;
}

static bool ppu_read_block5_compact_cursor(unsigned byte_count,
                                           ppu_block5_compact_cursor_t& cursor,
                                           act_vec_t& packed) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off
  packed = 0;
  if (byte_count == 0U || byte_count > static_cast<unsigned>(AXI_WORD_BYTES)) {
    return false;
  }

  const unsigned byte_index = cursor.byte_index.to_uint();
  const unsigned next_byte_index = byte_index + byte_count;
  const bool need_current = !cursor.valid;
  const bool need_next_word =
      next_byte_index > static_cast<unsigned>(AXI_WORD_BYTES);
  if (need_current && need_next_word) {
    return false;
  }

  axi_vec_t current_word = cursor.word;
  axi_vec_t next_word = 0;
  const bool need_fetch = need_current || need_next_word;
  if (need_fetch) {
    const u32_t read_offset =
        cursor.word_offset +
        static_cast<u32_t>(need_next_word ? AXI_WORD_BYTES : 0);
    axi_vec_t fetched = 0;
    if (!ppu_read_abs_word(read_offset, fetched)) {
      return false;
    }
    if (need_current) {
      current_word = fetched;
    } else {
      next_word = fetched;
    }
  }

  if (next_byte_index <= static_cast<unsigned>(AXI_WORD_BYTES)) {
    packed = static_cast<act_vec_t>(current_word >> (byte_index * 8U));
    if (next_byte_index == static_cast<unsigned>(AXI_WORD_BYTES)) {
      cursor.word_offset += static_cast<u32_t>(AXI_WORD_BYTES);
      cursor.byte_index = 0;
      cursor.word = 0;
      cursor.valid = false;
    } else {
      cursor.byte_index = static_cast<u8_t>(next_byte_index);
      cursor.word = current_word;
      cursor.valid = true;
    }
    return true;
  }

  ap_uint<AXI_WORD_BITS * 2> pair = 0;
  pair.range(AXI_WORD_BITS - 1, 0) = current_word;
  pair.range(AXI_WORD_BITS * 2 - 1, AXI_WORD_BITS) = next_word;
  packed = static_cast<act_vec_t>(pair >> (byte_index * 8U));
  cursor.word_offset += static_cast<u32_t>(AXI_WORD_BYTES);
  cursor.byte_index = static_cast<u8_t>(
      next_byte_index - static_cast<unsigned>(AXI_WORD_BYTES));
  cursor.word = next_word;
  cursor.valid = true;
  return true;
}

static bool ppu_read_block5_source_words(unsigned source0_bytes,
                                         unsigned source1_bytes,
                                         unsigned source2_bytes,
                                         unsigned source3_bytes,
                                         ppu_block5_compact_cursor_t& cursor0,
                                         ppu_block5_compact_cursor_t& cursor1,
                                         ppu_block5_compact_cursor_t& cursor2,
                                         ppu_block5_compact_cursor_t& cursor3,
                                         act_vec_t& source0,
                                         act_vec_t& source1,
                                         act_vec_t& source2,
                                         act_vec_t& source3) {
#pragma HLS INLINE
  const bool ok0 =
      ppu_read_block5_compact_cursor(source0_bytes, cursor0, source0);
  const bool ok1 =
      ppu_read_block5_compact_cursor(source1_bytes, cursor1, source1);
  const bool ok2 =
      ppu_read_block5_compact_cursor(source2_bytes, cursor2, source2);
  const bool ok3 =
      ppu_read_block5_compact_cursor(source3_bytes, cursor3, source3);
  return ok0 && ok1 && ok2 && ok3;
}

static void ppu_compose_block5_l2_words(const act_vec_t& source0,
                                        const act_vec_t& source1,
                                        const act_vec_t& source2,
                                        const act_vec_t& source3,
                                        const act_vec_t& row_word,
                                        act_vec_t& tile0_word,
                                        act_vec_t& tile1_word) {
#pragma HLS INLINE
  tile0_word = 0;
  tile0_word.range(127, 0) = source0.range(127, 0);
  tile0_word.range(223, 128) = source1.range(95, 0);
  tile0_word.range(255, 224) = source2.range(31, 0);

  tile1_word = 0;
  tile1_word.range(63, 0) = source2.range(95, 32);
  tile1_word.range(159, 64) = source3.range(95, 0);
  tile1_word.range(255, 160) = row_word.range(95, 0);
}

static void ppu_compose_block5_l3_words(const act_vec_t& source0,
                                        const act_vec_t& source1,
                                        const act_vec_t& source2,
                                        const act_vec_t& source3,
                                        const act_vec_t& row_word,
                                        act_vec_t& tile0_word,
                                        act_vec_t& tile1_word,
                                        act_vec_t& tile2_word,
                                        act_vec_t& tile3_word) {
#pragma HLS INLINE
  tile0_word = 0;
  tile0_word.range(223, 0) = source0.range(223, 0);
  tile0_word.range(255, 224) = source1.range(31, 0);

  tile1_word = 0;
  tile1_word.range(167, 0) = source1.range(199, 32);
  tile1_word.range(255, 168) = source2.range(87, 0);

  tile2_word = 0;
  tile2_word.range(111, 0) = source2.range(199, 88);
  tile2_word.range(255, 112) = source3.range(143, 0);

  tile3_word = 0;
  tile3_word.range(55, 0) = source3.range(199, 144);
  tile3_word.range(255, 56) = row_word.range(199, 0);
}

static bool ppu_load_block5_affine_bank(
    u8_t param_id, int tile_count,
    i32_t qmul[PPU_BLOCK_LANES][TM],
    i32_t qbias[PPU_BLOCK_LANES][TM],
    u8_t qshift[PPU_BLOCK_LANES][TM]) {
#pragma HLS INLINE
  for (int tile = 0; tile < 4; ++tile) {
#pragma HLS PIPELINE off
    if (tile >= tile_count) { break; }
    aff_q_t loaded;
#pragma HLS ARRAY_PARTITION variable=loaded.mul complete dim=1
#pragma HLS ARRAY_PARTITION variable=loaded.bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=loaded.shift complete dim=1
    if (!param_dma_get_affine_qparam(param_id, static_cast<u8_t>(tile), loaded)) {
      return false;
    }
    for (int group = 0; group < PPU_BLOCK_GROUPS; ++group) {
      for (int lane = 0; lane < PPU_BLOCK_LANES; ++lane) {
#pragma HLS UNROLL factor=4
        const int index = tile * PPU_BLOCK_GROUPS + group;
        const int source_lane = group * PPU_BLOCK_LANES + lane;
        qmul[lane][index] = loaded.mul[source_lane];
        qbias[lane][index] = loaded.bias[source_lane];
        qshift[lane][index] = loaded.shift[source_lane];
      }
    }
  }
  return true;
}

static void ppu_apply_block_word(
    const act_vec_t& lhs_word, const act_vec_t& rhs_word,
    bool has_residual, const add_q_t& add_qparam,
    const i32_t qmul[PPU_BLOCK_LANES][TM],
    const i32_t qbias[PPU_BLOCK_LANES][TM],
    const u8_t qshift[PPU_BLOCK_LANES][TM],
    int tile, u8_t act_type, act_vec_t& out_word) {
#pragma HLS INLINE
  ap_uint<32> groups[8];
#pragma HLS ARRAY_PARTITION variable=groups complete dim=1
  act_vec_t lhs_cursor = lhs_word;
  act_vec_t rhs_cursor = rhs_word;
  for (int group = 0; group < PPU_BLOCK_GROUPS; ++group) {
#pragma HLS PIPELINE II=1
    ap_uint<32> result = 0;
    const int index = tile * PPU_BLOCK_GROUPS + group;
    for (int lane = 0; lane < PPU_BLOCK_LANES; ++lane) {
#pragma HLS UNROLL factor=4
      i8_t lhs = 0;
      i8_t rhs = 0;
      lhs.range(7, 0) = lhs_cursor.range(lane * 8 + 7, lane * 8);
      rhs.range(7, 0) = rhs_cursor.range(lane * 8 + 7, lane * 8);
      // Saturation at the add boundary is part of the exported INT8 contract.
      const i8_t value = has_residual ? add_i8_bounded(lhs, rhs, add_qparam) : lhs;
      const i8_t transformed = affine_i8_to_i8_bounded(
          value, qmul[lane][index], qbias[lane][index], qshift[lane][index], act_type);
      result.range(lane * 8 + 7, lane * 8) = transformed.range(7, 0);
    }
    ppu_set_compact_group(groups, static_cast<u8_t>(group), result);
    lhs_cursor >>= 32;
    rhs_cursor >>= 32;
  }
  out_word = ppu_pack_compact_groups(groups);
}

static bool ppu_finalize_block5_emit_word(const tensor_desc_t& residual,
                                          const tensor_desc_t& final_dst,
                                          bool has_residual,
                                          const add_q_t& residual_add_qparam,
                                          const i32_t qmul[PPU_BLOCK_LANES][TM],
                                          const i32_t qbias[PPU_BLOCK_LANES][TM],
                                          const u8_t qshift[PPU_BLOCK_LANES][TM],
                                          u8_t act_type,
                                          u32_t residual_offset,
                                          u32_t dst_offset,
                                          int tile,
                                          const act_vec_t& cat_word) {
#pragma HLS INLINE
  act_vec_t residual_word = 0;
  act_vec_t out_word = 0;
  if (has_residual &&
      !on_chip_memory_read_packed_span(residual.bank_id, residual_offset,
                                       static_cast<u8_t>(TM), residual_word)) {
    return false;
  }
  ppu_apply_block_word(cat_word, residual_word, has_residual, residual_add_qparam,
                        qmul, qbias, qshift, tile, act_type, out_word);
  return conv_store_write_row_contiguous_word(final_dst, dst_offset, out_word);
}

static bool ppu_finalize_block5_static_row(const block5_sched_desc_t& sched,
                                           const tensor_desc_t& s0,
                                           const tensor_desc_t& s1,
                                           const tensor_desc_t& s2,
                                           const tensor_desc_t& s3,
                                           const tensor_desc_t& residual,
                                           const tensor_desc_t& final_dst,
                                           bool has_residual,
                                           const add_q_t& residual_add_qparam,
                                           u8_t act_type,
                                           u16_t local_row,
                                           u16_t abs_row,
                                           act_vec_t row_buf[MAX_FM_W]) {
#pragma HLS INLINE off
  static i32_t qmul[PPU_BLOCK_LANES][TM];
  static i32_t qbias[PPU_BLOCK_LANES][TM];
  static u8_t qshift[PPU_BLOCK_LANES][TM];
#pragma HLS ARRAY_PARTITION variable=qmul complete dim=1
#pragma HLS ARRAY_PARTITION variable=qbias complete dim=1
#pragma HLS ARRAY_PARTITION variable=qshift complete dim=1
#pragma HLS BIND_STORAGE variable=qmul type=ram_1p impl=lutram
#pragma HLS BIND_STORAGE variable=qbias type=ram_1p impl=lutram
#pragma HLS BIND_STORAGE variable=qshift type=ram_1p impl=lutram
#pragma HLS RESET variable=qmul off
#pragma HLS RESET variable=qbias off
#pragma HLS RESET variable=qshift off
  static bool affine_cache_valid = false;
  static u32_t affine_cache_generation = 0;
  static u8_t affine_cache_param = 0;
  static u8_t affine_cache_pattern = 0;
  static u16_t affine_cache_next_row = 0;
  const u32_t generation = param_dma_generation();
  const bool reuse_affine = affine_cache_valid && abs_row != 0 &&
      affine_cache_generation == generation && affine_cache_param == sched.affine_param_id &&
      affine_cache_pattern == sched.pattern && affine_cache_next_row == abs_row;
  // Any rejected/incomplete row invalidates the only coefficient cache.
  affine_cache_valid = false;
  if (!ppu_full_aligned_tile_write_plan_ok(final_dst, sched.valid_c)) {
    return false;
  }
  const unsigned pattern = sched.pattern.to_uint();
  const bool is_l3 = pattern == static_cast<unsigned>(BLOCK5_PATTERN_L3_C28_4C25);
  if (!is_l3 && pattern != static_cast<unsigned>(BLOCK5_PATTERN_L2_C16_4C12)) {
    return false;
  }
  if (abs_row >= final_dst.h || sched.out_w > final_dst.w ||
      (has_residual &&
       (abs_row >= residual.h || sched.out_w > residual.w ||
        !ppu_full_aligned_tile_write_plan_ok(residual, sched.valid_c)))) {
    return false;
  }

  ppu_block5_compact_cursor_t cursor0;
  ppu_block5_compact_cursor_t cursor1;
  ppu_block5_compact_cursor_t cursor2;
  ppu_block5_compact_cursor_t cursor3;
  const unsigned source0_bytes = is_l3 ? 28U : 16U;
  const unsigned source1_bytes = is_l3 ? 25U : 12U;
  const unsigned source2_bytes = is_l3 ? 25U : 12U;
  const unsigned source3_bytes = is_l3 ? 25U : 12U;
  if (!ppu_init_block5_source_cursors(s0,
                                      s1,
                                      s2,
                                      s3,
                                      local_row,
                                      source0_bytes,
                                      source1_bytes,
                                      source2_bytes,
                                      source3_bytes,
                                      cursor0,
                                      cursor1,
                                      cursor2,
                                      cursor3)) {
    return false;
  }

  const int tile_count = is_l3 ? 4 : 2;
  if (!reuse_affine &&
      !ppu_load_block5_affine_bank(sched.affine_param_id, tile_count, qmul, qbias, qshift)) {
    return false;
  }

  const int out_w_i = static_cast<int>(sched.out_w.to_uint());
  u32_t dst_pixel = ppu_compact_row_base(final_dst, abs_row);
  u32_t residual_pixel = has_residual ? ppu_compact_row_base(residual, abs_row)
                                      : static_cast<u32_t>(0);
  const u32_t dst_step = static_cast<u32_t>(tensor_phys_c(final_dst));
  const u32_t residual_step = has_residual
                                  ? static_cast<u32_t>(tensor_phys_c(residual))
                                  : static_cast<u32_t>(0);
  for (int ow_i = 0; ow_i < MAX_FM_W; ++ow_i) {
#pragma HLS PIPELINE off
    if (ow_i >= out_w_i) {
      break;
    }
    act_vec_t tile0_word = 0;
    act_vec_t tile1_word = 0;
    act_vec_t tile2_word = 0;
    act_vec_t tile3_word = 0;
    act_vec_t source0 = 0;
    act_vec_t source1 = 0;
    act_vec_t source2 = 0;
    act_vec_t source3 = 0;
    const act_vec_t row_word = row_buf[ow_i];
    const bool compose_ok = ppu_read_block5_source_words(source0_bytes,
                                                        source1_bytes,
                                                        source2_bytes,
                                                        source3_bytes,
                                                        cursor0,
                                                        cursor1,
                                                        cursor2,
                                                        cursor3,
                                                        source0,
                                                        source1,
                                                        source2,
                                                        source3);
    if (compose_ok) {
      if (is_l3) {
        ppu_compose_block5_l3_words(source0,
                                    source1,
                                    source2,
                                    source3,
                                    row_word,
                                    tile0_word,
                                    tile1_word,
                                    tile2_word,
                                    tile3_word);
      } else {
        ppu_compose_block5_l2_words(source0,
                                    source1,
                                    source2,
                                    source3,
                                    row_word,
                                    tile0_word,
                                    tile1_word);
      }
    }
    if (!compose_ok) {
      return false;
    }
    for (int tile = 0; tile < 4; ++tile) {
#pragma HLS PIPELINE off
      if (tile >= tile_count) { break; }
      act_vec_t cat_word = tile0_word;
      if (tile == 1) { cat_word = tile1_word; }
      else if (tile == 2) { cat_word = tile2_word; }
      else if (tile == 3) { cat_word = tile3_word; }
      const u32_t c = static_cast<u32_t>(tile * TM);
      if (!ppu_finalize_block5_emit_word(
              residual, final_dst, has_residual, residual_add_qparam,
              qmul, qbias, qshift, act_type, residual_pixel + c, dst_pixel + c,
              tile, cat_word)) {
        return false;
      }
    }
    dst_pixel += dst_step;
    residual_pixel += residual_step;
  }
  affine_cache_generation = generation;
  affine_cache_param = sched.affine_param_id;
  affine_cache_pattern = sched.pattern;
  affine_cache_next_row = abs_row + static_cast<u16_t>(1);
  affine_cache_valid = true;
  return true;
}


#if !defined(__SYNTHESIS__) && defined(ESP_INT8_CSIM_DUMP_LOWRES_LOGITS_SIDE)
static unsigned s_csim_lowres_logits_rows = 0;

static bool csim_dump_lowres_logits_row(u16_t out_row,
                                        u8_t valid_c,
                                        const act_vec_t row_buf[MAX_FM_W]) {
  const unsigned row = out_row.to_uint();
  const bool first_row = row == 0U || s_csim_lowres_logits_rows == 0U;
  std::ofstream out("csim_u72_lowres_logits.bin",
                    std::ios::binary | (first_row ? std::ios::trunc : std::ios::app));
  if (!out) {
    std::printf("[CSIM-DUMP] failed to open csim_u72_lowres_logits.bin\n");
    return false;
  }

  const unsigned class_count = valid_c.to_uint();
  std::uint8_t bytes[MAX_CLASS_C];
  for (unsigned ow = 0; ow < static_cast<unsigned>(ENCODER_OUT_W); ++ow) {
    const act_vec_t packed = row_buf[ow];
    for (unsigned lane = 0; lane < class_count; ++lane) {
      bytes[lane] = static_cast<std::uint8_t>(packed.range(lane * 8 + 7, lane * 8).to_uint());
    }
    out.write(reinterpret_cast<const char*>(bytes),
              static_cast<std::streamsize>(class_count));
  }

  ++s_csim_lowres_logits_rows;
  if (row == static_cast<unsigned>(ENCODER_OUT_H - 1)) {
    std::printf("[CSIM-DUMP] wrote csim_u72_lowres_logits.bin rows=%u shape=%ux%ux%u bytes=%u\n",
                s_csim_lowres_logits_rows,
                static_cast<unsigned>(ENCODER_OUT_H),
                static_cast<unsigned>(ENCODER_OUT_W),
                class_count,
                static_cast<unsigned>(ENCODER_OUT_H * ENCODER_OUT_W) * class_count);
  }
  return true;
}
#endif

bool ppu_consume_upsample_row(axi_vec_t* gmem_frame_out,
                              u16_t out_row,
                              u8_t valid_c,
                              const act_vec_t row_buf[MAX_FM_W]) {
#pragma HLS INLINE off
#if !defined(__SYNTHESIS__) && defined(ESP_INT8_CSIM_DUMP_LOWRES_LOGITS_SIDE)
  if (!csim_dump_lowres_logits_row(out_row, valid_c, row_buf)) {
    return false;
  }
#endif
  upsample_fused_consume_logits_row(gmem_frame_out, out_row, valid_c, row_buf);
  return true;
}

bool ppu_consume_block5_final_row(const block5_sched_desc_t& sched,
                                  const tensor_desc_t& scratch0,
                                  const tensor_desc_t& scratch1,
                                  const tensor_desc_t& scratch2,
                                  const tensor_desc_t& scratch3,
                                  const tensor_desc_t& residual,
                                  const tensor_desc_t& final_dst,
                                  bool has_residual,
                                  const add_q_t& residual_add_qparam,
                                  u8_t act_type,
                                  u16_t local_row,
                                  u16_t abs_row,
                                  act_vec_t row_buf[MAX_FM_W]) {
#pragma HLS INLINE off
  return ppu_finalize_block5_static_row(sched,
                                        scratch0,
                                        scratch1,
                                        scratch2,
                                        scratch3,
                                        residual,
                                        final_dst,
                                        has_residual,
                                        residual_add_qparam,
                                        act_type,
                                        local_row,
                                        abs_row,
                                        row_buf);
}

} // namespace esp_int8
