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

bool param_dma_get_affine_qparam(u8_t param_id, u8_t block_id, aff_q_t& qparam);
bool on_chip_memory_read_aligned_full_tile(const tensor_desc_t& desc,
                                           i32_t h,
                                           i32_t w,
                                           u16_t c_begin,
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
bool on_chip_memory_read_pool2_abs_word(u32_t byte_offset,
                                        axi_vec_t& packed);
bool on_chip_memory_write_fmbuf_abs_word(u8_t bank_id,
                                         u32_t byte_offset,
                                         act_vec_t packed);
bool on_chip_memory_write_pool2_abs_word(u32_t byte_offset,
                                         act_vec_t packed);
bool conv_store_write_aligned_tile(const tensor_desc_t& dst,
                                   u16_t h,
                                   u16_t w,
                                   u16_t c,
                                   const act_vec_t& word);
void upsample_fused_consume_logits_row(axi_vec_t* gmem_frame_out,
                                       u16_t encoder_row,
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

static u16_t ppu_tensor_phys_c(const tensor_desc_t& desc) {
#pragma HLS INLINE
  return (desc.reserved0.to_uint() == 0U) ? desc.c : desc.reserved0;
}

static bool ppu_can_use_aligned_full_tile(const tensor_desc_t& desc, u16_t c_begin) {
#pragma HLS INLINE
  const unsigned phys_c = ppu_tensor_phys_c(desc).to_uint();
  const unsigned start_c = desc.reserved1.to_uint() + c_begin.to_uint();
  const unsigned base_start = desc.base_offset.to_uint() + start_c;
  return phys_c >= static_cast<unsigned>(AXI_WORD_BYTES) &&
         ((phys_c & static_cast<unsigned>(AXI_WORD_BYTES - 1)) == 0U) &&
         c_begin.to_uint() + static_cast<unsigned>(AXI_WORD_BYTES) <= desc.c.to_uint() &&
         ((base_start & static_cast<unsigned>(AXI_WORD_BYTES - 1)) == 0U);
}

static bool ppu_full_aligned_tile_write_plan_ok(const tensor_desc_t& desc, u16_t total_c) {
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
    if (!ppu_can_use_aligned_full_tile(desc, c)) {
      return false;
    }
  }
  return true;
}

static u32_t ppu_compact_row_base(const tensor_desc_t& dst, u16_t out_row) {
#pragma HLS INLINE
  return dst.base_offset +
         static_cast<u32_t>(out_row) * static_cast<u32_t>(dst.w) *
             static_cast<u32_t>(ppu_tensor_phys_c(dst)) +
         static_cast<u32_t>(dst.reserved1);
}

static u8_t ppu_get_act_byte(const act_vec_t& word, int lane) {
#pragma HLS INLINE
  return word.range(lane * 8 + 7, lane * 8);
}

static void ppu_set_act_byte(act_vec_t& word, int lane, u8_t value) {
#pragma HLS INLINE
  word.range(lane * 8 + 7, lane * 8) = value;
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

static void ppu_apply_add_word(const act_vec_t& lhs_word,
                               const act_vec_t& rhs_word,
                               u8_t valid_c,
                               const add_q_t& add_qparam,
                               act_vec_t& out_word) {
#pragma HLS INLINE off
  i8_t out_lanes[TM];
#pragma HLS ARRAY_PARTITION variable=out_lanes complete dim=1
  const unsigned valid = valid_c.to_uint();
  for (int lane = 0; lane < TM; ++lane) {
#pragma HLS UNROLL factor=8
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

static void ppu_apply_c19_affine_word(const act_vec_t& in_word,
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
#pragma HLS UNROLL factor=8
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

static void ppu_apply_block_affine_word(const act_vec_t& in_word,
                                        const aff_q_t& qparam,
                                        u8_t act_type,
                                        act_vec_t& out_word) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=qparam.mul complete dim=1
#pragma HLS ARRAY_PARTITION variable=qparam.bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=qparam.shift complete dim=1
  i8_t out_lanes[TM];
#pragma HLS ARRAY_PARTITION variable=out_lanes complete dim=1
  for (int lane = 0; lane < TM; ++lane) {
#pragma HLS UNROLL factor=8
    const i8_t in_value = ppu_get_act_i8_dynamic(in_word, lane);
    out_lanes[lane] = affine_i8_to_i8(in_value,
                                      qparam.mul[lane],
                                      qparam.bias[lane],
                                      qparam.shift[lane],
                                      act_type);
  }
  ppu_pack_lanes(out_lanes, out_word);
}

static void ppu_apply_block_add_affine_word(const act_vec_t& lhs_word,
                                            const act_vec_t& rhs_word,
                                            const add_q_t& add_qparam,
                                            const aff_q_t& aff_qparam,
                                            u8_t act_type,
                                            act_vec_t& out_word) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=aff_qparam.mul complete dim=1
#pragma HLS ARRAY_PARTITION variable=aff_qparam.bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=aff_qparam.shift complete dim=1
  i8_t out_lanes[TM];
#pragma HLS ARRAY_PARTITION variable=out_lanes complete dim=1
  for (int lane = 0; lane < TM; ++lane) {
#pragma HLS UNROLL factor=8
    const i8_t lhs_value = ppu_get_act_i8_dynamic(lhs_word, lane);
    const i8_t rhs_value = ppu_get_act_i8_dynamic(rhs_word, lane);
    const i8_t added_value = add_i8(lhs_value, rhs_value, add_qparam);
    out_lanes[lane] = affine_i8_to_i8(added_value,
                                      aff_qparam.mul[lane],
                                      aff_qparam.bias[lane],
                                      aff_qparam.shift[lane],
                                      act_type);
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

static void ppu_add_other_row_to_buffer(const tensor_desc_t& other,
                                        u16_t out_row,
                                        const conv_cfg_t& cfg,
                                        u8_t valid_c,
                                        const add_q_t& qparam,
                                        act_vec_t row_buf[MAX_FM_W],
                                        bool& ok) {
#pragma HLS INLINE off
  const u16_t stride = ppu_effective_stride(cfg);
  const u16_t out_w = ppu_out_dim(cfg.in_w, stride);
  const int out_w_i = static_cast<int>(out_w.to_uint());
  for (int ow_i = 0; ow_i < MAX_FM_W; ++ow_i) {
#pragma HLS PIPELINE off
    if (ow_i >= out_w_i) {
      break;
    }
    act_vec_t other_word = 0;
    if (!on_chip_memory_read_packed_tile(other,
                                         static_cast<i32_t>(out_row),
                                         static_cast<i32_t>(ow_i),
                                         static_cast<u16_t>(0),
                                         valid_c,
                                         other_word)) {
      ok = false;
      return;
    }
    act_vec_t out_word = 0;
    ppu_apply_add_word(row_buf[ow_i], other_word, valid_c, qparam, out_word);
    row_buf[ow_i] = out_word;
  }
}

static void ppu_cat_other_row_to_buffer(const tensor_desc_t& other,
                                        u16_t out_row,
                                        const conv_cfg_t& cfg,
                                        u8_t prefix_c,
                                        u8_t tail_c,
                                        act_vec_t row_buf[MAX_FM_W],
                                        bool& ok) {
#pragma HLS INLINE off
  const u16_t stride = ppu_effective_stride(cfg);
  const u16_t out_w = ppu_out_dim(cfg.in_w, stride);
  const int out_w_i = static_cast<int>(out_w.to_uint());
  for (int ow_i = 0; ow_i < MAX_FM_W; ++ow_i) {
#pragma HLS PIPELINE off
    if (ow_i >= out_w_i) {
      break;
    }
    act_vec_t tail_word = 0;
    if (!on_chip_memory_read_packed_tile(other,
                                         static_cast<i32_t>(out_row),
                                         static_cast<i32_t>(ow_i),
                                         static_cast<u16_t>(0),
                                         tail_c,
                                         tail_word)) {
      ok = false;
      return;
    }
    act_vec_t merged_word = 0;
    ppu_cat_tail_word(row_buf[ow_i], tail_word, prefix_c, tail_c, merged_word);
    row_buf[ow_i] = merged_word;
  }
}

static void ppu_apply_row_affine(u8_t affine_param_id,
                                 u16_t affine_c_offset,
                                 u8_t act_type,
                                 const conv_cfg_t& cfg,
                                 u8_t valid_c,
                                 act_vec_t row_buf[MAX_FM_W],
                                 bool& ok) {
#pragma HLS INLINE off
  aff_q_t qparam0;
  aff_q_t qparam1;
#pragma HLS ARRAY_PARTITION variable=qparam0.mul complete dim=1
#pragma HLS ARRAY_PARTITION variable=qparam0.bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=qparam0.shift complete dim=1
#pragma HLS ARRAY_PARTITION variable=qparam1.mul complete dim=1
#pragma HLS ARRAY_PARTITION variable=qparam1.bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=qparam1.shift complete dim=1

  const unsigned c0 = affine_c_offset.to_uint();
  const unsigned lane0 = c0 & static_cast<unsigned>(TM - 1);
  const u8_t block0 = static_cast<u8_t>(c0 / static_cast<unsigned>(TM));
  const bool need_block1 = (lane0 + valid_c.to_uint()) > static_cast<unsigned>(TM);

  if (!param_dma_get_affine_qparam(affine_param_id, block0, qparam0)) {
    ok = false;
    return;
  }
  if (need_block1) {
    if (!param_dma_get_affine_qparam(affine_param_id,
                                     static_cast<u8_t>(block0.to_uint() + 1U),
                                     qparam1)) {
      ok = false;
      return;
    }
  } else {
    qparam1 = qparam0;
  }

  const u16_t stride = ppu_effective_stride(cfg);
  const u16_t out_w = ppu_out_dim(cfg.in_w, stride);
  const int out_w_i = static_cast<int>(out_w.to_uint());
  for (int ow_i = 0; ow_i < MAX_FM_W; ++ow_i) {
#pragma HLS PIPELINE off
    if (ow_i >= out_w_i) {
      break;
    }
    act_vec_t out_word = 0;
    ppu_apply_c19_affine_word(row_buf[ow_i],
                              valid_c,
                              qparam0,
                              qparam1,
                              static_cast<u8_t>(lane0),
                              act_type,
                              out_word);
    row_buf[ow_i] = out_word;
  }
}

static void ppu_write_compact_word(u8_t dst_bank,
                                   u32_t row_base,
                                   u32_t byte_offset,
                                   act_vec_t word,
                                   bool& ok) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off
  const u32_t abs_offset = row_base + byte_offset;
  const bool write_ok =
      (dst_bank.to_uint() == static_cast<unsigned>(BANK_BRAM_SCR1))
          ? on_chip_memory_write_pool2_abs_word(abs_offset, word)
          : on_chip_memory_write_fmbuf_abs_word(dst_bank, abs_offset, word);
  if (!write_ok) {
    ok = false;
  }
}

static bool ppu_store_compact_row_core(const tensor_desc_t& dst,
                                       u16_t out_row,
                                       int out_w_i,
                                       u8_t valid_c,
                                       u8_t phys_c,
                                       act_vec_t row_buf[MAX_FM_W]) {
#pragma HLS INLINE off
  bool ok = true;
  const u32_t row_base = ppu_compact_row_base(dst, out_row);
  const unsigned valid = valid_c.to_uint();
  const unsigned phys = phys_c.to_uint();
  if (valid == 0U || valid > static_cast<unsigned>(TM) ||
      phys < valid || phys > static_cast<unsigned>(TM)) {
    return false;
  }

  act_vec_t out_word = 0;
  int packed_lane = 0;
  u32_t byte_offset = 0;
  for (int pix = 0; pix < MAX_FM_W; ++pix) {
#pragma HLS PIPELINE off
    if (pix >= out_w_i) {
      break;
    }
    for (int ch = 0; ch < TM; ++ch) {
#pragma HLS PIPELINE off
      if (static_cast<unsigned>(ch) >= phys) {
        break;
      }
      const u8_t value = (static_cast<unsigned>(ch) < valid)
                             ? ppu_get_act_byte(row_buf[pix], ch)
                             : static_cast<u8_t>(0);
      ppu_set_act_byte(out_word, packed_lane, value);
      ++packed_lane;
      if (packed_lane == AXI_WORD_BYTES) {
        ppu_write_compact_word(dst.bank_id, row_base, byte_offset, out_word, ok);
        if (!ok) {
          return false;
        }
        out_word = 0;
        packed_lane = 0;
        byte_offset += static_cast<u32_t>(AXI_WORD_BYTES);
      }
    }
  }
  if (packed_lane != 0) {
    ppu_write_compact_word(dst.bank_id, row_base, byte_offset, out_word, ok);
  }
  return ok;
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
    case static_cast<unsigned>(STORE_LAYOUT_COMPACT_C19):
      valid_c = static_cast<u8_t>(19);
      phys_c = static_cast<u8_t>(19);
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

static bool ppu_store_compact_row(const tensor_desc_t& dst,
                                  u16_t out_row,
                                  u16_t store_layout,
                                  const conv_cfg_t& cfg,
                                  act_vec_t row_buf[MAX_FM_W]) {
#pragma HLS INLINE off
  const u16_t stride = ppu_effective_stride(cfg);
  const u16_t out_w = ppu_out_dim(cfg.in_w, stride);
  const int out_w_i = static_cast<int>(out_w.to_uint());
  if (store_layout.to_uint() != static_cast<unsigned>(STORE_LAYOUT_COMPACT_C12) &&
      store_layout.to_uint() != static_cast<unsigned>(STORE_LAYOUT_COMPACT_C19) &&
      store_layout.to_uint() != static_cast<unsigned>(STORE_LAYOUT_COMPACT_C25)) {
    return false;
  }
  u8_t valid = 0;
  u8_t phys = 0;
  if (!ppu_layout_to_compact_shape(store_layout, valid, phys)) {
    return false;
  }
  return ppu_store_compact_row_core(dst, out_row, out_w_i, valid, phys, row_buf);
}

static bool ppu_store_block5_scratch_row(const tensor_desc_t& dst,
                                         u16_t out_row,
                                         u16_t store_layout,
                                         const conv_cfg_t& cfg,
                                         act_vec_t row_buf[MAX_FM_W]) {
#pragma HLS INLINE off
  const u16_t stride = ppu_effective_stride(cfg);
  const u16_t out_w = ppu_out_dim(cfg.in_w, stride);
  const int out_w_i = static_cast<int>(out_w.to_uint());
  if (store_layout.to_uint() != static_cast<unsigned>(STORE_LAYOUT_COMPACT_C12) &&
      store_layout.to_uint() != static_cast<unsigned>(STORE_LAYOUT_COMPACT_C16) &&
      store_layout.to_uint() != static_cast<unsigned>(STORE_LAYOUT_COMPACT_C25) &&
      store_layout.to_uint() != static_cast<unsigned>(STORE_LAYOUT_COMPACT_C28)) {
    return false;
  }
  u8_t valid = 0;
  u8_t phys = 0;
  if (!ppu_layout_to_compact_shape(store_layout, valid, phys)) {
    return false;
  }
  return ppu_store_compact_row_core(dst, out_row, out_w_i, valid, phys, row_buf);
}

static bool ppu_block5_read_abs_word(const tensor_desc_t& desc, u32_t byte_offset, axi_vec_t& word) {
#pragma HLS INLINE
  if (desc.bank_id.to_uint() == static_cast<unsigned>(BANK_BRAM_SCR1)) {
    return on_chip_memory_read_pool2_abs_word(byte_offset, word);
  }
  return on_chip_memory_read_fmbuf_abs_word(desc.bank_id, byte_offset, word);
}

static u32_t ppu_tensor_byte_offset(const tensor_desc_t& desc, u16_t h, u16_t w, u16_t c_begin) {
#pragma HLS INLINE
  return static_cast<u32_t>((static_cast<u32_t>(h) * desc.w + w) *
                                static_cast<u32_t>(ppu_tensor_phys_c(desc)) +
                            desc.reserved1 + c_begin);
}

static bool ppu_read_block5_compact_bytes(const tensor_desc_t& src,
                                          u16_t local_row,
                                          u16_t ow,
                                          u16_t src_c,
                                          unsigned byte_count,
                                          act_vec_t& packed) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off
  packed = 0;
  if (byte_count == 0U) {
    return true;
  }
  const u32_t start_offset =
      src.base_offset + ppu_tensor_byte_offset(src, local_row, ow, src_c);
  const u32_t word0_offset = start_offset & static_cast<u32_t>(~(AXI_WORD_BYTES - 1));
  const unsigned byte0 = start_offset.to_uint() & static_cast<unsigned>(AXI_WORD_BYTES - 1);
  axi_vec_t word0 = 0;
  if (!ppu_block5_read_abs_word(src, word0_offset, word0)) {
    return false;
  }
  if (byte0 + byte_count <= static_cast<unsigned>(AXI_WORD_BYTES)) {
    packed = static_cast<act_vec_t>(word0 >> (byte0 * 8U));
    return true;
  }
  axi_vec_t word1 = 0;
  if (!ppu_block5_read_abs_word(src, word0_offset + static_cast<u32_t>(AXI_WORD_BYTES), word1)) {
    return false;
  }
  ap_uint<AXI_WORD_BITS * 2> pair = 0;
  pair.range(AXI_WORD_BITS - 1, 0) = word0;
  pair.range(AXI_WORD_BITS * 2 - 1, AXI_WORD_BITS) = word1;
  packed = static_cast<act_vec_t>(pair >> (byte0 * 8U));
  return true;
}

static void ppu_clear_lanes(i8_t lanes[TM]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
  for (int lane = 0; lane < TM; ++lane) {
#pragma HLS UNROLL
    lanes[lane] = 0;
  }
}

template <int SRC_C, int COUNT, int DST_LANE>
struct PpuBlock5LaneCopy {
  static void copy(const act_vec_t& packed, i8_t lanes[TM]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
    PpuBlock5LaneCopy<SRC_C, COUNT - 1, DST_LANE>::copy(packed, lanes);
    lanes[DST_LANE + COUNT - 1] = ppu_get_act_i8_dynamic(packed, SRC_C + COUNT - 1);
  }
};

template <int SRC_C, int DST_LANE>
struct PpuBlock5LaneCopy<SRC_C, 0, DST_LANE> {
  static void copy(const act_vec_t&, i8_t[TM]) {
#pragma HLS INLINE
  }
};

template <int SRC_C, int COUNT, int DST_LANE>
static bool ppu_read_compact_segment_to_lanes(const tensor_desc_t& src,
                                              u16_t local_row,
                                              u16_t ow,
                                              i8_t lanes[TM]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
  static_assert(SRC_C >= 0 && COUNT >= 0 && DST_LANE >= 0, "invalid BLOCK5 segment");
  static_assert(COUNT <= TM, "BLOCK5 segment copy exceeds tile width");
  static_assert(DST_LANE + COUNT <= TM, "BLOCK5 destination lane overflow");
  act_vec_t packed = 0;
  if (!ppu_read_block5_compact_bytes(src,
                                     local_row,
                                     ow,
                                     static_cast<u16_t>(SRC_C),
                                     static_cast<unsigned>(COUNT),
                                     packed)) {
    return false;
  }
  PpuBlock5LaneCopy<0, COUNT, DST_LANE>::copy(packed, lanes);
  return true;
}

template <int SRC_C, int COUNT, int DST_LANE>
static void ppu_read_rowbuf_segment_to_lanes(const act_vec_t row_buf[MAX_FM_W],
                                             u16_t ow,
                                             i8_t lanes[TM]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
  static_assert(SRC_C >= 0 && COUNT >= 0 && DST_LANE >= 0, "invalid BLOCK5 rowbuf segment");
  static_assert(SRC_C + COUNT <= TM, "BLOCK5 rowbuf source overflow");
  static_assert(DST_LANE + COUNT <= TM, "BLOCK5 rowbuf destination overflow");
  const act_vec_t packed = row_buf[static_cast<int>(ow.to_uint())];
  PpuBlock5LaneCopy<SRC_C, COUNT, DST_LANE>::copy(packed, lanes);
}

static bool ppu_block5_l2_tile0_to_word(const tensor_desc_t& s0,
                                        const tensor_desc_t& s1,
                                        const tensor_desc_t& s2,
                                        u16_t local_row,
                                        u16_t ow,
                                        act_vec_t& out_word) {
#pragma HLS INLINE off
  i8_t lanes[TM];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
  ppu_clear_lanes(lanes);
  if (!ppu_read_compact_segment_to_lanes<0, 16, 0>(s0, local_row, ow, lanes) ||
      !ppu_read_compact_segment_to_lanes<0, 12, 16>(s1, local_row, ow, lanes) ||
      !ppu_read_compact_segment_to_lanes<0, 4, 28>(s2, local_row, ow, lanes)) {
    return false;
  }
  ppu_pack_lanes(lanes, out_word);
  return true;
}

static bool ppu_block5_l2_tile1_to_word(const tensor_desc_t& s2,
                                        const tensor_desc_t& s3,
                                        const act_vec_t row_buf[MAX_FM_W],
                                        u16_t local_row,
                                        u16_t ow,
                                        act_vec_t& out_word) {
#pragma HLS INLINE off
  i8_t lanes[TM];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
  ppu_clear_lanes(lanes);
  if (!ppu_read_compact_segment_to_lanes<4, 8, 0>(s2, local_row, ow, lanes) ||
      !ppu_read_compact_segment_to_lanes<0, 12, 8>(s3, local_row, ow, lanes)) {
    return false;
  }
  ppu_read_rowbuf_segment_to_lanes<0, 12, 20>(row_buf, ow, lanes);
  ppu_pack_lanes(lanes, out_word);
  return true;
}

static bool ppu_block5_l3_tile0_to_word(const tensor_desc_t& s0,
                                        const tensor_desc_t& s1,
                                        u16_t local_row,
                                        u16_t ow,
                                        act_vec_t& out_word) {
#pragma HLS INLINE off
  i8_t lanes[TM];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
  ppu_clear_lanes(lanes);
  if (!ppu_read_compact_segment_to_lanes<0, 28, 0>(s0, local_row, ow, lanes) ||
      !ppu_read_compact_segment_to_lanes<0, 4, 28>(s1, local_row, ow, lanes)) {
    return false;
  }
  ppu_pack_lanes(lanes, out_word);
  return true;
}

static bool ppu_block5_l3_tile1_to_word(const tensor_desc_t& s1,
                                        const tensor_desc_t& s2,
                                        u16_t local_row,
                                        u16_t ow,
                                        act_vec_t& out_word) {
#pragma HLS INLINE off
  i8_t lanes[TM];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
  ppu_clear_lanes(lanes);
  if (!ppu_read_compact_segment_to_lanes<4, 21, 0>(s1, local_row, ow, lanes) ||
      !ppu_read_compact_segment_to_lanes<0, 11, 21>(s2, local_row, ow, lanes)) {
    return false;
  }
  ppu_pack_lanes(lanes, out_word);
  return true;
}

static bool ppu_block5_l3_tile2_to_word(const tensor_desc_t& s2,
                                        const tensor_desc_t& s3,
                                        u16_t local_row,
                                        u16_t ow,
                                        act_vec_t& out_word) {
#pragma HLS INLINE off
  i8_t lanes[TM];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
  ppu_clear_lanes(lanes);
  if (!ppu_read_compact_segment_to_lanes<11, 14, 0>(s2, local_row, ow, lanes) ||
      !ppu_read_compact_segment_to_lanes<0, 18, 14>(s3, local_row, ow, lanes)) {
    return false;
  }
  ppu_pack_lanes(lanes, out_word);
  return true;
}

static bool ppu_block5_l3_tile3_to_word(const tensor_desc_t& s3,
                                        const act_vec_t row_buf[MAX_FM_W],
                                        u16_t local_row,
                                        u16_t ow,
                                        act_vec_t& out_word) {
#pragma HLS INLINE off
  i8_t lanes[TM];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
  ppu_clear_lanes(lanes);
  if (!ppu_read_compact_segment_to_lanes<18, 7, 0>(s3, local_row, ow, lanes)) {
    return false;
  }
  ppu_read_rowbuf_segment_to_lanes<0, 25, 7>(row_buf, ow, lanes);
  ppu_pack_lanes(lanes, out_word);
  return true;
}

static bool ppu_block5_l2_tile_to_word(int tile_idx,
                                       const tensor_desc_t& s0,
                                       const tensor_desc_t& s1,
                                       const tensor_desc_t& s2,
                                       const tensor_desc_t& s3,
                                       const act_vec_t row_buf[MAX_FM_W],
                                       u16_t local_row,
                                       u16_t ow,
                                       act_vec_t& out_word,
                                       u16_t& c,
                                       u8_t& qparam_block) {
#pragma HLS INLINE off
  if (tile_idx == 0) {
    c = static_cast<u16_t>(0);
    qparam_block = static_cast<u8_t>(0);
    return ppu_block5_l2_tile0_to_word(s0, s1, s2, local_row, ow, out_word);
  }
  if (tile_idx == 1) {
    c = static_cast<u16_t>(TM);
    qparam_block = static_cast<u8_t>(1);
    return ppu_block5_l2_tile1_to_word(s2, s3, row_buf, local_row, ow, out_word);
  }
  return false;
}

static bool ppu_block5_l3_tile_to_word(int tile_idx,
                                       const tensor_desc_t& s0,
                                       const tensor_desc_t& s1,
                                       const tensor_desc_t& s2,
                                       const tensor_desc_t& s3,
                                       const act_vec_t row_buf[MAX_FM_W],
                                       u16_t local_row,
                                       u16_t ow,
                                       act_vec_t& out_word,
                                       u16_t& c,
                                       u8_t& qparam_block) {
#pragma HLS INLINE off
  if (tile_idx == 0) {
    c = static_cast<u16_t>(0);
    qparam_block = static_cast<u8_t>(0);
    return ppu_block5_l3_tile0_to_word(s0, s1, local_row, ow, out_word);
  }
  if (tile_idx == 1) {
    c = static_cast<u16_t>(TM);
    qparam_block = static_cast<u8_t>(1);
    return ppu_block5_l3_tile1_to_word(s1, s2, local_row, ow, out_word);
  }
  if (tile_idx == 2) {
    c = static_cast<u16_t>(TM * 2);
    qparam_block = static_cast<u8_t>(2);
    return ppu_block5_l3_tile2_to_word(s2, s3, local_row, ow, out_word);
  }
  if (tile_idx == 3) {
    c = static_cast<u16_t>(TM * 3);
    qparam_block = static_cast<u8_t>(3);
    return ppu_block5_l3_tile3_to_word(s3, row_buf, local_row, ow, out_word);
  }
  return false;
}

static bool ppu_finalize_block5_emit_word(const block5_sched_desc_t& sched,
                                          const tensor_desc_t& residual,
                                          const tensor_desc_t& final_dst,
                                          bool has_residual,
                                          const add_q_t& residual_add_qparam,
                                          u8_t act_type,
                                          u16_t abs_row,
                                          u16_t ow,
                                          u16_t c,
                                          u8_t qparam_block,
                                          const act_vec_t& cat_word) {
#pragma HLS INLINE off
  act_vec_t residual_word = 0;
  act_vec_t out_word = 0;
  aff_q_t aff_qparam;
#pragma HLS ARRAY_PARTITION variable=aff_qparam.mul complete dim=1
#pragma HLS ARRAY_PARTITION variable=aff_qparam.bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=aff_qparam.shift complete dim=1
  if (!param_dma_get_affine_qparam(sched.affine_param_id, qparam_block, aff_qparam)) {
    return false;
  }
  if (has_residual &&
      !on_chip_memory_read_aligned_full_tile(residual,
                                             static_cast<i32_t>(abs_row),
                                             static_cast<i32_t>(ow),
                                             c,
                                             residual_word)) {
    return false;
  }
  if (has_residual) {
    ppu_apply_block_add_affine_word(cat_word,
                                    residual_word,
                                    residual_add_qparam,
                                    aff_qparam,
                                    act_type,
                                    out_word);
  } else {
    ppu_apply_block_affine_word(cat_word, aff_qparam, act_type, out_word);
  }
  return conv_store_write_aligned_tile(final_dst, abs_row, ow, c, out_word);
}

static bool ppu_finalize_block5_l2_row(const block5_sched_desc_t& sched,
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
  if (!ppu_full_aligned_tile_write_plan_ok(final_dst, sched.valid_c)) {
    return false;
  }
  const int out_w_i = static_cast<int>(sched.out_w.to_uint());
  for (int ow_i = 0; ow_i < MAX_FM_W; ++ow_i) {
#pragma HLS PIPELINE off
    if (ow_i >= out_w_i) {
      break;
    }
    const u16_t ow = static_cast<u16_t>(ow_i);
    for (int tile_idx = 0; tile_idx < 2; ++tile_idx) {
#pragma HLS PIPELINE off
      act_vec_t cat_word = 0;
      u16_t c = 0;
      u8_t qparam_block = 0;
      if (!ppu_block5_l2_tile_to_word(tile_idx,
                                      s0,
                                      s1,
                                      s2,
                                      s3,
                                      row_buf,
                                      local_row,
                                      ow,
                                      cat_word,
                                      c,
                                      qparam_block) ||
          !ppu_finalize_block5_emit_word(sched,
                                         residual,
                                         final_dst,
                                         has_residual,
                                         residual_add_qparam,
                                         act_type,
                                         abs_row,
                                         ow,
                                         c,
                                         qparam_block,
                                         cat_word)) {
        return false;
      }
    }
  }
  return true;
}

static bool ppu_finalize_block5_l3_row(const block5_sched_desc_t& sched,
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
  if (!ppu_full_aligned_tile_write_plan_ok(final_dst, sched.valid_c)) {
    return false;
  }
  const int out_w_i = static_cast<int>(sched.out_w.to_uint());
  for (int ow_i = 0; ow_i < MAX_FM_W; ++ow_i) {
#pragma HLS PIPELINE off
    if (ow_i >= out_w_i) {
      break;
    }
    const u16_t ow = static_cast<u16_t>(ow_i);
    for (int tile_idx = 0; tile_idx < 4; ++tile_idx) {
#pragma HLS PIPELINE off
      act_vec_t cat_word = 0;
      u16_t c = 0;
      u8_t qparam_block = 0;
      if (!ppu_block5_l3_tile_to_word(tile_idx,
                                      s0,
                                      s1,
                                      s2,
                                      s3,
                                      row_buf,
                                      local_row,
                                      ow,
                                      cat_word,
                                      c,
                                      qparam_block) ||
          !ppu_finalize_block5_emit_word(sched,
                                         residual,
                                         final_dst,
                                         has_residual,
                                         residual_add_qparam,
                                         act_type,
                                         abs_row,
                                         ow,
                                         c,
                                         qparam_block,
                                         cat_word)) {
        return false;
      }
    }
  }
  return true;
}

#ifdef ESP_INT8_CSIM_DUMP_UPSAMPLE_INPUT
static void csim_dump_upsample_input_row(axi_vec_t* gmem_frame_out,
                                         u16_t out_row,
                                         const act_vec_t row_buf[MAX_FM_W]) {
  for (int ow_i = 0; ow_i < ENCODER_OUT_W; ++ow_i) {
    const act_vec_t packed = row_buf[ow_i];
    const unsigned base_byte =
        (out_row.to_uint() * static_cast<unsigned>(ENCODER_OUT_W) +
         static_cast<unsigned>(ow_i)) *
        static_cast<unsigned>(ENCODER_OUT_C);
    for (int lane = 0; lane < ENCODER_OUT_C; ++lane) {
      const unsigned byte_idx = base_byte + static_cast<unsigned>(lane);
      const unsigned word_idx = byte_idx / static_cast<unsigned>(AXI_WORD_BYTES);
      const unsigned byte_lane = byte_idx % static_cast<unsigned>(AXI_WORD_BYTES);
      axi_vec_t word = gmem_frame_out[word_idx];
      word.range(byte_lane * 8 + 7, byte_lane * 8) =
          packed.range(lane * 8 + 7, lane * 8);
      gmem_frame_out[word_idx] = word;
    }
  }
}
#endif

#if !defined(__SYNTHESIS__) && defined(ESP_INT8_CSIM_DUMP_LOWRES_LOGITS_SIDE)
static unsigned s_csim_lowres_logits_rows = 0;

static bool csim_dump_lowres_logits_row(u16_t out_row,
                                        const act_vec_t row_buf[MAX_FM_W]) {
  const unsigned row = out_row.to_uint();
  const bool first_row = row == 0U || s_csim_lowres_logits_rows == 0U;
  std::ofstream out("csim_u72_lowres_logits.bin",
                    std::ios::binary | (first_row ? std::ios::trunc : std::ios::app));
  if (!out) {
    std::printf("[CSIM-DUMP] failed to open csim_u72_lowres_logits.bin\n");
    return false;
  }

  std::uint8_t bytes[ENCODER_OUT_C];
  for (unsigned ow = 0; ow < static_cast<unsigned>(ENCODER_OUT_W); ++ow) {
    const act_vec_t packed = row_buf[ow];
    for (unsigned lane = 0; lane < static_cast<unsigned>(ENCODER_OUT_C); ++lane) {
      bytes[lane] = static_cast<std::uint8_t>(packed.range(lane * 8 + 7, lane * 8).to_uint());
    }
    out.write(reinterpret_cast<const char*>(bytes),
              static_cast<std::streamsize>(ENCODER_OUT_C));
  }

  ++s_csim_lowres_logits_rows;
  if (row == static_cast<unsigned>(ENCODER_OUT_H - 1)) {
    std::printf("[CSIM-DUMP] wrote csim_u72_lowres_logits.bin rows=%u shape=%ux%ux%u bytes=%u\n",
                s_csim_lowres_logits_rows,
                static_cast<unsigned>(ENCODER_OUT_H),
                static_cast<unsigned>(ENCODER_OUT_W),
                static_cast<unsigned>(ENCODER_OUT_C),
                static_cast<unsigned>(ENCODER_LOGITS_BYTES));
  }
  return true;
}
#endif

static bool ppu_consume_upsample_out(axi_vec_t* gmem_frame_out,
                                     u16_t out_row,
                                     act_vec_t row_buf[MAX_FM_W]) {
#pragma HLS INLINE off
#if !defined(__SYNTHESIS__) && defined(ESP_INT8_CSIM_DUMP_LOWRES_LOGITS_SIDE)
  if (!csim_dump_lowres_logits_row(out_row, row_buf)) {
    return false;
  }
#endif
#ifdef ESP_INT8_CSIM_DUMP_UPSAMPLE_INPUT
  csim_dump_upsample_input_row(gmem_frame_out, out_row, row_buf);
#else
  upsample_fused_consume_logits_row(gmem_frame_out, out_row, row_buf);
#endif
  return true;
}

bool ppu_consume_conv_row(const row_consumer_desc_t& consumer,
                          const tensor_desc_t& dst,
                          const tensor_desc_t& add_other,
                          bool has_add_other,
                          const add_q_t& add_qparam,
                          const conv_cfg_t& cfg,
                          axi_vec_t* gmem_frame_out,
                          u16_t out_row,
                          act_vec_t row_buf[MAX_FM_W]) {
#pragma HLS INLINE off
  const unsigned mode = consumer.mode.to_uint();
  const u8_t valid_c =
      (consumer.valid_c.to_uint() == 0U)
          ? static_cast<u8_t>(cfg.out_c.to_uint())
          : static_cast<u8_t>(consumer.valid_c.to_uint());

  if (mode == static_cast<unsigned>(ROW_CONSUMER_UPSAMPLE_OUT)) {
    return ppu_consume_upsample_out(gmem_frame_out, out_row, row_buf);
  }

  if (mode != static_cast<unsigned>(ROW_CONSUMER_NONE) &&
      mode != static_cast<unsigned>(ROW_CONSUMER_CAT_AFFINE_STORE)) {
    return false;
  }

  bool ok = true;
  if (has_add_other && mode == static_cast<unsigned>(ROW_CONSUMER_NONE)) {
    ppu_add_other_row_to_buffer(add_other, out_row, cfg, valid_c, add_qparam, row_buf, ok);
    if (!ok) {
      return false;
    }
  }

  if (mode == static_cast<unsigned>(ROW_CONSUMER_CAT_AFFINE_STORE)) {
    if (!has_add_other) {
      return false;
    }
    const unsigned prefix = cfg.out_c.to_uint();
    const unsigned final_c = valid_c.to_uint();
    if (prefix >= final_c || final_c > static_cast<unsigned>(TM)) {
      return false;
    }
    ppu_cat_other_row_to_buffer(add_other,
                                out_row,
                                cfg,
                                static_cast<u8_t>(prefix),
                                static_cast<u8_t>(final_c - prefix),
                                row_buf,
                                ok);
    if (!ok) {
      return false;
    }
    ppu_apply_row_affine(consumer.affine_param_id,
                         consumer.reserved1,
                         consumer.act_type,
                         cfg,
                         valid_c,
                         row_buf,
                         ok);
    if (!ok) {
      return false;
    }
  }

  if (consumer.reserved0.to_uint() == static_cast<unsigned>(STORE_LAYOUT_NONE)) {
    return false;
  }

  const unsigned layout = consumer.reserved0.to_uint();
  if (mode == static_cast<unsigned>(ROW_CONSUMER_NONE)) {
    const bool block5_scratch =
        has_add_other ||
        layout == static_cast<unsigned>(STORE_LAYOUT_COMPACT_C16) ||
        layout == static_cast<unsigned>(STORE_LAYOUT_COMPACT_C28);
    if (block5_scratch) {
      if (layout != static_cast<unsigned>(STORE_LAYOUT_COMPACT_C12) &&
          layout != static_cast<unsigned>(STORE_LAYOUT_COMPACT_C16) &&
          layout != static_cast<unsigned>(STORE_LAYOUT_COMPACT_C25) &&
          layout != static_cast<unsigned>(STORE_LAYOUT_COMPACT_C28)) {
        return false;
      }
      return ppu_store_block5_scratch_row(dst, out_row, consumer.reserved0, cfg, row_buf);
    }
    if (layout != static_cast<unsigned>(STORE_LAYOUT_COMPACT_C12) &&
        layout != static_cast<unsigned>(STORE_LAYOUT_COMPACT_C25)) {
      return false;
    }
    return ppu_store_compact_row(dst, out_row, consumer.reserved0, cfg, row_buf);
  }

  if (mode == static_cast<unsigned>(ROW_CONSUMER_CAT_AFFINE_STORE)) {
    if (layout != static_cast<unsigned>(STORE_LAYOUT_COMPACT_C19)) {
      return false;
    }
    return ppu_store_compact_row(dst, out_row, consumer.reserved0, cfg, row_buf);
  }

  return false;
}

bool ppu_consume_block5_final_row(const block5_sched_desc_t& sched,
                                  const tensor_desc_t& scratch0,
                                  const tensor_desc_t& scratch1,
                                  const tensor_desc_t& scratch2,
                                  const tensor_desc_t& scratch3,
                                  const tensor_desc_t& prev_branch,
                                  const tensor_desc_t& residual,
                                  const tensor_desc_t& final_dst,
                                  bool has_residual,
                                  const add_q_t& chain_add_qparam,
                                  const add_q_t& residual_add_qparam,
                                  const conv_cfg_t& cfg,
                                  u8_t act_type,
                                  u16_t local_row,
                                  u16_t abs_row,
                                  act_vec_t row_buf[MAX_FM_W]) {
#pragma HLS INLINE off
  bool ok = true;
  ppu_add_other_row_to_buffer(prev_branch,
                              local_row,
                              cfg,
                              static_cast<u8_t>(cfg.out_c.to_uint()),
                              chain_add_qparam,
                              row_buf,
                              ok);
  if (!ok) {
    return false;
  }
  const unsigned pattern = sched.pattern.to_uint();
  if (pattern == static_cast<unsigned>(BLOCK5_PATTERN_L2_C16_4C12)) {
    return ppu_finalize_block5_l2_row(sched,
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
  if (pattern == static_cast<unsigned>(BLOCK5_PATTERN_L3_C28_4C25)) {
    return ppu_finalize_block5_l3_row(sched,
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
  return false;
}

} // namespace esp_int8
