#include <cstdint>

#include "../include/npu_config.hpp"
#include "../include/npu_q.hpp"
#include "../include/npu_uop.hpp"

namespace esp_int8 {

void param_dma_init(const axi_vec_t* gmem_param);
bool param_dma_ready();
bool param_dma_get_pool_qparam(u8_t param_id, pool_q_t& qparam);
bool param_dma_get_conv_qparam(u8_t param_id, conv_q_t& qparam);
bool param_dma_get_weight_vec(u8_t param_id, u16_t oc, u16_t kt, const conv_cfg_t& cfg, wgt_vec_t& word);
bool param_dma_get_affine_qparam(u8_t param_id, u8_t block_id, aff_q_t& qparam);
bool param_dma_get_add_qparam(u8_t param_id, add_q_t& qparam);
void frame_dma_load(const axi_vec_t* gmem_frame_in);
bool avgpool_unit_checked(const tensor_desc_t& src,
                          const tensor_desc_t& dst,
                          const pool_q_t& qparam,
                          i8_t* fmbuf_base);
bool concat_writer(const tensor_desc_t& src,
                   const tensor_desc_t& dst,
                   u16_t c_offset,
                   u16_t valid_c);
bool on_chip_memory_read_packed_tile(const tensor_desc_t& desc,
                                     i32_t h,
                                     i32_t w,
                                     u16_t c_begin,
                                     u8_t valid_c,
                                     act_vec_t& packed);
bool on_chip_memory_read_aligned_full_tile(const tensor_desc_t& desc,
                                           i32_t h,
                                           i32_t w,
                                           u16_t c_begin,
                                           act_vec_t& packed);
bool on_chip_memory_write_aligned_full_tile(const tensor_desc_t& desc,
                                            u16_t h,
                                            u16_t w,
                                            u16_t c_begin,
                                            act_vec_t packed);
bool on_chip_memory_read_fmbuf_abs_word(u8_t bank_id,
                                        u32_t byte_offset,
                                        axi_vec_t& packed);
bool on_chip_memory_write_fmbuf_abs_word(u8_t bank_id,
                                         u32_t byte_offset,
                                         axi_vec_t packed);
bool on_chip_memory_read_pool2_abs_word(u32_t byte_offset,
                                        axi_vec_t& packed);
bool on_chip_memory_write_pool2_abs_word(u32_t byte_offset,
                                         axi_vec_t packed);
void window_generator_row(const tensor_desc_t& src_desc,
                          hls::stream<act_vec_t>& act_stream,
                          const conv_cfg_t& cfg,
                          u16_t out_row);
void systolic_array_core_row(hls::stream<act_vec_t>& act_stream,
                             hls::stream<wgt_vec_t>& wgt_stream,
                             hls::stream<psum_vec_t>& psum_stream,
                             const conv_cfg_t& cfg);
bool store_conv_output_row(const tensor_desc_t& dst,
                           u16_t out_row,
                           u16_t c_offset,
                           const conv_cfg_t& cfg,
                           act_vec_t row_buf[MAX_FM_W]);
void reset_scratch_state();
void select_scratch_region(const uop_t& uop, u16_t out_h);
bool resolve_tensor_read(u8_t tensor_id, tensor_desc_t& desc);
bool resolve_tensor_write(u8_t tensor_id, u16_t h, u16_t w, u16_t c, tensor_desc_t& desc);
bool alias_global_tensor_to_slice(u8_t tensor_id,
                                  const tensor_desc_t& base_desc,
                                  u16_t c_offset,
                                  u16_t c);
bool alias_scratch_tensor_to_slice(u8_t tensor_id,
                                   const tensor_desc_t& base_desc,
                                   u16_t c_offset,
                                   u16_t c);
void upsample_fused_begin();
void upsample_fused_consume_logits_row(axi_vec_t* gmem_frame_out,
                                       u16_t encoder_row,
                                       const act_vec_t row_buf[MAX_FM_W]);

static param_blob_header_t s_param_header;
static bool s_param_ready = false;

#ifndef __SYNTHESIS__
static unsigned s_csim_last_uop = 0;
static error_code_t s_csim_last_error = ERR_NONE;

unsigned csim_last_uop() {
  return s_csim_last_uop;
}

unsigned csim_last_error() {
  return static_cast<unsigned>(s_csim_last_error);
}
#endif

static u32_t axi_lane_u32(const axi_vec_t& word, int lane) {
#pragma HLS INLINE
  return word.range(lane * 32 + 31, lane * 32);
}

static bool is_aligned_section_offset(u32_t offset) {
#pragma HLS INLINE
  return (offset & (SECTION_ALIGNMENT_BYTES - 1)) == 0;
}

static std::uint32_t runtime_mode(std::uint32_t mode) {
#pragma HLS INLINE
  return mode & RUNTIME_MODE_MASK;
}

static u16_t ceil_div_u16(u16_t a, u16_t b) {
#pragma HLS INLINE
  return static_cast<u16_t>((a + b - 1) / b);
}

static u16_t effective_stride(const uop_t& uop) {
#pragma HLS INLINE
  return (uop.stride.to_uint() == 0U) ? static_cast<u16_t>(1) : static_cast<u16_t>(uop.stride);
}

static u16_t conv_out_dim(u16_t in_size, u16_t stride) {
#pragma HLS INLINE
  return ceil_div_u16(in_size, stride);
}

static u8_t tensor_lanes(u16_t remaining_c) {
#pragma HLS INLINE
  const unsigned rem = remaining_c.to_uint();
  return static_cast<u8_t>((rem > static_cast<unsigned>(TM)) ? TM : rem);
}

static bool is_full_tile_lanes(u8_t lanes) {
#pragma HLS INLINE
  return lanes.to_uint() == static_cast<unsigned>(TM);
}

static u16_t p6_desc_phys_c(const tensor_desc_t& desc) {
#pragma HLS INLINE
  return (desc.reserved0.to_uint() == 0U) ? desc.c : desc.reserved0;
}

static bool can_use_aligned_full_tile(const tensor_desc_t& desc, u16_t c_begin) {
#pragma HLS INLINE
  const unsigned phys_c = p6_desc_phys_c(desc).to_uint();
  const unsigned start_c = desc.reserved1.to_uint() + c_begin.to_uint();
  const unsigned base_start = desc.base_offset.to_uint() + start_c;
  return phys_c >= static_cast<unsigned>(AXI_WORD_BYTES) &&
         ((phys_c & static_cast<unsigned>(AXI_WORD_BYTES - 1)) == 0U) &&
         c_begin.to_uint() + static_cast<unsigned>(AXI_WORD_BYTES) <= desc.c.to_uint() &&
         ((base_start & static_cast<unsigned>(AXI_WORD_BYTES - 1)) == 0U);
}

static u32_t p6_tensor_byte_offset(const tensor_desc_t& desc, u16_t h, u16_t w, u16_t c_begin) {
#pragma HLS INLINE
  return static_cast<u32_t>((static_cast<u32_t>(h) * desc.w + w) * p6_desc_phys_c(desc) +
                            desc.reserved1 + c_begin);
}

static bool p6_read_aligned_abs_word(const tensor_desc_t& desc, u32_t byte_offset, axi_vec_t& word) {
#pragma HLS INLINE
  if (desc.bank_id.to_uint() == static_cast<unsigned>(BANK_BRAM_SCR1)) {
    return on_chip_memory_read_pool2_abs_word(byte_offset, word);
  }
  return on_chip_memory_read_fmbuf_abs_word(desc.bank_id, byte_offset, word);
}

static bool p6_write_aligned_abs_word(const tensor_desc_t& desc, u32_t byte_offset, axi_vec_t word) {
#pragma HLS INLINE
  if (desc.bank_id.to_uint() == static_cast<unsigned>(BANK_BRAM_SCR1)) {
    return on_chip_memory_write_pool2_abs_word(byte_offset, word);
  }
  return on_chip_memory_write_fmbuf_abs_word(desc.bank_id, byte_offset, word);
}

static axi_vec_t p6_low_byte_mask(unsigned byte_count) {
#pragma HLS INLINE
  if (byte_count >= static_cast<unsigned>(AXI_WORD_BYTES)) {
    return ~static_cast<axi_vec_t>(0);
  }
  return static_cast<axi_vec_t>((static_cast<axi_vec_t>(1) << (byte_count * 8U)) - 1U);
}

static bool p6_write_tensor_slice_narrow(const tensor_desc_t& desc,
                                         u16_t h,
                                         u16_t w,
                                         u16_t c_begin,
                                         u8_t valid_c,
                                         act_vec_t packed) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off
  const unsigned lanes = valid_c.to_uint();
  if (lanes == 0U || lanes > static_cast<unsigned>(TM) ||
      h >= desc.h || w >= desc.w ||
      c_begin.to_uint() + lanes > desc.c.to_uint()) {
    return false;
  }

  const u32_t start_offset = desc.base_offset + p6_tensor_byte_offset(desc, h, w, c_begin);
  const u32_t word0_offset = start_offset & static_cast<u32_t>(~(AXI_WORD_BYTES - 1));
  const unsigned byte0 = start_offset.to_uint() & static_cast<unsigned>(AXI_WORD_BYTES - 1);

  if (byte0 == 0U && lanes == static_cast<unsigned>(AXI_WORD_BYTES)) {
    return p6_write_aligned_abs_word(desc, word0_offset, packed);
  }

  axi_vec_t word0 = 0;
  if (!p6_read_aligned_abs_word(desc, word0_offset, word0)) {
    return false;
  }
  const unsigned first_count =
      ((byte0 + lanes) <= static_cast<unsigned>(AXI_WORD_BYTES))
          ? lanes
          : (static_cast<unsigned>(AXI_WORD_BYTES) - byte0);
  const axi_vec_t mask0 = static_cast<axi_vec_t>(p6_low_byte_mask(first_count) << (byte0 * 8U));
  const axi_vec_t shifted0 = static_cast<axi_vec_t>(packed << (byte0 * 8U));
  word0 = static_cast<axi_vec_t>((word0 & ~mask0) | (shifted0 & mask0));
  if (!p6_write_aligned_abs_word(desc, word0_offset, word0)) {
    return false;
  }

  if (first_count < lanes) {
    const u32_t word1_offset = word0_offset + static_cast<u32_t>(AXI_WORD_BYTES);
    axi_vec_t word1 = 0;
    if (!p6_read_aligned_abs_word(desc, word1_offset, word1)) {
      return false;
    }
    const unsigned second_count = lanes - first_count;
    const axi_vec_t mask1 = p6_low_byte_mask(second_count);
    const axi_vec_t shifted1 = static_cast<axi_vec_t>(packed >> (first_count * 8U));
    word1 = static_cast<axi_vec_t>((word1 & ~mask1) | (shifted1 & mask1));
    if (!p6_write_aligned_abs_word(desc, word1_offset, word1)) {
      return false;
    }
  }
  return true;
}

static bool same_tensor_shape(const tensor_desc_t& a, const tensor_desc_t& b) {
#pragma HLS INLINE
  return a.h.to_uint() == b.h.to_uint() &&
         a.w.to_uint() == b.w.to_uint() &&
         a.c.to_uint() == b.c.to_uint();
}

static bool alias_tensor_to_slice(u8_t tensor_id,
                                  const tensor_desc_t& base_desc,
                                  u16_t c_offset,
                                  u16_t c) {
#pragma HLS INLINE
  if (tensor_is_scratch(tensor_id)) {
    return alias_scratch_tensor_to_slice(tensor_id, base_desc, c_offset, c);
  }
  if (tensor_is_global(tensor_id)) {
    return alias_global_tensor_to_slice(tensor_id, base_desc, c_offset, c);
  }
  return false;
}

static conv_cfg_t conv_cfg_from_uop(const uop_t& uop) {
#pragma HLS INLINE
  conv_cfg_t cfg;
  cfg.in_h = uop.in_h;
  cfg.in_w = uop.in_w;
  cfg.in_c = uop.in_c;
  cfg.out_c = uop.out_c;
  cfg.kernel = (uop.kernel.to_uint() == 1U) ? ap_uint<2>(1) : ap_uint<2>(3);
  cfg.stride = (uop.stride.to_uint() == 0U) ? ap_uint<2>(1) : ap_uint<2>(uop.stride);
  cfg.dilation = (uop.dilation.to_uint() == 0U) ? ap_uint<5>(1) : ap_uint<5>(uop.dilation);
  cfg.bias_en = uop_has_bias(uop) ? ap_uint<1>(1) : ap_uint<1>(0);
  return cfg;
}

static u16_t conv_effective_stride(const conv_cfg_t& cfg) {
#pragma HLS INLINE
  return (cfg.stride == 0) ? static_cast<u16_t>(1) : static_cast<u16_t>(cfg.stride);
}

static u16_t conv_effective_kernel(const conv_cfg_t& cfg) {
#pragma HLS INLINE
  return (cfg.kernel == 1) ? static_cast<u16_t>(1) : static_cast<u16_t>(3);
}

static u16_t conv_kernel_flat(const conv_cfg_t& cfg) {
#pragma HLS INLINE
  const u16_t kernel = conv_effective_kernel(cfg);
  return static_cast<u16_t>(cfg.in_c * kernel * kernel);
}

static void set_act_vec_i8_dynamic(act_vec_t& word, int lane, i8_t value) {
#pragma HLS INLINE
  u8_t raw = 0;
  raw.range(7, 0) = value.range(7, 0);
  const act_vec_t widened = static_cast<act_vec_t>(raw);
  word |= static_cast<act_vec_t>(widened << (lane * 8));
}

static i8_t get_act_vec_i8_dynamic(const act_vec_t& word, int lane) {
#pragma HLS INLINE
  const u8_t raw = word.range(lane * 8 + 7, lane * 8);
  i8_t value = 0;
  value.range(7, 0) = raw;
  return value;
}

static i32_t get_psum_i32(const psum_vec_t& word, int lane) {
#pragma HLS INLINE
  i32_t value;
  value.range(31, 0) = word.range(lane * 32 + 31, lane * 32);
  return value;
}

static void feed_cached_weights(hls::stream<wgt_vec_t>& wgt_stream,
                                const wgt_vec_t cache[],
                                int count) {
#pragma HLS INLINE off
  for (int i = 0; i < count; ++i) {
#pragma HLS PIPELINE II=1
    wgt_stream.write(cache[i]);
  }
}

static void post_process_row_to_buffer(hls::stream<psum_vec_t>& psum_stream,
                                        act_vec_t row_buf[MAX_FM_W],
                                        const conv_cfg_t& cfg,
                                        const uop_t& uop,
                                        const conv_q_t& qparam) {
#pragma HLS INLINE off
  const u16_t stride = conv_effective_stride(cfg);
  const u16_t out_w = conv_out_dim(cfg.in_w, stride);
  const int out_w_i = static_cast<int>(out_w.to_uint());
  const int out_c_i = static_cast<int>(cfg.out_c.to_uint());

  for (int ow_i = 0; ow_i < MAX_FM_W; ++ow_i) {
    if (ow_i >= out_w_i) {
      break;
    }
    act_vec_t packed = 0;
    const psum_vec_t psum_word = psum_stream.read();
    for (int tm = 0; tm < TM; ++tm) {
#pragma HLS UNROLL factor=4
      if (tm < out_c_i) {
        const i32_t psum = get_psum_i32(psum_word, tm);
        const i8_t out = requant_i32_to_i8(psum,
                                           qparam.bias[tm],
                                           qparam.mult[tm],
                                           qparam.shift[tm],
                                           uop.act_type);
        set_act_vec_i8_dynamic(packed, tm, out);
      }
    }
    row_buf[ow_i] = packed;
  }
}

static void execute_conv_stream_row_region(const tensor_desc_t& src,
                                           const conv_cfg_t& cfg,
                                           const uop_t& uop,
                                           const conv_q_t& qparam,
                                           u16_t out_row,
                                           act_vec_t row_buf[MAX_FM_W],
                                           const wgt_vec_t cached_wgts[],
                                           int wgt_count) {
#pragma HLS INLINE off
  hls::stream<act_vec_t> act_stream;
  hls::stream<wgt_vec_t> wgt_stream;
  hls::stream<psum_vec_t> psum_stream;
#pragma HLS STREAM variable=act_stream depth=64
#pragma HLS STREAM variable=wgt_stream depth=1200
#pragma HLS STREAM variable=psum_stream depth=16
  // Keep wide dataflow FIFOs out of CLB LUTRAM; route congestion is the current implementation limiter.
#pragma HLS BIND_STORAGE variable=act_stream type=fifo impl=bram
#pragma HLS BIND_STORAGE variable=wgt_stream type=fifo impl=bram
#pragma HLS BIND_STORAGE variable=psum_stream type=fifo impl=bram
#pragma HLS DATAFLOW
  window_generator_row(src, act_stream, cfg, out_row);
  feed_cached_weights(wgt_stream, cached_wgts, wgt_count);
  systolic_array_core_row(act_stream, wgt_stream, psum_stream, cfg);
  post_process_row_to_buffer(psum_stream, row_buf, cfg, uop, qparam);
}

static void execute_conv_stream_datapath(const tensor_desc_t& src,
                                         const tensor_desc_t& dst,
                                         const conv_cfg_t& cfg,
                                         const uop_t& uop,
                                         const conv_q_t& qparam,
                                         axi_vec_t* gmem_frame_out,
                                         bool emit_fullres_mask,
                                         bool& ok_out) {
#pragma HLS INLINE off
  act_vec_t row_buf[MAX_FM_W];
#pragma HLS BIND_STORAGE variable=row_buf type=ram_2p impl=bram

  const u16_t stride = conv_effective_stride(cfg);
  const u16_t out_h = conv_out_dim(cfg.in_h, stride);
  const u16_t k_total = conv_kernel_flat(cfg);
  const u16_t k_tiles = ceil_div_u16(k_total, static_cast<u16_t>(TK));
  const int out_h_i = static_cast<int>(out_h.to_uint());
  const int k_tiles_i = static_cast<int>(k_tiles.to_uint());
  bool write_ok = true;

  const int wgt_count = k_tiles_i * TM;
  wgt_vec_t cached_wgts[MAX_K_TILE_COUNT * TM];
#pragma HLS BIND_STORAGE variable=cached_wgts type=ram_1p impl=bram
  {
    int wi = 0;
    for (int kt = 0; kt < MAX_K_TILE_COUNT; ++kt) {
      if (kt >= k_tiles_i) break;
      const u16_t k_tile = static_cast<u16_t>(kt);
      for (int tm = 0; tm < TM; ++tm) {
#pragma HLS PIPELINE off
        const u16_t oc = static_cast<u16_t>(tm);
        wgt_vec_t word = 0;
        param_dma_get_weight_vec(uop.param_id, oc, k_tile, cfg, word);
        cached_wgts[wi++] = word;
      }
    }
  }

  if (emit_fullres_mask) {
    upsample_fused_begin();
  }

  for (int oh_i = 0; oh_i < MAX_FM_H; ++oh_i) {
    if (oh_i >= out_h_i) {
      break;
    }
    const u16_t oh = static_cast<u16_t>(oh_i);
    execute_conv_stream_row_region(src,
                                   cfg,
                                   uop,
                                   qparam,
                                   oh,
                                   row_buf,
                                   cached_wgts,
                                   wgt_count);
    if (emit_fullres_mask) {
      upsample_fused_consume_logits_row(gmem_frame_out, oh, row_buf);
    } else {
      if (!store_conv_output_row(dst, oh, uop.c_offset, cfg, row_buf)) {
        write_ok = false;
      }
    }
  }
  ok_out = write_ok;
}

static error_code_t run_p6_pool_op(const uop_t& uop) {
#pragma HLS INLINE off
  tensor_desc_t src;
  tensor_desc_t dst;
  pool_q_t qparam;
  const u16_t stride = effective_stride(uop);
  const u16_t out_h = conv_out_dim(uop.in_h, stride);
  const u16_t out_w = conv_out_dim(uop.in_w, stride);

  select_scratch_region(uop, out_h);
  if (!resolve_tensor_read(uop.src0_tensor, src) ||
      !resolve_tensor_write(uop.dst_tensor, out_h, out_w, uop.out_c, dst)) {
    return ERR_TENSOR_DESC_RANGE;
  }
  if (!param_dma_get_pool_qparam(uop.param_id, qparam)) {
    return ERR_PARAM_DESC_RANGE;
  }
  if (!avgpool_unit_checked(src, dst, qparam, 0)) {
    return ERR_BANK_OVERFLOW;
  }
  return ERR_NONE;
}

static error_code_t run_p6_conv_op(const uop_t& uop, axi_vec_t* gmem_frame_out) {
#pragma HLS INLINE off
  tensor_desc_t src;
  tensor_desc_t dst;
  conv_q_t qparam;
#pragma HLS ARRAY_PARTITION variable=qparam.bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=qparam.mult complete dim=1
#pragma HLS ARRAY_PARTITION variable=qparam.shift complete dim=1
  const conv_cfg_t cfg = conv_cfg_from_uop(uop);
  const u16_t stride = effective_stride(uop);
  const u16_t out_h = conv_out_dim(uop.in_h, stride);
  const u16_t out_w = conv_out_dim(uop.in_w, stride);

  select_scratch_region(uop, out_h);
  if (!resolve_tensor_read(uop.src0_tensor, src)) {
    return ERR_TENSOR_DESC_RANGE;
  }
  if (!param_dma_get_conv_qparam(uop.param_id, qparam)) {
    return ERR_PARAM_DESC_RANGE;
  }
  // The exporter emits wide tensors as multiple <=TM conv/store chunks. Keeping
  // this contract avoids activation-stream rebroadcast and preserves one-pass
  // producer/consumer matching in the stream datapath.
  if (cfg.out_c.to_uint() > static_cast<unsigned>(TM)) {
    return ERR_UNSUPPORTED_OPCODE;
  }

  const bool emit_fullres_mask = uop.dst_tensor.to_uint() == static_cast<unsigned>(TID_OUT);
  if (emit_fullres_mask) {
    if (cfg.out_c.to_uint() != static_cast<unsigned>(ENCODER_OUT_C) ||
        out_h.to_uint() != static_cast<unsigned>(ENCODER_OUT_H) ||
        out_w.to_uint() != static_cast<unsigned>(ENCODER_OUT_W)) {
      return ERR_UNSUPPORTED_OPCODE;
    }
    dst = tensor_desc_t();
  } else {
    if (!resolve_tensor_write(uop.dst_tensor, out_h, out_w, uop.out_c, dst)) {
      return ERR_TENSOR_DESC_RANGE;
    }
    if (tensor_is_global(uop.dst_tensor) &&
        dst.c.to_uint() < uop.c_offset.to_uint() + uop.out_c.to_uint()) {
      return ERR_TENSOR_DESC_RANGE;
    }
  }

  bool conv_ok = false;
  execute_conv_stream_datapath(src,
                               dst,
                               cfg,
                               uop,
                               qparam,
                               gmem_frame_out,
                               emit_fullres_mask,
                               conv_ok);
  if (!conv_ok) {
    return ERR_BANK_OVERFLOW;
  }

  return ERR_NONE;
}

static error_code_t run_p6_affine_op(const uop_t& uop) {
#pragma HLS INLINE off
  tensor_desc_t src;
  tensor_desc_t dst;

  select_scratch_region(uop, uop.in_h);
  if (!resolve_tensor_read(uop.src0_tensor, src) ||
      !resolve_tensor_write(uop.dst_tensor, src.h, src.w, src.c, dst)) {
    return ERR_TENSOR_DESC_RANGE;
  }
  if (!same_tensor_shape(src, dst)) {
    return ERR_TENSOR_DESC_RANGE;
  }

  const u16_t valid_c = (uop.valid_c.to_uint() == 0U) ? src.c : uop.valid_c;
  if (valid_c.to_uint() > src.c.to_uint() || valid_c.to_uint() > dst.c.to_uint()) {
    return ERR_TENSOR_DESC_RANGE;
  }

  const int h_count = static_cast<int>(src.h.to_uint());
  const int w_count = static_cast<int>(src.w.to_uint());
  const int c_blocks = static_cast<int>((valid_c.to_uint() + static_cast<unsigned>(TM) - 1U) /
                                        static_cast<unsigned>(TM));

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
        const u16_t remaining = static_cast<u16_t>(valid_c - c);
        const u8_t lanes = tensor_lanes(remaining);
        const u8_t block_id = static_cast<u8_t>(c.to_uint() / static_cast<unsigned>(TM));
        aff_q_t qparam;
        act_vec_t in_packed = 0;
        act_vec_t out_packed = 0;
        const bool full_read = is_full_tile_lanes(lanes) && can_use_aligned_full_tile(src, c);
        const bool full_write = is_full_tile_lanes(lanes) && can_use_aligned_full_tile(dst, c);
        if (!param_dma_get_affine_qparam(uop.param_id, block_id, qparam)) {
          return ERR_PARAM_DESC_RANGE;
        }
        const bool read_ok = full_read
                                 ? on_chip_memory_read_aligned_full_tile(src,
                                                                         static_cast<i32_t>(h),
                                                                         static_cast<i32_t>(w),
                                                                         c,
                                                                         in_packed)
                                 : on_chip_memory_read_packed_tile(src,
                                                                   static_cast<i32_t>(h),
                                                                   static_cast<i32_t>(w),
                                                                   c,
                                                                   lanes,
                                                                   in_packed);
        if (!read_ok) {
          return ERR_BANK_OVERFLOW;
        }
        for (int lane = 0; lane < TM; ++lane) {
#pragma HLS UNROLL
          if (static_cast<unsigned>(lane) < lanes.to_uint()) {
            const i8_t in_value = get_act_vec_i8_dynamic(in_packed, lane);
            const i8_t out_value =
                affine_i8_to_i8(in_value, qparam.mul[lane], qparam.bias[lane], qparam.shift[lane], uop.act_type);
            set_act_vec_i8_dynamic(out_packed, lane, out_value);
          }
        }
        const bool write_ok = full_write
                                  ? on_chip_memory_write_aligned_full_tile(dst, h, w, c, out_packed)
                                  : p6_write_tensor_slice_narrow(dst, h, w, c, lanes, out_packed);
        if (!write_ok) {
          return ERR_BANK_OVERFLOW;
        }
      }
    }
  }

  return ERR_NONE;
}

static error_code_t run_p6_add_op(const uop_t& uop) {
#pragma HLS INLINE off
  tensor_desc_t src0;
  tensor_desc_t src1;
  tensor_desc_t dst;
  add_q_t qparam;

  select_scratch_region(uop, uop.in_h);
  if (!resolve_tensor_read(uop.src0_tensor, src0) ||
      !resolve_tensor_read(uop.src1_tensor, src1) ||
      !resolve_tensor_write(uop.dst_tensor, src0.h, src0.w, src0.c, dst)) {
    return ERR_TENSOR_DESC_RANGE;
  }
  if (!same_tensor_shape(src0, src1) || !same_tensor_shape(src0, dst)) {
    return ERR_TENSOR_DESC_RANGE;
  }
  if (!param_dma_get_add_qparam(uop.param_id, qparam)) {
    return ERR_PARAM_DESC_RANGE;
  }

  const u16_t valid_c = (uop.valid_c.to_uint() == 0U) ? src0.c : uop.valid_c;
  if (valid_c.to_uint() > src0.c.to_uint()) {
    return ERR_TENSOR_DESC_RANGE;
  }

  const int h_count = static_cast<int>(src0.h.to_uint());
  const int w_count = static_cast<int>(src0.w.to_uint());
  const int c_blocks = static_cast<int>((valid_c.to_uint() + static_cast<unsigned>(TM) - 1U) /
                                        static_cast<unsigned>(TM));

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
        const u16_t remaining = static_cast<u16_t>(valid_c - c);
        const u8_t lanes = tensor_lanes(remaining);
        act_vec_t a_packed = 0;
        act_vec_t b_packed = 0;
        act_vec_t out_packed = 0;
        const bool full_tile = is_full_tile_lanes(lanes);
        const bool full_read_a = full_tile && can_use_aligned_full_tile(src0, c);
        const bool full_read_b = full_tile && can_use_aligned_full_tile(src1, c);
        const bool full_write = full_tile && can_use_aligned_full_tile(dst, c);
        const bool read_a_ok = full_read_a
                                   ? on_chip_memory_read_aligned_full_tile(src0,
                                                                           static_cast<i32_t>(h),
                                                                           static_cast<i32_t>(w),
                                                                           c,
                                                                           a_packed)
                                   : on_chip_memory_read_packed_tile(src0,
                                                                     static_cast<i32_t>(h),
                                                                     static_cast<i32_t>(w),
                                                                     c,
                                                                     lanes,
                                                                     a_packed);
        const bool read_b_ok = full_read_b
                                   ? on_chip_memory_read_aligned_full_tile(src1,
                                                                           static_cast<i32_t>(h),
                                                                           static_cast<i32_t>(w),
                                                                           c,
                                                                           b_packed)
                                   : on_chip_memory_read_packed_tile(src1,
                                                                     static_cast<i32_t>(h),
                                                                     static_cast<i32_t>(w),
                                                                     c,
                                                                     lanes,
                                                                     b_packed);
        if (!read_a_ok || !read_b_ok) {
          return ERR_BANK_OVERFLOW;
        }
        for (int lane = 0; lane < TM; ++lane) {
#pragma HLS UNROLL
          if (static_cast<unsigned>(lane) < lanes.to_uint()) {
            const i8_t a_value = get_act_vec_i8_dynamic(a_packed, lane);
            const i8_t b_value = get_act_vec_i8_dynamic(b_packed, lane);
            const i8_t out_value = add_i8(a_value, b_value, qparam);
            set_act_vec_i8_dynamic(out_packed, lane, out_value);
          }
        }
        const bool write_ok = full_write
                                  ? on_chip_memory_write_aligned_full_tile(dst, h, w, c, out_packed)
                                  : p6_write_tensor_slice_narrow(dst, h, w, c, lanes, out_packed);
        if (!write_ok) {
          return ERR_BANK_OVERFLOW;
        }
      }
    }
  }

  return ERR_NONE;
}

static error_code_t run_p6_store_op(const uop_t& uop) {
#pragma HLS INLINE off
  tensor_desc_t src;
  tensor_desc_t dst;

  if (uop.dst_tensor.to_uint() == static_cast<unsigned>(TID_INVALID)) {
    return ERR_NONE;
  }

  select_scratch_region(uop, uop.in_h);
  if (!resolve_tensor_read(uop.src0_tensor, src) ||
      !resolve_tensor_write(uop.dst_tensor, src.h, src.w, static_cast<u16_t>(uop.out_c), dst)) {
    return ERR_TENSOR_DESC_RANGE;
  }

  const u16_t valid_c = (uop.valid_c.to_uint() == 0U) ? src.c : uop.valid_c;
  if (!concat_writer(src, dst, uop.c_offset, valid_c)) {
    return ERR_BANK_OVERFLOW;
  }
  return ERR_NONE;
}

static void load_param_header(const axi_vec_t* gmem_param, param_blob_header_t& header) {
#pragma HLS INLINE
  u32_t raw[32];
#pragma HLS ARRAY_PARTITION variable=raw complete dim=1

  for (int word_idx = 0; word_idx < PARAM_HEADER_AXI_WORDS; ++word_idx) {
#pragma HLS PIPELINE off
    const axi_vec_t word = gmem_param[word_idx];
    for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
      raw[word_idx * 8 + lane] = axi_lane_u32(word, lane);
    }
  }

  header.magic = raw[0];
  header.version = raw[1];
  header.tensor_desc_count = raw[2];
  header.scale_desc_count = raw[3];
  header.conv_desc_count = raw[4];
  header.affine_desc_count = raw[5];
  header.add_desc_count = raw[6];
  header.pool_desc_count = raw[7];
  header.uop_count = raw[8];
  header.reserved0 = raw[9];
  header.tensor_desc_offset = raw[10];
  header.scale_desc_offset = raw[11];
  header.conv_desc_offset = raw[12];
  header.affine_desc_offset = raw[13];
  header.add_desc_offset = raw[14];
  header.pool_desc_offset = raw[15];
  header.uop_offset = raw[16];
  header.weight_data_offset = raw[17];
  header.conv_qparam_offset = raw[18];
  header.affine_qparam_offset = raw[19];
  header.add_qparam_offset = raw[20];
  header.pool_qparam_offset = raw[21];

  for (int i = 0; i < 10; ++i) {
#pragma HLS UNROLL
    header.reserved1[i] = raw[22 + i];
  }
}

static error_code_t validate_param_header(const param_blob_header_t& header) {
#pragma HLS INLINE
  if (header.magic != PARAM_BLOB_MAGIC || header.version != PARAM_BLOB_VERSION) {
    return ERR_BAD_BLOB;
  }
  if (header.uop_count.to_uint() != static_cast<unsigned>(UOP_COUNT_ENCODER)) {
    return ERR_UOP_DECODE;
  }
  if (header.tensor_desc_count > MAX_TENSOR_DESC_COUNT ||
      header.scale_desc_count > SCALE_DESC_COUNT_MAX) {
    return ERR_TENSOR_DESC_RANGE;
  }
  if (header.conv_desc_count > MAX_CONV_PARAM_DESC_COUNT ||
      header.affine_desc_count > MAX_AFFINE_PARAM_DESC_COUNT ||
      header.add_desc_count > MAX_ADD_PARAM_DESC_COUNT ||
      header.pool_desc_count > MAX_POOL_PARAM_DESC_COUNT) {
    return ERR_PARAM_DESC_RANGE;
  }
  if (!is_aligned_section_offset(header.tensor_desc_offset) ||
      !is_aligned_section_offset(header.scale_desc_offset) ||
      !is_aligned_section_offset(header.conv_desc_offset) ||
      !is_aligned_section_offset(header.affine_desc_offset) ||
      !is_aligned_section_offset(header.add_desc_offset) ||
      !is_aligned_section_offset(header.pool_desc_offset) ||
      !is_aligned_section_offset(header.uop_offset) ||
      !is_aligned_section_offset(header.weight_data_offset) ||
      !is_aligned_section_offset(header.conv_qparam_offset) ||
      !is_aligned_section_offset(header.affine_qparam_offset) ||
      !is_aligned_section_offset(header.add_qparam_offset) ||
      !is_aligned_section_offset(header.pool_qparam_offset)) {
    return ERR_PARAM_DESC_RANGE;
  }

  return ERR_NONE;
}

static void core_mode_init(const axi_vec_t* gmem_param) {
#pragma HLS INLINE off
  reset_scratch_state();
  s_param_ready = false;

  load_param_header(gmem_param, s_param_header);
  const error_code_t err = validate_param_header(s_param_header);
  if (err != ERR_NONE) {
    return;
  }

  param_dma_init(gmem_param);
  if (!param_dma_ready()) {
    return;
  }
  s_param_ready = true;
}

static void fill_static_uop(uop_t& uop,
                            unsigned opcode,
                            unsigned flags,
                            unsigned src0,
                            unsigned src1,
                            unsigned dst,
                            unsigned param_id,
                            unsigned act_type,
                            unsigned in_h,
                            unsigned in_w,
                            unsigned in_c,
                            unsigned out_c,
                            unsigned kernel,
                            unsigned stride,
                            unsigned dilation,
                            unsigned padding,
                            unsigned c_offset,
                            unsigned valid_c,
                            unsigned qparam_id) {
#pragma HLS INLINE
  uop.opcode = static_cast<u8_t>(opcode);
  uop.flags = static_cast<u8_t>(flags);
  uop.src0_tensor = static_cast<u8_t>(src0);
  uop.src1_tensor = static_cast<u8_t>(src1);
  uop.dst_tensor = static_cast<u8_t>(dst);
  uop.param_id = static_cast<u8_t>(param_id);
  uop.act_type = static_cast<u8_t>(act_type);
  uop.reserved0 = 0;
  uop.in_h = static_cast<u16_t>(in_h);
  uop.in_w = static_cast<u16_t>(in_w);
  uop.in_c = static_cast<u16_t>(in_c);
  uop.out_c = static_cast<u16_t>(out_c);
  uop.kernel = static_cast<u8_t>(kernel);
  uop.stride = static_cast<u8_t>(stride);
  uop.dilation = static_cast<u8_t>(dilation);
  uop.padding = static_cast<u8_t>(padding);
  uop.c_offset = static_cast<u16_t>(c_offset);
  uop.valid_c = static_cast<u16_t>(valid_c);
  uop.qparam_id = static_cast<u16_t>(qparam_id);
  uop.reserved1 = 0;
  uop.reserved2 = 0;
}

template <unsigned IDX>
struct StaticUop;

#define ESP_INT8_STATIC_UOP(IDX, OPCODE, FLAGS, SRC0, SRC1, DST, PARAM_ID, ACT_TYPE, IN_H, IN_W, IN_C, OUT_C, KERNEL, STRIDE, DILATION, PADDING, C_OFFSET, VALID_C, QPARAM_ID) \
  template <>                                                                                                                                                                      \
  struct StaticUop<IDX> {                                                                                                                                                         \
    static void make(uop_t& uop) {                                                                                                                                                 \
      fill_static_uop(uop, OPCODE, FLAGS, SRC0, SRC1, DST, PARAM_ID, ACT_TYPE, IN_H, IN_W, IN_C, OUT_C, KERNEL, STRIDE, DILATION, PADDING, C_OFFSET, VALID_C, QPARAM_ID);          \
    }                                                                                                                                                                             \
  }

ESP_INT8_STATIC_UOP(0U, 1, 0, 255, 255, 0, 0, 0, 512, 1024, 3, 3, 0, 0, 0, 0, 0, 3, 0);
ESP_INT8_STATIC_UOP(1U, 3, 0, 0, 255, 1, 0, 0, 512, 1024, 3, 3, 3, 2, 0, 1, 0, 3, 0);
ESP_INT8_STATIC_UOP(2U, 2, 11, 0, 255, 2, 0, 1, 512, 1024, 3, 16, 3, 2, 1, 1, 0, 16, 0);
ESP_INT8_STATIC_UOP(3U, 6, 8, 1, 255, 2, 0, 0, 256, 512, 3, 19, 0, 0, 0, 0, 16, 3, 0);
ESP_INT8_STATIC_UOP(4U, 5, 80, 2, 255, 3, 0, 1, 256, 512, 19, 19, 0, 0, 0, 0, 0, 19, 0);
ESP_INT8_STATIC_UOP(5U, 3, 32, 0, 255, 18, 1, 0, 512, 1024, 3, 3, 3, 2, 0, 1, 0, 3, 0);
ESP_INT8_STATIC_UOP(6U, 3, 0, 18, 255, 8, 2, 0, 256, 512, 3, 3, 3, 2, 0, 1, 0, 3, 0);
ESP_INT8_STATIC_UOP(7U, 2, 0, 3, 255, 128, 1, 0, 256, 512, 19, 12, 3, 2, 1, 1, 0, 12, 0);
ESP_INT8_STATIC_UOP(8U, 2, 8, 128, 255, 4, 2, 0, 128, 256, 12, 16, 3, 1, 1, 1, 0, 16, 0);
ESP_INT8_STATIC_UOP(9U, 2, 0, 128, 255, 129, 3, 0, 128, 256, 12, 12, 3, 1, 2, 2, 0, 12, 0);
ESP_INT8_STATIC_UOP(10U, 6, 8, 129, 255, 4, 0, 0, 128, 256, 12, 64, 0, 0, 0, 0, 16, 12, 0);
ESP_INT8_STATIC_UOP(11U, 2, 0, 128, 255, 131, 4, 0, 128, 256, 12, 12, 3, 1, 4, 4, 0, 12, 0);
ESP_INT8_STATIC_UOP(12U, 4, 4, 129, 131, 130, 0, 0, 128, 256, 12, 12, 0, 0, 0, 0, 0, 12, 0);
ESP_INT8_STATIC_UOP(13U, 6, 8, 130, 255, 4, 0, 0, 128, 256, 12, 64, 0, 0, 0, 0, 28, 12, 0);
ESP_INT8_STATIC_UOP(14U, 2, 0, 128, 255, 131, 5, 0, 128, 256, 12, 12, 3, 1, 8, 8, 0, 12, 0);
ESP_INT8_STATIC_UOP(15U, 4, 4, 130, 131, 129, 1, 0, 128, 256, 12, 12, 0, 0, 0, 0, 0, 12, 0);
ESP_INT8_STATIC_UOP(16U, 6, 8, 129, 255, 4, 0, 0, 128, 256, 12, 64, 0, 0, 0, 0, 40, 12, 0);
ESP_INT8_STATIC_UOP(17U, 2, 0, 128, 255, 131, 6, 0, 128, 256, 12, 12, 3, 1, 16, 16, 0, 12, 0);
ESP_INT8_STATIC_UOP(18U, 4, 4, 129, 131, 130, 2, 0, 128, 256, 12, 12, 0, 0, 0, 0, 0, 12, 0);
ESP_INT8_STATIC_UOP(19U, 6, 8, 130, 255, 4, 0, 0, 128, 256, 12, 64, 0, 0, 0, 0, 52, 12, 0);
ESP_INT8_STATIC_UOP(20U, 5, 80, 4, 255, 5, 1, 1, 128, 256, 64, 64, 0, 0, 0, 0, 0, 64, 0);
ESP_INT8_STATIC_UOP(21U, 2, 0, 5, 255, 128, 7, 0, 128, 256, 64, 12, 1, 1, 1, 0, 0, 12, 0);
ESP_INT8_STATIC_UOP(22U, 2, 8, 128, 255, 6, 8, 0, 128, 256, 12, 16, 3, 1, 1, 1, 0, 16, 0);
ESP_INT8_STATIC_UOP(23U, 2, 0, 128, 255, 129, 9, 0, 128, 256, 12, 12, 3, 1, 2, 2, 0, 12, 0);
ESP_INT8_STATIC_UOP(24U, 6, 8, 129, 255, 6, 0, 0, 128, 256, 12, 64, 0, 0, 0, 0, 16, 12, 0);
ESP_INT8_STATIC_UOP(25U, 2, 0, 128, 255, 131, 10, 0, 128, 256, 12, 12, 3, 1, 4, 4, 0, 12, 0);
ESP_INT8_STATIC_UOP(26U, 4, 4, 129, 131, 130, 3, 0, 128, 256, 12, 12, 0, 0, 0, 0, 0, 12, 0);
ESP_INT8_STATIC_UOP(27U, 6, 8, 130, 255, 6, 0, 0, 128, 256, 12, 64, 0, 0, 0, 0, 28, 12, 0);
ESP_INT8_STATIC_UOP(28U, 2, 0, 128, 255, 131, 11, 0, 128, 256, 12, 12, 3, 1, 8, 8, 0, 12, 0);
ESP_INT8_STATIC_UOP(29U, 4, 4, 130, 131, 129, 4, 0, 128, 256, 12, 12, 0, 0, 0, 0, 0, 12, 0);
ESP_INT8_STATIC_UOP(30U, 6, 8, 129, 255, 6, 0, 0, 128, 256, 12, 64, 0, 0, 0, 0, 40, 12, 0);
ESP_INT8_STATIC_UOP(31U, 2, 0, 128, 255, 131, 12, 0, 128, 256, 12, 12, 3, 1, 16, 16, 0, 12, 0);
ESP_INT8_STATIC_UOP(32U, 4, 4, 129, 131, 130, 5, 0, 128, 256, 12, 12, 0, 0, 0, 0, 0, 12, 0);
ESP_INT8_STATIC_UOP(33U, 6, 8, 130, 255, 6, 0, 0, 128, 256, 12, 64, 0, 0, 0, 0, 52, 12, 0);
ESP_INT8_STATIC_UOP(34U, 4, 4, 6, 5, 6, 6, 0, 128, 256, 64, 64, 0, 0, 0, 0, 0, 64, 0);
ESP_INT8_STATIC_UOP(35U, 5, 80, 6, 255, 7, 2, 1, 128, 256, 64, 64, 0, 0, 0, 0, 0, 64, 0);
ESP_INT8_STATIC_UOP(36U, 6, 8, 7, 255, 9, 0, 0, 128, 256, 64, 131, 0, 0, 0, 0, 0, 64, 0);
ESP_INT8_STATIC_UOP(37U, 6, 8, 5, 255, 9, 0, 0, 128, 256, 64, 131, 0, 0, 0, 0, 64, 64, 0);
ESP_INT8_STATIC_UOP(38U, 6, 8, 8, 255, 9, 0, 0, 128, 256, 3, 131, 0, 0, 0, 0, 128, 3, 0);
ESP_INT8_STATIC_UOP(39U, 5, 80, 9, 255, 10, 3, 1, 128, 256, 131, 131, 0, 0, 0, 0, 0, 131, 0);
ESP_INT8_STATIC_UOP(40U, 2, 0, 10, 255, 128, 13, 0, 128, 256, 131, 25, 3, 2, 1, 1, 0, 25, 0);
ESP_INT8_STATIC_UOP(41U, 2, 8, 128, 255, 11, 14, 0, 64, 128, 25, 28, 3, 1, 1, 1, 0, 28, 0);
ESP_INT8_STATIC_UOP(42U, 2, 0, 128, 255, 129, 15, 0, 64, 128, 25, 25, 3, 1, 2, 2, 0, 25, 0);
ESP_INT8_STATIC_UOP(43U, 6, 8, 129, 255, 11, 0, 0, 64, 128, 25, 128, 0, 0, 0, 0, 28, 25, 0);
ESP_INT8_STATIC_UOP(44U, 2, 0, 128, 255, 131, 16, 0, 64, 128, 25, 25, 3, 1, 4, 4, 0, 25, 0);
ESP_INT8_STATIC_UOP(45U, 4, 4, 129, 131, 130, 7, 0, 64, 128, 25, 25, 0, 0, 0, 0, 0, 25, 0);
ESP_INT8_STATIC_UOP(46U, 6, 8, 130, 255, 11, 0, 0, 64, 128, 25, 128, 0, 0, 0, 0, 53, 25, 0);
ESP_INT8_STATIC_UOP(47U, 2, 0, 128, 255, 131, 17, 0, 64, 128, 25, 25, 3, 1, 8, 8, 0, 25, 0);
ESP_INT8_STATIC_UOP(48U, 4, 4, 130, 131, 129, 8, 0, 64, 128, 25, 25, 0, 0, 0, 0, 0, 25, 0);
ESP_INT8_STATIC_UOP(49U, 6, 8, 129, 255, 11, 0, 0, 64, 128, 25, 128, 0, 0, 0, 0, 78, 25, 0);
ESP_INT8_STATIC_UOP(50U, 2, 0, 128, 255, 131, 18, 0, 64, 128, 25, 25, 3, 1, 16, 16, 0, 25, 0);
ESP_INT8_STATIC_UOP(51U, 4, 4, 129, 131, 130, 9, 0, 64, 128, 25, 25, 0, 0, 0, 0, 0, 25, 0);
ESP_INT8_STATIC_UOP(52U, 6, 8, 130, 255, 11, 0, 0, 64, 128, 25, 128, 0, 0, 0, 0, 103, 25, 0);
ESP_INT8_STATIC_UOP(53U, 5, 80, 11, 255, 12, 4, 1, 64, 128, 128, 128, 0, 0, 0, 0, 0, 128, 0);
ESP_INT8_STATIC_UOP(54U, 2, 0, 12, 255, 128, 19, 0, 64, 128, 128, 25, 1, 1, 1, 0, 0, 25, 0);
ESP_INT8_STATIC_UOP(55U, 2, 8, 128, 255, 13, 20, 0, 64, 128, 25, 28, 3, 1, 1, 1, 0, 28, 0);
ESP_INT8_STATIC_UOP(56U, 2, 0, 128, 255, 129, 21, 0, 64, 128, 25, 25, 3, 1, 2, 2, 0, 25, 0);
ESP_INT8_STATIC_UOP(57U, 6, 8, 129, 255, 13, 0, 0, 64, 128, 25, 128, 0, 0, 0, 0, 28, 25, 0);
ESP_INT8_STATIC_UOP(58U, 2, 0, 128, 255, 131, 22, 0, 64, 128, 25, 25, 3, 1, 4, 4, 0, 25, 0);
ESP_INT8_STATIC_UOP(59U, 4, 4, 129, 131, 130, 10, 0, 64, 128, 25, 25, 0, 0, 0, 0, 0, 25, 0);
ESP_INT8_STATIC_UOP(60U, 6, 8, 130, 255, 13, 0, 0, 64, 128, 25, 128, 0, 0, 0, 0, 53, 25, 0);
ESP_INT8_STATIC_UOP(61U, 2, 0, 128, 255, 131, 23, 0, 64, 128, 25, 25, 3, 1, 8, 8, 0, 25, 0);
ESP_INT8_STATIC_UOP(62U, 4, 4, 130, 131, 129, 11, 0, 64, 128, 25, 25, 0, 0, 0, 0, 0, 25, 0);
ESP_INT8_STATIC_UOP(63U, 6, 8, 129, 255, 13, 0, 0, 64, 128, 25, 128, 0, 0, 0, 0, 78, 25, 0);
ESP_INT8_STATIC_UOP(64U, 2, 0, 128, 255, 131, 24, 0, 64, 128, 25, 25, 3, 1, 16, 16, 0, 25, 0);
ESP_INT8_STATIC_UOP(65U, 4, 4, 129, 131, 130, 12, 0, 64, 128, 25, 25, 0, 0, 0, 0, 0, 25, 0);
ESP_INT8_STATIC_UOP(66U, 6, 8, 130, 255, 13, 0, 0, 64, 128, 25, 128, 0, 0, 0, 0, 103, 25, 0);
ESP_INT8_STATIC_UOP(67U, 4, 4, 13, 12, 13, 13, 0, 64, 128, 128, 128, 0, 0, 0, 0, 0, 128, 0);
ESP_INT8_STATIC_UOP(68U, 5, 80, 13, 255, 14, 5, 1, 64, 128, 128, 128, 0, 0, 0, 0, 0, 128, 0);
ESP_INT8_STATIC_UOP(69U, 6, 8, 12, 255, 15, 0, 0, 64, 128, 128, 256, 0, 0, 0, 0, 0, 128, 0);
ESP_INT8_STATIC_UOP(70U, 6, 8, 14, 255, 15, 0, 0, 64, 128, 128, 256, 0, 0, 0, 0, 128, 128, 0);
ESP_INT8_STATIC_UOP(71U, 5, 16, 15, 255, 16, 6, 1, 64, 128, 256, 256, 0, 0, 0, 0, 0, 256, 0);
ESP_INT8_STATIC_UOP(72U, 2, 0, 16, 255, 17, 25, 0, 64, 128, 256, 2, 1, 1, 1, 0, 0, 2, 0);
ESP_INT8_STATIC_UOP(73U, 6, 0, 17, 255, 255, 0, 0, 64, 128, 2, 2, 0, 0, 0, 0, 0, 2, 0);
ESP_INT8_STATIC_UOP(74U, 15, 64, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

#undef ESP_INT8_STATIC_UOP

static bool build_p6_static_uop(unsigned uop_id, uop_t& uop) {
#pragma HLS INLINE off
  switch (uop_id) {
#define ESP_INT8_BUILD_CASE(ID) \
    case ID:                    \
      StaticUop<ID>::make(uop); \
      return true
    ESP_INT8_BUILD_CASE(0U);
    ESP_INT8_BUILD_CASE(1U);
    ESP_INT8_BUILD_CASE(2U);
    ESP_INT8_BUILD_CASE(3U);
    ESP_INT8_BUILD_CASE(4U);
    ESP_INT8_BUILD_CASE(5U);
    ESP_INT8_BUILD_CASE(6U);
    ESP_INT8_BUILD_CASE(7U);
    ESP_INT8_BUILD_CASE(8U);
    ESP_INT8_BUILD_CASE(9U);
    ESP_INT8_BUILD_CASE(10U);
    ESP_INT8_BUILD_CASE(11U);
    ESP_INT8_BUILD_CASE(12U);
    ESP_INT8_BUILD_CASE(13U);
    ESP_INT8_BUILD_CASE(14U);
    ESP_INT8_BUILD_CASE(15U);
    ESP_INT8_BUILD_CASE(16U);
    ESP_INT8_BUILD_CASE(17U);
    ESP_INT8_BUILD_CASE(18U);
    ESP_INT8_BUILD_CASE(19U);
    ESP_INT8_BUILD_CASE(20U);
    ESP_INT8_BUILD_CASE(21U);
    ESP_INT8_BUILD_CASE(22U);
    ESP_INT8_BUILD_CASE(23U);
    ESP_INT8_BUILD_CASE(24U);
    ESP_INT8_BUILD_CASE(25U);
    ESP_INT8_BUILD_CASE(26U);
    ESP_INT8_BUILD_CASE(27U);
    ESP_INT8_BUILD_CASE(28U);
    ESP_INT8_BUILD_CASE(29U);
    ESP_INT8_BUILD_CASE(30U);
    ESP_INT8_BUILD_CASE(31U);
    ESP_INT8_BUILD_CASE(32U);
    ESP_INT8_BUILD_CASE(33U);
    ESP_INT8_BUILD_CASE(34U);
    ESP_INT8_BUILD_CASE(35U);
    ESP_INT8_BUILD_CASE(36U);
    ESP_INT8_BUILD_CASE(37U);
    ESP_INT8_BUILD_CASE(38U);
    ESP_INT8_BUILD_CASE(39U);
    ESP_INT8_BUILD_CASE(40U);
    ESP_INT8_BUILD_CASE(41U);
    ESP_INT8_BUILD_CASE(42U);
    ESP_INT8_BUILD_CASE(43U);
    ESP_INT8_BUILD_CASE(44U);
    ESP_INT8_BUILD_CASE(45U);
    ESP_INT8_BUILD_CASE(46U);
    ESP_INT8_BUILD_CASE(47U);
    ESP_INT8_BUILD_CASE(48U);
    ESP_INT8_BUILD_CASE(49U);
    ESP_INT8_BUILD_CASE(50U);
    ESP_INT8_BUILD_CASE(51U);
    ESP_INT8_BUILD_CASE(52U);
    ESP_INT8_BUILD_CASE(53U);
    ESP_INT8_BUILD_CASE(54U);
    ESP_INT8_BUILD_CASE(55U);
    ESP_INT8_BUILD_CASE(56U);
    ESP_INT8_BUILD_CASE(57U);
    ESP_INT8_BUILD_CASE(58U);
    ESP_INT8_BUILD_CASE(59U);
    ESP_INT8_BUILD_CASE(60U);
    ESP_INT8_BUILD_CASE(61U);
    ESP_INT8_BUILD_CASE(62U);
    ESP_INT8_BUILD_CASE(63U);
    ESP_INT8_BUILD_CASE(64U);
    ESP_INT8_BUILD_CASE(65U);
    ESP_INT8_BUILD_CASE(66U);
    ESP_INT8_BUILD_CASE(67U);
    ESP_INT8_BUILD_CASE(68U);
    ESP_INT8_BUILD_CASE(69U);
    ESP_INT8_BUILD_CASE(70U);
    ESP_INT8_BUILD_CASE(71U);
    ESP_INT8_BUILD_CASE(72U);
    ESP_INT8_BUILD_CASE(73U);
    ESP_INT8_BUILD_CASE(74U);
#undef ESP_INT8_BUILD_CASE
    default:
      uop = uop_t();
      return false;
  }
}

static bool p6_csim_stop_before(unsigned uop_id) {
#pragma HLS INLINE
#ifdef ESP_INT8_CSIM_MAX_UOP
  return uop_id > static_cast<unsigned>(ESP_INT8_CSIM_MAX_UOP);
#else
  return false;
#endif
}

static bool p6_is_fused_conv_store_start(unsigned uop_id) {
#pragma HLS INLINE
  return uop_id == 9U || uop_id == 23U || uop_id == 42U || uop_id == 56U;
}

static bool p6_store_of(const uop_t& store_uop, const uop_t& producer_uop) {
#pragma HLS INLINE
  return store_uop.opcode.to_uint() == static_cast<unsigned>(UOP_STORE) &&
         store_uop.src0_tensor.to_uint() == producer_uop.dst_tensor.to_uint();
}

static error_code_t prepare_p6_conv_store_fusion(uop_t& conv_uop,
                                                 const uop_t& store_uop,
                                                 bool& alias_after_conv,
                                                 u8_t& alias_tensor,
                                                 tensor_desc_t& alias_dst) {
#pragma HLS INLINE off
  if (!p6_store_of(store_uop, conv_uop)) {
    return ERR_UOP_DECODE;
  }

  alias_after_conv = false;
  alias_tensor = conv_uop.dst_tensor;
  if (store_uop.dst_tensor.to_uint() == static_cast<unsigned>(TID_INVALID)) {
    return ERR_NONE;
  }

  const u16_t stride = effective_stride(conv_uop);
  const u16_t out_h = conv_out_dim(conv_uop.in_h, stride);
  const u16_t out_w = conv_out_dim(conv_uop.in_w, stride);
  tensor_desc_t resolved_dst;
  if (!resolve_tensor_write(store_uop.dst_tensor,
                            out_h,
                            out_w,
                            static_cast<u16_t>(store_uop.out_c),
                            resolved_dst)) {
    return ERR_TENSOR_DESC_RANGE;
  }
  if (resolved_dst.c.to_uint() < store_uop.c_offset.to_uint() + conv_uop.out_c.to_uint()) {
    return ERR_TENSOR_DESC_RANGE;
  }

  conv_uop.dst_tensor = store_uop.dst_tensor;
  conv_uop.c_offset = store_uop.c_offset;
  alias_dst = resolved_dst;
  alias_after_conv = true;
  return ERR_NONE;
}

static error_code_t run_p6_dispatch_uop(const uop_t& uop, axi_vec_t* gmem_frame_out) {
#pragma HLS INLINE off
  switch (uop.opcode.to_uint()) {
    case UOP_NOP:
    case UOP_LOAD_FM:
    case UOP_END:
      return ERR_NONE;
    case UOP_POOL:
      return run_p6_pool_op(uop);
    case UOP_CONV:
      return run_p6_conv_op(uop, gmem_frame_out);
    case UOP_ADD:
      return run_p6_add_op(uop);
    case UOP_AFFINE:
      return run_p6_affine_op(uop);
    case UOP_STORE:
      return run_p6_store_op(uop);
    default:
      return ERR_UNSUPPORTED_OPCODE;
  }
}

static error_code_t run_p6_static_graph(axi_vec_t* gmem_frame_out) {
#pragma HLS INLINE off
  // Fixed P6 schedule: UOP 1..6 stem, 7..39 level2, 40..71 level3,
  // and UOP 72 classifier/full-resolution output. The loop keeps one call
  // site for each operator datapath so HLS shares hardware instead of cloning
  // it once per static UOP.
  for (unsigned uop_id = 1U; uop_id <= 72U; ++uop_id) {
#pragma HLS PIPELINE off
#ifndef __SYNTHESIS__
    s_csim_last_uop = uop_id;
#endif
    if (p6_csim_stop_before(uop_id)) {
      return ERR_NONE;
    }

    uop_t uop;
    if (!build_p6_static_uop(uop_id, uop)) {
      return ERR_UOP_DECODE;
    }

    bool skip_next_uop = false;
    bool alias_after_conv = false;
    u8_t alias_tensor = static_cast<u8_t>(static_cast<unsigned>(TID_INVALID));
    tensor_desc_t alias_dst;
    if (p6_is_fused_conv_store_start(uop_id)) {
      uop_t store_uop;
      if (!build_p6_static_uop(uop_id + 1U, store_uop)) {
#ifndef __SYNTHESIS__
        s_csim_last_error = ERR_UOP_DECODE;
#endif
        return ERR_UOP_DECODE;
      }
      const error_code_t prep_err = prepare_p6_conv_store_fusion(uop,
                                                                 store_uop,
                                                                 alias_after_conv,
                                                                 alias_tensor,
                                                                 alias_dst);
      if (prep_err != ERR_NONE) {
#ifndef __SYNTHESIS__
        s_csim_last_error = prep_err;
#endif
        return prep_err;
      }
      skip_next_uop = true;
    }

    const error_code_t err = run_p6_dispatch_uop(uop, gmem_frame_out);
    if (err != ERR_NONE) {
#ifndef __SYNTHESIS__
      s_csim_last_error = err;
#endif
      return err;
    }
    if (alias_after_conv &&
        !alias_tensor_to_slice(alias_tensor, alias_dst, uop.c_offset, uop.out_c)) {
#ifndef __SYNTHESIS__
      s_csim_last_error = ERR_TENSOR_DESC_RANGE;
#endif
      return ERR_TENSOR_DESC_RANGE;
    }
    if (skip_next_uop) {
      ++uop_id;
    }
  }
#ifndef __SYNTHESIS__
  s_csim_last_error = ERR_NONE;
#endif
  return ERR_NONE;
}

static error_code_t core_mode_run(const axi_vec_t* gmem_frame_in,
                                  axi_vec_t* gmem_frame_out,
                                  u32_t expected_uop_count) {
#pragma HLS INLINE off
  if (!s_param_ready) {
#ifndef __SYNTHESIS__
    s_csim_last_error = ERR_BAD_BLOB;
#endif
    return ERR_BAD_BLOB;
  }

  if (expected_uop_count != UOP_COUNT_ENCODER ||
      s_param_header.uop_count.to_uint() != static_cast<unsigned>(UOP_COUNT_ENCODER)) {
#ifndef __SYNTHESIS__
    s_csim_last_error = ERR_UOP_DECODE;
#endif
    return ERR_UOP_DECODE;
  }

  frame_dma_load(gmem_frame_in);

  const error_code_t err = run_p6_static_graph(gmem_frame_out);
  if (err != ERR_NONE) {
    return err;
  }

  return ERR_NONE;
}

}  // namespace esp_int8

static_assert(esp_int8::INPUT_FRAME_AXI_WORDS == 49152,
              "Update gmem_frame_in m_axi depth when INPUT_FRAME_AXI_WORDS changes.");
static_assert(esp_int8::OUTPUT_FRAME_AXI_WORDS == 16384,
              "Update gmem_frame_out m_axi depth when OUTPUT_FRAME_AXI_WORDS changes.");

void espnet_encoder_int8_core(const esp_int8::axi_vec_t* gmem_frame_in,
                              esp_int8::axi_vec_t* gmem_frame_out,
                              const esp_int8::axi_vec_t* gmem_param,
                              std::uint32_t mode,
                              std::uint32_t uop_count) {
#ifdef ESP_INT8_COSIM_LITE
#pragma HLS INTERFACE ap_memory port=gmem_frame_in depth=49152
#pragma HLS INTERFACE ap_memory port=gmem_frame_out depth=16384
#pragma HLS INTERFACE ap_memory port=gmem_param depth=4096
#pragma HLS INTERFACE ap_none port=mode
#pragma HLS INTERFACE ap_none port=uop_count
#pragma HLS INTERFACE ap_ctrl_hs port=return
#else
#pragma HLS INTERFACE m_axi port=gmem_frame_in offset=slave bundle=gmem0 depth=49152 max_read_burst_length=64 num_read_outstanding=4
#pragma HLS INTERFACE m_axi port=gmem_frame_out offset=slave bundle=gmem1 depth=16384 max_write_burst_length=64 num_write_outstanding=4
#pragma HLS INTERFACE m_axi port=gmem_param offset=slave bundle=gmem2 depth=4096 max_read_burst_length=64 num_read_outstanding=4
#pragma HLS INTERFACE s_axilite port=gmem_frame_in bundle=control
#pragma HLS INTERFACE s_axilite port=gmem_frame_out bundle=control
#pragma HLS INTERFACE s_axilite port=gmem_param bundle=control
#pragma HLS INTERFACE s_axilite port=mode bundle=control
#pragma HLS INTERFACE s_axilite port=uop_count bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#endif

  const std::uint32_t mode_runtime = esp_int8::runtime_mode(mode);
  const esp_int8::u32_t expected_uop_count = uop_count;

  switch (mode_runtime) {
    case esp_int8::MODE_INIT:
      esp_int8::core_mode_init(gmem_param);
      break;
    case esp_int8::MODE_RUN:
      esp_int8::core_mode_run(gmem_frame_in,
                              gmem_frame_out,
                              expected_uop_count);
      break;
    case esp_int8::MODE_IDLE:
      break;
    default:
      break;
  }
}
