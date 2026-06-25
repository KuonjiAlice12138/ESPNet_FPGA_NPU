#include <cstdint>

#include "../include/npu_config.hpp"
#include "../include/npu_q.hpp"
#include "../include/npu_schedule.hpp"
#include "../include/npu_uop.hpp"

#ifndef __SYNTHESIS__
#include <fstream>
#endif

namespace esp_int8 {

void param_dma_init(const axi_vec_t* gmem_param);
bool param_dma_ready();
bool param_dma_is_schedule_blob();
bool param_dma_get_pool_qparam(u8_t param_id, pool_q_t& qparam);
bool param_dma_get_conv_qparam(u8_t param_id, conv_q_t& qparam);
bool param_dma_get_conv_exec_desc(u8_t id, conv_exec_desc_t& desc);
bool param_dma_get_window_sched(u8_t id, window_sched_desc_t& desc);
bool param_dma_get_row_consumer(u8_t id, row_consumer_desc_t& desc);
bool param_dma_get_exec_entry(u8_t pc, exec_plan_entry_t& entry);
bool param_dma_get_packed_weight_vec(const conv_exec_desc_t& desc, u16_t tm, u16_t kt, wgt_vec_t& word);
bool param_dma_get_affine_qparam(u8_t param_id, u8_t block_id, aff_q_t& qparam);
bool param_dma_get_add_qparam(u8_t param_id, add_q_t& qparam);
void frame_dma_load(const axi_vec_t* gmem_frame_in);
bool avgpool_unit_checked(const tensor_desc_t& src,
                          const tensor_desc_t& dst,
                          const pool_q_t& qparam,
                          i8_t* fmbuf_base);
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
void scheduled_window_generator_row(const tensor_desc_t& src_desc,
                                    const conv_exec_desc_t& conv_desc,
                                    const window_sched_desc_t& sched,
                                    hls::stream<act_vec_t>& act_stream,
                                    u16_t out_row);
void systolic_array_core_row(hls::stream<act_vec_t>& act_stream,
                             hls::stream<wgt_vec_t>& wgt_stream,
                             hls::stream<psum_vec_t>& psum_stream,
                             const conv_cfg_t& cfg);
bool store_conv_output_row(const tensor_desc_t& dst,
                           u16_t out_row,
                           u16_t c_offset,
                           u16_t store_layout,
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
static bool s_csim_tensor_dumped = false;

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

static conv_cfg_t conv_cfg_from_exec_desc(const conv_exec_desc_t& desc) {
#pragma HLS INLINE
  conv_cfg_t cfg;
  cfg.in_h = desc.in_h;
  cfg.in_w = desc.in_w;
  cfg.in_c = desc.in_c;
  cfg.out_c = desc.out_c;
  cfg.kernel = (desc.kernel.to_uint() == 1U) ? ap_uint<2>(1) : ap_uint<2>(3);
  cfg.stride = (desc.stride.to_uint() == 0U) ? ap_uint<2>(1) : ap_uint<2>(desc.stride);
  cfg.dilation = (desc.dilation.to_uint() == 0U) ? ap_uint<5>(1) : ap_uint<5>(desc.dilation);
  cfg.bias_en = (((desc.flags.to_uint() >> static_cast<unsigned>(UOP_FLAG_BIAS_EN)) & 0x1U) != 0U)
                    ? ap_uint<1>(1)
                    : ap_uint<1>(0);
  return cfg;
}

static u8_t conv_act_type_from_flags(u16_t flags) {
#pragma HLS INLINE
  return (((flags.to_uint() >> static_cast<unsigned>(UOP_FLAG_RELU_EN)) & 0x1U) != 0U)
             ? static_cast<u8_t>(static_cast<unsigned>(ACT_RELU))
             : static_cast<u8_t>(static_cast<unsigned>(ACT_NONE));
}

static void make_conv_uop_from_exec_desc(const conv_exec_desc_t& desc, uop_t& uop) {
#pragma HLS INLINE
  uop = uop_t();
  uop.opcode = static_cast<u8_t>(static_cast<unsigned>(UOP_CONV));
  uop.flags = static_cast<u8_t>(desc.flags.to_uint() & 0xffU);
  uop.src0_tensor = desc.src_tensor;
  uop.src1_tensor = static_cast<u8_t>(static_cast<unsigned>(TID_INVALID));
  uop.dst_tensor = desc.dst_tensor;
  uop.param_id = desc.param_id;
  uop.act_type = conv_act_type_from_flags(desc.flags);
  uop.in_h = desc.in_h;
  uop.in_w = desc.in_w;
  uop.in_c = desc.in_c;
  uop.out_c = desc.out_c;
  uop.kernel = desc.kernel;
  uop.stride = desc.stride;
  uop.dilation = desc.dilation;
  uop.padding = desc.padding;
  uop.c_offset = desc.dst_c_offset;
  uop.valid_c = desc.valid_c;
  uop.qparam_id = desc.qparam_id;
}

static u16_t conv_effective_stride(const conv_cfg_t& cfg) {
#pragma HLS INLINE
  return (cfg.stride == 0) ? static_cast<u16_t>(1) : static_cast<u16_t>(cfg.stride);
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
                                        u8_t act_type,
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
                                           act_type);
        set_act_vec_i8_dynamic(packed, tm, out);
      }
    }
    row_buf[ow_i] = packed;
  }
}

static void generate_conv_window_row(const tensor_desc_t& src,
                                     hls::stream<act_vec_t>& act_stream,
                                     const conv_cfg_t& cfg,
                                     const conv_exec_desc_t& conv_desc,
                                     const window_sched_desc_t& sched,
                                     u16_t out_row) {
#pragma HLS INLINE off
  (void)cfg;
  scheduled_window_generator_row(src, conv_desc, sched, act_stream, out_row);
}

static void execute_conv_stream_row_region(const tensor_desc_t& src,
                                           const conv_cfg_t& cfg,
                                           const conv_exec_desc_t& conv_desc,
                                           const window_sched_desc_t& sched,
                                           u8_t act_type,
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
  generate_conv_window_row(src, act_stream, cfg, conv_desc, sched, out_row);
  feed_cached_weights(wgt_stream, cached_wgts, wgt_count);
  systolic_array_core_row(act_stream, wgt_stream, psum_stream, cfg);
  post_process_row_to_buffer(psum_stream, row_buf, cfg, act_type, qparam);
}

static void add_other_row_to_buffer(const tensor_desc_t& other,
                                    u16_t out_row,
                                    const conv_cfg_t& cfg,
                                    u8_t valid_c,
                                    const add_q_t& qparam,
                                    act_vec_t row_buf[MAX_FM_W],
                                    bool& ok) {
#pragma HLS INLINE off
  const u16_t stride = conv_effective_stride(cfg);
  const u16_t out_w = conv_out_dim(cfg.in_w, stride);
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
    for (int lane = 0; lane < TM; ++lane) {
#pragma HLS UNROLL
      if (static_cast<unsigned>(lane) < valid_c.to_uint()) {
        const i8_t a_value = get_act_vec_i8_dynamic(row_buf[ow_i], lane);
        const i8_t b_value = get_act_vec_i8_dynamic(other_word, lane);
        const i8_t out_value = add_i8(a_value, b_value, qparam);
        set_act_vec_i8_dynamic(out_word, lane, out_value);
      }
    }
    row_buf[ow_i] = out_word;
  }
}

static bool store_consumer_row(const tensor_desc_t& dst,
                               u16_t out_row,
                               u16_t c_offset,
                               u8_t valid_c,
                               u16_t store_layout,
                               const conv_cfg_t& cfg,
                               act_vec_t row_buf[MAX_FM_W]) {
#pragma HLS INLINE off
  conv_cfg_t store_cfg = cfg;
  store_cfg.out_c = static_cast<u16_t>(valid_c);
  return store_conv_output_row(dst, out_row, c_offset, store_layout, store_cfg, row_buf);
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

static bool consume_conv_output_row(const row_consumer_desc_t& consumer,
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

  bool ok = true;
  if (mode == static_cast<unsigned>(ROW_CONSUMER_ADD_STORE) ||
      mode == static_cast<unsigned>(ROW_CONSUMER_ADD_AFFINE_STORE)) {
    if (!has_add_other) {
      return false;
    }
    add_other_row_to_buffer(add_other, out_row, cfg, valid_c, add_qparam, row_buf, ok);
    if (!ok) {
      return false;
    }
  }

  if (mode == static_cast<unsigned>(ROW_CONSUMER_NONE) ||
      mode == static_cast<unsigned>(ROW_CONSUMER_STORE) ||
      mode == static_cast<unsigned>(ROW_CONSUMER_ADD_STORE)) {
    return store_consumer_row(dst,
                              out_row,
                              consumer.store_c_offset,
                              valid_c,
                              consumer.reserved0,
                              cfg,
                              row_buf);
  }

  return false;
}

#if !defined(__SYNTHESIS__) && \
    (defined(ESP_INT8_CSIM_DUMP_DEBUG_SET) || \
     defined(ESP_INT8_CSIM_DUMP_L2_SET) || \
     (defined(ESP_INT8_CSIM_DUMP_AFTER_UOP) && defined(ESP_INT8_CSIM_DUMP_TENSOR_ID)))

static std::uint8_t csim_act_byte(const act_vec_t& word, int lane) {
  return static_cast<std::uint8_t>(word.range(lane * 8 + 7, lane * 8).to_uint());
}

static bool csim_dump_tensor_named(u8_t tensor_id,
                                   const char* filename,
                                   unsigned logical_uop) {
  tensor_desc_t desc;
  if (!resolve_tensor_read(tensor_id, desc)) {
    std::printf("[CSIM-DUMP] failed to resolve tensor=%u at logical_uop=%u file=%s\n",
                static_cast<unsigned>(tensor_id.to_uint()),
                logical_uop,
                filename);
    return false;
  }

  std::ofstream out(filename, std::ios::binary);
  if (!out) {
    std::printf("[CSIM-DUMP] failed to open %s\n", filename);
    return false;
  }

  const unsigned h_count = desc.h.to_uint();
  const unsigned w_count = desc.w.to_uint();
  const unsigned c_count = desc.c.to_uint();

  std::uint8_t bytes[TM];
  for (unsigned h = 0; h < h_count; ++h) {
    for (unsigned w = 0; w < w_count; ++w) {
      for (unsigned c = 0; c < c_count; c += static_cast<unsigned>(TM)) {
        const unsigned remaining = c_count - c;
        const unsigned lanes =
            (remaining < static_cast<unsigned>(TM))
                ? remaining
                : static_cast<unsigned>(TM);

        act_vec_t packed = 0;
        if (!on_chip_memory_read_packed_tile(desc,
                                             static_cast<i32_t>(h),
                                             static_cast<i32_t>(w),
                                             static_cast<u16_t>(c),
                                             static_cast<u8_t>(lanes),
                                             packed)) {
          std::printf("[CSIM-DUMP] read failed tensor=%u h=%u w=%u c=%u file=%s\n",
                      static_cast<unsigned>(tensor_id.to_uint()),
                      h,
                      w,
                      c,
                      filename);
          return false;
        }

        for (unsigned lane = 0; lane < lanes; ++lane) {
          bytes[lane] = csim_act_byte(packed, static_cast<int>(lane));
        }
        out.write(reinterpret_cast<const char*>(bytes),
                  static_cast<std::streamsize>(lanes));
      }
    }
  }

  std::printf("[CSIM-DUMP] wrote %s tensor=%u after/pre uop=%u shape=%ux%ux%u bytes=%u\n",
              filename,
              static_cast<unsigned>(tensor_id.to_uint()),
              logical_uop,
              h_count,
              w_count,
              c_count,
              h_count * w_count * c_count);
  return true;
}

static bool s_dump_u40 = false;
static bool s_dump_u53 = false;
static bool s_dump_pre_u68 = false;
static bool s_dump_u68 = false;
static bool s_dump_u70 = false;
static bool s_dump_u71 = false;
static bool s_dump_u38 = false;
static bool s_dump_u39 = false;
static bool s_dump_l2_u19 = false;
static bool s_dump_l2_u20 = false;
static bool s_dump_l2_u21 = false;
static bool s_dump_l2_u22 = false;
static bool s_dump_l2_u25 = false;
static bool s_dump_l2_u28 = false;
static bool s_dump_l2_u31 = false;
static bool s_dump_l2_u35 = false;

static bool csim_dump_tensor_set_pre(unsigned logical_uop) {
#ifdef ESP_INT8_CSIM_DUMP_DEBUG_SET
  // U67 ADD + U68 AFFINE are fused as EXEC_ADD_AFFINE.
  // Dump U67 inputs before executing the logical U68 entry.
  if (logical_uop == 68U && !s_dump_pre_u68) {
    if (!csim_dump_tensor_named(static_cast<u8_t>(static_cast<unsigned>(TID_L3B0_CAT)),
                                "csim_pre_u68_l3b0_cat.bin",
                                logical_uop)) {
      return false;
    }
    if (!csim_dump_tensor_named(static_cast<u8_t>(static_cast<unsigned>(TID_L30_ACT)),
                                "csim_pre_u68_l30_act.bin",
                                logical_uop)) {
      return false;
    }
    s_dump_pre_u68 = true;
  }
#else
  (void)logical_uop;
#endif
  return true;
}

static bool csim_dump_tensor_set_post(unsigned logical_uop) {
#if defined(ESP_INT8_CSIM_DUMP_AFTER_UOP) && defined(ESP_INT8_CSIM_DUMP_TENSOR_ID)
  static bool s_dump_generic = false;
  if (!s_dump_generic &&
      logical_uop == static_cast<unsigned>(ESP_INT8_CSIM_DUMP_AFTER_UOP)) {
    if (!csim_dump_tensor_named(static_cast<u8_t>(ESP_INT8_CSIM_DUMP_TENSOR_ID),
                                "csim_tensor_dump.bin",
                                logical_uop)) {
      return false;
    }
    s_dump_generic = true;
  }
#endif

#ifdef ESP_INT8_CSIM_DUMP_DEBUG_SET
  if (logical_uop == 40U && !s_dump_u40) {
    if (!csim_dump_tensor_named(static_cast<u8_t>(static_cast<unsigned>(LS_C1)),
                                "csim_u40_level3_0_c1.bin",
                                logical_uop)) {
      return false;
    }
    s_dump_u40 = true;
  }

  if (logical_uop == 38U && !s_dump_u38) {
    if (!csim_dump_tensor_named(static_cast<u8_t>(static_cast<unsigned>(TID_B2_CAT)),
                                "csim_u38_b2_cat.bin",
                                logical_uop)) {
      return false;
    }
    s_dump_u38 = true;
  }

  if (logical_uop == 39U && !s_dump_u39) {
    if (!csim_dump_tensor_named(static_cast<u8_t>(static_cast<unsigned>(TID_B2_ACT)),
                                "csim_u39_b2_bn.bin",
                                logical_uop)) {
      return false;
    }
    s_dump_u39 = true;
  }

  if (logical_uop == 53U && !s_dump_u53) {
    if (!csim_dump_tensor_named(static_cast<u8_t>(static_cast<unsigned>(TID_L30_ACT)),
                                "csim_u53_level3_0_bn.bin",
                                logical_uop)) {
      return false;
    }
    s_dump_u53 = true;
  }

  if (logical_uop == 68U && !s_dump_u68) {
    if (!csim_dump_tensor_named(static_cast<u8_t>(static_cast<unsigned>(TID_L3B0_ACT)),
                                "csim_u68_l3b0_act.bin",
                                logical_uop)) {
      return false;
    }
    s_dump_u68 = true;
  }

  if (logical_uop == 70U && !s_dump_u70) {
    if (!csim_dump_tensor_named(static_cast<u8_t>(static_cast<unsigned>(TID_B3_CAT)),
                                "csim_u70_b3_cat.bin",
                                logical_uop)) {
      return false;
    }
    s_dump_u70 = true;
  }

  if (logical_uop == 71U && !s_dump_u71) {
    if (!csim_dump_tensor_named(static_cast<u8_t>(static_cast<unsigned>(TID_B3_ACT)),
                                "csim_u71_b3_bn.bin",
                                logical_uop)) {
      return false;
    }
    s_dump_u71 = true;
  }
#endif

#ifdef ESP_INT8_CSIM_DUMP_L2_SET
  if (logical_uop == 19U && !s_dump_l2_u19) {
    if (!csim_dump_tensor_named(static_cast<u8_t>(static_cast<unsigned>(TID_L20_CAT)),
                                "csim_l2_u19_l20_cat.bin",
                                logical_uop)) {
      return false;
    }
    s_dump_l2_u19 = true;
  }
  if (logical_uop == 20U && !s_dump_l2_u20) {
    if (!csim_dump_tensor_named(static_cast<u8_t>(static_cast<unsigned>(TID_L20_ACT)),
                                "csim_l2_u20_l20_act.bin",
                                logical_uop)) {
      return false;
    }
    s_dump_l2_u20 = true;
  }
  if (logical_uop == 21U && !s_dump_l2_u21) {
    if (!csim_dump_tensor_named(static_cast<u8_t>(static_cast<unsigned>(LS_C1)),
                                "csim_l2_u21_l2b0_c1.bin",
                                logical_uop)) {
      return false;
    }
    s_dump_l2_u21 = true;
  }
  if (logical_uop == 22U && !s_dump_l2_u22) {
    if (!csim_dump_tensor_named(static_cast<u8_t>(static_cast<unsigned>(TID_L2B0_CAT)),
                                "csim_l2_u22_l2b0_cat.bin",
                                logical_uop)) {
      return false;
    }
    s_dump_l2_u22 = true;
  }
  if (logical_uop == 25U && !s_dump_l2_u25) {
    if (!csim_dump_tensor_named(static_cast<u8_t>(static_cast<unsigned>(TID_L2B0_CAT)),
                                "csim_l2_u25_l2b0_cat.bin",
                                logical_uop)) {
      return false;
    }
    s_dump_l2_u25 = true;
  }
  if (logical_uop == 28U && !s_dump_l2_u28) {
    if (!csim_dump_tensor_named(static_cast<u8_t>(static_cast<unsigned>(TID_L2B0_CAT)),
                                "csim_l2_u28_l2b0_cat.bin",
                                logical_uop)) {
      return false;
    }
    s_dump_l2_u28 = true;
  }
  if (logical_uop == 31U && !s_dump_l2_u31) {
    if (!csim_dump_tensor_named(static_cast<u8_t>(static_cast<unsigned>(TID_L2B0_CAT)),
                                "csim_l2_u31_l2b0_cat.bin",
                                logical_uop)) {
      return false;
    }
    s_dump_l2_u31 = true;
  }
  if (logical_uop == 35U && !s_dump_l2_u35) {
    if (!csim_dump_tensor_named(static_cast<u8_t>(static_cast<unsigned>(TID_L2B0_ACT)),
                                "csim_l2_u35_l2b0_act.bin",
                                logical_uop)) {
      return false;
    }
    s_dump_l2_u35 = true;
  }
#endif

  return true;
}

#else

static bool csim_dump_tensor_set_pre(unsigned) {
#pragma HLS INLINE
  return true;
}

static bool csim_dump_tensor_set_post(unsigned) {
#pragma HLS INLINE
  return true;
}

#endif

#if !defined(__SYNTHESIS__) && defined(ESP_INT8_CSIM_DUMP_U40_PRESTORE)
static unsigned s_csim_u40_prestore_rows = 0;

static std::uint8_t csim_rowbuf_byte(const act_vec_t& word, int lane) {
  return static_cast<std::uint8_t>(word.range(lane * 8 + 7, lane * 8).to_uint());
}

static bool csim_dump_u40_prestore_row(const conv_exec_desc_t& conv_desc,
                                       u16_t out_row,
                                       const conv_cfg_t& cfg,
                                       const act_vec_t row_buf[MAX_FM_W]) {
  const bool is_u40 =
      conv_desc.param_id.to_uint() == 13U &&
      conv_desc.dst_tensor.to_uint() == static_cast<unsigned>(LS_C1);
  if (!is_u40) {
    return true;
  }

  const unsigned row = out_row.to_uint();
  const bool first_row = row == 0U || s_csim_u40_prestore_rows == 0U;
  std::ofstream out("csim_u40_prestore_rowbuf.bin",
                    std::ios::binary | (first_row ? std::ios::trunc : std::ios::app));
  if (!out) {
    std::printf("[CSIM-DUMP] failed to open csim_u40_prestore_rowbuf.bin\n");
    return false;
  }

  const unsigned out_w = conv_out_dim(cfg.in_w, conv_effective_stride(cfg)).to_uint();
  const unsigned out_c = cfg.out_c.to_uint();
  std::uint8_t bytes[TM];
  for (unsigned ow = 0; ow < out_w; ++ow) {
    for (unsigned lane = 0; lane < out_c; ++lane) {
      bytes[lane] = csim_rowbuf_byte(row_buf[ow], static_cast<int>(lane));
    }
    out.write(reinterpret_cast<const char*>(bytes),
              static_cast<std::streamsize>(out_c));
  }

  ++s_csim_u40_prestore_rows;
  if (row == conv_out_dim(cfg.in_h, conv_effective_stride(cfg)).to_uint() - 1U) {
    std::printf("[CSIM-DUMP] wrote csim_u40_prestore_rowbuf.bin rows=%u shape=%ux%ux%u bytes=%u\n",
                s_csim_u40_prestore_rows,
                conv_out_dim(cfg.in_h, conv_effective_stride(cfg)).to_uint(),
                out_w,
                out_c,
                conv_out_dim(cfg.in_h, conv_effective_stride(cfg)).to_uint() * out_w * out_c);
  }
  return true;
}
#else
static bool csim_dump_u40_prestore_row(const conv_exec_desc_t&,
                                       u16_t,
                                       const conv_cfg_t&,
                                       const act_vec_t[MAX_FM_W]) {
#pragma HLS INLINE
  return true;
}
#endif

static void execute_conv_stream_datapath(const tensor_desc_t& src,
                                         const tensor_desc_t& dst,
                                         const tensor_desc_t& add_other,
                                         bool has_add_other,
                                         const conv_cfg_t& cfg,
                                         const conv_exec_desc_t& conv_desc,
                                         const window_sched_desc_t& sched,
                                         const row_consumer_desc_t& consumer,
                                         const conv_q_t& qparam,
                                         const add_q_t& add_qparam,
                                         axi_vec_t* gmem_frame_out,
                                         bool& ok_out) {
#pragma HLS INLINE off
  act_vec_t row_buf[MAX_FM_W];
#pragma HLS BIND_STORAGE variable=row_buf type=ram_2p impl=bram

  const u16_t stride = conv_effective_stride(cfg);
  const u16_t out_h = conv_out_dim(cfg.in_h, stride);
  const u16_t k_tiles = conv_desc.k_tiles;
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
        param_dma_get_packed_weight_vec(conv_desc, oc, k_tile, word);
        cached_wgts[wi++] = word;
      }
    }
  }

  if (consumer.mode.to_uint() == static_cast<unsigned>(ROW_CONSUMER_UPSAMPLE_OUT)) {
    upsample_fused_begin();
  }

  for (int oh_i = 0; oh_i < MAX_FM_H; ++oh_i) {
    if (oh_i >= out_h_i) {
      break;
    }
    const u16_t oh = static_cast<u16_t>(oh_i);
    execute_conv_stream_row_region(src,
                                   cfg,
                                   conv_desc,
                                   sched,
                                   conv_act_type_from_flags(conv_desc.flags),
                                   qparam,
                                   oh,
                                   row_buf,
                                   cached_wgts,
                                   wgt_count);
    if (!csim_dump_u40_prestore_row(conv_desc, oh, cfg, row_buf)) {
      write_ok = false;
    }
    if (!consume_conv_output_row(consumer,
                                 dst,
                                 add_other,
                                 has_add_other,
                                 add_qparam,
                                 cfg,
                                 gmem_frame_out,
                                 oh,
                                 row_buf)) {
      write_ok = false;
    }
  }
  ok_out = write_ok;
}

static error_code_t run_scheduled_pool_op(const uop_t& uop) {
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

static error_code_t run_scheduled_conv_op(u8_t desc_id, axi_vec_t* gmem_frame_out) {
#pragma HLS INLINE off
  conv_exec_desc_t conv_desc;
  window_sched_desc_t sched;
  row_consumer_desc_t consumer;
  tensor_desc_t src;
  tensor_desc_t dst;
  tensor_desc_t add_other;
  conv_q_t qparam;
  add_q_t add_qparam = add_q_t();
#pragma HLS ARRAY_PARTITION variable=qparam.bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=qparam.mult complete dim=1
#pragma HLS ARRAY_PARTITION variable=qparam.shift complete dim=1

  if (!param_dma_get_conv_exec_desc(desc_id, conv_desc) ||
      !param_dma_get_window_sched(conv_desc.window_sched_id, sched) ||
      !param_dma_get_row_consumer(conv_desc.row_consumer_id, consumer)) {
    return ERR_PARAM_DESC_RANGE;
  }

  const conv_cfg_t cfg = conv_cfg_from_exec_desc(conv_desc);
  const u16_t stride = conv_effective_stride(cfg);
  const u16_t out_h = conv_out_dim(cfg.in_h, stride);
  const u16_t out_w = conv_out_dim(cfg.in_w, stride);
  uop_t conv_uop;
  make_conv_uop_from_exec_desc(conv_desc, conv_uop);

  select_scratch_region(conv_uop, out_h);
  if (!resolve_tensor_read(conv_desc.src_tensor, src)) {
    return ERR_TENSOR_DESC_RANGE;
  }
  if (!param_dma_get_conv_qparam(conv_desc.qparam_id, qparam)) {
    return ERR_PARAM_DESC_RANGE;
  }

  if (cfg.out_c.to_uint() > static_cast<unsigned>(TM)) {
    return ERR_UNSUPPORTED_OPCODE;
  }

  const unsigned consumer_mode = consumer.mode.to_uint();
  const bool emit_fullres_mask =
      consumer_mode == static_cast<unsigned>(ROW_CONSUMER_UPSAMPLE_OUT);
  if (consumer_mode == static_cast<unsigned>(ROW_CONSUMER_NONE)) {
    consumer.store_dst_tensor = conv_desc.dst_tensor;
    consumer.store_c_offset = conv_desc.dst_c_offset;
    consumer.valid_c = conv_desc.valid_c;
  }

  bool has_add_other = false;
  if (consumer_mode == static_cast<unsigned>(ROW_CONSUMER_ADD_STORE)) {
    if (!resolve_tensor_read(consumer.add_other_tensor, add_other)) {
      return ERR_TENSOR_DESC_RANGE;
    }
    if (!param_dma_get_add_qparam(consumer.add_qparam_id, add_qparam)) {
      return ERR_PARAM_DESC_RANGE;
    }
    has_add_other = true;
  } else if (consumer_mode == static_cast<unsigned>(ROW_CONSUMER_ADD_AFFINE_STORE)) {
    return ERR_UNSUPPORTED_OPCODE;
  }

  if (emit_fullres_mask) {
    if (cfg.out_c.to_uint() != static_cast<unsigned>(ENCODER_OUT_C) ||
        out_h.to_uint() != static_cast<unsigned>(ENCODER_OUT_H) ||
        out_w.to_uint() != static_cast<unsigned>(ENCODER_OUT_W)) {
      return ERR_UNSUPPORTED_OPCODE;
    }
    dst = tensor_desc_t();
  } else {
    const u16_t dst_required_c =
        static_cast<u16_t>(consumer.store_c_offset + consumer.valid_c);
    if (!resolve_tensor_write(consumer.store_dst_tensor, out_h, out_w, dst_required_c, dst)) {
      return ERR_TENSOR_DESC_RANGE;
    }
    if (tensor_is_global(consumer.store_dst_tensor) &&
        dst.c.to_uint() < consumer.store_c_offset.to_uint() + consumer.valid_c.to_uint()) {
      return ERR_TENSOR_DESC_RANGE;
    }
  }

  bool conv_ok = false;
  execute_conv_stream_datapath(src,
                               dst,
                               add_other,
                               has_add_other,
                               cfg,
                               conv_desc,
                               sched,
                               consumer,
                               qparam,
                               add_qparam,
                               gmem_frame_out,
                               conv_ok);
  if (!conv_ok) {
    return ERR_BANK_OVERFLOW;
  }

  if (!emit_fullres_mask) {
    if (consumer_mode == static_cast<unsigned>(ROW_CONSUMER_STORE) &&
        conv_desc.dst_tensor.to_uint() != static_cast<unsigned>(TID_INVALID) &&
        conv_desc.dst_tensor.to_uint() != consumer.store_dst_tensor.to_uint() &&
        !alias_tensor_to_slice(conv_desc.dst_tensor,
                               dst,
                               consumer.store_c_offset,
                               consumer.valid_c)) {
      return ERR_TENSOR_DESC_RANGE;
    }
    if (consumer.alias_tensor.to_uint() != static_cast<unsigned>(TID_INVALID) &&
        !alias_tensor_to_slice(consumer.alias_tensor,
                               dst,
                               consumer.store_c_offset,
                               consumer.valid_c)) {
      return ERR_TENSOR_DESC_RANGE;
    }
  }

  return ERR_NONE;
}

static error_code_t run_scheduled_affine_op(const uop_t& uop) {
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

static error_code_t run_scheduled_add_affine_op(const uop_t& add_uop,
                                                const uop_t& affine_uop) {
#pragma HLS INLINE off
  tensor_desc_t src_a;
  tensor_desc_t src_b;
  tensor_desc_t dst;
  add_q_t add_qparam;

  if (add_uop.opcode.to_uint() != static_cast<unsigned>(UOP_ADD) ||
      affine_uop.opcode.to_uint() != static_cast<unsigned>(UOP_AFFINE) ||
      add_uop.dst_tensor.to_uint() != affine_uop.src0_tensor.to_uint()) {
    return ERR_UOP_DECODE;
  }

  select_scratch_region(affine_uop, affine_uop.in_h);
  if (!resolve_tensor_read(add_uop.src0_tensor, src_a) ||
      !resolve_tensor_read(add_uop.src1_tensor, src_b) ||
      !resolve_tensor_write(affine_uop.dst_tensor, src_a.h, src_a.w, src_a.c, dst)) {
    return ERR_TENSOR_DESC_RANGE;
  }
  if (!same_tensor_shape(src_a, src_b) || !same_tensor_shape(src_a, dst)) {
    return ERR_TENSOR_DESC_RANGE;
  }
  if (!param_dma_get_add_qparam(add_uop.param_id, add_qparam)) {
    return ERR_PARAM_DESC_RANGE;
  }

  const u16_t valid_c = (affine_uop.valid_c.to_uint() == 0U) ? src_a.c : affine_uop.valid_c;
  if (valid_c.to_uint() > src_a.c.to_uint() ||
      valid_c.to_uint() > src_b.c.to_uint() ||
      valid_c.to_uint() > dst.c.to_uint()) {
    return ERR_TENSOR_DESC_RANGE;
  }

  const int h_count = static_cast<int>(src_a.h.to_uint());
  const int w_count = static_cast<int>(src_a.w.to_uint());
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
        aff_q_t aff_qparam;
        act_vec_t a_packed = 0;
        act_vec_t b_packed = 0;
        act_vec_t out_packed = 0;
        if (!param_dma_get_affine_qparam(affine_uop.param_id, block_id, aff_qparam)) {
          return ERR_PARAM_DESC_RANGE;
        }

        const bool full_a = is_full_tile_lanes(lanes) && can_use_aligned_full_tile(src_a, c);
        const bool full_b = is_full_tile_lanes(lanes) && can_use_aligned_full_tile(src_b, c);
        const bool full_write = is_full_tile_lanes(lanes) && can_use_aligned_full_tile(dst, c);
        const bool read_a_ok =
            full_a ? on_chip_memory_read_aligned_full_tile(src_a,
                                                           static_cast<i32_t>(h),
                                                           static_cast<i32_t>(w),
                                                           c,
                                                           a_packed)
                   : on_chip_memory_read_packed_tile(src_a,
                                                     static_cast<i32_t>(h),
                                                     static_cast<i32_t>(w),
                                                     c,
                                                     lanes,
                                                     a_packed);
        const bool read_b_ok =
            full_b ? on_chip_memory_read_aligned_full_tile(src_b,
                                                           static_cast<i32_t>(h),
                                                           static_cast<i32_t>(w),
                                                           c,
                                                           b_packed)
                   : on_chip_memory_read_packed_tile(src_b,
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
            const i8_t added = add_i8(a_value, b_value, add_qparam);
            const i8_t out_value = affine_i8_to_i8(added,
                                                   aff_qparam.mul[lane],
                                                   aff_qparam.bias[lane],
                                                   aff_qparam.shift[lane],
                                                   affine_uop.act_type);
            set_act_vec_i8_dynamic(out_packed, lane, out_value);
          }
        }

        const bool write_ok =
            full_write ? on_chip_memory_write_aligned_full_tile(dst, h, w, c, out_packed)
                       : p6_write_tensor_slice_narrow(dst, h, w, c, lanes, out_packed);
        if (!write_ok) {
          return ERR_BANK_OVERFLOW;
        }
      }
    }
  }

  return ERR_NONE;
}

static bool same_tensor_hw(const tensor_desc_t& a, const tensor_desc_t& b) {
#pragma HLS INLINE
  return a.h.to_uint() == b.h.to_uint() && a.w.to_uint() == b.w.to_uint();
}

static bool tensor_slice_already_in_place(const tensor_desc_t& src,
                                          const tensor_desc_t& dst,
                                          u16_t c_offset,
                                          u16_t valid_c) {
#pragma HLS INLINE
  const unsigned dst_begin = dst.reserved1.to_uint() + c_offset.to_uint();
  return src.bank_id.to_uint() == dst.bank_id.to_uint() &&
         src.base_offset.to_uint() == dst.base_offset.to_uint() &&
         same_tensor_hw(src, dst) &&
         p6_desc_phys_c(src).to_uint() == p6_desc_phys_c(dst).to_uint() &&
         src.reserved1.to_uint() == dst_begin &&
         src.c.to_uint() >= valid_c.to_uint() &&
         dst.c.to_uint() >= c_offset.to_uint() + valid_c.to_uint();
}

static bool copy_tensor_slice_fixed(const tensor_desc_t& src,
                                    const tensor_desc_t& dst,
                                    u16_t c_offset,
                                    u16_t valid_c) {
#pragma HLS INLINE off
  if (!same_tensor_hw(src, dst) ||
      valid_c.to_uint() == 0U ||
      src.c.to_uint() < valid_c.to_uint() ||
      dst.c.to_uint() < c_offset.to_uint() + valid_c.to_uint()) {
    return false;
  }
  if (tensor_slice_already_in_place(src, dst, c_offset, valid_c)) {
    return true;
  }

  const int h_count = static_cast<int>(dst.h.to_uint());
  const int w_count = static_cast<int>(dst.w.to_uint());
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
        act_vec_t packed = 0;
        if (!on_chip_memory_read_packed_tile(src,
                                             static_cast<i32_t>(h),
                                             static_cast<i32_t>(w),
                                             c,
                                             lanes,
                                             packed)) {
          return false;
        }
        if (!p6_write_tensor_slice_narrow(dst,
                                          h,
                                          w,
                                          static_cast<u16_t>(c_offset + c),
                                          lanes,
                                          packed)) {
          return false;
        }
      }
    }
  }
  return true;
}

static error_code_t run_scheduled_store_op(const uop_t& uop) {
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
  if (!copy_tensor_slice_fixed(src, dst, uop.c_offset, valid_c)) {
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
  if (header.magic != PARAM_BLOB_MAGIC) {
    return ERR_BAD_BLOB;
  }
  if (header.version.to_uint() != PARAM_BLOB_VERSION_SCHED) {
    return ERR_BAD_BLOB;
  }
  if (header.uop_count.to_uint() != static_cast<unsigned>(UOP_COUNT_ENCODER)) {
    return ERR_UOP_DECODE;
  }
  if (header.tensor_desc_count > MAX_TENSOR_DESC_COUNT ||
      header.scale_desc_count > SCALE_DESC_COUNT_MAX) {
    return ERR_TENSOR_DESC_RANGE;
  }
  if (header.conv_desc_count.to_uint() > MAX_CONV_EXEC_DESC_COUNT ||
      header.affine_desc_count > MAX_AFFINE_PARAM_DESC_COUNT ||
      header.add_desc_count > MAX_ADD_PARAM_DESC_COUNT ||
      header.pool_desc_count > MAX_POOL_PARAM_DESC_COUNT) {
    return ERR_PARAM_DESC_RANGE;
  }
  if (!is_aligned_section_offset(header.tensor_desc_offset) ||
      !is_aligned_section_offset(header.scale_desc_offset) ||
      !is_aligned_section_offset(header.uop_offset) ||
      !is_aligned_section_offset(header.weight_data_offset) ||
      !is_aligned_section_offset(header.conv_qparam_offset) ||
      !is_aligned_section_offset(header.affine_qparam_offset) ||
      !is_aligned_section_offset(header.add_qparam_offset) ||
      !is_aligned_section_offset(header.pool_qparam_offset)) {
    return ERR_PARAM_DESC_RANGE;
  }
  if (!is_aligned_section_offset(header.reserved1[0]) ||
      !is_aligned_section_offset(header.reserved1[1]) ||
      !is_aligned_section_offset(header.reserved1[2]) ||
      !is_aligned_section_offset(header.reserved1[3]) ||
      !is_aligned_section_offset(header.reserved1[5])) {
    return ERR_PARAM_DESC_RANGE;
  }

  return ERR_NONE;
}

static void core_mode_init(const axi_vec_t* gmem_param) {
#pragma HLS INLINE off
  reset_scratch_state();
  s_param_ready = false;
#ifndef __SYNTHESIS__
  s_csim_tensor_dumped = false;
#endif

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

static bool build_legacy_uop_for_fixed_op(unsigned uop_id, uop_t& uop) {
#pragma HLS INLINE off
  switch (uop_id) {
#define ESP_INT8_BUILD_CASE(ID) \
    case ID:                    \
      StaticUop<ID>::make(uop); \
      return true
    ESP_INT8_BUILD_CASE(1U);
    ESP_INT8_BUILD_CASE(3U);
    ESP_INT8_BUILD_CASE(4U);
    ESP_INT8_BUILD_CASE(5U);
    ESP_INT8_BUILD_CASE(6U);
    ESP_INT8_BUILD_CASE(20U);
    ESP_INT8_BUILD_CASE(34U);
    ESP_INT8_BUILD_CASE(35U);
    ESP_INT8_BUILD_CASE(36U);
    ESP_INT8_BUILD_CASE(37U);
    ESP_INT8_BUILD_CASE(38U);
    ESP_INT8_BUILD_CASE(39U);
    ESP_INT8_BUILD_CASE(53U);
    ESP_INT8_BUILD_CASE(67U);
    ESP_INT8_BUILD_CASE(68U);
    ESP_INT8_BUILD_CASE(69U);
    ESP_INT8_BUILD_CASE(70U);
    ESP_INT8_BUILD_CASE(71U);
#undef ESP_INT8_BUILD_CASE
    default:
      uop = uop_t();
      return false;
  }
}

static bool csim_stop_before_logical_uop(unsigned uop_id) {
#pragma HLS INLINE
#ifdef ESP_INT8_CSIM_MAX_UOP
  return uop_id > static_cast<unsigned>(ESP_INT8_CSIM_MAX_UOP);
#else
  return false;
#endif
}

static error_code_t run_scheduled_fixed_op(const exec_plan_entry_t& entry) {
#pragma HLS INLINE off
  uop_t uop;
  if (!build_legacy_uop_for_fixed_op(entry.logical_uop_id.to_uint(), uop)) {
    return ERR_UOP_DECODE;
  }
  switch (entry.kind.to_uint()) {
    case static_cast<unsigned>(EXEC_POOL):
      if (uop.opcode.to_uint() != static_cast<unsigned>(UOP_POOL)) {
        return ERR_UOP_DECODE;
      }
      return run_scheduled_pool_op(uop);
    case static_cast<unsigned>(EXEC_AFFINE):
      if (uop.opcode.to_uint() != static_cast<unsigned>(UOP_AFFINE)) {
        return ERR_UOP_DECODE;
      }
      return run_scheduled_affine_op(uop);
    case static_cast<unsigned>(EXEC_ADD_AFFINE): {
      if (uop.opcode.to_uint() != static_cast<unsigned>(UOP_AFFINE) ||
          entry.logical_uop_id.to_uint() == 0U) {
        return ERR_UOP_DECODE;
      }
      uop_t add_uop;
      if (!build_legacy_uop_for_fixed_op(entry.logical_uop_id.to_uint() - 1U, add_uop)) {
        return ERR_UOP_DECODE;
      }
      return run_scheduled_add_affine_op(add_uop, uop);
    }
    case static_cast<unsigned>(EXEC_STORE):
      if (uop.opcode.to_uint() != static_cast<unsigned>(UOP_STORE)) {
        return ERR_UOP_DECODE;
      }
      return run_scheduled_store_op(uop);
    case static_cast<unsigned>(EXEC_NOP):
      return ERR_NONE;
    default:
      return ERR_UNSUPPORTED_OPCODE;
  }
}

static error_code_t run_scheduled_graph(axi_vec_t* gmem_frame_out) {
#pragma HLS INLINE off
  for (int pc = 0; pc < MAX_EXEC_PLAN_COUNT; ++pc) {
#pragma HLS PIPELINE off
    exec_plan_entry_t entry;
    if (!param_dma_get_exec_entry(static_cast<u8_t>(pc), entry)) {
      return ERR_UOP_DECODE;
    }
    const unsigned logical_uop = entry.logical_uop_id.to_uint();
#ifndef __SYNTHESIS__
    s_csim_last_uop = logical_uop;
#endif
    if (csim_stop_before_logical_uop(logical_uop)) {
      return ERR_NONE;
    }

    const unsigned kind = entry.kind.to_uint();
    if (kind == static_cast<unsigned>(EXEC_END)) {
#ifndef __SYNTHESIS__
      s_csim_last_error = ERR_NONE;
#endif
      return ERR_NONE;
    }

    if (!csim_dump_tensor_set_pre(logical_uop)) {
#ifndef __SYNTHESIS__
      s_csim_last_error = ERR_BANK_OVERFLOW;
#endif
      return ERR_BANK_OVERFLOW;
    }

    error_code_t err = ERR_NONE;
    if (kind == static_cast<unsigned>(EXEC_CONV)) {
      err = run_scheduled_conv_op(entry.desc_id, gmem_frame_out);
    } else if (kind == static_cast<unsigned>(EXEC_POOL) ||
               kind == static_cast<unsigned>(EXEC_AFFINE) ||
               kind == static_cast<unsigned>(EXEC_ADD_AFFINE) ||
               kind == static_cast<unsigned>(EXEC_STORE) ||
               kind == static_cast<unsigned>(EXEC_NOP)) {
      err = run_scheduled_fixed_op(entry);
    } else {
      err = ERR_UNSUPPORTED_OPCODE;
    }

    if (err != ERR_NONE) {
#ifndef __SYNTHESIS__
      s_csim_last_error = err;
#endif
      return err;
    }
    if (!csim_dump_tensor_set_post(logical_uop)) {
#ifndef __SYNTHESIS__
      s_csim_last_error = ERR_BANK_OVERFLOW;
#endif
      return ERR_BANK_OVERFLOW;
    }
  }
#ifndef __SYNTHESIS__
  s_csim_last_error = ERR_UOP_DECODE;
#endif
  return ERR_UOP_DECODE;
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
  if (!param_dma_is_schedule_blob()) {
#ifndef __SYNTHESIS__
    s_csim_last_error = ERR_BAD_BLOB;
#endif
    return ERR_BAD_BLOB;
  }

  frame_dma_load(gmem_frame_in);

  const error_code_t err = run_scheduled_graph(gmem_frame_out);
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
#pragma HLS INTERFACE ap_memory port=gmem_param depth=8192
#pragma HLS INTERFACE ap_none port=mode
#pragma HLS INTERFACE ap_none port=uop_count
#pragma HLS INTERFACE ap_ctrl_hs port=return
#else
#pragma HLS INTERFACE m_axi port=gmem_frame_in offset=slave bundle=gmem0 depth=49152 max_read_burst_length=64 num_read_outstanding=4
#pragma HLS INTERFACE m_axi port=gmem_frame_out offset=slave bundle=gmem1 depth=16384 max_write_burst_length=64 num_write_outstanding=4
#pragma HLS INTERFACE m_axi port=gmem_param offset=slave bundle=gmem2 depth=8192 max_read_burst_length=64 num_read_outstanding=4
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
