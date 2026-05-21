#include <cstdint>

#include "../include/npu_config.hpp"
#include "../include/npu_q.hpp"
#include "../include/npu_uop.hpp"

namespace esp_int8 {

void param_dma_init(const axi_vec_t* gmem_param);
bool param_dma_ready();
error_code_t param_dma_error();
bool param_dma_get_tensor_desc(u8_t tensor_id, tensor_desc_t& desc);
bool param_dma_get_uop(u16_t uop_id, uop_t& uop);
bool param_dma_get_pool_qparam(u8_t param_id, pool_q_t& qparam);
bool param_dma_get_conv_qparam(u8_t param_id, conv_q_t& qparam);
bool param_dma_get_weight_vec(u8_t param_id, u16_t oc, u16_t kt, const conv_cfg_t& cfg, wgt_vec_t& word);
bool param_dma_get_affine_qparam(u8_t param_id, u8_t block_id, aff_q_t& qparam);
bool param_dma_get_add_qparam(u8_t param_id, add_q_t& qparam);
void instruction_fetch_decode(const axi_vec_t* gmem_param, u32_t uop_count);
error_code_t if_dec_error();
u16_t if_dec_current_uop_id();
void frame_dma_load(const axi_vec_t* gmem_frame_in);
void frame_dma_store(axi_vec_t* gmem_frame_out);
bool avgpool_unit_checked(const tensor_desc_t& src,
                          const tensor_desc_t& dst,
                          const pool_q_t& qparam,
                          i8_t* fmbuf_base);
bool concat_writer(const tensor_desc_t& src,
                   const tensor_desc_t& dst,
                   u16_t c_offset,
                   u16_t valid_c);
bool on_chip_memory_read_tile(const tensor_desc_t& desc,
                              i32_t h,
                              i32_t w,
                              u16_t c_begin,
                              u8_t valid_c,
                              i8_t tile[TM]);
bool on_chip_memory_write_tile(const tensor_desc_t& desc,
                               u16_t h,
                               u16_t w,
                               u16_t c_begin,
                               u8_t valid_c,
                               const i8_t tile[TM]);
bool on_chip_memory_write_packed_tile(const tensor_desc_t& desc,
                                      u16_t h,
                                      u16_t w,
                                      u16_t c_begin,
                                      u8_t valid_c,
                                      act_vec_t packed);
bool on_chip_memory_write_aligned_row_word(const tensor_desc_t& desc,
                                           u16_t h,
                                           u32_t row_byte_offset,
                                           act_vec_t packed);
void window_generator_row(const tensor_desc_t& src_desc,
                          hls::stream<act_vec_t>& act_stream,
                          const conv_cfg_t& cfg,
                          u16_t out_row);
void systolic_array_core_row(hls::stream<act_vec_t>& act_stream,
                             hls::stream<wgt_vec_t>& wgt_stream,
                             hls::stream<psum_vec_t>& psum_stream,
                             const conv_cfg_t& cfg);

static param_blob_header_t s_param_header;
static core_status_t s_status;
static bool s_param_ready = false;
static tensor_desc_t s_scratch_desc[4];
static bool s_scratch_valid[4] = {false, false, false, false};
static unsigned s_scratch_region = 0;
static u32_t s_scratch_base = 0;
static u32_t s_scratch_slot_bytes = 0;
static u16_t s_scratch_phys_c = 0;
static u16_t s_scratch_channel_base = 0;
static u16_t s_scratch_channel_slot = 0;
static bool s_scratch_channel_view = false;

enum scratch_region_t : unsigned {
  SCRATCH_REGION_NONE = 0,
  SCRATCH_REGION_L20 = 1,
  SCRATCH_REGION_L2B0 = 2,
  SCRATCH_REGION_L30 = 3,
  SCRATCH_REGION_L3B0 = 4,
};

static u32_t axi_lane_u32(const axi_vec_t& word, int lane) {
#pragma HLS INLINE
  return word.range(lane * 32 + 31, lane * 32);
}

static bool is_aligned_section_offset(u32_t offset) {
#pragma HLS INLINE
  return (offset & (SECTION_ALIGNMENT_BYTES - 1)) == 0;
}

static u16_t error_code_to_u16(error_code_t err) {
#pragma HLS INLINE
  return static_cast<u16_t>(static_cast<unsigned>(err));
}

static void clear_status() {
#pragma HLS INLINE
  s_status.error_code = error_code_to_u16(ERR_NONE);
  s_status.current_uop_id = 0;
}

static void set_error(error_code_t err, u16_t uop_id) {
#pragma HLS INLINE
  s_status.error_code = error_code_to_u16(err);
  s_status.current_uop_id = uop_id;
}

static std::uint32_t runtime_mode(std::uint32_t mode) {
#pragma HLS INLINE
  return mode & RUNTIME_MODE_MASK;
}

static bool global_tensor_desc(u8_t tensor_id, tensor_desc_t& desc) {
#pragma HLS INLINE
  if (!tensor_is_global(tensor_id)) {
    desc = tensor_desc_t();
    return false;
  }
  return param_dma_get_tensor_desc(tensor_id, desc);
}

static bool scratch_index(u8_t tensor_id, unsigned& idx) {
#pragma HLS INLINE
  switch (tensor_id.to_uint()) {
    case static_cast<unsigned>(LS_C1):
      idx = 0;
      return true;
    case static_cast<unsigned>(LS_A):
      idx = 1;
      return true;
    case static_cast<unsigned>(LS_B):
      idx = 2;
      return true;
    case static_cast<unsigned>(LS_TMP):
      idx = 3;
      return true;
    default:
      idx = 0;
      return false;
  }
}

static void invalidate_scratch() {
#pragma HLS INLINE
  for (int i = 0; i < 4; ++i) {
#pragma HLS UNROLL
    s_scratch_valid[i] = false;
    s_scratch_desc[i] = tensor_desc_t();
  }
}

static void reset_scratch_state() {
#pragma HLS INLINE
  s_scratch_region = SCRATCH_REGION_NONE;
  s_scratch_base = 0;
  s_scratch_slot_bytes = 0;
  s_scratch_phys_c = 0;
  s_scratch_channel_base = 0;
  s_scratch_channel_slot = 0;
  s_scratch_channel_view = false;
  invalidate_scratch();
}

static void set_scratch_region(unsigned region) {
#pragma HLS INLINE
  u32_t next_base = 0;
  u32_t next_slot = 0;
  u16_t next_phys_c = 0;
  u16_t next_ch_base = 0;
  u16_t next_ch_slot = 0;
  bool next_view = false;

  switch (region) {
    case SCRATCH_REGION_L20:
      next_base = FMBUF_L2_SCRATCH_BASE;
      next_slot = FMBUF_L2_SCRATCH_SLOT_BYTES;
      next_phys_c = FMBUF_L20_PHYS_C;
      next_ch_base = 0;
      next_ch_slot = 16;
      break;
    case SCRATCH_REGION_L2B0:
      next_base = FMBUF_L2_SCRATCH_BASE;
      next_slot = FMBUF_L2_SCRATCH_SLOT_BYTES;
      break;
    case SCRATCH_REGION_L30:
      next_base = FMBUF_L30_SCRATCH_C1_BASE;
      next_slot = FMBUF_L30_SCRATCH_SLOT_BYTES;
      break;
    case SCRATCH_REGION_L3B0:
      next_base = 0x000000U;
      next_phys_c = 256;
      next_ch_base = 0;
      next_ch_slot = 32;
      next_view = true;
      break;
    default:
      next_base = 0;
      next_slot = 0;
      break;
  }

  if (region != s_scratch_region ||
      next_base != s_scratch_base ||
      next_slot != s_scratch_slot_bytes ||
      next_phys_c != s_scratch_phys_c ||
      next_ch_base != s_scratch_channel_base ||
      next_ch_slot != s_scratch_channel_slot ||
      next_view != s_scratch_channel_view) {
    s_scratch_region = region;
    s_scratch_base = next_base;
    s_scratch_slot_bytes = next_slot;
    s_scratch_phys_c = next_phys_c;
    s_scratch_channel_base = next_ch_base;
    s_scratch_channel_slot = next_ch_slot;
    s_scratch_channel_view = next_view;
    invalidate_scratch();
  }
}

static bool make_contiguous_scratch_desc(unsigned idx,
                                         u32_t base,
                                         u32_t slot_bytes,
                                         u16_t h,
                                         u16_t w,
                                         u16_t c,
                                         tensor_desc_t& desc) {
#pragma HLS INLINE
  const u32_t bytes = static_cast<u32_t>(h) * static_cast<u32_t>(w) * static_cast<u32_t>(c);
  if (slot_bytes.to_uint() == 0U || bytes > slot_bytes) {
    desc = tensor_desc_t();
    return false;
  }

  desc.bank_id = static_cast<u8_t>(static_cast<unsigned>(BANK_FMEM0));
  desc.elem_bytes = 1;
  desc.h = h;
  desc.w = w;
  desc.c = c;
  desc.reserved0 = c;
  desc.base_offset = base + static_cast<u32_t>(idx) * slot_bytes;
  desc.reserved1 = 0;
  return true;
}

static bool make_channel_view_scratch_desc(unsigned idx,
                                           u32_t base,
                                           u16_t phys_c,
                                           u16_t channel_base,
                                           u16_t channel_slot,
                                           u16_t h,
                                           u16_t w,
                                           u16_t c,
                                           tensor_desc_t& desc) {
#pragma HLS INLINE
  const u16_t ch_offset =
      static_cast<u16_t>(channel_base + static_cast<u16_t>(idx) * channel_slot);
  if (c > channel_slot ||
      static_cast<unsigned>(ch_offset.to_uint() + c.to_uint()) > phys_c.to_uint()) {
    desc = tensor_desc_t();
    return false;
  }

  desc.bank_id = static_cast<u8_t>(static_cast<unsigned>(BANK_FMEM0));
  desc.elem_bytes = 1;
  desc.h = h;
  desc.w = w;
  desc.c = c;
  desc.reserved0 = phys_c;
  desc.base_offset = base;
  desc.reserved1 = ch_offset;
  return true;
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

static unsigned conv_scratch_region(const uop_t& uop, u16_t out_h) {
#pragma HLS INLINE
  const unsigned pid = uop.param_id.to_uint();
  if (pid >= 1U && pid <= 6U) {
    return SCRATCH_REGION_L20;
  }
  if (pid >= 7U && pid <= 12U) {
    return SCRATCH_REGION_L2B0;
  }
  if (pid >= 13U && pid <= 18U) {
    return SCRATCH_REGION_L30;
  }
  if (pid >= 19U && pid <= 24U) {
    return SCRATCH_REGION_L3B0;
  }
  if (out_h.to_uint() >= 128U) {
    return (uop.src0_tensor.to_uint() == static_cast<unsigned>(TID_B1_ACT))
               ? SCRATCH_REGION_L20
               : SCRATCH_REGION_L2B0;
  }
  return (uop.src0_tensor.to_uint() == static_cast<unsigned>(TID_B2_ACT))
             ? SCRATCH_REGION_L30
             : SCRATCH_REGION_L3B0;
}

static unsigned add_scratch_region(const uop_t& uop) {
#pragma HLS INLINE
  const unsigned pid = uop.param_id.to_uint();
  if (pid <= 2U) {
    return SCRATCH_REGION_L20;
  }
  if (pid <= 6U) {
    return SCRATCH_REGION_L2B0;
  }
  if (pid <= 9U) {
    return SCRATCH_REGION_L30;
  }
  return SCRATCH_REGION_L3B0;
}

static unsigned store_scratch_region(const uop_t& uop) {
#pragma HLS INLINE
  switch (uop.dst_tensor.to_uint()) {
    case static_cast<unsigned>(TID_L20_CAT):
      return SCRATCH_REGION_L20;
    case static_cast<unsigned>(TID_L2B0_CAT):
      return SCRATCH_REGION_L2B0;
    case static_cast<unsigned>(TID_L30_CAT):
      return SCRATCH_REGION_L30;
    case static_cast<unsigned>(TID_L3B0_CAT):
      return SCRATCH_REGION_L3B0;
    default:
      break;
  }
  if (uop.in_h.to_uint() >= 128U) {
    return (s_scratch_region == SCRATCH_REGION_L20) ? SCRATCH_REGION_L20 : SCRATCH_REGION_L2B0;
  }
  return (s_scratch_region == SCRATCH_REGION_L30) ? SCRATCH_REGION_L30 : SCRATCH_REGION_L3B0;
}

static unsigned scratch_region_for_uop(const uop_t& uop, u16_t out_h) {
#pragma HLS INLINE
  const unsigned opcode = uop.opcode.to_uint();
  if (opcode == static_cast<unsigned>(UOP_CONV)) {
    return conv_scratch_region(uop, out_h);
  }
  if (opcode == static_cast<unsigned>(UOP_ADD)) {
    return add_scratch_region(uop);
  }
  if (opcode == static_cast<unsigned>(UOP_STORE)) {
    return store_scratch_region(uop);
  }
  if (out_h.to_uint() >= 128U) {
    return s_scratch_region == SCRATCH_REGION_L2B0 ? SCRATCH_REGION_L2B0 : SCRATCH_REGION_L20;
  }
  return s_scratch_region == SCRATCH_REGION_L3B0 ? SCRATCH_REGION_L3B0 : SCRATCH_REGION_L30;
}

static bool uop_uses_scratch(const uop_t& uop) {
#pragma HLS INLINE
  return tensor_is_scratch(uop.src0_tensor) ||
         tensor_is_scratch(uop.src1_tensor) ||
         tensor_is_scratch(uop.dst_tensor);
}

static void select_scratch_region(const uop_t& uop, u16_t out_h) {
#pragma HLS INLINE
  if (uop_uses_scratch(uop)) {
    set_scratch_region(scratch_region_for_uop(uop, out_h));
  }
}

static bool make_scratch_desc(u8_t tensor_id, u16_t h, u16_t w, u16_t c, tensor_desc_t& desc) {
#pragma HLS INLINE
  unsigned idx = 0;
  if (!scratch_index(tensor_id, idx) || s_scratch_region == SCRATCH_REGION_NONE) {
    desc = tensor_desc_t();
    return false;
  }

  if (s_scratch_region == SCRATCH_REGION_L20) {
    if (idx == 0U) {
      if (!make_contiguous_scratch_desc(0,
                                        FMBUF_L2_SCRATCH_BASE,
                                        FMBUF_L2_SCRATCH_SLOT_BYTES,
                                        h,
                                        w,
                                        c,
                                        desc)) {
        return false;
      }
    } else {
      if (!make_channel_view_scratch_desc(idx - 1U,
                                          FMBUF_L20_BASE,
                                          FMBUF_L20_PHYS_C,
                                          0,
                                          16,
                                          h,
                                          w,
                                          c,
                                          desc)) {
        return false;
      }
    }
    s_scratch_desc[idx] = desc;
    s_scratch_valid[idx] = true;
    return true;
  }

  if (s_scratch_region == SCRATCH_REGION_L30) {
    if (idx == 0U) {
      if (!make_contiguous_scratch_desc(0,
                                        FMBUF_L30_SCRATCH_C1_BASE,
                                        FMBUF_L30_SCRATCH_SLOT_BYTES,
                                        h,
                                        w,
                                        c,
                                        desc)) {
        return false;
      }
    } else {
      if (!make_contiguous_scratch_desc(idx - 1U,
                                        FMBUF_L30_SCRATCH_LOW_BASE,
                                        FMBUF_L30_SCRATCH_SLOT_BYTES,
                                        h,
                                        w,
                                        c,
                                        desc)) {
        return false;
      }
    }
    s_scratch_desc[idx] = desc;
    s_scratch_valid[idx] = true;
    return true;
  }

  if (s_scratch_channel_view) {
    if (!make_channel_view_scratch_desc(idx,
                                        s_scratch_base,
                                        s_scratch_phys_c,
                                        s_scratch_channel_base,
                                        s_scratch_channel_slot,
                                        h,
                                        w,
                                        c,
                                        desc)) {
      return false;
    }
  } else {
    if (!make_contiguous_scratch_desc(idx,
                                      s_scratch_base,
                                      s_scratch_slot_bytes,
                                      h,
                                      w,
                                      c,
                                      desc)) {
      return false;
    }
  }

  s_scratch_desc[idx] = desc;
  s_scratch_valid[idx] = true;
  return true;
}

static bool scratch_tensor_desc(u8_t tensor_id, tensor_desc_t& desc) {
#pragma HLS INLINE
  unsigned idx = 0;
  if (!scratch_index(tensor_id, idx) || !s_scratch_valid[idx]) {
    desc = tensor_desc_t();
    return false;
  }
  desc = s_scratch_desc[idx];
  return true;
}

static bool resolve_tensor_read(u8_t tensor_id, tensor_desc_t& desc) {
#pragma HLS INLINE
  if (tensor_is_global(tensor_id)) {
    return global_tensor_desc(tensor_id, desc);
  }
  return scratch_tensor_desc(tensor_id, desc);
}

static bool resolve_tensor_write(u8_t tensor_id, u16_t h, u16_t w, u16_t c, tensor_desc_t& desc) {
#pragma HLS INLINE
  if (tensor_is_global(tensor_id)) {
    return global_tensor_desc(tensor_id, desc);
  }
  return make_scratch_desc(tensor_id, h, w, c, desc);
}

static u8_t tensor_lanes(u16_t remaining_c) {
#pragma HLS INLINE
  const unsigned rem = remaining_c.to_uint();
  return static_cast<u8_t>((rem > static_cast<unsigned>(TM)) ? TM : rem);
}

static bool same_tensor_shape(const tensor_desc_t& a, const tensor_desc_t& b) {
#pragma HLS INLINE
  return a.h.to_uint() == b.h.to_uint() &&
         a.w.to_uint() == b.w.to_uint() &&
         a.c.to_uint() == b.c.to_uint();
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

static void zero_affine_arrays(i32_t aff_mul[32], i32_t aff_bias[32], u8_t aff_shift[32]) {
#pragma HLS INLINE
  for (int i = 0; i < TM; ++i) {
#pragma HLS UNROLL
    aff_mul[i] = 1;
    aff_bias[i] = 0;
    aff_shift[i] = 0;
  }
}

static u16_t conv_effective_stride(const conv_cfg_t& cfg) {
#pragma HLS INLINE
  return (cfg.stride == 0) ? static_cast<u16_t>(1) : static_cast<u16_t>(cfg.stride);
}

static u16_t conv_effective_dilation(const conv_cfg_t& cfg) {
#pragma HLS INLINE
  return (cfg.dilation == 0) ? static_cast<u16_t>(1) : static_cast<u16_t>(cfg.dilation);
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

template <int DST_LANE, int SRC_LANE, int COUNT>
static void copy_act_segment(act_vec_t& dst, const act_vec_t& src) {
#pragma HLS INLINE
  dst.range(DST_LANE * 8 + COUNT * 8 - 1, DST_LANE * 8) =
      src.range(SRC_LANE * 8 + COUNT * 8 - 1, SRC_LANE * 8);
}

static u16_t core_desc_phys_c(const tensor_desc_t& desc) {
#pragma HLS INLINE
  return (desc.reserved0.to_uint() == 0U) ? desc.c : desc.reserved0;
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

static bool store_compact_c12_row(const tensor_desc_t& dst,
                                  u16_t out_row,
                                  int out_w_i,
                                  act_vec_t row_buf[MAX_FM_W]) {
#pragma HLS INLINE off
  bool ok = true;
  for (int group = 0; group < MAX_FM_W / 8; ++group) {
    const int pix = group * 8;
    if (pix >= out_w_i) {
      break;
    }
    act_vec_t p0 = row_buf[pix + 0];
    act_vec_t p1 = row_buf[pix + 1];
    act_vec_t p2 = row_buf[pix + 2];
    act_vec_t p3 = row_buf[pix + 3];
    act_vec_t p4 = row_buf[pix + 4];
    act_vec_t p5 = row_buf[pix + 5];
    act_vec_t p6 = row_buf[pix + 6];
    act_vec_t p7 = row_buf[pix + 7];

    act_vec_t w0 = 0;
    copy_act_segment<0, 0, 12>(w0, p0);
    copy_act_segment<12, 0, 12>(w0, p1);
    copy_act_segment<24, 0, 8>(w0, p2);

    act_vec_t w1 = 0;
    copy_act_segment<0, 8, 4>(w1, p2);
    copy_act_segment<4, 0, 12>(w1, p3);
    copy_act_segment<16, 0, 12>(w1, p4);
    copy_act_segment<28, 0, 4>(w1, p5);

    act_vec_t w2 = 0;
    copy_act_segment<0, 4, 8>(w2, p5);
    copy_act_segment<8, 0, 12>(w2, p6);
    copy_act_segment<20, 0, 12>(w2, p7);

    const u32_t byte_offset = static_cast<u32_t>(group) * static_cast<u32_t>(96);
    if (!on_chip_memory_write_aligned_row_word(dst, out_row, byte_offset, w0)) {
      ok = false;
    }
    if (!on_chip_memory_write_aligned_row_word(
            dst, out_row, byte_offset + static_cast<u32_t>(32), w1)) {
      ok = false;
    }
    if (!on_chip_memory_write_aligned_row_word(
            dst, out_row, byte_offset + static_cast<u32_t>(64), w2)) {
      ok = false;
    }
  }
  return ok;
}

static bool store_compact_c16_row(const tensor_desc_t& dst,
                                  u16_t out_row,
                                  int out_w_i,
                                  act_vec_t row_buf[MAX_FM_W]) {
#pragma HLS INLINE off
  bool ok = true;
  for (int group = 0; group < MAX_FM_W / 2; ++group) {
#pragma HLS PIPELINE II=1
    const int pix = group * 2;
    if (pix >= out_w_i) {
      break;
    }
    act_vec_t word = 0;
    copy_act_segment<0, 0, 16>(word, row_buf[pix + 0]);
    copy_act_segment<16, 0, 16>(word, row_buf[pix + 1]);
    const u32_t byte_offset = static_cast<u32_t>(group) * static_cast<u32_t>(32);
    if (!on_chip_memory_write_aligned_row_word(dst, out_row, byte_offset, word)) {
      ok = false;
    }
  }
  return ok;
}

static void write_compact_row_word(const tensor_desc_t& dst,
                                   u16_t out_row,
                                   u32_t byte_offset,
                                   act_vec_t word,
                                   bool& ok) {
#pragma HLS INLINE
  if (!on_chip_memory_write_aligned_row_word(dst, out_row, byte_offset, word)) {
    ok = false;
  }
}

static bool store_compact_c2_row(const tensor_desc_t& dst,
                                 u16_t out_row,
                                 int out_w_i,
                                 act_vec_t row_buf[MAX_FM_W]) {
#pragma HLS INLINE off
  bool ok = true;
  for (int group = 0; group < MAX_FM_W / 16; ++group) {
#pragma HLS PIPELINE II=1
    const int pix = group * 16;
    if (pix >= out_w_i) {
      break;
    }
    act_vec_t word = 0;
    copy_act_segment<0, 0, 2>(word, row_buf[pix + 0]);
    copy_act_segment<2, 0, 2>(word, row_buf[pix + 1]);
    copy_act_segment<4, 0, 2>(word, row_buf[pix + 2]);
    copy_act_segment<6, 0, 2>(word, row_buf[pix + 3]);
    copy_act_segment<8, 0, 2>(word, row_buf[pix + 4]);
    copy_act_segment<10, 0, 2>(word, row_buf[pix + 5]);
    copy_act_segment<12, 0, 2>(word, row_buf[pix + 6]);
    copy_act_segment<14, 0, 2>(word, row_buf[pix + 7]);
    copy_act_segment<16, 0, 2>(word, row_buf[pix + 8]);
    copy_act_segment<18, 0, 2>(word, row_buf[pix + 9]);
    copy_act_segment<20, 0, 2>(word, row_buf[pix + 10]);
    copy_act_segment<22, 0, 2>(word, row_buf[pix + 11]);
    copy_act_segment<24, 0, 2>(word, row_buf[pix + 12]);
    copy_act_segment<26, 0, 2>(word, row_buf[pix + 13]);
    copy_act_segment<28, 0, 2>(word, row_buf[pix + 14]);
    copy_act_segment<30, 0, 2>(word, row_buf[pix + 15]);
    const u32_t byte_offset = static_cast<u32_t>(group) * static_cast<u32_t>(32);
    write_compact_row_word(dst, out_row, byte_offset, word, ok);
  }
  return ok;
}

static bool store_compact_c25_row(const tensor_desc_t& dst,
                                  u16_t out_row,
                                  int out_w_i,
                                  act_vec_t row_buf[MAX_FM_W]) {
#pragma HLS INLINE off
  bool ok = true;
  for (int group = 0; group < MAX_FM_W / 32; ++group) {
    const int pix = group * 32;
    if (pix >= out_w_i) {
      break;
    }
    const u32_t byte_offset = static_cast<u32_t>(group) * static_cast<u32_t>(800);

    act_vec_t w0 = 0;
    copy_act_segment<0, 0, 25>(w0, row_buf[pix + 0]);
    copy_act_segment<25, 0, 7>(w0, row_buf[pix + 1]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(0), w0, ok);
    act_vec_t w1 = 0;
    copy_act_segment<0, 7, 18>(w1, row_buf[pix + 1]);
    copy_act_segment<18, 0, 14>(w1, row_buf[pix + 2]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(32), w1, ok);
    act_vec_t w2 = 0;
    copy_act_segment<0, 14, 11>(w2, row_buf[pix + 2]);
    copy_act_segment<11, 0, 21>(w2, row_buf[pix + 3]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(64), w2, ok);
    act_vec_t w3 = 0;
    copy_act_segment<0, 21, 4>(w3, row_buf[pix + 3]);
    copy_act_segment<4, 0, 25>(w3, row_buf[pix + 4]);
    copy_act_segment<29, 0, 3>(w3, row_buf[pix + 5]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(96), w3, ok);
    act_vec_t w4 = 0;
    copy_act_segment<0, 3, 22>(w4, row_buf[pix + 5]);
    copy_act_segment<22, 0, 10>(w4, row_buf[pix + 6]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(128), w4, ok);
    act_vec_t w5 = 0;
    copy_act_segment<0, 10, 15>(w5, row_buf[pix + 6]);
    copy_act_segment<15, 0, 17>(w5, row_buf[pix + 7]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(160), w5, ok);
    act_vec_t w6 = 0;
    copy_act_segment<0, 17, 8>(w6, row_buf[pix + 7]);
    copy_act_segment<8, 0, 24>(w6, row_buf[pix + 8]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(192), w6, ok);
    act_vec_t w7 = 0;
    copy_act_segment<0, 24, 1>(w7, row_buf[pix + 8]);
    copy_act_segment<1, 0, 25>(w7, row_buf[pix + 9]);
    copy_act_segment<26, 0, 6>(w7, row_buf[pix + 10]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(224), w7, ok);
    act_vec_t w8 = 0;
    copy_act_segment<0, 6, 19>(w8, row_buf[pix + 10]);
    copy_act_segment<19, 0, 13>(w8, row_buf[pix + 11]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(256), w8, ok);
    act_vec_t w9 = 0;
    copy_act_segment<0, 13, 12>(w9, row_buf[pix + 11]);
    copy_act_segment<12, 0, 20>(w9, row_buf[pix + 12]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(288), w9, ok);
    act_vec_t w10 = 0;
    copy_act_segment<0, 20, 5>(w10, row_buf[pix + 12]);
    copy_act_segment<5, 0, 25>(w10, row_buf[pix + 13]);
    copy_act_segment<30, 0, 2>(w10, row_buf[pix + 14]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(320), w10, ok);
    act_vec_t w11 = 0;
    copy_act_segment<0, 2, 23>(w11, row_buf[pix + 14]);
    copy_act_segment<23, 0, 9>(w11, row_buf[pix + 15]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(352), w11, ok);
    act_vec_t w12 = 0;
    copy_act_segment<0, 9, 16>(w12, row_buf[pix + 15]);
    copy_act_segment<16, 0, 16>(w12, row_buf[pix + 16]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(384), w12, ok);
    act_vec_t w13 = 0;
    copy_act_segment<0, 16, 9>(w13, row_buf[pix + 16]);
    copy_act_segment<9, 0, 23>(w13, row_buf[pix + 17]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(416), w13, ok);
    act_vec_t w14 = 0;
    copy_act_segment<0, 23, 2>(w14, row_buf[pix + 17]);
    copy_act_segment<2, 0, 25>(w14, row_buf[pix + 18]);
    copy_act_segment<27, 0, 5>(w14, row_buf[pix + 19]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(448), w14, ok);
    act_vec_t w15 = 0;
    copy_act_segment<0, 5, 20>(w15, row_buf[pix + 19]);
    copy_act_segment<20, 0, 12>(w15, row_buf[pix + 20]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(480), w15, ok);
    act_vec_t w16 = 0;
    copy_act_segment<0, 12, 13>(w16, row_buf[pix + 20]);
    copy_act_segment<13, 0, 19>(w16, row_buf[pix + 21]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(512), w16, ok);
    act_vec_t w17 = 0;
    copy_act_segment<0, 19, 6>(w17, row_buf[pix + 21]);
    copy_act_segment<6, 0, 25>(w17, row_buf[pix + 22]);
    copy_act_segment<31, 0, 1>(w17, row_buf[pix + 23]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(544), w17, ok);
    act_vec_t w18 = 0;
    copy_act_segment<0, 1, 24>(w18, row_buf[pix + 23]);
    copy_act_segment<24, 0, 8>(w18, row_buf[pix + 24]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(576), w18, ok);
    act_vec_t w19 = 0;
    copy_act_segment<0, 8, 17>(w19, row_buf[pix + 24]);
    copy_act_segment<17, 0, 15>(w19, row_buf[pix + 25]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(608), w19, ok);
    act_vec_t w20 = 0;
    copy_act_segment<0, 15, 10>(w20, row_buf[pix + 25]);
    copy_act_segment<10, 0, 22>(w20, row_buf[pix + 26]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(640), w20, ok);
    act_vec_t w21 = 0;
    copy_act_segment<0, 22, 3>(w21, row_buf[pix + 26]);
    copy_act_segment<3, 0, 25>(w21, row_buf[pix + 27]);
    copy_act_segment<28, 0, 4>(w21, row_buf[pix + 28]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(672), w21, ok);
    act_vec_t w22 = 0;
    copy_act_segment<0, 4, 21>(w22, row_buf[pix + 28]);
    copy_act_segment<21, 0, 11>(w22, row_buf[pix + 29]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(704), w22, ok);
    act_vec_t w23 = 0;
    copy_act_segment<0, 11, 14>(w23, row_buf[pix + 29]);
    copy_act_segment<14, 0, 18>(w23, row_buf[pix + 30]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(736), w23, ok);
    act_vec_t w24 = 0;
    copy_act_segment<0, 18, 7>(w24, row_buf[pix + 30]);
    copy_act_segment<7, 0, 25>(w24, row_buf[pix + 31]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(768), w24, ok);
  }
  return ok;
}

static bool store_compact_c28_row(const tensor_desc_t& dst,
                                  u16_t out_row,
                                  int out_w_i,
                                  act_vec_t row_buf[MAX_FM_W]) {
#pragma HLS INLINE off
  bool ok = true;
  for (int group = 0; group < MAX_FM_W / 8; ++group) {
    const int pix = group * 8;
    if (pix >= out_w_i) {
      break;
    }
    const u32_t byte_offset = static_cast<u32_t>(group) * static_cast<u32_t>(224);
    act_vec_t w0 = 0;
    copy_act_segment<0, 0, 28>(w0, row_buf[pix + 0]);
    copy_act_segment<28, 0, 4>(w0, row_buf[pix + 1]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(0), w0, ok);
    act_vec_t w1 = 0;
    copy_act_segment<0, 4, 24>(w1, row_buf[pix + 1]);
    copy_act_segment<24, 0, 8>(w1, row_buf[pix + 2]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(32), w1, ok);
    act_vec_t w2 = 0;
    copy_act_segment<0, 8, 20>(w2, row_buf[pix + 2]);
    copy_act_segment<20, 0, 12>(w2, row_buf[pix + 3]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(64), w2, ok);
    act_vec_t w3 = 0;
    copy_act_segment<0, 12, 16>(w3, row_buf[pix + 3]);
    copy_act_segment<16, 0, 16>(w3, row_buf[pix + 4]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(96), w3, ok);
    act_vec_t w4 = 0;
    copy_act_segment<0, 16, 12>(w4, row_buf[pix + 4]);
    copy_act_segment<12, 0, 20>(w4, row_buf[pix + 5]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(128), w4, ok);
    act_vec_t w5 = 0;
    copy_act_segment<0, 20, 8>(w5, row_buf[pix + 5]);
    copy_act_segment<8, 0, 24>(w5, row_buf[pix + 6]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(160), w5, ok);
    act_vec_t w6 = 0;
    copy_act_segment<0, 24, 4>(w6, row_buf[pix + 6]);
    copy_act_segment<4, 0, 28>(w6, row_buf[pix + 7]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(192), w6, ok);
  }
  return ok;
}

static bool store_c16_into_c19_row(const tensor_desc_t& dst,
                                   u16_t out_row,
                                   int out_w_i,
                                   act_vec_t row_buf[MAX_FM_W]) {
#pragma HLS INLINE off
  bool ok = true;
  for (int group = 0; group < MAX_FM_W / 32; ++group) {
    const int pix = group * 32;
    if (pix >= out_w_i) {
      break;
    }
    const u32_t byte_offset = static_cast<u32_t>(group) * static_cast<u32_t>(608);

    act_vec_t w0 = 0;
    copy_act_segment<0, 0, 16>(w0, row_buf[pix + 0]);
    copy_act_segment<19, 0, 13>(w0, row_buf[pix + 1]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(0), w0, ok);
    act_vec_t w1 = 0;
    copy_act_segment<0, 13, 3>(w1, row_buf[pix + 1]);
    copy_act_segment<6, 0, 16>(w1, row_buf[pix + 2]);
    copy_act_segment<25, 0, 7>(w1, row_buf[pix + 3]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(32), w1, ok);
    act_vec_t w2 = 0;
    copy_act_segment<0, 7, 9>(w2, row_buf[pix + 3]);
    copy_act_segment<12, 0, 16>(w2, row_buf[pix + 4]);
    copy_act_segment<31, 0, 1>(w2, row_buf[pix + 5]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(64), w2, ok);
    act_vec_t w3 = 0;
    copy_act_segment<0, 1, 15>(w3, row_buf[pix + 5]);
    copy_act_segment<18, 0, 14>(w3, row_buf[pix + 6]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(96), w3, ok);
    act_vec_t w4 = 0;
    copy_act_segment<0, 14, 2>(w4, row_buf[pix + 6]);
    copy_act_segment<5, 0, 16>(w4, row_buf[pix + 7]);
    copy_act_segment<24, 0, 8>(w4, row_buf[pix + 8]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(128), w4, ok);
    act_vec_t w5 = 0;
    copy_act_segment<0, 8, 8>(w5, row_buf[pix + 8]);
    copy_act_segment<11, 0, 16>(w5, row_buf[pix + 9]);
    copy_act_segment<30, 0, 2>(w5, row_buf[pix + 10]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(160), w5, ok);
    act_vec_t w6 = 0;
    copy_act_segment<0, 2, 14>(w6, row_buf[pix + 10]);
    copy_act_segment<17, 0, 15>(w6, row_buf[pix + 11]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(192), w6, ok);
    act_vec_t w7 = 0;
    copy_act_segment<0, 15, 1>(w7, row_buf[pix + 11]);
    copy_act_segment<4, 0, 16>(w7, row_buf[pix + 12]);
    copy_act_segment<23, 0, 9>(w7, row_buf[pix + 13]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(224), w7, ok);
    act_vec_t w8 = 0;
    copy_act_segment<0, 9, 7>(w8, row_buf[pix + 13]);
    copy_act_segment<10, 0, 16>(w8, row_buf[pix + 14]);
    copy_act_segment<29, 0, 3>(w8, row_buf[pix + 15]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(256), w8, ok);
    act_vec_t w9 = 0;
    copy_act_segment<0, 3, 13>(w9, row_buf[pix + 15]);
    copy_act_segment<16, 0, 16>(w9, row_buf[pix + 16]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(288), w9, ok);
    act_vec_t w10 = 0;
    copy_act_segment<3, 0, 16>(w10, row_buf[pix + 17]);
    copy_act_segment<22, 0, 10>(w10, row_buf[pix + 18]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(320), w10, ok);
    act_vec_t w11 = 0;
    copy_act_segment<0, 10, 6>(w11, row_buf[pix + 18]);
    copy_act_segment<9, 0, 16>(w11, row_buf[pix + 19]);
    copy_act_segment<28, 0, 4>(w11, row_buf[pix + 20]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(352), w11, ok);
    act_vec_t w12 = 0;
    copy_act_segment<0, 4, 12>(w12, row_buf[pix + 20]);
    copy_act_segment<15, 0, 16>(w12, row_buf[pix + 21]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(384), w12, ok);
    act_vec_t w13 = 0;
    copy_act_segment<2, 0, 16>(w13, row_buf[pix + 22]);
    copy_act_segment<21, 0, 11>(w13, row_buf[pix + 23]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(416), w13, ok);
    act_vec_t w14 = 0;
    copy_act_segment<0, 11, 5>(w14, row_buf[pix + 23]);
    copy_act_segment<8, 0, 16>(w14, row_buf[pix + 24]);
    copy_act_segment<27, 0, 5>(w14, row_buf[pix + 25]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(448), w14, ok);
    act_vec_t w15 = 0;
    copy_act_segment<0, 5, 11>(w15, row_buf[pix + 25]);
    copy_act_segment<14, 0, 16>(w15, row_buf[pix + 26]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(480), w15, ok);
    act_vec_t w16 = 0;
    copy_act_segment<1, 0, 16>(w16, row_buf[pix + 27]);
    copy_act_segment<20, 0, 12>(w16, row_buf[pix + 28]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(512), w16, ok);
    act_vec_t w17 = 0;
    copy_act_segment<0, 12, 4>(w17, row_buf[pix + 28]);
    copy_act_segment<7, 0, 16>(w17, row_buf[pix + 29]);
    copy_act_segment<26, 0, 6>(w17, row_buf[pix + 30]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(544), w17, ok);
    act_vec_t w18 = 0;
    copy_act_segment<0, 6, 10>(w18, row_buf[pix + 30]);
    copy_act_segment<13, 0, 16>(w18, row_buf[pix + 31]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(576), w18, ok);
  }
  return ok;
}

static bool store_conv_output_row(const tensor_desc_t& dst,
                                  u16_t out_row,
                                  u16_t c_offset,
                                  const conv_cfg_t& cfg,
                                  act_vec_t row_buf[MAX_FM_W]) {
#pragma HLS INLINE off
  const u16_t stride = conv_effective_stride(cfg);
  const u16_t out_w = conv_out_dim(cfg.in_w, stride);
  const int out_w_i = static_cast<int>(out_w.to_uint());
  const u8_t valid_c = static_cast<u8_t>(cfg.out_c.to_uint());
  bool ok = true;

  const bool compact_row =
      c_offset.to_uint() == 0U &&
      core_desc_phys_c(dst).to_uint() == dst.c.to_uint() &&
      dst.c.to_uint() == cfg.out_c.to_uint();
  if (compact_row && cfg.out_c.to_uint() == 12U &&
      (out_w_i & 7) == 0) {
    return store_compact_c12_row(dst, out_row, out_w_i, row_buf);
  }
  if (compact_row && cfg.out_c.to_uint() == 16U &&
      (out_w_i & 1) == 0) {
    return store_compact_c16_row(dst, out_row, out_w_i, row_buf);
  }
  if (compact_row && cfg.out_c.to_uint() == 25U &&
      (out_w_i & 31) == 0) {
    return store_compact_c25_row(dst, out_row, out_w_i, row_buf);
  }
  if (compact_row && cfg.out_c.to_uint() == 28U &&
      (out_w_i & 7) == 0) {
    return store_compact_c28_row(dst, out_row, out_w_i, row_buf);
  }
  if (compact_row && cfg.out_c.to_uint() == 2U &&
      (out_w_i & 15) == 0) {
    return store_compact_c2_row(dst, out_row, out_w_i, row_buf);
  }
  if (c_offset.to_uint() == 0U &&
      cfg.out_c.to_uint() == 16U &&
      dst.c.to_uint() == 19U &&
      core_desc_phys_c(dst).to_uint() == 19U &&
      (out_w_i & 31) == 0) {
    return store_c16_into_c19_row(dst, out_row, out_w_i, row_buf);
  }

  for (int ow_i = 0; ow_i < MAX_FM_W; ++ow_i) {
#pragma HLS PIPELINE off
    if (ow_i >= out_w_i) {
      break;
    }
    const act_vec_t packed = row_buf[ow_i];
    if (!on_chip_memory_write_packed_tile(dst,
                                          out_row,
                                          static_cast<u16_t>(ow_i),
                                          c_offset,
                                          valid_c,
                                          packed)) {
      ok = false;
    }
  }
  return ok;
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
#pragma HLS STREAM variable=act_stream depth=32
#pragma HLS STREAM variable=wgt_stream depth=1200
#pragma HLS STREAM variable=psum_stream depth=16
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
                                         bool& ok_out) {
#pragma HLS INLINE off
  act_vec_t row_buf[MAX_FM_W];
#pragma HLS BIND_STORAGE variable=row_buf type=ram_2p impl=bram

  const u16_t stride = conv_effective_stride(cfg);
  const u16_t out_h = conv_out_dim(cfg.in_h, stride);
  const u16_t k_total = conv_kernel_flat(cfg);
  const u16_t k_tiles = ceil_div_u16(k_total, static_cast<u16_t>(TK));
  const u16_t oc_tiles = ceil_div_u16(cfg.out_c, static_cast<u16_t>(TM));
  const int out_h_i = static_cast<int>(out_h.to_uint());
  const int k_tiles_i = static_cast<int>(k_tiles.to_uint());
  const int oc_tiles_i = static_cast<int>(oc_tiles.to_uint());
  bool write_ok = true;

  const int wgt_count = k_tiles_i * TM * oc_tiles_i;
  wgt_vec_t cached_wgts[MAX_K_TILE_COUNT * TM];
#pragma HLS BIND_STORAGE variable=cached_wgts type=ram_1p impl=bram
  {
    int wi = 0;
    for (int oct = 0; oct < MAX_C_TILE_COUNT; ++oct) {
      if (oct >= oc_tiles_i) break;
      const u16_t oc_tile = static_cast<u16_t>(oct);
      for (int kt = 0; kt < MAX_K_TILE_COUNT; ++kt) {
        if (kt >= k_tiles_i) break;
        const u16_t k_tile = static_cast<u16_t>(kt);
        for (int tm = 0; tm < TM; ++tm) {
#pragma HLS PIPELINE II=1
          const u16_t oc = static_cast<u16_t>(oc_tile * TM + tm);
          wgt_vec_t word = 0;
          param_dma_get_weight_vec(uop.param_id, oc, k_tile, cfg, word);
          cached_wgts[wi++] = word;
        }
      }
    }
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
    if (!store_conv_output_row(dst, oh, uop.c_offset, cfg, row_buf)) {
      write_ok = false;
    }
  }
  ok_out = write_ok;
}

static error_code_t execute_pool_uop(const uop_t& uop) {
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

static error_code_t execute_conv_uop(const uop_t& uop) {
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
  if (!resolve_tensor_read(uop.src0_tensor, src) ||
      !resolve_tensor_write(uop.dst_tensor, out_h, out_w, uop.out_c, dst)) {
    return ERR_TENSOR_DESC_RANGE;
  }
  if (!param_dma_get_conv_qparam(uop.param_id, qparam)) {
    return ERR_PARAM_DESC_RANGE;
  }
  if (tensor_is_global(uop.dst_tensor) &&
      dst.c.to_uint() < uop.c_offset.to_uint() + uop.out_c.to_uint()) {
    return ERR_TENSOR_DESC_RANGE;
  }
  // The exporter emits wide tensors as multiple <=TM conv/store chunks. Keeping
  // this contract avoids activation-stream rebroadcast and preserves one-pass
  // producer/consumer matching in the stream datapath.
  if (cfg.out_c.to_uint() > static_cast<unsigned>(TM)) {
    return ERR_UNSUPPORTED_OPCODE;
  }

  bool conv_ok = false;
  execute_conv_stream_datapath(src,
                               dst,
                               cfg,
                               uop,
                               qparam,
                               conv_ok);
  if (!conv_ok) {
    return ERR_BANK_OVERFLOW;
  }

  return ERR_NONE;
}

static error_code_t execute_affine_uop(const uop_t& uop) {
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
        i8_t in_tile[TM];
        i8_t out_tile[TM];
#pragma HLS ARRAY_PARTITION variable=in_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=out_tile complete dim=1
        if (!param_dma_get_affine_qparam(uop.param_id, block_id, qparam)) {
          return ERR_PARAM_DESC_RANGE;
        }
        if (!on_chip_memory_read_tile(src, static_cast<i32_t>(h), static_cast<i32_t>(w), c, lanes, in_tile)) {
          return ERR_BANK_OVERFLOW;
        }
        for (int lane = 0; lane < TM; ++lane) {
#pragma HLS UNROLL
          out_tile[lane] = 0;
          if (static_cast<unsigned>(lane) < lanes.to_uint()) {
            out_tile[lane] =
                affine_i8_to_i8(in_tile[lane], qparam.mul[lane], qparam.bias[lane], qparam.shift[lane], uop.act_type);
          }
        }
        if (!on_chip_memory_write_tile(dst, h, w, c, lanes, out_tile)) {
          return ERR_BANK_OVERFLOW;
        }
      }
    }
  }

  return ERR_NONE;
}

static error_code_t execute_add_uop(const uop_t& uop) {
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
        i8_t a_tile[TM];
        i8_t b_tile[TM];
        i8_t out_tile[TM];
#pragma HLS ARRAY_PARTITION variable=a_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=b_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=out_tile complete dim=1
        if (!on_chip_memory_read_tile(src0, static_cast<i32_t>(h), static_cast<i32_t>(w), c, lanes, a_tile) ||
            !on_chip_memory_read_tile(src1, static_cast<i32_t>(h), static_cast<i32_t>(w), c, lanes, b_tile)) {
          return ERR_BANK_OVERFLOW;
        }
        for (int lane = 0; lane < TM; ++lane) {
#pragma HLS UNROLL
          out_tile[lane] = 0;
          if (static_cast<unsigned>(lane) < lanes.to_uint()) {
            out_tile[lane] = add_i8(a_tile[lane], b_tile[lane], qparam);
          }
        }
        if (!on_chip_memory_write_tile(dst, h, w, c, lanes, out_tile)) {
          return ERR_BANK_OVERFLOW;
        }
      }
    }
  }

  return ERR_NONE;
}

static error_code_t execute_store_uop(const uop_t& uop) {
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

static error_code_t execute_uop_partial(const uop_t& uop) {
#pragma HLS INLINE off
  const unsigned opcode = uop.opcode.to_uint();
  if (opcode == static_cast<unsigned>(UOP_NOP) ||
      opcode == static_cast<unsigned>(UOP_LOAD_FM)) {
    return ERR_NONE;
  }
  if (opcode == static_cast<unsigned>(UOP_POOL)) {
    return execute_pool_uop(uop);
  }
  if (opcode == static_cast<unsigned>(UOP_STORE)) {
    return execute_store_uop(uop);
  }
  if (opcode == static_cast<unsigned>(UOP_ADD)) {
    return execute_add_uop(uop);
  }
  if (opcode == static_cast<unsigned>(UOP_AFFINE)) {
    return execute_affine_uop(uop);
  }
  if (opcode == static_cast<unsigned>(UOP_CONV)) {
    return execute_conv_uop(uop);
  }
  if (opcode == static_cast<unsigned>(UOP_END)) {
    return ERR_NONE;
  }
  return ERR_UNSUPPORTED_OPCODE;
}

static void load_param_header(const axi_vec_t* gmem_param, param_blob_header_t& header) {
#pragma HLS INLINE
  u32_t raw[32];
#pragma HLS ARRAY_PARTITION variable=raw complete dim=1

  for (int word_idx = 0; word_idx < PARAM_HEADER_AXI_WORDS; ++word_idx) {
#pragma HLS PIPELINE II=1
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
  if (header.uop_count > MAX_UOP_COUNT) {
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
  clear_status();
  reset_scratch_state();
  s_param_ready = false;

  load_param_header(gmem_param, s_param_header);
  const error_code_t err = validate_param_header(s_param_header);
  if (err != ERR_NONE) {
    set_error(err, 0);
    return;
  }

  param_dma_init(gmem_param);
  if (!param_dma_ready()) {
    set_error(param_dma_error(), 0);
    return;
  }
  s_param_ready = true;
}

static void core_mode_run(const axi_vec_t* gmem_frame_in,
                          axi_vec_t* gmem_frame_out,
                          const axi_vec_t* gmem_param,
                          u32_t expected_uop_count) {
#pragma HLS INLINE off
  clear_status();

  if (!s_param_ready) {
    set_error(ERR_BAD_BLOB, 0);
    return;
  }

  if (expected_uop_count != s_param_header.uop_count.to_uint()) {
    set_error(ERR_UOP_DECODE, 0);
    return;
  }

  instruction_fetch_decode(gmem_param, expected_uop_count);
  if (if_dec_error() != ERR_NONE) {
    set_error(if_dec_error(), if_dec_current_uop_id());
    return;
  }

  frame_dma_load(gmem_frame_in);

  for (int uop_idx = 0; uop_idx < MAX_UOP_COUNT; ++uop_idx) {
    if (uop_idx >= static_cast<int>(expected_uop_count)) {
      break;
    }

    uop_t uop;
    if (!param_dma_get_uop(static_cast<u16_t>(uop_idx), uop)) {
      set_error(ERR_UOP_DECODE, static_cast<u16_t>(uop_idx));
      return;
    }
    s_status.current_uop_id = static_cast<u16_t>(uop_idx);
    if (uop.opcode.to_uint() == static_cast<unsigned>(UOP_END)) {
      break;
    }

    const error_code_t err = execute_uop_partial(uop);
    if (err != ERR_NONE) {
      set_error(err, static_cast<u16_t>(uop_idx));
      return;
    }
  }

  frame_dma_store(gmem_frame_out);
}

}  // namespace esp_int8

static_assert(esp_int8::INPUT_FRAME_AXI_WORDS == 49152,
              "Update gmem_frame_in m_axi depth when INPUT_FRAME_AXI_WORDS changes.");
static_assert(esp_int8::OUTPUT_FRAME_AXI_WORDS == 512,
              "Update gmem_frame_out m_axi depth when OUTPUT_FRAME_AXI_WORDS changes.");

void espnet_encoder_int8_core(const esp_int8::axi_vec_t* gmem_frame_in,
                              esp_int8::axi_vec_t* gmem_frame_out,
                              const esp_int8::axi_vec_t* gmem_param,
                              std::uint32_t mode,
                              std::uint32_t uop_count) {
#ifdef ESP_INT8_COSIM_LITE
#pragma HLS INTERFACE ap_memory port=gmem_frame_in depth=49152
#pragma HLS INTERFACE ap_memory port=gmem_frame_out depth=512
#pragma HLS INTERFACE ap_memory port=gmem_param depth=4096
#pragma HLS INTERFACE ap_none port=mode
#pragma HLS INTERFACE ap_none port=uop_count
#pragma HLS INTERFACE ap_ctrl_hs port=return
#else
#pragma HLS INTERFACE m_axi port=gmem_frame_in offset=slave bundle=gmem0 depth=49152 max_read_burst_length=64 num_read_outstanding=4
#pragma HLS INTERFACE m_axi port=gmem_frame_out offset=slave bundle=gmem1 depth=512 max_write_burst_length=64 num_write_outstanding=4
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
                              gmem_param,
                              expected_uop_count);
      break;
    case esp_int8::MODE_IDLE:
      break;
    default:
      esp_int8::set_error(esp_int8::ERR_INVALID_MODE, 0);
      break;
  }
}
