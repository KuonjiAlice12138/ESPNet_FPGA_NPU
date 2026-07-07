#include <cstdint>

#include "../include/npu_config.hpp"
#include "../include/npu_ctrl.hpp"
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
bool param_dma_get_fixed_exec_desc(u8_t id, fixed_exec_desc_t& desc);
bool param_dma_get_block5_sched(u8_t id, block5_sched_desc_t& desc);
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
void reset_scratch_state();
void select_scratch_region_for_fields(u8_t opcode,
                                      u8_t param_id,
                                      u8_t src0_tensor,
                                      u8_t src1_tensor,
                                      u8_t dst_tensor,
                                      u16_t in_h,
                                      u16_t out_h);
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
bool resolve_block5_scratch_descs(u16_t pattern,
                                  u8_t first_branch_param,
                                  u16_t rows,
                                  u16_t out_w,
                                  tensor_desc_t& s0,
                                  tensor_desc_t& s1,
                                  tensor_desc_t& s2,
                                  tensor_desc_t& s3,
                                  tensor_desc_t& s4);
bool backup_b2_src1_rows_before_write(const tensor_desc_t& src1,
                                      int write_row,
                                      bool src1_saved[MAX_FM_H]);
bool read_b2_backup_src1_tile(u16_t h,
                              u16_t w,
                              u16_t c,
                              u8_t lanes,
                              act_vec_t& packed);
void upsample_fused_begin();

static param_blob_header_t s_param_header;
static bool s_param_ready = false;

static u32_t s_prof_uop_count = 0;
static u32_t s_prof_conv_count = 0;
static u32_t s_prof_win_read_ops = 0;
static u32_t s_prof_win_words = 0;
static u32_t s_prof_wgt_words = 0;
static u32_t s_prof_sa_mac_steps = 0;
static u32_t s_prof_psum_words = 0;
static u32_t s_prof_out_tiles = 0;
static u32_t s_prof_out_rmw_ops = 0;
static u32_t s_prof_model_cycles = 0;
static u32_t s_prof2_win_saved_reads = 0;
static u32_t s_prof2_win_actual_reads = 0;
static u32_t s_prof2_out_direct_words = 0;
static u32_t s_prof2_out_rmw_reads = 0;
static u32_t s_prof3_wgt_cycles = 0;
static u32_t s_prof3_win_cycles = 0;
static u32_t s_prof3_sa_cycles = 0;
static u32_t s_prof3_post_cycles = 0;
static u32_t s_prof3_write_cycles = 0;
static u32_t s_prof3_row_region_cycles = 0;
static u32_t s_prof5_if_words = 0;
static u32_t s_prof5_frame_load_words = 0;
static u32_t s_prof5_frame_store_words = 0;
static u32_t s_prof5_uop_fetches = 0;
static u32_t s_prof5_pool_tiles = 0;
static u32_t s_prof5_affine_tiles = 0;
static u32_t s_prof5_add_tiles = 0;
static u32_t s_prof5_store_tiles = 0;
static u32_t s_prof5_nonconv_mem_ops = 0;
static u32_t s_prof5_conv_model_cycles = 0;
static u32_t s_prof5_total_work_units = 0;
static u32_t s_prof7_exec_conv = 0;
static u32_t s_prof7_exec_pool = 0;
static u32_t s_prof7_exec_affine = 0;
static u32_t s_prof7_exec_store = 0;
static u32_t s_prof7_exec_add_affine = 0;
static u32_t s_prof7_win_cmd_execs = 0;
static u32_t s_prof7_win_cache_loads = 0;
static u32_t s_prof7_row_regions = 0;
static u32_t s_prof7_fixed_iters = 0;
static u32_t s_prof7_narrow_rmw_writes = 0;
static u32_t s_prof7_upsample_rows = 0;
static u32_t s_prof7_win_interpreter_rows = 0;

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

static u32_t runtime_expected_uop_count(std::uint32_t raw_uop_count) {
#pragma HLS INLINE
  return static_cast<u32_t>(raw_uop_count & RUNTIME_UOP_COUNT_MASK);
}

static profile_ctrl_t runtime_profile_ctrl(std::uint32_t mode, std::uint32_t raw_uop_count) {
#pragma HLS INLINE
  profile_ctrl_t ctrl;
  ctrl.enable = (mode & RUNTIME_PROFILE_ENABLE_MASK) != 0U;
  ctrl.stop_before = (mode & RUNTIME_PROFILE_STOP_BEFORE_MASK) != 0U;
  ctrl.stop_pc_plus1 = static_cast<u16_t>(raw_uop_count >> RUNTIME_PROFILE_STOP_SHIFT);
  return ctrl;
}

static bool profile_stop_matches(const profile_ctrl_t& ctrl, int pc) {
#pragma HLS INLINE
  if (!ctrl.enable || ctrl.stop_pc_plus1.to_uint() == 0U) {
    return false;
  }
  return static_cast<unsigned>(pc + 1) == ctrl.stop_pc_plus1.to_uint();
}

static bool csim_stop_before_logical_uop(unsigned uop_id);
static bool csim_dump_tensor_set_pre(unsigned logical_uop);
static bool csim_dump_tensor_set_post(unsigned logical_uop);

static void profile_clear() {
#pragma HLS INLINE
  s_prof_uop_count = 0;
  s_prof_conv_count = 0;
  s_prof_win_read_ops = 0;
  s_prof_win_words = 0;
  s_prof_wgt_words = 0;
  s_prof_sa_mac_steps = 0;
  s_prof_psum_words = 0;
  s_prof_out_tiles = 0;
  s_prof_out_rmw_ops = 0;
  s_prof_model_cycles = 0;
  s_prof2_win_saved_reads = 0;
  s_prof2_win_actual_reads = 0;
  s_prof2_out_direct_words = 0;
  s_prof2_out_rmw_reads = 0;
  s_prof3_wgt_cycles = 0;
  s_prof3_win_cycles = 0;
  s_prof3_sa_cycles = 0;
  s_prof3_post_cycles = 0;
  s_prof3_write_cycles = 0;
  s_prof3_row_region_cycles = 0;
  s_prof5_if_words = 0;
  s_prof5_frame_load_words = 0;
  s_prof5_frame_store_words = 0;
  s_prof5_uop_fetches = 0;
  s_prof5_pool_tiles = 0;
  s_prof5_affine_tiles = 0;
  s_prof5_add_tiles = 0;
  s_prof5_store_tiles = 0;
  s_prof5_nonconv_mem_ops = 0;
  s_prof5_conv_model_cycles = 0;
  s_prof5_total_work_units = 0;
  s_prof7_exec_conv = 0;
  s_prof7_exec_pool = 0;
  s_prof7_exec_affine = 0;
  s_prof7_exec_store = 0;
  s_prof7_exec_add_affine = 0;
  s_prof7_win_cmd_execs = 0;
  s_prof7_win_cache_loads = 0;
  s_prof7_row_regions = 0;
  s_prof7_fixed_iters = 0;
  s_prof7_narrow_rmw_writes = 0;
  s_prof7_upsample_rows = 0;
  s_prof7_win_interpreter_rows = 0;
}

static void profile_publish(volatile std::uint32_t& prof_uop_count,
                            volatile std::uint32_t& prof_conv_count,
                            volatile std::uint32_t& prof_win_read_ops,
                            volatile std::uint32_t& prof_win_words,
                            volatile std::uint32_t& prof_wgt_words,
                            volatile std::uint32_t& prof_sa_mac_steps,
                            volatile std::uint32_t& prof_psum_words,
                            volatile std::uint32_t& prof_out_tiles,
                            volatile std::uint32_t& prof_out_rmw_ops,
                            volatile std::uint32_t& prof_model_cycles,
                            volatile std::uint32_t& prof2_win_saved_reads,
                            volatile std::uint32_t& prof2_win_actual_reads,
                            volatile std::uint32_t& prof2_out_direct_words,
                            volatile std::uint32_t& prof2_out_rmw_reads,
                            volatile std::uint32_t& prof3_wgt_cycles,
                            volatile std::uint32_t& prof3_win_cycles,
                            volatile std::uint32_t& prof3_sa_cycles,
                            volatile std::uint32_t& prof3_post_cycles,
                            volatile std::uint32_t& prof3_write_cycles,
                            volatile std::uint32_t& prof3_row_region_cycles,
                            volatile std::uint32_t& prof5_if_words,
                            volatile std::uint32_t& prof5_frame_load_words,
                            volatile std::uint32_t& prof5_frame_store_words,
                            volatile std::uint32_t& prof5_uop_fetches,
                            volatile std::uint32_t& prof5_pool_tiles,
                            volatile std::uint32_t& prof5_affine_tiles,
                            volatile std::uint32_t& prof5_add_tiles,
                            volatile std::uint32_t& prof5_store_tiles,
                            volatile std::uint32_t& prof5_nonconv_mem_ops,
                            volatile std::uint32_t& prof5_conv_model_cycles,
                            volatile std::uint32_t& prof5_total_work_units,
                            volatile std::uint32_t& prof7_exec_conv,
                            volatile std::uint32_t& prof7_exec_pool,
                            volatile std::uint32_t& prof7_exec_affine,
                            volatile std::uint32_t& prof7_exec_store,
                            volatile std::uint32_t& prof7_exec_add_affine,
                            volatile std::uint32_t& prof7_win_cmd_execs,
                            volatile std::uint32_t& prof7_win_cache_loads,
                            volatile std::uint32_t& prof7_row_regions,
                            volatile std::uint32_t& prof7_fixed_iters,
                            volatile std::uint32_t& prof7_narrow_rmw_writes,
                            volatile std::uint32_t& prof7_upsample_rows,
                            volatile std::uint32_t& prof7_win_interpreter_rows) {
#pragma HLS INLINE
  prof_uop_count = s_prof_uop_count.to_uint();
  prof_conv_count = s_prof_conv_count.to_uint();
  prof_win_read_ops = s_prof_win_read_ops.to_uint();
  prof_win_words = s_prof_win_words.to_uint();
  prof_wgt_words = s_prof_wgt_words.to_uint();
  prof_sa_mac_steps = s_prof_sa_mac_steps.to_uint();
  prof_psum_words = s_prof_psum_words.to_uint();
  prof_out_tiles = s_prof_out_tiles.to_uint();
  prof_out_rmw_ops = s_prof_out_rmw_ops.to_uint();
  prof_model_cycles = s_prof_model_cycles.to_uint();
  prof2_win_saved_reads = s_prof2_win_saved_reads.to_uint();
  prof2_win_actual_reads = s_prof2_win_actual_reads.to_uint();
  prof2_out_direct_words = s_prof2_out_direct_words.to_uint();
  prof2_out_rmw_reads = s_prof2_out_rmw_reads.to_uint();
  prof3_wgt_cycles = s_prof3_wgt_cycles.to_uint();
  prof3_win_cycles = s_prof3_win_cycles.to_uint();
  prof3_sa_cycles = s_prof3_sa_cycles.to_uint();
  prof3_post_cycles = s_prof3_post_cycles.to_uint();
  prof3_write_cycles = s_prof3_write_cycles.to_uint();
  prof3_row_region_cycles = s_prof3_row_region_cycles.to_uint();
  prof5_if_words = s_prof5_if_words.to_uint();
  prof5_frame_load_words = s_prof5_frame_load_words.to_uint();
  prof5_frame_store_words = s_prof5_frame_store_words.to_uint();
  prof5_uop_fetches = s_prof5_uop_fetches.to_uint();
  prof5_pool_tiles = s_prof5_pool_tiles.to_uint();
  prof5_affine_tiles = s_prof5_affine_tiles.to_uint();
  prof5_add_tiles = s_prof5_add_tiles.to_uint();
  prof5_store_tiles = s_prof5_store_tiles.to_uint();
  prof5_nonconv_mem_ops = s_prof5_nonconv_mem_ops.to_uint();
  prof5_conv_model_cycles = s_prof5_conv_model_cycles.to_uint();
  prof5_total_work_units = s_prof5_total_work_units.to_uint();
  prof7_exec_conv = s_prof7_exec_conv.to_uint();
  prof7_exec_pool = s_prof7_exec_pool.to_uint();
  prof7_exec_affine = s_prof7_exec_affine.to_uint();
  prof7_exec_store = s_prof7_exec_store.to_uint();
  prof7_exec_add_affine = s_prof7_exec_add_affine.to_uint();
  prof7_win_cmd_execs = s_prof7_win_cmd_execs.to_uint();
  prof7_win_cache_loads = s_prof7_win_cache_loads.to_uint();
  prof7_row_regions = s_prof7_row_regions.to_uint();
  prof7_fixed_iters = s_prof7_fixed_iters.to_uint();
  prof7_narrow_rmw_writes = s_prof7_narrow_rmw_writes.to_uint();
  prof7_upsample_rows = s_prof7_upsample_rows.to_uint();
  prof7_win_interpreter_rows = s_prof7_win_interpreter_rows.to_uint();
}

static u16_t ceil_div_u16(u16_t a, u16_t b) {
#pragma HLS INLINE
  return static_cast<u16_t>((a + b - 1) / b);
}

static u16_t conv_out_dim(u16_t in_size, u16_t stride) {
#pragma HLS INLINE
  return ceil_div_u16(in_size, stride);
}

static u16_t tensor_desc_phys_c_checked(const tensor_desc_t& desc) {
#pragma HLS INLINE
  return (desc.reserved0.to_uint() == 0U) ? desc.c : desc.reserved0;
}

bool alias_tensor_to_slice(u8_t tensor_id,
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

static u16_t conv_effective_stride(const conv_cfg_t& cfg) {
#pragma HLS INLINE
  return (cfg.stride == 0) ? static_cast<u16_t>(1) : static_cast<u16_t>(cfg.stride);
}

struct profile_store_stats_t {
  u32_t direct_words;
  u32_t rmw_reads;
  u32_t rmw_writes;
};

static u32_t profile_ceil_div_u32(u32_t a, u32_t b) {
#pragma HLS INLINE
  return (b == 0U) ? static_cast<u32_t>(0) : static_cast<u32_t>((a + b - 1U) / b);
}

static u32_t profile_max_u32(u32_t a, u32_t b) {
#pragma HLS INLINE
  return (a > b) ? a : b;
}

static u32_t profile_out_pixels(u16_t out_h, u16_t out_w) {
#pragma HLS INLINE
  return static_cast<u32_t>(out_h) * static_cast<u32_t>(out_w);
}

static u32_t profile_words_for_packed_row(u32_t out_w, u32_t valid_c) {
#pragma HLS INLINE
  return profile_ceil_div_u32(out_w * valid_c, static_cast<u32_t>(AXI_WORD_BYTES));
}

static u32_t profile_3x3_cache_cols(u32_t out_w, u8_t stride, u8_t dilation) {
#pragma HLS INLINE
  const unsigned stride_u = stride.to_uint();
  const unsigned dilation_u = dilation.to_uint();
  if (out_w == 0U) {
    return 0;
  }
  if (stride_u == 1U && dilation_u == 1U) {
    return out_w + 2U;
  }
  if (stride_u == 2U && dilation_u == 1U) {
    return out_w * 2U + 1U;
  }
  if (stride_u == 1U && dilation_u == 2U) {
    return out_w + 4U;
  }
  return out_w * 3U;
}

static profile_store_stats_t profile_row_store_stats(const tensor_desc_t& dst,
                                                     u16_t out_h,
                                                     u16_t out_w,
                                                     u16_t c_offset,
                                                     u8_t valid_c,
                                                     u16_t store_layout) {
#pragma HLS INLINE
  profile_store_stats_t stats;
  stats.direct_words = 0;
  stats.rmw_reads = 0;
  stats.rmw_writes = 0;

  const u32_t h_u = static_cast<u32_t>(out_h);
  const u32_t w_u = static_cast<u32_t>(out_w);
  const u32_t c_u = static_cast<u32_t>(valid_c);
  const u32_t pixels = h_u * w_u;
  const unsigned layout = store_layout.to_uint();

  switch (layout) {
    case static_cast<unsigned>(STORE_LAYOUT_COMPACT_C12):
    case static_cast<unsigned>(STORE_LAYOUT_COMPACT_C19):
    case static_cast<unsigned>(STORE_LAYOUT_COMPACT_C25):
      stats.direct_words = h_u * profile_words_for_packed_row(w_u, c_u);
      return stats;
    case static_cast<unsigned>(STORE_LAYOUT_ALIGNED_TILE_COPY):
      stats.direct_words = pixels;
      return stats;
    default:
      break;
  }

  const unsigned byte0 = c_offset.to_uint() & static_cast<unsigned>(AXI_WORD_BYTES - 1);
  const unsigned lanes = valid_c.to_uint();
  const bool aligned_direct =
      byte0 == 0U && lanes == static_cast<unsigned>(AXI_WORD_BYTES) &&
      ((tensor_desc_phys_c_checked(dst).to_uint() & static_cast<unsigned>(AXI_WORD_BYTES - 1)) == 0U);
  if (aligned_direct) {
    stats.direct_words = pixels;
    return stats;
  }

  const bool crosses = (byte0 + lanes) > static_cast<unsigned>(AXI_WORD_BYTES);
  const u32_t rmw_words_per_pixel = crosses ? static_cast<u32_t>(2) : static_cast<u32_t>(1);
  stats.rmw_reads = pixels * rmw_words_per_pixel;
  stats.rmw_writes = pixels * rmw_words_per_pixel;
  return stats;
}

static void profile_record_exec_entry(unsigned kind) {
#pragma HLS INLINE
  s_prof_uop_count = static_cast<u32_t>(s_prof_uop_count + 1U);
  switch (kind) {
    case static_cast<unsigned>(EXEC_CONV):
      s_prof7_exec_conv = static_cast<u32_t>(s_prof7_exec_conv + 1U);
      break;
    case static_cast<unsigned>(EXEC_POOL):
      s_prof7_exec_pool = static_cast<u32_t>(s_prof7_exec_pool + 1U);
      break;
    case static_cast<unsigned>(EXEC_BLOCK_AFFINE):
    case static_cast<unsigned>(EXEC_BLOCK_ADD_AFFINE):
      s_prof7_exec_affine = static_cast<u32_t>(s_prof7_exec_affine + 1U);
      break;
    default:
      break;
  }
}

bool main_ctrl_profile_stop_matches(const profile_ctrl_t& ctrl, int pc) {
#pragma HLS INLINE
  return profile_stop_matches(ctrl, pc);
}

void main_ctrl_profile_record_exec_entry(unsigned kind) {
#pragma HLS INLINE
  profile_record_exec_entry(kind);
}

void main_ctrl_record_exec_fetch() {
#pragma HLS INLINE
  s_prof5_uop_fetches = static_cast<u32_t>(s_prof5_uop_fetches + 1U);
}

void main_ctrl_set_csim_last_uop(unsigned logical_uop) {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
  s_csim_last_uop = logical_uop;
#else
  (void)logical_uop;
#endif
}

void main_ctrl_set_csim_last_error(error_code_t err) {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
  s_csim_last_error = err;
#else
  (void)err;
#endif
}

bool main_ctrl_csim_stop_before_logical_uop(unsigned logical_uop) {
#pragma HLS INLINE
  return csim_stop_before_logical_uop(logical_uop);
}

bool main_ctrl_csim_dump_tensor_set_pre(unsigned logical_uop) {
#pragma HLS INLINE
  return csim_dump_tensor_set_pre(logical_uop);
}

bool main_ctrl_csim_dump_tensor_set_post(unsigned logical_uop) {
#pragma HLS INLINE
  return csim_dump_tensor_set_post(logical_uop);
}

void profile_record_conv_desc(const conv_cfg_t& cfg,
                              const conv_exec_desc_t& conv_desc,
                              const window_sched_desc_t& sched,
                              const row_consumer_desc_t& consumer,
                              const tensor_desc_t& dst) {
#pragma HLS INLINE
  const u16_t stride = conv_effective_stride(cfg);
  const u16_t out_h = conv_out_dim(cfg.in_h, stride);
  const u16_t out_w = conv_out_dim(cfg.in_w, stride);
  const u32_t out_h_u = static_cast<u32_t>(out_h);
  const u32_t out_w_u = static_cast<u32_t>(out_w);
  const u32_t out_pixels = profile_out_pixels(out_h, out_w);
  const u32_t k_tiles = static_cast<u32_t>(conv_desc.k_tiles);
  const u32_t weight_words_per_conv = k_tiles * static_cast<u32_t>(TM);
  const u32_t win_words = out_pixels * k_tiles;
  const u32_t psum_words = out_pixels * static_cast<u32_t>(2);
  const u32_t sa_steps = out_pixels * k_tiles * static_cast<u32_t>(2);
  u32_t cmd_execs = 0;
  u32_t cache_loads = 0;

  if (sched.kernel.to_uint() == 3U) {
    const u32_t cols = profile_3x3_cache_cols(out_w_u, sched.stride, sched.dilation);
    cache_loads = out_h_u * cols * static_cast<u32_t>(3) * static_cast<u32_t>(sched.cache_chunks);
  } else {
    cache_loads = win_words;
  }

  profile_store_stats_t store_stats = profile_store_stats_t();
  if (consumer.mode.to_uint() == static_cast<unsigned>(ROW_CONSUMER_UPSAMPLE_OUT)) {
    store_stats.direct_words = static_cast<u32_t>(OUTPUT_FRAME_AXI_WORDS);
    s_prof5_frame_store_words =
        static_cast<u32_t>(s_prof5_frame_store_words + static_cast<u32_t>(OUTPUT_FRAME_AXI_WORDS));
    s_prof7_upsample_rows = static_cast<u32_t>(s_prof7_upsample_rows + out_h_u);
  } else if (consumer.mode.to_uint() == static_cast<unsigned>(ROW_CONSUMER_CAT_AFFINE_STORE) ||
             consumer.mode.to_uint() == static_cast<unsigned>(ROW_CONSUMER_NONE)) {
    const u8_t valid_c =
        (consumer.valid_c.to_uint() == 0U)
            ? static_cast<u8_t>(cfg.out_c.to_uint())
            : static_cast<u8_t>(consumer.valid_c.to_uint());
    store_stats = profile_row_store_stats(dst,
                                           out_h,
                                           out_w,
                                           consumer.store_c_offset,
                                           valid_c,
                                           consumer.reserved0);
  }

  const u32_t write_words =
      store_stats.direct_words + store_stats.rmw_reads + store_stats.rmw_writes;
  const u32_t post_cycles = out_pixels * static_cast<u32_t>(2);
  const u32_t denom_rows = (out_h_u.to_uint() == 0U) ? static_cast<u32_t>(1) : out_h_u;
  const u32_t row_region_units =
      out_h_u * profile_max_u32(cache_loads / denom_rows,
                                out_w_u * (k_tiles * 2U + 2U));
  const u32_t model_units =
      cmd_execs + cache_loads + weight_words_per_conv + sa_steps + post_cycles + write_words;

  s_prof_conv_count = static_cast<u32_t>(s_prof_conv_count + 1U);
  s_prof_win_read_ops = static_cast<u32_t>(s_prof_win_read_ops + cache_loads);
  s_prof_win_words = static_cast<u32_t>(s_prof_win_words + win_words);
  s_prof_wgt_words = static_cast<u32_t>(s_prof_wgt_words + weight_words_per_conv);
  s_prof_sa_mac_steps = static_cast<u32_t>(s_prof_sa_mac_steps + sa_steps);
  s_prof_psum_words = static_cast<u32_t>(s_prof_psum_words + psum_words);
  s_prof_out_tiles = static_cast<u32_t>(s_prof_out_tiles + out_pixels);
  s_prof_out_rmw_ops =
      static_cast<u32_t>(s_prof_out_rmw_ops + store_stats.rmw_reads + store_stats.rmw_writes);
  s_prof_model_cycles = static_cast<u32_t>(s_prof_model_cycles + model_units);
  s_prof2_win_actual_reads = static_cast<u32_t>(s_prof2_win_actual_reads + cache_loads);
  s_prof2_out_direct_words =
      static_cast<u32_t>(s_prof2_out_direct_words + store_stats.direct_words);
  s_prof2_out_rmw_reads = static_cast<u32_t>(s_prof2_out_rmw_reads + store_stats.rmw_reads);
  s_prof3_wgt_cycles = static_cast<u32_t>(s_prof3_wgt_cycles + weight_words_per_conv);
  s_prof3_win_cycles = static_cast<u32_t>(s_prof3_win_cycles + cache_loads + cmd_execs);
  s_prof3_sa_cycles = static_cast<u32_t>(s_prof3_sa_cycles + sa_steps);
  s_prof3_post_cycles = static_cast<u32_t>(s_prof3_post_cycles + post_cycles);
  s_prof3_write_cycles = static_cast<u32_t>(s_prof3_write_cycles + write_words);
  s_prof3_row_region_cycles = static_cast<u32_t>(s_prof3_row_region_cycles + row_region_units);
  s_prof5_conv_model_cycles = static_cast<u32_t>(s_prof5_conv_model_cycles + model_units);
  s_prof5_total_work_units = static_cast<u32_t>(s_prof5_total_work_units + model_units);
  s_prof7_win_cmd_execs = static_cast<u32_t>(s_prof7_win_cmd_execs + cmd_execs);
  s_prof7_win_cache_loads = static_cast<u32_t>(s_prof7_win_cache_loads + cache_loads);
  s_prof7_row_regions = static_cast<u32_t>(s_prof7_row_regions + out_h_u);
  s_prof7_narrow_rmw_writes =
      static_cast<u32_t>(s_prof7_narrow_rmw_writes + store_stats.rmw_writes);

  if (consumer.mode.to_uint() == static_cast<unsigned>(ROW_CONSUMER_CAT_AFFINE_STORE)) {
    s_prof5_affine_tiles = static_cast<u32_t>(s_prof5_affine_tiles + out_pixels);
    s_prof5_nonconv_mem_ops = static_cast<u32_t>(s_prof5_nonconv_mem_ops + out_pixels * 2U);
  }
}

static u32_t profile_fixed_tiles(u16_t h, u16_t w, u16_t c) {
#pragma HLS INLINE
  const u32_t c_blocks =
      profile_ceil_div_u32(static_cast<u32_t>(c), static_cast<u32_t>(TM));
  return static_cast<u32_t>(h) * static_cast<u32_t>(w) * c_blocks;
}

static void profile_record_pool_fixed(const fixed_exec_desc_t& desc) {
#pragma HLS INLINE
  const u16_t stride = (desc.stride.to_uint() == 0U) ? static_cast<u16_t>(1) : static_cast<u16_t>(desc.stride);
  const u16_t out_h = conv_out_dim(desc.in_h, stride);
  const u16_t out_w = conv_out_dim(desc.in_w, stride);
  const u32_t tiles = profile_fixed_tiles(out_h, out_w, desc.out_c);
  s_prof5_pool_tiles = static_cast<u32_t>(s_prof5_pool_tiles + tiles);
  s_prof7_fixed_iters = static_cast<u32_t>(s_prof7_fixed_iters + tiles);
  s_prof5_nonconv_mem_ops = static_cast<u32_t>(s_prof5_nonconv_mem_ops + tiles * 2U);
  s_prof5_total_work_units = static_cast<u32_t>(s_prof5_total_work_units + tiles * 2U);
}

void profile_record_affine_fixed(const fixed_exec_desc_t& desc, const tensor_desc_t& src) {
#pragma HLS INLINE
  const u16_t valid_c = (desc.valid_c.to_uint() == 0U) ? src.c : desc.valid_c;
  const u32_t tiles = profile_fixed_tiles(src.h, src.w, valid_c);
  s_prof5_affine_tiles = static_cast<u32_t>(s_prof5_affine_tiles + tiles);
  s_prof7_fixed_iters = static_cast<u32_t>(s_prof7_fixed_iters + tiles);
  s_prof5_nonconv_mem_ops = static_cast<u32_t>(s_prof5_nonconv_mem_ops + tiles * 2U);
  s_prof5_total_work_units = static_cast<u32_t>(s_prof5_total_work_units + tiles * 2U);
}

static error_code_t run_scheduled_pool_op(const fixed_exec_desc_t& desc) {
#pragma HLS INLINE off
  tensor_desc_t src;
  tensor_desc_t dst;
  pool_q_t qparam;
  const u16_t stride = (desc.stride.to_uint() == 0U) ? static_cast<u16_t>(1) : static_cast<u16_t>(desc.stride);
  const u16_t out_h = conv_out_dim(desc.in_h, stride);
  const u16_t out_w = conv_out_dim(desc.in_w, stride);

  select_scratch_region_for_fields(static_cast<u8_t>(static_cast<unsigned>(UOP_POOL)),
                                   desc.param_id,
                                   desc.src0_tensor,
                                   desc.src1_tensor,
                                   desc.dst_tensor,
                                   desc.in_h,
                                   out_h);
  if (!resolve_tensor_read(desc.src0_tensor, src) ||
      !resolve_tensor_write(desc.dst_tensor, out_h, out_w, desc.out_c, dst)) {
    return ERR_TENSOR_DESC_RANGE;
  }
  if (!param_dma_get_pool_qparam(desc.param_id, qparam)) {
    return ERR_PARAM_DESC_RANGE;
  }
  profile_record_pool_fixed(desc);
  if (!avgpool_unit_checked(src, dst, qparam, 0)) {
    return ERR_BANK_OVERFLOW;
  }
  return ERR_NONE;
}

error_code_t pool_engine_exec(const npu_issue_t& issue) {
#pragma HLS INLINE off
  if (issue.kind.to_uint() != static_cast<unsigned>(ISSUE_POOL_AVG)) {
    return ERR_UNSUPPORTED_OPCODE;
  }
  fixed_exec_desc_t desc;
  if (!param_dma_get_fixed_exec_desc(issue.fixed_desc_id, desc) ||
      desc.kind.to_uint() != static_cast<unsigned>(EXEC_POOL)) {
    return ERR_UOP_DECODE;
  }
  return run_scheduled_pool_op(desc);
}

error_code_t upsample_engine_exec(const npu_issue_t& issue,
                                  axi_vec_t* gmem_frame_out) {
#pragma HLS INLINE off
  (void)issue;
  (void)gmem_frame_out;
  return ERR_UNSUPPORTED_OPCODE;
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
      !is_aligned_section_offset(header.reserved1[4]) ||
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

static bool csim_stop_before_logical_uop(unsigned uop_id) {
#pragma HLS INLINE
#ifdef ESP_INT8_CSIM_MAX_UOP
  return uop_id > static_cast<unsigned>(ESP_INT8_CSIM_MAX_UOP);
#else
  return false;
#endif
}

static error_code_t core_mode_run(const axi_vec_t* gmem_frame_in,
                                  axi_vec_t* gmem_frame_out,
                                  u32_t expected_uop_count,
                                  const profile_ctrl_t& profile_ctrl) {
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

  reset_scratch_state();
  frame_dma_load(gmem_frame_in);
  s_prof5_if_words = static_cast<u32_t>(s_prof5_if_words + static_cast<u32_t>(INPUT_FRAME_AXI_WORDS));
  s_prof5_frame_load_words =
      static_cast<u32_t>(s_prof5_frame_load_words + static_cast<u32_t>(INPUT_FRAME_AXI_WORDS));

  const error_code_t err = main_ctrl_run(gmem_frame_out, profile_ctrl);
  if (err != ERR_NONE) {
    return err;
  }

  return ERR_NONE;
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
  // U67 ADD + U68 AFFINE are fused into the final conv row consumer.
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

bool csim_dump_u40_prestore_row(const conv_exec_desc_t& conv_desc,
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
bool csim_dump_u40_prestore_row(const conv_exec_desc_t&,
                                u16_t,
                                const conv_cfg_t&,
                                const act_vec_t[MAX_FM_W]) {
#pragma HLS INLINE
  return true;
}
#endif

}  // namespace esp_int8

static_assert(esp_int8::INPUT_FRAME_AXI_WORDS == 49152,
              "Update gmem_frame_in m_axi depth when INPUT_FRAME_AXI_WORDS changes.");
static_assert(esp_int8::OUTPUT_FRAME_AXI_WORDS == 16384,
              "Update gmem_frame_out m_axi depth when OUTPUT_FRAME_AXI_WORDS changes.");

void espnet_encoder_int8_core(const esp_int8::axi_vec_t* gmem_frame_in,
                              esp_int8::axi_vec_t* gmem_frame_out,
                              const esp_int8::axi_vec_t* gmem_param,
                              std::uint32_t mode,
                              std::uint32_t uop_count,
                              volatile std::uint32_t& prof_uop_count,
                              volatile std::uint32_t& prof_conv_count,
                              volatile std::uint32_t& prof_win_read_ops,
                              volatile std::uint32_t& prof_win_words,
                              volatile std::uint32_t& prof_wgt_words,
                              volatile std::uint32_t& prof_sa_mac_steps,
                              volatile std::uint32_t& prof_psum_words,
                              volatile std::uint32_t& prof_out_tiles,
                              volatile std::uint32_t& prof_out_rmw_ops,
                              volatile std::uint32_t& prof_model_cycles,
                              volatile std::uint32_t& prof2_win_saved_reads,
                              volatile std::uint32_t& prof2_win_actual_reads,
                              volatile std::uint32_t& prof2_out_direct_words,
                              volatile std::uint32_t& prof2_out_rmw_reads,
                              volatile std::uint32_t& prof3_wgt_cycles,
                              volatile std::uint32_t& prof3_win_cycles,
                              volatile std::uint32_t& prof3_sa_cycles,
                              volatile std::uint32_t& prof3_post_cycles,
                              volatile std::uint32_t& prof3_write_cycles,
                              volatile std::uint32_t& prof3_row_region_cycles,
                              volatile std::uint32_t& prof5_if_words,
                              volatile std::uint32_t& prof5_frame_load_words,
                              volatile std::uint32_t& prof5_frame_store_words,
                              volatile std::uint32_t& prof5_uop_fetches,
                              volatile std::uint32_t& prof5_pool_tiles,
                              volatile std::uint32_t& prof5_affine_tiles,
                              volatile std::uint32_t& prof5_add_tiles,
                              volatile std::uint32_t& prof5_store_tiles,
                              volatile std::uint32_t& prof5_nonconv_mem_ops,
                              volatile std::uint32_t& prof5_conv_model_cycles,
                              volatile std::uint32_t& prof5_total_work_units,
                              volatile std::uint32_t& prof7_exec_conv,
                              volatile std::uint32_t& prof7_exec_pool,
                              volatile std::uint32_t& prof7_exec_affine,
                              volatile std::uint32_t& prof7_exec_store,
                              volatile std::uint32_t& prof7_exec_add_affine,
                              volatile std::uint32_t& prof7_win_cmd_execs,
                              volatile std::uint32_t& prof7_win_cache_loads,
                              volatile std::uint32_t& prof7_row_regions,
                              volatile std::uint32_t& prof7_fixed_iters,
                              volatile std::uint32_t& prof7_narrow_rmw_writes,
                              volatile std::uint32_t& prof7_upsample_rows,
                              volatile std::uint32_t& prof7_win_interpreter_rows) {
#ifdef ESP_INT8_COSIM_LITE
#pragma HLS INTERFACE ap_memory port=gmem_frame_in depth=49152
#pragma HLS INTERFACE ap_memory port=gmem_frame_out depth=16384
#pragma HLS INTERFACE ap_memory port=gmem_param depth=8192
#pragma HLS INTERFACE ap_none port=mode
#pragma HLS INTERFACE ap_none port=uop_count
#pragma HLS INTERFACE ap_none port=prof_uop_count
#pragma HLS INTERFACE ap_none port=prof_conv_count
#pragma HLS INTERFACE ap_none port=prof_win_read_ops
#pragma HLS INTERFACE ap_none port=prof_win_words
#pragma HLS INTERFACE ap_none port=prof_wgt_words
#pragma HLS INTERFACE ap_none port=prof_sa_mac_steps
#pragma HLS INTERFACE ap_none port=prof_psum_words
#pragma HLS INTERFACE ap_none port=prof_out_tiles
#pragma HLS INTERFACE ap_none port=prof_out_rmw_ops
#pragma HLS INTERFACE ap_none port=prof_model_cycles
#pragma HLS INTERFACE ap_none port=prof2_win_saved_reads
#pragma HLS INTERFACE ap_none port=prof2_win_actual_reads
#pragma HLS INTERFACE ap_none port=prof2_out_direct_words
#pragma HLS INTERFACE ap_none port=prof2_out_rmw_reads
#pragma HLS INTERFACE ap_none port=prof3_wgt_cycles
#pragma HLS INTERFACE ap_none port=prof3_win_cycles
#pragma HLS INTERFACE ap_none port=prof3_sa_cycles
#pragma HLS INTERFACE ap_none port=prof3_post_cycles
#pragma HLS INTERFACE ap_none port=prof3_write_cycles
#pragma HLS INTERFACE ap_none port=prof3_row_region_cycles
#pragma HLS INTERFACE ap_none port=prof5_if_words
#pragma HLS INTERFACE ap_none port=prof5_frame_load_words
#pragma HLS INTERFACE ap_none port=prof5_frame_store_words
#pragma HLS INTERFACE ap_none port=prof5_uop_fetches
#pragma HLS INTERFACE ap_none port=prof5_pool_tiles
#pragma HLS INTERFACE ap_none port=prof5_affine_tiles
#pragma HLS INTERFACE ap_none port=prof5_add_tiles
#pragma HLS INTERFACE ap_none port=prof5_store_tiles
#pragma HLS INTERFACE ap_none port=prof5_nonconv_mem_ops
#pragma HLS INTERFACE ap_none port=prof5_conv_model_cycles
#pragma HLS INTERFACE ap_none port=prof5_total_work_units
#pragma HLS INTERFACE ap_none port=prof7_exec_conv
#pragma HLS INTERFACE ap_none port=prof7_exec_pool
#pragma HLS INTERFACE ap_none port=prof7_exec_affine
#pragma HLS INTERFACE ap_none port=prof7_exec_store
#pragma HLS INTERFACE ap_none port=prof7_exec_add_affine
#pragma HLS INTERFACE ap_none port=prof7_win_cmd_execs
#pragma HLS INTERFACE ap_none port=prof7_win_cache_loads
#pragma HLS INTERFACE ap_none port=prof7_row_regions
#pragma HLS INTERFACE ap_none port=prof7_fixed_iters
#pragma HLS INTERFACE ap_none port=prof7_narrow_rmw_writes
#pragma HLS INTERFACE ap_none port=prof7_upsample_rows
#pragma HLS INTERFACE ap_none port=prof7_win_interpreter_rows
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
#pragma HLS INTERFACE s_axilite port=prof_uop_count bundle=control
#pragma HLS INTERFACE s_axilite port=prof_conv_count bundle=control
#pragma HLS INTERFACE s_axilite port=prof_win_read_ops bundle=control
#pragma HLS INTERFACE s_axilite port=prof_win_words bundle=control
#pragma HLS INTERFACE s_axilite port=prof_wgt_words bundle=control
#pragma HLS INTERFACE s_axilite port=prof_sa_mac_steps bundle=control
#pragma HLS INTERFACE s_axilite port=prof_psum_words bundle=control
#pragma HLS INTERFACE s_axilite port=prof_out_tiles bundle=control
#pragma HLS INTERFACE s_axilite port=prof_out_rmw_ops bundle=control
#pragma HLS INTERFACE s_axilite port=prof_model_cycles bundle=control
#pragma HLS INTERFACE s_axilite port=prof2_win_saved_reads bundle=control
#pragma HLS INTERFACE s_axilite port=prof2_win_actual_reads bundle=control
#pragma HLS INTERFACE s_axilite port=prof2_out_direct_words bundle=control
#pragma HLS INTERFACE s_axilite port=prof2_out_rmw_reads bundle=control
#pragma HLS INTERFACE s_axilite port=prof3_wgt_cycles bundle=control
#pragma HLS INTERFACE s_axilite port=prof3_win_cycles bundle=control
#pragma HLS INTERFACE s_axilite port=prof3_sa_cycles bundle=control
#pragma HLS INTERFACE s_axilite port=prof3_post_cycles bundle=control
#pragma HLS INTERFACE s_axilite port=prof3_write_cycles bundle=control
#pragma HLS INTERFACE s_axilite port=prof3_row_region_cycles bundle=control
#pragma HLS INTERFACE s_axilite port=prof5_if_words bundle=control
#pragma HLS INTERFACE s_axilite port=prof5_frame_load_words bundle=control
#pragma HLS INTERFACE s_axilite port=prof5_frame_store_words bundle=control
#pragma HLS INTERFACE s_axilite port=prof5_uop_fetches bundle=control
#pragma HLS INTERFACE s_axilite port=prof5_pool_tiles bundle=control
#pragma HLS INTERFACE s_axilite port=prof5_affine_tiles bundle=control
#pragma HLS INTERFACE s_axilite port=prof5_add_tiles bundle=control
#pragma HLS INTERFACE s_axilite port=prof5_store_tiles bundle=control
#pragma HLS INTERFACE s_axilite port=prof5_nonconv_mem_ops bundle=control
#pragma HLS INTERFACE s_axilite port=prof5_conv_model_cycles bundle=control
#pragma HLS INTERFACE s_axilite port=prof5_total_work_units bundle=control
#pragma HLS INTERFACE s_axilite port=prof7_exec_conv bundle=control
#pragma HLS INTERFACE s_axilite port=prof7_exec_pool bundle=control
#pragma HLS INTERFACE s_axilite port=prof7_exec_affine bundle=control
#pragma HLS INTERFACE s_axilite port=prof7_exec_store bundle=control
#pragma HLS INTERFACE s_axilite port=prof7_exec_add_affine bundle=control
#pragma HLS INTERFACE s_axilite port=prof7_win_cmd_execs bundle=control
#pragma HLS INTERFACE s_axilite port=prof7_win_cache_loads bundle=control
#pragma HLS INTERFACE s_axilite port=prof7_row_regions bundle=control
#pragma HLS INTERFACE s_axilite port=prof7_fixed_iters bundle=control
#pragma HLS INTERFACE s_axilite port=prof7_narrow_rmw_writes bundle=control
#pragma HLS INTERFACE s_axilite port=prof7_upsample_rows bundle=control
#pragma HLS INTERFACE s_axilite port=prof7_win_interpreter_rows bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#endif

  esp_int8::profile_clear();
  esp_int8::profile_publish(prof_uop_count,
                            prof_conv_count,
                            prof_win_read_ops,
                            prof_win_words,
                            prof_wgt_words,
                            prof_sa_mac_steps,
                            prof_psum_words,
                            prof_out_tiles,
                            prof_out_rmw_ops,
                            prof_model_cycles,
                            prof2_win_saved_reads,
                            prof2_win_actual_reads,
                            prof2_out_direct_words,
                            prof2_out_rmw_reads,
                            prof3_wgt_cycles,
                            prof3_win_cycles,
                            prof3_sa_cycles,
                            prof3_post_cycles,
                            prof3_write_cycles,
                            prof3_row_region_cycles,
                            prof5_if_words,
                            prof5_frame_load_words,
                            prof5_frame_store_words,
                            prof5_uop_fetches,
                            prof5_pool_tiles,
                            prof5_affine_tiles,
                            prof5_add_tiles,
                            prof5_store_tiles,
                            prof5_nonconv_mem_ops,
                            prof5_conv_model_cycles,
                            prof5_total_work_units,
                            prof7_exec_conv,
                            prof7_exec_pool,
                            prof7_exec_affine,
                            prof7_exec_store,
                            prof7_exec_add_affine,
                            prof7_win_cmd_execs,
                            prof7_win_cache_loads,
                            prof7_row_regions,
                            prof7_fixed_iters,
                            prof7_narrow_rmw_writes,
                            prof7_upsample_rows,
                            prof7_win_interpreter_rows);
  const std::uint32_t mode_runtime = esp_int8::runtime_mode(mode);
  const esp_int8::u32_t expected_uop_count = esp_int8::runtime_expected_uop_count(uop_count);
  const esp_int8::profile_ctrl_t profile_ctrl = esp_int8::runtime_profile_ctrl(mode, uop_count);

  switch (mode_runtime) {
    case esp_int8::MODE_INIT:
      esp_int8::core_mode_init(gmem_param);
      break;
    case esp_int8::MODE_RUN:
      esp_int8::core_mode_run(gmem_frame_in,
                              gmem_frame_out,
                              expected_uop_count,
                              profile_ctrl);
      break;
    case esp_int8::MODE_IDLE:
      break;
    default:
      break;
  }
  esp_int8::profile_publish(prof_uop_count,
                            prof_conv_count,
                            prof_win_read_ops,
                            prof_win_words,
                            prof_wgt_words,
                            prof_sa_mac_steps,
                            prof_psum_words,
                            prof_out_tiles,
                            prof_out_rmw_ops,
                            prof_model_cycles,
                            prof2_win_saved_reads,
                            prof2_win_actual_reads,
                            prof2_out_direct_words,
                            prof2_out_rmw_reads,
                            prof3_wgt_cycles,
                            prof3_win_cycles,
                            prof3_sa_cycles,
                            prof3_post_cycles,
                            prof3_write_cycles,
                            prof3_row_region_cycles,
                            prof5_if_words,
                            prof5_frame_load_words,
                            prof5_frame_store_words,
                            prof5_uop_fetches,
                            prof5_pool_tiles,
                            prof5_affine_tiles,
                            prof5_add_tiles,
                            prof5_store_tiles,
                            prof5_nonconv_mem_ops,
                            prof5_conv_model_cycles,
                            prof5_total_work_units,
                            prof7_exec_conv,
                            prof7_exec_pool,
                            prof7_exec_affine,
                            prof7_exec_store,
                            prof7_exec_add_affine,
                            prof7_win_cmd_execs,
                            prof7_win_cache_loads,
                            prof7_row_regions,
                            prof7_fixed_iters,
                            prof7_narrow_rmw_writes,
                            prof7_upsample_rows,
                            prof7_win_interpreter_rows);
}
