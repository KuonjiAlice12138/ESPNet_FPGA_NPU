#include "../include/npu_config.hpp"
#include "../include/npu_ctrl.hpp"
#include "../include/npu_q.hpp"
#include "../include/npu_schedule.hpp"
#include "../include/npu_uop.hpp"

namespace esp_int8 {

bool param_dma_get_conv_qparam(u8_t param_id, conv_q_t& qparam);
bool param_dma_get_conv_exec_desc(u8_t id, conv_exec_desc_t& desc);
bool param_dma_get_fixed_exec_desc(u8_t id, fixed_exec_desc_t& desc);
bool param_dma_get_block5_sched(u8_t id, block5_sched_desc_t& desc);
bool param_dma_get_window_sched(u8_t id, window_sched_desc_t& desc);
bool param_dma_get_row_consumer(u8_t id, row_consumer_desc_t& desc);
bool param_dma_get_packed_weight_vec(const conv_exec_desc_t& desc,
                                     u16_t tm,
                                     u16_t kt,
                                     wgt_vec_t& word);
bool param_dma_get_add_qparam(u8_t param_id, add_q_t& qparam);
void select_scratch_region_for_fields(u8_t opcode,
                                      u8_t param_id,
                                      u8_t src0_tensor,
                                      u8_t src1_tensor,
                                      u8_t dst_tensor,
                                      u16_t in_h,
                                      u16_t out_h);
bool resolve_tensor_read(u8_t tensor_id, tensor_desc_t& desc);
bool resolve_tensor_write(u8_t tensor_id,
                          u16_t h,
                          u16_t w,
                          u16_t c,
                          tensor_desc_t& desc);
bool resolve_block5_scratch_descs(u16_t pattern,
                                  u8_t first_branch_param,
                                  u16_t rows,
                                  u16_t out_w,
                                  tensor_desc_t& s0,
                                  tensor_desc_t& s1,
                                  tensor_desc_t& s2,
                                  tensor_desc_t& s3,
                                  tensor_desc_t& s4);
bool alias_tensor_to_slice(u8_t tensor_id,
                           const tensor_desc_t& base_desc,
                           u16_t c_offset,
                           u16_t c);
void upsample_fused_begin();
void scheduled_window_generator_row(const tensor_desc_t& src_desc,
                                    const conv_exec_desc_t& conv_desc,
                                    const window_sched_desc_t& sched,
                                    hls::stream<act_vec_t>& act_stream,
                                    u16_t out_row);
void systolic_array_core_row(hls::stream<act_vec_t>& act_stream,
                             const wgt_vec_t weight_buf[TM][MAX_K_TILE_COUNT],
                             hls::stream<psum_half_vec_t>& psum_stream,
                             const conv_cfg_t& cfg);
void profile_record_conv_desc(const conv_cfg_t& cfg,
                              const conv_exec_desc_t& conv_desc,
                              const window_sched_desc_t& sched,
                              const row_consumer_desc_t& consumer,
                              const tensor_desc_t& dst);
bool ppu_consume_conv_row(const row_consumer_desc_t& consumer,
                          const tensor_desc_t& dst,
                          const tensor_desc_t& add_other,
                          bool has_add_other,
                          const add_q_t& add_qparam,
                          const conv_cfg_t& cfg,
                          axi_vec_t* gmem_frame_out,
                          u16_t out_row,
                          act_vec_t row_buf[MAX_FM_W]);
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
                                  act_vec_t row_buf[MAX_FM_W]);
bool csim_dump_u40_prestore_row(const conv_exec_desc_t& conv_desc,
                                u16_t out_row,
                                const conv_cfg_t& cfg,
                                const act_vec_t row_buf[MAX_FM_W]);

static act_vec_t s_shared_conv_row_buf[MAX_FM_W];
static wgt_vec_t s_shared_weight_buf[TM][MAX_K_TILE_COUNT];

enum conv_issue_kind_t {
  CONV_ISSUE_NORMAL = 0,
  CONV_ISSUE_BLOCK5_BRANCH = 1,
};

struct conv_issue_t {
  u8_t kind;
  u8_t conv_desc_id;
  u8_t src_tensor;
  u8_t branch_idx;
  u8_t block5_pattern;
  u8_t block5_sched_id;
  u8_t fixed_desc_id;
  u8_t block5_chain_add_base;
  u8_t block5_scratch_first_param;
  u8_t reserved0;
  u16_t conv_row_base;
  u16_t consumer_row_base;
  u16_t row_count;
  u16_t block5_out_h;
  u16_t block5_out_w;
};

struct conv_issue_result_t {
  conv_exec_desc_t conv_desc;
  row_consumer_desc_t consumer;
  tensor_desc_t dst;
  bool emit_fullres_mask;
};

struct conv_rows_task_t {
  tensor_desc_t src;
  tensor_desc_t dst;
  tensor_desc_t add_other;
  tensor_desc_t block5_scratch0;
  tensor_desc_t block5_scratch1;
  tensor_desc_t block5_scratch2;
  tensor_desc_t block5_scratch3;
  tensor_desc_t block5_prev_branch;
  tensor_desc_t block5_residual;
  tensor_desc_t block5_final_dst;
  conv_cfg_t cfg;
  conv_exec_desc_t conv_desc;
  window_sched_desc_t sched;
  block5_sched_desc_t block5_sched;
  row_consumer_desc_t consumer;
  conv_q_t qparam;
  add_q_t add_qparam;
  add_q_t block5_residual_add_qparam;
  u16_t conv_row_base;
  u16_t consumer_row_base;
  u16_t row_count;
  bool has_add_other;
  bool block5_finalize;
  bool block5_has_residual;
  u8_t block5_act_type;
};

static u16_t ceil_div_u16(u16_t a, u16_t b) {
#pragma HLS INLINE
  return static_cast<u16_t>((a + b - 1) / b);
}

static u16_t conv_out_dim(u16_t in_size, u16_t stride) {
#pragma HLS INLINE
  return ceil_div_u16(in_size, stride);
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

static u16_t conv_effective_stride(const conv_cfg_t& cfg) {
#pragma HLS INLINE
  return (cfg.stride == 0) ? static_cast<u16_t>(1) : static_cast<u16_t>(cfg.stride);
}

static void set_act_vec_i8_dynamic(act_vec_t& word, int lane, i8_t value) {
#pragma HLS INLINE
  u8_t raw = 0;
  raw.range(7, 0) = value.range(7, 0);
  const act_vec_t widened = static_cast<act_vec_t>(raw);
  const act_vec_t mask = static_cast<act_vec_t>(static_cast<act_vec_t>(0xff) << (lane * 8));
  word = static_cast<act_vec_t>((word & ~mask) | static_cast<act_vec_t>(widened << (lane * 8)));
}

static i32_t get_psum_half_i32(const psum_half_vec_t& word, int lane) {
#pragma HLS INLINE
  i32_t value;
  value.range(31, 0) = word.range(lane * 32 + 31, lane * 32);
  return value;
}

static u16_t block5_compact_layout(u8_t channels) {
#pragma HLS INLINE
  const unsigned c = channels.to_uint();
  if (c == 12U) {
    return static_cast<u16_t>(static_cast<unsigned>(STORE_LAYOUT_COMPACT_C12));
  }
  if (c == 16U) {
    return static_cast<u16_t>(static_cast<unsigned>(STORE_LAYOUT_COMPACT_C16));
  }
  if (c == 25U) {
    return static_cast<u16_t>(static_cast<unsigned>(STORE_LAYOUT_COMPACT_C25));
  }
  if (c == 28U) {
    return static_cast<u16_t>(static_cast<unsigned>(STORE_LAYOUT_COMPACT_C28));
  }
  return static_cast<u16_t>(static_cast<unsigned>(STORE_LAYOUT_NONE));
}

static void post_process_row_to_buffer(hls::stream<psum_half_vec_t>& psum_stream,
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
    const psum_half_vec_t psum_word0 = psum_stream.read();
    const psum_half_vec_t psum_word1 = psum_stream.read();
    for (int tm = 0; tm < TM; ++tm) {
#pragma HLS UNROLL factor=4
      if (tm < out_c_i) {
        const int half = TM / 2;
        const psum_half_vec_t& word = (tm < half) ? psum_word0 : psum_word1;
        const i32_t psum = get_psum_half_i32(word, (tm < half) ? tm : (tm - half));
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

static void shared_conv_row_engine(const tensor_desc_t& src,
                                   const conv_cfg_t& cfg,
                                   const conv_exec_desc_t& conv_desc,
                                   const window_sched_desc_t& sched,
                                   u8_t act_type,
                                   const conv_q_t& qparam,
                                   u16_t out_row,
                                   act_vec_t row_buf[MAX_FM_W],
                                   const wgt_vec_t weight_buf[TM][MAX_K_TILE_COUNT]) {
#pragma HLS INLINE off
  hls::stream<act_vec_t> act_stream;
  hls::stream<psum_half_vec_t> psum_stream;
#pragma HLS STREAM variable=act_stream depth=64
#pragma HLS STREAM variable=psum_stream depth=16
#pragma HLS BIND_STORAGE variable=act_stream type=fifo impl=bram
#pragma HLS BIND_STORAGE variable=psum_stream type=fifo impl=bram
#pragma HLS DATAFLOW
  generate_conv_window_row(src, act_stream, cfg, conv_desc, sched, out_row);
  systolic_array_core_row(act_stream, weight_buf, psum_stream, cfg);
  post_process_row_to_buffer(psum_stream, row_buf, cfg, act_type, qparam);
}

static void load_conv_weight_buffer(const conv_exec_desc_t& conv_desc,
                                    wgt_vec_t weight_buf[TM][MAX_K_TILE_COUNT]) {
#pragma HLS INLINE off
  const int k_tiles_i = static_cast<int>(conv_desc.k_tiles.to_uint());
  for (int kt = 0; kt < MAX_K_TILE_COUNT; ++kt) {
    if (kt >= k_tiles_i) {
      break;
    }
    const u16_t k_tile = static_cast<u16_t>(kt);
    for (int tm = 0; tm < TM; ++tm) {
#pragma HLS PIPELINE off
      const u16_t oc = static_cast<u16_t>(tm);
      wgt_vec_t word = 0;
      param_dma_get_packed_weight_vec(conv_desc, oc, k_tile, word);
      weight_buf[tm][kt] = word;
    }
  }
}

static bool run_conv_rows_task(const conv_rows_task_t& task, axi_vec_t* gmem_frame_out) {
#pragma HLS INLINE off
#pragma HLS BIND_STORAGE variable=s_shared_conv_row_buf type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=s_shared_weight_buf cyclic factor=16 dim=1
#pragma HLS BIND_STORAGE variable=s_shared_weight_buf type=ram_2p impl=bram
  load_conv_weight_buffer(task.conv_desc, s_shared_weight_buf);

  if (task.consumer.mode.to_uint() == static_cast<unsigned>(ROW_CONSUMER_UPSAMPLE_OUT)) {
    upsample_fused_begin();
  }

  bool write_ok = true;
  const int row_count_i = static_cast<int>(task.row_count.to_uint());
  for (int oh_i = 0; oh_i < MAX_FM_H; ++oh_i) {
    if (oh_i >= row_count_i) {
      break;
    }
    const u16_t conv_row =
        static_cast<u16_t>(task.conv_row_base.to_uint() + static_cast<unsigned>(oh_i));
    const u16_t consumer_row =
        static_cast<u16_t>(task.consumer_row_base.to_uint() + static_cast<unsigned>(oh_i));
    shared_conv_row_engine(task.src,
                           task.cfg,
                           task.conv_desc,
                           task.sched,
                           conv_act_type_from_flags(task.conv_desc.flags),
                           task.qparam,
                           conv_row,
                           s_shared_conv_row_buf,
                           s_shared_weight_buf);
    if (!csim_dump_u40_prestore_row(task.conv_desc, conv_row, task.cfg, s_shared_conv_row_buf)) {
      write_ok = false;
    }
    const bool ppu_ok =
        task.block5_finalize
            ? ppu_consume_block5_final_row(task.block5_sched,
                                           task.block5_scratch0,
                                           task.block5_scratch1,
                                           task.block5_scratch2,
                                           task.block5_scratch3,
                                           task.block5_prev_branch,
                                           task.block5_residual,
                                           task.block5_final_dst,
                                           task.block5_has_residual,
                                           task.add_qparam,
                                           task.block5_residual_add_qparam,
                                           task.cfg,
                                           task.block5_act_type,
                                           consumer_row,
                                           conv_row,
                                           s_shared_conv_row_buf)
            : ppu_consume_conv_row(task.consumer,
                                   task.dst,
                                   task.add_other,
                                   task.has_add_other,
                                   task.add_qparam,
                                   task.cfg,
                                   gmem_frame_out,
                                   consumer_row,
                                   s_shared_conv_row_buf);
    if (!ppu_ok) {
      write_ok = false;
    }
  }
  return write_ok;
}

static bool build_normal_conv_task(const conv_issue_t& issue,
                                   conv_rows_task_t& task,
                                   conv_issue_result_t& result) {
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

  if (!param_dma_get_conv_exec_desc(issue.conv_desc_id, conv_desc) ||
      !param_dma_get_window_sched(conv_desc.window_sched_id, sched) ||
      !param_dma_get_row_consumer(conv_desc.row_consumer_id, consumer)) {
    return false;
  }

  const conv_cfg_t cfg = conv_cfg_from_exec_desc(conv_desc);
  const u16_t stride = conv_effective_stride(cfg);
  const u16_t out_h = conv_out_dim(cfg.in_h, stride);
  const u16_t out_w = conv_out_dim(cfg.in_w, stride);

  select_scratch_region_for_fields(static_cast<u8_t>(static_cast<unsigned>(UOP_CONV)),
                                   conv_desc.param_id,
                                   conv_desc.src_tensor,
                                   static_cast<u8_t>(static_cast<unsigned>(TID_INVALID)),
                                   conv_desc.dst_tensor,
                                   conv_desc.in_h,
                                   out_h);
  if (!resolve_tensor_read(conv_desc.src_tensor, src) ||
      !param_dma_get_conv_qparam(conv_desc.qparam_id, qparam)) {
    return false;
  }

  if (cfg.out_c.to_uint() > static_cast<unsigned>(TM)) {
    return false;
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
  if (consumer_mode == static_cast<unsigned>(ROW_CONSUMER_CAT_AFFINE_STORE)) {
    if (!resolve_tensor_read(consumer.add_other_tensor, add_other)) {
      return false;
    }
    has_add_other = true;
  } else if (consumer_mode != static_cast<unsigned>(ROW_CONSUMER_NONE) &&
             consumer_mode != static_cast<unsigned>(ROW_CONSUMER_UPSAMPLE_OUT)) {
    return false;
  }

  if (emit_fullres_mask) {
    if (cfg.out_c.to_uint() != static_cast<unsigned>(ENCODER_OUT_C) ||
        out_h.to_uint() != static_cast<unsigned>(ENCODER_OUT_H) ||
        out_w.to_uint() != static_cast<unsigned>(ENCODER_OUT_W)) {
      return false;
    }
    dst = tensor_desc_t();
  } else {
    const u16_t dst_required_c =
        static_cast<u16_t>(consumer.store_c_offset + consumer.valid_c);
    if (!resolve_tensor_write(consumer.store_dst_tensor, out_h, out_w, dst_required_c, dst)) {
      return false;
    }
    if (tensor_is_global(consumer.store_dst_tensor) &&
        dst.c.to_uint() < consumer.store_c_offset.to_uint() + consumer.valid_c.to_uint()) {
      return false;
    }
  }

  profile_record_conv_desc(cfg, conv_desc, sched, consumer, dst);

  task = conv_rows_task_t();
  task.src = src;
  task.dst = dst;
  task.add_other = add_other;
  task.cfg = cfg;
  task.conv_desc = conv_desc;
  task.sched = sched;
  task.consumer = consumer;
  task.qparam = qparam;
  task.add_qparam = add_qparam;
  task.conv_row_base = static_cast<u16_t>(0);
  task.consumer_row_base = static_cast<u16_t>(0);
  task.row_count = out_h;
  task.has_add_other = has_add_other;

  result.conv_desc = conv_desc;
  result.consumer = consumer;
  result.dst = dst;
  result.emit_fullres_mask = emit_fullres_mask;
  return true;
}

static bool build_block5_branch_task(const conv_issue_t& issue, conv_rows_task_t& task) {
#pragma HLS INLINE off
  conv_exec_desc_t conv_desc;
  window_sched_desc_t sched;
  block5_sched_desc_t bsched;
  fixed_exec_desc_t fixed_desc;
  tensor_desc_t src;
  tensor_desc_t scratch_desc[5];
  tensor_desc_t residual;
  tensor_desc_t final_dst;
  conv_q_t qparam;
  add_q_t add_qparam = add_q_t();
  add_q_t residual_add_qparam = add_q_t();
#pragma HLS ARRAY_PARTITION variable=qparam.bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=qparam.mult complete dim=1
#pragma HLS ARRAY_PARTITION variable=qparam.shift complete dim=1

  if (!param_dma_get_conv_exec_desc(issue.conv_desc_id, conv_desc) ||
      !param_dma_get_window_sched(conv_desc.window_sched_id, sched) ||
      !param_dma_get_block5_sched(issue.block5_sched_id, bsched) ||
      !param_dma_get_fixed_exec_desc(issue.fixed_desc_id, fixed_desc) ||
      !param_dma_get_conv_qparam(conv_desc.qparam_id, qparam) ||
      !resolve_tensor_read(issue.src_tensor, src)) {
    return false;
  }
  if (bsched.pattern.to_uint() != issue.block5_pattern.to_uint() ||
      bsched.branch_count.to_uint() != 5U ||
      bsched.first_branch_conv_id.to_uint() + issue.branch_idx.to_uint() != issue.conv_desc_id.to_uint() ||
      bsched.out_w.to_uint() != issue.block5_out_w.to_uint() ||
      bsched.out_h.to_uint() != issue.block5_out_h.to_uint()) {
    return false;
  }
  if (!resolve_block5_scratch_descs(static_cast<u16_t>(static_cast<unsigned>(issue.block5_pattern.to_uint())),
                                    issue.block5_scratch_first_param,
                                    issue.row_count,
                                    issue.block5_out_w,
                                    scratch_desc[0],
                                    scratch_desc[1],
                                    scratch_desc[2],
                                    scratch_desc[3],
                                    scratch_desc[4])) {
    return false;
  }

  const unsigned branch = issue.branch_idx.to_uint();
  if (branch >= 5U) {
    return false;
  }
  const bool final_branch = branch == 4U;
  const u16_t store_layout =
      block5_compact_layout(static_cast<u8_t>(conv_desc.out_c.to_uint()));
  if (store_layout.to_uint() == static_cast<unsigned>(STORE_LAYOUT_NONE)) {
    return false;
  }

  tensor_desc_t prev;
  bool add_prev = branch >= 2U;
  if (add_prev) {
    prev = scratch_desc[branch - 1U];
    const u8_t add_id =
        static_cast<u8_t>(issue.block5_chain_add_base.to_uint() + branch - 2U);
    if (!param_dma_get_add_qparam(add_id, add_qparam)) {
      return false;
    }
  }

  bool has_residual = false;
  if (final_branch) {
    const unsigned pattern = issue.block5_pattern.to_uint();
    const bool is_l2 = pattern == static_cast<unsigned>(BLOCK5_PATTERN_L2_C16_4C12);
    const bool is_l3 = pattern == static_cast<unsigned>(BLOCK5_PATTERN_L3_C28_4C25);
    if ((!is_l2 && !is_l3) ||
        (is_l2 && bsched.finalizer_kind.to_uint() != static_cast<unsigned>(BLOCK5_FINALIZER_L2)) ||
        (is_l3 && bsched.finalizer_kind.to_uint() != static_cast<unsigned>(BLOCK5_FINALIZER_L3)) ||
        fixed_desc.valid_c.to_uint() != bsched.valid_c.to_uint()) {
      return false;
    }
    if (!resolve_tensor_write(bsched.dst_tensor,
                              bsched.out_h,
                              bsched.out_w,
                              bsched.valid_c,
                              final_dst)) {
      return false;
    }
    has_residual = bsched.add_tensor.to_uint() != static_cast<unsigned>(TID_INVALID);
    if (has_residual &&
        (!resolve_tensor_read(bsched.add_tensor, residual) ||
         !param_dma_get_add_qparam(bsched.residual_add_qparam_id, residual_add_qparam))) {
      return false;
    }
  }

  row_consumer_desc_t consumer = row_consumer_desc_t();
  consumer.mode = static_cast<u8_t>(static_cast<unsigned>(ROW_CONSUMER_NONE));
  consumer.store_dst_tensor = conv_desc.dst_tensor;
  consumer.store_c_offset = static_cast<u16_t>(0);
  consumer.valid_c = conv_desc.out_c;
  consumer.reserved0 = store_layout;

  task = conv_rows_task_t();
  task.src = src;
  task.dst = scratch_desc[branch];
  task.add_other = prev;
  task.cfg = conv_cfg_from_exec_desc(conv_desc);
  task.conv_desc = conv_desc;
  task.sched = sched;
  task.consumer = consumer;
  task.qparam = qparam;
  task.add_qparam = add_qparam;
  task.block5_residual_add_qparam = residual_add_qparam;
  task.conv_row_base = issue.conv_row_base;
  task.consumer_row_base = issue.consumer_row_base;
  task.row_count = issue.row_count;
  task.has_add_other = add_prev && !final_branch;
  task.block5_finalize = final_branch;
  task.block5_has_residual = has_residual;
  task.block5_act_type = fixed_desc.act_type;
  task.block5_sched = bsched;
  task.block5_scratch0 = scratch_desc[0];
  task.block5_scratch1 = scratch_desc[1];
  task.block5_scratch2 = scratch_desc[2];
  task.block5_scratch3 = scratch_desc[3];
  task.block5_prev_branch = prev;
  task.block5_residual = residual;
  task.block5_final_dst = final_dst;
  return true;
}

static bool build_conv_task_from_issue(const conv_issue_t& issue,
                                       conv_rows_task_t& task,
                                       conv_issue_result_t& result) {
#pragma HLS INLINE off
  result = conv_issue_result_t();
  if (issue.kind.to_uint() == static_cast<unsigned>(CONV_ISSUE_NORMAL)) {
    return build_normal_conv_task(issue, task, result);
  }
  if (issue.kind.to_uint() == static_cast<unsigned>(CONV_ISSUE_BLOCK5_BRANCH)) {
    return build_block5_branch_task(issue, task);
  }
  return false;
}

static bool run_conv_issue_once(const conv_issue_t& issue,
                                axi_vec_t* gmem_frame_out,
                                conv_issue_result_t& result) {
#pragma HLS INLINE off
  conv_rows_task_t task;
  if (!build_conv_task_from_issue(issue, task, result)) {
    return false;
  }
  return run_conv_rows_task(task, gmem_frame_out);
}

error_code_t conv_engine_exec(const npu_issue_t& issue,
                              axi_vec_t* gmem_frame_out) {
#pragma HLS INLINE off
  conv_issue_t conv_issue = conv_issue_t();
  conv_issue_result_t result;

  if (issue.kind.to_uint() == static_cast<unsigned>(ISSUE_CONV_NORMAL)) {
    conv_issue.kind = static_cast<u8_t>(static_cast<unsigned>(CONV_ISSUE_NORMAL));
    conv_issue.conv_desc_id = issue.exec_desc_id;
  } else if (issue.kind.to_uint() == static_cast<unsigned>(ISSUE_CONV_BLOCK5_BRANCH)) {
    conv_issue.kind = static_cast<u8_t>(static_cast<unsigned>(CONV_ISSUE_BLOCK5_BRANCH));
    conv_issue.conv_desc_id = issue.exec_desc_id;
    conv_issue.src_tensor = issue.src0_tensor;
    conv_issue.branch_idx = issue.branch_idx;
    conv_issue.block5_pattern = issue.pattern;
    conv_issue.block5_sched_id = issue.block5_sched_id;
    conv_issue.fixed_desc_id = issue.fixed_desc_id;
    conv_issue.block5_chain_add_base = issue.qparam_id;
    conv_issue.block5_scratch_first_param = issue.scratch_slot;
    conv_issue.conv_row_base = issue.row_begin;
    conv_issue.consumer_row_base = static_cast<u16_t>(0);
    conv_issue.row_count = issue.row_count;
    conv_issue.block5_out_h = issue.out_h;
    conv_issue.block5_out_w = issue.out_w;
  } else {
    return ERR_UNSUPPORTED_OPCODE;
  }

  if (!run_conv_issue_once(conv_issue, gmem_frame_out, result)) {
    return ERR_BANK_OVERFLOW;
  }

  if (issue.kind.to_uint() != static_cast<unsigned>(ISSUE_CONV_NORMAL)) {
    return ERR_NONE;
  }

  const unsigned consumer_mode = result.consumer.mode.to_uint();
  if (!result.emit_fullres_mask) {
    (void)consumer_mode;
    if (result.consumer.alias_tensor.to_uint() != static_cast<unsigned>(TID_INVALID) &&
        !alias_tensor_to_slice(result.consumer.alias_tensor,
                               result.dst,
                               result.consumer.store_c_offset,
                               result.consumer.valid_c)) {
      return ERR_TENSOR_DESC_RANGE;
    }
  }

  return ERR_NONE;
}

} // namespace esp_int8
