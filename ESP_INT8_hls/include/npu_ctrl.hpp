#ifndef ESP_INT8_NPU_CTRL_HPP
#define ESP_INT8_NPU_CTRL_HPP

#include "npu_config.hpp"
#include "npu_schedule.hpp"
#include "npu_types.hpp"

namespace esp_int8 {

enum npu_engine_t : unsigned {
  NPU_ENGINE_NONE = 0,
  NPU_ENGINE_CONV = 1,
  NPU_ENGINE_VEC = 2,
  NPU_ENGINE_POOL = 3,
};

enum npu_issue_kind_t : unsigned {
  ISSUE_CONV_NORMAL = 1,
  ISSUE_CONV_BLOCK5_BRANCH = 2,

  ISSUE_VEC_AFFINE = 11,

  ISSUE_POOL_AVG = 20,
};

enum npu_profile_stage_t : unsigned {
  PROF_STAGE_IDLE = 0,
  PROF_STAGE_PARAM_INIT = 1,
  PROF_STAGE_FRAME_LOAD = 2,
  PROF_STAGE_MAIN_CTRL = 3,
  PROF_STAGE_CONV_WEIGHT_LOAD = 4,
  PROF_STAGE_CONV_ROW_DATAPATH = 5,
  PROF_STAGE_PPU_ROW_CONSUME = 6,
  PROF_STAGE_PPU_BLOCK5_FINAL = 7,
  PROF_STAGE_VEC_FIXED = 8,
  PROF_STAGE_AVGPOOL = 9,
  PROF_STAGE_UPSAMPLE_OUT = 10,
  PROF_STAGE_FRAME_STORE = 11,
  PROF_STAGE_ERROR = 12,
  PROF_STAGE_COUNT = 13,
};

enum npu_profile_conv_win_state_t : unsigned {
  PROF_CONV_WIN_IDLE_OR_DONE = 0,
  PROF_CONV_WIN_ACTIVE = 1,
  PROF_CONV_WIN_STATE_COUNT = 2,
};

enum npu_profile_conv_sa_state_t : unsigned {
  PROF_CONV_SA_IDLE_OR_DONE = 0,
  PROF_CONV_SA_COMPUTE_OR_WAIT_ACT = 1,
  PROF_CONV_SA_PSUM_EMIT_OR_WAIT = 2,
  PROF_CONV_SA_STATE_COUNT = 3,
};

enum npu_profile_conv_post_state_t : unsigned {
  PROF_CONV_POST_IDLE_OR_DONE = 0,
  PROF_CONV_POST_REQUANT_OR_WAIT_PSUM = 1,
  PROF_CONV_POST_ROWBUF_WRITE = 2,
  PROF_CONV_POST_STATE_COUNT = 3,
};

inline void npu_profile_set_stage(volatile u8_t& prof_stage_id,
                                  npu_profile_stage_t stage) {
#pragma HLS INLINE
  prof_stage_id = static_cast<u8_t>(static_cast<unsigned>(stage));
}

inline void npu_profile_set_active(volatile ap_uint<1>& prof_active,
                                   bool active) {
#pragma HLS INLINE
  prof_active = active ? ap_uint<1>(1) : ap_uint<1>(0);
}

inline void npu_profile_set_conv_win_state(
    volatile u8_t& state,
    npu_profile_conv_win_state_t value) {
#pragma HLS INLINE
  state = static_cast<u8_t>(static_cast<unsigned>(value));
}

inline void npu_profile_set_conv_sa_state(
    volatile u8_t& state,
    npu_profile_conv_sa_state_t value) {
#pragma HLS INLINE
  state = static_cast<u8_t>(static_cast<unsigned>(value));
}

inline void npu_profile_set_conv_post_state(
    volatile u8_t& state,
    npu_profile_conv_post_state_t value) {
#pragma HLS INLINE
  state = static_cast<u8_t>(static_cast<unsigned>(value));
}

inline void npu_profile_reset_conv_states(volatile u8_t& win_state,
                                          volatile u8_t& sa_state,
                                          volatile u8_t& post_state) {
#pragma HLS INLINE
  npu_profile_set_conv_win_state(win_state, PROF_CONV_WIN_IDLE_OR_DONE);
  npu_profile_set_conv_sa_state(sa_state, PROF_CONV_SA_IDLE_OR_DONE);
  npu_profile_set_conv_post_state(post_state, PROF_CONV_POST_IDLE_OR_DONE);
}

struct npu_issue_t {
  u8_t engine;
  u8_t kind;

  u8_t exec_desc_id;
  u8_t window_sched_id;
  u8_t qparam_id;
  u8_t row_consumer_id;
  u8_t fixed_desc_id;
  u8_t block5_sched_id;

  u8_t branch_idx;
  u8_t pattern;
  u8_t scratch_slot;

  u8_t src0_tensor;
  u8_t src1_tensor;
  u8_t dst_tensor;
  u8_t aux_tensor;

  u16_t row_begin;
  u16_t row_count;
  u16_t out_h;
  u16_t out_w;
};

error_code_t main_ctrl_run(axi_vec_t* gmem_frame_out,
                           volatile u8_t& prof_stage_id,
                           volatile u8_t& prof_pc,
                           volatile u8_t& prof_issue_kind,
                           volatile u8_t& prof_conv_win_state,
                           volatile u8_t& prof_conv_sa_state,
                           volatile u8_t& prof_conv_post_state);

error_code_t conv_engine_exec(const npu_issue_t& issue,
                              axi_vec_t* gmem_frame_out,
                              volatile u8_t& prof_stage_id,
                              volatile u8_t& prof_conv_win_state,
                              volatile u8_t& prof_conv_sa_state,
                              volatile u8_t& prof_conv_post_state);
error_code_t vec_alu_engine_exec(const npu_issue_t& issue,
                                 axi_vec_t* gmem_frame_out,
                                 volatile u8_t& prof_stage_id);
error_code_t pool_engine_exec(const npu_issue_t& issue,
                              volatile u8_t& prof_stage_id);

void main_ctrl_set_csim_last_uop(unsigned logical_uop);
void main_ctrl_set_csim_last_error(error_code_t err);
bool main_ctrl_csim_stop_before_logical_uop(unsigned logical_uop);
bool main_ctrl_csim_dump_tensor_set_pre(unsigned logical_uop);
bool main_ctrl_csim_dump_tensor_set_post(unsigned logical_uop);

} // namespace esp_int8

#endif
