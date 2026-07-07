#ifndef ESP_INT8_NPU_CTRL_HPP
#define ESP_INT8_NPU_CTRL_HPP

#include "npu_config.hpp"
#include "npu_schedule.hpp"
#include "npu_types.hpp"

namespace esp_int8 {

struct profile_ctrl_t {
  bool enable;
  bool stop_before;
  u16_t stop_pc_plus1;
};

enum npu_engine_t : unsigned {
  NPU_ENGINE_NONE = 0,
  NPU_ENGINE_CONV = 1,
  NPU_ENGINE_VEC = 2,
  NPU_ENGINE_POOL = 3,
  NPU_ENGINE_UPSAMPLE = 5,
};

enum npu_issue_kind_t : unsigned {
  ISSUE_NONE = 0,

  ISSUE_CONV_NORMAL = 1,
  ISSUE_CONV_BLOCK5_BRANCH = 2,

  ISSUE_VEC_AFFINE = 11,

  ISSUE_POOL_AVG = 20,

  ISSUE_UPSAMPLE_ROW = 40,
};

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

struct main_ctrl_ctx_t {
  u8_t pc;
  u8_t exec_count;

  u8_t state;
  u8_t substate;

  u8_t block5_sched_id;
  u8_t block5_branch_idx;
  u16_t block5_row;
  u16_t block5_row_group_h;

  u8_t fixed_seq_idx;
  u16_t current_row;

  bool done;
  bool error;
  u8_t error_code;
};

error_code_t main_ctrl_run(axi_vec_t* gmem_frame_out,
                           const profile_ctrl_t& profile_ctrl);

error_code_t conv_engine_exec(const npu_issue_t& issue,
                              axi_vec_t* gmem_frame_out);
error_code_t vec_alu_engine_exec(const npu_issue_t& issue,
                                 axi_vec_t* gmem_frame_out);
error_code_t pool_engine_exec(const npu_issue_t& issue);
error_code_t upsample_engine_exec(const npu_issue_t& issue,
                                  axi_vec_t* gmem_frame_out);

void vec_alu_apply_affine_block(const act_vec_t& in_word,
                                u8_t valid_c,
                                aff_q_t aff_qparam,
                                u8_t act_type,
                                act_vec_t& out_word);

bool main_ctrl_profile_stop_matches(const profile_ctrl_t& ctrl, int pc);
void main_ctrl_profile_record_exec_entry(unsigned kind);
void main_ctrl_record_exec_fetch();

void main_ctrl_set_csim_last_uop(unsigned logical_uop);
void main_ctrl_set_csim_last_error(error_code_t err);
bool main_ctrl_csim_stop_before_logical_uop(unsigned logical_uop);
bool main_ctrl_csim_dump_tensor_set_pre(unsigned logical_uop);
bool main_ctrl_csim_dump_tensor_set_post(unsigned logical_uop);

} // namespace esp_int8

#endif
