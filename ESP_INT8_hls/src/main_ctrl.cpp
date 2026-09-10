#include "../include/npu_ctrl.hpp"
#include "../include/npu_uop.hpp"

namespace esp_int8 {

bool param_dma_get_exec_entry(u8_t pc, exec_plan_entry_t& entry);
bool param_dma_get_fixed_exec_desc(u8_t id, fixed_exec_desc_t& desc);
bool param_dma_get_block5_sched(u8_t id, block5_sched_desc_t& desc);
bool param_dma_get_conv_exec_desc(u8_t id, conv_exec_desc_t& desc);

static u8_t block5_pattern_from_fixed(const fixed_exec_desc_t& desc) {
#pragma HLS INLINE
  return static_cast<u8_t>(desc.reserved0.to_uint() & 0xffU);
}

static u8_t block5_sched_id_from_fixed(const fixed_exec_desc_t& desc) {
#pragma HLS INLINE
  return static_cast<u8_t>((desc.reserved0 >> 8).to_uint() & 0xffU);
}

static bool fixed_is_block5(const fixed_exec_desc_t& desc) {
#pragma HLS INLINE
  return (desc.flags.to_uint() & static_cast<unsigned>(FIXED_FLAG_BLOCK5_ROW_GROUP)) != 0U;
}

static bool block5_schedule_valid(const fixed_exec_desc_t& desc,
                                  const block5_sched_desc_t& sched) {
#pragma HLS INLINE
  const unsigned pattern = block5_pattern_from_fixed(desc).to_uint();
  const unsigned finalizer = sched.finalizer_kind.to_uint();
  if (sched.pattern.to_uint() != pattern ||
      sched.branch_count.to_uint() != 5U ||
      sched.row_group_h.to_uint() != static_cast<unsigned>(BLOCK5_ROW_BLOCK_ROWS) ||
      sched.valid_c.to_uint() != desc.valid_c.to_uint()) {
    return false;
  }
  if (pattern == static_cast<unsigned>(BLOCK5_PATTERN_L2_C16_4C12)) {
    return finalizer == static_cast<unsigned>(BLOCK5_FINALIZER_L2);
  }
  if (pattern == static_cast<unsigned>(BLOCK5_PATTERN_L3_C28_4C25)) {
    return finalizer == static_cast<unsigned>(BLOCK5_FINALIZER_L3);
  }
  return false;
}

static error_code_t execute_issue(const npu_issue_t& issue,
                                  axi_vec_t* gmem_frame_out,
                                  volatile u8_t& prof_stage_id,
                                  volatile u8_t& prof_conv_win_state,
                                  volatile u8_t& prof_conv_sa_state,
                                  volatile u8_t& prof_conv_post_state) {
#pragma HLS INLINE off
  npu_profile_reset_conv_states(prof_conv_win_state,
                                prof_conv_sa_state,
                                prof_conv_post_state);
  switch (issue.engine.to_uint()) {
    case static_cast<unsigned>(NPU_ENGINE_CONV):
      return conv_engine_exec(issue,
                              gmem_frame_out,
                              prof_stage_id,
                              prof_conv_win_state,
                              prof_conv_sa_state,
                              prof_conv_post_state);
    case static_cast<unsigned>(NPU_ENGINE_VEC):
      npu_profile_set_stage(prof_stage_id, PROF_STAGE_VEC_FIXED);
      return vec_alu_engine_exec(issue, gmem_frame_out, prof_stage_id);
    case static_cast<unsigned>(NPU_ENGINE_POOL):
      npu_profile_set_stage(prof_stage_id, PROF_STAGE_AVGPOOL);
      return pool_engine_exec(issue, prof_stage_id);
    case static_cast<unsigned>(NPU_ENGINE_NONE):
      return ERR_NONE;
    default:
      return ERR_UNSUPPORTED_OPCODE;
  }
}

struct ctrl_runtime_t {
  u8_t pc;
  u8_t exec_count;
  bool entry_active;
  bool stop_requested;
  exec_plan_entry_t entry;
  fixed_exec_desc_t fixed_desc;
  unsigned logical_uop;

  bool block5_active;
  u8_t block5_sched_id;
  u8_t block5_branch_idx;
  u8_t block5_scratch_first_param;
  u16_t block5_row;
};

static void init_ctrl_runtime(ctrl_runtime_t& ctx) {
#pragma HLS INLINE
  ctx = ctrl_runtime_t();
  ctx.pc = 0;
  ctx.exec_count = static_cast<u8_t>(MAX_EXEC_PLAN_COUNT);
  ctx.entry_active = false;
  ctx.stop_requested = false;
  ctx.logical_uop = 0U;
  ctx.block5_active = false;
  ctx.block5_sched_id = 0;
  ctx.block5_branch_idx = 0;
  ctx.block5_scratch_first_param = 0;
  ctx.block5_row = 0;
}

static void build_normal_conv_issue(const exec_plan_entry_t& entry,
                                    npu_issue_t& issue,
                                    bool& issue_valid) {
#pragma HLS INLINE off
  issue = npu_issue_t();
  issue.engine = static_cast<u8_t>(static_cast<unsigned>(NPU_ENGINE_CONV));
  issue.kind = static_cast<u8_t>(static_cast<unsigned>(ISSUE_CONV_NORMAL));
  issue.exec_desc_id = entry.desc_id;
  issue_valid = true;
}

static error_code_t build_fixed_issue(const exec_plan_entry_t& entry,
                                      const fixed_exec_desc_t& desc,
                                      npu_issue_t& issue,
                                      bool& issue_valid) {
#pragma HLS INLINE off
  issue = npu_issue_t();
  issue_valid = false;

  if (desc.kind.to_uint() != entry.kind.to_uint()) {
    return ERR_UOP_DECODE;
  }

  if (entry.kind.to_uint() == static_cast<unsigned>(EXEC_POOL)) {
    issue.engine = static_cast<u8_t>(static_cast<unsigned>(NPU_ENGINE_POOL));
    issue.kind = static_cast<u8_t>(static_cast<unsigned>(ISSUE_POOL_AVG));
    issue.fixed_desc_id = entry.desc_id;
    issue_valid = true;
    return ERR_NONE;
  }

  if (entry.kind.to_uint() == static_cast<unsigned>(EXEC_BLOCK_AFFINE)) {
    if (fixed_is_block5(desc)) {
      return ERR_UNSUPPORTED_OPCODE;
    }

    issue.engine = static_cast<u8_t>(static_cast<unsigned>(NPU_ENGINE_VEC));
    issue.kind = static_cast<u8_t>(static_cast<unsigned>(ISSUE_VEC_AFFINE));
    issue.fixed_desc_id = entry.desc_id;
    issue_valid = true;
    return ERR_NONE;
  }

  if (entry.kind.to_uint() == static_cast<unsigned>(EXEC_BLOCK_ADD_AFFINE)) {
    return ERR_UNSUPPORTED_OPCODE;
  }

  if (entry.kind.to_uint() == static_cast<unsigned>(EXEC_NOP)) {
    return ERR_NONE;
  }

  return ERR_UNSUPPORTED_OPCODE;
}

static error_code_t init_block5_state(ctrl_runtime_t& ctx,
                                      const exec_plan_entry_t& entry,
                                      const fixed_exec_desc_t& desc) {
#pragma HLS INLINE off
  if (desc.kind.to_uint() != entry.kind.to_uint()) {
    return ERR_UOP_DECODE;
  }

  block5_sched_desc_t sched;
  const u8_t sched_id = block5_sched_id_from_fixed(desc);
  if (!param_dma_get_block5_sched(sched_id, sched) ||
      !block5_schedule_valid(desc, sched)) {
    return ERR_PARAM_DESC_RANGE;
  }

  conv_exec_desc_t first_branch;
  if (!param_dma_get_conv_exec_desc(sched.first_branch_conv_id, first_branch)) {
    return ERR_PARAM_DESC_RANGE;
  }

  ctx.block5_active = true;
  ctx.block5_sched_id = sched_id;
  ctx.block5_branch_idx = 0;
  ctx.block5_scratch_first_param = first_branch.param_id;
  ctx.block5_row = 0;
  return ERR_NONE;
}

static error_code_t build_block5_issue_step(ctrl_runtime_t& ctx,
                                            npu_issue_t& issue,
                                            bool& issue_valid,
                                            bool& entry_complete) {
#pragma HLS INLINE off
  issue = npu_issue_t();
  issue_valid = false;
  entry_complete = false;

  block5_sched_desc_t sched;
  const u8_t sched_id = ctx.block5_sched_id;
  if (!param_dma_get_block5_sched(sched_id, sched) ||
      !block5_schedule_valid(ctx.fixed_desc, sched)) {
    return ERR_PARAM_DESC_RANGE;
  }

  const unsigned out_h_u = sched.out_h.to_uint();
  const unsigned row_u = ctx.block5_row.to_uint();
  if (row_u >= out_h_u) {
    ctx.block5_active = false;
    entry_complete = true;
    return ERR_NONE;
  }

  const unsigned remain = out_h_u - row_u;
  const unsigned local_rows_u =
      (remain < static_cast<unsigned>(BLOCK5_ROW_BLOCK_ROWS))
          ? remain
          : static_cast<unsigned>(BLOCK5_ROW_BLOCK_ROWS);
  const u16_t local_rows = static_cast<u16_t>(local_rows_u);

  const unsigned branch_u = ctx.block5_branch_idx.to_uint();
  if (branch_u >= sched.branch_count.to_uint()) {
    return ERR_UOP_DECODE;
  }

  issue.engine = static_cast<u8_t>(static_cast<unsigned>(NPU_ENGINE_CONV));
  issue.kind = static_cast<u8_t>(static_cast<unsigned>(ISSUE_CONV_BLOCK5_BRANCH));
  issue.exec_desc_id = static_cast<u8_t>(sched.first_branch_conv_id.to_uint() + branch_u);
  issue.qparam_id = sched.chain_add_qparam_id0;
  issue.fixed_desc_id = ctx.entry.desc_id;
  issue.block5_sched_id = sched_id;
  issue.branch_idx = static_cast<u8_t>(branch_u);
  issue.pattern = sched.pattern;
  issue.scratch_slot = ctx.block5_scratch_first_param;
  issue.src0_tensor = sched.src_tensor;
  issue.dst_tensor = sched.dst_tensor;
  issue.aux_tensor = sched.add_tensor;
  issue.row_begin = ctx.block5_row;
  issue.row_count = local_rows;
  issue.out_h = sched.out_h;
  issue.out_w = sched.out_w;
  issue_valid = true;

  const unsigned next_branch = branch_u + 1U;
  if (next_branch >= sched.branch_count.to_uint()) {
    const unsigned next_row = row_u + local_rows_u;
    ctx.block5_row = static_cast<u16_t>(next_row);
    ctx.block5_branch_idx = 0;
    if (next_row >= out_h_u) {
      ctx.block5_active = false;
      entry_complete = true;
    }
  } else {
    ctx.block5_branch_idx = static_cast<u8_t>(next_branch);
  }
  return ERR_NONE;
}

static error_code_t prepare_new_entry(ctrl_runtime_t& ctx,
                                      npu_issue_t& issue,
                                      bool& issue_valid,
                                      bool& entry_complete) {
#pragma HLS INLINE off
  issue = npu_issue_t();
  issue_valid = false;
  entry_complete = false;

  exec_plan_entry_t entry;
  if (!param_dma_get_exec_entry(ctx.pc, entry)) {
    main_ctrl_set_csim_last_error(ERR_UOP_DECODE);
    return ERR_UOP_DECODE;
  }

  const unsigned logical_uop = entry.logical_uop_id.to_uint();
  main_ctrl_set_csim_last_uop(logical_uop);
  if (main_ctrl_csim_stop_before_logical_uop(logical_uop)) {
    entry_complete = false;
    ctx.entry_active = false;
    ctx.stop_requested = true;
    return ERR_NONE;
  }

  const unsigned kind = entry.kind.to_uint();
  if (kind == static_cast<unsigned>(EXEC_END)) {
    main_ctrl_set_csim_last_error(ERR_NONE);
    entry_complete = false;
    ctx.entry_active = false;
    ctx.stop_requested = true;
    return ERR_NONE;
  }

  if (!main_ctrl_csim_dump_tensor_set_pre(logical_uop)) {
    main_ctrl_set_csim_last_error(ERR_BANK_OVERFLOW);
    return ERR_BANK_OVERFLOW;
  }

  ctx.entry = entry;
  ctx.logical_uop = logical_uop;
  ctx.entry_active = true;
  ctx.block5_active = false;

  if (kind == static_cast<unsigned>(EXEC_CONV)) {
    build_normal_conv_issue(entry, issue, issue_valid);
    entry_complete = true;
    return ERR_NONE;
  }

  fixed_exec_desc_t desc;
  if (!param_dma_get_fixed_exec_desc(entry.desc_id, desc) ||
      desc.kind.to_uint() != entry.kind.to_uint()) {
    main_ctrl_set_csim_last_error(ERR_UOP_DECODE);
    return ERR_UOP_DECODE;
  }
  ctx.fixed_desc = desc;

  if ((kind == static_cast<unsigned>(EXEC_BLOCK_AFFINE) ||
       kind == static_cast<unsigned>(EXEC_BLOCK_ADD_AFFINE)) &&
      fixed_is_block5(desc)) {
    const error_code_t init_err = init_block5_state(ctx, entry, desc);
    if (init_err != ERR_NONE) {
      return init_err;
    }
    return build_block5_issue_step(ctx, issue, issue_valid, entry_complete);
  }

  const error_code_t fixed_err = build_fixed_issue(entry, desc, issue, issue_valid);
  if (fixed_err != ERR_NONE) {
    return fixed_err;
  }
  entry_complete = true;
  return ERR_NONE;
}

static error_code_t complete_entry(ctrl_runtime_t& ctx) {
#pragma HLS INLINE off
  if (!main_ctrl_csim_dump_tensor_set_post(ctx.logical_uop)) {
    main_ctrl_set_csim_last_error(ERR_BANK_OVERFLOW);
    return ERR_BANK_OVERFLOW;
  }

  ctx.entry_active = false;
  ctx.block5_active = false;
  ctx.pc = static_cast<u8_t>(ctx.pc.to_uint() + 1U);
  return ERR_NONE;
}

error_code_t main_ctrl_run(axi_vec_t* gmem_frame_out,
                           volatile u8_t& prof_stage_id,
                           volatile u8_t& prof_pc,
                           volatile u8_t& prof_issue_kind,
                           volatile u8_t& prof_conv_win_state,
                           volatile u8_t& prof_conv_sa_state,
                           volatile u8_t& prof_conv_post_state) {
#pragma HLS INLINE off
  ctrl_runtime_t ctx;
  init_ctrl_runtime(ctx);

  enum { MAIN_CTRL_STEP_GUARD = MAX_EXEC_PLAN_COUNT * 64 };

  for (int step = 0; step < MAIN_CTRL_STEP_GUARD; ++step) {
#pragma HLS PIPELINE off
#pragma HLS UNROLL off
    npu_issue_t issue = npu_issue_t();
    bool issue_valid = false;
    bool entry_complete = false;
    error_code_t err = ERR_NONE;

    if (ctx.pc.to_uint() >= static_cast<unsigned>(MAX_EXEC_PLAN_COUNT)) {
      main_ctrl_set_csim_last_error(ERR_UOP_DECODE);
      return ERR_UOP_DECODE;
    }

    if (!ctx.entry_active) {
      err = prepare_new_entry(ctx, issue, issue_valid, entry_complete);
    } else if (ctx.block5_active) {
      err = build_block5_issue_step(ctx, issue, issue_valid, entry_complete);
    } else {
      entry_complete = true;
    }

    if (err != ERR_NONE) {
      main_ctrl_set_csim_last_error(err);
      return err;
    }

    if (ctx.stop_requested) {
      return ERR_NONE;
    }

    if (!ctx.entry_active && !issue_valid && !entry_complete) {
      return ERR_NONE;
    }

    // Single-tail dispatch: this must be the only non-definition call site of execute_issue().
    if (issue_valid) {
      prof_pc = ctx.pc;
      prof_issue_kind = issue.kind;
      npu_profile_set_stage(prof_stage_id, PROF_STAGE_MAIN_CTRL);
      err = execute_issue(issue,
                          gmem_frame_out,
                          prof_stage_id,
                          prof_conv_win_state,
                          prof_conv_sa_state,
                          prof_conv_post_state);
      if (err != ERR_NONE) {
        main_ctrl_set_csim_last_error(err);
        npu_profile_set_stage(prof_stage_id, PROF_STAGE_ERROR);
        return err;
      }
    }

    if (entry_complete) {
      err = complete_entry(ctx);
      if (err != ERR_NONE) {
        return err;
      }
      if (ctx.stop_requested) {
        return ERR_NONE;
      }
      if (!ctx.entry_active && ctx.pc.to_uint() >= static_cast<unsigned>(MAX_EXEC_PLAN_COUNT)) {
        main_ctrl_set_csim_last_error(ERR_UOP_DECODE);
        return ERR_UOP_DECODE;
      }
    }
  }

  main_ctrl_set_csim_last_error(ERR_UOP_DECODE);
  return ERR_UOP_DECODE;
}

} // namespace esp_int8
