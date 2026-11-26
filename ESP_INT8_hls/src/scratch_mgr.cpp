#include "../include/npu_config.hpp"
#include "../include/npu_schedule.hpp"
#include "../include/npu_uop.hpp"

namespace esp_int8 {

bool param_dma_get_tensor_desc(u8_t tensor_id, tensor_desc_t& desc);
static tensor_desc_t s_scratch_desc[4];
static bool s_scratch_valid[4] = {false, false, false, false};
static tensor_desc_t s_global_alias_desc[MAX_TENSOR_DESC_COUNT];
static bool s_global_alias_valid[MAX_TENSOR_DESC_COUNT] = {};
static unsigned s_scratch_region = 0;
static u32_t s_scratch_base = 0;
static u32_t s_scratch_slot_bytes = 0;
static bool s_b2_saved[MAX_INPUT_FRAME_H / 4];

bool on_chip_memory_read_aligned_tensor_word(const tensor_desc_t& desc,
                                            u32_t offset, act_vec_t& value);
bool on_chip_memory_write_fmbuf_abs_word(u8_t bank, u32_t offset, act_vec_t value);

enum scratch_region_t : unsigned {
  SCRATCH_REGION_NONE = 0,
  SCRATCH_REGION_L20 = 1,
  SCRATCH_REGION_L2B0 = 2,
  SCRATCH_REGION_L30 = 3,
  SCRATCH_REGION_L3B0 = 4,
};

static bool global_tensor_desc(u8_t tensor_id, tensor_desc_t& desc) {
#pragma HLS INLINE
  if (!tensor_is_global(tensor_id)) {
    desc = tensor_desc_t();
    return false;
  }
  const unsigned tid = tensor_id.to_uint();
  if (tid < static_cast<unsigned>(MAX_TENSOR_DESC_COUNT) &&
      s_global_alias_valid[tid]) {
    desc = s_global_alias_desc[tid];
    return true;
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
  }
}

static void invalidate_global_aliases() {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=s_global_alias_valid complete
  for (int i = 0; i < MAX_TENSOR_DESC_COUNT; ++i) {
#pragma HLS UNROLL
    s_global_alias_valid[i] = false;
  }
}

void reset_scratch_state() {
#pragma HLS INLINE
  s_scratch_region = SCRATCH_REGION_NONE;
  s_scratch_base = 0;
  s_scratch_slot_bytes = 0;
  for (int row = 0; row < MAX_INPUT_FRAME_H / 4; ++row) {
#pragma HLS PIPELINE II=1
    s_b2_saved[row] = false;
  }
  invalidate_scratch();
  invalidate_global_aliases();
}

static void set_scratch_region(unsigned region) {
#pragma HLS INLINE
  runtime_profile_t profile;
  runtime_get_profile(profile);
  u32_t next_base = 0;
  u32_t next_slot = 0;

  switch (region) {
    case SCRATCH_REGION_L20:
      next_base = profile.l20_c1_base;
      next_slot = profile.l2_scratch_slot_bytes;
      break;
    case SCRATCH_REGION_L2B0:
      next_base = profile.l2b0_c1_base;
      next_slot = profile.l2_scratch_slot_bytes;
      break;
    case SCRATCH_REGION_L30:
      next_base = profile.l30_c1_base;
      next_slot = profile.l3_scratch_slot_bytes;
      break;
    case SCRATCH_REGION_L3B0:
      next_base = profile.l3b0_c1_base;
      next_slot = profile.l3_scratch_slot_bytes;
      break;
    default:
      next_base = 0;
      next_slot = 0;
      break;
  }

  if (region != s_scratch_region ||
      next_base != s_scratch_base ||
      next_slot != s_scratch_slot_bytes) {
    s_scratch_region = region;
    s_scratch_base = next_base;
    s_scratch_slot_bytes = next_slot;
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
  const u32_t row_bytes =
      (static_cast<u32_t>(w) * static_cast<u32_t>(c) + 31U) & static_cast<u32_t>(~31U);
  const u32_t bytes = static_cast<u32_t>(h) * row_bytes;
  if (slot_bytes.to_uint() == 0U || bytes > slot_bytes) {
    desc = tensor_desc_t();
    return false;
  }

  desc.bank_id = static_cast<u8_t>(static_cast<unsigned>(BANK_FMEM0));
  desc.elem_bytes = 1;
  desc.h = h;
  desc.w = w;
  desc.c = c;
  desc.reserved0 = c | static_cast<u16_t>(TENSOR_ROW_ALIGN32);
  desc.base_offset = base + static_cast<u32_t>(idx) * slot_bytes;
  desc.reserved1 = 0;
  return true;
}

static u8_t block5_source_channels(u16_t pattern, int src_idx) {
#pragma HLS INLINE
  const unsigned pat = pattern.to_uint();
  if (pat == static_cast<unsigned>(BLOCK5_PATTERN_L2_C16_4C12)) {
    return static_cast<u8_t>(src_idx == 0 ? 16 : 12);
  }
  if (pat == static_cast<unsigned>(BLOCK5_PATTERN_L3_C28_4C25)) {
    return static_cast<u8_t>(src_idx == 0 ? 28 : 25);
  }
  return static_cast<u8_t>(0);
}

static u32_t block5_scratch_base(u16_t pattern, u8_t first_branch_param) {
#pragma HLS INLINE
  runtime_profile_t profile;
  runtime_get_profile(profile);
  if (pattern.to_uint() == static_cast<unsigned>(BLOCK5_PATTERN_L2_C16_4C12)) {
    return static_cast<u32_t>(first_branch_param.to_uint() < 7U
        ? profile.l20_block5_base : profile.l2b0_block5_base);
  }
  return (first_branch_param.to_uint() < 19U)
             ? static_cast<u32_t>(profile.l30_block5_base)
             : static_cast<u32_t>(profile.l3b0_block5_base);
}

static tensor_desc_t make_block5_scratch_desc(u32_t base_offset,
                                              u16_t rows,
                                              u16_t out_w,
                                              u8_t channels) {
#pragma HLS INLINE
  tensor_desc_t desc;
  desc.bank_id = static_cast<u8_t>(static_cast<unsigned>(BANK_FMEM0));
  desc.elem_bytes = static_cast<u8_t>(1);
  desc.reserved0 = static_cast<u16_t>(channels.to_uint() | TENSOR_ROW_ALIGN32);
  desc.base_offset = base_offset;
  desc.h = rows;
  desc.w = out_w;
  desc.c = static_cast<u16_t>(channels.to_uint());
  desc.reserved1 = static_cast<u16_t>(0);
  return desc;
}

bool resolve_block5_scratch_descs(u16_t pattern,
                                  u8_t first_branch_param,
                                  u16_t rows,
                                  u16_t out_w,
                                  tensor_desc_t& s0,
                                  tensor_desc_t& s1,
                                  tensor_desc_t& s2,
                                  tensor_desc_t& s3,
                                  tensor_desc_t& s4) {
#pragma HLS INLINE off
  const u8_t c0 = block5_source_channels(pattern, 0);
  const u8_t c1 = block5_source_channels(pattern, 1);
  const u8_t c2 = block5_source_channels(pattern, 2);
  const u8_t c3 = block5_source_channels(pattern, 3);
  const u8_t c4 = block5_source_channels(pattern, 4);
  if (rows == 0 || rows > BLOCK5_ROW_BLOCK_ROWS ||
      c0.to_uint() == 0U || c1.to_uint() == 0U || c2.to_uint() == 0U ||
      c3.to_uint() == 0U || c4.to_uint() == 0U) {
    return false;
  }
  u32_t cursor = block5_scratch_base(pattern, first_branch_param);
  s0 = make_block5_scratch_desc(cursor, rows, out_w, c0);
  cursor += static_cast<u32_t>(rows) * tensor_row_stride_bytes(s0);
  s1 = make_block5_scratch_desc(cursor, rows, out_w, c1);
  cursor += static_cast<u32_t>(rows) * tensor_row_stride_bytes(s1);
  s2 = make_block5_scratch_desc(cursor, rows, out_w, c2);
  cursor += static_cast<u32_t>(rows) * tensor_row_stride_bytes(s2);
  s3 = make_block5_scratch_desc(cursor, rows, out_w, c3);
  cursor += static_cast<u32_t>(rows) * tensor_row_stride_bytes(s3);
  s4 = make_block5_scratch_desc(cursor, rows, out_w, c4);
  return true;
}

static bool b2_residual_plan_ok(const tensor_desc_t& src, const runtime_profile_t& profile) {
#pragma HLS INLINE
  return src.bank_id.to_uint() == static_cast<unsigned>(BANK_FMEM0) &&
         src.base_offset.to_uint() == profile.l20_base && src.reserved1 == 0 &&
         src.h.to_uint() == profile.input_h / 4U && src.w.to_uint() == profile.input_w / 4U &&
         src.c == 64 && tensor_phys_c(src) == 64 && src.elem_bytes == 1 &&
         profile.b2_backup_rows != 0U && profile.b2_backup_rows <= 64U &&
         (profile.b2_backup_rows & (profile.b2_backup_rows - 1U)) == 0U;
}

bool b2_src1_row_base(const tensor_desc_t& src, u16_t row, u32_t& base) {
#pragma HLS INLINE
  runtime_profile_t profile;
  runtime_get_profile(profile);
  const unsigned r = row.to_uint();
  if (!b2_residual_plan_ok(src, profile) || r >= src.h.to_uint()) { return false; }
  const u32_t stride = tensor_row_stride_bytes(src);
  if (s_b2_saved[r]) {
    const unsigned slot = r & (profile.b2_backup_rows - 1U);
    base = static_cast<u32_t>(profile.b2_backup_base) + static_cast<u32_t>(slot) * stride;
    s_b2_saved[r] = false;
  } else {
    base = src.base_offset + static_cast<u32_t>(r) * stride;
  }
  return true;
}

static bool copy_b2_residual_row(const tensor_desc_t& src,
                                  const runtime_profile_t& profile, unsigned row) {
#pragma HLS INLINE off
  const unsigned stride = tensor_row_stride_bytes(src).to_uint();
  const unsigned words = stride / AXI_WORD_BYTES;
  const unsigned source = src.base_offset.to_uint() + row * stride;
  const unsigned target = profile.b2_backup_base +
      (row & (profile.b2_backup_rows - 1U)) * stride;
  for (int wi = 0; wi < MAX_INPUT_FRAME_W / 2; ++wi) {
#pragma HLS PIPELINE off
    if (static_cast<unsigned>(wi) >= words) { break; }
    const u32_t offset = static_cast<u32_t>(wi * AXI_WORD_BYTES);
    act_vec_t word = 0;
    if (!on_chip_memory_read_aligned_tensor_word(src, static_cast<u32_t>(source) + offset, word) ||
        !on_chip_memory_write_fmbuf_abs_word(static_cast<u8_t>(static_cast<unsigned>(BANK_FMEM0)),
                                              static_cast<u32_t>(target) + offset, word)) {
      return false;
    }
  }
  return true;
}

bool backup_b2_src1_rows_before_write(const tensor_desc_t& src,
                                      const tensor_desc_t& dst, u16_t write_row) {
#pragma HLS INLINE off
  runtime_profile_t profile;
  runtime_get_profile(profile);
  if (!b2_residual_plan_ok(src, profile) || write_row >= src.h ||
      dst.bank_id.to_uint() != static_cast<unsigned>(BANK_FMEM0) || dst.base_offset != 0 ||
      dst.reserved1 != 0 || dst.h != src.h || dst.w != src.w ||
      dst.c.to_uint() != 128U + profile.input_c || tensor_phys_c(dst) != dst.c) {
    return false;
  }
  const unsigned stride = tensor_row_stride_bytes(src).to_uint();
  const unsigned dst_stride = tensor_row_stride_bytes(dst).to_uint();
  const unsigned begin = write_row.to_uint() * dst_stride;
  const unsigned end = begin + dst_stride;
  for (int row = 0; row < MAX_INPUT_FRAME_H / 4; ++row) {
#pragma HLS PIPELINE off
    if (static_cast<unsigned>(row) >= write_row.to_uint()) { break; }
    const unsigned lo = profile.l20_base + static_cast<unsigned>(row) * stride;
    if (begin < lo + stride && lo < end && !s_b2_saved[row]) {
      if (!copy_b2_residual_row(src, profile, static_cast<unsigned>(row))) { return false; }
      s_b2_saved[row] = true;
    }
  }
  return true;
}

static unsigned conv_scratch_region_fields(u8_t param_id, u8_t src0_tensor, u16_t out_h) {
#pragma HLS INLINE
  const unsigned pid = param_id.to_uint();
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
    return (src0_tensor.to_uint() == static_cast<unsigned>(TID_B1_ACT))
               ? SCRATCH_REGION_L20
               : SCRATCH_REGION_L2B0;
  }
  return (src0_tensor.to_uint() == static_cast<unsigned>(TID_B2_ACT))
             ? SCRATCH_REGION_L30
             : SCRATCH_REGION_L3B0;
}

static unsigned add_scratch_region_fields(u8_t param_id) {
#pragma HLS INLINE
  const unsigned pid = param_id.to_uint();
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

static unsigned store_scratch_region_fields(u8_t dst_tensor, u16_t in_h) {
#pragma HLS INLINE
  switch (dst_tensor.to_uint()) {
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
  if (in_h.to_uint() >= 128U) {
    return (s_scratch_region == SCRATCH_REGION_L20) ? SCRATCH_REGION_L20 : SCRATCH_REGION_L2B0;
  }
  return (s_scratch_region == SCRATCH_REGION_L30) ? SCRATCH_REGION_L30 : SCRATCH_REGION_L3B0;
}

static unsigned scratch_region_for_fields(u8_t opcode,
                                          u8_t param_id,
                                          u8_t src0_tensor,
                                          u8_t dst_tensor,
                                          u16_t in_h,
                                          u16_t out_h) {
#pragma HLS INLINE
  const unsigned op = opcode.to_uint();
  if (op == static_cast<unsigned>(UOP_CONV)) {
    return conv_scratch_region_fields(param_id, src0_tensor, out_h);
  }
  if (op == static_cast<unsigned>(UOP_ADD)) {
    return add_scratch_region_fields(param_id);
  }
  if (op == static_cast<unsigned>(UOP_STORE)) {
    return store_scratch_region_fields(dst_tensor, in_h);
  }
  if (out_h.to_uint() >= 128U) {
    return s_scratch_region == SCRATCH_REGION_L2B0 ? SCRATCH_REGION_L2B0 : SCRATCH_REGION_L20;
  }
  return s_scratch_region == SCRATCH_REGION_L3B0 ? SCRATCH_REGION_L3B0 : SCRATCH_REGION_L30;
}

static bool fields_use_scratch(u8_t src0_tensor, u8_t src1_tensor, u8_t dst_tensor) {
#pragma HLS INLINE
  return tensor_is_scratch(src0_tensor) ||
         tensor_is_scratch(src1_tensor) ||
         tensor_is_scratch(dst_tensor);
}

void select_scratch_region_for_fields(u8_t opcode,
                                      u8_t param_id,
                                      u8_t src0_tensor,
                                      u8_t src1_tensor,
                                      u8_t dst_tensor,
                                      u16_t in_h,
                                      u16_t out_h) {
#pragma HLS INLINE
  if (fields_use_scratch(src0_tensor, src1_tensor, dst_tensor)) {
    set_scratch_region(scratch_region_for_fields(opcode,
                                                 param_id,
                                                 src0_tensor,
                                                 dst_tensor,
                                                 in_h,
                                                 out_h));
  }
}

static bool make_scratch_desc(u8_t tensor_id, u16_t h, u16_t w, u16_t c, tensor_desc_t& desc) {
#pragma HLS INLINE
  unsigned idx = 0;
  if (!scratch_index(tensor_id, idx) || idx != 0U || s_scratch_region == SCRATCH_REGION_NONE) {
    desc = tensor_desc_t();
    return false;
  }

  // Only C1 remains a full-height local tensor. Branch intermediates belong to
  // the row-group composer and must not recreate legacy full-layer LS_A/B/TMP.
  if (!make_contiguous_scratch_desc(0, s_scratch_base, s_scratch_slot_bytes,
                                     h, w, c, desc)) {
    return false;
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

bool resolve_tensor_read(u8_t tensor_id, tensor_desc_t& desc) {
#pragma HLS INLINE
  if (tensor_is_global(tensor_id)) {
    return global_tensor_desc(tensor_id, desc);
  }
  return scratch_tensor_desc(tensor_id, desc);
}

bool resolve_tensor_write(u8_t tensor_id, u16_t h, u16_t w, u16_t c, tensor_desc_t& desc) {
#pragma HLS INLINE
  if (tensor_is_global(tensor_id)) {
    const unsigned tid = tensor_id.to_uint();
    if (tid < static_cast<unsigned>(MAX_TENSOR_DESC_COUNT)) {
      s_global_alias_valid[tid] = false;
    }
    return global_tensor_desc(tensor_id, desc);
  }
  return make_scratch_desc(tensor_id, h, w, c, desc);
}

static bool make_slice_alias_desc(const tensor_desc_t& base_desc,
                                  u16_t c_offset,
                                  u16_t c,
                                  tensor_desc_t& alias) {
#pragma HLS INLINE
  const u16_t phys_c = tensor_phys_c(base_desc);
  const u16_t abs_c_offset = static_cast<u16_t>(base_desc.reserved1 + c_offset);
  if (static_cast<unsigned>(abs_c_offset.to_uint() + c.to_uint()) > phys_c.to_uint()) {
    alias = tensor_desc_t();
    return false;
  }

  alias = base_desc;
  alias.c = c;
  alias.reserved0 = phys_c | (base_desc.reserved0 & static_cast<u16_t>(TENSOR_ROW_ALIGN32));
  alias.reserved1 = abs_c_offset;
  return true;
}

bool alias_global_tensor_to_slice(u8_t tensor_id,
                                  const tensor_desc_t& base_desc,
                                  u16_t c_offset,
                                  u16_t c) {
#pragma HLS INLINE
  if (!tensor_is_global(tensor_id)) {
    return false;
  }
  const unsigned tid = tensor_id.to_uint();
  if (tid >= static_cast<unsigned>(MAX_TENSOR_DESC_COUNT)) {
    return false;
  }
  tensor_desc_t alias;
  if (!make_slice_alias_desc(base_desc, c_offset, c, alias)) {
    return false;
  }
  s_global_alias_desc[tid] = alias;
  s_global_alias_valid[tid] = true;
  return true;
}

bool alias_scratch_tensor_to_slice(u8_t tensor_id,
                                   const tensor_desc_t& base_desc,
                                   u16_t c_offset,
                                   u16_t c) {
#pragma HLS INLINE
  unsigned idx = 0;
  if (!scratch_index(tensor_id, idx)) {
    return false;
  }

  tensor_desc_t alias;
  if (!make_slice_alias_desc(base_desc, c_offset, c, alias)) {
    return false;
  }
  s_scratch_desc[idx] = alias;
  s_scratch_valid[idx] = true;
  return true;
}

}  // namespace esp_int8
