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
bool param_dma_get_fixed_exec_desc(u8_t id, fixed_exec_desc_t& desc);
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

static u32_t runtime_expected_uop_count(std::uint32_t raw_uop_count) {
#pragma HLS INLINE
  return static_cast<u32_t>(raw_uop_count & RUNTIME_UOP_COUNT_MASK);
}

static bool csim_stop_before_logical_uop(unsigned uop_id);
static bool csim_dump_tensor_set_pre(unsigned logical_uop);
static bool csim_dump_tensor_set_post(unsigned logical_uop);

static u16_t ceil_div_u16(u16_t a, u16_t b) {
#pragma HLS INLINE
  return static_cast<u16_t>((a + b - 1) / b);
}

static u16_t conv_out_dim(u16_t in_size, u16_t stride) {
#pragma HLS INLINE
  return ceil_div_u16(in_size, stride);
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
  if (!avgpool_unit_checked(src, dst, qparam, 0)) {
    return ERR_BANK_OVERFLOW;
  }
  return ERR_NONE;
}

error_code_t pool_engine_exec(const npu_issue_t& issue,
                              volatile u8_t& prof_stage_id) {
#pragma HLS INLINE off
  npu_profile_set_stage(prof_stage_id, PROF_STAGE_AVGPOOL);
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
  (void)uop_id;
  return false;
#endif
}

static error_code_t core_mode_run(const axi_vec_t* gmem_frame_in,
                                  axi_vec_t* gmem_frame_out,
                                  u32_t expected_uop_count,
                                  volatile u8_t& prof_stage_id,
                                  volatile u8_t& prof_pc,
                                  volatile u8_t& prof_issue_kind,
                                  volatile u8_t& prof_conv_win_state,
                                  volatile u8_t& prof_conv_sa_state,
                                  volatile u8_t& prof_conv_post_state) {
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
  npu_profile_set_stage(prof_stage_id, PROF_STAGE_FRAME_LOAD);
  frame_dma_load(gmem_frame_in);

  npu_profile_set_stage(prof_stage_id, PROF_STAGE_MAIN_CTRL);
  const error_code_t err = main_ctrl_run(gmem_frame_out,
                                         prof_stage_id,
                                         prof_pc,
                                         prof_issue_kind,
                                         prof_conv_win_state,
                                         prof_conv_sa_state,
                                         prof_conv_post_state);
  if (err != ERR_NONE) {
    npu_profile_set_stage(prof_stage_id, PROF_STAGE_ERROR);
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

static_assert(esp_int8::INPUT_FRAME_AXI_WORDS == 12288,
              "Update gmem_frame_in m_axi depth when INPUT_FRAME_AXI_WORDS changes.");
static_assert(esp_int8::OUTPUT_FRAME_AXI_WORDS == 4096,
              "Update gmem_frame_out m_axi depth when OUTPUT_FRAME_AXI_WORDS changes.");

void espnet_encoder_int8_core(const esp_int8::axi_vec_t* gmem_frame_in,
                              esp_int8::axi_vec_t* gmem_frame_out,
                              const esp_int8::axi_vec_t* gmem_param,
                              std::uint32_t mode,
                              std::uint32_t uop_count,
                              volatile esp_int8::u8_t& prof_stage_id,
                              volatile ap_uint<1>& prof_active,
                              volatile esp_int8::u8_t& prof_pc,
                              volatile esp_int8::u8_t& prof_issue_kind,
                              volatile esp_int8::u8_t& prof_conv_win_state,
                              volatile esp_int8::u8_t& prof_conv_sa_state,
                              volatile esp_int8::u8_t& prof_conv_post_state) {
#ifdef ESP_INT8_COSIM_LITE
#pragma HLS INTERFACE ap_memory port=gmem_frame_in depth=12288
#pragma HLS INTERFACE ap_memory port=gmem_frame_out depth=4096
#pragma HLS INTERFACE ap_memory port=gmem_param depth=8192
#pragma HLS INTERFACE ap_none port=mode
#pragma HLS INTERFACE ap_none port=uop_count
#pragma HLS INTERFACE ap_vld port=prof_stage_id register
#pragma HLS INTERFACE ap_none port=prof_active register
#pragma HLS INTERFACE ap_none port=prof_pc register
#pragma HLS INTERFACE ap_none port=prof_issue_kind register
#pragma HLS INTERFACE ap_none port=prof_conv_win_state register
#pragma HLS INTERFACE ap_none port=prof_conv_sa_state register
#pragma HLS INTERFACE ap_none port=prof_conv_post_state register
#pragma HLS INTERFACE ap_ctrl_hs port=return
#else
#pragma HLS INTERFACE m_axi port=gmem_frame_in offset=slave bundle=gmem0 depth=12288 max_read_burst_length=64 num_read_outstanding=4
#pragma HLS INTERFACE m_axi port=gmem_frame_out offset=slave bundle=gmem1 depth=4096 max_write_burst_length=64 num_write_outstanding=4
#pragma HLS INTERFACE m_axi port=gmem_param offset=slave bundle=gmem2 depth=8192 max_read_burst_length=64 num_read_outstanding=4
#pragma HLS INTERFACE ap_vld port=prof_stage_id register
#pragma HLS INTERFACE ap_none port=prof_active register
#pragma HLS INTERFACE ap_none port=prof_pc register
#pragma HLS INTERFACE ap_none port=prof_issue_kind register
#pragma HLS INTERFACE ap_none port=prof_conv_win_state register
#pragma HLS INTERFACE ap_none port=prof_conv_sa_state register
#pragma HLS INTERFACE ap_none port=prof_conv_post_state register
#pragma HLS INTERFACE s_axilite port=gmem_frame_in bundle=control
#pragma HLS INTERFACE s_axilite port=gmem_frame_out bundle=control
#pragma HLS INTERFACE s_axilite port=gmem_param bundle=control
#pragma HLS INTERFACE s_axilite port=mode bundle=control
#pragma HLS INTERFACE s_axilite port=uop_count bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#endif

  const std::uint32_t mode_runtime = esp_int8::runtime_mode(mode);
  const esp_int8::u32_t expected_uop_count = esp_int8::runtime_expected_uop_count(uop_count);
  esp_int8::npu_profile_reset_conv_states(prof_conv_win_state,
                                          prof_conv_sa_state,
                                          prof_conv_post_state);

  switch (mode_runtime) {
    case esp_int8::MODE_INIT:
      esp_int8::npu_profile_set_active(prof_active, true);
      esp_int8::npu_profile_set_stage(prof_stage_id, esp_int8::PROF_STAGE_PARAM_INIT);
      prof_pc = 0;
      prof_issue_kind = 0;
      esp_int8::core_mode_init(gmem_param);
      esp_int8::npu_profile_set_stage(prof_stage_id, esp_int8::PROF_STAGE_IDLE);
      esp_int8::npu_profile_set_active(prof_active, false);
      break;
    case esp_int8::MODE_RUN:
      esp_int8::npu_profile_set_active(prof_active, true);
      prof_pc = 0;
      prof_issue_kind = 0;
      {
        const esp_int8::error_code_t err =
            esp_int8::core_mode_run(gmem_frame_in,
                                    gmem_frame_out,
                                    expected_uop_count,
                                    prof_stage_id,
                                    prof_pc,
                                    prof_issue_kind,
                                    prof_conv_win_state,
                                    prof_conv_sa_state,
                                    prof_conv_post_state);
        if (err == esp_int8::ERR_NONE) {
          esp_int8::npu_profile_set_stage(prof_stage_id, esp_int8::PROF_STAGE_IDLE);
        } else {
          esp_int8::npu_profile_set_stage(prof_stage_id, esp_int8::PROF_STAGE_ERROR);
        }
      }
      esp_int8::npu_profile_set_active(prof_active, false);
      break;
    case esp_int8::MODE_IDLE:
      esp_int8::npu_profile_set_stage(prof_stage_id, esp_int8::PROF_STAGE_IDLE);
      esp_int8::npu_profile_set_active(prof_active, false);
      break;
    default:
      esp_int8::npu_profile_set_stage(prof_stage_id, esp_int8::PROF_STAGE_ERROR);
      esp_int8::npu_profile_set_active(prof_active, false);
      break;
  }
}
