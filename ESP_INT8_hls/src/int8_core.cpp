#include <cstdint>

#include "../include/npu_config.hpp"
#include "../include/npu_ctrl.hpp"
#include "../include/npu_q.hpp"
#include "../include/npu_schedule.hpp"
#include "../include/npu_uop.hpp"


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


}  // namespace esp_int8

static_assert(esp_int8::INPUT_FRAME_AXI_WORDS == 1800,
              "Update gmem_frame_in m_axi depth when INPUT_FRAME_AXI_WORDS changes.");
static_assert(esp_int8::OUTPUT_FRAME_AXI_WORDS == 1800,
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
#pragma HLS INTERFACE ap_memory port=gmem_frame_in depth=1800
#pragma HLS INTERFACE ap_memory port=gmem_frame_out depth=1800
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
#pragma HLS INTERFACE m_axi port=gmem_frame_in offset=slave bundle=gmem0 depth=1800 max_read_burst_length=64 num_read_outstanding=4
#pragma HLS INTERFACE m_axi port=gmem_frame_out offset=slave bundle=gmem1 depth=1800 max_write_burst_length=64 num_write_outstanding=4
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
