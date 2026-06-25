#pragma once

#include "npu_config.hpp"
#include "npu_types.hpp"

namespace esp_int8 {

enum window_mode_t : std::uint8_t {
  WIN_MODE_INVALID = 0,
  WIN_MODE_SMALLC_3X3_STAGED = 1,
  WIN_MODE_LARGEC_3X3_SEGMENT = 2,
  WIN_MODE_1X1_ALIGNED = 3,
  WIN_MODE_1X1_PACKED = 4,
  WIN_MODE_FIRST_C3 = 5,
};

enum pack_cmd_flags_t : std::uint8_t {
  PACK_CMD_VALID = 1 << 0,
  PACK_CMD_ZERO = 1 << 1,
  PACK_CMD_CONTIG_READ = 1 << 2,
  PACK_CMD_ALIGNED_READ = 1 << 3,
};

enum row_consumer_mode_t : std::uint8_t {
  ROW_CONSUMER_NONE = 0,
  ROW_CONSUMER_STORE = 1,
  ROW_CONSUMER_ADD_STORE = 2,
  ROW_CONSUMER_ADD_AFFINE_STORE = 3,
  ROW_CONSUMER_UPSAMPLE_OUT = 4,
};

enum store_layout_mode_t : std::uint8_t {
  STORE_LAYOUT_NONE = 0,
  STORE_LAYOUT_COMPACT_C2 = 1,
  STORE_LAYOUT_COMPACT_C12 = 2,
  STORE_LAYOUT_COMPACT_C16 = 3,
  STORE_LAYOUT_COMPACT_C25 = 4,
  STORE_LAYOUT_COMPACT_C28 = 5,
  STORE_LAYOUT_C16_INTO_C19 = 6,
  STORE_LAYOUT_ALIGNED_TILE_COPY = 7,
  STORE_LAYOUT_NARROW_FIXED = 8,
  STORE_LAYOUT_COLD_RMW_FALLBACK = 15,
};

enum exec_kind_t : std::uint8_t {
  EXEC_NOP = 0,
  EXEC_CONV = 1,
  EXEC_POOL = 2,
  EXEC_AFFINE = 3,
  EXEC_STORE = 4,
  EXEC_ADD_AFFINE = 5,
  EXEC_END = 255,
};

struct window_pack_cmd_t {
  u8_t spatial_id;
  u8_t src_c_begin;
  u8_t dst_lane_begin;
  u8_t byte_count;
  u8_t flags;
  u8_t reserved0;
  u16_t reserved1;
};

struct window_sched_desc_t {
  u8_t mode;
  u8_t kernel;
  u8_t stride;
  u8_t dilation;
  u16_t in_c;
  u16_t k_tiles;
  u16_t cmd_base;
  u16_t cmd_count;
  u16_t kt_cmd_base[MAX_K_TILE_COUNT + 1];
  u16_t flags;
};

struct conv_exec_desc_t {
  u8_t param_id;
  u8_t qparam_id;
  u8_t window_sched_id;
  u8_t row_consumer_id;
  u16_t in_h;
  u16_t in_w;
  u16_t in_c;
  u16_t out_c;
  u8_t kernel;
  u8_t stride;
  u8_t dilation;
  u8_t padding;
  u8_t src_tensor;
  u8_t dst_tensor;
  u16_t dst_c_offset;
  u16_t valid_c;
  u32_t packed_weight_word_offset;
  u16_t k_tiles;
  u16_t weight_words;
  u16_t flags;
  u16_t reserved;
};

struct row_consumer_desc_t {
  u8_t mode;
  u8_t add_other_tensor;
  u8_t store_dst_tensor;
  u8_t add_qparam_id;
  u16_t store_c_offset;
  u16_t valid_c;
  u8_t alias_tensor;
  u8_t affine_param_id;
  u8_t affine_block_base;
  u8_t act_type;
  u16_t reserved0;
  u16_t reserved1;
};

struct exec_plan_entry_t {
  u8_t kind;
  u8_t desc_id;
  u8_t logical_uop_id;
  u8_t flags;
};

static_assert(sizeof(window_pack_cmd_t) == WINDOW_PACK_CMD_BLOB_BYTES,
              "window_pack_cmd_t must match PARAM v3 blob layout");
static_assert(sizeof(window_sched_desc_t) == WINDOW_SCHED_DESC_BLOB_BYTES,
              "window_sched_desc_t must match PARAM v3 blob layout");
static_assert(sizeof(conv_exec_desc_t) == CONV_EXEC_DESC_BLOB_BYTES,
              "conv_exec_desc_t must match PARAM v3 blob layout");
static_assert(sizeof(row_consumer_desc_t) == ROW_CONSUMER_DESC_BLOB_BYTES,
              "row_consumer_desc_t must match PARAM v3 blob layout");
static_assert(sizeof(exec_plan_entry_t) == EXEC_PLAN_ENTRY_BLOB_BYTES,
              "exec_plan_entry_t must match PARAM v3 blob layout");

}  // namespace esp_int8
