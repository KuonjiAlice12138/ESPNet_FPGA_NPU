#pragma once

#include "npu_config.hpp"
#include "npu_types.hpp"

namespace esp_int8 {

enum window_mode_t : std::uint8_t {
  WIN_MODE_INVALID = 0,
  WIN_MODE_3X3_RESERVED = 1,
  WIN_MODE_1X1_ALIGNED = 2,
  WIN_MODE_1X1_PACKED = 3,
  WIN_MODE_3X3_STAGED_C3 = 4,
  WIN_MODE_3X3_STAGED_C12 = 5,
  WIN_MODE_3X3_STAGED_C19 = 6,
  WIN_MODE_3X3_STAGED_C25 = 7,
  WIN_MODE_3X3_STAGED_C131 = 8,
  WIN_MODE_3X3_STAGED_C28 = 9,
  WIN_MODE_3X3_STAGED_C64 = 10,
  WIN_MODE_3X3_STAGED_C128 = 11,
};

enum pack_cmd_flags_t : std::uint8_t {
  PACK_CMD_VALID = 1 << 0,
  PACK_CMD_ZERO = 1 << 1,
  PACK_CMD_CONTIG_READ = 1 << 2,
  PACK_CMD_ALIGNED_READ = 1 << 3,
};

enum window_sched_flags_t : std::uint8_t {
  WINDOW_SCHED_FLAG_PIXEL_PARALLEL_2 = 1 << 0,
  WINDOW_SCHED_FLAG_ODD_TAIL = 1 << 1,
};

enum window_loader_class_t : std::uint8_t {
  WIN_LOADER_DIRECT_1X1 = 0,
  WIN_LOADER_3X3_NARROW = 1,
  WIN_LOADER_3X3_WIDE = 2,
};

constexpr int WINDOW_LOADER_RUN_MAX = 3;
constexpr int WINDOW_LOADER_WARMUP_COUNT_WORD = 0;
constexpr int WINDOW_LOADER_WARMUP_RUN_WORD = 1;
constexpr int WINDOW_LOADER_STEADY_COUNT_WORD = 4;
constexpr int WINDOW_LOADER_STEADY_RUN_WORD = 5;
constexpr int WINDOW_LOADER_PHASE_SPLIT_WORD = 8;
constexpr int WINDOW_LOADER_ROW_REUSE_WORD = 9;
constexpr int WINDOW_LOADER_RESERVED_WORDS = 10;

enum window_row_reuse_mode_t : std::uint8_t {
  WINDOW_ROW_REUSE_NONE = 0,
  WINDOW_ROW_REUSE_STRIDE1_KEEP2 = 1,
  WINDOW_ROW_REUSE_STRIDE2_KEEP1 = 2,
};

constexpr unsigned WINDOW_ROW_REUSE_MODE_MASK = 0x3U;
constexpr unsigned WINDOW_ROW_REUSE_WORDS_SHIFT = 2U;
constexpr unsigned WINDOW_ROW_REUSE_WORDS_MASK = 0x7FU;
constexpr unsigned WINDOW_ROW_REUSE_RESERVED_SHIFT = 9U;

enum row_consumer_mode_t : std::uint8_t {
  ROW_CONSUMER_NONE = 0,
  ROW_CONSUMER_STORE = 1,
  ROW_CONSUMER_ADD_STORE = 2,
  ROW_CONSUMER_ADD_AFFINE_STORE = 3,
  ROW_CONSUMER_UPSAMPLE_OUT = 4,
  ROW_CONSUMER_AFFINE_STORE = 5,
  ROW_CONSUMER_CAT_AFFINE_STORE = 6,
  ROW_CONSUMER_ADD_STORE_BLOCK_ADD_AFFINE = 7,
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
  STORE_LAYOUT_COMPACT_C19 = 9,
  STORE_LAYOUT_PREFIX_ZERO_PAD = 10,
};

enum exec_kind_t : std::uint8_t {
  EXEC_NOP = 0,
  EXEC_CONV = 1,
  EXEC_POOL = 2,
  EXEC_BLOCK_AFFINE = 6,
  EXEC_BLOCK_ADD_AFFINE = 7,
  EXEC_END = 255,
};

enum fixed_exec_flags_t : std::uint8_t {
  FIXED_FLAG_BLOCK5_AFFINE = 1 << 0,
  FIXED_FLAG_BLOCK5_ADD_AFFINE = 1 << 1,
  FIXED_FLAG_BLOCK5_ROW_GROUP = 1 << 2,
  FIXED_FLAG_CBLOCK_MAJOR = 1 << 3,
  FIXED_FLAG_ROW_CONTIGUOUS_STORE = 1 << 7,
};

enum block5_pattern_t : std::uint8_t {
  BLOCK5_PATTERN_INVALID = 0,
  BLOCK5_PATTERN_L2_C16_4C12 = 1,
  BLOCK5_PATTERN_L3_C28_4C25 = 2,
};

enum block5_finalizer_kind_t : std::uint8_t {
  BLOCK5_FINALIZER_INVALID = 0,
  BLOCK5_FINALIZER_L2 = 1,
  BLOCK5_FINALIZER_L3 = 2,
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
  u8_t padding;
  u8_t cache_chunks;
  u8_t cache_col_slots;
  u8_t flags;
  u16_t in_c;
  u16_t out_w;
  u16_t k_tiles;
  u16_t cmd_base;
  u16_t cmd_count;
  u16_t kt_cmd_base[MAX_K_TILE_COUNT + 1];
  u8_t loader_class;
  u8_t loader_request_cols;
  u8_t loader_warmup_issues;
  u8_t loader_warmup_new_cols;
  u8_t loader_steady_new_cols;
  u8_t loader_words_per_col;
  u8_t loader_warmup_mask;
  u8_t loader_steady_mask;
  u16_t reserved[WINDOW_LOADER_RESERVED_WORDS];
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

struct fixed_exec_desc_t {
  u8_t kind;
  u8_t src0_tensor;
  u8_t src1_tensor;
  u8_t dst_tensor;
  u8_t param_id;
  u8_t add_param_id;
  u8_t act_type;
  u8_t flags;
  u16_t in_h;
  u16_t in_w;
  u16_t in_c;
  u16_t out_c;
  u8_t kernel;
  u8_t stride;
  u8_t dilation;
  u8_t padding;
  u16_t c_offset;
  u16_t valid_c;
  u16_t qparam_id;
  u16_t reserved0;
  u32_t reserved1;
};

struct exec_plan_entry_t {
  u8_t kind;
  u8_t desc_id;
  u8_t logical_uop_id;
  u8_t flags;
};

struct block5_sched_desc_t {
  u8_t pattern;
  u8_t branch_count;
  u8_t first_branch_conv_id;
  u8_t first_window_sched_id;
  u8_t first_conv_qparam_id;
  u8_t src_tensor;
  u8_t dst_tensor;
  u8_t add_tensor;
  u8_t chain_add_qparam_id0;
  u8_t chain_add_qparam_id1;
  u8_t chain_add_qparam_id2;
  u8_t residual_add_qparam_id;
  u8_t affine_param_id;
  u8_t affine_block_count;
  u8_t finalizer_kind;
  u8_t scratch_region;
  u16_t out_h;
  u16_t out_w;
  u16_t row_group_h;
  u16_t valid_c;
  u16_t reserved0;
  u16_t reserved1;
  u32_t reserved2;
};

static_assert(sizeof(window_pack_cmd_t) == WINDOW_PACK_CMD_BLOB_BYTES,
              "window_pack_cmd_t must match PARAM v4 blob layout");
static_assert(sizeof(window_sched_desc_t) == WINDOW_SCHED_DESC_BLOB_BYTES,
              "window_sched_desc_t must match PARAM v4 blob layout");
static_assert(sizeof(conv_exec_desc_t) == CONV_EXEC_DESC_BLOB_BYTES,
              "conv_exec_desc_t must match PARAM v4 blob layout");
static_assert(sizeof(row_consumer_desc_t) == ROW_CONSUMER_DESC_BLOB_BYTES,
              "row_consumer_desc_t must match PARAM v4 blob layout");
static_assert(sizeof(fixed_exec_desc_t) == FIXED_EXEC_DESC_BLOB_BYTES,
              "fixed_exec_desc_t must match PARAM v4 blob layout");
static_assert(sizeof(exec_plan_entry_t) == EXEC_PLAN_ENTRY_BLOB_BYTES,
              "exec_plan_entry_t must match PARAM v4 blob layout");
static_assert(sizeof(block5_sched_desc_t) == BLOCK5_SCHED_DESC_BLOB_BYTES,
              "block5_sched_desc_t must match PARAM v4 blob layout");

}  // namespace esp_int8
