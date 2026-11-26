#pragma once

#include <cstdint>

namespace esp_int8 {

constexpr std::uint32_t PARAM_BLOB_MAGIC = 0x544E4945U;  // "EINT" in little-endian word view.
constexpr std::uint32_t PARAM_BLOB_VERSION_SCHED = 4U;

constexpr int AXI_WORD_BITS = 256;
constexpr int AXI_WORD_BYTES = AXI_WORD_BITS / 8;
constexpr int PARAM_HEADER_BYTES = 128;
constexpr int PARAM_HEADER_AXI_WORDS = PARAM_HEADER_BYTES / AXI_WORD_BYTES;

constexpr int TM = 32;
constexpr int TK = 32;

constexpr int MAX_FM_H = 512;
constexpr int MAX_FM_W = 512;
constexpr int MAX_FM_C = 256;
constexpr int MAX_C_TILE_COUNT = (MAX_FM_C + TM - 1) / TM;

// Schedule-executor build: keep this at 40, not the general 3x3*C256 bound.
// Current ESPNet encoder max K-tile count is ceil(3*3*131/32)=37; C256 1x1 is 8.
// Raising this expands window_sched_desc_t and breaks the v4 blob contract.
constexpr int MAX_K_TILE_COUNT = 40;

// BEGIN GENERATED RUNTIME PROFILES
struct runtime_profile_t {
  std::uint32_t profile_id;
  std::uint32_t input_h;
  std::uint32_t input_w;
  std::uint32_t input_c;
  std::uint32_t logits_h;
  std::uint32_t logits_w;
  std::uint32_t input_bytes;
  std::uint32_t mask_bytes;
  std::uint32_t b1_base;
  std::uint32_t l20_base;
  std::uint32_t l30_base;
  std::uint32_t out_base;
  std::uint32_t pool1_base;
  std::uint32_t pool_tmp_base;
  std::uint32_t pool1_bytes;
  std::uint32_t pool2_bytes;
  std::uint32_t l20_c1_base;
  std::uint32_t l2b0_c1_base;
  std::uint32_t l30_c1_base;
  std::uint32_t l3b0_c1_base;
  std::uint32_t l2_scratch_slot_bytes;
  std::uint32_t l3_scratch_slot_bytes;
  std::uint32_t l20_block5_base;
  std::uint32_t l2b0_block5_base;
  std::uint32_t l30_block5_base;
  std::uint32_t l3b0_block5_base;
  std::uint32_t b2_backup_base;
  std::uint32_t b2_backup_rows;
  std::uint32_t block5_rows;
};
constexpr unsigned RUNTIME_PROFILE_COUNT = 8;
constexpr std::uint32_t RUNTIME_ABI_ID = 0x11160001U;
constexpr std::uint32_t RUNTIME_CAPABILITY = 0x00007FFFU;
constexpr runtime_profile_t RUNTIME_PROFILES[RUNTIME_PROFILE_COUNT] = {
  {0x0U, 0xF0U, 0xF0U, 0x1U, 0x1EU, 0x1EU, 0xE100U, 0xE100U, 0xE100U, 0x38400U, 0x38400U, 0x38400U, 0x380000U, 0x383C00U, 0x3C00U, 0xF00U, 0x0U, 0x380F00U, 0x380F00U, 0x380F00U, 0xE100U, 0x5A00U, 0xE100U, 0x38F000U, 0x0U, 0x54600U, 0x380F00U, 0x20U, 0x20U},
  {0x1U, 0xF0U, 0xF0U, 0x3U, 0x1EU, 0x1EU, 0x2A300U, 0xE100U, 0x2A300U, 0x38400U, 0x38400U, 0x38400U, 0x380000U, 0x38B400U, 0xB400U, 0x2D00U, 0x0U, 0x382D00U, 0x382D00U, 0x382D00U, 0xE100U, 0x5A00U, 0xE100U, 0x390E00U, 0x0U, 0x54600U, 0x382D00U, 0x20U, 0x20U},
  {0x2U, 0x100U, 0x200U, 0x1U, 0x20U, 0x40U, 0x20000U, 0x20000U, 0x20000U, 0x80000U, 0x80000U, 0x80000U, 0x380000U, 0x388000U, 0x8000U, 0x2000U, 0x0U, 0x382000U, 0x382000U, 0x382000U, 0x20000U, 0xC800U, 0x20000U, 0x3A2000U, 0x0U, 0xC0000U, 0x382000U, 0x20U, 0x20U},
  {0x3U, 0x100U, 0x200U, 0x3U, 0x20U, 0x40U, 0x60000U, 0x20000U, 0x60000U, 0x80000U, 0x80000U, 0x80000U, 0x380000U, 0x398000U, 0x18000U, 0x6000U, 0x0U, 0x386000U, 0x386000U, 0x386000U, 0x20000U, 0xC800U, 0x20000U, 0x3A6000U, 0x0U, 0xC0000U, 0x386000U, 0x20U, 0x20U},
  {0x4U, 0x200U, 0x200U, 0x1U, 0x40U, 0x40U, 0x40000U, 0x40000U, 0x40000U, 0x100000U, 0x100000U, 0x100000U, 0x380000U, 0x390000U, 0x10000U, 0x4000U, 0x0U, 0x384000U, 0x384000U, 0x384000U, 0x40000U, 0x19000U, 0x40000U, 0x3C4000U, 0x0U, 0x180000U, 0x384000U, 0x40U, 0x20U},
  {0x5U, 0x200U, 0x200U, 0x3U, 0x40U, 0x40U, 0xC0000U, 0x40000U, 0xC0000U, 0x100000U, 0x100000U, 0x100000U, 0x380000U, 0x3B0000U, 0x30000U, 0xC000U, 0x0U, 0x38C000U, 0x38C000U, 0x38C000U, 0x40000U, 0x19000U, 0x40000U, 0x3CC000U, 0x0U, 0x180000U, 0x38C000U, 0x40U, 0x20U},
  {0x6U, 0x200U, 0x400U, 0x1U, 0x40U, 0x80U, 0x80000U, 0x80000U, 0x80000U, 0x200000U, 0x200000U, 0x200000U, 0x408000U, 0x428000U, 0x20000U, 0x8000U, 0x0U, 0x410000U, 0x410000U, 0x410000U, 0x80000U, 0x32000U, 0x80000U, 0x490000U, 0x0U, 0x300000U, 0x410000U, 0x40U, 0x20U},
  {0x7U, 0x200U, 0x400U, 0x3U, 0x40U, 0x80U, 0x180000U, 0x80000U, 0x180000U, 0x200000U, 0x200000U, 0x200000U, 0x418000U, 0x478000U, 0x60000U, 0x18000U, 0x0U, 0x430000U, 0x430000U, 0x430000U, 0x80000U, 0x32000U, 0x80000U, 0x4B0000U, 0x0U, 0x300000U, 0x430000U, 0x40U, 0x20U},
};
inline bool runtime_profile_lookup(std::uint32_t id, runtime_profile_t& profile) {
#pragma HLS INLINE
  profile = {};
  if (id >= RUNTIME_PROFILE_COUNT) { return false; }
  profile = RUNTIME_PROFILES[id];
  return true;
}
// END GENERATED RUNTIME PROFILES

// BEGIN GENERATED RUNTIME IO
struct runtime_io_t {
  std::uint32_t model_profile_id;
  std::uint32_t src_mode;
  std::uint32_t mask_mode;
  std::uint32_t source_h;
  std::uint32_t source_w;
  std::uint32_t source_c;
  std::uint32_t input_bytes;
  std::uint32_t roi_top;
  std::uint32_t roi_left;
  std::uint32_t roi_h;
  std::uint32_t roi_w;
  std::uint32_t mask_h;
  std::uint32_t mask_w;
  std::uint32_t mask_bytes;
  std::uint32_t nearest_index;
  std::uint32_t nearest_error;
  std::uint32_t nearest_base_step;
  std::uint32_t nearest_error_step;
  std::uint32_t axis_denominator;
  std::uint32_t axis_step;
  std::int32_t axis_origin;
};
inline bool runtime_io_lookup(std::uint32_t profile, std::uint32_t source,
                              std::uint32_t mask, runtime_io_t& io) {
#pragma HLS INLINE
  io = {};
  if (profile >= RUNTIME_PROFILE_COUNT || source > 6U || mask > 1U) { return false; }
  switch ((profile << 4) | (source << 1) | mask) {
    case 0: io = {0, 0, 0, 240, 240, 1, 57600, 0, 0, 240, 240, 240, 240, 57600, 0, 0, 0, 0, 16, 2, -7}; return true;
    case 2: io = {0, 1, 0, 256, 512, 1, 131072, 0, 128, 256, 256, 240, 240, 57600, 0, 256, 1, 32, 16, 2, -7}; return true;
    case 3: io = {0, 1, 1, 256, 512, 1, 131072, 0, 128, 256, 256, 256, 256, 65536, 0, 256, 1, 32, 256, 30, -113}; return true;
    case 4: io = {0, 2, 0, 512, 1024, 1, 524288, 0, 256, 512, 512, 240, 240, 57600, 1, 32, 2, 64, 16, 2, -7}; return true;
    case 5: io = {0, 2, 1, 512, 1024, 1, 524288, 0, 256, 512, 512, 512, 512, 262144, 1, 32, 2, 64, 512, 30, -241}; return true;
    case 6: io = {0, 3, 0, 512, 512, 1, 262144, 0, 0, 512, 512, 240, 240, 57600, 1, 32, 2, 64, 16, 2, -7}; return true;
    case 7: io = {0, 3, 1, 512, 512, 1, 262144, 0, 0, 512, 512, 512, 512, 262144, 1, 32, 2, 64, 512, 30, -241}; return true;
    case 16: io = {1, 0, 0, 240, 240, 3, 172800, 0, 0, 240, 240, 240, 240, 57600, 0, 0, 0, 0, 16, 2, -7}; return true;
    case 24: io = {1, 4, 0, 256, 512, 3, 393216, 0, 128, 256, 256, 240, 240, 57600, 0, 256, 1, 32, 16, 2, -7}; return true;
    case 25: io = {1, 4, 1, 256, 512, 3, 393216, 0, 128, 256, 256, 256, 256, 65536, 0, 256, 1, 32, 256, 30, -113}; return true;
    case 26: io = {1, 5, 0, 512, 1024, 3, 1572864, 0, 256, 512, 512, 240, 240, 57600, 1, 32, 2, 64, 16, 2, -7}; return true;
    case 27: io = {1, 5, 1, 512, 1024, 3, 1572864, 0, 256, 512, 512, 512, 512, 262144, 1, 32, 2, 64, 512, 30, -241}; return true;
    case 28: io = {1, 6, 0, 512, 512, 3, 786432, 0, 0, 512, 512, 240, 240, 57600, 1, 32, 2, 64, 16, 2, -7}; return true;
    case 29: io = {1, 6, 1, 512, 512, 3, 786432, 0, 0, 512, 512, 512, 512, 262144, 1, 32, 2, 64, 512, 30, -241}; return true;
    case 32: io = {2, 0, 0, 256, 512, 1, 131072, 0, 0, 256, 512, 256, 512, 131072, 0, 0, 0, 0, 16, 2, -7}; return true;
    case 48: io = {3, 0, 0, 256, 512, 3, 393216, 0, 0, 256, 512, 256, 512, 131072, 0, 0, 0, 0, 16, 2, -7}; return true;
    case 64: io = {4, 0, 0, 512, 512, 1, 262144, 0, 0, 512, 512, 512, 512, 262144, 0, 0, 0, 0, 16, 2, -7}; return true;
    case 80: io = {5, 0, 0, 512, 512, 3, 786432, 0, 0, 512, 512, 512, 512, 262144, 0, 0, 0, 0, 16, 2, -7}; return true;
    case 96: io = {6, 0, 0, 512, 1024, 1, 524288, 0, 0, 512, 1024, 512, 1024, 524288, 0, 0, 0, 0, 16, 2, -7}; return true;
    case 112: io = {7, 0, 0, 512, 1024, 3, 1572864, 0, 0, 512, 1024, 512, 1024, 524288, 0, 0, 0, 0, 16, 2, -7}; return true;
    default: return false;
  }
}
// END GENERATED RUNTIME IO

constexpr unsigned DEFAULT_RUNTIME_PROFILE_ID = 4;
bool runtime_select_profile(std::uint32_t id);
bool runtime_profile_ready();
void runtime_get_profile(runtime_profile_t& profile);
bool runtime_select_io(std::uint32_t source, std::uint32_t mask);
bool runtime_io_ready();
void runtime_get_io(runtime_io_t& io);

constexpr int MAX_INPUT_FRAME_H = 512;
constexpr int MAX_INPUT_FRAME_W = 1024;
constexpr int MAX_INPUT_FRAME_C = 3;
constexpr int MAX_INPUT_FRAME_AXI_WORDS = 49152;
constexpr int MAX_OUTPUT_FRAME_AXI_WORDS = 16384;
constexpr int MAX_ENCODER_OUT_H = 64;
constexpr int MAX_ENCODER_OUT_W = 128;

// Default native512 dimensions are retained for testbench fixtures only.
constexpr int INPUT_FRAME_H = 512;
constexpr int INPUT_FRAME_W = 512;
constexpr int INPUT_FRAME_C = 1;
constexpr int ENCODER_OUT_H = 64;
constexpr int ENCODER_OUT_W = 64;
constexpr int MAX_CLASS_C = 32;
constexpr int UPSAMPLE_CLASS_LANES = 4;
// Binary native512 upsample and golden/debug compatibility.
constexpr int ENCODER_OUT_C = 2;
constexpr int UPSAMPLE_SCALE = 8;
constexpr int FULLRES_MASK_H = INPUT_FRAME_H;
constexpr int FULLRES_MASK_W = INPUT_FRAME_W;
constexpr int FULLRES_MASK_C = 1;

constexpr int INPUT_FRAME_BYTES = INPUT_FRAME_H * INPUT_FRAME_W * INPUT_FRAME_C;
constexpr int INPUT_FRAME_AXI_WORDS = (INPUT_FRAME_BYTES + AXI_WORD_BYTES - 1) / AXI_WORD_BYTES;
constexpr int ENCODER_LOGITS_MAX_BYTES = MAX_ENCODER_OUT_H * MAX_ENCODER_OUT_W * MAX_CLASS_C;
constexpr int ENCODER_LOGITS_BYTES = ENCODER_OUT_H * ENCODER_OUT_W * ENCODER_OUT_C;
constexpr int FULLRES_MASK_BYTES = FULLRES_MASK_H * FULLRES_MASK_W * FULLRES_MASK_C;
constexpr int OUTPUT_FRAME_BYTES = FULLRES_MASK_BYTES;
constexpr int OUTPUT_FRAME_AXI_WORDS = (OUTPUT_FRAME_BYTES + AXI_WORD_BYTES - 1) / AXI_WORD_BYTES;

static_assert(INPUT_FRAME_H == ENCODER_OUT_H * UPSAMPLE_SCALE,
              "Input/logits height must match the fixed upsample scale");
static_assert(INPUT_FRAME_W == ENCODER_OUT_W * UPSAMPLE_SCALE,
              "Input/logits width must match the fixed upsample scale");
static_assert(INPUT_FRAME_AXI_WORDS == 8192,
              "Native512 C1 input must occupy 8192 AXI words");
static_assert(OUTPUT_FRAME_AXI_WORDS == 8192,
              "Native512 mask must occupy 8192 AXI words");

// One physical allocation for all runtime profiles. Bases are latched at INIT.
constexpr int FMBUF_BYTES = 0x530000;
constexpr int FMBUF_URAM_BYTES = 0x380000;
constexpr int FMBUF_BRAM_BYTES = FMBUF_BYTES - FMBUF_URAM_BYTES;
constexpr int FMBUF_URAM_AXI_WORDS = (FMBUF_URAM_BYTES + AXI_WORD_BYTES - 1) / AXI_WORD_BYTES;
constexpr int FMBUF_BRAM_AXI_WORDS = (FMBUF_BRAM_BYTES + AXI_WORD_BYTES - 1) / AXI_WORD_BYTES;
constexpr int FMBUF_L20_PHYS_C = 64;
constexpr int FMBUF_L20_C_OFFSET = 0;
constexpr int BLOCK5_ROW_BLOCK_ROWS = 32;
constexpr int ROW_CONTIG_MAX_W = MAX_INPUT_FRAME_W / 2;
constexpr int ROW_CONTIG_MAX_C = 131;
constexpr int ROW_CONTIG_MAX_BYTES = (MAX_INPUT_FRAME_W / 4) * ROW_CONTIG_MAX_C;
constexpr int ROW_CONTIG_MAX_WORDS = (ROW_CONTIG_MAX_BYTES + AXI_WORD_BYTES - 1) / AXI_WORD_BYTES;
constexpr int BRAM_SCR0_BYTES = 256 * 512 * 3;
constexpr int BRAM_SCR1_BYTES = 128 * 256 * 3;
constexpr int BRAM_SCR0_AXI_WORDS = (BRAM_SCR0_BYTES + AXI_WORD_BYTES - 1) / AXI_WORD_BYTES;
constexpr int BRAM_SCR1_AXI_WORDS = (BRAM_SCR1_BYTES + AXI_WORD_BYTES - 1) / AXI_WORD_BYTES;
constexpr int WBUF_BYTES = 124 * 1024;
constexpr int WBUF_AXI_WORDS = (WBUF_BYTES + AXI_WORD_BYTES - 1) / AXI_WORD_BYTES;

constexpr int SCALE_DESC_COUNT_MAX = 128;
constexpr int CONV_PARAM_DESC_COUNT = 26;
constexpr int AFFINE_PARAM_DESC_COUNT = 7;
constexpr int ADD_PARAM_DESC_COUNT = 14;
constexpr int POOL_PARAM_DESC_COUNT = 3;

constexpr int UOP_COUNT_ENCODER = 75;

// All eight profiles execute the same finite ESPNet graph, not arbitrary networks.
constexpr int MAX_TENSOR_DESC_COUNT = 19;
constexpr int MAX_CONV_PARAM_DESC_COUNT = CONV_PARAM_DESC_COUNT;
constexpr int MAX_AFFINE_PARAM_DESC_COUNT = AFFINE_PARAM_DESC_COUNT;
constexpr int MAX_ADD_PARAM_DESC_COUNT = ADD_PARAM_DESC_COUNT;
constexpr int MAX_POOL_PARAM_DESC_COUNT = POOL_PARAM_DESC_COUNT;

// v4 schedule section upper bounds. Keep these tight; if exporter exceeds them,
// fix the schedule generator instead of expanding HLS arrays.
constexpr int MAX_CONV_EXEC_DESC_COUNT = CONV_PARAM_DESC_COUNT;
constexpr int MAX_WINDOW_SCHED_COUNT = 16;
constexpr int MAX_WINDOW_PACK_CMD_COUNT = 256;
constexpr int MAX_ROW_CONSUMER_DESC_COUNT = CONV_PARAM_DESC_COUNT;
constexpr int MAX_FIXED_EXEC_DESC_COUNT = 10;  // Includes v4 section-alignment slots.
constexpr int MAX_EXEC_PLAN_COUNT = 16;
constexpr int MAX_BLOCK5_SCHED_COUNT = 8;
constexpr int MAX_PACK_CMDS_PER_KT = 9;
constexpr int MAX_STAGED_WINDOW_PACK_CMDS = 64;
constexpr int MAX_3X3_CACHE_CHUNKS = 5;
constexpr int WINGEN_NARROW_CACHE_COL_SLOTS = 64;
constexpr int WINGEN_WIDE_CACHE_COL_SLOTS = 4;

constexpr int WINDOW_PACK_CMD_BLOB_BYTES = 8;
constexpr int WINDOW_SCHED_DESC_BLOB_BYTES = 128;
constexpr int CONV_EXEC_DESC_BLOB_BYTES = 36;
constexpr int ROW_CONSUMER_DESC_BLOB_BYTES = 16;
constexpr int FIXED_EXEC_DESC_BLOB_BYTES = 32;
constexpr int EXEC_PLAN_ENTRY_BLOB_BYTES = 4;
constexpr int BLOCK5_SCHED_DESC_BLOB_BYTES = 32;

constexpr int SECTION_ALIGNMENT_BYTES = 64;

static_assert(MAX_K_TILE_COUNT == 40, "v4 WINDOW_SCHED_DESC_BLOB_BYTES assumes 40 K tiles");
static_assert(WINDOW_SCHED_DESC_BLOB_BYTES == 18 + (MAX_K_TILE_COUNT + 1) * 2 + 28,
              "window_sched_desc_t blob size mismatch");
static_assert(ROW_CONTIG_MAX_WORDS == 1048, "Maximum compact row requires an 11-bit cursor");
static_assert(FMBUF_URAM_BYTES == 112 * 32768, "Unified URAM allocation must stay within ZU15EG");

}  // namespace esp_int8
