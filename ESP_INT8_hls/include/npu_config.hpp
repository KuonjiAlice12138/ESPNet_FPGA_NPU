#pragma once

#include <cstdint>

namespace esp_int8 {

constexpr std::uint32_t PARAM_BLOB_MAGIC = 0x544E4945U;  // "EINT" in little-endian word view.
constexpr std::uint32_t PARAM_BLOB_VERSION = 0x00010000U;

constexpr int AXI_WORD_BITS = 256;
constexpr int AXI_WORD_BYTES = AXI_WORD_BITS / 8;
constexpr int PARAM_HEADER_BYTES = 128;
constexpr int PARAM_HEADER_AXI_WORDS = PARAM_HEADER_BYTES / AXI_WORD_BYTES;

constexpr int TM = 32;
constexpr int TK = 32;

constexpr int MAX_FM_H = 512;
constexpr int MAX_FM_W = 1024;
constexpr int MAX_FM_C = 256;
constexpr int MAX_KERNEL_ELEMS = 9;
constexpr int MAX_C_TILE_COUNT = (MAX_FM_C + TM - 1) / TM;
constexpr int MAX_K_TILE_COUNT = (MAX_FM_C * MAX_KERNEL_ELEMS + TK - 1) / TK;

constexpr int INPUT_FRAME_BYTES = 512 * 1024 * 3;
constexpr int INPUT_FRAME_AXI_WORDS = (INPUT_FRAME_BYTES + AXI_WORD_BYTES - 1) / AXI_WORD_BYTES;
constexpr int OUTPUT_FRAME_BYTES = 64 * 128 * 2;
constexpr int OUTPUT_FRAME_AXI_WORDS = (OUTPUT_FRAME_BYTES + AXI_WORD_BYTES - 1) / AXI_WORD_BYTES;

constexpr std::uint32_t RUNTIME_MODE_MASK = 0x0000000fU;
constexpr std::uint32_t DEBUG_MODE_ENABLE_MASK = 0x00000010U;
constexpr int DEBUG_MODE_DUMP_TENSOR_SHIFT = 8;
constexpr int DEBUG_MODE_DUMP_WORDS_SHIFT = 16;
constexpr std::uint32_t DEBUG_MODE_DUMP_TENSOR_MASK = 0x0000ff00U;
constexpr std::uint32_t DEBUG_MODE_DUMP_WORDS_MASK = 0xffff0000U;
constexpr std::uint32_t DEBUG_UOP_COUNT_MASK = 0x0000ffffU;
constexpr int DEBUG_STOP_AFTER_SHIFT = 16;
constexpr std::uint32_t DEBUG_STOP_AFTER_MASK = 0xffff0000U;
constexpr std::uint32_t DEBUG_DUMP_TENSOR_DISABLED = 0xffU;
constexpr int DEBUG_DUMP_MAX_AXI_WORDS = OUTPUT_FRAME_AXI_WORDS;

constexpr int FMBUF_BYTES = 0x598000;
constexpr int FMBUF_BANK_COUNT = 3;
constexpr int FMBUF_URAM_BYTES = 0x380000;
constexpr int FMBUF_BRAM_BYTES = FMBUF_BYTES - FMBUF_URAM_BYTES;
constexpr int FMEM0_BYTES = 0x418000;
constexpr int FMEM1_BYTES = 0x200000;
constexpr int FMEM2_BYTES = 0x000000;
constexpr int FMEM0_AXI_WORDS = (FMEM0_BYTES + AXI_WORD_BYTES - 1) / AXI_WORD_BYTES;
constexpr int FMEM1_AXI_WORDS = (FMEM1_BYTES + AXI_WORD_BYTES - 1) / AXI_WORD_BYTES;
constexpr int FMEM2_AXI_WORDS = (FMEM2_BYTES + AXI_WORD_BYTES - 1) / AXI_WORD_BYTES;
constexpr int FMBUF_URAM_AXI_WORDS = (FMBUF_URAM_BYTES + AXI_WORD_BYTES - 1) / AXI_WORD_BYTES;
constexpr int FMBUF_BRAM_AXI_WORDS = (FMBUF_BRAM_BYTES + AXI_WORD_BYTES - 1) / AXI_WORD_BYTES;
constexpr int FMBUF_L20_BASE = 0x000000;
constexpr int FMBUF_L30_BASE = 0x418000;
constexpr int FMBUF_POOL1_BASE = 0x3E0000;
constexpr int FMBUF_POOL_TMP_BASE = 0x440000;
constexpr int FMBUF_L20_PHYS_C = 131;
constexpr int FMBUF_L20_C_OFFSET = 64;
constexpr int FMBUF_L2_SCRATCH_BASE = 0x418000;
constexpr int FMBUF_L2_SCRATCH_SLOT_BYTES = 128 * 256 * 12;
constexpr int FMBUF_L30_SCRATCH_C1_BASE = 0x518000;
constexpr int FMBUF_L30_SCRATCH_LOW_BASE = 0x000000;
constexpr int FMBUF_L30_SCRATCH_SLOT_BYTES = 64 * 128 * 25;
constexpr int BRAM_SCR0_BYTES = 256 * 512 * 3;
constexpr int BRAM_SCR1_BYTES = 128 * 256 * 3;
constexpr int BRAM_SCR0_AXI_WORDS = (BRAM_SCR0_BYTES + AXI_WORD_BYTES - 1) / AXI_WORD_BYTES;
constexpr int BRAM_SCR1_AXI_WORDS = (BRAM_SCR1_BYTES + AXI_WORD_BYTES - 1) / AXI_WORD_BYTES;
constexpr int WBUF_BYTES = 128 * 1024;
constexpr int WBUF_AXI_WORDS = (WBUF_BYTES + AXI_WORD_BYTES - 1) / AXI_WORD_BYTES;
constexpr int QBUF0_BYTES = 64 * 1024;
constexpr int QBUF1_BYTES = 64 * 1024;
constexpr int QBUF_BYTES = QBUF0_BYTES + QBUF1_BYTES;
constexpr int SCRATCH_BRAM_BYTES = 64 * 1024;

constexpr int TENSOR_DESC_COUNT = 19;
constexpr int SCALE_DESC_COUNT_MAX = 128;
constexpr int CONV_PARAM_DESC_COUNT = 26;
constexpr int AFFINE_PARAM_DESC_COUNT = 7;
constexpr int ADD_PARAM_DESC_COUNT = 14;
constexpr int POOL_PARAM_DESC_COUNT = 3;

constexpr int UOP_COUNT_ENCODER = 75;
constexpr int MAX_UOP_COUNT = 128;

constexpr int MAX_TENSOR_DESC_COUNT = 64;
constexpr int MAX_CONV_PARAM_DESC_COUNT = 32;
constexpr int MAX_AFFINE_PARAM_DESC_COUNT = 16;
constexpr int MAX_ADD_PARAM_DESC_COUNT = 16;
constexpr int MAX_POOL_PARAM_DESC_COUNT = 8;

constexpr int SECTION_ALIGNMENT_BYTES = 64;

static_assert(FMBUF_L2_SCRATCH_BASE + 4 * FMBUF_L2_SCRATCH_SLOT_BYTES <= FMBUF_BYTES,
              "Level2 scratch must fit in FMBUF");
static_assert(FMBUF_L30_BASE + 64 * 128 * 128 <= FMBUF_L30_SCRATCH_C1_BASE,
              "Level3 compact tensor must not overlap LS_C1 scratch");
static_assert(FMBUF_L30_SCRATCH_C1_BASE + FMBUF_L30_SCRATCH_SLOT_BYTES <= FMBUF_BYTES,
              "Level3 LS_C1 scratch must fit in FMBUF");
static_assert(FMBUF_POOL_TMP_BASE + BRAM_SCR0_BYTES <= FMBUF_BYTES,
              "Pool scratch must fit in FMBUF");

}  // namespace esp_int8
