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

constexpr int MAX_FM_H = 256;
constexpr int MAX_FM_W = 512;
constexpr int MAX_FM_C = 256;
constexpr int MAX_KERNEL_ELEMS = 9;
constexpr int MAX_C_TILE_COUNT = (MAX_FM_C + TM - 1) / TM;

// Schedule-executor build: keep this at 40, not the general 3x3*C256 bound.
// Current ESPNet encoder max K-tile count is ceil(3*3*131/32)=37; C256 1x1 is 8.
// Raising this expands window_sched_desc_t and breaks the v4 blob contract.
constexpr int MAX_K_TILE_COUNT = 40;

constexpr int INPUT_FRAME_H = 256;
constexpr int INPUT_FRAME_W = 512;
constexpr int INPUT_FRAME_C = 3;
constexpr int ENCODER_OUT_H = 32;
constexpr int ENCODER_OUT_W = 64;
constexpr int MAX_CLASS_C = TM;
constexpr int UPSAMPLE_CLASS_LANES = 4;
// Binary golden/debug TB compatibility only; the synthesized path uses valid_c.
constexpr int ENCODER_OUT_C = 2;
constexpr int UPSAMPLE_SCALE = 8;
constexpr int FULLRES_MASK_H = INPUT_FRAME_H;
constexpr int FULLRES_MASK_W = INPUT_FRAME_W;
constexpr int FULLRES_MASK_C = 1;

constexpr int INPUT_FRAME_BYTES = INPUT_FRAME_H * INPUT_FRAME_W * INPUT_FRAME_C;
constexpr int INPUT_FRAME_AXI_WORDS = (INPUT_FRAME_BYTES + AXI_WORD_BYTES - 1) / AXI_WORD_BYTES;
constexpr int ENCODER_LOGITS_MAX_BYTES = ENCODER_OUT_H * ENCODER_OUT_W * MAX_CLASS_C;
constexpr int ENCODER_LOGITS_BYTES = ENCODER_OUT_H * ENCODER_OUT_W * ENCODER_OUT_C;
constexpr int FULLRES_MASK_BYTES = FULLRES_MASK_H * FULLRES_MASK_W * FULLRES_MASK_C;
constexpr int OUTPUT_FRAME_BYTES = FULLRES_MASK_BYTES;
constexpr int OUTPUT_FRAME_AXI_WORDS = (OUTPUT_FRAME_BYTES + AXI_WORD_BYTES - 1) / AXI_WORD_BYTES;

static_assert(INPUT_FRAME_H == ENCODER_OUT_H * UPSAMPLE_SCALE,
              "Input/logits height must match the fixed upsample scale");
static_assert(INPUT_FRAME_W == ENCODER_OUT_W * UPSAMPLE_SCALE,
              "Input/logits width must match the fixed upsample scale");
static_assert(INPUT_FRAME_AXI_WORDS == 12288,
              "H256W512 INT8 input must occupy 12288 AXI words");
static_assert(OUTPUT_FRAME_AXI_WORDS == 4096,
              "H256W512 mask must occupy 4096 AXI words");

constexpr std::uint32_t RUNTIME_MODE_MASK = 0x0000000fU;
constexpr std::uint32_t RUNTIME_UOP_COUNT_MASK = 0x0000ffffU;

// H256 lifetime-allocated shared feature memory. Main tensors and BLOCK5
// workspaces occupy the URAM prefix; Pool tensors occupy the BRAM-only tail.
constexpr int FMBUF_BYTES = 0x1F0000;
constexpr int FMBUF_URAM_BYTES = 0x1C0000;
constexpr int FMBUF_BRAM_BYTES = FMBUF_BYTES - FMBUF_URAM_BYTES;
constexpr int FMBUF_URAM_AXI_WORDS = (FMBUF_URAM_BYTES + AXI_WORD_BYTES - 1) / AXI_WORD_BYTES;
constexpr int FMBUF_BRAM_AXI_WORDS = (FMBUF_BRAM_BYTES + AXI_WORD_BYTES - 1) / AXI_WORD_BYTES;
constexpr int FMBUF_PRIMARY_BASE = 0x000000;
constexpr int FMBUF_BLOCK5_BASE = 0x080000;
constexpr int FMBUF_B1_BASE = 0x108000;
constexpr int FMBUF_L20_BASE = 0x108000;
constexpr int FMBUF_L30_BASE = 0x108000;
constexpr int FMBUF_OUT_BASE = 0x108000;
constexpr int FMBUF_POOL1_BASE = 0x1C0000;
constexpr int FMBUF_POOL_TMP_BASE = 0x1D8000;
constexpr int FMBUF_L20_PHYS_C = 64;
constexpr int FMBUF_L20_C_OFFSET = 0;
constexpr int FMBUF_L2_SCRATCH_BASE = 0x1A0000;
constexpr int FMBUF_L2_SCRATCH_SLOT_BYTES = 64 * 128 * 16;
constexpr int FMBUF_L30_SCRATCH_C1_BASE = 0x1A0000;
constexpr int FMBUF_L30_SCRATCH_LOW_BASE = FMBUF_BLOCK5_BASE;
constexpr int FMBUF_L30_SCRATCH_SLOT_BYTES = 32 * 64 * 25;
constexpr int FMBUF_L3B0_SCRATCH_BASE = 0x1A0000;
constexpr int FMBUF_L3B0_SCRATCH_SLOT_BYTES = 32 * 64 * 25;
constexpr int BLOCK5_ROW_BLOCK_ROWS = 64;
constexpr int FMBUF_L2_BLOCK5_BASE = FMBUF_BLOCK5_BASE;
constexpr int FMBUF_L30_BLOCK5_BASE = FMBUF_BLOCK5_BASE;
constexpr int FMBUF_L3B0_BLOCK5_BASE = FMBUF_BLOCK5_BASE;
constexpr int ROW_CONTIG_MAX_W = 128;
constexpr int ROW_CONTIG_MAX_C = 131;
constexpr int ROW_CONTIG_MAX_BYTES = ROW_CONTIG_MAX_W * ROW_CONTIG_MAX_C;
constexpr int ROW_CONTIG_MAX_WORDS = (ROW_CONTIG_MAX_BYTES + AXI_WORD_BYTES - 1) / AXI_WORD_BYTES;
constexpr int BRAM_SCR0_BYTES = 128 * 256 * 3;
constexpr int BRAM_SCR1_BYTES = 64 * 128 * 3;
constexpr int BRAM_SCR0_AXI_WORDS = (BRAM_SCR0_BYTES + AXI_WORD_BYTES - 1) / AXI_WORD_BYTES;
constexpr int BRAM_SCR1_AXI_WORDS = (BRAM_SCR1_BYTES + AXI_WORD_BYTES - 1) / AXI_WORD_BYTES;
constexpr int FMBUF_POOL2_ALIAS_BASE = FMBUF_POOL1_BASE;
constexpr int FMBUF_POOL2_ALIAS_BYTES = BRAM_SCR1_BYTES;
constexpr int WBUF_BYTES = 124 * 1024;
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

constexpr int MAX_TENSOR_DESC_COUNT = 64;
constexpr int MAX_CONV_PARAM_DESC_COUNT = 32;
constexpr int MAX_AFFINE_PARAM_DESC_COUNT = 16;
constexpr int MAX_ADD_PARAM_DESC_COUNT = 16;
constexpr int MAX_POOL_PARAM_DESC_COUNT = 8;

// v4 schedule section upper bounds. Keep these tight; if exporter exceeds them,
// fix the schedule generator instead of expanding HLS arrays.
constexpr int MAX_CONV_EXEC_DESC_COUNT = 32;
constexpr int MAX_WINDOW_SCHED_COUNT = 16;
constexpr int MAX_WINDOW_PACK_CMD_COUNT = 256;
constexpr int MAX_ROW_CONSUMER_DESC_COUNT = 64;
constexpr int MAX_FIXED_EXEC_DESC_COUNT = 64;
constexpr int MAX_EXEC_PLAN_COUNT = 96;
constexpr int MAX_BLOCK5_SCHED_COUNT = 8;
constexpr int MAX_PACK_CMDS_PER_KT = 9;
constexpr int MAX_STAGED_WINDOW_PACK_CMDS = 64;
constexpr int MAX_3X3_CACHE_CHUNKS = 5;
constexpr int WINGEN_NARROW_CACHE_COL_SLOTS = 64;
constexpr int WINGEN_NARROW_ROW_WORDS = 96;
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
static_assert(64 * 128 * 131 <= FMBUF_L20_BASE,
              "B2 output must end before the residual slot without a backup");
static_assert(FMBUF_B1_BASE + 128 * 256 * 19 <= FMBUF_L2_SCRATCH_BASE,
              "The B1 tensor must end before C1 scratch");
static_assert(FMBUF_L2_BLOCK5_BASE + 64 * 128 * 64 <= FMBUF_L20_BASE,
              "Level2 BLOCK5 workspace must end before the residual slot");
static_assert(FMBUF_L2_SCRATCH_BASE + FMBUF_L2_SCRATCH_SLOT_BYTES <= FMBUF_URAM_BYTES,
              "Level2 C1 scratch must fit in the URAM prefix");
static_assert(FMBUF_L30_BASE + 32 * 64 * 128 <= FMBUF_L30_SCRATCH_C1_BASE,
              "Level3 compact tensor must not overlap LS_C1 scratch");
static_assert(FMBUF_L30_SCRATCH_C1_BASE + FMBUF_L30_SCRATCH_SLOT_BYTES <= FMBUF_URAM_BYTES,
              "Level3 LS_C1 scratch must fit in the URAM prefix");
static_assert(FMBUF_L3B0_SCRATCH_BASE + FMBUF_L3B0_SCRATCH_SLOT_BYTES <= FMBUF_URAM_BYTES,
              "Level3B0 LS_C1 scratch must fit in the URAM prefix");
static_assert(FMBUF_L30_BLOCK5_BASE + 32 * 64 * 128 <= FMBUF_L20_BASE,
              "Level3 BLOCK5 workspace must end before the residual slot");
static_assert(FMBUF_POOL1_BASE == FMBUF_URAM_BYTES,
              "Pool storage must start at the BRAM-only boundary");
static_assert(FMBUF_POOL_TMP_BASE + BRAM_SCR0_BYTES <= FMBUF_BYTES,
              "Pool scratch must fit in FMBUF");
static_assert(FMBUF_POOL2_ALIAS_BASE + FMBUF_POOL2_ALIAS_BYTES <= FMBUF_BYTES,
              "Pool2 alias must fit in FMBUF");

}  // namespace esp_int8
