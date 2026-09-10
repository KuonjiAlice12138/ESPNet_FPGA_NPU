#include "../include/npu_config.hpp"
#include "../include/npu_ctrl.hpp"
#include "../include/npu_schedule.hpp"
#include "../include/npu_types.hpp"

#if !defined(__SYNTHESIS__) && defined(ESP_INT8_CSIM_DUMP_U40_ACT)
#include <cstdint>
#include <cstdio>
#include <fstream>
#endif

namespace esp_int8 {

bool on_chip_memory_read_packed_tile(const tensor_desc_t& desc,
                                     i32_t h,
                                     i32_t w,
                                     u16_t c_begin,
                                     u8_t valid_c,
                                     act_vec_t& packed);

bool on_chip_memory_read_aligned_full_tile(const tensor_desc_t& desc,
                                           i32_t h,
                                           i32_t w,
                                           u16_t c_begin,
                                           act_vec_t& packed);

bool on_chip_memory_prepare_row_base(const tensor_desc_t& desc,
                                     i32_t h,
                                     u32_t& row_base,
                                     bool& row_valid);

bool on_chip_memory_read_aligned_tensor_word(const tensor_desc_t& desc,
                                             u32_t byte_offset,
                                             act_vec_t& packed);

static act_vec_t read_tile_or_zero(const tensor_desc_t& desc,
                                   i32_t h,
                                   i32_t w,
                                   u16_t c_begin,
                                   u8_t valid_c) {
#pragma HLS INLINE
    act_vec_t packed = 0;
    const bool ok = on_chip_memory_read_packed_tile(desc, h, w, c_begin, valid_c, packed);
    return ok ? packed : act_vec_t(0);
}

static act_vec_t read_aligned_or_zero(const tensor_desc_t& desc,
                                      i32_t h,
                                      i32_t w,
                                      u16_t c_begin) {
#pragma HLS INLINE
    act_vec_t packed = 0;
    const bool ok = on_chip_memory_read_aligned_full_tile(desc, h, w, c_begin, packed);
    return ok ? packed : act_vec_t(0);
}

static u16_t win_desc_phys_c(const tensor_desc_t& desc) {
#pragma HLS INLINE
    return (desc.reserved0.to_uint() == 0U) ? desc.c : desc.reserved0;
}

static bool win_can_read_aligned_1x1(const tensor_desc_t& desc, u16_t c_begin) {
#pragma HLS INLINE
    const unsigned phys_c = win_desc_phys_c(desc).to_uint();
    const unsigned start_c = desc.reserved1.to_uint() + c_begin.to_uint();
    const unsigned base_start = desc.base_offset.to_uint() + start_c;
    return phys_c >= static_cast<unsigned>(AXI_WORD_BYTES) &&
           ((phys_c & static_cast<unsigned>(AXI_WORD_BYTES - 1)) == 0U) &&
           c_begin.to_uint() + static_cast<unsigned>(AXI_WORD_BYTES) <= desc.c.to_uint() &&
           ((base_start & static_cast<unsigned>(AXI_WORD_BYTES - 1)) == 0U);
}

struct narrow_3x3_window_t {
    act_vec_t spatial[9];
};

// One physical owner serves two mutually exclusive compiled modes:
// - legacy narrow schedules use [0, 64) as the horizontal column ring;
// - row-reuse schedules use [0, 96) as packed physical source-row words.
// Four 64-bit stripes preserve one 256-bit read/write per cycle while allowing
// a row-reuse pixel to read up to three adjacent stripes from single-port RAMs.
constexpr int WINGEN_NARROW_WORD_LANES = AXI_WORD_BYTES / 8;
static u64_t
    s_narrow_row_bank0[WINGEN_NARROW_WORD_LANES][WINGEN_NARROW_ROW_WORDS];
static u64_t
    s_narrow_row_bank1[WINGEN_NARROW_WORD_LANES][WINGEN_NARROW_ROW_WORDS];
static u64_t
    s_narrow_row_bank2[WINGEN_NARROW_WORD_LANES][WINGEN_NARROW_ROW_WORDS];

static u32_t s_narrow_reuse_row_tag[3];
static bool s_narrow_reuse_row_valid[3];
static bool s_narrow_reuse_identity_valid = false;
static u8_t s_narrow_reuse_bank_id = 0;
static u32_t s_narrow_reuse_base_offset = 0;
static u16_t s_narrow_reuse_width = 0;
static u16_t s_narrow_reuse_phys_c = 0;
static u16_t s_narrow_reuse_c_offset = 0;
static u8_t s_narrow_reuse_mode = 0;
static u8_t s_narrow_reuse_window_mode = 0;
static u16_t s_narrow_reuse_last_out_row = 0;

struct wide_3x3_cache_t {
    act_vec_t data[3][WINGEN_WIDE_CACHE_COL_SLOTS][MAX_3X3_CACHE_CHUNKS];
};

static wide_3x3_cache_t s_wide_cache;

struct packed_word_cursor_t {
    act_vec_t word0;
    act_vec_t word1;
    u32_t tag0;
    u32_t tag1;
    bool valid0;
    bool valid1;
};

static packed_word_cursor_t s_packed_word_cursor[3];

struct window_column_meta_t {
    u8_t slot;
    u8_t update;
};

struct window_load_word_t {
    act_vec_t word;
    u8_t slot;
    u8_t row;
    u8_t chunk;
};

struct narrow_paired_issue_meta_t {
    u8_t slot00;
    u8_t slot01;
    u8_t slot02;
    u8_t slot10;
    u8_t slot11;
    u8_t slot12;
    u8_t update_cols;
    u8_t phase0_update_cols;
    u8_t has_second_window;
};

struct window_row_cfg_t {
    tensor_desc_t src_desc;
    u32_t row_base0;
    u32_t row_base1;
    u32_t row_base2;
    u8_t row_valid_mask;
    u8_t mode;
    u8_t stride;
    u8_t dilation;
    u8_t padding;
    u8_t cache_chunks;
    u8_t cache_col_slots;
    u8_t flags;
    u8_t loader_class;
    u8_t loader_request_cols;
    u8_t loader_warmup_issues;
    u8_t loader_warmup_new_cols;
    u8_t loader_steady_new_cols;
    u8_t loader_words_per_col;
    u8_t loader_warmup_mask;
    u8_t loader_steady_mask;
    u8_t loader_warmup_run_count;
    u8_t loader_steady_run_count;
    u8_t loader_warmup_phase0_cols;
    u8_t loader_steady_phase0_cols;
    u16_t loader_warmup_run0;
    u16_t loader_warmup_run1;
    u16_t loader_warmup_run2;
    u16_t loader_steady_run0;
    u16_t loader_steady_run1;
    u16_t loader_steady_run2;
    u8_t row_reuse_mode;
    u8_t packed_words_per_source_row;
    u16_t out_row;
    u16_t in_c;
    u16_t out_w;
    u16_t k_tiles;
};

struct window_assembler_cfg_t {
    u8_t mode;
    u8_t loader_class;
    u8_t flags;
    u16_t out_w;
    u8_t loader_request_cols;
    u8_t loader_words_per_col;
};

static u32_t window_cfg_row_base(const window_row_cfg_t& cfg, int row) {
#pragma HLS INLINE
    if (row == 0) {
        return cfg.row_base0;
    }
    if (row == 1) {
        return cfg.row_base1;
    }
    return cfg.row_base2;
}

static bool window_cfg_row_valid(const window_row_cfg_t& cfg, int row) {
#pragma HLS INLINE
    return ((cfg.row_valid_mask.to_uint() >> row) & 1U) != 0U;
}

static int window_issue_count(const window_row_cfg_t& cfg) {
#pragma HLS INLINE
    const int out_w = static_cast<int>(cfg.out_w.to_uint());
    const bool paired =
        (cfg.flags.to_uint() &
         static_cast<unsigned>(WINDOW_SCHED_FLAG_PIXEL_PARALLEL_2)) != 0U;
    return paired ? ((out_w + 1) / 2) : out_w;
}

static i32_t window_request_column(const window_row_cfg_t& cfg,
                                   int issue,
                                   int request) {
#pragma HLS INLINE
    const bool paired =
        (cfg.flags.to_uint() &
         static_cast<unsigned>(WINDOW_SCHED_FLAG_PIXEL_PARALLEL_2)) != 0U;
    const int phase = request >= 3 ? 1 : 0;
    const int kw = phase != 0 ? request - 3 : request;
    const int pixel0 = paired ? issue * 2 : issue;
    const int pixel = pixel0 + phase;
    return static_cast<i32_t>(
        pixel * static_cast<int>(cfg.stride.to_uint()) -
        static_cast<int>(cfg.padding.to_uint()) +
        kw * static_cast<int>(cfg.dilation.to_uint()));
}

static void configure_window_row(const tensor_desc_t& src_desc,
                                 const window_sched_desc_t& sched,
                                 u16_t out_row,
                                 hls::stream<window_row_cfg_t>& loader_cfg_stream,
                                 hls::stream<window_assembler_cfg_t>& assembler_cfg_stream) {
#pragma HLS INLINE off
    window_row_cfg_t cfg;
    cfg.src_desc = src_desc;
    cfg.mode = sched.mode;
    cfg.stride = sched.stride;
    cfg.dilation = sched.dilation;
    cfg.padding = sched.padding;
    cfg.cache_chunks = sched.cache_chunks;
    cfg.cache_col_slots = sched.cache_col_slots;
    cfg.flags = sched.flags;
    cfg.loader_class = sched.loader_class;
    cfg.loader_request_cols = sched.loader_request_cols;
    cfg.loader_warmup_issues = sched.loader_warmup_issues;
    cfg.loader_warmup_new_cols = sched.loader_warmup_new_cols;
    cfg.loader_steady_new_cols = sched.loader_steady_new_cols;
    cfg.loader_words_per_col = sched.loader_words_per_col;
    cfg.loader_warmup_mask = sched.loader_warmup_mask;
    cfg.loader_steady_mask = sched.loader_steady_mask;
    cfg.loader_warmup_run_count =
        sched.reserved[WINDOW_LOADER_WARMUP_COUNT_WORD];
    cfg.loader_steady_run_count =
        sched.reserved[WINDOW_LOADER_STEADY_COUNT_WORD];
    cfg.loader_warmup_phase0_cols =
        sched.reserved[WINDOW_LOADER_PHASE_SPLIT_WORD].range(7, 0);
    cfg.loader_steady_phase0_cols =
        sched.reserved[WINDOW_LOADER_PHASE_SPLIT_WORD].range(15, 8);
    cfg.loader_warmup_run0 =
        sched.reserved[WINDOW_LOADER_WARMUP_RUN_WORD + 0];
    cfg.loader_warmup_run1 =
        sched.reserved[WINDOW_LOADER_WARMUP_RUN_WORD + 1];
    cfg.loader_warmup_run2 =
        sched.reserved[WINDOW_LOADER_WARMUP_RUN_WORD + 2];
    cfg.loader_steady_run0 =
        sched.reserved[WINDOW_LOADER_STEADY_RUN_WORD + 0];
    cfg.loader_steady_run1 =
        sched.reserved[WINDOW_LOADER_STEADY_RUN_WORD + 1];
    cfg.loader_steady_run2 =
        sched.reserved[WINDOW_LOADER_STEADY_RUN_WORD + 2];
    const unsigned row_reuse_word =
        sched.reserved[WINDOW_LOADER_ROW_REUSE_WORD].to_uint();
    cfg.row_reuse_mode = static_cast<u8_t>(
        row_reuse_word & WINDOW_ROW_REUSE_MODE_MASK);
    cfg.packed_words_per_source_row = static_cast<u8_t>(
        (row_reuse_word >> WINDOW_ROW_REUSE_WORDS_SHIFT) &
        WINDOW_ROW_REUSE_WORDS_MASK);
    cfg.out_row = out_row;
    cfg.in_c = sched.in_c;
    cfg.out_w = sched.out_w;
    cfg.k_tiles = sched.k_tiles;

    const i32_t base_h =
        static_cast<i32_t>(
            out_row.to_uint() * static_cast<unsigned>(sched.stride.to_uint())) -
        static_cast<i32_t>(sched.padding.to_uint());
    u32_t row_base[3];
    bool row_valid[3];
#pragma HLS ARRAY_PARTITION variable=row_base complete dim=1
#pragma HLS ARRAY_PARTITION variable=row_valid complete dim=1
    for (int kh = 0; kh < 3; ++kh) {
#pragma HLS UNROLL
        on_chip_memory_prepare_row_base(
            src_desc,
            base_h + static_cast<i32_t>(
                         kh * static_cast<int>(sched.dilation.to_uint())),
            row_base[kh],
            row_valid[kh]);
    }
    cfg.row_base0 = row_base[0];
    cfg.row_base1 = row_base[1];
    cfg.row_base2 = row_base[2];
    cfg.row_valid_mask = 0;
    for (int kh = 0; kh < 3; ++kh) {
#pragma HLS UNROLL
        cfg.row_valid_mask[kh] = row_valid[kh] ? 1 : 0;
    }

    loader_cfg_stream.write(cfg);
    window_assembler_cfg_t assembler_cfg;
    assembler_cfg.mode = cfg.mode;
    assembler_cfg.loader_class = cfg.loader_class;
    assembler_cfg.flags = cfg.flags;
    assembler_cfg.out_w = cfg.out_w;
    assembler_cfg.loader_request_cols = cfg.loader_request_cols;
    assembler_cfg.loader_words_per_col = cfg.loader_words_per_col;
    assembler_cfg_stream.write(assembler_cfg);
}

static act_vec_t win_low_byte_mask(unsigned byte_count) {
#pragma HLS INLINE
    if (byte_count == 0U) {
        return 0;
    }
    if (byte_count >= static_cast<unsigned>(AXI_WORD_BYTES)) {
        return ~act_vec_t(0);
    }
    return static_cast<act_vec_t>(
        (static_cast<act_vec_t>(1) << (byte_count * 8U)) - 1);
}

static void reset_packed_word_cursors() {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=s_packed_word_cursor complete dim=1
    for (int kh = 0; kh < 3; ++kh) {
#pragma HLS UNROLL
        s_packed_word_cursor[kh].valid0 = false;
        s_packed_word_cursor[kh].valid1 = false;
    }
}

template <int LANE>
static u64_t narrow_row_bank_lane_read(unsigned bank, unsigned word_index) {
#pragma HLS INLINE
    if (bank == 0U) {
        return s_narrow_row_bank0[LANE][word_index];
    }
    if (bank == 1U) {
        return s_narrow_row_bank1[LANE][word_index];
    }
    return s_narrow_row_bank2[LANE][word_index];
}

template <int LANE>
static void narrow_row_bank_lane_write(unsigned bank,
                                       unsigned word_index,
                                       const u64_t& lane_word) {
#pragma HLS INLINE
    if (bank == 0U) {
        s_narrow_row_bank0[LANE][word_index] = lane_word;
    } else if (bank == 1U) {
        s_narrow_row_bank1[LANE][word_index] = lane_word;
    } else {
        s_narrow_row_bank2[LANE][word_index] = lane_word;
    }
}

static act_vec_t narrow_row_bank_read(unsigned bank, unsigned word_index) {
#pragma HLS INLINE
    act_vec_t word = 0;
    word.range(63, 0) = narrow_row_bank_lane_read<0>(bank, word_index);
    word.range(127, 64) = narrow_row_bank_lane_read<1>(bank, word_index);
    word.range(191, 128) = narrow_row_bank_lane_read<2>(bank, word_index);
    word.range(255, 192) = narrow_row_bank_lane_read<3>(bank, word_index);
    return word;
}

static void narrow_row_bank_write(unsigned bank,
                                  unsigned word_index,
                                  const act_vec_t& word) {
#pragma HLS INLINE
    narrow_row_bank_lane_write<0>(
        bank, word_index, static_cast<u64_t>(word.range(63, 0)));
    narrow_row_bank_lane_write<1>(
        bank, word_index, static_cast<u64_t>(word.range(127, 64)));
    narrow_row_bank_lane_write<2>(
        bank, word_index, static_cast<u64_t>(word.range(191, 128)));
    narrow_row_bank_lane_write<3>(
        bank, word_index, static_cast<u64_t>(word.range(255, 192)));
}

static bool narrow_reuse_identity_matches(const window_row_cfg_t& cfg) {
#pragma HLS INLINE
    const unsigned out_row = cfg.out_row.to_uint();
    const bool contiguous =
        out_row == s_narrow_reuse_last_out_row.to_uint() + 1U;
    return s_narrow_reuse_identity_valid &&
           contiguous &&
           s_narrow_reuse_bank_id == cfg.src_desc.bank_id &&
           s_narrow_reuse_base_offset == cfg.src_desc.base_offset &&
           s_narrow_reuse_width == cfg.src_desc.w &&
           s_narrow_reuse_phys_c == win_desc_phys_c(cfg.src_desc) &&
           s_narrow_reuse_c_offset == cfg.src_desc.reserved1 &&
           s_narrow_reuse_mode == cfg.row_reuse_mode &&
           s_narrow_reuse_window_mode == cfg.mode;
}

static void narrow_reuse_update_identity(const window_row_cfg_t& cfg) {
#pragma HLS INLINE
    s_narrow_reuse_identity_valid = true;
    s_narrow_reuse_bank_id = cfg.src_desc.bank_id;
    s_narrow_reuse_base_offset = cfg.src_desc.base_offset;
    s_narrow_reuse_width = cfg.src_desc.w;
    s_narrow_reuse_phys_c = win_desc_phys_c(cfg.src_desc);
    s_narrow_reuse_c_offset = cfg.src_desc.reserved1;
    s_narrow_reuse_mode = cfg.row_reuse_mode;
    s_narrow_reuse_window_mode = cfg.mode;
    s_narrow_reuse_last_out_row = cfg.out_row;
}

static void narrow_reuse_load_source_row(const window_row_cfg_t& cfg,
                                         unsigned bank,
                                         u32_t row_base) {
#pragma HLS INLINE off
    const int packed_words =
        static_cast<int>(cfg.packed_words_per_source_row.to_uint());
    for (int word_index = 0;
         word_index < WINGEN_NARROW_ROW_WORDS;
         ++word_index) {
#pragma HLS PIPELINE II=1
        if (word_index >= packed_words) {
            break;
        }
        act_vec_t word = 0;
        on_chip_memory_read_aligned_tensor_word(
            cfg.src_desc,
            row_base +
                static_cast<u32_t>(word_index * AXI_WORD_BYTES),
            word);
        narrow_row_bank_write(
            bank, static_cast<unsigned>(word_index), word);
    }
}

static void narrow_reuse_prepare_rows(const window_row_cfg_t& cfg,
                                      u8_t logical_bank[3]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=logical_bank complete dim=1
    bool used[3];
#pragma HLS ARRAY_PARTITION variable=used complete dim=1
    if (!narrow_reuse_identity_matches(cfg)) {
        for (int bank = 0; bank < 3; ++bank) {
#pragma HLS UNROLL
            s_narrow_reuse_row_valid[bank] = false;
        }
    }
    for (int bank = 0; bank < 3; ++bank) {
#pragma HLS UNROLL
        used[bank] = false;
    }
    for (int kh = 0; kh < 3; ++kh) {
#pragma HLS UNROLL
        logical_bank[kh] = static_cast<u8_t>(3);
    }

    for (int kh = 0; kh < 3; ++kh) {
#pragma HLS PIPELINE off
        if (!window_cfg_row_valid(cfg, kh)) {
            continue;
        }
        const u32_t required_tag = window_cfg_row_base(cfg, kh);
        for (int bank = 0; bank < 3; ++bank) {
#pragma HLS UNROLL
            if (!used[bank] &&
                s_narrow_reuse_row_valid[bank] &&
                s_narrow_reuse_row_tag[bank] == required_tag) {
                logical_bank[kh] = static_cast<u8_t>(bank);
                used[bank] = true;
            }
        }
    }

    for (int kh = 0; kh < 3; ++kh) {
#pragma HLS PIPELINE off
        if (!window_cfg_row_valid(cfg, kh) ||
            logical_bank[kh].to_uint() < 3U) {
            continue;
        }
        unsigned selected = 3U;
        for (int bank = 0; bank < 3; ++bank) {
#pragma HLS UNROLL
            if (!used[bank] && selected == 3U) {
                selected = static_cast<unsigned>(bank);
            }
        }
        if (selected >= 3U) {
            continue;
        }
        used[selected] = true;
        logical_bank[kh] = static_cast<u8_t>(selected);
        const u32_t required_tag = window_cfg_row_base(cfg, kh);
        narrow_reuse_load_source_row(cfg, selected, required_tag);
        s_narrow_reuse_row_tag[selected] = required_tag;
        s_narrow_reuse_row_valid[selected] = true;
    }
    narrow_reuse_update_identity(cfg);
}

static bool read_packed_tile_from_reuse_row(
    const window_row_cfg_t& cfg,
    unsigned bank,
    bool row_valid,
    i32_t input_col,
    act_vec_t& packed) {
#pragma HLS INLINE
    packed = 0;
    if (!row_valid ||
        bank >= 3U ||
        input_col < 0 ||
        input_col >= static_cast<i32_t>(cfg.src_desc.w)) {
        return true;
    }
    const unsigned lane_count = cfg.in_c.to_uint();
    const unsigned byte_offset =
        static_cast<unsigned>(input_col.to_int()) *
        win_desc_phys_c(cfg.src_desc).to_uint();
    const unsigned packed_words =
        cfg.packed_words_per_source_row.to_uint();
    const unsigned packed_bytes =
        packed_words * static_cast<unsigned>(AXI_WORD_BYTES);
    if (lane_count == 0U ||
        byte_offset + lane_count > packed_bytes) {
        return false;
    }

    const unsigned word_index =
        byte_offset / static_cast<unsigned>(AXI_WORD_BYTES);
    const unsigned stripe =
        (byte_offset / 8U) &
        static_cast<unsigned>(WINGEN_NARROW_WORD_LANES - 1);
    const unsigned byte_in_stripe = byte_offset & 7U;
    const unsigned next_word_index =
        (word_index + 1U < packed_words) ? word_index + 1U : word_index;

    // Any C3/C12 pixel occupies at most three consecutive 64-bit stripes.
    // Read each physical stripe once, selecting the next word only after wrap.
    const u64_t lane0 = narrow_row_bank_lane_read<0>(
        bank, (stripe > 0U) ? next_word_index : word_index);
    const u64_t lane1 = narrow_row_bank_lane_read<1>(
        bank, (stripe > 1U) ? next_word_index : word_index);
    const u64_t lane2 = narrow_row_bank_lane_read<2>(
        bank, (stripe > 2U) ? next_word_index : word_index);
    const u64_t lane3 =
        narrow_row_bank_lane_read<3>(bank, word_index);

    ap_uint<256> rotated = 0;
    switch (stripe) {
        case 0:
            rotated.range(63, 0) = lane0;
            rotated.range(127, 64) = lane1;
            rotated.range(191, 128) = lane2;
            rotated.range(255, 192) = lane3;
            break;
        case 1:
            rotated.range(63, 0) = lane1;
            rotated.range(127, 64) = lane2;
            rotated.range(191, 128) = lane3;
            rotated.range(255, 192) = lane0;
            break;
        case 2:
            rotated.range(63, 0) = lane2;
            rotated.range(127, 64) = lane3;
            rotated.range(191, 128) = lane0;
            rotated.range(255, 192) = lane1;
            break;
        default:
            rotated.range(63, 0) = lane3;
            rotated.range(127, 64) = lane0;
            rotated.range(191, 128) = lane1;
            rotated.range(255, 192) = lane2;
            break;
    }
    packed = static_cast<act_vec_t>(rotated >> (byte_in_stripe * 8U));
    packed &= win_low_byte_mask(lane_count);
    return true;
}

static bool read_aligned_word_cached(const tensor_desc_t& src_desc,
                                     packed_word_cursor_t& cursor,
                                     u32_t word_offset,
                                     act_vec_t& word) {
#pragma HLS INLINE
    if (cursor.valid0 && cursor.tag0 == word_offset) {
        word = cursor.word0;
        return true;
    }
    if (cursor.valid1 && cursor.tag1 == word_offset) {
        word = cursor.word1;
        return true;
    }

    act_vec_t loaded = 0;
    if (!on_chip_memory_read_aligned_tensor_word(src_desc, word_offset, loaded)) {
        word = 0;
        return false;
    }
    cursor.word1 = cursor.word0;
    cursor.tag1 = cursor.tag0;
    cursor.valid1 = cursor.valid0;
    cursor.word0 = loaded;
    cursor.tag0 = word_offset;
    cursor.valid0 = true;
    word = loaded;
    return true;
}

static bool read_packed_tile_from_row_cached(const tensor_desc_t& src_desc,
                                             packed_word_cursor_t& cursor,
                                             u32_t row_base,
                                             bool row_valid,
                                             i32_t input_col,
                                             u16_t c_begin,
                                             u8_t valid_c,
                                             act_vec_t& packed) {
#pragma HLS INLINE
    packed = 0;
    unsigned lane_count = valid_c.to_uint();
    if (lane_count == 0U || lane_count > static_cast<unsigned>(TK)) {
        lane_count = static_cast<unsigned>(TK);
    }
    if (!row_valid || input_col < 0 || input_col >= static_cast<i32_t>(src_desc.w) ||
        c_begin >= src_desc.c) {
        return true;
    }

    const unsigned remaining =
        static_cast<unsigned>(src_desc.c.to_uint() - c_begin.to_uint());
    const unsigned read_count = lane_count < remaining ? lane_count : remaining;
    if (read_count == 0U) {
        return true;
    }

    const u32_t start_offset =
        row_base + static_cast<u32_t>(input_col) *
                       static_cast<u32_t>(win_desc_phys_c(src_desc)) +
        static_cast<u32_t>(c_begin);
    const u32_t word0_offset =
        start_offset & static_cast<u32_t>(~(AXI_WORD_BYTES - 1));
    const unsigned byte0 =
        start_offset.to_uint() & static_cast<unsigned>(AXI_WORD_BYTES - 1);

    act_vec_t word0 = 0;
    if (!read_aligned_word_cached(src_desc, cursor, word0_offset, word0)) {
        return false;
    }
    const unsigned shift_bits = byte0 * 8U;
    packed = static_cast<act_vec_t>(word0 >> shift_bits);
    if (byte0 + read_count <= static_cast<unsigned>(AXI_WORD_BYTES)) {
        // The first aligned word contains the complete compact tile.
    } else {
        act_vec_t word1 = 0;
        if (!read_aligned_word_cached(
                src_desc, cursor, word0_offset + AXI_WORD_BYTES, word1)) {
            return false;
        }
        const unsigned carry_bits =
            (static_cast<unsigned>(AXI_WORD_BYTES) - byte0) * 8U;
        packed |= static_cast<act_vec_t>(word1 << carry_bits);
    }
    if (read_count < static_cast<unsigned>(AXI_WORD_BYTES)) {
        packed &= win_low_byte_mask(read_count);
    }
    return true;
}

static act_vec_t read_staged_cache(const narrow_3x3_window_t& window,
                                   int kh,
                                   int kw,
                                   int,
                                   int) {
#pragma HLS INLINE
    return window.spatial[kh * 3 + kw];
}

static act_vec_t read_staged_cache(const wide_3x3_cache_t& cache,
                                   int kh,
                                   int,
                                   int slot,
                                   int chunk) {
#pragma HLS INLINE
    return cache.data[kh][slot][chunk];
}

static void window_loader_emit_column(
    const window_row_cfg_t& cfg,
    i32_t input_col,
    u8_t slot,
    hls::stream<window_load_word_t>& load_word_stream) {
#pragma HLS INLINE off
    for (int kh = 0; kh < 3; ++kh) {
#pragma HLS PIPELINE off
        packed_word_cursor_t& cursor = s_packed_word_cursor[kh];
        for (int chunk = 0; chunk < MAX_3X3_CACHE_CHUNKS; ++chunk) {
#pragma HLS PIPELINE off
            if (chunk >= static_cast<int>(cfg.cache_chunks.to_uint())) {
                break;
            }
            act_vec_t word = 0;
            const unsigned c_begin_u = static_cast<unsigned>(chunk * TK);
            unsigned valid = cfg.in_c.to_uint() - c_begin_u;
            if (valid > static_cast<unsigned>(TK)) {
                valid = static_cast<unsigned>(TK);
            }
            read_packed_tile_from_row_cached(cfg.src_desc,
                                             cursor,
                                             window_cfg_row_base(cfg, kh),
                                             window_cfg_row_valid(cfg, kh),
                                             input_col,
                                             static_cast<u16_t>(c_begin_u),
                                             static_cast<u8_t>(valid),
                                             word);
            window_load_word_t token;
            token.word = word;
            token.slot = slot;
            token.row = static_cast<u8_t>(kh);
            token.chunk = static_cast<u8_t>(chunk);
            load_word_stream.write(token);
        }
    }
}

static bool window_cfg_is_narrow_paired(const window_row_cfg_t& cfg) {
#pragma HLS INLINE
    return cfg.loader_class.to_uint() ==
               static_cast<unsigned>(WIN_LOADER_3X3_NARROW) &&
           (cfg.flags.to_uint() &
            static_cast<unsigned>(WINDOW_SCHED_FLAG_PIXEL_PARALLEL_2)) != 0U;
}

static bool window_cfg_uses_row_reuse(const window_row_cfg_t& cfg) {
#pragma HLS INLINE
    return cfg.row_reuse_mode.to_uint() !=
           static_cast<unsigned>(WINDOW_ROW_REUSE_NONE);
}

static u16_t window_cfg_loader_run_word(const window_row_cfg_t& cfg,
                                        bool warmup,
                                        int run_index) {
#pragma HLS INLINE
    if (warmup) {
        if (run_index == 0) {
            return cfg.loader_warmup_run0;
        }
        if (run_index == 1) {
            return cfg.loader_warmup_run1;
        }
        return cfg.loader_warmup_run2;
    }
    if (run_index == 0) {
        return cfg.loader_steady_run0;
    }
    if (run_index == 1) {
        return cfg.loader_steady_run1;
    }
    return cfg.loader_steady_run2;
}

static u8_t window_cache_slot(i32_t input_col, unsigned slot_mask) {
#pragma HLS INLINE
    return static_cast<u8_t>(
        static_cast<unsigned>(input_col.to_int()) & slot_mask);
}

static narrow_paired_issue_meta_t make_narrow_paired_issue_meta(
    const window_row_cfg_t& cfg,
    int issue,
    bool warmup,
    unsigned slot_mask) {
#pragma HLS INLINE
    const int stride = static_cast<int>(cfg.stride.to_uint());
    const int dilation = static_cast<int>(cfg.dilation.to_uint());
    const int padding = static_cast<int>(cfg.padding.to_uint());
    const int pixel0 = issue * 2;
    const i32_t base0 =
        static_cast<i32_t>(pixel0 * stride - padding);
    const i32_t base1 =
        static_cast<i32_t>((pixel0 + 1) * stride - padding);

    narrow_paired_issue_meta_t meta;
    meta.slot00 = window_cache_slot(base0, slot_mask);
    meta.slot01 =
        window_cache_slot(base0 + static_cast<i32_t>(dilation), slot_mask);
    meta.slot02 =
        window_cache_slot(base0 + static_cast<i32_t>(2 * dilation), slot_mask);
    meta.slot10 = window_cache_slot(base1, slot_mask);
    meta.slot11 =
        window_cache_slot(base1 + static_cast<i32_t>(dilation), slot_mask);
    meta.slot12 =
        window_cache_slot(base1 + static_cast<i32_t>(2 * dilation), slot_mask);
    meta.update_cols =
        warmup ? cfg.loader_warmup_new_cols
               : cfg.loader_steady_new_cols;
    meta.phase0_update_cols =
        warmup ? cfg.loader_warmup_phase0_cols
               : cfg.loader_steady_phase0_cols;
    const bool odd_tail =
        (cfg.flags.to_uint() &
         static_cast<unsigned>(WINDOW_SCHED_FLAG_ODD_TAIL)) != 0U;
    const bool tail_issue =
        odd_tail && issue == window_issue_count(cfg) - 1;
    meta.has_second_window =
        tail_issue ? static_cast<u8_t>(0)
                   : static_cast<u8_t>(1);
    return meta;
}

static void window_loader_update_narrow_run(
    const window_row_cfg_t& cfg,
    i32_t start_col,
    int column_count,
    unsigned slot_mask) {
#pragma HLS INLINE
    unsigned valid = cfg.in_c.to_uint();
    if (valid > static_cast<unsigned>(TK)) {
        valid = static_cast<unsigned>(TK);
    }
    for (int kh = 0; kh < 3; ++kh) {
#pragma HLS PIPELINE off
        packed_word_cursor_t& cursor = s_packed_word_cursor[kh];
        const u32_t row_base = window_cfg_row_base(cfg, kh);
        const bool row_valid = window_cfg_row_valid(cfg, kh);
        for (int offset = 0; offset < 6; ++offset) {
#pragma HLS PIPELINE off
            if (offset >= column_count) {
                break;
            }
            const i32_t input_col =
                start_col + static_cast<i32_t>(offset);
            act_vec_t word = 0;
            read_packed_tile_from_row_cached(
                cfg.src_desc,
                cursor,
                row_base,
                row_valid,
                input_col,
                static_cast<u16_t>(0),
                static_cast<u8_t>(valid),
                word);
            narrow_row_bank_write(
                static_cast<unsigned>(kh),
                window_cache_slot(input_col, slot_mask).to_uint(),
                word);
        }
    }
}

static void window_loader_emit_narrow_cached_window(
    const u8_t slot_for_kw[3],
    hls::stream<window_load_word_t>& load_word_stream) {
#pragma HLS INLINE
    for (int kh = 0; kh < 3; ++kh) {
#pragma HLS PIPELINE off
        for (int kw = 0; kw < 3; ++kw) {
#pragma HLS PIPELINE II=1
            window_load_word_t token;
            token.word = narrow_row_bank_read(
                static_cast<unsigned>(kh),
                slot_for_kw[kw].to_uint());
            token.slot = 0;
            token.row = static_cast<u8_t>(kh);
            token.chunk = static_cast<u8_t>(kw);
            load_word_stream.write(token);
        }
    }
}

static void window_loader_emit_narrow_reuse_window(
    const window_row_cfg_t& cfg,
    const u8_t logical_bank[3],
    int pixel,
    hls::stream<window_load_word_t>& load_word_stream) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=logical_bank complete dim=1
    const int stride = static_cast<int>(cfg.stride.to_uint());
    const int dilation = static_cast<int>(cfg.dilation.to_uint());
    const int padding = static_cast<int>(cfg.padding.to_uint());
    for (int kh = 0; kh < 3; ++kh) {
#pragma HLS PIPELINE off
        for (int kw = 0; kw < 3; ++kw) {
#pragma HLS PIPELINE II=1
            const i32_t input_col = static_cast<i32_t>(
                pixel * stride - padding + kw * dilation);
            act_vec_t word = 0;
            read_packed_tile_from_reuse_row(
                cfg,
                logical_bank[kh].to_uint(),
                window_cfg_row_valid(cfg, kh),
                input_col,
                word);
            window_load_word_t token;
            token.word = word;
            token.slot = 0;
            token.row = static_cast<u8_t>(kh);
            token.chunk = static_cast<u8_t>(kw);
            load_word_stream.write(token);
        }
    }
}

static void window_row_loader_narrow_paired(
    const window_row_cfg_t& cfg,
    hls::stream<narrow_paired_issue_meta_t>& narrow_issue_stream,
    hls::stream<window_load_word_t>& load_word_stream) {
#pragma HLS INLINE off
    const int issue_count = window_issue_count(cfg);
    const unsigned slot_mask =
        static_cast<unsigned>(cfg.cache_col_slots.to_uint()) - 1U;
    const int stride = static_cast<int>(cfg.stride.to_uint());
    const int padding = static_cast<int>(cfg.padding.to_uint());
    for (int issue = 0; issue < MAX_FM_W; ++issue) {
        if (issue >= issue_count) {
            break;
        }
        const bool warmup =
            issue < static_cast<int>(cfg.loader_warmup_issues.to_uint());
        const int run_count = static_cast<int>(
            warmup ? cfg.loader_warmup_run_count.to_uint()
                   : cfg.loader_steady_run_count.to_uint());
        const narrow_paired_issue_meta_t meta =
            make_narrow_paired_issue_meta(
                cfg, issue, warmup, slot_mask);
        narrow_issue_stream.write(meta);
        u8_t slot_for_kw0[3] = {
            meta.slot00, meta.slot01, meta.slot02};
        u8_t slot_for_kw1[3] = {
            meta.slot10, meta.slot11, meta.slot12};
#pragma HLS ARRAY_PARTITION variable=slot_for_kw0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=slot_for_kw1 complete dim=1

        const int base_column = issue * 2 * stride - padding;
        const int total_columns =
            static_cast<int>(meta.update_cols.to_uint());
        const int phase0_columns =
            static_cast<int>(meta.phase0_update_cols.to_uint());
        const int first_phase_end =
            phase0_columns != 0 ? phase0_columns : total_columns;
        for (int phase = 0; phase < 2; ++phase) {
#pragma HLS PIPELINE off
            const int phase_begin =
                phase == 0 ? 0 : first_phase_end;
            const int phase_end =
                phase == 0 ? first_phase_end : total_columns;
            int run_column_base = 0;
            if (phase_begin < phase_end) {
              for (int run_index = 0;
                   run_index < WINDOW_LOADER_RUN_MAX;
                   ++run_index) {
#pragma HLS PIPELINE off
                if (run_index >= run_count) {
                    break;
                }
                const u16_t run_word =
                    window_cfg_loader_run_word(
                        cfg, warmup, run_index);
                int start_delta =
                    static_cast<int>(run_word.to_uint() & 0xFFU);
                if ((start_delta & 0x80) != 0) {
                    start_delta -= 0x100;
                }
                const int run_columns = static_cast<int>(
                    (run_word.to_uint() >> 8) & 0xFFU);
                int local_begin = phase_begin - run_column_base;
                if (local_begin < 0) {
                    local_begin = 0;
                }
                int local_end = phase_end - run_column_base;
                if (local_end > run_columns) {
                    local_end = run_columns;
                }
                const int emit_columns = local_end - local_begin;
                if (emit_columns > 0) {
                    const i32_t start_col =
                        static_cast<i32_t>(
                            base_column + start_delta + local_begin);
                    window_loader_update_narrow_run(
                        cfg,
                        start_col,
                        emit_columns,
                        slot_mask);
                }
                run_column_base += run_columns;
              }
            }
            if (phase == 0) {
                window_loader_emit_narrow_cached_window(
                    slot_for_kw0, load_word_stream);
            } else if (meta.has_second_window.to_uint() != 0U) {
                window_loader_emit_narrow_cached_window(
                    slot_for_kw1, load_word_stream);
            }
        }
    }
}

static void window_row_loader_narrow_unpaired(
    const window_row_cfg_t& cfg,
    hls::stream<narrow_paired_issue_meta_t>& narrow_issue_stream,
    hls::stream<window_load_word_t>& load_word_stream) {
#pragma HLS INLINE off
    const int issue_count = window_issue_count(cfg);
    const unsigned slot_mask =
        static_cast<unsigned>(cfg.cache_col_slots.to_uint()) - 1U;
    for (int issue = 0; issue < MAX_FM_W; ++issue) {
        if (issue >= issue_count) {
            break;
        }
        const bool warmup =
            issue < static_cast<int>(cfg.loader_warmup_issues.to_uint());
        const unsigned update_mask =
            warmup ? cfg.loader_warmup_mask.to_uint()
                   : cfg.loader_steady_mask.to_uint();
        u8_t slot_for_kw[3];
#pragma HLS ARRAY_PARTITION variable=slot_for_kw complete dim=1
        for (int request = 0; request < 3; ++request) {
#pragma HLS PIPELINE off
            const i32_t input_col =
                window_request_column(cfg, issue, request);
            slot_for_kw[request] =
                window_cache_slot(input_col, slot_mask);
            if (((update_mask >> request) & 1U) != 0U) {
                window_loader_update_narrow_run(
                    cfg, input_col, 1, slot_mask);
            }
        }
        narrow_paired_issue_meta_t meta = narrow_paired_issue_meta_t();
        meta.has_second_window = 0;
        narrow_issue_stream.write(meta);
        window_loader_emit_narrow_cached_window(
            slot_for_kw, load_word_stream);
    }
}

static void window_row_loader_narrow_reuse(
    const window_row_cfg_t& cfg,
    hls::stream<narrow_paired_issue_meta_t>& narrow_issue_stream,
    hls::stream<window_load_word_t>& load_word_stream) {
#pragma HLS INLINE off
    u8_t logical_bank[3];
#pragma HLS ARRAY_PARTITION variable=logical_bank complete dim=1
    narrow_reuse_prepare_rows(cfg, logical_bank);
    const int issue_count = window_issue_count(cfg);
    const bool odd_tail =
        (cfg.flags.to_uint() &
         static_cast<unsigned>(WINDOW_SCHED_FLAG_ODD_TAIL)) != 0U;
    for (int issue = 0; issue < MAX_FM_W; ++issue) {
        if (issue >= issue_count) {
            break;
        }
        const bool tail_issue =
            odd_tail && issue == issue_count - 1;
        narrow_paired_issue_meta_t meta = narrow_paired_issue_meta_t();
        meta.has_second_window =
            tail_issue ? static_cast<u8_t>(0)
                       : static_cast<u8_t>(1);
        narrow_issue_stream.write(meta);
        window_loader_emit_narrow_reuse_window(
            cfg, logical_bank, issue * 2, load_word_stream);
        if (!tail_issue) {
            window_loader_emit_narrow_reuse_window(
                cfg, logical_bank, issue * 2 + 1, load_word_stream);
        }
    }
}

static void window_row_loader(
    hls::stream<window_row_cfg_t>& loader_cfg_stream,
    hls::stream<window_column_meta_t>& column_meta_stream,
    hls::stream<narrow_paired_issue_meta_t>& narrow_issue_stream,
    hls::stream<window_load_word_t>& load_word_stream) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=s_narrow_row_bank0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=s_narrow_row_bank1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=s_narrow_row_bank2 complete dim=1
#pragma HLS BIND_STORAGE variable=s_narrow_row_bank0 type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=s_narrow_row_bank1 type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=s_narrow_row_bank2 type=ram_1p impl=bram
#pragma HLS RESET variable=s_narrow_row_bank0 off
#pragma HLS RESET variable=s_narrow_row_bank1 off
#pragma HLS RESET variable=s_narrow_row_bank2 off
    const window_row_cfg_t cfg = loader_cfg_stream.read();
    reset_packed_word_cursors();
    if (cfg.loader_class.to_uint() ==
        static_cast<unsigned>(WIN_LOADER_3X3_NARROW)) {
        if (window_cfg_uses_row_reuse(cfg)) {
            window_row_loader_narrow_reuse(
                cfg, narrow_issue_stream, load_word_stream);
        } else {
            s_narrow_reuse_identity_valid = false;
            if (window_cfg_is_narrow_paired(cfg)) {
                window_row_loader_narrow_paired(
                    cfg, narrow_issue_stream, load_word_stream);
            } else {
                window_row_loader_narrow_unpaired(
                    cfg, narrow_issue_stream, load_word_stream);
            }
        }
        return;
    }

    const int issue_count = window_issue_count(cfg);
    const int request_cols =
        static_cast<int>(cfg.loader_request_cols.to_uint());
    const unsigned slot_mask =
        static_cast<unsigned>(cfg.cache_col_slots.to_uint()) - 1U;
    for (int issue = 0; issue < MAX_FM_W; ++issue) {
        if (issue >= issue_count) {
            break;
        }
        const bool warmup =
            issue < static_cast<int>(cfg.loader_warmup_issues.to_uint());
        const unsigned update_mask =
            warmup ? cfg.loader_warmup_mask.to_uint()
                   : cfg.loader_steady_mask.to_uint();
        for (int request = 0; request < 6; ++request) {
#pragma HLS PIPELINE off
            if (request >= request_cols) {
                break;
            }
            const i32_t input_col =
                window_request_column(cfg, issue, request);
            const unsigned slot =
                static_cast<unsigned>(input_col.to_int()) & slot_mask;
            const bool update = ((update_mask >> request) & 1U) != 0U;

            window_column_meta_t meta;
            meta.slot = static_cast<u8_t>(slot);
            meta.update = update ? static_cast<u8_t>(1)
                                 : static_cast<u8_t>(0);
            column_meta_stream.write(meta);
            if (update) {
                window_loader_emit_column(
                    cfg,
                    input_col,
                    static_cast<u8_t>(slot),
                    load_word_stream);
            }
        }
    }
}

template <int SPATIAL, int SRC_LANE, int DST_LANE, int COUNT, typename CacheT>
static void copy_staged_cache_segment(
    const CacheT& cache,
    const u8_t slot_for_kw[3],
    act_vec_t& word) {
#pragma HLS INLINE
    const int kh = SPATIAL / 3;
    const int kw = SPATIAL - kh * 3;
    const int slot = static_cast<int>(slot_for_kw[kw].to_uint());
    const act_vec_t src = read_staged_cache(cache, kh, kw, slot, 0);
    word.range(DST_LANE * 8 + COUNT * 8 - 1, DST_LANE * 8) =
        src.range(SRC_LANE * 8 + COUNT * 8 - 1, SRC_LANE * 8);
}

template <int SPATIAL, int CHUNK, int SRC_LANE, int DST_LANE, int COUNT, typename CacheT>
static void copy_staged_cache_chunk_segment(
    const CacheT& cache,
    const u8_t slot_for_kw[3],
    act_vec_t& word) {
#pragma HLS INLINE
    const int kh = SPATIAL / 3;
    const int kw = SPATIAL - kh * 3;
    const int slot = static_cast<int>(slot_for_kw[kw].to_uint());
    const act_vec_t src = read_staged_cache(cache, kh, kw, slot, CHUNK);
    word.range(DST_LANE * 8 + COUNT * 8 - 1, DST_LANE * 8) =
        src.range(SRC_LANE * 8 + COUNT * 8 - 1, SRC_LANE * 8);
}

template <typename CacheT>
static void build_staged_c3_word(
    const CacheT& cache,
    const u8_t slot_for_kw[3],
    int,
    act_vec_t& word) {
#pragma HLS INLINE
    copy_staged_cache_segment<0, 0, 0, 3>(cache, slot_for_kw, word);
    copy_staged_cache_segment<1, 0, 3, 3>(cache, slot_for_kw, word);
    copy_staged_cache_segment<2, 0, 6, 3>(cache, slot_for_kw, word);
    copy_staged_cache_segment<3, 0, 9, 3>(cache, slot_for_kw, word);
    copy_staged_cache_segment<4, 0, 12, 3>(cache, slot_for_kw, word);
    copy_staged_cache_segment<5, 0, 15, 3>(cache, slot_for_kw, word);
    copy_staged_cache_segment<6, 0, 18, 3>(cache, slot_for_kw, word);
    copy_staged_cache_segment<7, 0, 21, 3>(cache, slot_for_kw, word);
    copy_staged_cache_segment<8, 0, 24, 3>(cache, slot_for_kw, word);
}

template <typename CacheT>
static void build_staged_c12_word(
    const CacheT& cache,
    const u8_t slot_for_kw[3],
    int kt,
    act_vec_t& word) {
#pragma HLS INLINE
    switch (kt) {
        case 0:
            copy_staged_cache_segment<0, 0, 0, 12>(cache, slot_for_kw, word);
            copy_staged_cache_segment<1, 0, 12, 12>(cache, slot_for_kw, word);
            copy_staged_cache_segment<2, 0, 24, 8>(cache, slot_for_kw, word);
            break;
        case 1:
            copy_staged_cache_segment<2, 8, 0, 4>(cache, slot_for_kw, word);
            copy_staged_cache_segment<3, 0, 4, 12>(cache, slot_for_kw, word);
            copy_staged_cache_segment<4, 0, 16, 12>(cache, slot_for_kw, word);
            copy_staged_cache_segment<5, 0, 28, 4>(cache, slot_for_kw, word);
            break;
        case 2:
            copy_staged_cache_segment<5, 4, 0, 8>(cache, slot_for_kw, word);
            copy_staged_cache_segment<6, 0, 8, 12>(cache, slot_for_kw, word);
            copy_staged_cache_segment<7, 0, 20, 12>(cache, slot_for_kw, word);
            break;
        case 3:
            copy_staged_cache_segment<8, 0, 0, 12>(cache, slot_for_kw, word);
            break;
        default:
            break;
    }
}

template <typename CacheT>
static void build_staged_c19_word(
    const CacheT& cache,
    const u8_t slot_for_kw[3],
    int kt,
    act_vec_t& word) {
#pragma HLS INLINE
    switch (kt) {
        case 0:
            copy_staged_cache_segment<0, 0, 0, 19>(cache, slot_for_kw, word);
            copy_staged_cache_segment<1, 0, 19, 13>(cache, slot_for_kw, word);
            break;
        case 1:
            copy_staged_cache_segment<1, 13, 0, 6>(cache, slot_for_kw, word);
            copy_staged_cache_segment<2, 0, 6, 19>(cache, slot_for_kw, word);
            copy_staged_cache_segment<3, 0, 25, 7>(cache, slot_for_kw, word);
            break;
        case 2:
            copy_staged_cache_segment<3, 7, 0, 12>(cache, slot_for_kw, word);
            copy_staged_cache_segment<4, 0, 12, 19>(cache, slot_for_kw, word);
            copy_staged_cache_segment<5, 0, 31, 1>(cache, slot_for_kw, word);
            break;
        case 3:
            copy_staged_cache_segment<5, 1, 0, 18>(cache, slot_for_kw, word);
            copy_staged_cache_segment<6, 0, 18, 14>(cache, slot_for_kw, word);
            break;
        case 4:
            copy_staged_cache_segment<6, 14, 0, 5>(cache, slot_for_kw, word);
            copy_staged_cache_segment<7, 0, 5, 19>(cache, slot_for_kw, word);
            copy_staged_cache_segment<8, 0, 24, 8>(cache, slot_for_kw, word);
            break;
        case 5:
            copy_staged_cache_segment<8, 8, 0, 11>(cache, slot_for_kw, word);
            break;
        default:
            break;
    }
}

template <typename CacheT>
static void build_staged_c25_word(
    const CacheT& cache,
    const u8_t slot_for_kw[3],
    int kt,
    act_vec_t& word) {
#pragma HLS INLINE
    switch (kt) {
        case 0:
            copy_staged_cache_segment<0, 0, 0, 25>(cache, slot_for_kw, word);
            copy_staged_cache_segment<1, 0, 25, 7>(cache, slot_for_kw, word);
            break;
        case 1:
            copy_staged_cache_segment<1, 7, 0, 18>(cache, slot_for_kw, word);
            copy_staged_cache_segment<2, 0, 18, 14>(cache, slot_for_kw, word);
            break;
        case 2:
            copy_staged_cache_segment<2, 14, 0, 11>(cache, slot_for_kw, word);
            copy_staged_cache_segment<3, 0, 11, 21>(cache, slot_for_kw, word);
            break;
        case 3:
            copy_staged_cache_segment<3, 21, 0, 4>(cache, slot_for_kw, word);
            copy_staged_cache_segment<4, 0, 4, 25>(cache, slot_for_kw, word);
            copy_staged_cache_segment<5, 0, 29, 3>(cache, slot_for_kw, word);
            break;
        case 4:
            copy_staged_cache_segment<5, 3, 0, 22>(cache, slot_for_kw, word);
            copy_staged_cache_segment<6, 0, 22, 10>(cache, slot_for_kw, word);
            break;
        case 5:
            copy_staged_cache_segment<6, 10, 0, 15>(cache, slot_for_kw, word);
            copy_staged_cache_segment<7, 0, 15, 17>(cache, slot_for_kw, word);
            break;
        case 6:
            copy_staged_cache_segment<7, 17, 0, 8>(cache, slot_for_kw, word);
            copy_staged_cache_segment<8, 0, 8, 24>(cache, slot_for_kw, word);
            break;
        case 7:
            copy_staged_cache_segment<8, 24, 0, 1>(cache, slot_for_kw, word);
            break;
        default:
            break;
    }
}

template <typename CacheT>
static void build_staged_c28_word(
    const CacheT& cache,
    const u8_t slot_for_kw[3],
    int kt,
    act_vec_t& word) {
#pragma HLS INLINE off
    switch (kt) {
        case 0:
            copy_staged_cache_chunk_segment<0, 0, 0, 0, 28>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<1, 0, 0, 28, 4>(cache, slot_for_kw, word);
            break;
        case 1:
            copy_staged_cache_chunk_segment<1, 0, 4, 0, 24>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<2, 0, 0, 24, 8>(cache, slot_for_kw, word);
            break;
        case 2:
            copy_staged_cache_chunk_segment<2, 0, 8, 0, 20>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<3, 0, 0, 20, 12>(cache, slot_for_kw, word);
            break;
        case 3:
            copy_staged_cache_chunk_segment<3, 0, 12, 0, 16>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<4, 0, 0, 16, 16>(cache, slot_for_kw, word);
            break;
        case 4:
            copy_staged_cache_chunk_segment<4, 0, 16, 0, 12>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<5, 0, 0, 12, 20>(cache, slot_for_kw, word);
            break;
        case 5:
            copy_staged_cache_chunk_segment<5, 0, 20, 0, 8>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<6, 0, 0, 8, 24>(cache, slot_for_kw, word);
            break;
        case 6:
            copy_staged_cache_chunk_segment<6, 0, 24, 0, 4>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<7, 0, 0, 4, 28>(cache, slot_for_kw, word);
            break;
        case 7:
            copy_staged_cache_chunk_segment<8, 0, 0, 0, 28>(cache, slot_for_kw, word);
            break;
        default:
            break;
    }
}

template <typename CacheT>
static void build_staged_c64_word(
    const CacheT& cache,
    const u8_t slot_for_kw[3],
    int kt,
    act_vec_t& word) {
#pragma HLS INLINE off
    switch (kt) {
        case 0:
            copy_staged_cache_chunk_segment<0, 0, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 1:
            copy_staged_cache_chunk_segment<0, 1, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 2:
            copy_staged_cache_chunk_segment<1, 0, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 3:
            copy_staged_cache_chunk_segment<1, 1, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 4:
            copy_staged_cache_chunk_segment<2, 0, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 5:
            copy_staged_cache_chunk_segment<2, 1, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 6:
            copy_staged_cache_chunk_segment<3, 0, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 7:
            copy_staged_cache_chunk_segment<3, 1, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 8:
            copy_staged_cache_chunk_segment<4, 0, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 9:
            copy_staged_cache_chunk_segment<4, 1, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 10:
            copy_staged_cache_chunk_segment<5, 0, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 11:
            copy_staged_cache_chunk_segment<5, 1, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 12:
            copy_staged_cache_chunk_segment<6, 0, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 13:
            copy_staged_cache_chunk_segment<6, 1, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 14:
            copy_staged_cache_chunk_segment<7, 0, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 15:
            copy_staged_cache_chunk_segment<7, 1, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 16:
            copy_staged_cache_chunk_segment<8, 0, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 17:
            copy_staged_cache_chunk_segment<8, 1, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        default:
            break;
    }
}

template <typename CacheT>
static void build_staged_c128_word(
    const CacheT& cache,
    const u8_t slot_for_kw[3],
    int kt,
    act_vec_t& word) {
#pragma HLS INLINE off
    switch (kt) {
        case 0:
            copy_staged_cache_chunk_segment<0, 0, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 1:
            copy_staged_cache_chunk_segment<0, 1, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 2:
            copy_staged_cache_chunk_segment<0, 2, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 3:
            copy_staged_cache_chunk_segment<0, 3, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 4:
            copy_staged_cache_chunk_segment<1, 0, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 5:
            copy_staged_cache_chunk_segment<1, 1, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 6:
            copy_staged_cache_chunk_segment<1, 2, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 7:
            copy_staged_cache_chunk_segment<1, 3, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 8:
            copy_staged_cache_chunk_segment<2, 0, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 9:
            copy_staged_cache_chunk_segment<2, 1, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 10:
            copy_staged_cache_chunk_segment<2, 2, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 11:
            copy_staged_cache_chunk_segment<2, 3, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 12:
            copy_staged_cache_chunk_segment<3, 0, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 13:
            copy_staged_cache_chunk_segment<3, 1, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 14:
            copy_staged_cache_chunk_segment<3, 2, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 15:
            copy_staged_cache_chunk_segment<3, 3, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 16:
            copy_staged_cache_chunk_segment<4, 0, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 17:
            copy_staged_cache_chunk_segment<4, 1, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 18:
            copy_staged_cache_chunk_segment<4, 2, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 19:
            copy_staged_cache_chunk_segment<4, 3, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 20:
            copy_staged_cache_chunk_segment<5, 0, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 21:
            copy_staged_cache_chunk_segment<5, 1, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 22:
            copy_staged_cache_chunk_segment<5, 2, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 23:
            copy_staged_cache_chunk_segment<5, 3, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 24:
            copy_staged_cache_chunk_segment<6, 0, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 25:
            copy_staged_cache_chunk_segment<6, 1, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 26:
            copy_staged_cache_chunk_segment<6, 2, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 27:
            copy_staged_cache_chunk_segment<6, 3, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 28:
            copy_staged_cache_chunk_segment<7, 0, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 29:
            copy_staged_cache_chunk_segment<7, 1, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 30:
            copy_staged_cache_chunk_segment<7, 2, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 31:
            copy_staged_cache_chunk_segment<7, 3, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 32:
            copy_staged_cache_chunk_segment<8, 0, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 33:
            copy_staged_cache_chunk_segment<8, 1, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 34:
            copy_staged_cache_chunk_segment<8, 2, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 35:
            copy_staged_cache_chunk_segment<8, 3, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        default:
            break;
    }
}

template <typename CacheT>
static void build_staged_c131_word(
    const CacheT& cache,
    const u8_t slot_for_kw[3],
    int kt,
    act_vec_t& word) {
#pragma HLS INLINE off
    switch (kt) {
        case 0:
            copy_staged_cache_chunk_segment<0, 0, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 1:
            copy_staged_cache_chunk_segment<0, 1, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 2:
            copy_staged_cache_chunk_segment<0, 2, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 3:
            copy_staged_cache_chunk_segment<0, 3, 0, 0, 32>(cache, slot_for_kw, word);
            break;
        case 4:
            copy_staged_cache_chunk_segment<0, 4, 0, 0, 3>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<1, 0, 0, 3, 29>(cache, slot_for_kw, word);
            break;
        case 5:
            copy_staged_cache_chunk_segment<1, 0, 29, 0, 3>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<1, 1, 0, 3, 29>(cache, slot_for_kw, word);
            break;
        case 6:
            copy_staged_cache_chunk_segment<1, 1, 29, 0, 3>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<1, 2, 0, 3, 29>(cache, slot_for_kw, word);
            break;
        case 7:
            copy_staged_cache_chunk_segment<1, 2, 29, 0, 3>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<1, 3, 0, 3, 29>(cache, slot_for_kw, word);
            break;
        case 8:
            copy_staged_cache_chunk_segment<1, 3, 29, 0, 3>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<1, 4, 0, 3, 3>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<2, 0, 0, 6, 26>(cache, slot_for_kw, word);
            break;
        case 9:
            copy_staged_cache_chunk_segment<2, 0, 26, 0, 6>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<2, 1, 0, 6, 26>(cache, slot_for_kw, word);
            break;
        case 10:
            copy_staged_cache_chunk_segment<2, 1, 26, 0, 6>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<2, 2, 0, 6, 26>(cache, slot_for_kw, word);
            break;
        case 11:
            copy_staged_cache_chunk_segment<2, 2, 26, 0, 6>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<2, 3, 0, 6, 26>(cache, slot_for_kw, word);
            break;
        case 12:
            copy_staged_cache_chunk_segment<2, 3, 26, 0, 6>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<2, 4, 0, 6, 3>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<3, 0, 0, 9, 23>(cache, slot_for_kw, word);
            break;
        case 13:
            copy_staged_cache_chunk_segment<3, 0, 23, 0, 9>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<3, 1, 0, 9, 23>(cache, slot_for_kw, word);
            break;
        case 14:
            copy_staged_cache_chunk_segment<3, 1, 23, 0, 9>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<3, 2, 0, 9, 23>(cache, slot_for_kw, word);
            break;
        case 15:
            copy_staged_cache_chunk_segment<3, 2, 23, 0, 9>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<3, 3, 0, 9, 23>(cache, slot_for_kw, word);
            break;
        case 16:
            copy_staged_cache_chunk_segment<3, 3, 23, 0, 9>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<3, 4, 0, 9, 3>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<4, 0, 0, 12, 20>(cache, slot_for_kw, word);
            break;
        case 17:
            copy_staged_cache_chunk_segment<4, 0, 20, 0, 12>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<4, 1, 0, 12, 20>(cache, slot_for_kw, word);
            break;
        case 18:
            copy_staged_cache_chunk_segment<4, 1, 20, 0, 12>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<4, 2, 0, 12, 20>(cache, slot_for_kw, word);
            break;
        case 19:
            copy_staged_cache_chunk_segment<4, 2, 20, 0, 12>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<4, 3, 0, 12, 20>(cache, slot_for_kw, word);
            break;
        case 20:
            copy_staged_cache_chunk_segment<4, 3, 20, 0, 12>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<4, 4, 0, 12, 3>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<5, 0, 0, 15, 17>(cache, slot_for_kw, word);
            break;
        case 21:
            copy_staged_cache_chunk_segment<5, 0, 17, 0, 15>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<5, 1, 0, 15, 17>(cache, slot_for_kw, word);
            break;
        case 22:
            copy_staged_cache_chunk_segment<5, 1, 17, 0, 15>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<5, 2, 0, 15, 17>(cache, slot_for_kw, word);
            break;
        case 23:
            copy_staged_cache_chunk_segment<5, 2, 17, 0, 15>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<5, 3, 0, 15, 17>(cache, slot_for_kw, word);
            break;
        case 24:
            copy_staged_cache_chunk_segment<5, 3, 17, 0, 15>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<5, 4, 0, 15, 3>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<6, 0, 0, 18, 14>(cache, slot_for_kw, word);
            break;
        case 25:
            copy_staged_cache_chunk_segment<6, 0, 14, 0, 18>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<6, 1, 0, 18, 14>(cache, slot_for_kw, word);
            break;
        case 26:
            copy_staged_cache_chunk_segment<6, 1, 14, 0, 18>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<6, 2, 0, 18, 14>(cache, slot_for_kw, word);
            break;
        case 27:
            copy_staged_cache_chunk_segment<6, 2, 14, 0, 18>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<6, 3, 0, 18, 14>(cache, slot_for_kw, word);
            break;
        case 28:
            copy_staged_cache_chunk_segment<6, 3, 14, 0, 18>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<6, 4, 0, 18, 3>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<7, 0, 0, 21, 11>(cache, slot_for_kw, word);
            break;
        case 29:
            copy_staged_cache_chunk_segment<7, 0, 11, 0, 21>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<7, 1, 0, 21, 11>(cache, slot_for_kw, word);
            break;
        case 30:
            copy_staged_cache_chunk_segment<7, 1, 11, 0, 21>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<7, 2, 0, 21, 11>(cache, slot_for_kw, word);
            break;
        case 31:
            copy_staged_cache_chunk_segment<7, 2, 11, 0, 21>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<7, 3, 0, 21, 11>(cache, slot_for_kw, word);
            break;
        case 32:
            copy_staged_cache_chunk_segment<7, 3, 11, 0, 21>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<7, 4, 0, 21, 3>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<8, 0, 0, 24, 8>(cache, slot_for_kw, word);
            break;
        case 33:
            copy_staged_cache_chunk_segment<8, 0, 8, 0, 24>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<8, 1, 0, 24, 8>(cache, slot_for_kw, word);
            break;
        case 34:
            copy_staged_cache_chunk_segment<8, 1, 8, 0, 24>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<8, 2, 0, 24, 8>(cache, slot_for_kw, word);
            break;
        case 35:
            copy_staged_cache_chunk_segment<8, 2, 8, 0, 24>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<8, 3, 0, 24, 8>(cache, slot_for_kw, word);
            break;
        case 36:
            copy_staged_cache_chunk_segment<8, 3, 8, 0, 24>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<8, 4, 0, 24, 3>(cache, slot_for_kw, word);
            break;
        default:
            break;
    }
}

static void emit_narrow_words_for_mode(
    unsigned mode,
    const narrow_3x3_window_t& window,
    const u8_t slot_for_kw[3],
    hls::stream<act_vec_t>& act_stream) {
#pragma HLS INLINE off
    if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C3)) {
        for (int kt = 0; kt < 1; ++kt) {
#pragma HLS PIPELINE II=1
            act_vec_t word = 0;
            build_staged_c3_word(window, slot_for_kw, kt, word);
            act_stream.write(word);
        }
        return;
    }
    if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C12)) {
        for (int kt = 0; kt < 4; ++kt) {
#pragma HLS PIPELINE II=1
            act_vec_t word = 0;
            build_staged_c12_word(window, slot_for_kw, kt, word);
            act_stream.write(word);
        }
        return;
    }
    if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C19)) {
        for (int kt = 0; kt < 6; ++kt) {
#pragma HLS PIPELINE II=1
            act_vec_t word = 0;
            build_staged_c19_word(window, slot_for_kw, kt, word);
            act_stream.write(word);
        }
        return;
    }
    if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C25)) {
        for (int kt = 0; kt < 8; ++kt) {
#pragma HLS PIPELINE II=1
            act_vec_t word = 0;
            build_staged_c25_word(window, slot_for_kw, kt, word);
            act_stream.write(word);
        }
        return;
    }
}

static void emit_narrow_word_pair_for_mode(
    unsigned mode,
    const narrow_3x3_window_t& window0,
    const narrow_3x3_window_t& window1,
    const u8_t slot_for_kw0[3],
    const u8_t slot_for_kw1[3],
    hls::stream<act_vec_t>& act_stream0,
    hls::stream<act_vec_t>& act_stream1) {
#pragma HLS INLINE off
    const int k_tiles = (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C3))
                            ? 1
                            : (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C12))
                                  ? 4
                                  : (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C19)) ? 6 : 8;
    for (int kt = 0; kt < 8; ++kt) {
#pragma HLS PIPELINE II=1
        if (kt >= k_tiles) {
            break;
        }
        act_vec_t word0 = 0;
        act_vec_t word1 = 0;
        if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C3)) {
            build_staged_c3_word(window0, slot_for_kw0, kt, word0);
            build_staged_c3_word(window1, slot_for_kw1, kt, word1);
        } else if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C12)) {
            build_staged_c12_word(window0, slot_for_kw0, kt, word0);
            build_staged_c12_word(window1, slot_for_kw1, kt, word1);
        } else if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C19)) {
            build_staged_c19_word(window0, slot_for_kw0, kt, word0);
            build_staged_c19_word(window1, slot_for_kw1, kt, word1);
        } else {
            build_staged_c25_word(window0, slot_for_kw0, kt, word0);
            build_staged_c25_word(window1, slot_for_kw1, kt, word1);
        }
        act_stream0.write(word0);
        act_stream1.write(word1);
    }
}

static void emit_wide_words_for_mode(
    unsigned mode,
    const wide_3x3_cache_t& cache,
    const u8_t slot_for_kw[3],
    hls::stream<act_vec_t>& act_stream) {
#pragma HLS INLINE off
    if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C28)) {
        for (int kt = 0; kt < 8; ++kt) {
            act_vec_t word = 0;
            build_staged_c28_word(cache, slot_for_kw, kt, word);
            act_stream.write(word);
        }
        return;
    }
    if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C64)) {
        for (int kt = 0; kt < 18; ++kt) {
            act_vec_t word = 0;
            build_staged_c64_word(cache, slot_for_kw, kt, word);
            act_stream.write(word);
        }
        return;
    }
    if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C128)) {
        for (int kt = 0; kt < 36; ++kt) {
            act_vec_t word = 0;
            build_staged_c128_word(cache, slot_for_kw, kt, word);
            act_stream.write(word);
        }
        return;
    }
    if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C131)) {
        for (int kt = 0; kt < 37; ++kt) {
            act_vec_t word = 0;
            build_staged_c131_word(cache, slot_for_kw, kt, word);
            act_stream.write(word);
        }
        return;
    }
}

static void window_assembler_store_wide_word(
    const window_load_word_t& token) {
#pragma HLS INLINE
    const unsigned slot = token.slot.to_uint();
    const unsigned row = token.row.to_uint();
    const unsigned chunk = token.chunk.to_uint();
    if (row == 0U) {
        s_wide_cache.data[0][slot][chunk] = token.word;
    } else if (row == 1U) {
        s_wide_cache.data[1][slot][chunk] = token.word;
    } else {
        s_wide_cache.data[2][slot][chunk] = token.word;
    }
}

static void window_row_assembler_read_narrow_window(
    hls::stream<window_load_word_t>& load_word_stream,
    narrow_3x3_window_t& window) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=window.spatial complete dim=1
    for (int spatial = 0; spatial < 9; ++spatial) {
#pragma HLS PIPELINE II=1
        window.spatial[spatial] = load_word_stream.read().word;
    }
}

static void window_row_assembler(
    hls::stream<window_assembler_cfg_t>& assembler_cfg_stream,
    hls::stream<window_column_meta_t>& column_meta_stream,
    hls::stream<narrow_paired_issue_meta_t>& narrow_issue_stream,
    hls::stream<window_load_word_t>& load_word_stream,
    hls::stream<act_vec_t>& act_stream0,
    hls::stream<act_vec_t>& act_stream1) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=s_wide_cache.data complete dim=1
#pragma HLS ARRAY_PARTITION variable=s_wide_cache.data complete dim=3
#pragma HLS BIND_STORAGE variable=s_wide_cache.data type=ram_2p impl=lutram
#pragma HLS RESET variable=s_wide_cache off

    const window_assembler_cfg_t cfg = assembler_cfg_stream.read();
    const unsigned mode = cfg.mode.to_uint();
    const bool narrow =
        cfg.loader_class.to_uint() ==
        static_cast<unsigned>(WIN_LOADER_3X3_NARROW);
    const bool paired =
        (cfg.flags.to_uint() &
         static_cast<unsigned>(WINDOW_SCHED_FLAG_PIXEL_PARALLEL_2)) != 0U;
    const int out_w = static_cast<int>(cfg.out_w.to_uint());
    const int issue_count = paired ? ((out_w + 1) / 2) : out_w;
    const int request_cols =
        static_cast<int>(cfg.loader_request_cols.to_uint());
    const int words_per_col =
        static_cast<int>(cfg.loader_words_per_col.to_uint());

    for (int issue = 0; issue < MAX_FM_W; ++issue) {
        if (issue >= issue_count) {
            break;
        }
        if (narrow) {
            const narrow_paired_issue_meta_t meta =
                narrow_issue_stream.read();
            const bool has_second_window =
                meta.has_second_window.to_uint() != 0U;
            narrow_3x3_window_t window0;
            narrow_3x3_window_t window1;
#pragma HLS ARRAY_PARTITION variable=window0.spatial complete dim=1
#pragma HLS ARRAY_PARTITION variable=window1.spatial complete dim=1
            for (int spatial = 0; spatial < 9; ++spatial) {
#pragma HLS UNROLL
                window0.spatial[spatial] = 0;
                window1.spatial[spatial] = 0;
            }
            window_row_assembler_read_narrow_window(
                load_word_stream, window0);
            if (has_second_window) {
                window_row_assembler_read_narrow_window(
                    load_word_stream, window1);
            }
            u8_t unused_slots0[3] = {0, 0, 0};
            u8_t unused_slots1[3] = {0, 0, 0};
#pragma HLS ARRAY_PARTITION variable=unused_slots0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=unused_slots1 complete dim=1
            if (!paired) {
                emit_narrow_words_for_mode(
                    mode, window0, unused_slots0, act_stream0);
            } else {
                emit_narrow_word_pair_for_mode(
                    mode,
                    window0,
                    window1,
                    unused_slots0,
                    unused_slots1,
                    act_stream0,
                    act_stream1);
            }
            continue;
        }

        u8_t slot_for_kw0[3];
        u8_t slot_for_kw1[3];
#pragma HLS ARRAY_PARTITION variable=slot_for_kw0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=slot_for_kw1 complete dim=1
        for (int request = 0; request < 6; ++request) {
#pragma HLS PIPELINE off
            if (request >= request_cols) {
                break;
            }
            const window_column_meta_t meta =
                column_meta_stream.read();
            if (meta.update.to_uint() != 0U) {
                for (int word_index = 0;
                     word_index < 3 * MAX_3X3_CACHE_CHUNKS;
                     ++word_index) {
#pragma HLS PIPELINE off
                    if (word_index >= words_per_col) {
                        break;
                    }
                    const window_load_word_t token =
                        load_word_stream.read();
                    window_assembler_store_wide_word(token);
                }
            }

            const bool second_phase = request >= 3;
            const int kw =
                second_phase ? request - 3 : request;
            if (second_phase) {
                slot_for_kw1[kw] = meta.slot;
            } else {
                slot_for_kw0[kw] = meta.slot;
            }
        }
        emit_wide_words_for_mode(
            mode, s_wide_cache, slot_for_kw0, act_stream0);
    }
}

static void scheduled_3x3_window_row_pipeline(
    const tensor_desc_t& src_desc,
    const window_sched_desc_t& sched,
    hls::stream<act_vec_t>& act_stream0,
    hls::stream<act_vec_t>& act_stream1,
    u16_t out_row) {
#pragma HLS INLINE off
    hls::stream<window_row_cfg_t> loader_cfg_stream;
    hls::stream<window_assembler_cfg_t> assembler_cfg_stream;
    hls::stream<window_column_meta_t> column_meta_stream;
    hls::stream<narrow_paired_issue_meta_t> narrow_issue_stream;
    hls::stream<window_load_word_t> load_word_stream;
#pragma HLS STREAM variable=loader_cfg_stream depth=2
#pragma HLS STREAM variable=assembler_cfg_stream depth=3
#pragma HLS STREAM variable=column_meta_stream depth=16
#pragma HLS STREAM variable=narrow_issue_stream depth=16
#pragma HLS STREAM variable=load_word_stream depth=16
#pragma HLS BIND_STORAGE variable=assembler_cfg_stream type=fifo impl=lutram
#pragma HLS BIND_STORAGE variable=column_meta_stream type=fifo impl=lutram
#pragma HLS BIND_STORAGE variable=narrow_issue_stream type=fifo impl=lutram
#pragma HLS BIND_STORAGE variable=load_word_stream type=fifo impl=lutram
#pragma HLS DATAFLOW
    configure_window_row(
        src_desc,
        sched,
        out_row,
        loader_cfg_stream,
        assembler_cfg_stream);
    window_row_loader(
        loader_cfg_stream,
        column_meta_stream,
        narrow_issue_stream,
        load_word_stream);
    window_row_assembler(
        assembler_cfg_stream,
        column_meta_stream,
        narrow_issue_stream,
        load_word_stream,
        act_stream0,
        act_stream1);
}

static void scheduled_1x1_window_row(const tensor_desc_t& src_desc,
                                     const window_sched_desc_t& sched,
                                     hls::stream<act_vec_t>& act_stream0,
                                     hls::stream<act_vec_t>& act_stream1,
                                     u16_t out_row,
                                     bool aligned_full_tile) {
#pragma HLS INLINE off
    const int stride_i = static_cast<int>(sched.stride.to_uint());
    const int out_w_i = static_cast<int>(sched.out_w.to_uint());
    const int k_tiles_i = static_cast<int>(sched.k_tiles.to_uint());
    const int in_c_i = static_cast<int>(sched.in_c.to_uint());
    const i32_t ih = static_cast<i32_t>(out_row.to_uint() * static_cast<unsigned>(stride_i));

    const bool paired =
        (sched.flags.to_uint() & static_cast<unsigned>(WINDOW_SCHED_FLAG_PIXEL_PARALLEL_2)) != 0U;
    const bool odd_tail =
        (sched.flags.to_uint() & static_cast<unsigned>(WINDOW_SCHED_FLAG_ODD_TAIL)) != 0U;
    const int full_issue_count = paired ? ((out_w_i + 1) / 2) : out_w_i;
    for (int issue_i = 0; issue_i < MAX_FM_W; ++issue_i) {
        if (issue_i >= full_issue_count) {
            break;
        }
        const int pixel0 = paired ? issue_i * 2 : issue_i;
        const i32_t iw0 = static_cast<i32_t>(pixel0 * stride_i);
        const i32_t iw1 = static_cast<i32_t>((pixel0 + 1) * stride_i);
        for (int kt = 0; kt < MAX_K_TILE_COUNT; ++kt) {
#pragma HLS PIPELINE off
            if (kt >= k_tiles_i) {
                break;
            }
            const unsigned c_begin = static_cast<unsigned>(kt) * static_cast<unsigned>(TK);
            act_vec_t word0 = 0;
            if (aligned_full_tile &&
                win_can_read_aligned_1x1(src_desc, static_cast<u16_t>(c_begin))) {
                word0 = read_aligned_or_zero(src_desc, ih, iw0, static_cast<u16_t>(c_begin));
            } else {
                unsigned valid = (c_begin < static_cast<unsigned>(in_c_i))
                                     ? (static_cast<unsigned>(in_c_i) - c_begin)
                                     : 0U;
                if (valid > static_cast<unsigned>(TK)) {
                    valid = static_cast<unsigned>(TK);
                }
                word0 = read_tile_or_zero(src_desc,
                                         ih,
                                         iw0,
                                         static_cast<u16_t>(c_begin),
                                         static_cast<u8_t>(valid));
            }
            act_stream0.write(word0);
            if (paired) {
                act_vec_t word1 = 0;
                const bool tail_issue = odd_tail && issue_i == full_issue_count - 1;
                if (!tail_issue) {
                    if (aligned_full_tile &&
                        win_can_read_aligned_1x1(src_desc, static_cast<u16_t>(c_begin))) {
                        word1 = read_aligned_or_zero(src_desc, ih, iw1, static_cast<u16_t>(c_begin));
                    } else {
                        unsigned valid = (c_begin < static_cast<unsigned>(in_c_i))
                                             ? (static_cast<unsigned>(in_c_i) - c_begin)
                                             : 0U;
                        if (valid > static_cast<unsigned>(TK)) {
                            valid = static_cast<unsigned>(TK);
                        }
                        word1 = read_tile_or_zero(src_desc,
                                                  ih,
                                                  iw1,
                                                  static_cast<u16_t>(c_begin),
                                                  static_cast<u8_t>(valid));
                    }
                }
                act_stream1.write(word1);
            }
        }
    }
}

void scheduled_window_generator_row(const tensor_desc_t& src_desc,
                                     const window_sched_desc_t& sched,
                                     hls::stream<act_vec_t>& act_stream0,
                                     hls::stream<act_vec_t>& act_stream1,
                                     u16_t out_row,
                                     volatile u8_t& prof_conv_win_state) {
#pragma HLS INLINE off
    npu_profile_set_conv_win_state(prof_conv_win_state, PROF_CONV_WIN_ACTIVE);
    const unsigned mode = sched.mode.to_uint();
    const unsigned loader_class = sched.loader_class.to_uint();
    if (loader_class == static_cast<unsigned>(WIN_LOADER_3X3_NARROW) ||
        loader_class == static_cast<unsigned>(WIN_LOADER_3X3_WIDE)) {
        scheduled_3x3_window_row_pipeline(
            src_desc, sched, act_stream0, act_stream1, out_row);
    } else if (mode == static_cast<unsigned>(WIN_MODE_1X1_ALIGNED)) {
        scheduled_1x1_window_row(src_desc, sched, act_stream0, act_stream1, out_row, true);
    } else if (mode == static_cast<unsigned>(WIN_MODE_1X1_PACKED)) {
        scheduled_1x1_window_row(src_desc, sched, act_stream0, act_stream1, out_row, false);
    }
    npu_profile_set_conv_win_state(prof_conv_win_state,
                                   PROF_CONV_WIN_IDLE_OR_DONE);
}

}  // namespace esp_int8
