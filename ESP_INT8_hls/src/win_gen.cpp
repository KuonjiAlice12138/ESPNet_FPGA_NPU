#include "../include/npu_config.hpp"
#include "../include/npu_ctrl.hpp"
#include "../include/npu_schedule.hpp"
#include "../include/npu_types.hpp"

namespace esp_int8 {

#if !defined(__SYNTHESIS__) && \
    defined(ESP_INT8_CSIM_VALIDATE_INCREMENTAL_TRANSFER)
static unsigned long long s_csim_narrow_transfer_words = 0;

void csim_reset_narrow_transfer_words() {
    s_csim_narrow_transfer_words = 0;
}

unsigned long long csim_get_narrow_transfer_words() {
    return s_csim_narrow_transfer_words;
}

static void csim_count_narrow_transfer_word() {
    ++s_csim_narrow_transfer_words;
}
#else
static void csim_count_narrow_transfer_word() {
#pragma HLS INLINE
}
#endif

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


static bool win_can_read_aligned_1x1(const tensor_desc_t& desc, u16_t c_begin) {
#pragma HLS INLINE
    const unsigned phys_c = tensor_phys_c(desc).to_uint();
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

// Column-delta storage is owned exclusively by the assembler.
struct narrow_delta_cache_t {
    act_vec_t data[3][WINGEN_NARROW_CACHE_COL_SLOTS];
};
static narrow_delta_cache_t s_delta_cache;

// The raw C1 frame has 240-byte rows; no padded copy is stored in FM memory.
static act_vec_t s_c1_rows[3][WINGEN_C1_PACKED_ROW_WORDS];
static bool s_c1_rows_valid = false;
static u16_t s_c1_last_row = 0;
static u32_t s_c1_source_base = 0;
static u8_t s_c1_source_bank = 0;


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

static u32_t win_build_lane_mask(u8_t byte_count) {
#pragma HLS INLINE
    u32_t lane_mask = 0;
    for (int lane = 0; lane < AXI_WORD_BYTES; ++lane) {
#pragma HLS UNROLL
        lane_mask[lane] =
            lane < static_cast<int>(byte_count.to_uint()) ? 1 : 0;
    }
    return lane_mask;
}

static act_vec_t win_apply_lane_mask(const act_vec_t& word,
                                     const u32_t& lane_mask) {
#pragma HLS INLINE
    act_vec_t masked;
    for (int lane = 0; lane < AXI_WORD_BYTES; ++lane) {
#pragma HLS UNROLL
        for (int bit = 0; bit < 8; ++bit) {
#pragma HLS UNROLL
            const int bit_index = lane * 8 + bit;
            masked[bit_index] = word[bit_index] & lane_mask[lane];
        }
    }
    return masked;
}

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
    u8_t loader_words_per_col;
    u8_t loader_warmup_mask;
    u8_t loader_steady_mask;
    u16_t loader_warmup_run0;
    u16_t loader_steady_run0;
    u8_t narrow_transfer_protocol;
    u8_t narrow_valid_c;
    u8_t last_chunk_valid_c;
    u32_t narrow_lane_mask;
    u32_t last_chunk_lane_mask;
    u16_t out_w;
};

struct window_assembler_cfg_t {
    u8_t mode;
    u8_t loader_class;
    u8_t flags;
    u16_t out_w;
    u8_t loader_request_cols;
    u8_t loader_words_per_col;
    u8_t narrow_transfer_protocol;
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
    cfg.loader_words_per_col = sched.loader_words_per_col;
    cfg.loader_warmup_mask = sched.loader_warmup_mask;
    cfg.loader_steady_mask = sched.loader_steady_mask;
    cfg.loader_warmup_run0 = sched.reserved[WINDOW_LOADER_WARMUP_RUN_WORD];
    cfg.loader_steady_run0 = sched.reserved[WINDOW_LOADER_STEADY_RUN_WORD];
    cfg.narrow_transfer_protocol = static_cast<u8_t>(
        (sched.reserved[WINDOW_LOADER_ROW_REUSE_WORD].to_uint() >>
         WINDOW_TRANSFER_PROTOCOL_SHIFT) & WINDOW_TRANSFER_PROTOCOL_MASK);
    const unsigned input_channels = sched.in_c.to_uint();
    unsigned narrow_valid_c = input_channels;
    if (narrow_valid_c == 0U ||
        narrow_valid_c > static_cast<unsigned>(AXI_WORD_BYTES)) {
        narrow_valid_c = static_cast<unsigned>(AXI_WORD_BYTES);
    }
    const unsigned cache_chunks = sched.cache_chunks.to_uint();
    const unsigned last_chunk_begin =
        cache_chunks == 0U
            ? 0U
            : (cache_chunks - 1U) * static_cast<unsigned>(AXI_WORD_BYTES);
    unsigned last_chunk_valid_c =
        input_channels > last_chunk_begin
            ? input_channels - last_chunk_begin
            : static_cast<unsigned>(AXI_WORD_BYTES);
    if (last_chunk_valid_c == 0U ||
        last_chunk_valid_c > static_cast<unsigned>(AXI_WORD_BYTES)) {
        last_chunk_valid_c = static_cast<unsigned>(AXI_WORD_BYTES);
    }
    cfg.narrow_valid_c = static_cast<u8_t>(narrow_valid_c);
    cfg.last_chunk_valid_c = static_cast<u8_t>(last_chunk_valid_c);
    cfg.narrow_lane_mask = win_build_lane_mask(cfg.narrow_valid_c);
    cfg.last_chunk_lane_mask = win_build_lane_mask(cfg.last_chunk_valid_c);
    cfg.out_w = sched.out_w;

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
    assembler_cfg.narrow_transfer_protocol = cfg.narrow_transfer_protocol;
    assembler_cfg_stream.write(assembler_cfg);
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
                                             u32_t lane_mask,
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
                       static_cast<u32_t>(tensor_phys_c(src_desc)) +
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
        packed = win_apply_lane_mask(packed, lane_mask);
    }
    return true;
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
            const bool last_chunk =
                chunk + 1 == static_cast<int>(cfg.cache_chunks.to_uint());
            const u8_t valid =
                last_chunk ? cfg.last_chunk_valid_c
                           : static_cast<u8_t>(AXI_WORD_BYTES);
            const u32_t lane_mask =
                last_chunk ? cfg.last_chunk_lane_mask
                           : static_cast<u32_t>(~u32_t(0));
            read_packed_tile_from_row_cached(cfg.src_desc,
                                             cursor,
                                             window_cfg_row_base(cfg, kh),
                                             window_cfg_row_valid(cfg, kh),
                                             input_col,
                                             static_cast<u16_t>(c_begin_u),
                                             valid,
                                             lane_mask,
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

static int window_loader_run_start_delta(u16_t run_word) {
#pragma HLS INLINE
    int delta = static_cast<int>(run_word.to_uint() & 0xFFU);
    if ((delta & 0x80) != 0) {
        delta -= 0x100;
    }
    return delta;
}

template <int COLUMN_COUNT>
static void window_loader_emit_narrow_direct_run(
    const window_row_cfg_t& cfg,
    i32_t start_col,
    hls::stream<window_load_word_t>& load_word_stream) {
#pragma HLS INLINE
    for (int offset = 0; offset < COLUMN_COUNT; ++offset) {
#pragma HLS PIPELINE off
        const i32_t input_col = start_col + static_cast<i32_t>(offset);
        for (int kh = 0; kh < 3; ++kh) {
#pragma HLS PIPELINE off
            act_vec_t word = 0;
            read_packed_tile_from_row_cached(
                cfg.src_desc,
                s_packed_word_cursor[kh],
                window_cfg_row_base(cfg, kh),
                window_cfg_row_valid(cfg, kh),
                input_col,
                static_cast<u16_t>(0),
                cfg.narrow_valid_c,
                cfg.narrow_lane_mask,
                word);
            window_load_word_t token;
            token.word = word;
            token.slot = static_cast<u8_t>(offset);
            token.row = static_cast<u8_t>(kh);
            token.chunk = 0;
            csim_count_narrow_transfer_word();
            load_word_stream.write(token);
        }
    }
}

static void window_row_loader_narrow_incremental_c12(
    const window_row_cfg_t& cfg,
    hls::stream<window_load_word_t>& load_word_stream) {
#pragma HLS INLINE off
    const int issue_count = window_issue_count(cfg);
    const int stride = static_cast<int>(cfg.stride.to_uint());
    const int padding = static_cast<int>(cfg.padding.to_uint());
    for (int issue = 0; issue < MAX_FM_W; ++issue) {
#pragma HLS PIPELINE off
        if (issue >= issue_count) {
            break;
        }
        const bool warmup = issue == 0;
        const u16_t run_word =
            warmup ? cfg.loader_warmup_run0 : cfg.loader_steady_run0;
        const i32_t start_col = static_cast<i32_t>(
            issue * 2 * stride - padding +
            window_loader_run_start_delta(run_word));
        if (warmup) {
            window_loader_emit_narrow_direct_run<4>(
                cfg, start_col, load_word_stream);
        } else {
            window_loader_emit_narrow_direct_run<2>(
                cfg, start_col, load_word_stream);
        }
    }
}

static void window_loader_emit_delta_column(
    const window_row_cfg_t& cfg, i32_t input_col, u8_t slot,
    hls::stream<window_load_word_t>& load_word_stream) {
#pragma HLS INLINE
    for (int kh = 0; kh < 3; ++kh) {
#pragma HLS PIPELINE off
        window_load_word_t token;
        token.word = 0;
        read_packed_tile_from_row_cached(
            cfg.src_desc, s_packed_word_cursor[kh], window_cfg_row_base(cfg, kh),
            window_cfg_row_valid(cfg, kh), input_col, static_cast<u16_t>(0),
            cfg.narrow_valid_c, cfg.narrow_lane_mask, token.word);
        token.slot = slot;
        token.row = static_cast<u8_t>(kh);
        token.chunk = 0;
        csim_count_narrow_transfer_word();
        load_word_stream.write(token);
    }
}

static void window_row_loader(
    hls::stream<window_row_cfg_t>& loader_cfg_stream,
    hls::stream<window_column_meta_t>& column_meta_stream,
    hls::stream<window_load_word_t>& load_word_stream) {
#pragma HLS INLINE off
    const window_row_cfg_t cfg = loader_cfg_stream.read();
    reset_packed_word_cursors();
    const bool narrow = cfg.loader_class.to_uint() ==
                        static_cast<unsigned>(WIN_LOADER_3X3_NARROW);
    if (cfg.narrow_transfer_protocol.to_uint() ==
        static_cast<unsigned>(WINDOW_TRANSFER_C12_S1_INCREMENTAL)) {
        window_row_loader_narrow_incremental_c12(cfg, load_word_stream);
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
                if (narrow) {
                    window_loader_emit_delta_column(cfg, input_col,
                                                    static_cast<u8_t>(slot), load_word_stream);
                } else {
                    window_loader_emit_column(cfg, input_col,
                                              static_cast<u8_t>(slot), load_word_stream);
                }
            }
        }
    }
}

template <int SPATIAL, int SRC_LANE, int DST_LANE, int COUNT>
static void copy_staged_cache_segment(
    const narrow_3x3_window_t& cache,
    act_vec_t& word) {
#pragma HLS INLINE
    const act_vec_t src = cache.spatial[SPATIAL];
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

static void build_staged_c12_word(
    const narrow_3x3_window_t& cache,
    int kt,
    act_vec_t& word) {
#pragma HLS INLINE
    switch (kt) {
        case 0:
            copy_staged_cache_segment<0, 0, 0, 12>(cache, word);
            copy_staged_cache_segment<1, 0, 12, 12>(cache, word);
            copy_staged_cache_segment<2, 0, 24, 8>(cache, word);
            break;
        case 1:
            copy_staged_cache_segment<2, 8, 0, 4>(cache, word);
            copy_staged_cache_segment<3, 0, 4, 12>(cache, word);
            copy_staged_cache_segment<4, 0, 16, 12>(cache, word);
            copy_staged_cache_segment<5, 0, 28, 4>(cache, word);
            break;
        case 2:
            copy_staged_cache_segment<5, 4, 0, 8>(cache, word);
            copy_staged_cache_segment<6, 0, 8, 12>(cache, word);
            copy_staged_cache_segment<7, 0, 20, 12>(cache, word);
            break;
        case 3:
            copy_staged_cache_segment<8, 0, 0, 12>(cache, word);
            break;
        default:
            break;
    }
}

static void build_staged_c17_word(
    const narrow_3x3_window_t& cache,
    int kt,
    act_vec_t& word) {
#pragma HLS INLINE
    switch (kt) {
        case 0:
            copy_staged_cache_segment<0, 0, 0, 17>(cache, word);
            copy_staged_cache_segment<1, 0, 17, 15>(cache, word);
            break;
        case 1:
            copy_staged_cache_segment<1, 15, 0, 2>(cache, word);
            copy_staged_cache_segment<2, 0, 2, 17>(cache, word);
            copy_staged_cache_segment<3, 0, 19, 13>(cache, word);
            break;
        case 2:
            copy_staged_cache_segment<3, 13, 0, 4>(cache, word);
            copy_staged_cache_segment<4, 0, 4, 17>(cache, word);
            copy_staged_cache_segment<5, 0, 21, 11>(cache, word);
            break;
        case 3:
            copy_staged_cache_segment<5, 11, 0, 6>(cache, word);
            copy_staged_cache_segment<6, 0, 6, 17>(cache, word);
            copy_staged_cache_segment<7, 0, 23, 9>(cache, word);
            break;
        case 4:
            copy_staged_cache_segment<7, 9, 0, 8>(cache, word);
            copy_staged_cache_segment<8, 0, 8, 17>(cache, word);
            break;
        default:
            break;
    }
}

static void build_staged_c25_word(
    const narrow_3x3_window_t& cache,
    int kt,
    act_vec_t& word) {
#pragma HLS INLINE
    switch (kt) {
        case 0:
            copy_staged_cache_segment<0, 0, 0, 25>(cache, word);
            copy_staged_cache_segment<1, 0, 25, 7>(cache, word);
            break;
        case 1:
            copy_staged_cache_segment<1, 7, 0, 18>(cache, word);
            copy_staged_cache_segment<2, 0, 18, 14>(cache, word);
            break;
        case 2:
            copy_staged_cache_segment<2, 14, 0, 11>(cache, word);
            copy_staged_cache_segment<3, 0, 11, 21>(cache, word);
            break;
        case 3:
            copy_staged_cache_segment<3, 21, 0, 4>(cache, word);
            copy_staged_cache_segment<4, 0, 4, 25>(cache, word);
            copy_staged_cache_segment<5, 0, 29, 3>(cache, word);
            break;
        case 4:
            copy_staged_cache_segment<5, 3, 0, 22>(cache, word);
            copy_staged_cache_segment<6, 0, 22, 10>(cache, word);
            break;
        case 5:
            copy_staged_cache_segment<6, 10, 0, 15>(cache, word);
            copy_staged_cache_segment<7, 0, 15, 17>(cache, word);
            break;
        case 6:
            copy_staged_cache_segment<7, 17, 0, 8>(cache, word);
            copy_staged_cache_segment<8, 0, 8, 24>(cache, word);
            break;
        case 7:
            copy_staged_cache_segment<8, 24, 0, 1>(cache, word);
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
static void build_staged_c129_word(
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
            copy_staged_cache_chunk_segment<0, 4, 0, 0, 1>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<1, 0, 0, 1, 31>(cache, slot_for_kw, word);
            break;
        case 5:
            copy_staged_cache_chunk_segment<1, 0, 31, 0, 1>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<1, 1, 0, 1, 31>(cache, slot_for_kw, word);
            break;
        case 6:
            copy_staged_cache_chunk_segment<1, 1, 31, 0, 1>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<1, 2, 0, 1, 31>(cache, slot_for_kw, word);
            break;
        case 7:
            copy_staged_cache_chunk_segment<1, 2, 31, 0, 1>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<1, 3, 0, 1, 31>(cache, slot_for_kw, word);
            break;
        case 8:
            copy_staged_cache_chunk_segment<1, 3, 31, 0, 1>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<1, 4, 0, 1, 1>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<2, 0, 0, 2, 30>(cache, slot_for_kw, word);
            break;
        case 9:
            copy_staged_cache_chunk_segment<2, 0, 30, 0, 2>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<2, 1, 0, 2, 30>(cache, slot_for_kw, word);
            break;
        case 10:
            copy_staged_cache_chunk_segment<2, 1, 30, 0, 2>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<2, 2, 0, 2, 30>(cache, slot_for_kw, word);
            break;
        case 11:
            copy_staged_cache_chunk_segment<2, 2, 30, 0, 2>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<2, 3, 0, 2, 30>(cache, slot_for_kw, word);
            break;
        case 12:
            copy_staged_cache_chunk_segment<2, 3, 30, 0, 2>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<2, 4, 0, 2, 1>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<3, 0, 0, 3, 29>(cache, slot_for_kw, word);
            break;
        case 13:
            copy_staged_cache_chunk_segment<3, 0, 29, 0, 3>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<3, 1, 0, 3, 29>(cache, slot_for_kw, word);
            break;
        case 14:
            copy_staged_cache_chunk_segment<3, 1, 29, 0, 3>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<3, 2, 0, 3, 29>(cache, slot_for_kw, word);
            break;
        case 15:
            copy_staged_cache_chunk_segment<3, 2, 29, 0, 3>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<3, 3, 0, 3, 29>(cache, slot_for_kw, word);
            break;
        case 16:
            copy_staged_cache_chunk_segment<3, 3, 29, 0, 3>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<3, 4, 0, 3, 1>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<4, 0, 0, 4, 28>(cache, slot_for_kw, word);
            break;
        case 17:
            copy_staged_cache_chunk_segment<4, 0, 28, 0, 4>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<4, 1, 0, 4, 28>(cache, slot_for_kw, word);
            break;
        case 18:
            copy_staged_cache_chunk_segment<4, 1, 28, 0, 4>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<4, 2, 0, 4, 28>(cache, slot_for_kw, word);
            break;
        case 19:
            copy_staged_cache_chunk_segment<4, 2, 28, 0, 4>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<4, 3, 0, 4, 28>(cache, slot_for_kw, word);
            break;
        case 20:
            copy_staged_cache_chunk_segment<4, 3, 28, 0, 4>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<4, 4, 0, 4, 1>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<5, 0, 0, 5, 27>(cache, slot_for_kw, word);
            break;
        case 21:
            copy_staged_cache_chunk_segment<5, 0, 27, 0, 5>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<5, 1, 0, 5, 27>(cache, slot_for_kw, word);
            break;
        case 22:
            copy_staged_cache_chunk_segment<5, 1, 27, 0, 5>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<5, 2, 0, 5, 27>(cache, slot_for_kw, word);
            break;
        case 23:
            copy_staged_cache_chunk_segment<5, 2, 27, 0, 5>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<5, 3, 0, 5, 27>(cache, slot_for_kw, word);
            break;
        case 24:
            copy_staged_cache_chunk_segment<5, 3, 27, 0, 5>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<5, 4, 0, 5, 1>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<6, 0, 0, 6, 26>(cache, slot_for_kw, word);
            break;
        case 25:
            copy_staged_cache_chunk_segment<6, 0, 26, 0, 6>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<6, 1, 0, 6, 26>(cache, slot_for_kw, word);
            break;
        case 26:
            copy_staged_cache_chunk_segment<6, 1, 26, 0, 6>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<6, 2, 0, 6, 26>(cache, slot_for_kw, word);
            break;
        case 27:
            copy_staged_cache_chunk_segment<6, 2, 26, 0, 6>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<6, 3, 0, 6, 26>(cache, slot_for_kw, word);
            break;
        case 28:
            copy_staged_cache_chunk_segment<6, 3, 26, 0, 6>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<6, 4, 0, 6, 1>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<7, 0, 0, 7, 25>(cache, slot_for_kw, word);
            break;
        case 29:
            copy_staged_cache_chunk_segment<7, 0, 25, 0, 7>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<7, 1, 0, 7, 25>(cache, slot_for_kw, word);
            break;
        case 30:
            copy_staged_cache_chunk_segment<7, 1, 25, 0, 7>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<7, 2, 0, 7, 25>(cache, slot_for_kw, word);
            break;
        case 31:
            copy_staged_cache_chunk_segment<7, 2, 25, 0, 7>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<7, 3, 0, 7, 25>(cache, slot_for_kw, word);
            break;
        case 32:
            copy_staged_cache_chunk_segment<7, 3, 25, 0, 7>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<7, 4, 0, 7, 1>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<8, 0, 0, 8, 24>(cache, slot_for_kw, word);
            break;
        case 33:
            copy_staged_cache_chunk_segment<8, 0, 24, 0, 8>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<8, 1, 0, 8, 24>(cache, slot_for_kw, word);
            break;
        case 34:
            copy_staged_cache_chunk_segment<8, 1, 24, 0, 8>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<8, 2, 0, 8, 24>(cache, slot_for_kw, word);
            break;
        case 35:
            copy_staged_cache_chunk_segment<8, 2, 24, 0, 8>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<8, 3, 0, 8, 24>(cache, slot_for_kw, word);
            break;
        case 36:
            copy_staged_cache_chunk_segment<8, 3, 24, 0, 8>(cache, slot_for_kw, word);
            copy_staged_cache_chunk_segment<8, 4, 0, 8, 1>(cache, slot_for_kw, word);
            break;
        default:
            break;
    }
}

static void emit_narrow_words_for_mode(
    unsigned mode,
    const narrow_3x3_window_t& window,
    hls::stream<act_vec_t>& act_stream) {
#pragma HLS INLINE off
    if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C12)) {
        for (int kt = 0; kt < 4; ++kt) {
#pragma HLS PIPELINE II=1
            act_vec_t word = 0;
            build_staged_c12_word(window, kt, word);
            act_stream.write(word);
        }
        return;
    }
    if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C17)) {
        for (int kt = 0; kt < 5; ++kt) {
#pragma HLS PIPELINE II=1
            act_vec_t word = 0;
            build_staged_c17_word(window, kt, word);
            act_stream.write(word);
        }
        return;
    }
    if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C25)) {
        for (int kt = 0; kt < 8; ++kt) {
#pragma HLS PIPELINE II=1
            act_vec_t word = 0;
            build_staged_c25_word(window, kt, word);
            act_stream.write(word);
        }
        return;
    }
}

static void emit_narrow_word_pair_for_mode(
    unsigned mode,
    const narrow_3x3_window_t& window0,
    const narrow_3x3_window_t& window1,
    hls::stream<act_vec_t>& act_stream0,
    hls::stream<act_vec_t>& act_stream1) {
#pragma HLS INLINE off
    const int k_tiles = (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C12))
                                  ? 4
                                  : (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C17)) ? 5 : 8;
    for (int kt = 0; kt < 8; ++kt) {
#pragma HLS PIPELINE II=1
        if (kt >= k_tiles) {
            break;
        }
        act_vec_t word0 = 0;
        act_vec_t word1 = 0;
        if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C12)) {
            build_staged_c12_word(window0, kt, word0);
            build_staged_c12_word(window1, kt, word1);
        } else if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C17)) {
            build_staged_c17_word(window0, kt, word0);
            build_staged_c17_word(window1, kt, word1);
        } else {
            build_staged_c25_word(window0, kt, word0);
            build_staged_c25_word(window1, kt, word1);
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
    if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C129)) {
        for (int kt = 0; kt < 37; ++kt) {
            act_vec_t word = 0;
            build_staged_c129_word(cache, slot_for_kw, kt, word);
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

struct narrow_incremental_column_t {
    act_vec_t row0;
    act_vec_t row1;
    act_vec_t row2;
};

struct narrow_incremental_cache_t {
    narrow_incremental_column_t col0;
    narrow_incremental_column_t col1;
    narrow_incremental_column_t col2;
    narrow_incremental_column_t col3;
};

static void window_assembler_read_incremental_column(
    hls::stream<window_load_word_t>& load_word_stream,
    narrow_incremental_column_t& column) {
#pragma HLS INLINE
    const window_load_word_t row0 = load_word_stream.read();
    const window_load_word_t row1 = load_word_stream.read();
    const window_load_word_t row2 = load_word_stream.read();
    column.row0 = row0.word;
    column.row1 = row1.word;
    column.row2 = row2.word;
}

static void window_assembler_clear_narrow_window(
    narrow_3x3_window_t& window) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=window.spatial complete dim=1
    for (int spatial = 0; spatial < 9; ++spatial) {
#pragma HLS UNROLL
        window.spatial[spatial] = 0;
    }
}

template <int KW>
static void window_assembler_set_narrow_column(
    narrow_3x3_window_t& window,
    const narrow_incremental_column_t& column) {
#pragma HLS INLINE
    window.spatial[KW] = column.row0;
    window.spatial[3 + KW] = column.row1;
    window.spatial[6 + KW] = column.row2;
}

static void window_assembler_consume_c12_incremental(
    int issue,
    bool has_second_window,
    hls::stream<window_load_word_t>& load_word_stream,
    narrow_incremental_cache_t& cache,
    narrow_3x3_window_t& window0,
    narrow_3x3_window_t& window1) {
#pragma HLS INLINE
    if (issue == 0) {
        window_assembler_read_incremental_column(load_word_stream, cache.col0);
        window_assembler_read_incremental_column(load_word_stream, cache.col1);
        window_assembler_read_incremental_column(load_word_stream, cache.col2);
        window_assembler_read_incremental_column(load_word_stream, cache.col3);
    } else {
        cache.col0 = cache.col2;
        cache.col1 = cache.col3;
        window_assembler_read_incremental_column(load_word_stream, cache.col2);
        window_assembler_read_incremental_column(load_word_stream, cache.col3);
    }
    window_assembler_set_narrow_column<0>(window0, cache.col0);
    window_assembler_set_narrow_column<1>(window0, cache.col1);
    window_assembler_set_narrow_column<2>(window0, cache.col2);
    if (has_second_window) {
        window_assembler_set_narrow_column<0>(window1, cache.col1);
        window_assembler_set_narrow_column<1>(window1, cache.col2);
        window_assembler_set_narrow_column<2>(window1, cache.col3);
    }
}

static void window_assembler_store_delta_word(const window_load_word_t& token) {
#pragma HLS INLINE
    const unsigned slot = token.slot.to_uint();
    if (token.row.to_uint() == 0U) {
        s_delta_cache.data[0][slot] = token.word;
    } else if (token.row.to_uint() == 1U) {
        s_delta_cache.data[1][slot] = token.word;
    } else {
        s_delta_cache.data[2][slot] = token.word;
    }
}

template <int KW>
static void window_assembler_snapshot_delta_column(u8_t slot, narrow_3x3_window_t& window) {
#pragma HLS INLINE
    window.spatial[KW] = s_delta_cache.data[0][slot.to_uint()];
    window.spatial[3 + KW] = s_delta_cache.data[1][slot.to_uint()];
    window.spatial[6 + KW] = s_delta_cache.data[2][slot.to_uint()];
}

static void window_assembler_consume_delta_phase(
    hls::stream<window_column_meta_t>& column_meta_stream,
    hls::stream<window_load_word_t>& load_word_stream,
    narrow_3x3_window_t& window) {
#pragma HLS INLINE
    u8_t slots[3];
#pragma HLS ARRAY_PARTITION variable=slots complete dim=1
    for (int kw = 0; kw < 3; ++kw) {
#pragma HLS PIPELINE off
        const window_column_meta_t meta = column_meta_stream.read();
        slots[kw] = meta.slot;
        if (meta.update.to_uint() != 0U) {
            for (int row = 0; row < 3; ++row) {
#pragma HLS PIPELINE II=1
                const window_load_word_t token = load_word_stream.read();
                window_assembler_store_delta_word(token);
            }
        }
    }
    window_assembler_snapshot_delta_column<0>(slots[0], window);
    window_assembler_snapshot_delta_column<1>(slots[1], window);
    window_assembler_snapshot_delta_column<2>(slots[2], window);
}

static void window_row_assembler(
    hls::stream<window_assembler_cfg_t>& assembler_cfg_stream,
    hls::stream<window_column_meta_t>& column_meta_stream,
    hls::stream<window_load_word_t>& load_word_stream,
    hls::stream<act_vec_t>& act_stream0,
    hls::stream<act_vec_t>& act_stream1) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=s_wide_cache.data complete dim=1
#pragma HLS ARRAY_PARTITION variable=s_wide_cache.data complete dim=3
#pragma HLS BIND_STORAGE variable=s_wide_cache.data type=ram_2p impl=lutram
#pragma HLS RESET variable=s_wide_cache off
#pragma HLS ARRAY_PARTITION variable=s_delta_cache.data complete dim=1
#pragma HLS BIND_STORAGE variable=s_delta_cache.data type=ram_1p impl=bram
#pragma HLS RESET variable=s_delta_cache off

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
    narrow_incremental_cache_t incremental_cache;

    for (int issue = 0; issue < MAX_FM_W; ++issue) {
        if (issue >= issue_count) {
            break;
        }
        if (narrow) {
            const bool has_second_window =
                paired && issue * 2 + 1 < out_w;
            narrow_3x3_window_t window0;
            narrow_3x3_window_t window1;
#pragma HLS ARRAY_PARTITION variable=window0.spatial complete dim=1
#pragma HLS ARRAY_PARTITION variable=window1.spatial complete dim=1
            window_assembler_clear_narrow_window(window0);
            window_assembler_clear_narrow_window(window1);
            if (cfg.narrow_transfer_protocol.to_uint() ==
                static_cast<unsigned>(
                    WINDOW_TRANSFER_C12_S1_INCREMENTAL)) {
                window_assembler_consume_c12_incremental(
                    issue,
                    has_second_window,
                    load_word_stream,
                    incremental_cache,
                    window0,
                    window1);
            } else {
                window_assembler_consume_delta_phase(column_meta_stream, load_word_stream, window0);
                if (paired) {
                    // Snapshot phase 0 before phase 1 can replace an aliased ring slot.
                    window_assembler_consume_delta_phase(column_meta_stream, load_word_stream, window1);
                    if (!has_second_window) {
                        window_assembler_clear_narrow_window(window1);
                    }
                }
            }
            if (!paired) {
                emit_narrow_words_for_mode(
                    mode, window0, act_stream0);
            } else {
                emit_narrow_word_pair_for_mode(
                    mode,
                    window0,
                    window1,
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
    hls::stream<window_load_word_t> load_word_stream;
#pragma HLS STREAM variable=loader_cfg_stream depth=2
#pragma HLS STREAM variable=assembler_cfg_stream depth=3
#pragma HLS STREAM variable=column_meta_stream depth=16
#pragma HLS STREAM variable=load_word_stream depth=16
#pragma HLS BIND_STORAGE variable=assembler_cfg_stream type=fifo impl=lutram
#pragma HLS BIND_STORAGE variable=column_meta_stream type=fifo impl=lutram
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
        load_word_stream);
    window_row_assembler(
        assembler_cfg_stream,
        column_meta_stream,
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

template <int ROW>
static void c1_load_packed_source_row(const tensor_desc_t& src, int source_row) {
#pragma HLS INLINE
    const bool valid = source_row >= 0 && source_row < INPUT_FRAME_H;
    const unsigned row_byte = src.base_offset.to_uint() +
                              static_cast<unsigned>(valid ? source_row : 0) * INPUT_FRAME_W;
    const unsigned phase = row_byte & 31U;
    const unsigned aligned = row_byte & ~31U;
    act_vec_t previous = 0;
    for (int wi = 0; wi < WINGEN_C1_PACKED_ROW_WORDS; ++wi) {
#pragma HLS PIPELINE off
        act_vec_t current = 0;
        if (valid) {
            on_chip_memory_read_aligned_tensor_word(src,
                static_cast<u32_t>(aligned + static_cast<unsigned>(wi) * AXI_WORD_BYTES), current);
        }
        if (phase == 0U) {
            s_c1_rows[ROW][wi] = current;
        } else if (wi != 0) {
            s_c1_rows[ROW][wi - 1] = (previous >> 128) | (current << 128);
        }
        previous = current;
    }
    if (phase != 0U) {
        s_c1_rows[ROW][WINGEN_C1_PACKED_ROW_WORDS - 1] = previous >> 128;
    }
    // 240 real bytes followed by 16 local zeros, including the last raw frame row.
    s_c1_rows[ROW][WINGEN_C1_PACKED_ROW_WORDS - 1].range(255, 128) = 0;
}

template <int ROW>
static void c1_pack_window_pair(int first_col, act_vec_t& word0, act_vec_t& word1) {
#pragma HLS INLINE
    const unsigned col = static_cast<unsigned>(first_col < 0 ? 0 : first_col);
    const unsigned wi = col >> 5;
    const unsigned offset = col & 31U;
    ap_uint<40> bytes = s_c1_rows[ROW][wi] >> (offset * 8U);
    if (offset > 27U && wi + 1U < static_cast<unsigned>(WINGEN_C1_PACKED_ROW_WORDS)) {
        bytes |= ap_uint<40>(s_c1_rows[ROW][wi + 1U]) << ((32U - offset) * 8U);
    }
    if (first_col < 0) {
        bytes <<= 8;
    }
    word0.range((ROW * 3 + 0) * 8 + 7, (ROW * 3 + 0) * 8) = bytes.range(7, 0);
    word0.range((ROW * 3 + 1) * 8 + 7, (ROW * 3 + 1) * 8) = bytes.range(15, 8);
    word0.range((ROW * 3 + 2) * 8 + 7, (ROW * 3 + 2) * 8) = bytes.range(23, 16);
    word1.range((ROW * 3 + 0) * 8 + 7, (ROW * 3 + 0) * 8) = bytes.range(23, 16);
    word1.range((ROW * 3 + 1) * 8 + 7, (ROW * 3 + 1) * 8) = bytes.range(31, 24);
    word1.range((ROW * 3 + 2) * 8 + 7, (ROW * 3 + 2) * 8) = bytes.range(39, 32);
}

static void scheduled_c1_packed_row(const tensor_desc_t& src, u16_t out_row,
                                    hls::stream<act_vec_t>& act_stream0,
                                    hls::stream<act_vec_t>& act_stream1) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=s_c1_rows complete dim=1
#pragma HLS BIND_STORAGE variable=s_c1_rows type=ram_2p impl=lutram
#pragma HLS RESET variable=s_c1_rows off
    const bool continuous = s_c1_rows_valid && out_row != 0 &&
                            out_row.to_uint() == s_c1_last_row.to_uint() + 1U &&
                            src.base_offset == s_c1_source_base &&
                            src.bank_id.to_uint() == s_c1_source_bank.to_uint();
    const int base_h = static_cast<int>(out_row.to_uint()) * 2 - 1;
    if (continuous) {
        for (int wi = 0; wi < WINGEN_C1_PACKED_ROW_WORDS; ++wi) {
#pragma HLS PIPELINE II=1
            s_c1_rows[0][wi] = s_c1_rows[2][wi];
        }
    } else {
        c1_load_packed_source_row<0>(src, base_h);
    }
    c1_load_packed_source_row<1>(src, base_h + 1);
    c1_load_packed_source_row<2>(src, base_h + 2);
    s_c1_source_base = src.base_offset;
    s_c1_source_bank = static_cast<u8_t>(src.bank_id.to_uint());
    s_c1_last_row = out_row;
    s_c1_rows_valid = true;
    for (int issue = 0; issue < INPUT_FRAME_W / 4; ++issue) {
#pragma HLS PIPELINE II=1
        act_vec_t word0 = 0;
        act_vec_t word1 = 0;
        c1_pack_window_pair<0>(issue * 4 - 1, word0, word1);
        c1_pack_window_pair<1>(issue * 4 - 1, word0, word1);
        c1_pack_window_pair<2>(issue * 4 - 1, word0, word1);
        act_stream0.write(word0);
        act_stream1.write(word1);
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
    if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C1)) {
        scheduled_c1_packed_row(src_desc, out_row, act_stream0, act_stream1);
    } else if (loader_class == static_cast<unsigned>(WIN_LOADER_3X3_NARROW) ||
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
