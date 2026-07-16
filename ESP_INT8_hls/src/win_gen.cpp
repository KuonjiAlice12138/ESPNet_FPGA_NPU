#include "../include/npu_config.hpp"
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

bool on_chip_memory_read_packed_tile_from_row(const tensor_desc_t& desc,
                                              u32_t row_base,
                                              bool row_valid,
                                              i32_t w,
                                              u16_t c_begin,
                                              u8_t valid_c,
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

static act_vec_t s_narrow_cache_row0[WINGEN_NARROW_CACHE_COL_SLOTS];
static act_vec_t s_narrow_cache_row1[WINGEN_NARROW_CACHE_COL_SLOTS];
static act_vec_t s_narrow_cache_row2[WINGEN_NARROW_CACHE_COL_SLOTS];

struct wide_3x3_cache_t {
    act_vec_t data[3][WINGEN_WIDE_CACHE_COL_SLOTS][MAX_3X3_CACHE_CHUNKS];
};

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

static void load_narrow_cache_column(const tensor_desc_t& src_desc,
                                     const u32_t row_base[3],
                                     const bool row_valid[3],
                                     i32_t input_col,
                                     u8_t slot,
                                     const window_sched_desc_t& sched) {
#pragma HLS INLINE off
    const int slot_i = static_cast<int>(slot.to_uint());
    u8_t valid_c = 0;
    valid_c.range(7, 0) = sched.in_c.range(7, 0);
    for (int kh = 0; kh < 3; ++kh) {
#pragma HLS PIPELINE off
        act_vec_t word = 0;
        on_chip_memory_read_packed_tile_from_row(src_desc,
                                                 row_base[kh],
                                                 row_valid[kh],
                                                 input_col,
                                                 static_cast<u16_t>(0),
                                                 valid_c,
                                                 word);
        if (kh == 0) {
            s_narrow_cache_row0[slot_i] = word;
        } else if (kh == 1) {
            s_narrow_cache_row1[slot_i] = word;
        } else {
            s_narrow_cache_row2[slot_i] = word;
        }
    }
}

static void load_wide_cache_column(const tensor_desc_t& src_desc,
                                   const u32_t row_base[3],
                                   const bool row_valid[3],
                                   i32_t input_col,
                                   u8_t slot,
                                   const window_sched_desc_t& sched,
                                   wide_3x3_cache_t& cache) {
#pragma HLS INLINE off
    const int slot_i = static_cast<int>(slot.to_uint());
    for (int kh = 0; kh < 3; ++kh) {
#pragma HLS PIPELINE off
        for (int chunk = 0; chunk < MAX_3X3_CACHE_CHUNKS; ++chunk) {
#pragma HLS PIPELINE off
            act_vec_t word = 0;
            if (chunk < static_cast<int>(sched.cache_chunks.to_uint())) {
                const unsigned c_begin_u = static_cast<unsigned>(chunk * TK);
                unsigned valid = sched.in_c.to_uint() - c_begin_u;
                if (valid > static_cast<unsigned>(TK)) {
                    valid = static_cast<unsigned>(TK);
                }
                on_chip_memory_read_packed_tile_from_row(src_desc,
                                                         row_base[kh],
                                                         row_valid[kh],
                                                         input_col,
                                                         static_cast<u16_t>(c_begin_u),
                                                         static_cast<u8_t>(valid),
                                                         word);
            }
            cache.data[kh][slot_i][chunk] = word;
        }
    }
}

static void update_narrow_direct_cache(const tensor_desc_t& src_desc,
                                       const window_sched_desc_t& sched,
                                       const u32_t row_base[3],
                                       const bool row_valid[3],
                                       const i32_t want_col[3],
                                       i32_t col_tag[WINGEN_NARROW_CACHE_COL_SLOTS],
                                       u8_t slot_for_kw[3]) {
#pragma HLS INLINE off
    const unsigned slot_mask = sched.cache_col_slots.to_uint() - 1U;
    for (int kw = 0; kw < 3; ++kw) {
#pragma HLS PIPELINE off
        const unsigned slot = static_cast<unsigned>(want_col[kw].to_int()) & slot_mask;
        slot_for_kw[kw] = static_cast<u8_t>(slot);
        if (col_tag[slot] != want_col[kw]) {
            load_narrow_cache_column(src_desc,
                                     row_base,
                                     row_valid,
                                     want_col[kw],
                                     static_cast<u8_t>(slot),
                                     sched);
            col_tag[slot] = want_col[kw];
        }
    }
}

static void stage_narrow_3x3_window(const u8_t slot_for_kw[3],
                                    narrow_3x3_window_t& window) {
#pragma HLS INLINE off
    for (int kw = 0; kw < 3; ++kw) {
#pragma HLS PIPELINE II=1
        const int slot = static_cast<int>(slot_for_kw[kw].to_uint());
        window.spatial[kw] = s_narrow_cache_row0[slot];
        window.spatial[3 + kw] = s_narrow_cache_row1[slot];
        window.spatial[6 + kw] = s_narrow_cache_row2[slot];
    }
}

static void update_wide_direct_cache(const tensor_desc_t& src_desc,
                                     const window_sched_desc_t& sched,
                                     const u32_t row_base[3],
                                     const bool row_valid[3],
                                     const i32_t want_col[3],
                                     i32_t col_tag[WINGEN_WIDE_CACHE_COL_SLOTS],
                                     u8_t slot_for_kw[3],
                                     wide_3x3_cache_t& cache) {
#pragma HLS INLINE off
    for (int kw = 0; kw < 3; ++kw) {
#pragma HLS PIPELINE off
        const unsigned slot = static_cast<unsigned>(want_col[kw].to_int()) &
                              static_cast<unsigned>(WINGEN_WIDE_CACHE_COL_SLOTS - 1);
        slot_for_kw[kw] = static_cast<u8_t>(slot);
        if (col_tag[slot] != want_col[kw]) {
            load_wide_cache_column(src_desc,
                                   row_base,
                                   row_valid,
                                   want_col[kw],
                                   static_cast<u8_t>(slot),
                                   sched,
                                   cache);
            col_tag[slot] = want_col[kw];
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

static void scheduled_narrow_3x3_window_row(const tensor_desc_t& src_desc,
                                            const window_sched_desc_t& sched,
                                            unsigned mode,
                                            hls::stream<act_vec_t>& act_stream0,
                                            hls::stream<act_vec_t>& act_stream1,
                                            u16_t out_row) {
#pragma HLS INLINE off
#pragma HLS BIND_STORAGE variable=s_narrow_cache_row0 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=s_narrow_cache_row1 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=s_narrow_cache_row2 type=ram_2p impl=bram
#pragma HLS RESET variable=s_narrow_cache_row0 off
#pragma HLS RESET variable=s_narrow_cache_row1 off
#pragma HLS RESET variable=s_narrow_cache_row2 off

    i32_t col_tag[WINGEN_NARROW_CACHE_COL_SLOTS];
#pragma HLS BIND_STORAGE variable=col_tag type=ram_2p impl=lutram
    u8_t slot_for_kw[3];
#pragma HLS ARRAY_PARTITION variable=slot_for_kw complete dim=1

    for (int slot = 0; slot < WINGEN_NARROW_CACHE_COL_SLOTS; ++slot) {
#pragma HLS PIPELINE II=1
        col_tag[slot] = static_cast<i32_t>(-32768);
    }

    const int out_w_i = static_cast<int>(sched.out_w.to_uint());
    const int stride_i = static_cast<int>(sched.stride.to_uint());
    const int dilation_i = static_cast<int>(sched.dilation.to_uint());
    const int padding_i = static_cast<int>(sched.padding.to_uint());
    const i32_t base_h =
        static_cast<i32_t>(out_row.to_uint() * static_cast<unsigned>(stride_i)) -
        static_cast<i32_t>(padding_i);
    u32_t row_base[3];
    bool row_valid[3];
#pragma HLS ARRAY_PARTITION variable=row_base complete dim=1
#pragma HLS ARRAY_PARTITION variable=row_valid complete dim=1
    for (int kh = 0; kh < 3; ++kh) {
#pragma HLS UNROLL
        on_chip_memory_prepare_row_base(src_desc,
                                        base_h + static_cast<i32_t>(kh * dilation_i),
                                        row_base[kh],
                                        row_valid[kh]);
    }

    const bool paired =
        (sched.flags.to_uint() & static_cast<unsigned>(WINDOW_SCHED_FLAG_PIXEL_PARALLEL_2)) != 0U;
    const bool odd_tail =
        (sched.flags.to_uint() & static_cast<unsigned>(WINDOW_SCHED_FLAG_ODD_TAIL)) != 0U;
    const int issue_count = paired ? ((out_w_i + 1) / 2) : out_w_i;
    for (int issue_i = 0; issue_i < MAX_FM_W; ++issue_i) {
        if (issue_i >= issue_count) {
            break;
        }
        const int pixel0 = paired ? issue_i * 2 : issue_i;
        const i32_t base_w0 =
            static_cast<i32_t>(pixel0 * stride_i) - static_cast<i32_t>(padding_i);
        i32_t want_col0[3];
#pragma HLS ARRAY_PARTITION variable=want_col0 complete dim=1
        for (int kw = 0; kw < 3; ++kw) {
#pragma HLS UNROLL
            want_col0[kw] = base_w0 + static_cast<i32_t>(kw * dilation_i);
        }
        update_narrow_direct_cache(src_desc, sched, row_base, row_valid, want_col0, col_tag, slot_for_kw);
        narrow_3x3_window_t window0;
#pragma HLS ARRAY_PARTITION variable=window0.spatial complete dim=1
        stage_narrow_3x3_window(slot_for_kw, window0);

        if (!paired) {
            emit_narrow_words_for_mode(mode, window0, slot_for_kw, act_stream0);
            continue;
        }

        u8_t slot_for_kw1[3];
#pragma HLS ARRAY_PARTITION variable=slot_for_kw1 complete dim=1
        narrow_3x3_window_t window1;
#pragma HLS ARRAY_PARTITION variable=window1.spatial complete dim=1
        const bool tail_issue = odd_tail && issue_i == issue_count - 1;
        if (!tail_issue) {
            const i32_t base_w1 =
                static_cast<i32_t>((pixel0 + 1) * stride_i) - static_cast<i32_t>(padding_i);
            i32_t want_col1[3];
#pragma HLS ARRAY_PARTITION variable=want_col1 complete dim=1
            for (int kw = 0; kw < 3; ++kw) {
#pragma HLS UNROLL
                want_col1[kw] = base_w1 + static_cast<i32_t>(kw * dilation_i);
            }
            update_narrow_direct_cache(src_desc, sched, row_base, row_valid, want_col1, col_tag, slot_for_kw1);
            stage_narrow_3x3_window(slot_for_kw1, window1);
        } else {
            for (int kw = 0; kw < 3; ++kw) {
#pragma HLS UNROLL
                slot_for_kw1[kw] = 0;
            }
            for (int spatial = 0; spatial < 9; ++spatial) {
#pragma HLS UNROLL
                window1.spatial[spatial] = 0;
            }
        }
        emit_narrow_word_pair_for_mode(
            mode, window0, window1, slot_for_kw, slot_for_kw1, act_stream0, act_stream1);
    }
}

static void scheduled_wide_3x3_window_row(const tensor_desc_t& src_desc,
                                          const window_sched_desc_t& sched,
                                          unsigned mode,
                                          hls::stream<act_vec_t>& act_stream,
                                          u16_t out_row) {
#pragma HLS INLINE off
    wide_3x3_cache_t cache;
#pragma HLS ARRAY_PARTITION variable=cache.data complete dim=1
#pragma HLS BIND_STORAGE variable=cache.data type=ram_2p impl=lutram
    i32_t col_tag[WINGEN_WIDE_CACHE_COL_SLOTS];
#pragma HLS ARRAY_PARTITION variable=col_tag complete dim=1
    u8_t slot_for_kw[3];
#pragma HLS ARRAY_PARTITION variable=slot_for_kw complete dim=1

    for (int slot = 0; slot < WINGEN_WIDE_CACHE_COL_SLOTS; ++slot) {
#pragma HLS UNROLL
        col_tag[slot] = static_cast<i32_t>(-32768);
    }

    const int out_w_i = static_cast<int>(sched.out_w.to_uint());
    const int stride_i = static_cast<int>(sched.stride.to_uint());
    const int dilation_i = static_cast<int>(sched.dilation.to_uint());
    const int padding_i = static_cast<int>(sched.padding.to_uint());
    const i32_t base_h =
        static_cast<i32_t>(out_row.to_uint() * static_cast<unsigned>(stride_i)) -
        static_cast<i32_t>(padding_i);
    u32_t row_base[3];
    bool row_valid[3];
#pragma HLS ARRAY_PARTITION variable=row_base complete dim=1
#pragma HLS ARRAY_PARTITION variable=row_valid complete dim=1
    for (int kh = 0; kh < 3; ++kh) {
#pragma HLS UNROLL
        on_chip_memory_prepare_row_base(src_desc,
                                        base_h + static_cast<i32_t>(kh * dilation_i),
                                        row_base[kh],
                                        row_valid[kh]);
    }

    for (int ow_i = 0; ow_i < MAX_FM_W; ++ow_i) {
        if (ow_i >= out_w_i) {
            break;
        }
        const i32_t base_w =
            static_cast<i32_t>(ow_i * stride_i) - static_cast<i32_t>(padding_i);
        i32_t want_col[3];
#pragma HLS ARRAY_PARTITION variable=want_col complete dim=1
        for (int kw = 0; kw < 3; ++kw) {
#pragma HLS UNROLL
            want_col[kw] = base_w + static_cast<i32_t>(kw * dilation_i);
        }
        update_wide_direct_cache(src_desc,
                                 sched,
                                 row_base,
                                 row_valid,
                                 want_col,
                                 col_tag,
                                 slot_for_kw,
                                 cache);
        emit_wide_words_for_mode(mode, cache, slot_for_kw, act_stream);
    }
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
    const int issue_count = paired ? ((out_w_i + 1) / 2) : out_w_i;
    for (int issue_i = 0; issue_i < MAX_FM_W; ++issue_i) {
        if (issue_i >= issue_count) {
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
                const bool tail_issue = odd_tail && issue_i == issue_count - 1;
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
                                    const conv_exec_desc_t&,
                                    const window_sched_desc_t& sched,
                                    hls::stream<act_vec_t>& act_stream0,
                                    hls::stream<act_vec_t>& act_stream1,
                                    u16_t out_row) {
#pragma HLS INLINE off
    const unsigned mode = sched.mode.to_uint();
    if (mode >= static_cast<unsigned>(WIN_MODE_3X3_STAGED_C3) &&
        mode <= static_cast<unsigned>(WIN_MODE_3X3_STAGED_C25)) {
        scheduled_narrow_3x3_window_row(src_desc, sched, mode, act_stream0, act_stream1, out_row);
        return;
    }
    if (mode >= static_cast<unsigned>(WIN_MODE_3X3_STAGED_C131) &&
        mode <= static_cast<unsigned>(WIN_MODE_3X3_STAGED_C128)) {
        scheduled_wide_3x3_window_row(src_desc, sched, mode, act_stream0, out_row);
        return;
    }
    if (mode == static_cast<unsigned>(WIN_MODE_1X1_ALIGNED)) {
        scheduled_1x1_window_row(src_desc, sched, act_stream0, act_stream1, out_row, true);
        return;
    }
    if (mode == static_cast<unsigned>(WIN_MODE_1X1_PACKED)) {
        scheduled_1x1_window_row(src_desc, sched, act_stream0, act_stream1, out_row, false);
        return;
    }
}

}  // namespace esp_int8
