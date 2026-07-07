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

static void read_pixel_chunks_or_zero(const tensor_desc_t& desc,
                                      i32_t ih,
                                      i32_t iw,
                                      u16_t in_c,
                                      u8_t cache_chunks,
                                      act_vec_t out_chunks[MAX_3X3_CACHE_CHUNKS]) {
#pragma HLS INLINE off
    const int chunk_count = static_cast<int>(cache_chunks.to_uint());
    for (int chunk = 0; chunk < MAX_3X3_CACHE_CHUNKS; ++chunk) {
#pragma HLS PIPELINE off
        act_vec_t word = 0;
        if (chunk < chunk_count) {
            const unsigned c_begin_u = static_cast<unsigned>(chunk * TK);
            unsigned valid = 0U;
            if (c_begin_u < in_c.to_uint()) {
                valid = in_c.to_uint() - c_begin_u;
                if (valid > static_cast<unsigned>(TK)) {
                    valid = static_cast<unsigned>(TK);
                }
            }
            if (valid != 0U) {
                word = read_tile_or_zero(desc,
                                         ih,
                                         iw,
                                         static_cast<u16_t>(c_begin_u),
                                         static_cast<u8_t>(valid));
            }
        }
        out_chunks[chunk] = word;
    }
}

static void load_3x3_cache_column(const tensor_desc_t& src_desc,
                                  i32_t base_h,
                                  i32_t input_col,
                                  u8_t slot,
                                  const window_sched_desc_t& sched,
                                  act_vec_t cache[3][WINGEN_CACHE_COL_SLOTS][MAX_3X3_CACHE_CHUNKS]) {
#pragma HLS INLINE off
    const int slot_i = static_cast<int>(slot.to_uint());
    const int dilation_i = static_cast<int>(sched.dilation.to_uint());
    for (int kh = 0; kh < 3; ++kh) {
#pragma HLS PIPELINE off
        const i32_t ih = base_h + static_cast<i32_t>(kh * dilation_i);
        read_pixel_chunks_or_zero(src_desc,
                                  ih,
                                  input_col,
                                  sched.in_c,
                                  sched.cache_chunks,
                                  cache[kh][slot_i]);
    }
}

static int find_cached_col(const i32_t col_tag[WINGEN_CACHE_COL_SLOTS], i32_t col) {
#pragma HLS INLINE
    int hit = -1;
    for (int slot = 0; slot < WINGEN_CACHE_COL_SLOTS; ++slot) {
#pragma HLS UNROLL
        if (col_tag[slot] == col) {
            hit = slot;
        }
    }
    return hit;
}

static int select_replacement_slot(const bool used[WINGEN_CACHE_COL_SLOTS]) {
#pragma HLS INLINE
    int chosen = 0;
    for (int slot = 0; slot < WINGEN_CACHE_COL_SLOTS; ++slot) {
#pragma HLS UNROLL
        if (!used[slot]) {
            chosen = slot;
            break;
        }
    }
    return chosen;
}

static void update_3x3_column_cache(const tensor_desc_t& src_desc,
                                    const window_sched_desc_t& sched,
                                    i32_t base_h,
                                    i32_t want_col[3],
                                    i32_t col_tag[WINGEN_CACHE_COL_SLOTS],
                                    u8_t slot_for_kw[3],
                                    act_vec_t cache[3][WINGEN_CACHE_COL_SLOTS][MAX_3X3_CACHE_CHUNKS]) {
#pragma HLS INLINE off
    bool used[WINGEN_CACHE_COL_SLOTS];
#pragma HLS ARRAY_PARTITION variable=used complete dim=1

    for (int slot = 0; slot < WINGEN_CACHE_COL_SLOTS; ++slot) {
#pragma HLS UNROLL
        used[slot] = false;
    }
    for (int kw = 0; kw < 3; ++kw) {
#pragma HLS UNROLL
        slot_for_kw[kw] = static_cast<u8_t>(0xff);
    }

    for (int kw = 0; kw < 3; ++kw) {
#pragma HLS UNROLL
        const int hit = find_cached_col(col_tag, want_col[kw]);
        if (hit >= 0) {
            slot_for_kw[kw] = static_cast<u8_t>(hit);
            used[hit] = true;
        }
    }

    for (int kw = 0; kw < 3; ++kw) {
#pragma HLS PIPELINE off
        if (slot_for_kw[kw].to_uint() == 0xffU) {
            const int slot = select_replacement_slot(used);
            used[slot] = true;
            col_tag[slot] = want_col[kw];
            slot_for_kw[kw] = static_cast<u8_t>(slot);
            load_3x3_cache_column(src_desc,
                                  base_h,
                                  want_col[kw],
                                  static_cast<u8_t>(slot),
                                  sched,
                                  cache);
        }
    }
}

template <int SPATIAL, int SRC_LANE, int DST_LANE, int COUNT>
static void copy_staged_cache_segment(
    const act_vec_t cache[3][WINGEN_CACHE_COL_SLOTS][MAX_3X3_CACHE_CHUNKS],
    const u8_t slot_for_kw[3],
    act_vec_t& word) {
#pragma HLS INLINE
    const int kh = SPATIAL / 3;
    const int kw = SPATIAL - kh * 3;
    const int slot = static_cast<int>(slot_for_kw[kw].to_uint());
    const act_vec_t src = cache[kh][slot][0];
    word.range(DST_LANE * 8 + COUNT * 8 - 1, DST_LANE * 8) =
        src.range(SRC_LANE * 8 + COUNT * 8 - 1, SRC_LANE * 8);
}

template <int SPATIAL, int CHUNK, int SRC_LANE, int DST_LANE, int COUNT>
static void copy_staged_cache_chunk_segment(
    const act_vec_t cache[3][WINGEN_CACHE_COL_SLOTS][MAX_3X3_CACHE_CHUNKS],
    const u8_t slot_for_kw[3],
    act_vec_t& word) {
#pragma HLS INLINE
    const int kh = SPATIAL / 3;
    const int kw = SPATIAL - kh * 3;
    const int slot = static_cast<int>(slot_for_kw[kw].to_uint());
    const act_vec_t src = cache[kh][slot][CHUNK];
    word.range(DST_LANE * 8 + COUNT * 8 - 1, DST_LANE * 8) =
        src.range(SRC_LANE * 8 + COUNT * 8 - 1, SRC_LANE * 8);
}

static void build_staged_c3_word(
    const act_vec_t cache[3][WINGEN_CACHE_COL_SLOTS][MAX_3X3_CACHE_CHUNKS],
    const u8_t slot_for_kw[3],
    int,
    act_vec_t& word) {
#pragma HLS INLINE off
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

static void build_staged_c12_word(
    const act_vec_t cache[3][WINGEN_CACHE_COL_SLOTS][MAX_3X3_CACHE_CHUNKS],
    const u8_t slot_for_kw[3],
    int kt,
    act_vec_t& word) {
#pragma HLS INLINE off
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

static void build_staged_c19_word(
    const act_vec_t cache[3][WINGEN_CACHE_COL_SLOTS][MAX_3X3_CACHE_CHUNKS],
    const u8_t slot_for_kw[3],
    int kt,
    act_vec_t& word) {
#pragma HLS INLINE off
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

static void build_staged_c25_word(
    const act_vec_t cache[3][WINGEN_CACHE_COL_SLOTS][MAX_3X3_CACHE_CHUNKS],
    const u8_t slot_for_kw[3],
    int kt,
    act_vec_t& word) {
#pragma HLS INLINE off
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

static void build_staged_c28_word(
    const act_vec_t cache[3][WINGEN_CACHE_COL_SLOTS][MAX_3X3_CACHE_CHUNKS],
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

static void build_staged_c64_word(
    const act_vec_t cache[3][WINGEN_CACHE_COL_SLOTS][MAX_3X3_CACHE_CHUNKS],
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

static void build_staged_c128_word(
    const act_vec_t cache[3][WINGEN_CACHE_COL_SLOTS][MAX_3X3_CACHE_CHUNKS],
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

static void build_staged_c131_word(
    const act_vec_t cache[3][WINGEN_CACHE_COL_SLOTS][MAX_3X3_CACHE_CHUNKS],
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

static void build_staged_word_for_mode(
    unsigned mode,
    const act_vec_t cache[3][WINGEN_CACHE_COL_SLOTS][MAX_3X3_CACHE_CHUNKS],
    const u8_t slot_for_kw[3],
    int kt,
    act_vec_t& word) {
#pragma HLS INLINE off
    if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C3)) {
        build_staged_c3_word(cache, slot_for_kw, kt, word);
        return;
    }
    if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C12)) {
        build_staged_c12_word(cache, slot_for_kw, kt, word);
        return;
    }
    if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C19)) {
        build_staged_c19_word(cache, slot_for_kw, kt, word);
        return;
    }
    if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C25)) {
        build_staged_c25_word(cache, slot_for_kw, kt, word);
        return;
    }
    if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C28)) {
        build_staged_c28_word(cache, slot_for_kw, kt, word);
        return;
    }
    if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C64)) {
        build_staged_c64_word(cache, slot_for_kw, kt, word);
        return;
    }
    if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C128)) {
        build_staged_c128_word(cache, slot_for_kw, kt, word);
        return;
    }
    if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C131)) {
        build_staged_c131_word(cache, slot_for_kw, kt, word);
        return;
    }
}

static void scheduled_3x3_staged_window_row(const tensor_desc_t& src_desc,
                                            const window_sched_desc_t& sched,
                                            unsigned mode,
                                            hls::stream<act_vec_t>& act_stream,
                                            u16_t out_row) {
#pragma HLS INLINE off
    act_vec_t cache[3][WINGEN_CACHE_COL_SLOTS][MAX_3X3_CACHE_CHUNKS];
#pragma HLS BIND_STORAGE variable=cache type=ram_2p impl=lutram

    i32_t col_tag[WINGEN_CACHE_COL_SLOTS];
#pragma HLS ARRAY_PARTITION variable=col_tag complete dim=1
    u8_t slot_for_kw[3];
#pragma HLS ARRAY_PARTITION variable=slot_for_kw complete dim=1

    for (int slot = 0; slot < WINGEN_CACHE_COL_SLOTS; ++slot) {
#pragma HLS UNROLL
        col_tag[slot] = static_cast<i32_t>(-32768);
    }

    const int out_w_i = static_cast<int>(sched.out_w.to_uint());
    const int stride_i = static_cast<int>(sched.stride.to_uint());
    const int dilation_i = static_cast<int>(sched.dilation.to_uint());
    const int padding_i = static_cast<int>(sched.padding.to_uint());
    const int k_tiles_i = static_cast<int>(sched.k_tiles.to_uint());
    const i32_t base_h =
        static_cast<i32_t>(out_row.to_uint() * static_cast<unsigned>(stride_i)) -
        static_cast<i32_t>(padding_i);

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

        update_3x3_column_cache(src_desc,
                                sched,
                                base_h,
                                want_col,
                                col_tag,
                                slot_for_kw,
                                cache);

        for (int kt = 0; kt < MAX_K_TILE_COUNT; ++kt) {
            if (kt >= k_tiles_i) {
                break;
            }
            act_vec_t word = 0;
            build_staged_word_for_mode(mode, cache, slot_for_kw, kt, word);
            act_stream.write(word);
        }
    }
}

static void scheduled_1x1_window_row(const tensor_desc_t& src_desc,
                                     const window_sched_desc_t& sched,
                                     hls::stream<act_vec_t>& act_stream,
                                     u16_t out_row,
                                     bool aligned_full_tile) {
#pragma HLS INLINE off
    const int stride_i = static_cast<int>(sched.stride.to_uint());
    const int out_w_i = static_cast<int>(sched.out_w.to_uint());
    const int k_tiles_i = static_cast<int>(sched.k_tiles.to_uint());
    const int in_c_i = static_cast<int>(sched.in_c.to_uint());
    const i32_t ih = static_cast<i32_t>(out_row.to_uint() * static_cast<unsigned>(stride_i));

    for (int ow_i = 0; ow_i < MAX_FM_W; ++ow_i) {
        if (ow_i >= out_w_i) {
            break;
        }
        const i32_t iw = static_cast<i32_t>(ow_i * stride_i);
        for (int kt = 0; kt < MAX_K_TILE_COUNT; ++kt) {
#pragma HLS PIPELINE off
            if (kt >= k_tiles_i) {
                break;
            }
            const unsigned c_begin = static_cast<unsigned>(kt) * static_cast<unsigned>(TK);
            act_vec_t word = 0;
            if (aligned_full_tile &&
                win_can_read_aligned_1x1(src_desc, static_cast<u16_t>(c_begin))) {
                word = read_aligned_or_zero(src_desc, ih, iw, static_cast<u16_t>(c_begin));
            } else {
                unsigned valid = (c_begin < static_cast<unsigned>(in_c_i))
                                     ? (static_cast<unsigned>(in_c_i) - c_begin)
                                     : 0U;
                if (valid > static_cast<unsigned>(TK)) {
                    valid = static_cast<unsigned>(TK);
                }
                word = read_tile_or_zero(src_desc,
                                         ih,
                                         iw,
                                         static_cast<u16_t>(c_begin),
                                         static_cast<u8_t>(valid));
            }
            act_stream.write(word);
        }
    }
}

void scheduled_window_generator_row(const tensor_desc_t& src_desc,
                                    const conv_exec_desc_t&,
                                    const window_sched_desc_t& sched,
                                    hls::stream<act_vec_t>& act_stream,
                                    u16_t out_row) {
#pragma HLS INLINE off
    const unsigned mode = sched.mode.to_uint();
    if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C3)) {
        scheduled_3x3_staged_window_row(src_desc, sched, mode, act_stream, out_row);
        return;
    }
    if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C12)) {
        scheduled_3x3_staged_window_row(src_desc, sched, mode, act_stream, out_row);
        return;
    }
    if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C19)) {
        scheduled_3x3_staged_window_row(src_desc, sched, mode, act_stream, out_row);
        return;
    }
    if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C25)) {
        scheduled_3x3_staged_window_row(src_desc, sched, mode, act_stream, out_row);
        return;
    }
    if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C28)) {
        scheduled_3x3_staged_window_row(src_desc, sched, mode, act_stream, out_row);
        return;
    }
    if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C64)) {
        scheduled_3x3_staged_window_row(src_desc, sched, mode, act_stream, out_row);
        return;
    }
    if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C128)) {
        scheduled_3x3_staged_window_row(src_desc, sched, mode, act_stream, out_row);
        return;
    }
    if (mode == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C131)) {
        scheduled_3x3_staged_window_row(src_desc, sched, mode, act_stream, out_row);
        return;
    }
    if (mode == static_cast<unsigned>(WIN_MODE_1X1_ALIGNED)) {
        scheduled_1x1_window_row(src_desc, sched, act_stream, out_row, true);
        return;
    }
    if (mode == static_cast<unsigned>(WIN_MODE_1X1_PACKED)) {
        scheduled_1x1_window_row(src_desc, sched, act_stream, out_row, false);
        return;
    }
}

}  // namespace esp_int8
