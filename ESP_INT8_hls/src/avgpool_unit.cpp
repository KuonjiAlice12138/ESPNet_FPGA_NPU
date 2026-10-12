#include "../include/npu_q.hpp"
#include "../include/npu_config.hpp"

namespace esp_int8 {

constexpr int AVGPOOL_C1_GROUP_PIXELS = 32;
constexpr int AVGPOOL_C1_GROUP_INPUT_COLS = 65;
constexpr int AVGPOOL_C1_CACHE_WORDS = 3;

bool on_chip_memory_read_main_uram_word(u8_t bank_id,
                                        u32_t byte_offset,
                                        act_vec_t& packed);
bool on_chip_memory_read_pool_bram_word(u8_t bank_id,
                                        u32_t byte_offset,
                                        act_vec_t& packed);
bool on_chip_memory_write_pool_bram_word(u8_t bank_id,
                                         u32_t byte_offset,
                                         act_vec_t packed);

static i32_t round_div9(i32_t x) {
#pragma HLS INLINE
    return (x >= 0) ? static_cast<i32_t>((x + 4) / 9)
                    : static_cast<i32_t>((x - 4) / 9);
}

static i8_t requant_pool_avg(i32_t avg, const pool_q_t& qparam) {
#pragma HLS INLINE
    if (qparam.same_scale.to_uint() != 0U) {
        return apply_act(clamp_i8(avg), qparam.act_type);
    }
    const i64_t scaled = static_cast<i64_t>(avg) * static_cast<i64_t>(qparam.mult);
    return apply_act(clamp_i8(round_shift(scaled, qparam.shift)), qparam.act_type);
}

static bool read_avgpool_abs_word(u8_t src_bank,
                                  u32_t byte_offset,
                                  act_vec_t& word) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off
    const unsigned bank = src_bank.to_uint();
    if (bank == static_cast<unsigned>(BANK_BRAM_SCR0) ||
        bank == static_cast<unsigned>(BANK_BRAM_SCR1)) {
        return on_chip_memory_read_pool_bram_word(src_bank, byte_offset, word);
    }
    return on_chip_memory_read_main_uram_word(src_bank, byte_offset, word);
}

static bool pool_shape_supported(const tensor_desc_t& src,
                                 const tensor_desc_t& dst,
                                 const pool_q_t& qparam) {
#pragma HLS INLINE
    const unsigned kernel = qparam.kernel.to_uint() == 0U ? 3U : qparam.kernel.to_uint();
    const unsigned stride = qparam.stride.to_uint() == 0U ? 2U : qparam.stride.to_uint();
    return kernel == 3U && stride == 2U &&
           src.c.to_uint() == 1U && dst.c.to_uint() == 1U &&
           tensor_phys_c(src).to_uint() == 1U &&
           tensor_phys_c(dst).to_uint() == 1U &&
           src.reserved1.to_uint() == 0U && dst.reserved1.to_uint() == 0U &&
           (src.reserved0.to_uint() & TENSOR_ROW_ALIGN32) == 0U &&
           (dst.reserved0.to_uint() & TENSOR_ROW_ALIGN32) == 0U &&
           src.h.to_uint() > 0U && src.h.to_uint() <= MAX_FM_H &&
           src.w.to_uint() > 0U && src.w.to_uint() <= MAX_FM_W &&
           dst.h.to_uint() == (src.h.to_uint() + 1U) / 2U &&
           dst.w.to_uint() == (src.w.to_uint() + 1U) / 2U;
}

static bool read_c1_group_cache_row(const tensor_desc_t& src,
                                    int h,
                                    int base_w,
                                    act_vec_t cache[AVGPOOL_C1_CACHE_WORDS],
                                    u8_t& byte0,
                                    u8_t& left_pad,
                                    u8_t& valid_cols) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off
    for (int word = 0; word < AVGPOOL_C1_CACHE_WORDS; ++word) {
#pragma HLS UNROLL
        cache[word] = 0;
    }
    byte0 = 0;
    left_pad = 0;
    valid_cols = 0;
    if (h < 0 || h >= static_cast<int>(src.h.to_uint())) {
        return true;
    }
    const int width = static_cast<int>(src.w.to_uint());
    const int begin = base_w < 0 ? 0 : base_w;
    const int end_unclipped = base_w + AVGPOOL_C1_GROUP_INPUT_COLS;
    const int end = end_unclipped < width ? end_unclipped : width;
    if (begin >= end) {
        return true;
    }
    const u32_t first_byte = src.base_offset +
        static_cast<u32_t>(h) * tensor_row_stride_bytes(src) +
        static_cast<u32_t>(begin);
    byte0 = static_cast<u8_t>(first_byte.to_uint() & 31U);
    left_pad = static_cast<u8_t>(begin - base_w);
    valid_cols = static_cast<u8_t>(end - begin);
    const u32_t word0 = first_byte & static_cast<u32_t>(~31U);
    const unsigned read_bytes = byte0.to_uint() + valid_cols.to_uint();
    for (int word = 0; word < AVGPOOL_C1_CACHE_WORDS; ++word) {
#pragma HLS PIPELINE off
        if (static_cast<unsigned>(word * AXI_WORD_BYTES) < read_bytes &&
            !read_avgpool_abs_word(src.bank_id,
                word0 + static_cast<u32_t>(word * AXI_WORD_BYTES), cache[word])) {
            return false;
        }
    }
    return true;
}

static i32_t cached_c1_col(const act_vec_t cache[AVGPOOL_C1_CACHE_WORDS],
                           u8_t byte0, u8_t left_pad, u8_t valid_cols,
                           int logical_col) {
#pragma HLS INLINE
    const int col = logical_col - static_cast<int>(left_pad.to_uint());
    if (col < 0 || col >= static_cast<int>(valid_cols.to_uint())) {
        return static_cast<i32_t>(0);
    }
    const unsigned index = byte0.to_uint() + static_cast<unsigned>(col);
    act_vec_t word = 0;
    switch (index >> 5) {
        case 0: word = cache[0]; break;
        case 1: word = cache[1]; break;
        default: word = cache[2]; break;
    }
    const act_vec_t shifted = word >> ((index & 31U) * 8U);
    i8_t value;
    value.range(7, 0) = shifted.range(7, 0);
    return static_cast<i32_t>(value);
}

static bool avgpool_unit_c1(const tensor_desc_t& src,
                            const tensor_desc_t& dst,
                            const pool_q_t& qparam) {
#pragma HLS INLINE off
    act_vec_t row0[AVGPOOL_C1_CACHE_WORDS];
    act_vec_t row1[AVGPOOL_C1_CACHE_WORDS];
    act_vec_t row2[AVGPOOL_C1_CACHE_WORDS];
#pragma HLS ARRAY_PARTITION variable=row0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=row1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=row2 complete dim=1
    act_vec_t out_word = 0;
    unsigned fill = 0;
    u32_t word_offset = dst.base_offset;
    for (int oh = 0; oh < MAX_FM_H; ++oh) {
#pragma HLS PIPELINE off
        if (oh >= static_cast<int>(dst.h.to_uint())) {
            break;
        }
        for (int group = 0; group < (MAX_FM_W + 31) / 32; ++group) {
#pragma HLS PIPELINE off
            const int ow_begin = group * AVGPOOL_C1_GROUP_PIXELS;
            if (ow_begin >= static_cast<int>(dst.w.to_uint())) {
                break;
            }
            u8_t byte0, byte1, byte2, left0, left1, left2, valid0, valid1, valid2;
            const int base_h = oh * 2 - 1;
            const int base_w = ow_begin * 2 - 1;
            if (!read_c1_group_cache_row(src, base_h, base_w, row0, byte0, left0, valid0) ||
                !read_c1_group_cache_row(src, base_h + 1, base_w, row1, byte1, left1, valid1) ||
                !read_c1_group_cache_row(src, base_h + 2, base_w, row2, byte2, left2, valid2)) {
                return false;
            }
            for (int pix = 0; pix < AVGPOOL_C1_GROUP_PIXELS; ++pix) {
#pragma HLS PIPELINE off
                if (ow_begin + pix >= static_cast<int>(dst.w.to_uint())) {
                    break;
                }
                i32_t sum = 0;
                for (int kw = 0; kw < 3; ++kw) {
#pragma HLS UNROLL
                    const int col = pix * 2 + kw;
                    sum += cached_c1_col(row0, byte0, left0, valid0, col) +
                           cached_c1_col(row1, byte1, left1, valid1, col) +
                           cached_c1_col(row2, byte2, left2, valid2, col);
                }
                const i8_t value = requant_pool_avg(round_div9(sum), qparam);
                // Packing persists across rows; only complete words touch memory.
                out_word >>= 8;
                out_word.range(255, 248) = static_cast<u8_t>(value);
                ++fill;
                if (fill == AXI_WORD_BYTES) {
                    if (!on_chip_memory_write_pool_bram_word(dst.bank_id, word_offset, out_word)) {
                        return false;
                    }
                    word_offset += AXI_WORD_BYTES;
                    out_word = 0;
                    fill = 0;
                }
            }
        }
    }
    if (fill != 0U) {
        out_word >>= (AXI_WORD_BYTES - fill) * 8U;
        return on_chip_memory_write_pool_bram_word(dst.bank_id, word_offset, out_word);
    }
    return true;
}

bool avgpool_unit_checked(const tensor_desc_t& src,
                           const tensor_desc_t& dst,
                           const pool_q_t& qparam,
                           i8_t* fmbuf_base) {
#pragma HLS INLINE off
    (void)fmbuf_base;
    if (!pool_shape_supported(src, dst, qparam)) {
        return false;
    }
    return avgpool_unit_c1(src, dst, qparam);
}

}  // namespace esp_int8
