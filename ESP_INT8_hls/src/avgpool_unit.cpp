#include "../include/npu_q.hpp"
#include "../include/npu_config.hpp"

namespace esp_int8 {

constexpr int AVGPOOL_C3_GROUP_PIXELS = 32;
constexpr int AVGPOOL_C3_GROUP_INPUT_COLS = 65;
constexpr int AVGPOOL_C3_GROUP_BYTES = AVGPOOL_C3_GROUP_INPUT_COLS * 3;
constexpr int AVGPOOL_C3_CACHE_WORDS = 7;

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

static i8_t packed_i8(act_vec_t word, int lane) {
#pragma HLS INLINE
    const act_vec_t shifted = static_cast<act_vec_t>(word >> (lane * 8));
    const u8_t raw = shifted.range(7, 0);
    i8_t value;
    value.range(7, 0) = raw;
    return value;
}

static void set_packed_i8(act_vec_t& word, int lane, i8_t value) {
#pragma HLS INLINE
    word.range(lane * 8 + 7, lane * 8) = static_cast<u8_t>(value);
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

static bool write_avgpool_aligned_abs_word(u8_t dst_bank,
                                           u32_t byte_offset,
                                           act_vec_t word) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off
    return on_chip_memory_write_pool_bram_word(dst_bank, byte_offset, word);
}

static bool pool_shape_supported(const tensor_desc_t& src,
                                 const tensor_desc_t& dst,
                                 const pool_q_t& qparam) {
#pragma HLS INLINE
    const unsigned kernel = (qparam.kernel.to_uint() == 0U) ? 3U : qparam.kernel.to_uint();
    const unsigned stride = (qparam.stride.to_uint() == 0U) ? 2U : qparam.stride.to_uint();
    const unsigned expect_h = (src.h.to_uint() + stride - 1U) / stride;
    const unsigned expect_w = (src.w.to_uint() + stride - 1U) / stride;
    return kernel == 3U &&
           stride == 2U &&
           src.c.to_uint() == dst.c.to_uint() &&
           dst.c.to_uint() == 3U &&
           dst.h.to_uint() == expect_h &&
           dst.w.to_uint() == expect_w;
}

static u16_t pool_desc_phys_c(const tensor_desc_t& desc) {
#pragma HLS INLINE
    return (desc.reserved0.to_uint() == 0U) ? desc.c : desc.reserved0;
}

static u32_t pool_row_base(const tensor_desc_t& desc, u16_t h) {
#pragma HLS INLINE
    return desc.base_offset +
           static_cast<u32_t>(h) * static_cast<u32_t>(desc.w) *
               static_cast<u32_t>(pool_desc_phys_c(desc)) +
           static_cast<u32_t>(desc.reserved1);
}

static void quantize_c3_avg_pixel(const i32_t sum[3],
                                  const pool_q_t& qparam,
                                  act_vec_t& out_packed) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=sum complete dim=1
    out_packed = 0;
    for (int c = 0; c < 3; ++c) {
#pragma HLS UNROLL
        set_packed_i8(out_packed, c, requant_pool_avg(round_div9(sum[c]), qparam));
    }
}

template <int DST_BYTE, int SRC_BYTE, int COUNT_BYTES>
static void copy_c3_segment(act_vec_t& dst, const act_vec_t& src) {
#pragma HLS INLINE
    dst.range(DST_BYTE * 8 + COUNT_BYTES * 8 - 1, DST_BYTE * 8) =
        src.range(SRC_BYTE * 8 + COUNT_BYTES * 8 - 1, SRC_BYTE * 8);
}

static bool read_c3_group_cache_row(const tensor_desc_t& src,
                                    i32_t h,
                                    i32_t base_w,
                                    act_vec_t cache[AVGPOOL_C3_CACHE_WORDS],
                                    u8_t& byte0,
                                    u8_t& left_pad_cols,
                                    u8_t& valid_cols) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off

    for (int word = 0; word < AVGPOOL_C3_CACHE_WORDS; ++word) {
#pragma HLS UNROLL
        cache[word] = 0;
    }
    byte0 = 0;
    left_pad_cols = 0;
    valid_cols = 0;

    const unsigned phys_c = pool_desc_phys_c(src).to_uint();
    const unsigned c_offset = src.reserved1.to_uint();
    if (phys_c != 3U || c_offset != 0U) {
        return false;
    }

    if (h < 0 || h >= static_cast<i32_t>(src.h)) {
        return true;
    }

    const i32_t group_last_w = base_w + AVGPOOL_C3_GROUP_INPUT_COLS - 1;
    const i32_t valid_start_w = (base_w < 0) ? static_cast<i32_t>(0) : base_w;
    const i32_t src_last_w = static_cast<i32_t>(src.w) - 1;
    const i32_t valid_end_w = (group_last_w > src_last_w) ? src_last_w : group_last_w;
    if (valid_start_w > valid_end_w) {
        return true;
    }

    left_pad_cols = static_cast<u8_t>(
        static_cast<unsigned>(valid_start_w - base_w));
    valid_cols = static_cast<u8_t>(
        static_cast<unsigned>(valid_end_w - valid_start_w + 1));

    const u32_t start_offset =
        src.base_offset +
        (static_cast<u32_t>(static_cast<u16_t>(h)) * static_cast<u32_t>(src.w) +
         static_cast<u32_t>(static_cast<u16_t>(valid_start_w))) *
            static_cast<u32_t>(phys_c);
    const u32_t word0_offset =
        start_offset & static_cast<u32_t>(~(AXI_WORD_BYTES - 1));
    byte0 = static_cast<u8_t>(start_offset.to_uint() &
                              static_cast<unsigned>(AXI_WORD_BYTES - 1));
    if (byte0.to_uint() + AVGPOOL_C3_GROUP_BYTES >
        static_cast<unsigned>(AVGPOOL_C3_CACHE_WORDS * AXI_WORD_BYTES)) {
        return false;
    }

    const unsigned needed_bytes =
        static_cast<unsigned>(valid_cols.to_uint()) * 3U;
    const unsigned read_window_bytes =
        static_cast<unsigned>(byte0.to_uint()) + needed_bytes;

    for (int word = 0; word < AVGPOOL_C3_CACHE_WORDS; ++word) {
#pragma HLS UNROLL
        if (static_cast<unsigned>(word * AXI_WORD_BYTES) < read_window_bytes) {
            if (!read_avgpool_abs_word(
                    src.bank_id,
                    word0_offset + static_cast<u32_t>(word * AXI_WORD_BYTES),
                    cache[word])) {
                return false;
            }
        }
    }
    return true;
}

static act_vec_t select_c3_cache_word(const act_vec_t cache[AVGPOOL_C3_CACHE_WORDS],
                                      int word_index) {
#pragma HLS INLINE
    act_vec_t word = 0;
    switch (word_index) {
        case 0: word = cache[0]; break;
        case 1: word = cache[1]; break;
        case 2: word = cache[2]; break;
        case 3: word = cache[3]; break;
        case 4: word = cache[4]; break;
        case 5: word = cache[5]; break;
        default: word = cache[6]; break;
    }
    return word;
}

static act_vec_t cached_c3_col(const act_vec_t cache[AVGPOOL_C3_CACHE_WORDS],
                               u8_t byte0,
                               u8_t left_pad_cols,
                               u8_t valid_cols,
                               int logical_col) {
#pragma HLS INLINE
    act_vec_t packed = 0;
    const int valid_col = logical_col - static_cast<int>(left_pad_cols.to_uint());
    if (valid_col < 0 || valid_col >= static_cast<int>(valid_cols.to_uint())) {
        return packed;
    }

    const int abs_index = static_cast<int>(byte0.to_uint()) + valid_col * 3;
    const int word_index = abs_index >> 5;
    const int byte_lane = abs_index & (AXI_WORD_BYTES - 1);

    const act_vec_t word = select_c3_cache_word(cache, word_index);
    if (byte_lane <= AXI_WORD_BYTES - 3) {
        packed.range(23, 0) = word.range(byte_lane * 8 + 23, byte_lane * 8);
    } else {
        const act_vec_t next = select_c3_cache_word(cache, word_index + 1);
        if (byte_lane == AXI_WORD_BYTES - 2) {
            packed.range(15, 0) = word.range(255, 240);
            packed.range(23, 16) = next.range(7, 0);
        } else {
            packed.range(7, 0) = word.range(255, 248);
            packed.range(23, 8) = next.range(15, 0);
        }
    }
    return packed;
}

static void sum_c3_cached_pixel(const act_vec_t row0[AVGPOOL_C3_CACHE_WORDS],
                                const act_vec_t row1[AVGPOOL_C3_CACHE_WORDS],
                                const act_vec_t row2[AVGPOOL_C3_CACHE_WORDS],
                                u8_t byte0,
                                u8_t byte1,
                                u8_t byte2,
                                u8_t left0,
                                u8_t left1,
                                u8_t left2,
                                u8_t valid0,
                                u8_t valid1,
                                u8_t valid2,
                                int pix,
                                i32_t sum[3]) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off
#pragma HLS ARRAY_PARTITION variable=sum complete dim=1
    const int col0 = pix * 2;
    const act_vec_t r0c0 = cached_c3_col(row0, byte0, left0, valid0, col0 + 0);
    const act_vec_t r0c1 = cached_c3_col(row0, byte0, left0, valid0, col0 + 1);
    const act_vec_t r0c2 = cached_c3_col(row0, byte0, left0, valid0, col0 + 2);
    const act_vec_t r1c0 = cached_c3_col(row1, byte1, left1, valid1, col0 + 0);
    const act_vec_t r1c1 = cached_c3_col(row1, byte1, left1, valid1, col0 + 1);
    const act_vec_t r1c2 = cached_c3_col(row1, byte1, left1, valid1, col0 + 2);
    const act_vec_t r2c0 = cached_c3_col(row2, byte2, left2, valid2, col0 + 0);
    const act_vec_t r2c1 = cached_c3_col(row2, byte2, left2, valid2, col0 + 1);
    const act_vec_t r2c2 = cached_c3_col(row2, byte2, left2, valid2, col0 + 2);
    for (int c = 0; c < 3; ++c) {
#pragma HLS UNROLL
        sum[c] = static_cast<i32_t>(packed_i8(r0c0, c)) +
                 static_cast<i32_t>(packed_i8(r0c1, c)) +
                 static_cast<i32_t>(packed_i8(r0c2, c)) +
                 static_cast<i32_t>(packed_i8(r1c0, c)) +
                 static_cast<i32_t>(packed_i8(r1c1, c)) +
                 static_cast<i32_t>(packed_i8(r1c2, c)) +
                 static_cast<i32_t>(packed_i8(r2c0, c)) +
                 static_cast<i32_t>(packed_i8(r2c1, c)) +
                 static_cast<i32_t>(packed_i8(r2c2, c));
    }
}

static void store_c3_pixel_to_group_words(int pix,
                                          const act_vec_t& pixel,
                                          act_vec_t& w0,
                                          act_vec_t& w1,
                                          act_vec_t& w2) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off
    switch (pix) {
        case 0:  copy_c3_segment<0, 0, 3>(w0, pixel); break;
        case 1:  copy_c3_segment<3, 0, 3>(w0, pixel); break;
        case 2:  copy_c3_segment<6, 0, 3>(w0, pixel); break;
        case 3:  copy_c3_segment<9, 0, 3>(w0, pixel); break;
        case 4:  copy_c3_segment<12, 0, 3>(w0, pixel); break;
        case 5:  copy_c3_segment<15, 0, 3>(w0, pixel); break;
        case 6:  copy_c3_segment<18, 0, 3>(w0, pixel); break;
        case 7:  copy_c3_segment<21, 0, 3>(w0, pixel); break;
        case 8:  copy_c3_segment<24, 0, 3>(w0, pixel); break;
        case 9:  copy_c3_segment<27, 0, 3>(w0, pixel); break;
        case 10:
            copy_c3_segment<30, 0, 2>(w0, pixel);
            copy_c3_segment<0, 2, 1>(w1, pixel);
            break;
        case 11: copy_c3_segment<1, 0, 3>(w1, pixel); break;
        case 12: copy_c3_segment<4, 0, 3>(w1, pixel); break;
        case 13: copy_c3_segment<7, 0, 3>(w1, pixel); break;
        case 14: copy_c3_segment<10, 0, 3>(w1, pixel); break;
        case 15: copy_c3_segment<13, 0, 3>(w1, pixel); break;
        case 16: copy_c3_segment<16, 0, 3>(w1, pixel); break;
        case 17: copy_c3_segment<19, 0, 3>(w1, pixel); break;
        case 18: copy_c3_segment<22, 0, 3>(w1, pixel); break;
        case 19: copy_c3_segment<25, 0, 3>(w1, pixel); break;
        case 20: copy_c3_segment<28, 0, 3>(w1, pixel); break;
        case 21:
            copy_c3_segment<31, 0, 1>(w1, pixel);
            copy_c3_segment<0, 1, 2>(w2, pixel);
            break;
        case 22: copy_c3_segment<2, 0, 3>(w2, pixel); break;
        case 23: copy_c3_segment<5, 0, 3>(w2, pixel); break;
        case 24: copy_c3_segment<8, 0, 3>(w2, pixel); break;
        case 25: copy_c3_segment<11, 0, 3>(w2, pixel); break;
        case 26: copy_c3_segment<14, 0, 3>(w2, pixel); break;
        case 27: copy_c3_segment<17, 0, 3>(w2, pixel); break;
        case 28: copy_c3_segment<20, 0, 3>(w2, pixel); break;
        case 29: copy_c3_segment<23, 0, 3>(w2, pixel); break;
        case 30: copy_c3_segment<26, 0, 3>(w2, pixel); break;
        default: copy_c3_segment<29, 0, 3>(w2, pixel); break;
    }
}

static bool avgpool_c3_group32_pack_write(const tensor_desc_t& src,
                                          const tensor_desc_t& dst,
                                          const pool_q_t& qparam,
                                          int oh_i,
                                          int ow_start,
                                          u32_t word_offset) {
#pragma HLS INLINE off
    act_vec_t row0[AVGPOOL_C3_CACHE_WORDS];
    act_vec_t row1[AVGPOOL_C3_CACHE_WORDS];
    act_vec_t row2[AVGPOOL_C3_CACHE_WORDS];
    u8_t byte0 = 0;
    u8_t byte1 = 0;
    u8_t byte2 = 0;
    u8_t left0 = 0;
    u8_t left1 = 0;
    u8_t left2 = 0;
    u8_t valid0 = 0;
    u8_t valid1 = 0;
    u8_t valid2 = 0;
#pragma HLS ARRAY_PARTITION variable=row0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=row1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=row2 complete dim=1

    const i32_t base_h = static_cast<i32_t>(oh_i) * 2 - 1;
    const i32_t base_w = static_cast<i32_t>(ow_start) * 2 - 1;
    if (!read_c3_group_cache_row(
            src, base_h + 0, base_w, row0, byte0, left0, valid0) ||
        !read_c3_group_cache_row(
            src, base_h + 1, base_w, row1, byte1, left1, valid1) ||
        !read_c3_group_cache_row(
            src, base_h + 2, base_w, row2, byte2, left2, valid2)) {
        return false;
    }

    act_vec_t w0 = 0;
    act_vec_t w1 = 0;
    act_vec_t w2 = 0;
    for (int pix = 0; pix < AVGPOOL_C3_GROUP_PIXELS; ++pix) {
#pragma HLS PIPELINE off
        i32_t sum[3];
#pragma HLS ARRAY_PARTITION variable=sum complete dim=1
        act_vec_t out_packed = 0;
        sum_c3_cached_pixel(row0,
                            row1,
                            row2,
                            byte0,
                            byte1,
                            byte2,
                            left0,
                            left1,
                            left2,
                            valid0,
                            valid1,
                            valid2,
                            pix,
                            sum);
        quantize_c3_avg_pixel(sum, qparam, out_packed);
        store_c3_pixel_to_group_words(pix, out_packed, w0, w1, w2);
    }

    if (!write_avgpool_aligned_abs_word(dst.bank_id, word_offset, w0)) {
        return false;
    }
    if (!write_avgpool_aligned_abs_word(dst.bank_id,
                                        word_offset + static_cast<u32_t>(32),
                                        w1)) {
        return false;
    }
    return write_avgpool_aligned_abs_word(dst.bank_id,
                                          word_offset + static_cast<u32_t>(64),
                                          w2);
}

static bool avgpool_unit_c3_fast(const tensor_desc_t& src,
                                 const tensor_desc_t& dst,
                                 const pool_q_t& qparam) {
#pragma HLS INLINE off
    const int out_h_i = static_cast<int>(dst.h.to_uint());
    const int out_w_i = static_cast<int>(dst.w.to_uint());
    const u16_t dst_phys_c = pool_desc_phys_c(dst);
    if (dst_phys_c.to_uint() != 3U) {
        return false;
    }
    if (((out_w_i * 3) & (AXI_WORD_BYTES - 1)) != 0) {
        return false;
    }

    if (out_h_i <= 0 || out_w_i <= 0) {
        return true;
    }

    for (int oh_i = 0; oh_i < MAX_FM_H; ++oh_i) {
#pragma HLS PIPELINE off
        if (oh_i >= out_h_i) {
            break;
        }
        const u16_t oh = static_cast<u16_t>(oh_i);
        const u32_t row_base = pool_row_base(dst, oh);

        for (int group = 0; group < MAX_FM_W / 32; ++group) {
#pragma HLS PIPELINE off
            const int ow_start = group * 32;
            if (ow_start >= out_w_i) {
                break;
            }
            const u32_t word_offset =
                row_base + static_cast<u32_t>(group) * static_cast<u32_t>(96);
            if (!avgpool_c3_group32_pack_write(src,
                                               dst,
                                               qparam,
                                               oh_i,
                                               ow_start,
                                               word_offset)) {
                return false;
            }
        }
    }

    return true;
}

bool avgpool_unit_checked(
    const tensor_desc_t& src,
    const tensor_desc_t& dst,
    const pool_q_t& qparam,
    i8_t* fmbuf_base) {
#pragma HLS INLINE off
    (void)fmbuf_base;

    if (!pool_shape_supported(src, dst, qparam)) {
        return false;
    }

    return avgpool_unit_c3_fast(src, dst, qparam);
}

}  // namespace esp_int8
