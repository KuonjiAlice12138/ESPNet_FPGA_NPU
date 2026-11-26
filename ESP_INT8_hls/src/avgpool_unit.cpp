#include "../include/npu_q.hpp"
#include "../include/npu_config.hpp"

namespace esp_int8 {

constexpr int AVGPOOL_GROUP_PIXELS = 32;
constexpr int AVGPOOL_GROUP_INPUT_COLS = 65;
constexpr int AVGPOOL_CACHE_WORDS = 7;

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
    const unsigned channels = src.c.to_uint();
    return kernel == 3U && stride == 2U &&
           (channels == 1U || channels == 3U) && dst.c.to_uint() == channels &&
           tensor_phys_c(src).to_uint() == channels &&
           tensor_phys_c(dst).to_uint() == channels &&
           src.reserved1.to_uint() == 0U && dst.reserved1.to_uint() == 0U &&
           (dst.base_offset.to_uint() & 31U) == 0U &&
           (tensor_row_stride_bytes(dst).to_uint() & 31U) == 0U &&
           src.h.to_uint() > 0U && src.h.to_uint() <= MAX_INPUT_FRAME_H &&
           src.w.to_uint() > 0U && src.w.to_uint() <= MAX_INPUT_FRAME_W &&
           dst.h.to_uint() == (src.h.to_uint() + 1U) / 2U &&
           dst.w.to_uint() == (src.w.to_uint() + 1U) / 2U;
}

static bool read_pool_group_cache_row(const tensor_desc_t& src,
                                    int h,
                                    int base_w,
                                    act_vec_t cache[AVGPOOL_CACHE_WORDS],
                                    u8_t& byte0,
                                    u8_t& left_pad,
                                    u8_t& valid_cols) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off
    for (int word = 0; word < AVGPOOL_CACHE_WORDS; ++word) {
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
    const int end_unclipped = base_w + AVGPOOL_GROUP_INPUT_COLS;
    const int end = end_unclipped < width ? end_unclipped : width;
    if (begin >= end) {
        return true;
    }
    const u32_t first_byte = src.base_offset +
        static_cast<u32_t>(h) * tensor_row_stride_bytes(src) +
        static_cast<u32_t>(begin * src.c.to_uint());
    byte0 = static_cast<u8_t>(first_byte.to_uint() & 31U);
    left_pad = static_cast<u8_t>(begin - base_w);
    valid_cols = static_cast<u8_t>(end - begin);
    const u32_t word0 = first_byte & static_cast<u32_t>(~31U);
    const unsigned read_bytes = byte0.to_uint() + valid_cols.to_uint() * src.c.to_uint();
    if (read_bytes > AVGPOOL_CACHE_WORDS * AXI_WORD_BYTES) { return false; }
    for (int word = 0; word < AVGPOOL_CACHE_WORDS; ++word) {
#pragma HLS PIPELINE off
        if (static_cast<unsigned>(word * AXI_WORD_BYTES) >= read_bytes) { break; }
        if (!read_avgpool_abs_word(src.bank_id,
                word0 + static_cast<u32_t>(word * AXI_WORD_BYTES), cache[word])) {
            return false;
        }
    }
    return true;
}

static act_vec_t select_pool_cache_word(const act_vec_t cache[AVGPOOL_CACHE_WORDS],
                                        unsigned index) {
#pragma HLS INLINE
    act_vec_t word = 0;
    switch (index) {
        case 0: word = cache[0]; break;
        case 1: word = cache[1]; break;
        case 2: word = cache[2]; break;
        case 3: word = cache[3]; break;
        case 4: word = cache[4]; break;
        case 5: word = cache[5]; break;
        case 6: word = cache[6]; break;
        default: break;
    }
    return word;
}

static ap_uint<24> cached_pool_pixel(const act_vec_t cache[AVGPOOL_CACHE_WORDS],
                                     u8_t byte0, u8_t left, u8_t valid,
                                     int logical_col, bool rgb) {
#pragma HLS INLINE
    const int col = logical_col - static_cast<int>(left.to_uint());
    if (col < 0 || col >= static_cast<int>(valid.to_uint())) { return 0; }
    const unsigned channels = rgb ? 3U : 1U;
    const unsigned byte = byte0.to_uint() + static_cast<unsigned>(col) * channels;
    const unsigned lane = byte & 31U;
    const unsigned index = byte >> 5;
    ap_uint<24> pixel = select_pool_cache_word(cache, index) >> (lane * 8U);
    if (lane + channels > 32U) {
        pixel |= ap_uint<24>(select_pool_cache_word(cache, index + 1U)) << ((32U - lane) * 8U);
    }
    if (!rgb) { pixel.range(23, 8) = 0; }
    return pixel;
}

static void compute_pool_group_words(
    const act_vec_t row0[AVGPOOL_CACHE_WORDS],
    const act_vec_t row1[AVGPOOL_CACHE_WORDS],
    const act_vec_t row2[AVGPOOL_CACHE_WORDS],
    u8_t byte0, u8_t left0, u8_t valid0,
    u8_t byte1, u8_t left1, u8_t valid1,
    u8_t byte2, u8_t left2, u8_t valid2,
    const pool_q_t& qparam, int pixel_count, bool rgb, act_vec_t packed[3]) {
#pragma HLS INLINE off
    u8_t values[AVGPOOL_GROUP_PIXELS][3];
#pragma HLS ARRAY_PARTITION variable=values complete dim=0
    for (int pix = 0; pix < AVGPOOL_GROUP_PIXELS; ++pix) {
#pragma HLS PIPELINE II=1
        i32_t sum[3] = {0, 0, 0};
#pragma HLS ARRAY_PARTITION variable=sum complete dim=1
        for (int kw = 0; kw < 3; ++kw) {
#pragma HLS UNROLL
            const int col = pix * 2 + kw;
            const ap_uint<24> a = cached_pool_pixel(row0, byte0, left0, valid0, col, rgb);
            const ap_uint<24> b = cached_pool_pixel(row1, byte1, left1, valid1, col, rgb);
            const ap_uint<24> c = cached_pool_pixel(row2, byte2, left2, valid2, col, rgb);
            for (int ch = 0; ch < 3; ++ch) {
#pragma HLS UNROLL
                i8_t av, bv, cv;
                av.range(7, 0) = a.range(ch * 8 + 7, ch * 8);
                bv.range(7, 0) = b.range(ch * 8 + 7, ch * 8);
                cv.range(7, 0) = c.range(ch * 8 + 7, ch * 8);
                sum[ch] += static_cast<i32_t>(av) + static_cast<i32_t>(bv) + static_cast<i32_t>(cv);
            }
        }
        for (int ch = 0; ch < 3; ++ch) {
#pragma HLS UNROLL
            const i8_t result = requant_pool_avg(round_div9(sum[ch]), qparam);
            values[pix][ch] = pix < pixel_count && (rgb || ch == 0)
                                ? static_cast<u8_t>(result) : u8_t(0);
        }
    }
    for (int wi = 0; wi < 3; ++wi) {
#pragma HLS UNROLL
        packed[wi] = 0;
        for (int lane = 0; lane < 32; ++lane) {
#pragma HLS UNROLL
            const int byte = wi * 32 + lane;
            const u8_t value = rgb ? values[byte / 3][byte % 3]
                                   : wi == 0 ? values[lane][0] : u8_t(0);
            packed[wi].range(lane * 8 + 7, lane * 8) = value;
        }
    }
}

static bool avgpool_group_engine(const tensor_desc_t& src,
                                  const tensor_desc_t& dst,
                                  const pool_q_t& qparam) {
#pragma HLS INLINE off
    act_vec_t row0[AVGPOOL_CACHE_WORDS];
    act_vec_t row1[AVGPOOL_CACHE_WORDS];
    act_vec_t row2[AVGPOOL_CACHE_WORDS];
#pragma HLS ARRAY_PARTITION variable=row0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=row1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=row2 complete dim=1
    const bool rgb = src.c.to_uint() == 3U;
    const unsigned channels = rgb ? 3U : 1U;
    for (int oh = 0; oh < MAX_FM_H; ++oh) {
#pragma HLS PIPELINE off
        if (oh >= static_cast<int>(dst.h.to_uint())) {
            break;
        }
        for (int group = 0; group < (MAX_FM_W + 31) / 32; ++group) {
#pragma HLS PIPELINE off
            const int ow_begin = group * AVGPOOL_GROUP_PIXELS;
            if (ow_begin >= static_cast<int>(dst.w.to_uint())) {
                break;
            }
            u8_t byte0, byte1, byte2, left0, left1, left2, valid0, valid1, valid2;
            const int base_h = oh * 2 - 1;
            const int base_w = ow_begin * 2 - 1;
            if (!read_pool_group_cache_row(src, base_h, base_w, row0, byte0, left0, valid0) ||
                !read_pool_group_cache_row(src, base_h + 1, base_w, row1, byte1, left1, valid1) ||
                !read_pool_group_cache_row(src, base_h + 2, base_w, row2, byte2, left2, valid2)) {
                return false;
            }
            const int remaining = static_cast<int>(dst.w.to_uint()) - ow_begin;
            const int pixels = remaining < AVGPOOL_GROUP_PIXELS ? remaining : AVGPOOL_GROUP_PIXELS;
            act_vec_t packed[3];
#pragma HLS ARRAY_PARTITION variable=packed complete dim=1
            compute_pool_group_words(row0, row1, row2,
                byte0, left0, valid0, byte1, left1, valid1,
                byte2, left2, valid2, qparam, pixels, rgb, packed);
            const u32_t word_offset = dst.base_offset +
                static_cast<u32_t>(oh) * tensor_row_stride_bytes(dst) + ow_begin * channels;
            const unsigned words = (static_cast<unsigned>(pixels) * channels + 31U) >> 5;
            for (int wi = 0; wi < 3; ++wi) {
#pragma HLS PIPELINE off
                if (static_cast<unsigned>(wi) >= words) { break; }
                if (!on_chip_memory_write_pool_bram_word(dst.bank_id,
                    word_offset + wi * AXI_WORD_BYTES, packed[wi])) {
                    return false;
                }
            }
        }
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
    return avgpool_group_engine(src, dst, qparam);
}

}  // namespace esp_int8
