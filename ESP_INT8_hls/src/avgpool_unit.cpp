#include "../include/npu_q.hpp"
#include "../include/npu_config.hpp"

namespace esp_int8 {

bool on_chip_memory_read_fmbuf_abs_word(u8_t bank_id,
                                        u32_t byte_offset,
                                        act_vec_t& packed);
bool on_chip_memory_read_pool2_abs_word(u32_t byte_offset,
                                        act_vec_t& packed);
bool on_chip_memory_write_fmbuf_abs_word(u8_t bank_id,
                                         u32_t byte_offset,
                                         act_vec_t packed);
bool on_chip_memory_write_pool2_abs_word(u32_t byte_offset,
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
    return (src_bank.to_uint() == static_cast<unsigned>(BANK_BRAM_SCR1))
               ? on_chip_memory_read_pool2_abs_word(byte_offset, word)
               : on_chip_memory_read_fmbuf_abs_word(src_bank, byte_offset, word);
}

static bool write_avgpool_aligned_abs_word(u8_t dst_bank,
                                           u32_t byte_offset,
                                           act_vec_t word) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off
    return (dst_bank.to_uint() == static_cast<unsigned>(BANK_BRAM_SCR1))
               ? on_chip_memory_write_pool2_abs_word(byte_offset, word)
               : on_chip_memory_write_fmbuf_abs_word(dst_bank, byte_offset, word);
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

template <int BYTE_COUNT>
static bool read_avgpool_c3_row_bytes(const tensor_desc_t& src,
                                      i32_t h,
                                      i32_t w,
                                      act_vec_t& packed) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off
    packed = 0;

    const bool spatial_valid = (h >= 0 && w >= 0 &&
                                h < static_cast<i32_t>(src.h) &&
                                w < static_cast<i32_t>(src.w));
    if (!spatial_valid) {
        return true;
    }

    const unsigned phys_c = pool_desc_phys_c(src).to_uint();
    const unsigned c_offset = src.reserved1.to_uint();
    if (phys_c == 0U || c_offset >= phys_c) {
        return true;
    }

    const unsigned w_u = static_cast<unsigned>(static_cast<u16_t>(w));
    const unsigned remaining_row_bytes =
        (src.w.to_uint() - w_u) * phys_c - c_offset;
    const unsigned read_count =
        (BYTE_COUNT < static_cast<int>(remaining_row_bytes))
            ? static_cast<unsigned>(BYTE_COUNT)
            : remaining_row_bytes;
    if (read_count == 0U) {
        return true;
    }

    const u32_t start_offset =
        src.base_offset +
        (static_cast<u32_t>(static_cast<u16_t>(h)) * static_cast<u32_t>(src.w) +
         static_cast<u32_t>(static_cast<u16_t>(w))) *
            static_cast<u32_t>(phys_c) +
        static_cast<u32_t>(c_offset);
    const u32_t word0_offset = start_offset & static_cast<u32_t>(~(AXI_WORD_BYTES - 1));
    const unsigned byte0 = start_offset.to_uint() & static_cast<unsigned>(AXI_WORD_BYTES - 1);

    act_vec_t word0 = 0;
    act_vec_t word1 = 0;
    if (!read_avgpool_abs_word(src.bank_id, word0_offset, word0)) {
        return false;
    }
    const bool crosses_word = (byte0 + read_count) > static_cast<unsigned>(AXI_WORD_BYTES);
    if (crosses_word) {
        if (!read_avgpool_abs_word(src.bank_id, word0_offset + AXI_WORD_BYTES, word1)) {
            return false;
        }
    }

    for (int i = 0; i < BYTE_COUNT; ++i) {
#pragma HLS UNROLL
        u8_t raw = 0;
        if (static_cast<unsigned>(i) < read_count) {
            const unsigned src_byte = byte0 + static_cast<unsigned>(i);
            raw = (src_byte < static_cast<unsigned>(AXI_WORD_BYTES))
                      ? word0.range(src_byte * 8U + 7U, src_byte * 8U)
                      : word1.range((src_byte - AXI_WORD_BYTES) * 8U + 7U,
                                    (src_byte - AXI_WORD_BYTES) * 8U);
        }
        packed.range(i * 8 + 7, i * 8) = raw;
    }

    return true;
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

static bool avgpool_pixel_c3_generic_sum(const tensor_desc_t& src,
                                         u16_t oh,
                                         u16_t ow,
                                         i32_t sum[3]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=sum complete dim=1

    for (int c = 0; c < 3; ++c) {
#pragma HLS UNROLL
        sum[c] = 0;
    }

    for (int kh = 0; kh < 3; ++kh) {
        for (int kw = 0; kw < 3; ++kw) {
#pragma HLS PIPELINE off
            act_vec_t in_packed = 0;
            const i32_t ih = static_cast<i32_t>(oh) * 2 + kh - 1;
            const i32_t iw = static_cast<i32_t>(ow) * 2 + kw - 1;
            if (!read_avgpool_c3_row_bytes<3>(src, ih, iw, in_packed)) {
                return false;
            }
            for (int c = 0; c < 3; ++c) {
#pragma HLS UNROLL
                sum[c] += static_cast<i32_t>(packed_i8(in_packed, c));
            }
        }
    }

    return true;
}

static bool read_c3_fast_rows(const tensor_desc_t& src,
                              i32_t base_h,
                              i32_t base_w,
                              act_vec_t& row0,
                              act_vec_t& row1,
                              act_vec_t& row2) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off
    if (!read_avgpool_c3_row_bytes<9>(src, base_h + 0, base_w, row0)) {
        return false;
    }
    if (!read_avgpool_c3_row_bytes<9>(src, base_h + 1, base_w, row1)) {
        return false;
    }
    if (!read_avgpool_c3_row_bytes<9>(src, base_h + 2, base_w, row2)) {
        return false;
    }
    return true;
}

static void sum_c3_fast_rows(act_vec_t row0,
                             act_vec_t row1,
                             act_vec_t row2,
                             i32_t sum[3]) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off
#pragma HLS ARRAY_PARTITION variable=sum complete dim=1
    i32_t row_sum0[3];
    i32_t row_sum1[3];
    i32_t row_sum2[3];
#pragma HLS ARRAY_PARTITION variable=row_sum0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=row_sum1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=row_sum2 complete dim=1

    for (int c = 0; c < 3; ++c) {
#pragma HLS UNROLL
        row_sum0[c] = static_cast<i32_t>(packed_i8(row0, c)) +
                      static_cast<i32_t>(packed_i8(row0, c + 3)) +
                      static_cast<i32_t>(packed_i8(row0, c + 6));
        row_sum1[c] = static_cast<i32_t>(packed_i8(row1, c)) +
                      static_cast<i32_t>(packed_i8(row1, c + 3)) +
                      static_cast<i32_t>(packed_i8(row1, c + 6));
        row_sum2[c] = static_cast<i32_t>(packed_i8(row2, c)) +
                      static_cast<i32_t>(packed_i8(row2, c + 3)) +
                      static_cast<i32_t>(packed_i8(row2, c + 6));
    }

    for (int c = 0; c < 3; ++c) {
#pragma HLS UNROLL
        sum[c] = row_sum0[c] + row_sum1[c] + row_sum2[c];
    }
}

static bool avgpool_pixel_c3_inner_fast_sum(const tensor_desc_t& src,
                                            u16_t oh,
                                            u16_t ow,
                                            i32_t sum[3]) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off
    const i32_t base_h = static_cast<i32_t>(oh) * 2 - 1;
    const i32_t base_w = static_cast<i32_t>(ow) * 2 - 1;
    act_vec_t row0 = 0;
    act_vec_t row1 = 0;
    act_vec_t row2 = 0;
#pragma HLS ARRAY_PARTITION variable=sum complete dim=1

    if (!read_c3_fast_rows(src, base_h, base_w, row0, row1, row2)) {
        return false;
    }

    sum_c3_fast_rows(row0, row1, row2, sum);
    return true;
}

static bool avgpool_pixel_c3_pack(const tensor_desc_t& src,
                                  const pool_q_t& qparam,
                                  int oh_i,
                                  int ow_i,
                                  bool use_inner_fast,
                                  act_vec_t& out_packed) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off
    i32_t sum[3];
#pragma HLS ARRAY_PARTITION variable=sum complete dim=1

    const u16_t oh = static_cast<u16_t>(oh_i);
    const u16_t ow = static_cast<u16_t>(ow_i);
    const bool ok = use_inner_fast
                        ? avgpool_pixel_c3_inner_fast_sum(src, oh, ow, sum)
                        : avgpool_pixel_c3_generic_sum(src, oh, ow, sum);
    if (!ok) {
      return false;
    }

    quantize_c3_avg_pixel(sum, qparam, out_packed);
    return true;
}

static bool write_c3_group32(u8_t dst_bank,
                             u32_t word_offset,
                             const act_vec_t pixels[32]) {
#pragma HLS INLINE
    act_vec_t w0 = 0;
    copy_c3_segment<0, 0, 3>(w0, pixels[0]);
    copy_c3_segment<3, 0, 3>(w0, pixels[1]);
    copy_c3_segment<6, 0, 3>(w0, pixels[2]);
    copy_c3_segment<9, 0, 3>(w0, pixels[3]);
    copy_c3_segment<12, 0, 3>(w0, pixels[4]);
    copy_c3_segment<15, 0, 3>(w0, pixels[5]);
    copy_c3_segment<18, 0, 3>(w0, pixels[6]);
    copy_c3_segment<21, 0, 3>(w0, pixels[7]);
    copy_c3_segment<24, 0, 3>(w0, pixels[8]);
    copy_c3_segment<27, 0, 3>(w0, pixels[9]);
    copy_c3_segment<30, 0, 2>(w0, pixels[10]);

    act_vec_t w1 = 0;
    copy_c3_segment<0, 2, 1>(w1, pixels[10]);
    copy_c3_segment<1, 0, 3>(w1, pixels[11]);
    copy_c3_segment<4, 0, 3>(w1, pixels[12]);
    copy_c3_segment<7, 0, 3>(w1, pixels[13]);
    copy_c3_segment<10, 0, 3>(w1, pixels[14]);
    copy_c3_segment<13, 0, 3>(w1, pixels[15]);
    copy_c3_segment<16, 0, 3>(w1, pixels[16]);
    copy_c3_segment<19, 0, 3>(w1, pixels[17]);
    copy_c3_segment<22, 0, 3>(w1, pixels[18]);
    copy_c3_segment<25, 0, 3>(w1, pixels[19]);
    copy_c3_segment<28, 0, 3>(w1, pixels[20]);
    copy_c3_segment<31, 0, 1>(w1, pixels[21]);

    act_vec_t w2 = 0;
    copy_c3_segment<0, 1, 2>(w2, pixels[21]);
    copy_c3_segment<2, 0, 3>(w2, pixels[22]);
    copy_c3_segment<5, 0, 3>(w2, pixels[23]);
    copy_c3_segment<8, 0, 3>(w2, pixels[24]);
    copy_c3_segment<11, 0, 3>(w2, pixels[25]);
    copy_c3_segment<14, 0, 3>(w2, pixels[26]);
    copy_c3_segment<17, 0, 3>(w2, pixels[27]);
    copy_c3_segment<20, 0, 3>(w2, pixels[28]);
    copy_c3_segment<23, 0, 3>(w2, pixels[29]);
    copy_c3_segment<26, 0, 3>(w2, pixels[30]);
    copy_c3_segment<29, 0, 3>(w2, pixels[31]);

    if (!write_avgpool_aligned_abs_word(dst_bank, word_offset, w0)) {
        return false;
    }
    if (!write_avgpool_aligned_abs_word(dst_bank, word_offset + static_cast<u32_t>(32), w1)) {
        return false;
    }
    return write_avgpool_aligned_abs_word(dst_bank, word_offset + static_cast<u32_t>(64), w2);
}

static bool avgpool_c3_group32_pack_write(const tensor_desc_t& src,
                                          const tensor_desc_t& dst,
                                          const pool_q_t& qparam,
                                          int oh_i,
                                          int ow_start,
                                          u32_t word_offset,
                                          bool use_inner_fast) {
#pragma HLS INLINE off
    act_vec_t pixels[32];
#pragma HLS ARRAY_PARTITION variable=pixels complete dim=1

    for (int pix = 0; pix < 32; ++pix) {
#pragma HLS PIPELINE off
        act_vec_t out_packed = 0;
        if (!avgpool_pixel_c3_pack(src,
                                   qparam,
                                   oh_i,
                                   ow_start + pix,
                                   use_inner_fast,
                                   out_packed)) {
            return false;
        }
        pixels[pix] = out_packed;
    }

    return write_c3_group32(dst.bank_id, word_offset, pixels);
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
            const bool use_inner_fast = (oh_i > 0) && (group > 0);
            if (!avgpool_c3_group32_pack_write(src,
                                               dst,
                                               qparam,
                                               oh_i,
                                               ow_start,
                                               word_offset,
                                               use_inner_fast)) {
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
