#include "../include/npu_q.hpp"
#include "../include/npu_config.hpp"

namespace esp_int8 {

bool on_chip_memory_read_packed_contiguous(const tensor_desc_t& desc,
                                           i32_t h,
                                           i32_t w,
                                           u16_t c_begin,
                                           u8_t valid_bytes,
                                           act_vec_t& packed);
bool on_chip_memory_write_packed_tile(const tensor_desc_t& desc,
                                      u16_t h,
                                      u16_t w,
                                      u16_t c_begin,
                                      u8_t valid_c,
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

static bool avgpool_pixel_c3_generic(const tensor_desc_t& src,
                                     const tensor_desc_t& dst,
                                     const pool_q_t& qparam,
                                     u16_t oh,
                                     u16_t ow) {
#pragma HLS INLINE off
    i32_t sum[3];
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
            if (!on_chip_memory_read_packed_contiguous(
                    src, ih, iw, 0, static_cast<u8_t>(3), in_packed)) {
                return false;
            }
            for (int c = 0; c < 3; ++c) {
#pragma HLS UNROLL
                sum[c] += static_cast<i32_t>(packed_i8(in_packed, c));
            }
        }
    }

    act_vec_t out_packed = 0;
    for (int c = 0; c < 3; ++c) {
#pragma HLS UNROLL
        set_packed_i8(out_packed, c, requant_pool_avg(round_div9(sum[c]), qparam));
    }
    return on_chip_memory_write_packed_tile(dst, oh, ow, 0, static_cast<u8_t>(3), out_packed);
}

static bool read_c3_fast_rows(const tensor_desc_t& src,
                              i32_t base_h,
                              i32_t base_w,
                              act_vec_t& row0,
                              act_vec_t& row1,
                              act_vec_t& row2) {
#pragma HLS INLINE off
    if (!on_chip_memory_read_packed_contiguous(
            src, base_h + 0, base_w, 0, static_cast<u8_t>(9), row0)) {
        return false;
    }
    if (!on_chip_memory_read_packed_contiguous(
            src, base_h + 1, base_w, 0, static_cast<u8_t>(9), row1)) {
        return false;
    }
    if (!on_chip_memory_read_packed_contiguous(
            src, base_h + 2, base_w, 0, static_cast<u8_t>(9), row2)) {
        return false;
    }
    return true;
}

static void sum_c3_fast_rows(act_vec_t row0,
                             act_vec_t row1,
                             act_vec_t row2,
                             i32_t sum[3]) {
#pragma HLS INLINE off
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

static bool write_c3_avg_pixel(const tensor_desc_t& dst,
                               const pool_q_t& qparam,
                               u16_t oh,
                               u16_t ow,
                               const i32_t sum[3]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=sum complete dim=1
    act_vec_t out_packed = 0;
    for (int c = 0; c < 3; ++c) {
#pragma HLS UNROLL
        set_packed_i8(out_packed, c, requant_pool_avg(round_div9(sum[c]), qparam));
    }
    return on_chip_memory_write_packed_tile(dst, oh, ow, 0, static_cast<u8_t>(3), out_packed);
}

static bool avgpool_pixel_c3_inner_fast(const tensor_desc_t& src,
                                        const tensor_desc_t& dst,
                                        const pool_q_t& qparam,
                                        u16_t oh,
                                        u16_t ow) {
#pragma HLS INLINE off
    const i32_t base_h = static_cast<i32_t>(oh) * 2 - 1;
    const i32_t base_w = static_cast<i32_t>(ow) * 2 - 1;
    act_vec_t row0 = 0;
    act_vec_t row1 = 0;
    act_vec_t row2 = 0;
    i32_t sum[3];
#pragma HLS ARRAY_PARTITION variable=sum complete dim=1

    if (!read_c3_fast_rows(src, base_h, base_w, row0, row1, row2)) {
        return false;
    }

    sum_c3_fast_rows(row0, row1, row2, sum);
    return write_c3_avg_pixel(dst, qparam, oh, ow, sum);
}

static bool avgpool_unit_c3_fast(const tensor_desc_t& src,
                                 const tensor_desc_t& dst,
                                 const pool_q_t& qparam) {
#pragma HLS INLINE off
    const int out_h_i = static_cast<int>(dst.h.to_uint());
    const int out_w_i = static_cast<int>(dst.w.to_uint());
    const int src_h_i = static_cast<int>(src.h.to_uint());
    const int src_w_i = static_cast<int>(src.w.to_uint());
    int fast_h_end = (src_h_i >= 2) ? ((src_h_i - 2) / 2 + 1) : 0;
    int fast_w_end = (src_w_i >= 2) ? ((src_w_i - 2) / 2 + 1) : 0;
    if (fast_h_end > out_h_i) {
        fast_h_end = out_h_i;
    }
    if (fast_w_end > out_w_i) {
        fast_w_end = out_w_i;
    }

    for (int oh_i = 0; oh_i < MAX_FM_H; ++oh_i) {
        if (oh_i >= out_h_i) {
            break;
        }
        const u16_t oh = static_cast<u16_t>(oh_i);
        if (oh_i == 0 || oh_i >= fast_h_end) {
            for (int ow_i = 0; ow_i < MAX_FM_W; ++ow_i) {
                if (ow_i >= out_w_i) {
                    break;
                }
                if (!avgpool_pixel_c3_generic(src, dst, qparam, oh, static_cast<u16_t>(ow_i))) {
                    return false;
                }
            }
            continue;
        }

        if (!avgpool_pixel_c3_generic(src, dst, qparam, oh, static_cast<u16_t>(0))) {
            return false;
        }
        for (int ow_i = 1; ow_i < MAX_FM_W; ++ow_i) {
#pragma HLS PIPELINE II=1
            if (ow_i >= fast_w_end) {
                break;
            }
            if (!avgpool_pixel_c3_inner_fast(src, dst, qparam, oh, static_cast<u16_t>(ow_i))) {
                return false;
            }
        }
        for (int ow_i = fast_w_end; ow_i < MAX_FM_W; ++ow_i) {
            if (ow_i >= out_w_i) {
                break;
            }
            if (!avgpool_pixel_c3_generic(src, dst, qparam, oh, static_cast<u16_t>(ow_i))) {
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
