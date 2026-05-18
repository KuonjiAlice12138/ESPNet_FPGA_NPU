#include "../include/npu_q.hpp"
#include "../include/npu_config.hpp"

namespace esp_int8 {

bool on_chip_memory_read_tile(const tensor_desc_t& desc,
                              i32_t h,
                              i32_t w,
                              u16_t c_begin,
                              u8_t valid_c,
                              i8_t tile[TM]);
bool on_chip_memory_write_tile(const tensor_desc_t& desc,
                               u16_t h,
                               u16_t w,
                               u16_t c_begin,
                               u8_t valid_c,
                               const i8_t tile[TM]);
bool on_chip_memory_read_packed_contiguous(const tensor_desc_t& desc,
                                           i32_t h,
                                           i32_t w,
                                           u16_t c_begin,
                                           u8_t valid_bytes,
                                           act_vec_t& packed);

static u8_t pool_lanes(u16_t remaining_c) {
#pragma HLS INLINE
    const unsigned rem = remaining_c.to_uint();
    return static_cast<u8_t>((rem > static_cast<unsigned>(TM)) ? TM : rem);
}

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
    i8_t out_tile[TM];
#pragma HLS ARRAY_PARTITION variable=sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=out_tile complete dim=1

    for (int lane = 0; lane < TM; ++lane) {
#pragma HLS UNROLL
        out_tile[lane] = 0;
    }
    for (int c = 0; c < 3; ++c) {
#pragma HLS UNROLL
        sum[c] = 0;
    }

    for (int kh = 0; kh < 3; ++kh) {
        for (int kw = 0; kw < 3; ++kw) {
            i8_t in_tile[TM];
#pragma HLS ARRAY_PARTITION variable=in_tile complete dim=1
            const i32_t ih = static_cast<i32_t>(oh) * 2 + kh - 1;
            const i32_t iw = static_cast<i32_t>(ow) * 2 + kw - 1;
            if (!on_chip_memory_read_tile(src, ih, iw, 0, static_cast<u8_t>(3), in_tile)) {
                return false;
            }
            for (int c = 0; c < 3; ++c) {
#pragma HLS UNROLL
                sum[c] += static_cast<i32_t>(in_tile[c]);
            }
        }
    }

    for (int c = 0; c < 3; ++c) {
#pragma HLS UNROLL
        out_tile[c] = requant_pool_avg(round_div9(sum[c]), qparam);
    }
    return on_chip_memory_write_tile(dst, oh, ow, 0, static_cast<u8_t>(3), out_tile);
}

static void accumulate_c3_row(act_vec_t row, i32_t sum[3]) {
#pragma HLS INLINE
    for (int px = 0; px < 3; ++px) {
#pragma HLS UNROLL
        for (int c = 0; c < 3; ++c) {
#pragma HLS UNROLL
            sum[c] += static_cast<i32_t>(packed_i8(row, px * 3 + c));
        }
    }
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
    i8_t out_tile[TM];
#pragma HLS ARRAY_PARTITION variable=sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=out_tile complete dim=1

    for (int lane = 0; lane < TM; ++lane) {
#pragma HLS UNROLL
        out_tile[lane] = 0;
    }
    for (int c = 0; c < 3; ++c) {
#pragma HLS UNROLL
        sum[c] = 0;
    }

    if (!on_chip_memory_read_packed_contiguous(
            src, base_h + 0, base_w, 0, static_cast<u8_t>(9), row0) ||
        !on_chip_memory_read_packed_contiguous(
            src, base_h + 1, base_w, 0, static_cast<u8_t>(9), row1) ||
        !on_chip_memory_read_packed_contiguous(
            src, base_h + 2, base_w, 0, static_cast<u8_t>(9), row2)) {
        return false;
    }

    accumulate_c3_row(row0, sum);
    accumulate_c3_row(row1, sum);
    accumulate_c3_row(row2, sum);

    for (int c = 0; c < 3; ++c) {
#pragma HLS UNROLL
        out_tile[c] = requant_pool_avg(round_div9(sum[c]), qparam);
    }
    return on_chip_memory_write_tile(dst, oh, ow, 0, static_cast<u8_t>(3), out_tile);
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

    if (src.c.to_uint() == 3U && dst.c.to_uint() == 3U) {
        return avgpool_unit_c3_fast(src, dst, qparam);
    }

    const int out_h_i = static_cast<int>(dst.h.to_uint());
    const int out_w_i = static_cast<int>(dst.w.to_uint());
    const int c_blocks = static_cast<int>((dst.c.to_uint() + static_cast<unsigned>(TM) - 1U) /
                                          static_cast<unsigned>(TM));

    for (int oh_i = 0; oh_i < MAX_FM_H; ++oh_i) {
        if (oh_i >= out_h_i) {
            break;
        }
        const u16_t oh = static_cast<u16_t>(oh_i);
        for (int ow_i = 0; ow_i < MAX_FM_W; ++ow_i) {
            if (ow_i >= out_w_i) {
                break;
            }
            const u16_t ow = static_cast<u16_t>(ow_i);
            for (int c_blk = 0; c_blk < MAX_C_TILE_COUNT; ++c_blk) {
#pragma HLS PIPELINE off
                if (c_blk >= c_blocks) {
                    break;
                }
                const u16_t cb = static_cast<u16_t>(c_blk * TM);
                const u16_t remaining = static_cast<u16_t>(dst.c - cb);
                const u8_t lanes = pool_lanes(remaining);
                i32_t sum[TM];
                i8_t out_tile[TM];
#pragma HLS ARRAY_PARTITION variable=sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=out_tile complete dim=1

                for (int lane = 0; lane < TM; ++lane) {
#pragma HLS UNROLL
                    sum[lane] = 0;
                    out_tile[lane] = 0;
                }

                for (int kh = 0; kh < 3; ++kh) {
                    for (int kw = 0; kw < 3; ++kw) {
                        i8_t in_tile[TM];
#pragma HLS ARRAY_PARTITION variable=in_tile complete dim=1
                        const i32_t ih = static_cast<i32_t>(oh) * 2 + kh - 1;
                        const i32_t iw = static_cast<i32_t>(ow) * 2 + kw - 1;
                        if (!on_chip_memory_read_tile(src, ih, iw, cb, lanes, in_tile)) {
                            return false;
                        }
                        for (int lane = 0; lane < TM; ++lane) {
#pragma HLS UNROLL
                            if (static_cast<unsigned>(lane) < lanes.to_uint()) {
                                sum[lane] += static_cast<i32_t>(in_tile[lane]);
                            }
                        }
                    }
                }

                for (int lane = 0; lane < TM; ++lane) {
#pragma HLS UNROLL
                    if (static_cast<unsigned>(lane) < lanes.to_uint()) {
                        out_tile[lane] = requant_pool_avg(round_div9(sum[lane]), qparam);
                    }
                }

                if (!on_chip_memory_write_tile(dst, oh, ow, cb, lanes, out_tile)) {
                    return false;
                }
            }
        }
    }

    return true;
}

void avgpool_unit(
    const tensor_desc_t& src,
    const tensor_desc_t& dst,
    const pool_q_t& qparam,
    i8_t* fmbuf_base) {
    (void)avgpool_unit_checked(src, dst, qparam, fmbuf_base);
}

}  // namespace esp_int8
