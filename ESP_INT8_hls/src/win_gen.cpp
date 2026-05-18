#include "../include/npu_config.hpp"
#include "../include/npu_types.hpp"

namespace esp_int8 {

bool on_chip_memory_read_tile(const tensor_desc_t& desc,
                              i32_t h,
                              i32_t w,
                              u16_t c_begin,
                              u8_t valid_c,
                              i8_t tile[TM]);

bool on_chip_memory_read_packed_tile(const tensor_desc_t& desc,
                                     i32_t h,
                                     i32_t w,
                                     u16_t c_begin,
                                     u8_t valid_c,
                                     act_vec_t& packed);

bool on_chip_memory_read_packed_contiguous(const tensor_desc_t& desc,
                                           i32_t h,
                                           i32_t w,
                                           u16_t c_begin,
                                           u8_t valid_bytes,
                                           act_vec_t& packed);

bool on_chip_memory_read_aligned_full_tile(const tensor_desc_t& desc,
                                           i32_t h,
                                           i32_t w,
                                           u16_t c_begin,
                                           act_vec_t& packed);

static u16_t ceil_div_u16(u16_t a, u16_t b) {
#pragma HLS INLINE
    return static_cast<u16_t>((a + b - 1) / b);
}

static u16_t effective_stride(const conv_cfg_t& cfg) {
#pragma HLS INLINE
    return (cfg.stride == 0) ? static_cast<u16_t>(1) : static_cast<u16_t>(cfg.stride);
}

static u16_t effective_dilation(const conv_cfg_t& cfg) {
#pragma HLS INLINE
    return (cfg.dilation == 0) ? static_cast<u16_t>(1) : static_cast<u16_t>(cfg.dilation);
}

static u16_t effective_kernel(const conv_cfg_t& cfg) {
#pragma HLS INLINE
    return (cfg.kernel == 1) ? static_cast<u16_t>(1) : static_cast<u16_t>(3);
}

static u16_t conv_out_dim(u16_t in_size, u16_t stride) {
#pragma HLS INLINE
    return ceil_div_u16(in_size, stride);
}

static u16_t kernel_flat(const conv_cfg_t& cfg) {
#pragma HLS INLINE
    const u16_t k = effective_kernel(cfg);
    return static_cast<u16_t>(cfg.in_c * k * k);
}

#ifndef __SYNTHESIS__
static bool map_k_index(const conv_cfg_t& cfg,
                        u16_t oh,
                        u16_t ow,
                        u16_t k_idx,
                        i32_t& ih,
                        i32_t& iw,
                        u16_t& cin) {
#pragma HLS INLINE
    const u16_t k_total = kernel_flat(cfg);
    if (k_idx >= k_total) {
        ih = 0;
        iw = 0;
        cin = 0;
        return false;
    }

    const u16_t kernel = effective_kernel(cfg);
    const u16_t stride = effective_stride(cfg);
    const u16_t dilation = effective_dilation(cfg);

    if (kernel == 1) {
        cin = k_idx;
        ih = static_cast<i32_t>(oh * stride);
        iw = static_cast<i32_t>(ow * stride);
    } else {
        cin = static_cast<u16_t>(k_idx % cfg.in_c);
        const u16_t spatial_idx = static_cast<u16_t>(k_idx / cfg.in_c);
        const u16_t kh = static_cast<u16_t>(spatial_idx / kernel);
        const u16_t kw = static_cast<u16_t>(spatial_idx % kernel);
        const i32_t pad = static_cast<i32_t>(dilation);
        ih = static_cast<i32_t>(oh * stride) + static_cast<i32_t>(kh * dilation) - pad;
        iw = static_cast<i32_t>(ow * stride) + static_cast<i32_t>(kw * dilation) - pad;
    }

    if (cin >= cfg.in_c ||
        ih < 0 || iw < 0 ||
        ih >= static_cast<i32_t>(cfg.in_h) ||
        iw >= static_cast<i32_t>(cfg.in_w)) {
        return false;
    }
    return true;
}
#endif

static void set_vec_i8(act_vec_t& word, int lane, i8_t value) {
#pragma HLS INLINE
    word.range(lane * 8 + 7, lane * 8) = value.range(7, 0);
}

static void set_vec_i8_dynamic(act_vec_t& word, int lane, i8_t value) {
#pragma HLS INLINE
    u8_t raw = 0;
    raw.range(7, 0) = value.range(7, 0);
    const act_vec_t widened = static_cast<act_vec_t>(raw);
    word |= static_cast<act_vec_t>(widened << (lane * 8));
}

static i8_t packed_vec_i8(act_vec_t word, int lane) {
#pragma HLS INLINE
    const act_vec_t shifted = static_cast<act_vec_t>(word >> (lane * 8));
    const u8_t raw = shifted.range(7, 0);
    i8_t value;
    value.range(7, 0) = raw;
    return value;
}

static act_vec_t low_byte_mask(int count) {
#pragma HLS INLINE
    if (count <= 0) {
        return 0;
    }
    if (count >= AXI_WORD_BYTES) {
        return ~act_vec_t(0);
    }
    return static_cast<act_vec_t>((static_cast<act_vec_t>(1) << (count * 8)) - 1);
}

static void insert_packed_segment(act_vec_t& word,
                                  int lane_offset,
                                  int count,
                                  act_vec_t segment) {
#pragma HLS INLINE
    const act_vec_t mask = static_cast<act_vec_t>(low_byte_mask(count) << (lane_offset * 8));
    const act_vec_t shifted = static_cast<act_vec_t>(segment << (lane_offset * 8));
    word = static_cast<act_vec_t>((word & ~mask) | (shifted & mask));
}

template <int DST_LANE, int SRC_LANE, int COUNT>
static void copy_const_segment(act_vec_t& dst, const act_vec_t& src) {
#pragma HLS INLINE
    dst.range(DST_LANE * 8 + COUNT * 8 - 1, DST_LANE * 8) =
        src.range(SRC_LANE * 8 + COUNT * 8 - 1, SRC_LANE * 8);
}

static bool is_first_layer_3x3(const conv_cfg_t& cfg) {
#pragma HLS INLINE
    return cfg.in_c.to_uint() == 3U &&
           cfg.kernel.to_uint() == 3U &&
           cfg.stride.to_uint() == 2U &&
           cfg.dilation.to_uint() == 1U;
}

static void emit_first_layer_c3_word(const act_vec_t spatial_word[9],
                                     hls::stream<act_vec_t>& act_stream) {
#pragma HLS INLINE
    act_vec_t word = 0;
    copy_const_segment<0, 0, 3>(word, spatial_word[0]);
    copy_const_segment<3, 0, 3>(word, spatial_word[1]);
    copy_const_segment<6, 0, 3>(word, spatial_word[2]);
    copy_const_segment<9, 0, 3>(word, spatial_word[3]);
    copy_const_segment<12, 0, 3>(word, spatial_word[4]);
    copy_const_segment<15, 0, 3>(word, spatial_word[5]);
    copy_const_segment<18, 0, 3>(word, spatial_word[6]);
    copy_const_segment<21, 0, 3>(word, spatial_word[7]);
    copy_const_segment<24, 0, 3>(word, spatial_word[8]);
    act_stream.write(word);
}

static void load_first_layer_3x3_col(const tensor_desc_t& src_desc,
                                     i32_t base_h,
                                     i32_t iw,
                                     int col,
                                     act_vec_t spatial_word[9]) {
#pragma HLS INLINE
    for (int kh = 0; kh < 3; ++kh) {
#pragma HLS UNROLL
        const i32_t ih = base_h + static_cast<i32_t>(kh);
        on_chip_memory_read_packed_tile(src_desc,
                                        ih,
                                        iw,
                                        0,
                                        static_cast<u8_t>(3),
                                        spatial_word[kh * 3 + col]);
    }
}

static void emit_first_layer_c3_border_word(const tensor_desc_t& src_desc,
                                            hls::stream<act_vec_t>& act_stream,
                                            i32_t base_h,
                                            i32_t base_w) {
#pragma HLS INLINE
    act_vec_t spatial_word[9];
#pragma HLS ARRAY_PARTITION variable=spatial_word complete dim=1
    load_first_layer_3x3_col(src_desc, base_h, base_w + 0, 0, spatial_word);
    load_first_layer_3x3_col(src_desc, base_h, base_w + 1, 1, spatial_word);
    load_first_layer_3x3_col(src_desc, base_h, base_w + 2, 2, spatial_word);
    emit_first_layer_c3_word(spatial_word, act_stream);
}

static void emit_first_layer_c3_row_segment_word(const tensor_desc_t& src_desc,
                                                 hls::stream<act_vec_t>& act_stream,
                                                 i32_t base_h,
                                                 i32_t base_w) {
#pragma HLS INLINE
    act_vec_t row0 = 0;
    act_vec_t row1 = 0;
    act_vec_t row2 = 0;
    on_chip_memory_read_packed_contiguous(
        src_desc, base_h + 0, base_w, 0, static_cast<u8_t>(9), row0);
    on_chip_memory_read_packed_contiguous(
        src_desc, base_h + 1, base_w, 0, static_cast<u8_t>(9), row1);
    on_chip_memory_read_packed_contiguous(
        src_desc, base_h + 2, base_w, 0, static_cast<u8_t>(9), row2);

    act_vec_t word = 0;
    copy_const_segment<0, 0, 9>(word, row0);
    copy_const_segment<9, 0, 9>(word, row1);
    copy_const_segment<18, 0, 9>(word, row2);
    act_stream.write(word);
}

static void split_first_layer_c3_pair_row(const act_vec_t& segment,
                                          int row,
                                          act_vec_t spatial0[9],
                                          act_vec_t spatial1[9]) {
#pragma HLS INLINE
    act_vec_t px0 = 0;
    act_vec_t px1 = 0;
    act_vec_t px2 = 0;
    act_vec_t px3 = 0;
    act_vec_t px4 = 0;
    copy_const_segment<0, 0, 3>(px0, segment);
    copy_const_segment<0, 3, 3>(px1, segment);
    copy_const_segment<0, 6, 3>(px2, segment);
    copy_const_segment<0, 9, 3>(px3, segment);
    copy_const_segment<0, 12, 3>(px4, segment);
    spatial0[row * 3 + 0] = px0;
    spatial0[row * 3 + 1] = px1;
    spatial0[row * 3 + 2] = px2;
    spatial1[row * 3 + 0] = px2;
    spatial1[row * 3 + 1] = px3;
    spatial1[row * 3 + 2] = px4;
}

static void emit_first_layer_c3_pair_words(const tensor_desc_t& src_desc,
                                           hls::stream<act_vec_t>& act_stream,
                                           i32_t base_h,
                                           i32_t base_w) {
#pragma HLS INLINE
    act_vec_t spatial0[9];
    act_vec_t spatial1[9];
#pragma HLS ARRAY_PARTITION variable=spatial0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=spatial1 complete dim=1

    act_vec_t row0 = 0;
    act_vec_t row1 = 0;
    act_vec_t row2 = 0;
    on_chip_memory_read_packed_contiguous(
        src_desc, base_h + 0, base_w, 0, static_cast<u8_t>(15), row0);
    on_chip_memory_read_packed_contiguous(
        src_desc, base_h + 1, base_w, 0, static_cast<u8_t>(15), row1);
    on_chip_memory_read_packed_contiguous(
        src_desc, base_h + 2, base_w, 0, static_cast<u8_t>(15), row2);
    split_first_layer_c3_pair_row(row0, 0, spatial0, spatial1);
    split_first_layer_c3_pair_row(row1, 1, spatial0, spatial1);
    split_first_layer_c3_pair_row(row2, 2, spatial0, spatial1);
    emit_first_layer_c3_word(spatial0, act_stream);
    emit_first_layer_c3_word(spatial1, act_stream);
}

static bool is_aligned_full_channel_source(const tensor_desc_t& desc) {
#pragma HLS INLINE
    return ((desc.base_offset.to_uint() |
             desc.reserved0.to_uint() |
             desc.reserved1.to_uint()) &
           static_cast<unsigned>(AXI_WORD_BYTES - 1)) == 0U;
}

static void emit_smallc_c12_words(const act_vec_t spatial_word[9],
                                  hls::stream<act_vec_t>& act_stream) {
#pragma HLS INLINE
    act_vec_t w0 = 0;
    copy_const_segment<0, 0, 12>(w0, spatial_word[0]);
    copy_const_segment<12, 0, 12>(w0, spatial_word[1]);
    copy_const_segment<24, 0, 8>(w0, spatial_word[2]);
    act_stream.write(w0);

    act_vec_t w1 = 0;
    copy_const_segment<0, 8, 4>(w1, spatial_word[2]);
    copy_const_segment<4, 0, 12>(w1, spatial_word[3]);
    copy_const_segment<16, 0, 12>(w1, spatial_word[4]);
    copy_const_segment<28, 0, 4>(w1, spatial_word[5]);
    act_stream.write(w1);

    act_vec_t w2 = 0;
    copy_const_segment<0, 4, 8>(w2, spatial_word[5]);
    copy_const_segment<8, 0, 12>(w2, spatial_word[6]);
    copy_const_segment<20, 0, 12>(w2, spatial_word[7]);
    act_stream.write(w2);

    act_vec_t w3 = 0;
    copy_const_segment<0, 0, 12>(w3, spatial_word[8]);
    act_stream.write(w3);
}

static void emit_smallc_c19_words(const act_vec_t spatial_word[9],
                                  hls::stream<act_vec_t>& act_stream) {
#pragma HLS INLINE
    act_vec_t w0 = 0;
    copy_const_segment<0, 0, 19>(w0, spatial_word[0]);
    copy_const_segment<19, 0, 13>(w0, spatial_word[1]);
    act_stream.write(w0);

    act_vec_t w1 = 0;
    copy_const_segment<0, 13, 6>(w1, spatial_word[1]);
    copy_const_segment<6, 0, 19>(w1, spatial_word[2]);
    copy_const_segment<25, 0, 7>(w1, spatial_word[3]);
    act_stream.write(w1);

    act_vec_t w2 = 0;
    copy_const_segment<0, 7, 12>(w2, spatial_word[3]);
    copy_const_segment<12, 0, 19>(w2, spatial_word[4]);
    copy_const_segment<31, 0, 1>(w2, spatial_word[5]);
    act_stream.write(w2);

    act_vec_t w3 = 0;
    copy_const_segment<0, 1, 18>(w3, spatial_word[5]);
    copy_const_segment<18, 0, 14>(w3, spatial_word[6]);
    act_stream.write(w3);

    act_vec_t w4 = 0;
    copy_const_segment<0, 14, 5>(w4, spatial_word[6]);
    copy_const_segment<5, 0, 19>(w4, spatial_word[7]);
    copy_const_segment<24, 0, 8>(w4, spatial_word[8]);
    act_stream.write(w4);

    act_vec_t w5 = 0;
    copy_const_segment<0, 8, 11>(w5, spatial_word[8]);
    act_stream.write(w5);
}

static void emit_smallc_c25_words(const act_vec_t spatial_word[9],
                                  hls::stream<act_vec_t>& act_stream) {
#pragma HLS INLINE
    act_vec_t w0 = 0;
    copy_const_segment<0, 0, 25>(w0, spatial_word[0]);
    copy_const_segment<25, 0, 7>(w0, spatial_word[1]);
    act_stream.write(w0);

    act_vec_t w1 = 0;
    copy_const_segment<0, 7, 18>(w1, spatial_word[1]);
    copy_const_segment<18, 0, 14>(w1, spatial_word[2]);
    act_stream.write(w1);

    act_vec_t w2 = 0;
    copy_const_segment<0, 14, 11>(w2, spatial_word[2]);
    copy_const_segment<11, 0, 21>(w2, spatial_word[3]);
    act_stream.write(w2);

    act_vec_t w3 = 0;
    copy_const_segment<0, 21, 4>(w3, spatial_word[3]);
    copy_const_segment<4, 0, 25>(w3, spatial_word[4]);
    copy_const_segment<29, 0, 3>(w3, spatial_word[5]);
    act_stream.write(w3);

    act_vec_t w4 = 0;
    copy_const_segment<0, 3, 22>(w4, spatial_word[5]);
    copy_const_segment<22, 0, 10>(w4, spatial_word[6]);
    act_stream.write(w4);

    act_vec_t w5 = 0;
    copy_const_segment<0, 10, 15>(w5, spatial_word[6]);
    copy_const_segment<15, 0, 17>(w5, spatial_word[7]);
    act_stream.write(w5);

    act_vec_t w6 = 0;
    copy_const_segment<0, 17, 8>(w6, spatial_word[7]);
    copy_const_segment<8, 0, 24>(w6, spatial_word[8]);
    act_stream.write(w6);

    act_vec_t w7 = 0;
    copy_const_segment<0, 24, 1>(w7, spatial_word[8]);
    act_stream.write(w7);
}

static void emit_smallc_generic_words(const act_vec_t spatial_word[9],
                                      hls::stream<act_vec_t>& act_stream,
                                      int in_c_i,
                                      int k_total_i,
                                      int k_tiles_i) {
#pragma HLS INLINE off
    int emitted_k = 0;
    int spatial_idx = 0;
    int cin_idx = 0;
    for (int kt = 0; kt < MAX_K_TILE_COUNT; ++kt) {
        if (kt >= k_tiles_i) break;
        act_vec_t word = 0;
        for (int lane = 0; lane < TK; ++lane) {
#pragma HLS PIPELINE II=1
            if (emitted_k < k_total_i) {
                set_vec_i8_dynamic(word, lane, packed_vec_i8(spatial_word[spatial_idx], cin_idx));
                ++emitted_k;
                ++cin_idx;
                if (cin_idx >= in_c_i) {
                    cin_idx = 0;
                    ++spatial_idx;
                }
            }
        }
        act_stream.write(word);
    }
}

#ifndef __SYNTHESIS__
static void emit_first_layer_3x3_windows(const tensor_desc_t& src_desc,
                                         hls::stream<act_vec_t>& act_stream,
                                         const conv_cfg_t& cfg,
                                         u16_t out_h,
                                         u16_t out_w,
                                         volatile std::uint32_t& dbg_act_words) {
#pragma HLS INLINE off
    const int out_h_i = static_cast<int>(out_h.to_uint());
    const int out_w_i = static_cast<int>(out_w.to_uint());
    std::uint32_t emitted = 0;

    for (int oh_i = 0; oh_i < MAX_FM_H; ++oh_i) {
        if (oh_i >= out_h_i) {
            break;
        }
        for (int ow_i = 0; ow_i < MAX_FM_W; ++ow_i) {
            if (ow_i >= out_w_i) {
                break;
            }

            act_vec_t word = 0;
            const i32_t base_h = static_cast<i32_t>(oh_i * 2 - 1);
            const i32_t base_w = static_cast<i32_t>(ow_i * 2 - 1);

            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
#pragma HLS PIPELINE off
                    i8_t tile[TM];
#pragma HLS ARRAY_PARTITION variable=tile complete dim=1
                    const i32_t ih = base_h + kh;
                    const i32_t iw = base_w + kw;
                    const bool ok =
                        on_chip_memory_read_tile(src_desc, ih, iw, 0, static_cast<u8_t>(3), tile);

                    for (int c = 0; c < 3; ++c) {
#pragma HLS UNROLL
                        const int lane = (kh * 3 + kw) * 3 + c;
                        set_vec_i8(word, lane, ok ? tile[c] : i8_t(0));
                    }
                }
            }

            act_stream.write(word);
            ++emitted;
            if ((emitted & 0x3ffU) == 0U) {
                dbg_act_words = emitted;
            }
        }
    }
    dbg_act_words = emitted;
}
#endif

static void emit_first_layer_3x3_window_row(const tensor_desc_t& src_desc,
                                            hls::stream<act_vec_t>& act_stream,
                                            u16_t out_row,
                                            u16_t out_w) {
#pragma HLS INLINE off
    const int out_w_i = static_cast<int>(out_w.to_uint());
    const i32_t base_h = static_cast<i32_t>(out_row * 2 - 1);
    if (out_w_i <= 0) {
        return;
    }

    emit_first_layer_c3_border_word(src_desc, act_stream, base_h, static_cast<i32_t>(-1));

    int inner_end = out_w_i;
    if (out_w_i > 1) {
        const i32_t last_base_w = static_cast<i32_t>((out_w_i - 1) * 2 - 1);
        if (last_base_w + 2 >= static_cast<i32_t>(src_desc.w)) {
            inner_end = out_w_i - 1;
        }
    }

    int ow_i = 1;
    for (; ow_i + 1 < MAX_FM_W; ow_i += 2) {
        if (ow_i + 1 >= inner_end) {
            break;
        }
        const i32_t base_w = static_cast<i32_t>(ow_i * 2 - 1);
        emit_first_layer_c3_pair_words(src_desc, act_stream, base_h, base_w);
    }

    for (; ow_i < MAX_FM_W; ++ow_i) {
        if (ow_i >= inner_end) {
            break;
        }
        const i32_t base_w = static_cast<i32_t>(ow_i * 2 - 1);
        emit_first_layer_c3_row_segment_word(src_desc, act_stream, base_h, base_w);
    }

    for (int ow_i = 0; ow_i < 1; ++ow_i) {
        const int tail_ow = inner_end + ow_i;
        if (tail_ow >= out_w_i) {
            break;
        }
        const i32_t base_w = static_cast<i32_t>(tail_ow * 2 - 1);
        emit_first_layer_c3_border_word(src_desc, act_stream, base_h, base_w);
    }
}

#ifndef __SYNTHESIS__
static bool can_merge_lane(const conv_cfg_t& cfg,
                           u16_t oh,
                           u16_t ow,
                           u16_t k_idx,
                           i32_t base_ih,
                           i32_t base_iw,
                           u16_t expected_cin) {
#pragma HLS INLINE
    i32_t ih = 0;
    i32_t iw = 0;
    u16_t cin = 0;
    if (!map_k_index(cfg, oh, ow, k_idx, ih, iw, cin)) {
        return false;
    }
    return ih == base_ih && iw == base_iw && cin == expected_cin;
}

static void load_window_line(const tensor_desc_t& src_desc,
                             const conv_cfg_t& cfg,
                             u16_t oh,
                             u16_t kt,
                             u16_t out_w,
                             i8_t line_buf[1024][TK]) {
#pragma HLS INLINE off
    const int out_w_i = static_cast<int>(out_w.to_uint());

    if (effective_kernel(cfg) == 1) {
        const u16_t stride = effective_stride(cfg);
        const u16_t c_begin = static_cast<u16_t>(kt * TK);
        const u16_t remaining = (c_begin < cfg.in_c) ? static_cast<u16_t>(cfg.in_c - c_begin) : static_cast<u16_t>(0);
        const u8_t lanes = (remaining > static_cast<u16_t>(TK)) ? static_cast<u8_t>(TK) : static_cast<u8_t>(remaining);
        const i32_t ih = static_cast<i32_t>(oh * stride);

        for (int ow_i = 0; ow_i < MAX_FM_W; ++ow_i) {
#pragma HLS PIPELINE II=1
            if (ow_i >= out_w_i) {
                break;
            }
            const i32_t iw = static_cast<i32_t>(static_cast<u16_t>(ow_i) * stride);
            i8_t tile[TM];
#pragma HLS ARRAY_PARTITION variable=tile complete dim=1
            bool ok = false;
            if (lanes != 0) {
                ok = on_chip_memory_read_tile(src_desc, ih, iw, c_begin, lanes, tile);
            }
            for (int lane = 0; lane < TK; ++lane) {
#pragma HLS UNROLL
                line_buf[ow_i][lane] = (ok && static_cast<unsigned>(lane) < lanes.to_uint()) ? tile[lane] : i8_t(0);
            }
        }
        return;
    }

    for (int ow_i = 0; ow_i < MAX_FM_W; ++ow_i) {
        if (ow_i >= out_w_i) {
            break;
        }
        const u16_t ow = static_cast<u16_t>(ow_i);
        int lane = 0;
        for (int lane_iter = 0; lane_iter < TK; ++lane_iter) {
#pragma HLS PIPELINE off
            if (lane >= TK) {
                break;
            }
            const u16_t k_idx = static_cast<u16_t>(kt * TK + lane);
            i32_t ih = 0;
            i32_t iw = 0;
            u16_t cin = 0;

            if (!map_k_index(cfg, oh, ow, k_idx, ih, iw, cin)) {
                line_buf[ow_i][lane] = 0;
                ++lane;
                continue;
            }

            int run = 1;
            for (int run_i = 1; run_i < TM; ++run_i) {
                if (lane + run_i >= TK ||
                    !can_merge_lane(cfg,
                                    oh,
                                    ow,
                                    static_cast<u16_t>(k_idx + run_i),
                                    ih,
                                    iw,
                                    static_cast<u16_t>(cin + run_i))) {
                    break;
                }
                run = run_i + 1;
            }

            i8_t tile[TM];
#pragma HLS ARRAY_PARTITION variable=tile complete dim=1
            if (!on_chip_memory_read_tile(src_desc, ih, iw, cin, static_cast<u8_t>(run), tile)) {
                for (int i = 0; i < TM; ++i) {
#pragma HLS UNROLL
                    if (i < run) {
                        line_buf[ow_i][lane + i] = 0;
                    }
                }
            } else {
                for (int i = 0; i < TM; ++i) {
#pragma HLS UNROLL
                    if (i < run) {
                        line_buf[ow_i][lane + i] = tile[i];
                    }
                }
            }
            lane += run;
        }
    }
}

static void load_window_vector(const tensor_desc_t& src_desc,
                               const conv_cfg_t& cfg,
                               u16_t oh,
                               u16_t ow,
                               u16_t kt,
                               i8_t vec_buf[TK]) {
#pragma HLS INLINE off
    if (effective_kernel(cfg) == 1) {
        const u16_t stride = effective_stride(cfg);
        const u16_t c_begin = static_cast<u16_t>(kt * TK);
        const u16_t remaining = (c_begin < cfg.in_c) ? static_cast<u16_t>(cfg.in_c - c_begin) : static_cast<u16_t>(0);
        const u8_t lanes = (remaining > static_cast<u16_t>(TK)) ? static_cast<u8_t>(TK) : static_cast<u8_t>(remaining);
        const i32_t ih = static_cast<i32_t>(oh * stride);
        const i32_t iw = static_cast<i32_t>(ow * stride);

        i8_t tile[TM];
#pragma HLS ARRAY_PARTITION variable=tile complete dim=1
        bool ok = false;
        if (lanes != 0) {
            ok = on_chip_memory_read_tile(src_desc, ih, iw, c_begin, lanes, tile);
        }
        for (int lane = 0; lane < TK; ++lane) {
#pragma HLS UNROLL
            vec_buf[lane] = (ok && static_cast<unsigned>(lane) < lanes.to_uint()) ? tile[lane] : i8_t(0);
        }
        return;
    }

    int lane = 0;
    for (int lane_iter = 0; lane_iter < TK; ++lane_iter) {
#pragma HLS PIPELINE off
        if (lane >= TK) {
            break;
        }
        const u16_t k_idx = static_cast<u16_t>(kt * TK + lane);
        i32_t ih = 0;
        i32_t iw = 0;
        u16_t cin = 0;

        if (!map_k_index(cfg, oh, ow, k_idx, ih, iw, cin)) {
            vec_buf[lane] = 0;
            ++lane;
            continue;
        }

        int run = 1;
        for (int run_i = 1; run_i < TM; ++run_i) {
            if (lane + run_i >= TK ||
                !can_merge_lane(cfg,
                                oh,
                                ow,
                                static_cast<u16_t>(k_idx + run_i),
                                ih,
                                iw,
                                static_cast<u16_t>(cin + run_i))) {
                break;
            }
            run = run_i + 1;
        }

        i8_t tile[TM];
#pragma HLS ARRAY_PARTITION variable=tile complete dim=1
        if (!on_chip_memory_read_tile(src_desc, ih, iw, cin, static_cast<u8_t>(run), tile)) {
            for (int i = 0; i < TM; ++i) {
#pragma HLS UNROLL
                if (i < run) {
                    vec_buf[lane + i] = 0;
                }
            }
        } else {
            for (int i = 0; i < TM; ++i) {
#pragma HLS UNROLL
                if (i < run) {
                    vec_buf[lane + i] = tile[i];
                }
            }
        }
        lane += run;
    }
}

void window_generator(const tensor_desc_t& src_desc,
                      hls::stream<act_vec_t>& act_stream,
                      const conv_cfg_t& cfg,
                      volatile std::uint32_t& dbg_act_words) {
#pragma HLS INLINE off
    i8_t line_buf[1024][TK];
#pragma HLS BIND_STORAGE variable=line_buf type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=line_buf complete dim=2
    i8_t vec_buf[TK];
#pragma HLS ARRAY_PARTITION variable=vec_buf complete dim=1

    const u16_t stride = effective_stride(cfg);
    const u16_t out_h = conv_out_dim(cfg.in_h, stride);
    const u16_t out_w = conv_out_dim(cfg.in_w, stride);
    const u16_t k_total = kernel_flat(cfg);
    const u16_t k_tiles = ceil_div_u16(k_total, TK);
    const int out_h_i = static_cast<int>(out_h.to_uint());
    const int out_w_i = static_cast<int>(out_w.to_uint());
    const int k_tiles_i = static_cast<int>(k_tiles.to_uint());
    std::uint32_t emitted = 0;
    dbg_act_words = 0;

    if (k_tiles == 1 && is_first_layer_3x3(cfg)) {
        emit_first_layer_3x3_windows(src_desc, act_stream, cfg, out_h, out_w, dbg_act_words);
        return;
    }

    for (int oh_i = 0; oh_i < MAX_FM_H; ++oh_i) {
        if (oh_i >= out_h_i) {
            break;
        }
        const u16_t oh = static_cast<u16_t>(oh_i);
        if (k_tiles == 1) {
            load_window_line(src_desc, cfg, oh, 0, out_w, line_buf);
            for (int ow_i = 0; ow_i < MAX_FM_W; ++ow_i) {
#pragma HLS PIPELINE II=1
                if (ow_i >= out_w_i) {
                    break;
                }
                act_vec_t word = 0;
                for (int lane = 0; lane < TK; ++lane) {
#pragma HLS UNROLL
                    set_vec_i8(word, lane, line_buf[ow_i][lane]);
                }
                act_stream.write(word);
                ++emitted;
                if ((emitted & 0x3ffU) == 0U) {
                    dbg_act_words = emitted;
                }
            }
        } else {
            for (int ow_i = 0; ow_i < MAX_FM_W; ++ow_i) {
                if (ow_i >= out_w_i) {
                    break;
                }
                const u16_t ow = static_cast<u16_t>(ow_i);
                for (int kt_i = 0; kt_i < MAX_K_TILE_COUNT; ++kt_i) {
                    if (kt_i >= k_tiles_i) {
                        break;
                    }
                    const u16_t kt = static_cast<u16_t>(kt_i);
                    load_window_vector(src_desc, cfg, oh, ow, kt, vec_buf);
                    act_vec_t word = 0;
                    for (int lane = 0; lane < TK; ++lane) {
#pragma HLS UNROLL
                        set_vec_i8(word, lane, vec_buf[lane]);
                    }
                    act_stream.write(word);
                    ++emitted;
                    if ((emitted & 0x3ffU) == 0U) {
                        dbg_act_words = emitted;
                    }
                }
            }
        }
    }
    dbg_act_words = emitted;
}

static void emit_window_row_1ktile(const tensor_desc_t& src_desc,
                                   hls::stream<act_vec_t>& act_stream,
                                   const conv_cfg_t& cfg,
                                   u16_t out_row,
                                   u16_t out_w) {
#pragma HLS INLINE off
    i8_t line_buf[1024][TK];
#pragma HLS BIND_STORAGE variable=line_buf type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=line_buf complete dim=2
    const int out_w_i = static_cast<int>(out_w.to_uint());

    load_window_line(src_desc, cfg, out_row, 0, out_w, line_buf);
    for (int ow_i = 0; ow_i < MAX_FM_W; ++ow_i) {
#pragma HLS PIPELINE II=1
        if (ow_i >= out_w_i) {
            break;
        }
        act_vec_t word = 0;
        for (int lane = 0; lane < TK; ++lane) {
#pragma HLS UNROLL
            set_vec_i8(word, lane, line_buf[ow_i][lane]);
        }
        act_stream.write(word);
    }
}

static void emit_window_row_multikt(const tensor_desc_t& src_desc,
                                    hls::stream<act_vec_t>& act_stream,
                                    const conv_cfg_t& cfg,
                                    u16_t out_row,
                                    u16_t out_w) {
#pragma HLS INLINE off
    i8_t vec_buf[TK];
#pragma HLS ARRAY_PARTITION variable=vec_buf complete dim=1
    const u16_t k_total = kernel_flat(cfg);
    const u16_t k_tiles = ceil_div_u16(k_total, TK);
    const int out_w_i = static_cast<int>(out_w.to_uint());
    const int k_tiles_i = static_cast<int>(k_tiles.to_uint());

    for (int ow_i = 0; ow_i < MAX_FM_W; ++ow_i) {
        if (ow_i >= out_w_i) {
            break;
        }
        const u16_t ow = static_cast<u16_t>(ow_i);
        for (int kt_i = 0; kt_i < MAX_K_TILE_COUNT; ++kt_i) {
            if (kt_i >= k_tiles_i) {
                break;
            }
            const u16_t kt = static_cast<u16_t>(kt_i);
            load_window_vector(src_desc, cfg, out_row, ow, kt, vec_buf);
            act_vec_t word = 0;
            for (int lane = 0; lane < TK; ++lane) {
#pragma HLS UNROLL
                set_vec_i8(word, lane, vec_buf[lane]);
            }
            act_stream.write(word);
        }
    }
}
#endif

static void emit_window_row_3x3_smallc_fast(const tensor_desc_t& src_desc,
                                            hls::stream<act_vec_t>& act_stream,
                                            const conv_cfg_t& cfg,
                                            u16_t out_row,
                                            u16_t out_w) {
#pragma HLS INLINE off
    const u16_t stride = effective_stride(cfg);
    const u16_t dilation = effective_dilation(cfg);
    const int out_w_i = static_cast<int>(out_w.to_uint());
    const int in_c_i = static_cast<int>(cfg.in_c.to_uint());
    const i32_t pad = static_cast<i32_t>(dilation.to_uint());
    const i32_t base_h = static_cast<i32_t>(out_row.to_uint() * stride.to_uint()) - pad;
    const int k_total_i = in_c_i * 9;
    const int k_tiles_i = (k_total_i + TK - 1) / TK;

    for (int ow_i = 0; ow_i < MAX_FM_W; ++ow_i) {
        if (ow_i >= out_w_i) break;
        const i32_t base_w = static_cast<i32_t>(ow_i * static_cast<int>(stride.to_uint())) - pad;

        act_vec_t spatial_word[9];
#pragma HLS ARRAY_PARTITION variable=spatial_word complete dim=1

        for (int sp = 0; sp < 9; ++sp) {
            const int kh = sp / 3;
            const int kw = sp - kh * 3;
            const i32_t ih = base_h + static_cast<i32_t>(kh * static_cast<int>(dilation.to_uint()));
            const i32_t iw = base_w + static_cast<i32_t>(kw * static_cast<int>(dilation.to_uint()));
            on_chip_memory_read_packed_tile(src_desc,
                                            ih,
                                            iw,
                                            0,
                                            static_cast<u8_t>(cfg.in_c.to_uint()),
                                            spatial_word[sp]);
        }

        if (in_c_i == 12) {
            emit_smallc_c12_words(spatial_word, act_stream);
        } else if (in_c_i == 19) {
            emit_smallc_c19_words(spatial_word, act_stream);
        } else if (in_c_i == 25) {
            emit_smallc_c25_words(spatial_word, act_stream);
        } else {
            emit_smallc_generic_words(spatial_word, act_stream, in_c_i, k_total_i, k_tiles_i);
        }
    }
}

static void load_smallc_spatial_col(const tensor_desc_t& src_desc,
                                    const conv_cfg_t& cfg,
                                    i32_t base_h,
                                    i32_t iw,
                                    int col,
                                    act_vec_t spatial_word[9]) {
#pragma HLS INLINE
    for (int kh = 0; kh < 3; ++kh) {
#pragma HLS UNROLL
        const i32_t ih = base_h + static_cast<i32_t>(kh);
        on_chip_memory_read_packed_tile(src_desc,
                                        ih,
                                        iw,
                                        0,
                                        static_cast<u8_t>(cfg.in_c.to_uint()),
                                        spatial_word[kh * 3 + col]);
    }
}

static void load_smallc_c19_row_segment(const tensor_desc_t& src_desc,
                                        i32_t ih,
                                        i32_t base_w,
                                        int row,
                                        act_vec_t spatial_word[9]) {
#pragma HLS INLINE
    act_vec_t seg0 = 0;
    act_vec_t seg1 = 0;
    on_chip_memory_read_packed_contiguous(
        src_desc, ih, base_w, 0, static_cast<u8_t>(32), seg0);
    on_chip_memory_read_packed_contiguous(
        src_desc, ih, base_w + 1, static_cast<u16_t>(13), static_cast<u8_t>(25), seg1);

    act_vec_t px0 = 0;
    act_vec_t px1 = 0;
    act_vec_t px2 = 0;
    copy_const_segment<0, 0, 19>(px0, seg0);
    copy_const_segment<0, 19, 13>(px1, seg0);
    copy_const_segment<13, 0, 6>(px1, seg1);
    copy_const_segment<0, 6, 19>(px2, seg1);
    spatial_word[row * 3 + 0] = px0;
    spatial_word[row * 3 + 1] = px1;
    spatial_word[row * 3 + 2] = px2;
}

static void split_smallc_c19_pair_row(const act_vec_t& seg0,
                                      const act_vec_t& seg1,
                                      const act_vec_t& seg2,
                                      int row,
                                      act_vec_t spatial0[9],
                                      act_vec_t spatial1[9]) {
#pragma HLS INLINE
    act_vec_t px0 = 0;
    act_vec_t px1 = 0;
    act_vec_t px2 = 0;
    act_vec_t px3 = 0;
    act_vec_t px4 = 0;
    copy_const_segment<0, 0, 19>(px0, seg0);
    copy_const_segment<0, 19, 13>(px1, seg0);
    copy_const_segment<13, 0, 6>(px1, seg1);
    copy_const_segment<0, 6, 19>(px2, seg1);
    copy_const_segment<0, 25, 7>(px3, seg1);
    copy_const_segment<7, 0, 12>(px3, seg2);
    copy_const_segment<0, 12, 19>(px4, seg2);

    spatial0[row * 3 + 0] = px0;
    spatial0[row * 3 + 1] = px1;
    spatial0[row * 3 + 2] = px2;
    spatial1[row * 3 + 0] = px2;
    spatial1[row * 3 + 1] = px3;
    spatial1[row * 3 + 2] = px4;
}

static void load_smallc_c19_pair_row_segment(const tensor_desc_t& src_desc,
                                             i32_t ih,
                                             i32_t base_w,
                                             int row,
                                             act_vec_t spatial0[9],
                                             act_vec_t spatial1[9]) {
#pragma HLS INLINE
    act_vec_t seg0 = 0;
    act_vec_t seg1 = 0;
    act_vec_t seg2 = 0;
    on_chip_memory_read_packed_contiguous(
        src_desc, ih, base_w, 0, static_cast<u8_t>(32), seg0);
    on_chip_memory_read_packed_contiguous(
        src_desc, ih, base_w + 1, static_cast<u16_t>(13), static_cast<u8_t>(32), seg1);
    on_chip_memory_read_packed_contiguous(
        src_desc, ih, base_w + 3, static_cast<u16_t>(7), static_cast<u8_t>(31), seg2);
    split_smallc_c19_pair_row(seg0, seg1, seg2, row, spatial0, spatial1);
}

static void emit_window_row_3x3_c19_stride2_single(const tensor_desc_t& src_desc,
                                                   hls::stream<act_vec_t>& act_stream,
                                                   const conv_cfg_t& cfg,
                                                   i32_t base_h,
                                                   i32_t base_w) {
#pragma HLS INLINE
    act_vec_t spatial_word[9];
#pragma HLS ARRAY_PARTITION variable=spatial_word complete dim=1

    const bool inner =
        (base_h >= 0) &&
        (base_h + 2 < static_cast<i32_t>(src_desc.h)) &&
        (base_w >= 0) &&
        (base_w + 2 < static_cast<i32_t>(src_desc.w));
    if (inner) {
        load_smallc_c19_row_segment(src_desc, base_h + 0, base_w, 0, spatial_word);
        load_smallc_c19_row_segment(src_desc, base_h + 1, base_w, 1, spatial_word);
        load_smallc_c19_row_segment(src_desc, base_h + 2, base_w, 2, spatial_word);
    } else {
        load_smallc_spatial_col(src_desc, cfg, base_h, base_w + 0, 0, spatial_word);
        load_smallc_spatial_col(src_desc, cfg, base_h, base_w + 1, 1, spatial_word);
        load_smallc_spatial_col(src_desc, cfg, base_h, base_w + 2, 2, spatial_word);
    }

    emit_smallc_c19_words(spatial_word, act_stream);
}

static void emit_window_row_3x3_c19_stride2_fast(const tensor_desc_t& src_desc,
                                                 hls::stream<act_vec_t>& act_stream,
                                                 const conv_cfg_t& cfg,
                                                 u16_t out_row,
                                                 u16_t out_w) {
#pragma HLS INLINE off
    const int out_w_i = static_cast<int>(out_w.to_uint());
    const i32_t base_h = static_cast<i32_t>(out_row.to_uint() * 2U) - 1;

    if (out_w_i <= 0) {
        return;
    }

    emit_window_row_3x3_c19_stride2_single(src_desc, act_stream, cfg, base_h, static_cast<i32_t>(-1));

    int ow_i = 1;
    for (; ow_i + 1 < MAX_FM_W; ow_i += 2) {
        if (ow_i + 1 >= out_w_i) break;
        const i32_t base_w = static_cast<i32_t>(ow_i * 2) - 1;
        const bool inner_pair =
            (base_h >= 0) &&
            (base_h + 2 < static_cast<i32_t>(src_desc.h)) &&
            (base_w >= 0) &&
            (base_w + 4 < static_cast<i32_t>(src_desc.w));
        if (inner_pair) {
            act_vec_t spatial0[9];
            act_vec_t spatial1[9];
#pragma HLS ARRAY_PARTITION variable=spatial0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=spatial1 complete dim=1
            load_smallc_c19_pair_row_segment(src_desc, base_h + 0, base_w, 0, spatial0, spatial1);
            load_smallc_c19_pair_row_segment(src_desc, base_h + 1, base_w, 1, spatial0, spatial1);
            load_smallc_c19_pair_row_segment(src_desc, base_h + 2, base_w, 2, spatial0, spatial1);
            emit_smallc_c19_words(spatial0, act_stream);
            emit_smallc_c19_words(spatial1, act_stream);
        } else {
            emit_window_row_3x3_c19_stride2_single(src_desc, act_stream, cfg, base_h, base_w);
            emit_window_row_3x3_c19_stride2_single(src_desc,
                                                   act_stream,
                                                   cfg,
                                                   base_h,
                                                   static_cast<i32_t>(base_w + 2));
        }
    }

    for (; ow_i < MAX_FM_W; ++ow_i) {
        if (ow_i >= out_w_i) break;
        const i32_t base_w = static_cast<i32_t>(ow_i * 2) - 1;
        emit_window_row_3x3_c19_stride2_single(src_desc, act_stream, cfg, base_h, base_w);
    }
}

static void emit_window_row_3x3_smallc_stride1_reuse(const tensor_desc_t& src_desc,
                                                     hls::stream<act_vec_t>& act_stream,
                                                     const conv_cfg_t& cfg,
                                                     u16_t out_row,
                                                     u16_t out_w) {
#pragma HLS INLINE off
    const int out_w_i = static_cast<int>(out_w.to_uint());
    const int in_c_i = static_cast<int>(cfg.in_c.to_uint());
    const i32_t base_h = static_cast<i32_t>(out_row.to_uint()) - 1;
    const int k_total_i = in_c_i * 9;
    const int k_tiles_i = (k_total_i + TK - 1) / TK;
    act_vec_t spatial_word[9];
#pragma HLS ARRAY_PARTITION variable=spatial_word complete dim=1

    for (int ow_i = 0; ow_i < MAX_FM_W; ++ow_i) {
        if (ow_i >= out_w_i) break;
        const i32_t base_w = static_cast<i32_t>(ow_i) - 1;
        if (ow_i == 0) {
            load_smallc_spatial_col(src_desc, cfg, base_h, base_w + 0, 0, spatial_word);
            load_smallc_spatial_col(src_desc, cfg, base_h, base_w + 1, 1, spatial_word);
            load_smallc_spatial_col(src_desc, cfg, base_h, base_w + 2, 2, spatial_word);
        } else {
            for (int kh = 0; kh < 3; ++kh) {
#pragma HLS UNROLL
                spatial_word[kh * 3 + 0] = spatial_word[kh * 3 + 1];
                spatial_word[kh * 3 + 1] = spatial_word[kh * 3 + 2];
            }
            load_smallc_spatial_col(src_desc, cfg, base_h, base_w + 2, 2, spatial_word);
        }

        if (in_c_i == 12) {
            emit_smallc_c12_words(spatial_word, act_stream);
        } else if (in_c_i == 19) {
            emit_smallc_c19_words(spatial_word, act_stream);
        } else if (in_c_i == 25) {
            emit_smallc_c25_words(spatial_word, act_stream);
        } else {
            emit_smallc_generic_words(spatial_word, act_stream, in_c_i, k_total_i, k_tiles_i);
        }
    }
}

static void read_3x3_spatial_segment_packed(const tensor_desc_t& src_desc,
                                            const conv_cfg_t& cfg,
                                            i32_t base_h,
                                            i32_t base_w,
                                            int spatial_idx,
                                            int cin,
                                            int count,
                                            act_vec_t& packed) {
#pragma HLS INLINE
    const u16_t dilation = effective_dilation(cfg);
    const int kh = spatial_idx / 3;
    const int kw = spatial_idx - kh * 3;
    const i32_t ih = base_h + static_cast<i32_t>(kh * static_cast<int>(dilation.to_uint()));
    const i32_t iw = base_w + static_cast<i32_t>(kw * static_cast<int>(dilation.to_uint()));
    on_chip_memory_read_packed_tile(src_desc,
                                    ih,
                                    iw,
                                    static_cast<u16_t>(cin),
                                    static_cast<u8_t>(count),
                                    packed);
}

static void emit_window_row_3x3_fast(const tensor_desc_t& src_desc,
                                      hls::stream<act_vec_t>& act_stream,
                                      const conv_cfg_t& cfg,
                                      u16_t out_row,
                                      u16_t out_w) {
#pragma HLS INLINE off
    const u16_t stride = effective_stride(cfg);
    const u16_t dilation = effective_dilation(cfg);
    const u16_t k_total = static_cast<u16_t>(cfg.in_c * static_cast<u16_t>(9));
    const u16_t k_tiles = ceil_div_u16(k_total, TK);
    const int out_w_i = static_cast<int>(out_w.to_uint());
    const int k_tiles_i = static_cast<int>(k_tiles.to_uint());
    const int in_c_i = static_cast<int>(cfg.in_c.to_uint());
    const int k_total_i = static_cast<int>(k_total.to_uint());
    const i32_t pad = static_cast<i32_t>(dilation.to_uint());
    const i32_t base_h = static_cast<i32_t>(out_row.to_uint() * stride.to_uint()) - pad;

    for (int ow_i = 0; ow_i < MAX_FM_W; ++ow_i) {
        if (ow_i >= out_w_i) break;
        const i32_t base_w = static_cast<i32_t>(ow_i * static_cast<int>(stride.to_uint())) - pad;

        int emitted_k = 0;
        int spatial_idx = 0;
        int cin_idx = 0;
        for (int kt = 0; kt < MAX_K_TILE_COUNT; ++kt) {
            if (kt >= k_tiles_i) break;
            act_vec_t word = 0;
            int lane_offset = 0;
            int lanes_left = TK;

            for (int segment = 0; segment < 2; ++segment) {
                if (lanes_left <= 0 || emitted_k >= k_total_i) break;
                int count = in_c_i - cin_idx;
                if (count > lanes_left) {
                    count = lanes_left;
                }
                if (count > (k_total_i - emitted_k)) {
                    count = k_total_i - emitted_k;
                }

                act_vec_t segment_word = 0;
                read_3x3_spatial_segment_packed(src_desc,
                                                cfg,
                                                base_h,
                                                base_w,
                                                spatial_idx,
                                                cin_idx,
                                                count,
                                                segment_word);
                insert_packed_segment(word, lane_offset, count, segment_word);

                lane_offset += count;
                lanes_left -= count;
                emitted_k += count;
                cin_idx += count;
                if (cin_idx >= in_c_i) {
                    cin_idx = 0;
                    ++spatial_idx;
                }
            }
            act_stream.write(word);
        }
    }
}

static void emit_window_row_3x3_c131_stride2_fast(const tensor_desc_t& src_desc,
                                                  hls::stream<act_vec_t>& act_stream,
                                                  const conv_cfg_t& cfg,
                                                  u16_t out_row,
                                                  u16_t out_w) {
#pragma HLS INLINE off
    const int out_w_i = static_cast<int>(out_w.to_uint());
    const i32_t base_h = static_cast<i32_t>(out_row.to_uint() * 2U) - 1;
    const int k_total_i = 131 * 9;
    const int k_tiles_i = (k_total_i + TK - 1) / TK;

    for (int ow_i = 0; ow_i < MAX_FM_W; ++ow_i) {
        if (ow_i >= out_w_i) break;
        const i32_t base_w = static_cast<i32_t>(ow_i * 2) - 1;
        for (int kt = 0; kt < MAX_K_TILE_COUNT; ++kt) {
#pragma HLS PIPELINE II=1
            if (kt >= k_tiles_i) break;
            const int start_k = kt * TK;
            const int spatial_idx = start_k / 131;
            const int cin_idx = start_k - spatial_idx * 131;
            const int remaining = k_total_i - start_k;
            int count0 = 131 - cin_idx;
            if (count0 > TK) {
                count0 = TK;
            }
            if (count0 > remaining) {
                count0 = remaining;
            }

            act_vec_t word = 0;
            act_vec_t seg0 = 0;
            read_3x3_spatial_segment_packed(src_desc,
                                            cfg,
                                            base_h,
                                            base_w,
                                            spatial_idx,
                                            cin_idx,
                                            count0,
                                            seg0);
            insert_packed_segment(word, 0, count0, seg0);

            int count1 = remaining - count0;
            if (count1 > (TK - count0)) {
                count1 = TK - count0;
            }
            if (count1 > 0) {
                act_vec_t seg1 = 0;
                read_3x3_spatial_segment_packed(src_desc,
                                                cfg,
                                                base_h,
                                                base_w,
                                                spatial_idx + 1,
                                                0,
                                                count1,
                                                seg1);
                insert_packed_segment(word, count0, count1, seg1);
            }
            act_stream.write(word);
        }
    }
}

static void emit_window_row_1x1_fast(const tensor_desc_t& src_desc,
                                      hls::stream<act_vec_t>& act_stream,
                                      const conv_cfg_t& cfg,
                                      u16_t out_row,
                                      u16_t out_w) {
#pragma HLS INLINE off
    const u16_t stride = (cfg.stride == 0) ? static_cast<u16_t>(1) : static_cast<u16_t>(cfg.stride);
    const u16_t k_total = cfg.in_c;
    const u16_t k_tiles = ceil_div_u16(k_total, TK);
    const int out_w_i = static_cast<int>(out_w.to_uint());
    const int k_tiles_i = static_cast<int>(k_tiles.to_uint());
    const i32_t ih = static_cast<i32_t>(out_row * stride);

    for (int ow_i = 0; ow_i < MAX_FM_W; ++ow_i) {
        if (ow_i >= out_w_i) break;
        const i32_t iw = static_cast<i32_t>(ow_i * stride);
        for (int kt = 0; kt < MAX_K_TILE_COUNT; ++kt) {
#pragma HLS PIPELINE II=1
            if (kt >= k_tiles_i) break;
            const u16_t c_begin = static_cast<u16_t>(kt * TK);
            const u16_t remaining = static_cast<u16_t>(cfg.in_c.to_uint() - c_begin.to_uint());
            const u8_t lanes = (remaining > static_cast<u16_t>(TK))
                                   ? static_cast<u8_t>(TK)
                                   : static_cast<u8_t>(remaining);
            act_vec_t word = 0;
            on_chip_memory_read_packed_tile(src_desc, ih, iw, c_begin, lanes, word);
            act_stream.write(word);
        }
    }
}

static void emit_window_row_1x1_aligned_full_fast(const tensor_desc_t& src_desc,
                                                  hls::stream<act_vec_t>& act_stream,
                                                  const conv_cfg_t& cfg,
                                                  u16_t out_row,
                                                  u16_t out_w) {
#pragma HLS INLINE off
    const u16_t stride = (cfg.stride == 0) ? static_cast<u16_t>(1) : static_cast<u16_t>(cfg.stride);
    const u16_t k_tiles = ceil_div_u16(cfg.in_c, static_cast<u16_t>(TK));
    const int out_w_i = static_cast<int>(out_w.to_uint());
    const int k_tiles_i = static_cast<int>(k_tiles.to_uint());
    const i32_t ih = static_cast<i32_t>(out_row * stride);

    for (int ow_i = 0; ow_i < MAX_FM_W; ++ow_i) {
        if (ow_i >= out_w_i) break;
        const i32_t iw = static_cast<i32_t>(ow_i * stride);
        for (int kt = 0; kt < MAX_K_TILE_COUNT; ++kt) {
#pragma HLS PIPELINE II=1
            if (kt >= k_tiles_i) break;
            act_vec_t word = 0;
            on_chip_memory_read_aligned_full_tile(src_desc,
                                                  ih,
                                                  iw,
                                                  static_cast<u16_t>(kt * TK),
                                                  word);
            act_stream.write(word);
        }
    }
}

void window_generator_row(const tensor_desc_t& src_desc,
                          hls::stream<act_vec_t>& act_stream,
                          const conv_cfg_t& cfg,
                          u16_t out_row) {
#pragma HLS INLINE off
    const u16_t stride = (cfg.stride == 0) ? static_cast<u16_t>(1) : static_cast<u16_t>(cfg.stride);
    const u16_t out_w = conv_out_dim(cfg.in_w, stride);
    const u16_t k_total = kernel_flat(cfg);
    const u16_t k_tiles = ceil_div_u16(k_total, TK);
    const u16_t kernel = (cfg.kernel == 1) ? static_cast<u16_t>(1) : static_cast<u16_t>(3);

    if (k_tiles == 1 && is_first_layer_3x3(cfg)) {
        emit_first_layer_3x3_window_row(src_desc, act_stream, out_row, out_w);
        return;
    }

    if (kernel == 3) {
        if (cfg.in_c.to_uint() <= 32U) {
            if (cfg.in_c.to_uint() == 19U &&
                stride.to_uint() == 2U &&
                cfg.dilation.to_uint() == 1U) {
                emit_window_row_3x3_c19_stride2_fast(src_desc, act_stream, cfg, out_row, out_w);
                return;
            }
            if (stride.to_uint() == 1U && cfg.dilation.to_uint() == 1U) {
                emit_window_row_3x3_smallc_stride1_reuse(src_desc, act_stream, cfg, out_row, out_w);
                return;
            }
            emit_window_row_3x3_smallc_fast(src_desc, act_stream, cfg, out_row, out_w);
            return;
        }
        if (cfg.in_c.to_uint() == 131U &&
            stride.to_uint() == 2U &&
            cfg.dilation.to_uint() == 1U) {
            emit_window_row_3x3_c131_stride2_fast(src_desc, act_stream, cfg, out_row, out_w);
            return;
        }
        emit_window_row_3x3_fast(src_desc, act_stream, cfg, out_row, out_w);
        return;
    }

    if (kernel == 1 &&
        (cfg.in_c.to_uint() & static_cast<unsigned>(TK - 1)) == 0U &&
        is_aligned_full_channel_source(src_desc)) {
        emit_window_row_1x1_aligned_full_fast(src_desc, act_stream, cfg, out_row, out_w);
        return;
    }

    emit_window_row_1x1_fast(src_desc, act_stream, cfg, out_row, out_w);
}

}  // namespace esp_int8
