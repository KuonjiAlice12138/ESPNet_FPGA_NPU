#include "../include/npu_config.hpp"
#include "../include/npu_types.hpp"

namespace esp_int8 {

constexpr int MAX_SA_K_TILES = MAX_K_TILE_COUNT;

static i8_t get_vec_i8(ap_uint<256> word, int lane) {
#pragma HLS INLINE
    u8_t bits = word.range(lane * 8 + 7, lane * 8);
    i8_t value;
    value.range(7, 0) = bits;
    return value;
}

static u16_t ceil_div_u16(u16_t a, u16_t b) {
#pragma HLS INLINE
    return static_cast<u16_t>((a + b - 1) / b);
}

static u16_t conv_out_dim(u16_t in_size, ap_uint<2> stride) {
#pragma HLS INLINE
    const u16_t s = (stride == 0) ? static_cast<u16_t>(1) : static_cast<u16_t>(stride);
    return ceil_div_u16(in_size, s);
}

static u16_t kernel_flat(const conv_cfg_t& cfg) {
#pragma HLS INLINE
    const u16_t k = (cfg.kernel == 1) ? static_cast<u16_t>(1) : static_cast<u16_t>(3);
    return static_cast<u16_t>(cfg.in_c * k * k);
}

static int valid_tm_for_oc_tile(u16_t oc_tile, u16_t out_c) {
#pragma HLS INLINE
    const unsigned oc_base = oc_tile.to_uint() * static_cast<unsigned>(TM);
    const unsigned out_c_u = out_c.to_uint();
    if (oc_base >= out_c_u) {
        return 0;
    }
    const unsigned remaining = out_c_u - oc_base;
    return (remaining > static_cast<unsigned>(TM)) ? TM : static_cast<int>(remaining);
}

static void load_weight_buffer(hls::stream<wgt_vec_t>& wgt_stream,
                               i8_t weight_buf[MAX_SA_K_TILES][TM][TK],
                               u16_t k_tiles,
                               u16_t k_total,
                               u16_t oc_tile,
                               u16_t out_c) {
#pragma HLS INLINE off
    const int k_tiles_i = static_cast<int>(k_tiles.to_uint());
    for (int kt_i = 0; kt_i < MAX_SA_K_TILES; ++kt_i) {
        if (kt_i >= k_tiles_i) {
            break;
        }
        const u16_t kt = static_cast<u16_t>(kt_i);
        for (int tm = 0; tm < TM; ++tm) {
#pragma HLS PIPELINE II=1
            const wgt_vec_t wgt_word = wgt_stream.read();
            const u16_t oc = static_cast<u16_t>(oc_tile * TM + tm);
            for (int tk = 0; tk < TK; ++tk) {
#pragma HLS UNROLL
                const u16_t k_idx = static_cast<u16_t>(kt * TK + tk);
                const bool valid = (kt < MAX_SA_K_TILES && k_idx < k_total && oc < out_c);
                weight_buf[kt][tm][tk] = valid ? get_vec_i8(wgt_word, tk) : i8_t(0);
            }
        }
    }
}

static void unpack_act(act_vec_t act_word, i8_t act_lane[TK]) {
#pragma HLS INLINE
    for (int tk = 0; tk < TK; ++tk) {
#pragma HLS UNROLL
        act_lane[tk] = get_vec_i8(act_word, tk);
    }
}

static void set_psum_i32(psum_vec_t& word, int lane, i32_t value) {
#pragma HLS INLINE
    word.range(lane * 32 + 31, lane * 32) = value.range(31, 0);
}

static void mac_tile(const i8_t act_lane[TK],
                     const i8_t weight_buf[MAX_SA_K_TILES][TM][TK],
                     i32_t psum[TM],
                     u16_t kt,
                     u16_t k_total,
                     u16_t oc_tile,
                     u16_t out_c) {
#pragma HLS INLINE
    for (int tm = 0; tm < TM; ++tm) {
#pragma HLS UNROLL
        const u16_t oc = static_cast<u16_t>(oc_tile * TM + tm);
        i32_t partial = 0;
        if (oc < out_c && kt < MAX_SA_K_TILES) {
            for (int tk = 0; tk < TK; ++tk) {
#pragma HLS UNROLL
                const u16_t k_idx = static_cast<u16_t>(kt * TK + tk);
                if (k_idx < k_total) {
                    partial += static_cast<i32_t>(act_lane[tk]) *
                               static_cast<i32_t>(weight_buf[kt][tm][tk]);
                }
            }
        }
        psum[tm] += partial;
    }
}

void systolic_array_core_row(
    hls::stream<act_vec_t>& act_stream,
    hls::stream<wgt_vec_t>& wgt_stream,
    hls::stream<psum_vec_t>& psum_stream,
    const conv_cfg_t& cfg) {
#pragma HLS INLINE off
    const u16_t out_w = conv_out_dim(cfg.in_w, cfg.stride);
    const u16_t k_total = kernel_flat(cfg);
    const u16_t k_tiles = ceil_div_u16(k_total, TK);
    const u16_t oc_tiles = ceil_div_u16(cfg.out_c, TM);
    const int out_w_i = static_cast<int>(out_w.to_uint());
    const int k_tiles_i = static_cast<int>(k_tiles.to_uint());
    const int oc_tiles_i = static_cast<int>(oc_tiles.to_uint());

    for (int oc_tile_i = 0; oc_tile_i < MAX_C_TILE_COUNT; ++oc_tile_i) {
        if (oc_tile_i >= oc_tiles_i) {
            break;
        }
        const u16_t oc_tile = static_cast<u16_t>(oc_tile_i);
        const int valid_tm_i = valid_tm_for_oc_tile(oc_tile, cfg.out_c);
        i8_t weight_buf[MAX_SA_K_TILES][TM][TK];
#pragma HLS ARRAY_PARTITION variable=weight_buf complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight_buf complete dim=3

        load_weight_buffer(wgt_stream, weight_buf, k_tiles, k_total, oc_tile, cfg.out_c);

        for (int ow_i = 0; ow_i < MAX_FM_W; ++ow_i) {
            if (ow_i >= out_w_i) {
                break;
            }
            i32_t psum[TM];
#pragma HLS ARRAY_PARTITION variable=psum complete dim=1

            for (int tm = 0; tm < TM; ++tm) {
#pragma HLS UNROLL
                psum[tm] = 0;
            }

            for (int kt_i = 0; kt_i < MAX_SA_K_TILES; ++kt_i) {
#pragma HLS PIPELINE II=1
                if (kt_i >= k_tiles_i) {
                    break;
                }
                const u16_t kt = static_cast<u16_t>(kt_i);
                const act_vec_t act_word = act_stream.read();
                i8_t act_lane[TK];
#pragma HLS ARRAY_PARTITION variable=act_lane complete dim=1

                unpack_act(act_word, act_lane);
                mac_tile(act_lane, weight_buf, psum, kt, k_total, oc_tile, cfg.out_c);
            }

            psum_vec_t psum_word = 0;
            for (int tm = 0; tm < TM; ++tm) {
#pragma HLS UNROLL
                const i32_t lane_psum = (tm < valid_tm_i) ? psum[tm] : i32_t(0);
                set_psum_i32(psum_word, tm, lane_psum);
            }
            psum_stream.write(psum_word);
        }
    }
}

}  // namespace esp_int8
