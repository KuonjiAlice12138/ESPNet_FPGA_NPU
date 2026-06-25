#include "../include/npu_config.hpp"
#include "../include/npu_types.hpp"

namespace esp_int8 {

constexpr int MAX_SA_K_TILES = MAX_K_TILE_COUNT;
constexpr int SA_GROUP_COUNT = 2;
constexpr int SA_GROUP_TM = TM / SA_GROUP_COUNT;
constexpr int SA_SEGMENT_TM = 8;
static_assert(TM == 32 && TK == 32 && SA_GROUP_TM == 16,
              "Grouped SA assumes the P6 2x16x32 organization");
static_assert(SA_GROUP_TM % SA_SEGMENT_TM == 0,
              "SA group must split into equal physical segments");

static i8_t get_vec_i8(ap_uint<256> word, int lane) {
#pragma HLS INLINE
    u8_t bits = word.range(lane * 8 + 7, lane * 8);
    i8_t value;
    value.range(7, 0) = bits;
    return value;
}

static void load_weight_buffer(hls::stream<wgt_vec_t>& wgt_stream,
                               wgt_vec_t weight_buf[TM][MAX_SA_K_TILES],
                               u16_t k_tiles) {
#pragma HLS INLINE off
    const int k_tiles_i = static_cast<int>(k_tiles.to_uint());
    for (int kt_i = 0; kt_i < MAX_SA_K_TILES; ++kt_i) {
        if (kt_i >= k_tiles_i) {
            break;
        }
        for (int tm = 0; tm < TM; ++tm) {
#pragma HLS PIPELINE II=1
            weight_buf[tm][kt_i] = wgt_stream.read();
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

static void compute_k_valid_lanes(u16_t kt,
                                  u16_t k_total,
                                  bool k_valid[TK]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=k_valid complete dim=1
    const u16_t k_base = static_cast<u16_t>(kt * TK);
    for (int tk = 0; tk < TK; ++tk) {
#pragma HLS UNROLL
        k_valid[tk] = static_cast<u16_t>(k_base + tk) < k_total;
    }
}

template <int BASE_TM>
static void mac_tile_segment(const i8_t act_lane[TK],
                             const wgt_vec_t weight_buf[TM][MAX_SA_K_TILES],
                             i32_t psum[TM],
                             u16_t kt,
                             u16_t k_total) {
#pragma HLS INLINE
    bool k_valid[TK];
#pragma HLS ARRAY_PARTITION variable=k_valid complete dim=1
    compute_k_valid_lanes(kt, k_total, k_valid);

    for (int lane = 0; lane < SA_SEGMENT_TM; ++lane) {
#pragma HLS UNROLL
        const int tm = BASE_TM + lane;
        i32_t partial = 0;
        const wgt_vec_t wgt_word = weight_buf[tm][kt];
        for (int tk = 0; tk < TK; ++tk) {
#pragma HLS UNROLL
            if (k_valid[tk]) {
                partial += static_cast<i32_t>(act_lane[tk]) *
                           static_cast<i32_t>(get_vec_i8(wgt_word, tk));
            }
        }
        psum[tm] += partial;
    }
}

static void mac_tile_all_lanes(const i8_t act_lane[TK],
                               const wgt_vec_t weight_buf[TM][MAX_SA_K_TILES],
                               i32_t psum[TM],
                               u16_t kt,
                               u16_t k_total) {
#pragma HLS INLINE
    mac_tile_segment<0>(act_lane, weight_buf, psum, kt, k_total);
    mac_tile_segment<SA_SEGMENT_TM>(act_lane, weight_buf, psum, kt, k_total);
    mac_tile_segment<SA_GROUP_TM>(act_lane, weight_buf, psum, kt, k_total);
    mac_tile_segment<SA_GROUP_TM + SA_SEGMENT_TM>(act_lane, weight_buf, psum, kt, k_total);
}

void systolic_array_core_row(
    hls::stream<act_vec_t>& act_stream,
    hls::stream<wgt_vec_t>& wgt_stream,
    hls::stream<psum_vec_t>& psum_stream,
    const conv_cfg_t& cfg) {
#pragma HLS INLINE off
    const u16_t stride = (cfg.stride == 0)
                             ? static_cast<u16_t>(1)
                             : static_cast<u16_t>(cfg.stride);
    const u16_t out_w = static_cast<u16_t>((cfg.in_w + stride - 1) / stride);
    const u16_t kernel_size = (cfg.kernel == 1) ? static_cast<u16_t>(1)
                                                : static_cast<u16_t>(3);
    const u16_t k_total = static_cast<u16_t>(cfg.in_c * kernel_size * kernel_size);
    const u16_t k_tiles = static_cast<u16_t>((k_total + TK - 1) / TK);
    const int out_w_i = static_cast<int>(out_w.to_uint());
    const int k_tiles_i = static_cast<int>(k_tiles.to_uint());
    const int valid_tm_i = (cfg.out_c.to_uint() > static_cast<unsigned>(TM))
                               ? TM
                               : static_cast<int>(cfg.out_c.to_uint());

    wgt_vec_t weight_buf[TM][MAX_SA_K_TILES];
#pragma HLS ARRAY_PARTITION variable=weight_buf complete dim=1
#pragma HLS BIND_STORAGE variable=weight_buf type=ram_1p impl=bram

    load_weight_buffer(wgt_stream, weight_buf, k_tiles);

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
            if (kt_i >= k_tiles_i) {
                break;
            }
#pragma HLS PIPELINE II=1
            const act_vec_t act_word = act_stream.read();
            i8_t act_lane[TK];
#pragma HLS ARRAY_PARTITION variable=act_lane complete dim=1

            unpack_act(act_word, act_lane);
            mac_tile_all_lanes(act_lane,
                               weight_buf,
                               psum,
                               static_cast<u16_t>(kt_i),
                               k_total);
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

}  // namespace esp_int8
