#include "../include/npu_config.hpp"
#include "../include/npu_schedule.hpp"
#include "../include/npu_types.hpp"

namespace esp_int8 {

constexpr int MAX_SA_K_TILES = MAX_K_TILE_COUNT;
constexpr int SA_ACTIVE_TM = 32;
constexpr int SA_OUTPUT_TM = 16;
constexpr int SA_K_TILE_II = 1;
constexpr int SA_OUTPUT_GROUP_COUNT = TM / SA_OUTPUT_TM;
static_assert(TM == 32 && TK == 32 && SA_ACTIVE_TM == 32,
              "Full SA assumes the P7 32x32 organization");
static_assert(TM % SA_ACTIVE_TM == 0 && TM % SA_OUTPUT_TM == 0,
              "SA compute/output groups must divide TM");

static i8_t get_vec_i8(ap_uint<256> word, int lane) {
#pragma HLS INLINE
    u8_t bits = word.range(lane * 8 + 7, lane * 8);
    i8_t value;
    value.range(7, 0) = bits;
    return value;
}

static void unpack_act(act_vec_t act_word, i8_t act_lane[TK]) {
#pragma HLS INLINE
    for (int tk = 0; tk < TK; ++tk) {
#pragma HLS UNROLL
        act_lane[tk] = get_vec_i8(act_word, tk);
    }
}

static void set_psum_half_i32(psum_half_vec_t& word, int lane, i32_t value) {
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

static void mac_tile_active_lanes(const i8_t act_lane0[TK],
                                  const i8_t act_lane1[TK],
                                  const wgt_vec_t weight_buf[TM][MAX_SA_K_TILES],
                                  i32_t psum[TM],
                                  u16_t kt,
                                  u16_t k_total,
                                  bool paired) {
#pragma HLS INLINE
    bool k_valid[TK];
#pragma HLS ARRAY_PARTITION variable=k_valid complete dim=1
    compute_k_valid_lanes(kt, k_total, k_valid);

    for (int lane = 0; lane < SA_ACTIVE_TM; ++lane) {
#pragma HLS UNROLL
        const int weight_lane = paired && lane >= SA_OUTPUT_TM ? lane - SA_OUTPUT_TM : lane;
        const bool use_second_pixel = paired && lane >= SA_OUTPUT_TM;
        i32_t partial = 0;
        const wgt_vec_t wgt_word = weight_buf[weight_lane][kt];
        for (int tk = 0; tk < TK; ++tk) {
#pragma HLS UNROLL
            if (k_valid[tk]) {
                const i8_t act = use_second_pixel ? act_lane1[tk] : act_lane0[tk];
                partial += static_cast<i32_t>(act) *
                           static_cast<i32_t>(get_vec_i8(wgt_word, tk));
            }
        }
        psum[lane] += partial;
    }
}

void systolic_array_core_row(
    hls::stream<act_vec_t>& act_stream0,
    hls::stream<act_vec_t>& act_stream1,
    const wgt_vec_t weight_buf[TM][MAX_SA_K_TILES],
    hls::stream<psum_half_vec_t>& psum_stream,
    hls::stream<conv_cfg_t>& cfg_stream,
    hls::stream<u8_t>& sched_flags_stream) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=weight_buf cyclic factor=SA_ACTIVE_TM dim=1
    const conv_cfg_t cfg = cfg_stream.read();
    const u8_t sched_flags = sched_flags_stream.read();
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
    const bool paired =
        (sched_flags.to_uint() & static_cast<unsigned>(WINDOW_SCHED_FLAG_PIXEL_PARALLEL_2)) != 0U;
    const int issue_count = paired ? ((out_w_i + 1) / 2) : out_w_i;

    for (int issue_i = 0; issue_i < MAX_FM_W; ++issue_i) {
        if (issue_i >= issue_count) {
            break;
        }

        i32_t psum[TM];
#pragma HLS ARRAY_PARTITION variable=psum cyclic factor=SA_ACTIVE_TM dim=1
        for (int tm = 0; tm < TM; ++tm) {
#pragma HLS UNROLL
            psum[tm] = 0;
        }

        for (int kt_i = 0; kt_i < MAX_SA_K_TILES; ++kt_i) {
            if (kt_i >= k_tiles_i) {
                break;
            }
#pragma HLS PIPELINE II=SA_K_TILE_II
            const act_vec_t act_word0 = act_stream0.read();
            const act_vec_t act_word1 = paired ? act_stream1.read() : act_vec_t(0);
            i8_t act_lane0[TK];
            i8_t act_lane1[TK];
#pragma HLS ARRAY_PARTITION variable=act_lane0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=act_lane1 complete dim=1

            unpack_act(act_word0, act_lane0);
            unpack_act(act_word1, act_lane1);
            mac_tile_active_lanes(act_lane0,
                                  act_lane1,
                                  weight_buf,
                                  psum,
                                  static_cast<u16_t>(kt_i),
                                  k_total,
                                  paired);
        }

        const int valid_group_count = paired
                                          ? SA_OUTPUT_GROUP_COUNT
                                          : ((valid_tm_i <= SA_OUTPUT_TM) ? 1 : SA_OUTPUT_GROUP_COUNT);
        for (int group = 0; group < SA_OUTPUT_GROUP_COUNT; ++group) {
#pragma HLS PIPELINE II=1
            if (group >= valid_group_count) {
                break;
            }
            psum_half_vec_t group_word = 0;
            for (int lane = 0; lane < SA_OUTPUT_TM; ++lane) {
#pragma HLS UNROLL
                const int tm = group * SA_OUTPUT_TM + lane;
                const bool lane_valid = paired ? (lane < valid_tm_i) : (tm < valid_tm_i);
                const i32_t lane_psum = lane_valid ? psum[tm] : i32_t(0);
                set_psum_half_i32(group_word, lane, lane_psum);
            }
            psum_stream.write(group_word);
        }
    }
}

}  // namespace esp_int8
