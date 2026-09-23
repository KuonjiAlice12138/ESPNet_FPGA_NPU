#include "../include/npu_config.hpp"
#include "../include/npu_schedule.hpp"
#include "../include/npu_types.hpp"

#include <cstdint>
#include <cstdio>

namespace esp_int8 {
void systolic_array_core_row(
    hls::stream<act_vec_t>& act_stream0,
    hls::stream<act_vec_t>& act_stream1,
    const wgt_vec_t weight_buf[TM][MAX_K_TILE_COUNT],
    hls::stream<psum_half_vec_t>& psum_stream,
    hls::stream<conv_cfg_t>& cfg_stream,
    hls::stream<u8_t>& sched_flags_stream,
    volatile u8_t& prof_conv_sa_state);
}  // namespace esp_int8

static int g_failures = 0;

static void set_vec_i8(ap_uint<256>& word, int lane, std::int8_t value) {
    word.range(lane * 8 + 7, lane * 8) = static_cast<std::uint8_t>(value);
}

static esp_int8::act_vec_t pack_act(const std::int8_t* values, int k_total, int k_tile) {
    esp_int8::act_vec_t word = 0;
    for (int tk = 0; tk < esp_int8::TK; ++tk) {
        const int idx = k_tile * esp_int8::TK + tk;
        set_vec_i8(word, tk, (idx < k_total) ? values[idx] : 0);
    }
    return word;
}

static esp_int8::wgt_vec_t pack_wgt(const std::int8_t* values, int k_total, int k_tile) {
    esp_int8::wgt_vec_t word = 0;
    for (int tk = 0; tk < esp_int8::TK; ++tk) {
        const int idx = k_tile * esp_int8::TK + tk;
        set_vec_i8(word, tk, (idx < k_total) ? values[idx] : 0);
    }
    return word;
}

static std::int32_t dot_ref(const std::int8_t* act, const std::int8_t* wgt, int k_total) {
    std::int32_t sum = 0;
    for (int i = 0; i < k_total; ++i) {
        sum += static_cast<std::int32_t>(act[i]) * static_cast<std::int32_t>(wgt[i]);
    }
    return sum;
}

static esp_int8::i32_t get_psum_i32(const esp_int8::psum_half_vec_t& word, int lane) {
    esp_int8::i32_t value;
    value.range(31, 0) = word.range(lane * 32 + 31, lane * 32);
    return value;
}

static void expect_eq(const char* tag, int index, esp_int8::i32_t got, std::int32_t expected) {
    const std::int32_t got_i32 = got.to_int();
    if (got_i32 != expected) {
        std::printf("[FAIL] %s idx=%d got=%d expected=%d\n", tag, index, got_i32, expected);
        ++g_failures;
    }
}

static void run_case(const char* tag,
                     const esp_int8::conv_cfg_t& cfg,
                     const std::int8_t* act_by_pixel,
                     const std::int8_t* wgt_by_oc,
                     int k_total,
                     int out_pixels,
                     bool paired = false) {
    hls::stream<esp_int8::act_vec_t> act_stream0;
    hls::stream<esp_int8::act_vec_t> act_stream1;
    hls::stream<esp_int8::psum_half_vec_t> psum_stream;
    hls::stream<esp_int8::conv_cfg_t> cfg_stream;
    hls::stream<esp_int8::u8_t> sched_flags_stream;
    volatile esp_int8::u8_t prof_conv_sa_state = 0;
    esp_int8::wgt_vec_t weight_buf[esp_int8::TM][esp_int8::MAX_K_TILE_COUNT] = {};

    const int k_tiles = (k_total + esp_int8::TK - 1) / esp_int8::TK;
    for (int kt = 0; kt < k_tiles; ++kt) {
        for (int tm = 0; tm < esp_int8::TM; ++tm) {
            const int oc = tm;
            const std::int8_t* wgt = (oc < cfg.out_c.to_int())
                ? (wgt_by_oc + oc * k_total)
                : (wgt_by_oc);
            weight_buf[tm][kt] = pack_wgt(wgt, k_total, kt);
        }
    }

    const int issue_count = paired ? (out_pixels + 1) / 2 : out_pixels;
    for (int issue = 0; issue < issue_count; ++issue) {
        const int pix0 = paired ? issue * 2 : issue;
        const std::int8_t* act0 = act_by_pixel + pix0 * k_total;
        for (int kt = 0; kt < k_tiles; ++kt) {
            act_stream0.write(pack_act(act0, k_total, kt));
            if (paired) {
                const std::int8_t* act1 = (pix0 + 1 < out_pixels)
                                              ? act_by_pixel + (pix0 + 1) * k_total
                                              : act0;
                act_stream1.write(pack_act(act1, k_total, kt));
            }
        }
    }

    const esp_int8::u8_t flags = paired
                                     ? static_cast<esp_int8::u8_t>(
                                           static_cast<unsigned>(esp_int8::WINDOW_SCHED_FLAG_PIXEL_PARALLEL_2))
                                     : static_cast<esp_int8::u8_t>(0);
    cfg_stream.write(cfg);
    sched_flags_stream.write(flags);
    esp_int8::systolic_array_core_row(
        act_stream0,
        act_stream1,
        weight_buf,
        psum_stream,
        cfg_stream,
        sched_flags_stream,
        prof_conv_sa_state);

    int out_idx = 0;
    for (int pix = 0; pix < out_pixels; ++pix) {
        if (paired && (pix & 1) != 0) {
            continue;
        }
        const esp_int8::psum_half_vec_t psum_word0 = psum_stream.read();
        esp_int8::psum_half_vec_t psum_word1 = 0;
        if (paired || cfg.out_c.to_int() > esp_int8::TM / 2) {
            psum_word1 = psum_stream.read();
        }
        const int pixel_count = paired && pix + 1 < out_pixels ? 2 : 1;
        for (int pixel_lane = 0; pixel_lane < pixel_count; ++pixel_lane) {
          const std::int8_t* pixel_act = act_by_pixel + (pix + pixel_lane) * k_total;
          for (int oc = 0; oc < cfg.out_c.to_int(); ++oc) {
            if (oc < cfg.out_c.to_int()) {
                const bool second_group = paired ? (pixel_lane != 0) : (oc >= esp_int8::TM / 2);
                const esp_int8::psum_half_vec_t& psum_word = second_group ? psum_word1 : psum_word0;
                const int psum_lane = paired ? oc : (oc % (esp_int8::TM / 2));
                const std::int8_t* wgt = wgt_by_oc + oc * k_total;
                expect_eq(tag,
                          out_idx,
                          get_psum_i32(psum_word, psum_lane),
                          dot_ref(pixel_act, wgt, k_total));
                ++out_idx;
            }
          }
        }
    }

    if (!psum_stream.empty()) {
        std::printf("[FAIL] %s produced extra psum data\n", tag);
        ++g_failures;
    }
}

static void test_1x1_basic() {
    esp_int8::conv_cfg_t cfg;
    cfg.in_h = 1;
    cfg.in_w = 2;
    cfg.in_c = 3;
    cfg.out_c = 2;
    cfg.kernel = 1;
    cfg.stride = 1;
    cfg.dilation = 1;
    cfg.bias_en = 0;

    const std::int8_t act[] = {
        1, -2, 3,
        -4, 5, -6,
    };
    const std::int8_t wgt[] = {
        2, 3, 4,
        -1, 2, -3,
    };

    run_case("SA_1X1_BASIC", cfg, act, wgt, 3, 2);
}

static void test_tail_and_multi_ktile() {
    static std::int8_t act[35];
    static std::int8_t wgt[31 * 35];

    for (int k = 0; k < 35; ++k) {
        act[k] = static_cast<std::int8_t>((k % 9) - 4);
    }
    for (int oc = 0; oc < 31; ++oc) {
        for (int k = 0; k < 35; ++k) {
            wgt[oc * 35 + k] = static_cast<std::int8_t>(((oc + 2 * k) % 7) - 3);
        }
    }

    esp_int8::conv_cfg_t cfg;
    cfg.in_h = 1;
    cfg.in_w = 1;
    cfg.in_c = 35;
    cfg.out_c = 31;
    cfg.kernel = 1;
    cfg.stride = 1;
    cfg.dilation = 1;
    cfg.bias_en = 0;

    run_case("SA_TAIL_MULTI_KTILE", cfg, act, wgt, 35, 1);
}

static void run_k_tile_case(int expected_tiles,
                            int in_c,
                            int kernel,
                            const char* tag) {
    constexpr int MAX_TEST_K = esp_int8::TK * 37;
    static std::int8_t act[MAX_TEST_K];
    static std::int8_t wgt[esp_int8::TM * MAX_TEST_K];
    const int k_total = in_c * kernel * kernel;
    const int actual_tiles = (k_total + esp_int8::TK - 1) / esp_int8::TK;
    if (actual_tiles != expected_tiles || k_total > MAX_TEST_K) {
        std::printf("[FAIL] %s setup tiles=%d expected=%d k_total=%d\n",
                    tag, actual_tiles, expected_tiles, k_total);
        ++g_failures;
        return;
    }
    for (int k = 0; k < k_total; ++k) {
        act[k] = static_cast<std::int8_t>((7 * k + expected_tiles) % 17 - 8);
    }
    for (int oc = 0; oc < 31; ++oc) {
        for (int k = 0; k < k_total; ++k) {
            wgt[oc * k_total + k] =
                static_cast<std::int8_t>((5 * oc + 3 * k) % 19 - 9);
        }
    }

    esp_int8::conv_cfg_t cfg;
    cfg.in_h = 1;
    cfg.in_w = 1;
    cfg.in_c = in_c;
    cfg.out_c = 31;
    cfg.kernel = kernel;
    cfg.stride = 1;
    cfg.dilation = 1;
    cfg.bias_en = 0;
    run_case(tag, cfg, act, wgt, k_total, 1);
}

static void test_k_tile_depths() {
    run_k_tile_case(1, 31, 1, "SA_KTILE_1");
    run_k_tile_case(2, 33, 1, "SA_KTILE_2");
    run_k_tile_case(4, 12, 3, "SA_KTILE_4");
    run_k_tile_case(8, 25, 3, "SA_KTILE_8");
    run_k_tile_case(37, 131, 3, "SA_KTILE_37");
}

static void test_signed_extremes() {
    static std::int8_t act[esp_int8::TK];
    static std::int8_t wgt[2 * esp_int8::TK];
    for (int k = 0; k < esp_int8::TK; ++k) {
        act[k] = -128;
        wgt[k] = -128;
        wgt[esp_int8::TK + k] = 127;
    }

    esp_int8::conv_cfg_t cfg;
    cfg.in_h = 1;
    cfg.in_w = 1;
    cfg.in_c = esp_int8::TK;
    cfg.out_c = 2;
    cfg.kernel = 1;
    cfg.stride = 1;
    cfg.dilation = 1;
    cfg.bias_en = 0;
    run_case("SA_SIGNED_EXTREMES", cfg, act, wgt, esp_int8::TK, 1);
}

static void test_3x3_flatten() {
    static std::int8_t act[18];
    static std::int8_t wgt[4 * 18];

    for (int k = 0; k < 18; ++k) {
        act[k] = static_cast<std::int8_t>((k % 5) - 2);
    }
    for (int oc = 0; oc < 4; ++oc) {
        for (int k = 0; k < 18; ++k) {
            wgt[oc * 18 + k] = static_cast<std::int8_t>(((3 * oc + k) % 11) - 5);
        }
    }

    esp_int8::conv_cfg_t cfg;
    cfg.in_h = 1;
    cfg.in_w = 1;
    cfg.in_c = 2;
    cfg.out_c = 4;
    cfg.kernel = 3;
    cfg.stride = 1;
    cfg.dilation = 1;
    cfg.bias_en = 0;

    run_case("SA_3X3_FLATTEN", cfg, act, wgt, 18, 1);
}

static void test_two_pixel_odd_tail() {
    static std::int8_t act[3 * 12];
    static std::int8_t wgt[12 * 12];
    for (int pix = 0; pix < 3; ++pix) {
        for (int k = 0; k < 12; ++k) {
            act[pix * 12 + k] = static_cast<std::int8_t>(((pix + 2) * (k + 1)) % 9 - 4);
        }
    }
    for (int oc = 0; oc < 12; ++oc) {
        for (int k = 0; k < 12; ++k) {
            wgt[oc * 12 + k] = static_cast<std::int8_t>((oc + 3 * k) % 7 - 3);
        }
    }
    esp_int8::conv_cfg_t cfg;
    cfg.in_h = 1;
    cfg.in_w = 3;
    cfg.in_c = 12;
    cfg.out_c = 12;
    cfg.kernel = 1;
    cfg.stride = 1;
    cfg.dilation = 1;
    cfg.bias_en = 0;
    run_case("SA_TWO_PIXEL_ODD_TAIL", cfg, act, wgt, 12, 3, true);
}

int main() {
    test_1x1_basic();
    test_tail_and_multi_ktile();
    test_k_tile_depths();
    test_signed_extremes();
    test_3x3_flatten();
    test_two_pixel_odd_tail();

    if (g_failures != 0) {
        std::printf("sa_core_tb failed: %d failure(s)\n", g_failures);
        return 1;
    }

    std::printf("sa_core_tb passed\n");
    return 0;
}
