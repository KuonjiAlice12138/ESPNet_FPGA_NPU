#include "../include/npu_config.hpp"
#include "../include/npu_types.hpp"

#include <cstdint>
#include <cstdio>

namespace esp_int8 {
void systolic_array_core_row(
    hls::stream<act_vec_t>& act_stream,
    hls::stream<wgt_vec_t>& wgt_stream,
    hls::stream<psum_vec_t>& psum_stream,
    const conv_cfg_t& cfg);
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

static esp_int8::i32_t get_psum_i32(const esp_int8::psum_vec_t& word, int lane) {
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
                     int out_pixels) {
    hls::stream<esp_int8::act_vec_t> act_stream;
    hls::stream<esp_int8::wgt_vec_t> wgt_stream;
    hls::stream<esp_int8::psum_vec_t> psum_stream;

    const int k_tiles = (k_total + esp_int8::TK - 1) / esp_int8::TK;
    for (int kt = 0; kt < k_tiles; ++kt) {
        for (int tm = 0; tm < esp_int8::TM; ++tm) {
            const int oc = tm;
            const std::int8_t* wgt = (oc < cfg.out_c.to_int())
                ? (wgt_by_oc + oc * k_total)
                : (wgt_by_oc);
            wgt_stream.write(pack_wgt(wgt, k_total, kt));
        }
    }

    for (int pix = 0; pix < out_pixels; ++pix) {
        const std::int8_t* act = act_by_pixel + pix * k_total;
        for (int kt = 0; kt < k_tiles; ++kt) {
            act_stream.write(pack_act(act, k_total, kt));
        }
    }

    esp_int8::systolic_array_core_row(act_stream, wgt_stream, psum_stream, cfg);

    int out_idx = 0;
    for (int pix = 0; pix < out_pixels; ++pix) {
        const std::int8_t* act = act_by_pixel + pix * k_total;
        const esp_int8::psum_vec_t psum_word = psum_stream.read();
        for (int tm = 0; tm < esp_int8::TM; ++tm) {
            const int oc = tm;
            if (oc < cfg.out_c.to_int()) {
                const std::int8_t* wgt = wgt_by_oc + oc * k_total;
                expect_eq(tag, out_idx, get_psum_i32(psum_word, tm), dot_ref(act, wgt, k_total));
                ++out_idx;
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

int main() {
    test_1x1_basic();
    test_tail_and_multi_ktile();
    test_3x3_flatten();

    if (g_failures != 0) {
        std::printf("sa_core_tb failed: %d failure(s)\n", g_failures);
        return 1;
    }

    std::printf("sa_core_tb passed\n");
    return 0;
}
