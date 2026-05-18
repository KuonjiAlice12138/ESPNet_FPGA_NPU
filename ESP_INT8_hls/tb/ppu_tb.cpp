#include "../include/npu_config.hpp"
#include "../include/npu_q.hpp"

#include <cstdint>
#include <cstdio>

namespace esp_int8 {
void post_process_unit(
    hls::stream<i32_t>& psum_stream,
    hls::stream<i8_t>& add_a_stream,
    hls::stream<i8_t>& add_b_stream,
    hls::stream<i8_t>& out_stream,
    const post_cfg_t& cfg,
    const i32_t bias[32],
    const i32_t mult[32],
    const u8_t shift[32],
    const i32_t aff_mul[32],
    const i32_t aff_bias[32],
    const u8_t aff_shift[32]);
}  // namespace esp_int8

static int g_failures = 0;

static std::int32_t ref_round_shift(std::int64_t x, std::uint8_t shift) {
    if (shift == 0) {
        return static_cast<std::int32_t>(x);
    }
    const std::int64_t bias = std::int64_t(1) << (shift - 1);
    return static_cast<std::int32_t>((x >= 0) ? ((x + bias) >> shift) : ((x - bias) >> shift));
}

static std::int8_t ref_clamp_i8(std::int32_t x) {
    if (x > 127) {
        return 127;
    }
    if (x < -128) {
        return -128;
    }
    return static_cast<std::int8_t>(x);
}

static std::int8_t ref_act(std::int8_t x, std::uint8_t act_type) {
    return (act_type == esp_int8::ACT_RELU && x < 0) ? 0 : x;
}

static std::int8_t to_i8(esp_int8::i8_t x) {
    return static_cast<std::int8_t>(x.to_int());
}

static void expect_eq(const char* tag, int lane, esp_int8::i8_t got, std::int8_t expected) {
    const std::int8_t got_i8 = to_i8(got);
    if (got_i8 != expected) {
        std::printf("[FAIL] %s lane=%d got=%d expected=%d\n",
                    tag,
                    lane,
                    static_cast<int>(got_i8),
                    static_cast<int>(expected));
        ++g_failures;
    }
}

static void zero_params(esp_int8::i32_t bias[32],
                        esp_int8::i32_t mult[32],
                        esp_int8::u8_t shift[32],
                        esp_int8::i32_t aff_mul[32],
                        esp_int8::i32_t aff_bias[32],
                        esp_int8::u8_t aff_shift[32]) {
    for (int i = 0; i < esp_int8::TM; ++i) {
        bias[i] = 0;
        mult[i] = 1;
        shift[i] = 0;
        aff_mul[i] = 1;
        aff_bias[i] = 0;
        aff_shift[i] = 0;
    }
}

static void test_conv_post() {
    hls::stream<esp_int8::i32_t> psum_stream;
    hls::stream<esp_int8::i8_t> add_a_stream;
    hls::stream<esp_int8::i8_t> add_b_stream;
    hls::stream<esp_int8::i8_t> out_stream;

    esp_int8::i32_t bias[32];
    esp_int8::i32_t mult[32];
    esp_int8::u8_t shift[32];
    esp_int8::i32_t aff_mul[32];
    esp_int8::i32_t aff_bias[32];
    esp_int8::u8_t aff_shift[32];
    zero_params(bias, mult, shift, aff_mul, aff_bias, aff_shift);

    const std::int32_t psum[8] = {10, -10, 1000, -1000, 63, -65, 5, -5};
    const std::int32_t b[8] = {1, -3, 0, 0, 0, 0, -20, -20};
    const std::int32_t m[8] = {2, 3, 1, 1, 1, 1, 4, 4};
    const std::uint8_t s[8] = {1, 1, 0, 0, 1, 1, 2, 2};

    for (int i = 0; i < 8; ++i) {
        psum_stream.write(psum[i]);
        bias[i] = b[i];
        mult[i] = m[i];
        shift[i] = s[i];
    }

    esp_int8::post_cfg_t cfg;
    cfg.mode = esp_int8::POST_CONV;
    cfg.act_type = esp_int8::ACT_RELU;
    cfg.requant_bypass = 0;
    cfg.valid_tm = 8;

    esp_int8::post_process_unit(psum_stream, add_a_stream, add_b_stream, out_stream,
                                cfg, bias, mult, shift, aff_mul, aff_bias, aff_shift);

    for (int i = 0; i < 8; ++i) {
        const std::int64_t scaled = (static_cast<std::int64_t>(psum[i]) + b[i]) * m[i];
        const std::int8_t exp = ref_act(ref_clamp_i8(ref_round_shift(scaled, s[i])), esp_int8::ACT_RELU);
        expect_eq("CONV_POST", i, out_stream.read(), exp);
    }
    if (!out_stream.empty()) {
        std::printf("[FAIL] CONV_POST produced extra output\n");
        ++g_failures;
    }

    psum_stream.write(-9);
    bias[0] = 0;
    mult[0] = 1;
    shift[0] = 1;
    cfg.act_type = esp_int8::ACT_NONE;
    cfg.valid_tm = 1;

    esp_int8::post_process_unit(psum_stream, add_a_stream, add_b_stream, out_stream,
                                cfg, bias, mult, shift, aff_mul, aff_bias, aff_shift);
    expect_eq("CONV_POST_NO_RELU", 0, out_stream.read(), ref_round_shift(-9, 1));
}

static void test_add_post() {
    hls::stream<esp_int8::i32_t> psum_stream;
    hls::stream<esp_int8::i8_t> add_a_stream;
    hls::stream<esp_int8::i8_t> add_b_stream;
    hls::stream<esp_int8::i8_t> out_stream;

    esp_int8::i32_t bias[32];
    esp_int8::i32_t mult[32];
    esp_int8::u8_t shift[32];
    esp_int8::i32_t aff_mul[32];
    esp_int8::i32_t aff_bias[32];
    esp_int8::u8_t aff_shift[32];
    zero_params(bias, mult, shift, aff_mul, aff_bias, aff_shift);

    const std::int8_t a[6] = {100, 100, -100, -100, -30, 20};
    const std::int8_t b[6] = {40, 20, -50, 10, 20, -70};
    for (int i = 0; i < 6; ++i) {
        add_a_stream.write(a[i]);
        add_b_stream.write(b[i]);
    }

    esp_int8::post_cfg_t cfg;
    cfg.mode = esp_int8::POST_ADD;
    cfg.act_type = esp_int8::ACT_NONE;
    cfg.requant_bypass = 1;
    cfg.valid_tm = 6;

    esp_int8::post_process_unit(psum_stream, add_a_stream, add_b_stream, out_stream,
                                cfg, bias, mult, shift, aff_mul, aff_bias, aff_shift);

    for (int i = 0; i < 6; ++i) {
        expect_eq("ADD_POST_BYPASS", i, out_stream.read(), ref_clamp_i8(a[i] + b[i]));
    }

    add_a_stream.write(-80);
    add_b_stream.write(10);
    cfg.act_type = esp_int8::ACT_RELU;
    cfg.requant_bypass = 1;
    cfg.valid_tm = 1;

    esp_int8::post_process_unit(psum_stream, add_a_stream, add_b_stream, out_stream,
                                cfg, bias, mult, shift, aff_mul, aff_bias, aff_shift);
    expect_eq("ADD_POST_BYPASS_RELU", 0, out_stream.read(), 0);

    add_a_stream.write(-20);
    add_b_stream.write(10);
    mult[0] = 3;
    shift[0] = 1;
    cfg.requant_bypass = 0;
    cfg.act_type = esp_int8::ACT_RELU;
    cfg.valid_tm = 1;

    esp_int8::post_process_unit(psum_stream, add_a_stream, add_b_stream, out_stream,
                                cfg, bias, mult, shift, aff_mul, aff_bias, aff_shift);
    const std::int64_t scaled = static_cast<std::int64_t>(-10) * 3;
    const std::int8_t exp = ref_act(ref_clamp_i8(ref_round_shift(scaled, 1)), esp_int8::ACT_RELU);
    expect_eq("ADD_POST_REQUANT", 0, out_stream.read(), exp);

    add_a_stream.write(-7);
    add_b_stream.write(0);
    mult[0] = 1;
    shift[0] = 1;
    cfg.requant_bypass = 0;
    cfg.act_type = esp_int8::ACT_NONE;
    cfg.valid_tm = 1;

    esp_int8::post_process_unit(psum_stream, add_a_stream, add_b_stream, out_stream,
                                cfg, bias, mult, shift, aff_mul, aff_bias, aff_shift);
    expect_eq("ADD_POST_REQUANT_NO_RELU", 0, out_stream.read(), ref_round_shift(-7, 1));
}

static void test_affine_post() {
    hls::stream<esp_int8::i32_t> psum_stream;
    hls::stream<esp_int8::i8_t> add_a_stream;
    hls::stream<esp_int8::i8_t> add_b_stream;
    hls::stream<esp_int8::i8_t> out_stream;

    esp_int8::i32_t bias[32];
    esp_int8::i32_t mult[32];
    esp_int8::u8_t shift[32];
    esp_int8::i32_t aff_mul[32];
    esp_int8::i32_t aff_bias[32];
    esp_int8::u8_t aff_shift[32];
    zero_params(bias, mult, shift, aff_mul, aff_bias, aff_shift);

    const std::int8_t x[5] = {10, -10, 64, -64, 127};
    const std::int32_t mul[5] = {3, 3, 4, 4, 2};
    const std::int32_t add[5] = {1, -1, 0, 0, 0};
    const std::uint8_t sh[5] = {1, 1, 0, 0, 0};
    for (int i = 0; i < 5; ++i) {
        add_a_stream.write(x[i]);
        aff_mul[i] = mul[i];
        aff_bias[i] = add[i];
        aff_shift[i] = sh[i];
    }

    esp_int8::post_cfg_t cfg;
    cfg.mode = esp_int8::POST_AFFINE;
    cfg.act_type = esp_int8::ACT_RELU;
    cfg.requant_bypass = 0;
    cfg.valid_tm = 5;

    esp_int8::post_process_unit(psum_stream, add_a_stream, add_b_stream, out_stream,
                                cfg, bias, mult, shift, aff_mul, aff_bias, aff_shift);

    for (int i = 0; i < 5; ++i) {
        const std::int64_t scaled = static_cast<std::int64_t>(x[i]) * mul[i] + add[i];
        const std::int8_t exp = ref_act(ref_clamp_i8(ref_round_shift(scaled, sh[i])), esp_int8::ACT_RELU);
        expect_eq("AFFINE_POST", i, out_stream.read(), exp);
    }

    add_a_stream.write(-5);
    aff_mul[0] = 3;
    aff_bias[0] = 0;
    aff_shift[0] = 1;
    cfg.act_type = esp_int8::ACT_NONE;
    cfg.valid_tm = 1;

    esp_int8::post_process_unit(psum_stream, add_a_stream, add_b_stream, out_stream,
                                cfg, bias, mult, shift, aff_mul, aff_bias, aff_shift);
    expect_eq("AFFINE_POST_NO_RELU", 0, out_stream.read(), ref_round_shift(-15, 1));
}

static void test_valid_tm_tail() {
    hls::stream<esp_int8::i32_t> psum_stream;
    hls::stream<esp_int8::i8_t> add_a_stream;
    hls::stream<esp_int8::i8_t> add_b_stream;
    hls::stream<esp_int8::i8_t> out_stream;

    esp_int8::i32_t bias[32];
    esp_int8::i32_t mult[32];
    esp_int8::u8_t shift[32];
    esp_int8::i32_t aff_mul[32];
    esp_int8::i32_t aff_bias[32];
    esp_int8::u8_t aff_shift[32];
    zero_params(bias, mult, shift, aff_mul, aff_bias, aff_shift);

    for (int i = 0; i < 3; ++i) {
        psum_stream.write(i + 1);
        mult[i] = 1;
        shift[i] = 0;
    }

    esp_int8::post_cfg_t cfg;
    cfg.mode = esp_int8::POST_CONV;
    cfg.act_type = esp_int8::ACT_NONE;
    cfg.requant_bypass = 0;
    cfg.valid_tm = 3;

    esp_int8::post_process_unit(psum_stream, add_a_stream, add_b_stream, out_stream,
                                cfg, bias, mult, shift, aff_mul, aff_bias, aff_shift);

    for (int i = 0; i < 3; ++i) {
        expect_eq("VALID_TM_TAIL", i, out_stream.read(), static_cast<std::int8_t>(i + 1));
    }
    if (!out_stream.empty()) {
        std::printf("[FAIL] VALID_TM_TAIL produced extra output\n");
        ++g_failures;
    }
}

int main() {
    test_conv_post();
    test_add_post();
    test_affine_post();
    test_valid_tm_tail();

    if (g_failures != 0) {
        std::printf("ppu_tb failed: %d failure(s)\n", g_failures);
        return 1;
    }

    std::printf("ppu_tb passed\n");
    return 0;
}
