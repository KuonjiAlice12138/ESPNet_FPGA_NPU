#include "../include/npu_config.hpp"
#include "../include/npu_uop.hpp"

#include <cstdint>
#include <cstdio>

namespace esp_int8 {
bool on_chip_memory_get_tensor_desc(u8_t tensor_id, tensor_desc_t& desc);
bool on_chip_memory_write_tile(u8_t tensor_id, u16_t h, u16_t w, u16_t c_begin, u8_t valid_c, const i8_t tile[TM]);
void window_generator(const tensor_desc_t& src_desc,
                      hls::stream<act_vec_t>& act_stream,
                      const conv_cfg_t& cfg);
}  // namespace esp_int8

static int g_failures = 0;

static int u16_to_int(esp_int8::u16_t value) {
    return static_cast<int>(value.to_uint());
}

static int cfg_kernel(const esp_int8::conv_cfg_t& cfg) {
    return (cfg.kernel == 1) ? 1 : 3;
}

static int cfg_stride(const esp_int8::conv_cfg_t& cfg) {
    return (cfg.stride == 0) ? 1 : static_cast<int>(cfg.stride.to_uint());
}

static int cfg_dilation(const esp_int8::conv_cfg_t& cfg) {
    return (cfg.dilation == 0) ? 1 : static_cast<int>(cfg.dilation.to_uint());
}

static int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

static std::int8_t lane_i8(esp_int8::act_vec_t word, int lane) {
    esp_int8::u8_t bits = word.range(lane * 8 + 7, lane * 8);
    esp_int8::i8_t value;
    value.range(7, 0) = bits;
    return static_cast<std::int8_t>(value.to_int());
}

static std::int8_t input_pattern(int h, int w, int c) {
    const int raw = (h * 17 + w * 11 + c * 5 + 13) % 127;
    return static_cast<std::int8_t>(raw - 63);
}

static void expect(bool cond, const char* msg) {
    if (!cond) {
        if (g_failures < 40) {
            std::printf("[FAIL] %s\n", msg);
        }
        ++g_failures;
    }
}

static bool fill_tensor_pattern(std::uint8_t tensor_id, int h, int w, int c) {
    for (int yy = 0; yy < h; ++yy) {
        for (int xx = 0; xx < w; ++xx) {
            for (int cb = 0; cb < c; cb += esp_int8::TM) {
                esp_int8::i8_t tile[esp_int8::TM];
                for (int lane = 0; lane < esp_int8::TM; ++lane) {
                    const int ch = cb + lane;
                    if (ch < c) {
                        tile[lane] = static_cast<esp_int8::i8_t>(input_pattern(yy, xx, ch));
                    } else {
                        tile[lane] = 0;
                    }
                }

                const int valid = ((c - cb) > esp_int8::TM) ? esp_int8::TM : (c - cb);
                if (!esp_int8::on_chip_memory_write_tile(static_cast<esp_int8::u8_t>(tensor_id),
                                                         static_cast<esp_int8::u16_t>(yy),
                                                         static_cast<esp_int8::u16_t>(xx),
                                                         static_cast<esp_int8::u16_t>(cb),
                                                         static_cast<esp_int8::u8_t>(valid),
                                                         tile)) {
                    return false;
                }
            }
        }
    }
    return true;
}

static std::int8_t expected_value(const esp_int8::conv_cfg_t& cfg, int oh, int ow, int k_idx) {
    const int in_h = u16_to_int(cfg.in_h);
    const int in_w = u16_to_int(cfg.in_w);
    const int in_c = u16_to_int(cfg.in_c);
    const int kernel = cfg_kernel(cfg);
    const int stride = cfg_stride(cfg);
    const int dilation = cfg_dilation(cfg);
    const int k_total = in_c * kernel * kernel;

    if (k_idx >= k_total) {
        return 0;
    }

    int ih = 0;
    int iw = 0;
    int cin = 0;
    if (kernel == 1) {
        cin = k_idx;
        ih = oh * stride;
        iw = ow * stride;
    } else {
        cin = k_idx % in_c;
        const int spatial_idx = k_idx / in_c;
        const int kh = spatial_idx / kernel;
        const int kw = spatial_idx % kernel;
        ih = oh * stride + kh * dilation - dilation;
        iw = ow * stride + kw * dilation - dilation;
    }

    if (ih < 0 || iw < 0 || ih >= in_h || iw >= in_w || cin >= in_c) {
        return 0;
    }

    return input_pattern(ih, iw, cin);
}

static esp_int8::conv_cfg_t make_cfg(int in_h, int in_w, int in_c, int kernel, int stride, int dilation) {
    esp_int8::conv_cfg_t cfg;
    cfg.in_h = static_cast<esp_int8::u16_t>(in_h);
    cfg.in_w = static_cast<esp_int8::u16_t>(in_w);
    cfg.in_c = static_cast<esp_int8::u16_t>(in_c);
    cfg.out_c = 1;
    cfg.kernel = static_cast<ap_uint<2> >(kernel);
    cfg.stride = static_cast<ap_uint<2> >(stride);
    cfg.dilation = static_cast<ap_uint<5> >(dilation);
    cfg.bias_en = 0;
    return cfg;
}

static void check_generator_all(const char* tag, std::uint8_t tensor_id, const esp_int8::conv_cfg_t& cfg) {
    esp_int8::tensor_desc_t desc;
    expect(esp_int8::on_chip_memory_get_tensor_desc(static_cast<esp_int8::u8_t>(tensor_id), desc), "tensor desc available");

    hls::stream<esp_int8::act_vec_t> act_stream;
    esp_int8::window_generator(desc, act_stream, cfg);

    const int out_h = ceil_div(u16_to_int(cfg.in_h), cfg_stride(cfg));
    const int out_w = ceil_div(u16_to_int(cfg.in_w), cfg_stride(cfg));
    const int k_total = u16_to_int(cfg.in_c) * cfg_kernel(cfg) * cfg_kernel(cfg);
    const int k_tiles = ceil_div(k_total, esp_int8::TK);

    int word_idx = 0;
    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            for (int kt = 0; kt < k_tiles; ++kt) {
                const esp_int8::act_vec_t word = act_stream.read();
                for (int lane = 0; lane < esp_int8::TK; ++lane) {
                    const int k_idx = kt * esp_int8::TK + lane;
                    const std::int8_t got = lane_i8(word, lane);
                    const std::int8_t expected = expected_value(cfg, oh, ow, k_idx);
                    if (got != expected) {
                        if (g_failures < 40) {
                            std::printf("[FAIL] %s word=%d oh=%d ow=%d kt=%d lane=%d got=%d expected=%d\n",
                                        tag,
                                        word_idx,
                                        oh,
                                        ow,
                                        kt,
                                        lane,
                                        static_cast<int>(got),
                                        static_cast<int>(expected));
                        }
                        ++g_failures;
                    }
                }
                ++word_idx;
            }
        }
    }

    if (!act_stream.empty()) {
        if (g_failures < 40) {
            std::printf("[FAIL] %s produced extra activation vectors\n", tag);
        }
        ++g_failures;
    }
}

static void test_1x1_direct_pack() {
    const esp_int8::conv_cfg_t cfg = make_cfg(2, 2, 35, 1, 1, 1);
    expect(fill_tensor_pattern(esp_int8::TID_B2_CAT, 2, 2, 35), "fill 1x1 tensor");
    check_generator_all("WIN_1X1_DIRECT", esp_int8::TID_B2_CAT, cfg);
}

static void test_3x3_stride1_padding() {
    const esp_int8::conv_cfg_t cfg = make_cfg(3, 3, 3, 3, 1, 1);
    expect(fill_tensor_pattern(esp_int8::TID_B2_CAT, 3, 3, 3), "fill 3x3 stride1 tensor");
    check_generator_all("WIN_3X3_STRIDE1_PAD", esp_int8::TID_B2_CAT, cfg);
}

static void test_3x3_stride2() {
    const esp_int8::conv_cfg_t cfg = make_cfg(5, 5, 2, 3, 2, 1);
    expect(fill_tensor_pattern(esp_int8::TID_B2_CAT, 5, 5, 2), "fill 3x3 stride2 tensor");
    check_generator_all("WIN_3X3_STRIDE2", esp_int8::TID_B2_CAT, cfg);
}

static void test_dilation_modes() {
    expect(fill_tensor_pattern(esp_int8::TID_B2_CAT, 40, 40, 1), "fill dilation tensor");

    const int dilations[] = {1, 2, 4, 8, 16};
    for (int i = 0; i < 5; ++i) {
        char tag[32];
        std::snprintf(tag, sizeof(tag), "WIN_DILATION_%d", dilations[i]);
        const esp_int8::conv_cfg_t cfg = make_cfg(40, 40, 1, 3, 1, dilations[i]);
        check_generator_all(tag, esp_int8::TID_B2_CAT, cfg);
    }
}

int main() {
    test_1x1_direct_pack();
    test_3x3_stride1_padding();
    test_3x3_stride2();
    test_dilation_modes();

    if (g_failures != 0) {
        std::printf("win_gen_tb failed: %d failure(s)\n", g_failures);
        return 1;
    }

    std::printf("win_gen_tb passed\n");
    return 0;
}
