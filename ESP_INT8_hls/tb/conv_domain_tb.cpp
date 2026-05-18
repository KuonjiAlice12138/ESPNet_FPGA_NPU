#include "../include/npu_config.hpp"
#include "../include/npu_q.hpp"
#include "../include/npu_uop.hpp"

#include <cstdint>
#include <cstdio>

namespace esp_int8 {
bool on_chip_memory_get_tensor_desc(u8_t tensor_id, tensor_desc_t& desc);
bool on_chip_memory_read_tile(u8_t tensor_id, i32_t h, i32_t w, u16_t c_begin, u8_t valid_c, i8_t tile[TM]);
bool on_chip_memory_write_tile(u8_t tensor_id, u16_t h, u16_t w, u16_t c_begin, u8_t valid_c, const i8_t tile[TM]);

void window_generator(const tensor_desc_t& src_desc,
                      hls::stream<act_vec_t>& act_stream,
                      const conv_cfg_t& cfg);

void systolic_array_core(hls::stream<act_vec_t>& act_stream,
                         hls::stream<wgt_vec_t>& wgt_stream,
                         hls::stream<i32_t>& psum_stream,
                         const conv_cfg_t& cfg);

void post_process_unit(hls::stream<i32_t>& psum_stream,
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

static int to_int(esp_int8::u16_t value) {
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

static std::int8_t to_i8(esp_int8::i8_t value) {
    return static_cast<std::int8_t>(value.to_int());
}

static void set_vec_i8(ap_uint<256>& word, int lane, std::int8_t value) {
    word.range(lane * 8 + 7, lane * 8) = static_cast<std::uint8_t>(value);
}

static std::int8_t input_pattern(int h, int w, int c) {
    const int raw = (h * 19 + w * 13 + c * 7 + 23) % 127;
    return static_cast<std::int8_t>(raw - 63);
}

static std::int8_t weight_pattern(int oc, int k_idx) {
    const int raw = (oc * 11 + k_idx * 5 + 29) % 31;
    return static_cast<std::int8_t>(raw - 15);
}

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

static bool fill_input_tensor(std::uint8_t tensor_id, int h, int w, int c) {
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

static esp_int8::wgt_vec_t pack_weight_row(int oc, int k_total, int kt) {
    esp_int8::wgt_vec_t word = 0;
    for (int lane = 0; lane < esp_int8::TK; ++lane) {
        const int k_idx = kt * esp_int8::TK + lane;
        const std::int8_t value = (k_idx < k_total) ? weight_pattern(oc, k_idx) : 0;
        set_vec_i8(word, lane, value);
    }
    return word;
}

static void push_weights(hls::stream<esp_int8::wgt_vec_t>& wgt_stream,
                         const esp_int8::conv_cfg_t& cfg) {
    const int kernel = cfg_kernel(cfg);
    const int k_total = to_int(cfg.in_c) * kernel * kernel;
    const int k_tiles = ceil_div(k_total, esp_int8::TK);
    const int oc_tiles = ceil_div(to_int(cfg.out_c), esp_int8::TM);

    for (int oc_tile = 0; oc_tile < oc_tiles; ++oc_tile) {
        for (int kt = 0; kt < k_tiles; ++kt) {
            for (int tm = 0; tm < esp_int8::TM; ++tm) {
                const int oc = oc_tile * esp_int8::TM + tm;
                wgt_stream.write((oc < to_int(cfg.out_c)) ? pack_weight_row(oc, k_total, kt) : esp_int8::wgt_vec_t(0));
            }
        }
    }
}

static std::int8_t ref_input_at(const esp_int8::conv_cfg_t& cfg, int oh, int ow, int k_idx) {
    const int kernel = cfg_kernel(cfg);
    const int stride = cfg_stride(cfg);
    const int dilation = cfg_dilation(cfg);
    const int in_h = to_int(cfg.in_h);
    const int in_w = to_int(cfg.in_w);
    const int in_c = to_int(cfg.in_c);
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

static std::int32_t ref_psum(const esp_int8::conv_cfg_t& cfg, int oh, int ow, int oc) {
    const int kernel = cfg_kernel(cfg);
    const int k_total = to_int(cfg.in_c) * kernel * kernel;
    std::int32_t sum = 0;
    for (int k_idx = 0; k_idx < k_total; ++k_idx) {
        sum += static_cast<std::int32_t>(ref_input_at(cfg, oh, ow, k_idx)) *
               static_cast<std::int32_t>(weight_pattern(oc, k_idx));
    }
    return sum;
}

static std::int8_t ref_post(std::int32_t psum,
                            std::int32_t bias,
                            std::int32_t mult,
                            std::uint8_t shift,
                            std::uint8_t act_type) {
    const std::int64_t scaled = (static_cast<std::int64_t>(psum) + bias) * mult;
    return ref_act(ref_clamp_i8(ref_round_shift(scaled, shift)), act_type);
}

static void init_qparams(esp_int8::i32_t bias[32],
                         esp_int8::i32_t mult[32],
                         esp_int8::u8_t shift[32],
                         esp_int8::i32_t aff_mul[32],
                         esp_int8::i32_t aff_bias[32],
                         esp_int8::u8_t aff_shift[32]) {
    for (int i = 0; i < esp_int8::TM; ++i) {
        bias[i] = static_cast<esp_int8::i32_t>(i * 3 - 17);
        mult[i] = static_cast<esp_int8::i32_t>((i % 4) + 1);
        shift[i] = static_cast<esp_int8::u8_t>(i % 3);
        aff_mul[i] = 1;
        aff_bias[i] = 0;
        aff_shift[i] = 0;
    }
}

static esp_int8::conv_cfg_t make_cfg(int in_h, int in_w, int in_c, int out_c, int kernel, int stride, int dilation) {
    esp_int8::conv_cfg_t cfg;
    cfg.in_h = static_cast<esp_int8::u16_t>(in_h);
    cfg.in_w = static_cast<esp_int8::u16_t>(in_w);
    cfg.in_c = static_cast<esp_int8::u16_t>(in_c);
    cfg.out_c = static_cast<esp_int8::u16_t>(out_c);
    cfg.kernel = static_cast<ap_uint<2> >(kernel);
    cfg.stride = static_cast<ap_uint<2> >(stride);
    cfg.dilation = static_cast<ap_uint<5> >(dilation);
    cfg.bias_en = 1;
    return cfg;
}

static void print_perf_model(const char* tag, const esp_int8::conv_cfg_t& cfg) {
    const int kernel = cfg_kernel(cfg);
    const int k_total = to_int(cfg.in_c) * kernel * kernel;
    const int k_tiles = ceil_div(k_total, esp_int8::TK);
    const int out_h = ceil_div(to_int(cfg.in_h), cfg_stride(cfg));
    const int out_w = ceil_div(to_int(cfg.in_w), cfg_stride(cfg));
    const int out_pixels = out_h * out_w;
    const int oc_tiles = ceil_div(to_int(cfg.out_c), esp_int8::TM);
    const long long act_vecs = static_cast<long long>(out_pixels) * k_tiles * oc_tiles;
    const long long wgt_vec_reads = static_cast<long long>(oc_tiles) * k_tiles * esp_int8::TM;
    const long long useful_macs = static_cast<long long>(out_pixels) * to_int(cfg.out_c) * k_total;
    const long long ideal_sa_cycles = act_vecs;
    const long long psum_write_cycles = static_cast<long long>(out_pixels) * to_int(cfg.out_c);
    const long long sa_core_lower_bound = wgt_vec_reads + ideal_sa_cycles + psum_write_cycles;

    std::printf("[PERF_MODEL] %s out_pixels=%d k_total=%d k_tiles=%d out_c=%d "
                "act_vecs=%lld weight_vec_reads=%lld useful_macs=%lld "
                "ideal_32x32_cycles>=%lld sa_core_cycles_lb>=%lld\n",
                tag,
                out_pixels,
                k_total,
                k_tiles,
                to_int(cfg.out_c),
                act_vecs,
                wgt_vec_reads,
                useful_macs,
                ideal_sa_cycles,
                sa_core_lower_bound);
}

static void run_conv_case(const char* tag,
                          const esp_int8::conv_cfg_t& cfg,
                          std::uint8_t src_tensor,
                          std::uint8_t dst_tensor,
                          std::uint8_t act_type) {
    if (!fill_input_tensor(src_tensor, to_int(cfg.in_h), to_int(cfg.in_w), to_int(cfg.in_c))) {
        std::printf("[FAIL] %s failed to fill input tensor\n", tag);
        ++g_failures;
        return;
    }

    esp_int8::tensor_desc_t src_desc;
    if (!esp_int8::on_chip_memory_get_tensor_desc(static_cast<esp_int8::u8_t>(src_tensor), src_desc)) {
        std::printf("[FAIL] %s missing source tensor desc\n", tag);
        ++g_failures;
        return;
    }

    hls::stream<esp_int8::act_vec_t> act_stream;
    hls::stream<esp_int8::wgt_vec_t> wgt_stream;
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
    init_qparams(bias, mult, shift, aff_mul, aff_bias, aff_shift);

    esp_int8::window_generator(src_desc, act_stream, cfg);
    push_weights(wgt_stream, cfg);
    esp_int8::systolic_array_core(act_stream, wgt_stream, psum_stream, cfg);

    const int out_h = ceil_div(to_int(cfg.in_h), cfg_stride(cfg));
    const int out_w = ceil_div(to_int(cfg.in_w), cfg_stride(cfg));
    const int out_c = to_int(cfg.out_c);

    esp_int8::post_cfg_t post_cfg;
    post_cfg.mode = esp_int8::POST_CONV;
    post_cfg.act_type = static_cast<ap_uint<2> >(act_type);
    post_cfg.requant_bypass = 0;
    post_cfg.valid_tm = static_cast<ap_uint<6> >(out_c);

    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            esp_int8::post_process_unit(psum_stream,
                                        add_a_stream,
                                        add_b_stream,
                                        out_stream,
                                        post_cfg,
                                        bias,
                                        mult,
                                        shift,
                                        aff_mul,
                                        aff_bias,
                                        aff_shift);

            esp_int8::i8_t out_tile[esp_int8::TM];
            for (int lane = 0; lane < esp_int8::TM; ++lane) {
                if (lane < out_c) {
                    out_tile[lane] = out_stream.read();
                } else {
                    out_tile[lane] = 0;
                }
            }

            if (!esp_int8::on_chip_memory_write_tile(static_cast<esp_int8::u8_t>(dst_tensor),
                                                     static_cast<esp_int8::u16_t>(oh),
                                                     static_cast<esp_int8::u16_t>(ow),
                                                     0,
                                                     static_cast<esp_int8::u8_t>(out_c),
                                                     out_tile)) {
                std::printf("[FAIL] %s failed to write output tile oh=%d ow=%d\n", tag, oh, ow);
                ++g_failures;
            }
        }
    }

    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            esp_int8::i8_t got_tile[esp_int8::TM];
            if (!esp_int8::on_chip_memory_read_tile(static_cast<esp_int8::u8_t>(dst_tensor),
                                                    static_cast<esp_int8::i32_t>(oh),
                                                    static_cast<esp_int8::i32_t>(ow),
                                                    0,
                                                    static_cast<esp_int8::u8_t>(out_c),
                                                    got_tile)) {
                std::printf("[FAIL] %s failed to read output tile oh=%d ow=%d\n", tag, oh, ow);
                ++g_failures;
                continue;
            }

            for (int oc = 0; oc < out_c; ++oc) {
                const std::int32_t psum = ref_psum(cfg, oh, ow, oc);
                const std::int8_t expected = ref_post(psum,
                                                      bias[oc].to_int(),
                                                      mult[oc].to_int(),
                                                      static_cast<std::uint8_t>(shift[oc].to_uint()),
                                                      act_type);
                const std::int8_t got = to_i8(got_tile[oc]);
                if (got != expected) {
                    if (g_failures < 40) {
                        std::printf("[FAIL] %s oh=%d ow=%d oc=%d got=%d expected=%d psum=%d\n",
                                    tag,
                                    oh,
                                    ow,
                                    oc,
                                    static_cast<int>(got),
                                    static_cast<int>(expected),
                                    psum);
                    }
                    ++g_failures;
                }
            }
        }
    }

    if (!act_stream.empty() || !wgt_stream.empty() || !psum_stream.empty() || !out_stream.empty()) {
        std::printf("[FAIL] %s left non-empty stream(s)\n", tag);
        ++g_failures;
    }

    print_perf_model(tag, cfg);
}

int main() {
    run_conv_case("CONV_1X1_CIN_TAIL", make_cfg(2, 3, 35, 7, 1, 1, 1),
                  esp_int8::TID_B2_CAT, esp_int8::TID_B1_ACT, esp_int8::ACT_NONE);
    run_conv_case("CONV_3X3_STRIDE1_RELU", make_cfg(4, 5, 3, 5, 3, 1, 1),
                  esp_int8::TID_B2_CAT, esp_int8::TID_B1_ACT, esp_int8::ACT_RELU);
    run_conv_case("CONV_3X3_STRIDE2", make_cfg(5, 5, 2, 3, 3, 2, 1),
                  esp_int8::TID_B2_CAT, esp_int8::TID_B1_ACT, esp_int8::ACT_NONE);
    run_conv_case("CONV_3X3_DILATION16", make_cfg(35, 35, 1, 2, 3, 1, 16),
                  esp_int8::TID_B2_CAT, esp_int8::TID_B1_ACT, esp_int8::ACT_NONE);

    if (g_failures != 0) {
        std::printf("conv_domain_tb failed: %d failure(s)\n", g_failures);
        return 1;
    }

    std::printf("conv_domain_tb passed\n");
    return 0;
}
