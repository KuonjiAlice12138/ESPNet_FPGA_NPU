#include "../include/npu_q.hpp"
#include "../include/npu_config.hpp"

namespace esp_int8 {

static u8_t effective_valid_tm(const post_cfg_t& cfg) {
#pragma HLS INLINE
    const u8_t n = cfg.valid_tm;
    if (n == 0 || n > TM) {
        return TM;
    }
    return n;
}

static i8_t add_post_i8(i8_t a, i8_t b, const post_cfg_t& cfg, i32_t mult, u8_t shift) {
#pragma HLS INLINE
    const i32_t sum = static_cast<i32_t>(a) + static_cast<i32_t>(b);
    if (cfg.requant_bypass != 0) {
        return apply_act(clamp_i8(sum), cfg.act_type);
    }

    const i64_t scaled = static_cast<i64_t>(sum) * static_cast<i64_t>(mult);
    return apply_act(clamp_i8(round_shift(scaled, shift)), cfg.act_type);
}

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
    const u8_t aff_shift[32]) {
#pragma HLS INLINE off
    const u8_t lanes = effective_valid_tm(cfg);

    for (int lane = 0; lane < TM; ++lane) {
#pragma HLS PIPELINE II=1
        if (lane >= static_cast<int>(lanes.to_uint())) {
            break;
        }

        i8_t out = 0;
        if (cfg.mode.to_uint() == static_cast<unsigned>(POST_CONV)) {
            const i32_t psum = psum_stream.read();
            out = requant_i32_to_i8(psum, bias[lane], mult[lane], shift[lane], cfg.act_type);
        } else if (cfg.mode.to_uint() == static_cast<unsigned>(POST_ADD)) {
            const i8_t a = add_a_stream.read();
            const i8_t b = add_b_stream.read();
            out = add_post_i8(a, b, cfg, mult[0], shift[0]);
        } else if (cfg.mode.to_uint() == static_cast<unsigned>(POST_AFFINE)) {
            const i8_t x = add_a_stream.read();
            out = affine_i8_to_i8(x, aff_mul[lane], aff_bias[lane], aff_shift[lane], cfg.act_type);
        }
        out_stream.write(out);
    }
}

}  // namespace esp_int8
