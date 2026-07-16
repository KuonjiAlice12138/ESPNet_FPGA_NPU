#include "../include/npu_config.hpp"
#include "../include/npu_schedule.hpp"
#include "../include/npu_types.hpp"

#include <cstdint>
#include <cstdio>

namespace esp_int8 {

static std::int8_t mock_value(int h, int w, int c) {
    return static_cast<std::int8_t>(((h * 17 + w * 11 + c * 5 + 63) % 127) - 63);
}

static void set_lane(act_vec_t& word, int lane, std::int8_t value) {
    word.range(lane * 8 + 7, lane * 8) = static_cast<std::uint8_t>(value);
}

bool on_chip_memory_prepare_row_base(const tensor_desc_t& desc,
                                     i32_t h,
                                     u32_t& row_base,
                                     bool& row_valid) {
    row_valid = h >= 0 && h < static_cast<i32_t>(desc.h);
    row_base = row_valid ? static_cast<u32_t>(h * desc.w * desc.reserved0) : u32_t(0);
    return true;
}

bool on_chip_memory_read_packed_tile_from_row(const tensor_desc_t& desc,
                                              u32_t row_base,
                                              bool row_valid,
                                              i32_t w,
                                              u16_t c_begin,
                                              u8_t valid_c,
                                              act_vec_t& packed) {
    packed = 0;
    if (!row_valid || w < 0 || w >= static_cast<i32_t>(desc.w)) {
        return true;
    }
    const int h = static_cast<int>((row_base / (desc.w * desc.reserved0)).to_uint());
    const int count = static_cast<int>(valid_c.to_uint());
    for (int lane = 0; lane < TK; ++lane) {
        if (lane < count && c_begin.to_uint() + lane < desc.c.to_uint()) {
            set_lane(packed, lane, mock_value(h, w.to_int(), c_begin.to_uint() + lane));
        }
    }
    return true;
}

bool on_chip_memory_read_packed_tile(const tensor_desc_t& desc,
                                     i32_t h,
                                     i32_t w,
                                     u16_t c_begin,
                                     u8_t valid_c,
                                     act_vec_t& packed) {
    u32_t row_base = 0;
    bool row_valid = false;
    on_chip_memory_prepare_row_base(desc, h, row_base, row_valid);
    return on_chip_memory_read_packed_tile_from_row(
        desc, row_base, row_valid, w, c_begin, valid_c, packed);
}

bool on_chip_memory_read_aligned_full_tile(const tensor_desc_t& desc,
                                           i32_t h,
                                           i32_t w,
                                           u16_t c_begin,
                                           act_vec_t& packed) {
    return on_chip_memory_read_packed_tile(
        desc, h, w, c_begin, static_cast<u8_t>(TK), packed);
}

void scheduled_window_generator_row(const tensor_desc_t& src_desc,
                                    const conv_exec_desc_t& conv_desc,
                                    const window_sched_desc_t& sched,
                                    hls::stream<act_vec_t>& act_stream0,
                                    hls::stream<act_vec_t>& act_stream1,
                                    u16_t out_row);

}  // namespace esp_int8

static int g_failures = 0;

static std::int8_t lane_i8(const esp_int8::act_vec_t& word, int lane) {
    esp_int8::i8_t value;
    value.range(7, 0) = word.range(lane * 8 + 7, lane * 8);
    return static_cast<std::int8_t>(value.to_int());
}

static std::int8_t expected(const esp_int8::window_sched_desc_t& sched,
                            int in_h,
                            int in_w,
                            int out_row,
                            int out_col,
                            int k_index) {
    const int kernel = sched.kernel.to_uint();
    const int in_c = sched.in_c.to_uint();
    if (k_index >= kernel * kernel * in_c) {
        return 0;
    }
    const int spatial = k_index / in_c;
    const int channel = k_index % in_c;
    const int kh = kernel == 1 ? 0 : spatial / 3;
    const int kw = kernel == 1 ? 0 : spatial % 3;
    const int h = out_row * sched.stride.to_uint() + kh * sched.dilation.to_uint() - sched.padding.to_uint();
    const int w = out_col * sched.stride.to_uint() + kw * sched.dilation.to_uint() - sched.padding.to_uint();
    if (h < 0 || h >= in_h || w < 0 || w >= in_w) {
        return 0;
    }
    return esp_int8::mock_value(h, w, channel);
}

static void check_case(const char* tag,
                       int in_h,
                       int in_w,
                       int in_c,
                       int kernel,
                       int stride,
                       int dilation,
                       unsigned mode,
                       bool paired) {
    esp_int8::tensor_desc_t desc = {};
    desc.h = in_h;
    desc.w = in_w;
    desc.c = in_c;
    desc.reserved0 = in_c;

    esp_int8::window_sched_desc_t sched = {};
    sched.mode = mode;
    sched.kernel = kernel;
    sched.stride = stride;
    sched.dilation = dilation;
    sched.padding = kernel == 3 ? dilation : 0;
    sched.cache_chunks = kernel == 3 ? (in_c + esp_int8::TK - 1) / esp_int8::TK : 0;
    sched.cache_col_slots = kernel == 3 && in_c <= 25 ? 4 * dilation : 4;
    unsigned sched_flags = paired
                               ? static_cast<unsigned>(esp_int8::WINDOW_SCHED_FLAG_PIXEL_PARALLEL_2)
                               : 0U;
    sched.in_c = in_c;
    sched.out_w = (in_w + stride - 1) / stride;
    if (paired && (sched.out_w.to_uint() & 1U) != 0U) {
        sched_flags |= static_cast<unsigned>(esp_int8::WINDOW_SCHED_FLAG_ODD_TAIL);
    }
    sched.flags = static_cast<esp_int8::u8_t>(sched_flags);
    sched.k_tiles = (kernel * kernel * in_c + esp_int8::TK - 1) / esp_int8::TK;

    esp_int8::conv_exec_desc_t conv = {};
    hls::stream<esp_int8::act_vec_t> stream0;
    hls::stream<esp_int8::act_vec_t> stream1;
    const int out_h = (in_h + stride - 1) / stride;
    for (int oh = 0; oh < out_h; ++oh) {
        esp_int8::scheduled_window_generator_row(
            desc, conv, sched, stream0, stream1, static_cast<esp_int8::u16_t>(oh));
        const int issue_count = paired ? (sched.out_w.to_uint() + 1) / 2 : sched.out_w.to_uint();
        for (int issue = 0; issue < issue_count; ++issue) {
            const int pixel0 = paired ? issue * 2 : issue;
            for (int kt = 0; kt < sched.k_tiles.to_uint(); ++kt) {
                const esp_int8::act_vec_t word0 = stream0.read();
                const esp_int8::act_vec_t word1 = paired ? stream1.read() : esp_int8::act_vec_t(0);
                for (int lane = 0; lane < esp_int8::TK; ++lane) {
                    const int k = kt * esp_int8::TK + lane;
                    const std::int8_t exp0 = expected(sched, in_h, in_w, oh, pixel0, k);
                    if (lane_i8(word0, lane) != exp0) {
                        if (g_failures < 20) {
                            std::printf("[FAIL] %s oh=%d ow=%d kt=%d lane=%d got=%d exp=%d\n",
                                        tag, oh, pixel0, kt, lane,
                                        static_cast<int>(lane_i8(word0, lane)), static_cast<int>(exp0));
                        }
                        ++g_failures;
                    }
                    if (paired) {
                        const int pixel1 = pixel0 + 1;
                        const std::int8_t exp1 = pixel1 < sched.out_w.to_uint()
                                                     ? expected(sched, in_h, in_w, oh, pixel1, k)
                                                     : 0;
                        if (lane_i8(word1, lane) != exp1) {
                            if (g_failures < 20) {
                                std::printf("[FAIL] %s-p1 oh=%d ow=%d kt=%d lane=%d got=%d exp=%d\n",
                                            tag, oh, pixel1, kt, lane,
                                            static_cast<int>(lane_i8(word1, lane)), static_cast<int>(exp1));
                            }
                            ++g_failures;
                        }
                    }
                }
            }
        }
    }
    if (!stream0.empty() || !stream1.empty()) {
        std::printf("[FAIL] %s left unread stream data\n", tag);
        ++g_failures;
    }
}

int main() {
    check_case("C3-single", 7, 9, 3, 3, 2, 1, esp_int8::WIN_MODE_3X3_STAGED_C3, false);
    check_case("C3-paired", 7, 9, 3, 3, 2, 1, esp_int8::WIN_MODE_3X3_STAGED_C3, true);
    check_case("C12-single", 6, 8, 12, 3, 1, 2, esp_int8::WIN_MODE_3X3_STAGED_C12, false);
    check_case("C12-paired", 6, 8, 12, 3, 1, 2, esp_int8::WIN_MODE_3X3_STAGED_C12, true);
    check_case("C64-1x1-single", 4, 6, 64, 1, 1, 1, esp_int8::WIN_MODE_1X1_ALIGNED, false);
    check_case("C64-1x1-paired", 4, 6, 64, 1, 1, 1, esp_int8::WIN_MODE_1X1_ALIGNED, true);
    if (g_failures != 0) {
        std::printf("win_gen_tb failed: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("win_gen_tb passed\n");
    return 0;
}
