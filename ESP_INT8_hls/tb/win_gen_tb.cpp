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
    row_base = 0;
    if (row_valid) {
        u32_t row_offset = static_cast<u32_t>(h);
        row_offset *= static_cast<u32_t>(desc.w);
        row_offset *= static_cast<u32_t>(desc.reserved0);
        row_base = desc.base_offset + row_offset + static_cast<u32_t>(desc.reserved1);
    }
    return true;
}

bool on_chip_memory_read_aligned_tensor_word(const tensor_desc_t& desc,
                                             u32_t byte_offset,
                                             act_vec_t& packed) {
    packed = 0;
    for (int lane = 0; lane < TK; ++lane) {
        const unsigned absolute = byte_offset.to_uint() + static_cast<unsigned>(lane);
        if (absolute < desc.base_offset.to_uint()) {
            continue;
        }
        const unsigned relative = absolute - desc.base_offset.to_uint();
        const unsigned phys_c = desc.reserved0.to_uint();
        const unsigned pixel = relative / phys_c;
        const unsigned physical_channel = relative % phys_c;
        const unsigned channel_offset = desc.reserved1.to_uint();
        if (pixel >= desc.h.to_uint() * desc.w.to_uint() ||
            physical_channel < channel_offset ||
            physical_channel >= channel_offset + desc.c.to_uint()) {
            continue;
        }
        const int h = static_cast<int>(pixel / desc.w.to_uint());
        const int w = static_cast<int>(pixel % desc.w.to_uint());
        const int c = static_cast<int>(physical_channel - channel_offset);
        set_lane(packed, lane, mock_value(h, w, c));
    }
    return true;
}

bool on_chip_memory_read_packed_tile(const tensor_desc_t& desc,
                                     i32_t h,
                                     i32_t w,
                                     u16_t c_begin,
                                     u8_t valid_c,
                                     act_vec_t& packed) {
    packed = 0;
    if (h < 0 || h >= static_cast<i32_t>(desc.h) ||
        w < 0 || w >= static_cast<i32_t>(desc.w)) {
        return true;
    }
    const int count = static_cast<int>(valid_c.to_uint());
    for (int lane = 0; lane < TK; ++lane) {
        if (lane < count && c_begin.to_uint() + lane < desc.c.to_uint()) {
            set_lane(packed,
                     lane,
                     mock_value(h.to_int(), w.to_int(), c_begin.to_uint() + lane));
        }
    }
    return true;
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

static void compile_loader_runs(esp_int8::window_sched_desc_t& sched,
                                int issue,
                                unsigned update_mask,
                                int count_word,
                                int first_run_word) {
    const int pixel0 = issue * 2;
    const int base_column =
        pixel0 * sched.stride.to_uint() - sched.padding.to_uint();
    int deltas[6];
    int delta_count = 0;
    for (int request = 0; request < 6; ++request) {
        if ((update_mask & (1U << request)) == 0U) {
            continue;
        }
        const bool second = request >= 3;
        const int pixel = pixel0 + (second ? 1 : 0);
        const int kw = second ? request - 3 : request;
        const int column =
            pixel * sched.stride.to_uint() -
            sched.padding.to_uint() +
            kw * sched.dilation.to_uint();
        const int delta = column - base_column;
        bool duplicate = false;
        for (int index = 0; index < delta_count; ++index) {
            duplicate |= deltas[index] == delta;
        }
        if (!duplicate) {
            deltas[delta_count++] = delta;
        }
    }
    for (int index = 1; index < delta_count; ++index) {
        const int value = deltas[index];
        int insert = index;
        while (insert > 0 && deltas[insert - 1] > value) {
            deltas[insert] = deltas[insert - 1];
            --insert;
        }
        deltas[insert] = value;
    }

    int run_count = 0;
    int index = 0;
    while (index < delta_count) {
        const int start = deltas[index];
        int count = 1;
        while (index + count < delta_count &&
               deltas[index + count] == start + count) {
            ++count;
        }
        if (run_count >= esp_int8::WINDOW_LOADER_RUN_MAX) {
            std::printf("[FAIL] loader run compiler exceeded max runs\n");
            ++g_failures;
            return;
        }
        sched.reserved[first_run_word + run_count] =
            static_cast<esp_int8::u16_t>(
                ((count & 0xFF) << 8) | (start & 0xFF));
        ++run_count;
        index += count;
    }
    sched.reserved[count_word] =
        static_cast<esp_int8::u16_t>(run_count);
}

static int compile_loader_phase_split(
    const esp_int8::window_sched_desc_t& sched,
    int issue,
    unsigned update_mask) {
    const int pixel0 = issue * 2;
    const int base0 =
        pixel0 * sched.stride.to_uint() - sched.padding.to_uint();
    const int base1 =
        (pixel0 + 1) * sched.stride.to_uint() - sched.padding.to_uint();
    int phase0_columns[3];
    int phase1_columns[3];
    int phase0_count = 0;
    int phase1_count = 0;
    for (int kw = 0; kw < 3; ++kw) {
        if ((update_mask & (1U << kw)) != 0U) {
            phase0_columns[phase0_count++] =
                base0 + kw * sched.dilation.to_uint();
        }
        if ((update_mask & (1U << (kw + 3))) != 0U) {
            phase1_columns[phase1_count++] =
                base1 + kw * sched.dilation.to_uint();
        }
    }
    const int slot_mask = sched.cache_col_slots.to_uint() - 1;
    for (int phase1 = 0; phase1 < phase1_count; ++phase1) {
        for (int kw = 0; kw < 3; ++kw) {
            const int window0_column =
                base0 + kw * sched.dilation.to_uint();
            if (phase1_columns[phase1] != window0_column &&
                (phase1_columns[phase1] & slot_mask) ==
                    (window0_column & slot_mask)) {
                return phase0_count;
            }
        }
    }
    return 0;
}

static void configure_loader_contract(esp_int8::window_sched_desc_t& sched) {
    for (int index = 0;
         index < esp_int8::WINDOW_LOADER_RESERVED_WORDS;
         ++index) {
        sched.reserved[index] = 0;
    }
    if (sched.kernel.to_uint() != 3U) {
        sched.loader_class = static_cast<esp_int8::u8_t>(
            static_cast<unsigned>(esp_int8::WIN_LOADER_DIRECT_1X1));
        return;
    }

    const bool paired =
        (sched.flags.to_uint() &
         static_cast<unsigned>(esp_int8::WINDOW_SCHED_FLAG_PIXEL_PARALLEL_2)) != 0U;
    const int request_cols = paired ? 6 : 3;
    const int issue_count =
        paired ? (sched.out_w.to_uint() + 1) / 2 : sched.out_w.to_uint();
    const int slots = sched.cache_col_slots.to_uint();
    int tags[esp_int8::WINGEN_NARROW_CACHE_COL_SLOTS];
    int miss_counts[esp_int8::MAX_FM_W];
    unsigned update_masks[esp_int8::MAX_FM_W];
    for (int slot = 0; slot < esp_int8::WINGEN_NARROW_CACHE_COL_SLOTS; ++slot) {
        tags[slot] = -32768;
    }

    for (int issue = 0; issue < issue_count; ++issue) {
        const int pixel0 = issue * (paired ? 2 : 1);
        int misses = 0;
        unsigned mask = 0;
        for (int request = 0; request < request_cols; ++request) {
            const bool second = request >= 3;
            const int pixel = pixel0 + (second ? 1 : 0);
            if (second && pixel >= sched.out_w.to_uint()) {
                continue;
            }
            const int kw = second ? request - 3 : request;
            const int column =
                pixel * sched.stride.to_uint() -
                sched.padding.to_uint() +
                kw * sched.dilation.to_uint();
            const int slot = column & (slots - 1);
            if (tags[slot] != column) {
                tags[slot] = column;
                ++misses;
                mask |= 1U << request;
            }
        }
        miss_counts[issue] = misses;
        update_masks[issue] = mask;
    }

    const int steady_new_cols = miss_counts[issue_count - 1];
    int warmup_issues = 0;
    for (int candidate = 0; candidate < issue_count; ++candidate) {
        bool steady = true;
        for (int issue = candidate; issue < issue_count; ++issue) {
            if (miss_counts[issue] != steady_new_cols) {
                steady = false;
            }
        }
        if (steady) {
            warmup_issues = candidate;
            break;
        }
    }

    const unsigned loader_class =
        sched.in_c.to_uint() <= 25U
            ? static_cast<unsigned>(esp_int8::WIN_LOADER_3X3_NARROW)
            : static_cast<unsigned>(esp_int8::WIN_LOADER_3X3_WIDE);
    sched.loader_class =
        static_cast<esp_int8::u8_t>(loader_class);
    sched.loader_request_cols = static_cast<esp_int8::u8_t>(request_cols);
    sched.loader_warmup_issues = static_cast<esp_int8::u8_t>(warmup_issues);
    sched.loader_warmup_new_cols = static_cast<esp_int8::u8_t>(
        warmup_issues > 0 ? miss_counts[0] : steady_new_cols);
    sched.loader_steady_new_cols =
        static_cast<esp_int8::u8_t>(steady_new_cols);
    sched.loader_words_per_col =
        static_cast<esp_int8::u8_t>(3U * sched.cache_chunks.to_uint());
    sched.loader_warmup_mask = static_cast<esp_int8::u8_t>(
        warmup_issues > 0 ? update_masks[0]
                          : update_masks[issue_count - 1]);
    sched.loader_steady_mask =
        static_cast<esp_int8::u8_t>(update_masks[issue_count - 1]);
    if (paired &&
        loader_class ==
            static_cast<unsigned>(esp_int8::WIN_LOADER_3X3_NARROW)) {
        compile_loader_runs(
            sched,
            0,
            sched.loader_warmup_mask.to_uint(),
            esp_int8::WINDOW_LOADER_WARMUP_COUNT_WORD,
            esp_int8::WINDOW_LOADER_WARMUP_RUN_WORD);
        compile_loader_runs(
            sched,
            warmup_issues > 0 ? warmup_issues : 1,
            sched.loader_steady_mask.to_uint(),
            esp_int8::WINDOW_LOADER_STEADY_COUNT_WORD,
            esp_int8::WINDOW_LOADER_STEADY_RUN_WORD);
        const int warmup_split = compile_loader_phase_split(
            sched, 0, sched.loader_warmup_mask.to_uint());
        const int steady_split = compile_loader_phase_split(
            sched,
            warmup_issues > 0 ? warmup_issues : 1,
            sched.loader_steady_mask.to_uint());
        sched.reserved[esp_int8::WINDOW_LOADER_PHASE_SPLIT_WORD] =
            static_cast<esp_int8::u16_t>(
                ((steady_split & 0xFF) << 8) |
                (warmup_split & 0xFF));
    }
}

static void check_case(const char* tag,
                       int in_h,
                       int in_w,
                       int in_c,
                       int kernel,
                       int stride,
                       int dilation,
                       unsigned mode,
                       bool paired,
                       bool row_reuse = false) {
    esp_int8::tensor_desc_t desc = {};
    desc.h = in_h;
    desc.w = in_w;
    desc.c = in_c;
    desc.base_offset = kernel == 3 && !row_reuse ? 13 : 0;
    desc.reserved0 = kernel == 3 && !row_reuse ? in_c + 5 : in_c;
    desc.reserved1 = kernel == 3 && !row_reuse ? 3 : 0;

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
    configure_loader_contract(sched);
    if (row_reuse) {
        const unsigned packed_words =
            static_cast<unsigned>(in_w * in_c / esp_int8::AXI_WORD_BYTES);
        const unsigned reuse_mode =
            stride == 1
                ? static_cast<unsigned>(
                      esp_int8::WINDOW_ROW_REUSE_STRIDE1_KEEP2)
                : static_cast<unsigned>(
                      esp_int8::WINDOW_ROW_REUSE_STRIDE2_KEEP1);
        sched.reserved[esp_int8::WINDOW_LOADER_ROW_REUSE_WORD] =
            static_cast<esp_int8::u16_t>(
                reuse_mode |
                (packed_words << esp_int8::WINDOW_ROW_REUSE_WORDS_SHIFT));
    }

    hls::stream<esp_int8::act_vec_t> stream0;
    hls::stream<esp_int8::act_vec_t> stream1;
    const int out_h = (in_h + stride - 1) / stride;
    for (int oh = 0; oh < out_h; ++oh) {
        const int issue_count = paired ? (sched.out_w.to_uint() + 1) / 2 : sched.out_w.to_uint();
        esp_int8::scheduled_window_generator_row(
            desc, sched, stream0, stream1, static_cast<esp_int8::u16_t>(oh));
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
    check_case("C3-row-reuse", 5, 1024, 3, 3, 2, 1, esp_int8::WIN_MODE_3X3_STAGED_C3, true, true);
    check_case("C12-paired-d1", 4, 12, 12, 3, 1, 1, esp_int8::WIN_MODE_3X3_STAGED_C12, true);
    check_case("C12-row-reuse", 4, 256, 12, 3, 1, 1, esp_int8::WIN_MODE_3X3_STAGED_C12, true, true);
    check_case("C12-single", 6, 8, 12, 3, 1, 2, esp_int8::WIN_MODE_3X3_STAGED_C12, false);
    check_case("C12-paired", 6, 8, 12, 3, 1, 2, esp_int8::WIN_MODE_3X3_STAGED_C12, true);
    check_case("C12-paired-d16", 3, 40, 12, 3, 1, 16, esp_int8::WIN_MODE_3X3_STAGED_C12, true);
    check_case("C19-paired", 6, 9, 19, 3, 1, 1, esp_int8::WIN_MODE_3X3_STAGED_C19, true);
    check_case("C19-paired-s2", 6, 11, 19, 3, 2, 1, esp_int8::WIN_MODE_3X3_STAGED_C19, true);
    check_case("C25-paired", 5, 7, 25, 3, 1, 2, esp_int8::WIN_MODE_3X3_STAGED_C25, true);
    check_case("C131-wide", 4, 5, 131, 3, 1, 1, esp_int8::WIN_MODE_3X3_STAGED_C131, false);
    check_case("C64-1x1-single", 4, 6, 64, 1, 1, 1, esp_int8::WIN_MODE_1X1_ALIGNED, false);
    check_case("C64-1x1-paired", 4, 6, 64, 1, 1, 1, esp_int8::WIN_MODE_1X1_ALIGNED, true);
    if (g_failures != 0) {
        std::printf("win_gen_tb failed: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("win_gen_tb passed\n");
    return 0;
}
