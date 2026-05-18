#include "../include/npu_config.hpp"
#include "../include/npu_uop.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <vector>

void espnet_encoder_int8_core(const esp_int8::axi_vec_t* gmem_frame_in,
                              esp_int8::axi_vec_t* gmem_frame_out,
                              const esp_int8::axi_vec_t* gmem_param,
                              std::uint32_t mode,
                              std::uint32_t uop_count,
                              volatile std::uint32_t& dbg_status,
                              volatile std::uint32_t& dbg_heartbeat,
                              volatile std::uint32_t& dbg_act_words,
                              volatile std::uint32_t& dbg_wgt_words,
                              volatile std::uint32_t& dbg_psum_words,
                              volatile std::uint32_t& dbg_out_words,
                              volatile std::uint32_t& dbg_hw_version,
                              volatile std::uint32_t& prof_uop_count,
                              volatile std::uint32_t& prof_conv_count,
                              volatile std::uint32_t& prof_win_read_ops,
                              volatile std::uint32_t& prof_win_words,
                              volatile std::uint32_t& prof_wgt_words,
                              volatile std::uint32_t& prof_sa_mac_steps,
                              volatile std::uint32_t& prof_psum_words,
                              volatile std::uint32_t& prof_out_tiles,
                              volatile std::uint32_t& prof_out_rmw_ops,
                              volatile std::uint32_t& prof_model_cycles,
                              volatile std::uint32_t& prof2_win_saved_reads,
                              volatile std::uint32_t& prof2_win_actual_reads,
                              volatile std::uint32_t& prof2_out_direct_words,
                              volatile std::uint32_t& prof2_out_rmw_reads,
                              volatile std::uint32_t& prof3_wgt_cycles,
                              volatile std::uint32_t& prof3_win_cycles,
                              volatile std::uint32_t& prof3_sa_cycles,
                              volatile std::uint32_t& prof3_post_cycles,
                              volatile std::uint32_t& prof3_write_cycles,
                              volatile std::uint32_t& prof3_row_region_cycles);

static bool read_binary(const char* path, std::vector<std::uint8_t>& bytes) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::printf("[FAIL] failed to open %s\n", path);
        return false;
    }
    bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return true;
}

static bool write_binary(const char* path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::printf("[FAIL] failed to write %s\n", path);
        return false;
    }
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(out);
}

static void set_byte(esp_int8::axi_vec_t& word, int lane, std::uint8_t value) {
    word.range(lane * 8 + 7, lane * 8) = value;
}

static std::uint8_t get_byte(const esp_int8::axi_vec_t& word, int lane) {
    return static_cast<std::uint8_t>(word.range(lane * 8 + 7, lane * 8).to_uint());
}

static void pack_bytes(const std::vector<std::uint8_t>& bytes,
                       esp_int8::axi_vec_t* words,
                       std::size_t word_count) {
    for (std::size_t i = 0; i < word_count; ++i) {
        words[i] = 0;
    }
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        set_byte(words[i / esp_int8::AXI_WORD_BYTES],
                 static_cast<int>(i % esp_int8::AXI_WORD_BYTES),
                 bytes[i]);
    }
}

static std::int8_t as_i8(std::uint8_t value) {
    return static_cast<std::int8_t>(value);
}

static int argmax2(std::int8_t c0, std::int8_t c1) {
    return (c1 > c0) ? 1 : 0;
}

int main() {
    static constexpr int MAX_ALLOWED_MASK_MISMATCHES = 128;
    const char* artifact_dir = "D:/ESP_INT8/hw_artifacts/hw_constrained_qat_3ep_single";
    const char* param_path = "D:/ESP_INT8/hw_artifacts/hw_constrained_qat_3ep_single/param_blob.bin";
    const char* input_path = "D:/ESP_INT8/hw_artifacts/hw_constrained_qat_3ep_single/input_q.bin";
    const char* golden_path = "D:/ESP_INT8/hw_artifacts/hw_constrained_qat_3ep_single/golden_output_q.bin";
    const char* hls_output_path = "hls_output_q.bin";

    std::vector<std::uint8_t> param_bytes;
    std::vector<std::uint8_t> input_bytes;
    std::vector<std::uint8_t> golden_bytes;
    if (!read_binary(param_path, param_bytes) ||
        !read_binary(input_path, input_bytes) ||
        !read_binary(golden_path, golden_bytes)) {
        return 1;
    }
    if (input_bytes.size() != static_cast<std::size_t>(esp_int8::INPUT_FRAME_BYTES) ||
        golden_bytes.size() != static_cast<std::size_t>(esp_int8::OUTPUT_FRAME_BYTES)) {
        std::printf("[FAIL] unexpected artifact sizes in %s: input=%zu golden=%zu\n",
                    artifact_dir,
                    input_bytes.size(),
                    golden_bytes.size());
        return 1;
    }

    const std::size_t param_words =
        (param_bytes.size() + esp_int8::AXI_WORD_BYTES - 1U) / esp_int8::AXI_WORD_BYTES;
    static esp_int8::axi_vec_t frame_in[esp_int8::INPUT_FRAME_AXI_WORDS];
    static esp_int8::axi_vec_t frame_out[esp_int8::OUTPUT_FRAME_AXI_WORDS];
    static esp_int8::axi_vec_t param[4096];
    volatile std::uint32_t dbg_status = 0;
    volatile std::uint32_t dbg_heartbeat = 0;
    volatile std::uint32_t dbg_act_words = 0;
    volatile std::uint32_t dbg_wgt_words = 0;
    volatile std::uint32_t dbg_psum_words = 0;
    volatile std::uint32_t dbg_out_words = 0;
    volatile std::uint32_t dbg_hw_version = 0;
    volatile std::uint32_t prof_uop_count = 0;
    volatile std::uint32_t prof_conv_count = 0;
    volatile std::uint32_t prof_win_read_ops = 0;
    volatile std::uint32_t prof_win_words = 0;
    volatile std::uint32_t prof_wgt_words = 0;
    volatile std::uint32_t prof_sa_mac_steps = 0;
    volatile std::uint32_t prof_psum_words = 0;
    volatile std::uint32_t prof_out_tiles = 0;
    volatile std::uint32_t prof_out_rmw_ops = 0;
    volatile std::uint32_t prof_model_cycles = 0;
    volatile std::uint32_t prof2_win_saved_reads = 0;
    volatile std::uint32_t prof2_win_actual_reads = 0;
    volatile std::uint32_t prof2_out_direct_words = 0;
    volatile std::uint32_t prof2_out_rmw_reads = 0;
    volatile std::uint32_t prof3_wgt_cycles = 0;
    volatile std::uint32_t prof3_win_cycles = 0;
    volatile std::uint32_t prof3_sa_cycles = 0;
    volatile std::uint32_t prof3_post_cycles = 0;
    volatile std::uint32_t prof3_write_cycles = 0;
    volatile std::uint32_t prof3_row_region_cycles = 0;

    if (param_words > 4096U) {
        std::printf("[FAIL] param blob too large for top TB buffer: words=%zu\n", param_words);
        return 1;
    }

    pack_bytes(input_bytes, frame_in, esp_int8::INPUT_FRAME_AXI_WORDS);
    pack_bytes(param_bytes, param, 4096U);
    for (int i = 0; i < esp_int8::OUTPUT_FRAME_AXI_WORDS; ++i) {
        frame_out[i] = 0;
    }

    espnet_encoder_int8_core(frame_in, frame_out, param,
                             esp_int8::MODE_INIT,
                             esp_int8::UOP_COUNT_ENCODER,
                             dbg_status,
                             dbg_heartbeat,
                             dbg_act_words,
                             dbg_wgt_words,
                             dbg_psum_words,
                             dbg_out_words,
                             dbg_hw_version,
                             prof_uop_count,
                             prof_conv_count,
                             prof_win_read_ops,
                             prof_win_words,
                             prof_wgt_words,
                             prof_sa_mac_steps,
                             prof_psum_words,
                             prof_out_tiles,
                             prof_out_rmw_ops,
                             prof_model_cycles,
                             prof2_win_saved_reads,
                             prof2_win_actual_reads,
                             prof2_out_direct_words,
                             prof2_out_rmw_reads,
                             prof3_wgt_cycles,
                             prof3_win_cycles,
                             prof3_sa_cycles,
                             prof3_post_cycles,
                             prof3_write_cycles,
                             prof3_row_region_cycles);
    espnet_encoder_int8_core(frame_in, frame_out, param,
                             esp_int8::MODE_RUN,
                             esp_int8::UOP_COUNT_ENCODER,
                             dbg_status,
                             dbg_heartbeat,
                             dbg_act_words,
                             dbg_wgt_words,
                             dbg_psum_words,
                             dbg_out_words,
                             dbg_hw_version,
                             prof_uop_count,
                             prof_conv_count,
                             prof_win_read_ops,
                             prof_win_words,
                             prof_wgt_words,
                             prof_sa_mac_steps,
                             prof_psum_words,
                             prof_out_tiles,
                             prof_out_rmw_ops,
                             prof_model_cycles,
                             prof2_win_saved_reads,
                             prof2_win_actual_reads,
                             prof2_out_direct_words,
                             prof2_out_rmw_reads,
                             prof3_wgt_cycles,
                             prof3_win_cycles,
                             prof3_sa_cycles,
                             prof3_post_cycles,
                             prof3_write_cycles,
                             prof3_row_region_cycles);

    std::vector<std::uint8_t> hls_output(golden_bytes.size());
    for (std::size_t i = 0; i < hls_output.size(); ++i) {
        hls_output[i] = get_byte(frame_out[i / esp_int8::AXI_WORD_BYTES],
                                 static_cast<int>(i % esp_int8::AXI_WORD_BYTES));
    }
    if (!write_binary(hls_output_path, hls_output)) {
        return 1;
    }
    std::printf("wrote HLS output: %s bytes=%zu\n", hls_output_path, hls_output.size());
    std::printf("top golden dbg status=0x%08x hb=0x%08x act=%u wgt=%u psum=%u out=%u hw=0x%08x\n",
                static_cast<unsigned>(dbg_status),
                static_cast<unsigned>(dbg_heartbeat),
                static_cast<unsigned>(dbg_act_words),
                static_cast<unsigned>(dbg_wgt_words),
                static_cast<unsigned>(dbg_psum_words),
                static_cast<unsigned>(dbg_out_words),
                static_cast<unsigned>(dbg_hw_version));
    std::printf("top golden prof uop=%u conv=%u win_read=%u win_words=%u wgt=%u "
                "sa_steps=%u psum=%u out_tiles=%u rmw_ops=%u model_cycles=%u "
                "win_saved=%u win_actual=%u direct_words=%u rmw_reads=%u "
                "p3_wgt=%u p3_win=%u p3_sa=%u p3_post=%u p3_write=%u p3_row=%u\n",
                static_cast<unsigned>(prof_uop_count),
                static_cast<unsigned>(prof_conv_count),
                static_cast<unsigned>(prof_win_read_ops),
                static_cast<unsigned>(prof_win_words),
                static_cast<unsigned>(prof_wgt_words),
                static_cast<unsigned>(prof_sa_mac_steps),
                static_cast<unsigned>(prof_psum_words),
                static_cast<unsigned>(prof_out_tiles),
                static_cast<unsigned>(prof_out_rmw_ops),
                static_cast<unsigned>(prof_model_cycles),
                static_cast<unsigned>(prof2_win_saved_reads),
                static_cast<unsigned>(prof2_win_actual_reads),
                static_cast<unsigned>(prof2_out_direct_words),
                static_cast<unsigned>(prof2_out_rmw_reads),
                static_cast<unsigned>(prof3_wgt_cycles),
                static_cast<unsigned>(prof3_win_cycles),
                static_cast<unsigned>(prof3_sa_cycles),
                static_cast<unsigned>(prof3_post_cycles),
                static_cast<unsigned>(prof3_write_cycles),
                static_cast<unsigned>(prof3_row_region_cycles));

    int failures = 0;
    int max_abs_diff = 0;
    int max_signed_diff = std::numeric_limits<int>::min();
    int min_signed_diff = std::numeric_limits<int>::max();
    long long sum_abs_diff = 0;
    long long sum_sq_diff = 0;
    int diff_le_1 = 0;
    int diff_le_2 = 0;
    int diff_le_4 = 0;
    int diff_le_8 = 0;
    int diff_le_16 = 0;
    for (std::size_t i = 0; i < golden_bytes.size(); ++i) {
        const std::uint8_t got = hls_output[i];
        const std::uint8_t expected = golden_bytes[i];
        const int diff = static_cast<int>(as_i8(got)) - static_cast<int>(as_i8(expected));
        const int abs_diff = (diff < 0) ? -diff : diff;
        if (abs_diff > max_abs_diff) {
            max_abs_diff = abs_diff;
        }
        if (diff > max_signed_diff) {
            max_signed_diff = diff;
        }
        if (diff < min_signed_diff) {
            min_signed_diff = diff;
        }
        sum_abs_diff += abs_diff;
        sum_sq_diff += static_cast<long long>(diff) * static_cast<long long>(diff);
        if (abs_diff <= 1) {
            ++diff_le_1;
        }
        if (abs_diff <= 2) {
            ++diff_le_2;
        }
        if (abs_diff <= 4) {
            ++diff_le_4;
        }
        if (abs_diff <= 8) {
            ++diff_le_8;
        }
        if (abs_diff <= 16) {
            ++diff_le_16;
        }
        if (got != expected) {
            if (failures < 32) {
                std::printf("[DIFF] top golden mismatch byte=%zu got=%d expected=%d\n",
                            i,
                            static_cast<std::int8_t>(got),
                            static_cast<std::int8_t>(expected));
            }
            ++failures;
        }
    }

    int mask_mismatches = 0;
    int margin_mismatch_le_1 = 0;
    int margin_mismatch_le_2 = 0;
    int margin_mismatch_le_4 = 0;
    int margin_mismatch_le_8 = 0;
    const std::size_t pixel_count = golden_bytes.size() / 2U;
    for (std::size_t p = 0; p < pixel_count; ++p) {
        const std::size_t off = p * 2U;
        const std::int8_t got0 = as_i8(hls_output[off]);
        const std::int8_t got1 = as_i8(hls_output[off + 1U]);
        const std::int8_t exp0 = as_i8(golden_bytes[off]);
        const std::int8_t exp1 = as_i8(golden_bytes[off + 1U]);
        const int got_mask = argmax2(got0, got1);
        const int exp_mask = argmax2(exp0, exp1);
        if (got_mask != exp_mask) {
            ++mask_mismatches;
            const int exp_margin = static_cast<int>(exp1) - static_cast<int>(exp0);
            const int abs_margin = (exp_margin < 0) ? -exp_margin : exp_margin;
            if (abs_margin <= 1) {
                ++margin_mismatch_le_1;
            }
            if (abs_margin <= 2) {
                ++margin_mismatch_le_2;
            }
            if (abs_margin <= 4) {
                ++margin_mismatch_le_4;
            }
            if (abs_margin <= 8) {
                ++margin_mismatch_le_8;
            }
            if (mask_mismatches <= 16) {
                const int h = static_cast<int>(p / 128U);
                const int w = static_cast<int>(p % 128U);
                std::printf("[MASK] pixel=(%d,%d) got=(%d,%d)->%d expected=(%d,%d)->%d exp_margin=%d\n",
                            h,
                            w,
                            static_cast<int>(got0),
                            static_cast<int>(got1),
                            got_mask,
                            static_cast<int>(exp0),
                            static_cast<int>(exp1),
                            exp_mask,
                            exp_margin);
            }
        }
    }

    std::printf("top golden score diff stats: mismatches=%d/%zu max_abs=%d min_diff=%d max_diff=%d "
                "mean_abs=%.6f mse=%.6f <=1=%d <=2=%d <=4=%d <=8=%d <=16=%d\n",
                failures,
                golden_bytes.size(),
                max_abs_diff,
                min_signed_diff,
                max_signed_diff,
                static_cast<double>(sum_abs_diff) / static_cast<double>(golden_bytes.size()),
                static_cast<double>(sum_sq_diff) / static_cast<double>(golden_bytes.size()),
                diff_le_1,
                diff_le_2,
                diff_le_4,
                diff_le_8,
                diff_le_16);
    std::printf("top golden argmax mask stats: mismatches=%d/%zu mismatch_rate=%.6f "
                "mismatch_exp_margin<=1:%d <=2:%d <=4:%d <=8:%d\n",
                mask_mismatches,
                pixel_count,
                static_cast<double>(mask_mismatches) / static_cast<double>(pixel_count),
                margin_mismatch_le_1,
                margin_mismatch_le_2,
                margin_mismatch_le_4,
                margin_mismatch_le_8);

    if (failures != 0) {
        std::printf("top_golden_sample_tb byte-diff diagnostic: mismatches=%d of %zu bytes\n",
                    failures,
                    golden_bytes.size());
    }
    if (mask_mismatches > MAX_ALLOWED_MASK_MISMATCHES) {
        std::printf("top_golden_sample_tb failed: mask mismatches=%d exceeds threshold=%d\n",
                    mask_mismatches,
                    MAX_ALLOWED_MASK_MISMATCHES);
        return 1;
    }

    std::printf("top_golden_sample_tb passed: mask mismatches=%d threshold=%d compared=%zu bytes\n",
                mask_mismatches,
                MAX_ALLOWED_MASK_MISMATCHES,
                golden_bytes.size());
    return 0;
}
