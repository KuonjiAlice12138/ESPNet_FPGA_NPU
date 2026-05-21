#include "../include/npu_config.hpp"
#include "../include/npu_uop.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#ifndef DBG_NAME
#define DBG_NAME "dbg"
#endif

#ifndef DBG_STOP_AFTER
#define DBG_STOP_AFTER 4
#endif

#ifndef DBG_DUMP_TENSOR
#define DBG_DUMP_TENSOR esp_int8::TID_B1_ACT
#endif

#ifndef DBG_DUMP_WORDS
#define DBG_DUMP_WORDS esp_int8::OUTPUT_FRAME_AXI_WORDS
#endif

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
                              volatile std::uint32_t& prof3_row_region_cycles,
                              volatile std::uint32_t& prof5_if_words,
                              volatile std::uint32_t& prof5_frame_load_words,
                              volatile std::uint32_t& prof5_frame_store_words,
                              volatile std::uint32_t& prof5_uop_fetches,
                              volatile std::uint32_t& prof5_pool_tiles,
                              volatile std::uint32_t& prof5_affine_tiles,
                              volatile std::uint32_t& prof5_add_tiles,
                              volatile std::uint32_t& prof5_store_tiles,
                              volatile std::uint32_t& prof5_nonconv_mem_ops,
                              volatile std::uint32_t& prof5_conv_model_cycles,
                              volatile std::uint32_t& prof5_total_work_units);

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

int main() {
    const char* param_path = "D:/ESP_INT8/hw_artifacts/hw_constrained_qat_3ep_single/param_blob.bin";
    const char* input_path = "D:/ESP_INT8/hw_artifacts/hw_constrained_qat_3ep_single/input_q.bin";

    std::vector<std::uint8_t> param_bytes;
    std::vector<std::uint8_t> input_bytes;
    if (!read_binary(param_path, param_bytes) ||
        !read_binary(input_path, input_bytes)) {
        return 1;
    }
    if (input_bytes.size() != static_cast<std::size_t>(esp_int8::INPUT_FRAME_BYTES)) {
        std::printf("[FAIL] unexpected input size: %zu\n", input_bytes.size());
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
    volatile std::uint32_t prof5_if_words = 0;
    volatile std::uint32_t prof5_frame_load_words = 0;
    volatile std::uint32_t prof5_frame_store_words = 0;
    volatile std::uint32_t prof5_uop_fetches = 0;
    volatile std::uint32_t prof5_pool_tiles = 0;
    volatile std::uint32_t prof5_affine_tiles = 0;
    volatile std::uint32_t prof5_add_tiles = 0;
    volatile std::uint32_t prof5_store_tiles = 0;
    volatile std::uint32_t prof5_nonconv_mem_ops = 0;
    volatile std::uint32_t prof5_conv_model_cycles = 0;
    volatile std::uint32_t prof5_total_work_units = 0;

    if (param_words > 4096U) {
        std::printf("[FAIL] param blob too large: words=%zu\n", param_words);
        return 1;
    }

    pack_bytes(input_bytes, frame_in, esp_int8::INPUT_FRAME_AXI_WORDS);
    pack_bytes(param_bytes, param, 4096U);
    for (int i = 0; i < esp_int8::OUTPUT_FRAME_AXI_WORDS; ++i) {
        frame_out[i] = 0;
    }

    const std::uint32_t dump_words = static_cast<std::uint32_t>(DBG_DUMP_WORDS);
    const std::uint32_t dump_tensor = static_cast<std::uint32_t>(DBG_DUMP_TENSOR) & 0xffU;
    const std::uint32_t stop_after = static_cast<std::uint32_t>(DBG_STOP_AFTER);
    const std::uint32_t debug_mode =
        esp_int8::MODE_RUN |
        esp_int8::DEBUG_MODE_ENABLE_MASK |
        (dump_tensor << esp_int8::DEBUG_MODE_DUMP_TENSOR_SHIFT) |
        (dump_words << esp_int8::DEBUG_MODE_DUMP_WORDS_SHIFT);
    const std::uint32_t debug_uop_count =
        esp_int8::UOP_COUNT_ENCODER |
        ((stop_after + 1U) << esp_int8::DEBUG_STOP_AFTER_SHIFT);

    std::printf("debug seq %s: stop_after=%u dump_tensor=%u dump_words=%u\n",
                DBG_NAME,
                stop_after,
                dump_tensor,
                dump_words);

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
                             prof3_row_region_cycles,
                             prof5_if_words,
                             prof5_frame_load_words,
                             prof5_frame_store_words,
                             prof5_uop_fetches,
                             prof5_pool_tiles,
                             prof5_affine_tiles,
                             prof5_add_tiles,
                             prof5_store_tiles,
                             prof5_nonconv_mem_ops,
                             prof5_conv_model_cycles,
                             prof5_total_work_units);
    espnet_encoder_int8_core(frame_in, frame_out, param,
                             debug_mode,
                             debug_uop_count,
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
                             prof3_row_region_cycles,
                             prof5_if_words,
                             prof5_frame_load_words,
                             prof5_frame_store_words,
                             prof5_uop_fetches,
                             prof5_pool_tiles,
                             prof5_affine_tiles,
                             prof5_add_tiles,
                             prof5_store_tiles,
                             prof5_nonconv_mem_ops,
                             prof5_conv_model_cycles,
                             prof5_total_work_units);

    std::vector<std::uint8_t> output(static_cast<std::size_t>(dump_words) *
                                     static_cast<std::size_t>(esp_int8::AXI_WORD_BYTES));
    int nonzero = 0;
    unsigned checksum = 0;
    for (std::size_t i = 0; i < output.size(); ++i) {
        const std::uint8_t value = get_byte(frame_out[i / esp_int8::AXI_WORD_BYTES],
                                            static_cast<int>(i % esp_int8::AXI_WORD_BYTES));
        output[i] = value;
        checksum = (checksum * 131U) + static_cast<unsigned>(value);
        if (value != 0U) {
            ++nonzero;
        }
    }

    char out_path[128];
    std::snprintf(out_path, sizeof(out_path), "dbg_%s_out.bin", DBG_NAME);
    if (!write_binary(out_path, output)) {
        return 1;
    }

    std::printf("debug seq %s wrote %s bytes=%zu nonzero=%d checksum=0x%08x\n",
                DBG_NAME,
                out_path,
                output.size(),
                nonzero,
                checksum);
    std::printf("debug seq %s dbg status=0x%08x hb=0x%08x act=%u wgt=%u psum=%u out=%u\n",
                DBG_NAME,
                static_cast<unsigned>(dbg_status),
                static_cast<unsigned>(dbg_heartbeat),
                static_cast<unsigned>(dbg_act_words),
                static_cast<unsigned>(dbg_wgt_words),
                static_cast<unsigned>(dbg_psum_words),
                static_cast<unsigned>(dbg_out_words));
    std::printf("debug seq %s first bytes:", DBG_NAME);
    const std::size_t preview = (output.size() < 16U) ? output.size() : 16U;
    for (std::size_t i = 0; i < preview; ++i) {
        std::printf(" %02x", static_cast<unsigned>(output[i]));
    }
    std::printf("\n");

    return 0;
}
