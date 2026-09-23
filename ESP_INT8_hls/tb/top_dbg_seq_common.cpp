#include "../include/npu_config.hpp"
#include "../include/npu_uop.hpp"
#include "top_call.hpp"

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
    const char* param_path = "D:/ESP_INT8/hw_artifacts/binary2_int8_h256w512_r2_v4/PARAM.BIN";
    const char* input_path = "D:/ESP_INT8/hw_artifacts/binary2_int8_h256w512_r2_v4/input_q.bin";

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
    static esp_int8::axi_vec_t param[8192];
    if (param_words > 8192U) {
        std::printf("[FAIL] param blob too large: words=%zu\n", param_words);
        return 1;
    }

    pack_bytes(input_bytes, frame_in, esp_int8::INPUT_FRAME_AXI_WORDS);
    pack_bytes(param_bytes, param, 8192U);
    for (int i = 0; i < esp_int8::OUTPUT_FRAME_AXI_WORDS; ++i) {
        frame_out[i] = 0;
    }

    const std::uint32_t dump_words = static_cast<std::uint32_t>(DBG_DUMP_WORDS);
    const std::uint32_t dump_tensor = static_cast<std::uint32_t>(DBG_DUMP_TENSOR) & 0xffU;
    const std::uint32_t stop_after = static_cast<std::uint32_t>(DBG_STOP_AFTER);
    const std::uint32_t debug_mode = esp_int8::MODE_RUN;
    const std::uint32_t debug_uop_count = esp_int8::UOP_COUNT_ENCODER;

    std::printf("debug seq %s: stop_after=%u dump_tensor=%u dump_words=%u\n",
                DBG_NAME,
                stop_after,
                dump_tensor,
                dump_words);

    call_espnet_encoder_int8_core(frame_in, frame_out, param,
                                  esp_int8::MODE_INIT,
                                  esp_int8::UOP_COUNT_ENCODER);
    call_espnet_encoder_int8_core(frame_in, frame_out, param,
                                  debug_mode,
                                  debug_uop_count);

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
