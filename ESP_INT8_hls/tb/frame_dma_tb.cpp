#include "../include/npu_config.hpp"
#include "../include/npu_uop.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

namespace esp_int8 {
void frame_dma_load(const axi_vec_t* gmem_frame_in);
void frame_dma_store(axi_vec_t* gmem_frame_out);
bool on_chip_memory_read_byte(u8_t tensor_id, u32_t elem_offset, u8_t& value);
bool on_chip_memory_copy_tensor_prefix(u8_t src_tensor, u8_t dst_tensor, u32_t byte_count);
}  // namespace esp_int8

static std::uint8_t pattern(std::uint32_t byte_idx) {
    return static_cast<std::uint8_t>((byte_idx * 37U + 19U) & 0xffU);
}

static bool load_input_file(std::vector<std::uint8_t>& bytes) {
    const char* path = "D:/ESP_INT8/hw_artifacts/sched_v3_single_p7_hwconv_0623/input_q.bin";
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return bytes.size() == static_cast<std::size_t>(esp_int8::INPUT_FRAME_BYTES);
}

static std::uint8_t source_byte(const std::vector<std::uint8_t>& file_bytes, std::uint32_t byte_idx) {
    if (!file_bytes.empty()) {
        return file_bytes[byte_idx];
    }
    return pattern(byte_idx);
}

static void set_byte(esp_int8::axi_vec_t& word, int lane, std::uint8_t value) {
    word.range(lane * 8 + 7, lane * 8) = value;
}

static std::uint8_t get_byte(const esp_int8::axi_vec_t& word, int lane) {
    return static_cast<std::uint8_t>(word.range(lane * 8 + 7, lane * 8).to_uint());
}

int main() {
    static esp_int8::axi_vec_t frame_in[esp_int8::INPUT_FRAME_AXI_WORDS];
    static esp_int8::axi_vec_t frame_out[esp_int8::OUTPUT_FRAME_AXI_WORDS];
    std::vector<std::uint8_t> file_bytes;
    const bool using_file = load_input_file(file_bytes);

    for (int word_idx = 0; word_idx < esp_int8::INPUT_FRAME_AXI_WORDS; ++word_idx) {
        esp_int8::axi_vec_t word = 0;
        for (int lane = 0; lane < esp_int8::AXI_WORD_BYTES; ++lane) {
            const std::uint32_t byte_idx =
                static_cast<std::uint32_t>(word_idx * esp_int8::AXI_WORD_BYTES + lane);
            set_byte(word, lane, source_byte(file_bytes, byte_idx));
        }
        frame_in[word_idx] = word;
    }

    for (int i = 0; i < esp_int8::OUTPUT_FRAME_AXI_WORDS; ++i) {
        frame_out[i] = 0;
    }

    esp_int8::frame_dma_load(frame_in);

    const std::uint32_t probe_offsets[] = {
        0U,
        31U,
        32U,
        1024U,
        static_cast<std::uint32_t>(esp_int8::INPUT_FRAME_BYTES - 1),
    };
    for (std::uint32_t off : probe_offsets) {
        esp_int8::u8_t value = 0;
        if (!esp_int8::on_chip_memory_read_byte(esp_int8::TID_INPUT, off, value)) {
            std::printf("failed to read T_INPUT offset %u\n", off);
            return 1;
        }
        const std::uint8_t exp = source_byte(file_bytes, off);
        if (value != exp) {
            std::printf("T_INPUT mismatch at %u: got=%u expected=%u\n",
                        off,
                        static_cast<unsigned>(value),
                        static_cast<unsigned>(exp));
            return 1;
        }
    }

    if (!esp_int8::on_chip_memory_copy_tensor_prefix(
            esp_int8::TID_INPUT, esp_int8::TID_OUT, esp_int8::OUTPUT_FRAME_BYTES)) {
        std::printf("failed to copy T_INPUT prefix to T_OUT\n");
        return 1;
    }

    esp_int8::frame_dma_store(frame_out);

    for (int word_idx = 0; word_idx < esp_int8::OUTPUT_FRAME_AXI_WORDS; ++word_idx) {
        for (int lane = 0; lane < esp_int8::AXI_WORD_BYTES; ++lane) {
            const std::uint32_t byte_idx =
                static_cast<std::uint32_t>(word_idx * esp_int8::AXI_WORD_BYTES + lane);
            const std::uint8_t got = get_byte(frame_out[word_idx], lane);
            const std::uint8_t exp = source_byte(file_bytes, byte_idx);
            if (got != exp) {
                std::printf("frame_out mismatch at %u: got=%u expected=%u\n",
                            byte_idx,
                            static_cast<unsigned>(got),
                            static_cast<unsigned>(exp));
                return 1;
            }
        }
    }

    std::printf("frame_dma_tb passed: input_bytes=%d output_bytes=%d source=%s\n",
                esp_int8::INPUT_FRAME_BYTES,
                esp_int8::OUTPUT_FRAME_BYTES,
                using_file ? "input_q.bin" : "pattern");
    return 0;
}
