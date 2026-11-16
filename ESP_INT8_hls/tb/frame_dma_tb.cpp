#include "../include/npu_config.hpp"
#include "../include/npu_uop.hpp"
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

namespace esp_int8 {
void frame_dma_load(const axi_vec_t* gmem_frame_in);
bool on_chip_memory_read_packed_tile(const tensor_desc_t&, i32_t, i32_t, u16_t, u8_t, act_vec_t&);
bool on_chip_memory_read_main_uram_word(u8_t, u32_t, act_vec_t&);
bool on_chip_memory_write_fmbuf_abs_word(u8_t, u32_t, axi_vec_t);
bool on_chip_memory_write_input_axi_word(u32_t, axi_vec_t);
}

int main() {
    using namespace esp_int8;
    std::ifstream in("D:/ESP_INT8/hw_artifacts/binary2_int8_h512w512_gray_finetuned_v4/input_q.bin",
                     std::ios::binary);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)), {});
    if (!in || bytes.size() != INPUT_FRAME_BYTES) {
        std::printf("[FAIL] missing/malformed C1 frame input\n");
        return 1;
    }
    static axi_vec_t input[INPUT_FRAME_AXI_WORDS];
    for (unsigned i = 0; i < bytes.size(); ++i) {
        input[i / AXI_WORD_BYTES].range((i % AXI_WORD_BYTES) * 8 + 7,
                                        (i % AXI_WORD_BYTES) * 8) = bytes[i];
    }
    const u8_t bank = static_cast<unsigned>(BANK_FMEM0);
    const axi_vec_t guard = ~axi_vec_t(0);
    if (!on_chip_memory_write_fmbuf_abs_word(bank, INPUT_FRAME_BYTES, guard)) {
        std::printf("[FAIL] unable to initialize DMA end guard\n");
        return 1;
    }
    frame_dma_load(input);
    act_vec_t after = 0;
    if (on_chip_memory_write_input_axi_word(INPUT_FRAME_AXI_WORDS, 0) ||
        !on_chip_memory_read_main_uram_word(bank, INPUT_FRAME_BYTES, after) || after != guard) {
        std::printf("[FAIL] input DMA end guard or bound rejection\n");
        return 1;
    }
    for (int i = 0; i < INPUT_FRAME_AXI_WORDS; ++i) {
        act_vec_t word = 0;
        if (!on_chip_memory_read_main_uram_word(bank, i * AXI_WORD_BYTES, word) ||
            word != input[i]) {
            std::printf("[FAIL] input DMA word=%d\n", i);
            return 1;
        }
    }
    tensor_desc_t desc = {};
    desc.bank_id = bank;
    desc.elem_bytes = 1;
    desc.h = INPUT_FRAME_H;
    desc.w = INPUT_FRAME_W;
    desc.c = INPUT_FRAME_C;
    const unsigned offsets[] = {0, 31, 32, 239, 240, 495, 496, 511, 512, INPUT_FRAME_BYTES - 1};
    for (unsigned off : offsets) {
        act_vec_t word = 0;
        if (!on_chip_memory_read_packed_tile(desc, off / INPUT_FRAME_W,
                                            off % INPUT_FRAME_W, 0, 1, word) ||
            word.range(7, 0).to_uint() != bytes[off] || (word >> 8) != 0) {
            std::printf("[FAIL] C1 frame byte=%u\n", off);
            return 1;
        }
    }
    std::printf("frame_dma_tb passed: %d bytes / %d words\n",
                INPUT_FRAME_BYTES, INPUT_FRAME_AXI_WORDS);
    return 0;
}
