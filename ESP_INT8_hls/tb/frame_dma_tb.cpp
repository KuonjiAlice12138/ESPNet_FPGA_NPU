#include "../include/npu_config.hpp"
#include "../include/npu_uop.hpp"
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace esp_int8 {
void frame_dma_load(const axi_vec_t* gmem_frame_in, std::uint32_t src_mode);
bool on_chip_memory_read_packed_tile(const tensor_desc_t&, i32_t, i32_t, u16_t, u8_t, act_vec_t&);
bool on_chip_memory_read_main_uram_word(u8_t, u32_t, act_vec_t&);
bool on_chip_memory_write_fmbuf_abs_word(u8_t, u32_t, axi_vec_t);
}

static bool read_bytes(const std::string& path, std::vector<std::uint8_t>& bytes) {
    std::ifstream file(path, std::ios::binary);
    bytes.assign(std::istreambuf_iterator<char>(file), {});
    return static_cast<bool>(file);
}

static bool check_frame(std::uint32_t mode,
                        const std::vector<std::uint8_t>& source,
                        const std::vector<std::uint8_t>& expected) {
    using namespace esp_int8;
    if (source.empty() || source.size() > INPUT_SOURCE_MAX_BYTES ||
        source.size() % AXI_WORD_BYTES != 0 || expected.size() != INPUT_FRAME_BYTES) {
        std::printf("[FAIL] bad DMA test sizes for mode=%u\n", mode);
        return false;
    }
    std::vector<axi_vec_t> input(source.size() / AXI_WORD_BYTES, 0);
    for (std::size_t i = 0; i < source.size(); ++i) {
        input[i / AXI_WORD_BYTES].range((i % AXI_WORD_BYTES) * 8 + 7,
                                        (i % AXI_WORD_BYTES) * 8) = source[i];
    }
    const u8_t bank = static_cast<unsigned>(BANK_FMEM0);
    const axi_vec_t sentinel = ~static_cast<axi_vec_t>(0);
    if (!on_chip_memory_write_fmbuf_abs_word(bank, INPUT_FRAME_BYTES, sentinel)) {
        return false;
    }
    frame_dma_load(input.data(), mode);
    for (int i = 0; i < INPUT_FRAME_AXI_WORDS; ++i) {
        axi_vec_t golden = 0;
        for (int lane = 0; lane < AXI_WORD_BYTES; ++lane) {
            golden.range(lane * 8 + 7, lane * 8) = expected[i * AXI_WORD_BYTES + lane];
        }
        act_vec_t actual = 0;
        if (!on_chip_memory_read_main_uram_word(bank, i * AXI_WORD_BYTES, actual) || actual != golden) {
            std::printf("[FAIL] mode=%u word=%d\n", mode, i);
            return false;
        }
    }
    act_vec_t guard = 0;
    if (!on_chip_memory_read_main_uram_word(bank, INPUT_FRAME_BYTES, guard) || guard != sentinel) {
        std::printf("[FAIL] DMA wrote past the 57600-byte encoder input\n");
        return false;
    }
    std::printf("DMA mode=%u: 0/%d byte mismatches; guard and FIFO drain PASS\n",
                mode, INPUT_FRAME_BYTES);
    return true;
}

static bool check_synthetic(std::uint32_t mode) {
    using namespace esp_int8;
    const unsigned side = mode == INPUT_SRC_H512W1024 ? 512U : 256U;
    std::vector<std::uint8_t> source(side * side * 2);
    std::uint32_t random = 1023U + mode;
    for (auto& byte : source) {
        random = random * 1664525U + 1013904223U;
        byte = static_cast<std::uint8_t>(random >> 24);
    }
    // Exercise signed extrema at the first four independently selected pixels.
    const unsigned values[] = {0x80, 0xff, 0, 0x7f};
    const unsigned sy = side / 480U;
    for (unsigned x = 0; x < 4; ++x) {
        const unsigned sx = side / 2U + ((2U * x + 1U) * side) / 480U;
        source[sy * side * 2U + sx] = values[x];
    }
    std::vector<std::uint8_t> expected(INPUT_FRAME_BYTES);
    for (unsigned y = 0; y < INPUT_FRAME_H; ++y) {
        const unsigned src_y = ((2U * y + 1U) * side) / 480U;
        for (unsigned x = 0; x < INPUT_FRAME_W; ++x) {
            const unsigned src_x = side / 2U + ((2U * x + 1U) * side) / 480U;
            expected[y * INPUT_FRAME_W + x] = source[src_y * side * 2U + src_x];
        }
    }
    return check_frame(mode, source, expected);
}

int main() {
    using namespace esp_int8;
    std::ifstream in("D:/ESP_INT8/hw_artifacts/binary2_int8_h240w240_gray_r6_v4/input_q.bin",
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
    frame_dma_load(input, INPUT_SRC_NATIVE);
    const u8_t bank = static_cast<unsigned>(BANK_FMEM0);
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
    const unsigned offsets[] = {0, 31, 32, 239, 240, INPUT_FRAME_BYTES - 1};
    for (unsigned off : offsets) {
        act_vec_t word = 0;
        if (!on_chip_memory_read_packed_tile(desc, off / INPUT_FRAME_W,
                                            off % INPUT_FRAME_W, 0, 1, word) ||
            word.range(7, 0).to_uint() != bytes[off] || (word >> 8) != 0) {
            std::printf("[FAIL] C1 frame byte=%u\n", off);
            return 1;
        }
    }
    if (!check_synthetic(INPUT_SRC_H512W1024) || !check_synthetic(INPUT_SRC_H256W512) ||
        !check_frame(INPUT_SRC_NATIVE, bytes, bytes)) {
        return 1;
    }
    const std::string root = "D:/ESPNet/resolution_h240w240_gray/input_resize_1023/";
    for (unsigned mode : {INPUT_SRC_H256W512, INPUT_SRC_H512W1024}) {
        const std::string side = mode == INPUT_SRC_H256W512 ? "256" : "512";
        std::vector<std::uint8_t> source, expected;
        if (!read_bytes(root + "selected_int8_reference/I" + side + "Q.BIN", source) ||
            !read_bytes(root + "selected_reference/Q" + side + ".BIN", expected) ||
            !check_frame(mode, source, expected)) {
            std::printf("[FAIL] prequantized reference mode=%u\n", mode);
            return 1;
        }
    }
    if (!check_frame(INPUT_SRC_NATIVE, bytes, bytes)) {
        return 1;
    }
    std::printf("frame_dma_tb passed: native/re-entry, two signed-pattern modes, two reference frames\n");
    return 0;
}
