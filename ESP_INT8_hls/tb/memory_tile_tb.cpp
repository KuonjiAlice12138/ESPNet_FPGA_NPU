#include "../include/npu_config.hpp"
#include "../include/npu_uop.hpp"
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

namespace esp_int8 {
void param_dma_init(const axi_vec_t*);
bool param_dma_ready();
bool param_dma_get_tensor_desc(u8_t, tensor_desc_t&);
bool on_chip_memory_write_fmbuf_abs_word(u8_t, u32_t, axi_vec_t);
bool on_chip_memory_read_packed_tile(const tensor_desc_t&, i32_t, i32_t, u16_t, u8_t, act_vec_t&);
bool on_chip_memory_prepare_row_base(const tensor_desc_t&, i32_t, u32_t&, bool&);
}

static unsigned pattern(unsigned offset) {
    return (offset * 37U + 19U) & 255U;
}

static bool test_padded_tensor(unsigned tensor_id, unsigned channels, unsigned expected_stride) {
    using namespace esp_int8;
    tensor_desc_t desc = {};
    if (!param_dma_get_tensor_desc(tensor_id, desc) ||
        tensor_phys_c(desc).to_uint() != channels ||
        tensor_row_stride_bytes(desc).to_uint() != expected_stride ||
        (desc.reserved0.to_uint() & TENSOR_ROW_ALIGN32) == 0U) {
        return false;
    }
    for (unsigned i = 0; i < 2U * expected_stride; i += AXI_WORD_BYTES) {
        axi_vec_t word = 0;
        for (unsigned lane = 0; lane < AXI_WORD_BYTES; ++lane) {
            word.range(lane * 8 + 7, lane * 8) = pattern(i + lane);
        }
        if (!on_chip_memory_write_fmbuf_abs_word(desc.bank_id, desc.base_offset + i, word)) {
            return false;
        }
    }
    const unsigned cols[] = {0, desc.w.to_uint() - 1U};
    for (unsigned row = 0; row < 2; ++row) {
        u32_t row_base = 0;
        bool valid = false;
        if (!on_chip_memory_prepare_row_base(desc, row, row_base, valid) || !valid ||
            row_base != desc.base_offset + row * expected_stride) {
            return false;
        }
        for (unsigned col : cols) {
            act_vec_t word = 0;
            const unsigned begin = channels > 32 ? channels - 1U : 0U;
            if (!on_chip_memory_read_packed_tile(desc, row, col, begin, 32, word)) {
                return false;
            }
            for (unsigned lane = 0; lane < 32; ++lane) {
                const unsigned expected = begin + lane < channels
                    ? pattern(row * expected_stride + col * channels + begin + lane) : 0U;
                if (word.range(lane * 8 + 7, lane * 8).to_uint() != expected) {
                    return false;
                }
            }
        }
    }
    act_vec_t padding = ~act_vec_t(0);
    return on_chip_memory_read_packed_tile(desc, -1, 0, 0, 32, padding) && padding == 0;
}

int main() {
    using namespace esp_int8;
    std::ifstream in("D:/ESP_INT8/hw_artifacts/binary2_int8_h240w240_gray_v4/PARAM.BIN",
                     std::ios::binary);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)), {});
    if (!in || bytes.empty()) {
        return 1;
    }
    std::vector<axi_vec_t> param((bytes.size() + 31U) / 32U, 0);
    for (unsigned i = 0; i < bytes.size(); ++i) {
        param[i / 32].range((i % 32) * 8 + 7, (i % 32) * 8) = bytes[i];
    }
    param_dma_init(param.data());
    if (!param_dma_ready() ||
        !test_padded_tensor(TID_B1_CAT, 17, 2048) ||
        !test_padded_tensor(TID_B2_CAT, 129, 7744)) {
        std::printf("[FAIL] C17/C129 padded-row addressing\n");
        return 1;
    }
    std::printf("memory_tile_tb passed: C17/C129 row padding, tails and boundary reads\n");
    return 0;
}
