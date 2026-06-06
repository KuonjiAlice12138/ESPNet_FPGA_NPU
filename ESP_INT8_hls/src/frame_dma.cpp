#include "../include/npu_config.hpp"
#include "../include/npu_uop.hpp"

namespace esp_int8 {

bool on_chip_memory_write_axi_word(u8_t tensor_id, u32_t word_offset, axi_vec_t value);

void frame_dma_load(const axi_vec_t* gmem_frame_in) {
#pragma HLS INLINE off
    const u8_t input_tensor = static_cast<u8_t>(static_cast<unsigned>(TID_INPUT));
    for (int word_idx = 0; word_idx < INPUT_FRAME_AXI_WORDS; ++word_idx) {
#pragma HLS PIPELINE II=1
        on_chip_memory_write_axi_word(input_tensor, static_cast<u32_t>(word_idx), gmem_frame_in[word_idx]);
    }
}

void frame_dma_store(axi_vec_t* gmem_frame_out) {
#pragma HLS INLINE off
    (void)gmem_frame_out;
}

}  // namespace esp_int8
