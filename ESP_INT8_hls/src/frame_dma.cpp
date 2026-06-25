#include "../include/npu_config.hpp"
#include "../include/npu_uop.hpp"

namespace esp_int8 {

bool on_chip_memory_write_input_axi_word(u32_t word_offset, axi_vec_t value);

void frame_dma_load(const axi_vec_t* gmem_frame_in) {
#pragma HLS INLINE off
    for (int word_idx = 0; word_idx < INPUT_FRAME_AXI_WORDS; ++word_idx) {
#pragma HLS PIPELINE II=1
        on_chip_memory_write_input_axi_word(static_cast<u32_t>(word_idx), gmem_frame_in[word_idx]);
    }
}

}  // namespace esp_int8
