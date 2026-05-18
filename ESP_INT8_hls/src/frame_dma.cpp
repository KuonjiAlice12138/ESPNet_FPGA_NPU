#include "../include/npu_config.hpp"
#include "../include/npu_uop.hpp"

namespace esp_int8 {

bool on_chip_memory_write_byte(u8_t tensor_id, u32_t elem_offset, u8_t value);
bool on_chip_memory_read_byte(u8_t tensor_id, u32_t elem_offset, u8_t& value);
bool on_chip_memory_write_axi_word(u8_t tensor_id, u32_t word_offset, axi_vec_t value);
bool on_chip_memory_read_axi_word(u8_t tensor_id, u32_t word_offset, axi_vec_t& value);
bool on_chip_memory_read_elem(const tensor_desc_t& desc, u16_t h, u16_t w, u16_t c, i8_t& value);

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
    const u8_t output_tensor = static_cast<u8_t>(static_cast<unsigned>(TID_OUT));
    for (int word_idx = 0; word_idx < OUTPUT_FRAME_AXI_WORDS; ++word_idx) {
#pragma HLS PIPELINE II=1
        axi_vec_t word = 0;
        on_chip_memory_read_axi_word(output_tensor, static_cast<u32_t>(word_idx), word);
        gmem_frame_out[word_idx] = word;
    }
}

void frame_dma_store_tensor(axi_vec_t* gmem_frame_out,
                            const tensor_desc_t& desc,
                            u32_t max_words) {
#pragma HLS INLINE off
    u16_t h = 0;
    u16_t w = 0;
    u16_t c = 0;
    bool done = (desc.h == 0 || desc.w == 0 || desc.c == 0);
    const u32_t limit_words =
        (max_words == 0 || max_words > DEBUG_DUMP_MAX_AXI_WORDS)
            ? static_cast<u32_t>(DEBUG_DUMP_MAX_AXI_WORDS)
            : max_words;

    for (int word_idx = 0; word_idx < DEBUG_DUMP_MAX_AXI_WORDS; ++word_idx) {
#pragma HLS PIPELINE off
        if (static_cast<u32_t>(word_idx) >= limit_words || done) {
            break;
        }

        axi_vec_t word = 0;
        for (int lane = 0; lane < AXI_WORD_BYTES; ++lane) {
#pragma HLS PIPELINE off
            if (!done) {
                i8_t value = 0;
                if (!on_chip_memory_read_elem(desc, h, w, c, value)) {
                    done = true;
                    break;
                }
                word.range(lane * 8 + 7, lane * 8) = value.range(7, 0);

                ++c;
                if (c >= desc.c) {
                    c = 0;
                    ++w;
                    if (w >= desc.w) {
                        w = 0;
                        ++h;
                        if (h >= desc.h) {
                            done = true;
                        }
                    }
                }
            }
        }
        gmem_frame_out[word_idx] = word;
    }
}

}  // namespace esp_int8
