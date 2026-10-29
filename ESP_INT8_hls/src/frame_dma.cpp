#include "../include/npu_config.hpp"
#include "../include/npu_uop.hpp"

#ifndef __SYNTHESIS__
#include <cassert>
#endif

namespace esp_int8 {

bool on_chip_memory_write_input_axi_word(u32_t word_offset, axi_vec_t value);

static void frame_dma_advance_axis(ap_uint<10>& index, ap_uint<10>& error, bool large) {
#pragma HLS INLINE
    const ap_uint<10> sum = error + (large ? INPUT_RESIZE_LARGE_ERROR_STEP
                                          : INPUT_RESIZE_SMALL_ERROR_STEP);
    const bool carry = sum >= INPUT_RESIZE_DENOM;
    error = carry ? static_cast<ap_uint<10>>(sum - INPUT_RESIZE_DENOM) : sum;
    index += (large ? INPUT_RESIZE_LARGE_BASE_STEP : INPUT_RESIZE_SMALL_BASE_STEP) + carry;
}

static void frame_dma_read_source(const axi_vec_t* gmem_frame_in,
                                  std::uint32_t src_mode,
                                  hls::stream<axi_vec_t>& source_words) {
#pragma HLS INLINE off
    if (src_mode == INPUT_SRC_NATIVE) {
        for (int word_idx = 0; word_idx < INPUT_FRAME_AXI_WORDS; ++word_idx) {
#pragma HLS PIPELINE II=1
            source_words.write(gmem_frame_in[word_idx]);
        }
    } else {
        const bool large = src_mode == INPUT_SRC_H512W1024;
        const int roi_words = large ? 16 : 8;
        ap_uint<10> src_y = large ? INPUT_RESIZE_LARGE_INITIAL_INDEX : INPUT_RESIZE_SMALL_INITIAL_INDEX;
        ap_uint<10> y_error = large ? INPUT_RESIZE_LARGE_INITIAL_ERROR : INPUT_RESIZE_SMALL_INITIAL_ERROR;
        for (int row = 0; row < INPUT_FRAME_H; ++row) {
#pragma HLS PIPELINE off
            const ap_uint<14> row_base =
                (static_cast<ap_uint<14>>(src_y) << (large ? 5 : 4)) + (large ? 8 : 4);
            // One contiguous, aligned ROI burst per selected source row.
            for (int word_idx = 0; word_idx < roi_words; ++word_idx) {
#pragma HLS PIPELINE II=1
                source_words.write(gmem_frame_in[row_base.to_uint() + word_idx]);
            }
            frame_dma_advance_axis(src_y, y_error, large);
        }
    }
}

static ap_uint<120> frame_dma_compact_block(axi_vec_t word, bool large, bool upper_half) {
#pragma HLS INLINE
    const ap_uint<128> half = upper_half ? static_cast<ap_uint<128>>(word.range(255, 128))
                                         : static_cast<ap_uint<128>>(word.range(127, 0));
    ap_uint<120> compact;
    // The half-pixel map repeats every 15 outputs: 16 or 32 source bytes.
    for (int lane = 0; lane < 15; ++lane) {
#pragma HLS UNROLL
        const int small_byte = lane < 7 ? lane : lane + 1;
        const int large_byte = lane < 7 ? 2 * lane + 1 : 2 * lane + 2;
        compact.range(lane * 8 + 7, lane * 8) =
            large ? static_cast<u8_t>(word.range(large_byte * 8 + 7, large_byte * 8))
                  : static_cast<u8_t>(half.range(small_byte * 8 + 7, small_byte * 8));
    }
    return compact;
}

static void frame_dma_select_pack(hls::stream<axi_vec_t>& source_words,
                                  std::uint32_t src_mode,
                                  hls::stream<axi_vec_t>& packed_words) {
#pragma HLS INLINE off
    if (src_mode == INPUT_SRC_NATIVE) {
        for (int word_idx = 0; word_idx < INPUT_FRAME_AXI_WORDS; ++word_idx) {
#pragma HLS PIPELINE II=1
            packed_words.write(source_words.read());
        }
    } else {
        const bool large = src_mode == INPUT_SRC_H512W1024;
        constexpr int service_count = INPUT_FRAME_H * (INPUT_FRAME_W / 15) * 4;
        static_assert(INPUT_FRAME_W % 15 == 0 && INPUT_RESIZE_LANES == 4,
                      "Input compaction requires whole 15-pixel blocks and four-byte groups");
        axi_vec_t source = 0;
        axi_vec_t packed = 0;
        ap_uint<128> groups = 0;
        ap_uint<24> carry = 0;
        ap_uint<2> carry_bytes = 0;
        ap_uint<3> pack_phase = 0;
        bool upper_half = false;
        bool full_block = false;
        for (int service = 0; service < service_count; ++service) {
#pragma HLS PIPELINE II=1
            const ap_uint<2> slot = static_cast<ap_uint<2>>(service);
            if (slot == 0) {
                if (large || !upper_half) {
                    source = source_words.read();
                }
                const ap_uint<120> compact = frame_dma_compact_block(source, large, upper_half);
                const ap_uint<5> carry_shift = static_cast<ap_uint<5>>(carry_bytes) << 3;
                groups = (static_cast<ap_uint<128>>(compact) << carry_shift) |
                         static_cast<ap_uint<128>>(carry);
                full_block = carry_bytes != 0;
                // Four blocks emit 3,4,4,4 groups; only 0..3 bytes need carrying.
                switch (carry_bytes.to_uint()) {
                case 0: carry = compact.range(119, 96); break;
                case 1: carry = 0; break;
                case 2: carry = static_cast<u8_t>(compact.range(119, 112)); break;
                default: carry = static_cast<u16_t>(compact.range(119, 104)); break;
                }
                --carry_bytes;
            }
            if (slot != 3 || full_block) {
                // Keep the half-word carry across 240-byte row boundaries.
                packed >>= 32;
                packed.range(255, 224) = groups.range(31, 0);
                groups >>= 32;
                if (pack_phase == 7) {
                    packed_words.write(packed);
                }
                ++pack_phase;
            }
            if (slot == 3) {
                upper_half = !upper_half;
            }
        }
#ifndef __SYNTHESIS__
        assert(carry_bytes == 0 && carry == 0 && pack_phase == 0);
#endif
    }
}

static void frame_dma_write_input(hls::stream<axi_vec_t>& packed_words) {
#pragma HLS INLINE off
    for (int word_idx = 0; word_idx < INPUT_FRAME_AXI_WORDS; ++word_idx) {
#pragma HLS PIPELINE II=1
        on_chip_memory_write_input_axi_word(static_cast<u32_t>(word_idx), packed_words.read());
    }
}

// The caller validates src_mode before starting this fixed-output-token DAG.
void frame_dma_load(const axi_vec_t* gmem_frame_in, std::uint32_t src_mode) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    hls::stream<axi_vec_t> source_words("input_source_words");
    hls::stream<axi_vec_t> packed_words("input_packed_words");
#pragma HLS STREAM variable=source_words depth=16
#pragma HLS STREAM variable=packed_words depth=4
#pragma HLS BIND_STORAGE variable=source_words type=fifo impl=srl
#pragma HLS BIND_STORAGE variable=packed_words type=fifo impl=srl
    frame_dma_read_source(gmem_frame_in, src_mode, source_words);
    frame_dma_select_pack(source_words, src_mode, packed_words);
    frame_dma_write_input(packed_words);
#ifndef __SYNTHESIS__
    assert(source_words.empty() && packed_words.empty());
#endif
}

}  // namespace esp_int8
