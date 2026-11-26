#include "../include/npu_config.hpp"
#include "../include/npu_uop.hpp"
#ifndef __SYNTHESIS__
#include <cassert>
#endif

namespace esp_int8 {

bool on_chip_memory_write_input_axi_word(u32_t word_offset, axi_vec_t value);

static void advance_input_axis(ap_uint<10>& index, ap_uint<10>& error,
                                unsigned base_step, unsigned error_step) {
#pragma HLS INLINE
    const ap_uint<10> sum = error + error_step;
    const bool carry = sum >= 480U;
    error = carry ? static_cast<ap_uint<10>>(sum - 480U) : sum;
    index += base_step + static_cast<unsigned>(carry);
}

static void frame_dma_read_source(const axi_vec_t* input, bool native, unsigned encoder_words,
                                   unsigned source_w, unsigned channels, unsigned roi_left,
                                   unsigned side, unsigned initial_y, unsigned initial_error,
                                   unsigned base_step, unsigned error_step,
                                   hls::stream<axi_vec_t>& words) {
#pragma HLS INLINE off
    if (native) {
        for (int wi = 0; wi < MAX_INPUT_FRAME_AXI_WORDS; ++wi) {
#pragma HLS PIPELINE II=1
            if (static_cast<unsigned>(wi) >= encoder_words) { break; }
            words.write(input[wi]);
        }
    } else {
        ap_uint<10> y = initial_y, error = initial_error;
        const unsigned row_words = side * channels / AXI_WORD_BYTES;
        for (int row = 0; row < 240; ++row) {
#pragma HLS PIPELINE off
            const unsigned base = (y.to_uint() * source_w + roi_left) * channels / AXI_WORD_BYTES;
            for (int wi = 0; wi < 48; ++wi) {
#pragma HLS PIPELINE II=1
                if (static_cast<unsigned>(wi) >= row_words) { break; }
                words.write(input[base + wi]);
            }
            advance_input_axis(y, error, base_step, error_step);
        }
    }
}

template <int CHANNELS>
static ap_uint<360> compact_input_block(const ap_uint<768>& source, bool large) {
#pragma HLS INLINE
    ap_uint<360> compact = 0;
    // Fifteen outputs repeat over 16 or 32 source pixels, without a divider.
    for (int pixel = 0; pixel < 15; ++pixel) {
#pragma HLS UNROLL
        const int small = pixel < 7 ? pixel : pixel + 1;
        const int big = pixel < 7 ? pixel * 2 + 1 : pixel * 2 + 2;
        compact.range((pixel + 1) * CHANNELS * 8 - 1, pixel * CHANNELS * 8) =
            large ? source.range((big + 1) * CHANNELS * 8 - 1, big * CHANNELS * 8)
                  : source.range((small + 1) * CHANNELS * 8 - 1, small * CHANNELS * 8);
    }
    return compact;
}

static void frame_dma_select_pack(hls::stream<axi_vec_t>& source_words,
                                   bool native, unsigned encoder_words, bool rgb, bool large,
                                   hls::stream<axi_vec_t>& packed_words) {
#pragma HLS INLINE off
    if (native) {
        for (int wi = 0; wi < MAX_INPUT_FRAME_AXI_WORDS; ++wi) {
#pragma HLS PIPELINE II=1
            if (static_cast<unsigned>(wi) >= encoder_words) { break; }
            packed_words.write(source_words.read());
        }
    } else {
        axi_vec_t source_word = 0, packed = 0;
        bool upper = false;
        ap_uint<24> carry = 0;
        ap_uint<2> carry_bytes = 0;
        ap_uint<3> pack_phase = 0;
        const unsigned halves = (rgb ? 3U : 1U) * (large ? 2U : 1U);
        for (int block_index = 0; block_index < 240 * 16; ++block_index) {
#pragma HLS PIPELINE off
            ap_uint<768> block = 0;
            for (int part = 0; part < 6; ++part) {
#pragma HLS PIPELINE off
                if (static_cast<unsigned>(part) >= halves) { break; }
                if (!upper) { source_word = source_words.read(); }
                const ap_uint<128> half = upper ? ap_uint<128>(source_word.range(255, 128))
                                                : ap_uint<128>(source_word.range(127, 0));
                switch (part) {
                    case 0: block.range(127, 0) = half; break;
                    case 1: block.range(255, 128) = half; break;
                    case 2: block.range(383, 256) = half; break;
                    case 3: block.range(511, 384) = half; break;
                    case 4: block.range(639, 512) = half; break;
                    default: block.range(767, 640) = half; break;
                }
                upper = !upper;
            }
            const ap_uint<360> compact = rgb ? compact_input_block<3>(block, large)
                                            : compact_input_block<1>(block, large);
            const unsigned bytes = (rgb ? 45U : 15U) + carry_bytes.to_uint();
            const unsigned groups = bytes >> 2;
            ap_uint<384> pending = (ap_uint<384>(compact) << (carry_bytes.to_uint() * 8U)) | carry;
            for (int group = 0; group < 12; ++group) {
#pragma HLS PIPELINE II=1
                if (static_cast<unsigned>(group) >= groups) { break; }
                packed >>= 32;
                packed.range(255, 224) = pending.range(31, 0);
                pending >>= 32;
                if (pack_phase == 7) { packed_words.write(packed); }
                ++pack_phase;
            }
            carry = pending.range(23, 0);
            carry_bytes = bytes & 3U;
        }
#ifndef __SYNTHESIS__
        assert(!upper && carry_bytes == 0 && carry == 0 && pack_phase == 0);
#endif
    }
}

static void frame_dma_write_input(hls::stream<axi_vec_t>& words, unsigned count) {
#pragma HLS INLINE off
    for (int wi = 0; wi < MAX_INPUT_FRAME_AXI_WORDS; ++wi) {
#pragma HLS PIPELINE II=1
        if (static_cast<unsigned>(wi) >= count) { break; }
        on_chip_memory_write_input_axi_word(static_cast<u32_t>(wi), words.read());
    }
}

static void frame_dma_pipeline(const axi_vec_t* input, unsigned encoder_words, bool native,
                                unsigned source_w, unsigned channels, unsigned roi_left,
                                unsigned side, unsigned initial_y, unsigned initial_error,
                                unsigned base_step, unsigned error_step) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    hls::stream<axi_vec_t> source_words("input_source_words");
    hls::stream<axi_vec_t> packed_words("input_packed_words");
#pragma HLS STREAM variable=source_words depth=16
#pragma HLS STREAM variable=packed_words depth=4
#pragma HLS BIND_STORAGE variable=source_words type=fifo impl=srl
#pragma HLS BIND_STORAGE variable=packed_words type=fifo impl=srl
    frame_dma_read_source(input, native, encoder_words, source_w, channels, roi_left,
                          side, initial_y, initial_error, base_step, error_step, source_words);
    frame_dma_select_pack(source_words, native, encoder_words, channels == 3U,
                          side == 512U, packed_words);
    frame_dma_write_input(packed_words, encoder_words);
#ifndef __SYNTHESIS__
    assert(source_words.empty() && packed_words.empty());
#endif
}

void frame_dma_load(const axi_vec_t* gmem_frame_in) {
#pragma HLS INLINE off
    if (!runtime_profile_ready() || !runtime_io_ready()) { return; }
    runtime_profile_t profile;
    runtime_io_t io;
    runtime_get_profile(profile);
    runtime_get_io(io);
    frame_dma_pipeline(gmem_frame_in, profile.input_bytes / AXI_WORD_BYTES, io.src_mode == 0U,
        io.source_w, io.source_c, io.roi_left, io.roi_h, io.nearest_index, io.nearest_error,
        io.nearest_base_step, io.nearest_error_step);
}

}  // namespace esp_int8
