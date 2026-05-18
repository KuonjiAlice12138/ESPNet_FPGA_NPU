#include "../include/npu_config.hpp"
#include "../include/npu_uop.hpp"

#include <cstdint>
#include <cstdio>

void espnet_encoder_int8_core(const esp_int8::axi_vec_t* gmem_frame_in,
                              esp_int8::axi_vec_t* gmem_frame_out,
                              const esp_int8::axi_vec_t* gmem_param,
                              std::uint32_t mode,
                              std::uint32_t uop_count,
                              volatile std::uint32_t& dbg_status,
                              volatile std::uint32_t& dbg_heartbeat,
                              volatile std::uint32_t& dbg_act_words,
                              volatile std::uint32_t& dbg_wgt_words,
                              volatile std::uint32_t& dbg_psum_words,
                              volatile std::uint32_t& dbg_out_words,
                              volatile std::uint32_t& dbg_hw_version,
                              volatile std::uint32_t& prof_uop_count,
                              volatile std::uint32_t& prof_conv_count,
                              volatile std::uint32_t& prof_win_read_ops,
                              volatile std::uint32_t& prof_win_words,
                              volatile std::uint32_t& prof_wgt_words,
                              volatile std::uint32_t& prof_sa_mac_steps,
                              volatile std::uint32_t& prof_psum_words,
                              volatile std::uint32_t& prof_out_tiles,
                              volatile std::uint32_t& prof_out_rmw_ops,
                              volatile std::uint32_t& prof_model_cycles,
                              volatile std::uint32_t& prof2_win_saved_reads,
                              volatile std::uint32_t& prof2_win_actual_reads,
                              volatile std::uint32_t& prof2_out_direct_words,
                              volatile std::uint32_t& prof2_out_rmw_reads,
                              volatile std::uint32_t& prof3_wgt_cycles,
                              volatile std::uint32_t& prof3_win_cycles,
                              volatile std::uint32_t& prof3_sa_cycles,
                              volatile std::uint32_t& prof3_post_cycles,
                              volatile std::uint32_t& prof3_write_cycles,
                              volatile std::uint32_t& prof3_row_region_cycles);

static constexpr std::uint32_t OFF_TENSOR = 128;
static constexpr std::uint32_t OFF_SCALE = 512;
static constexpr std::uint32_t OFF_CONV = 576;
static constexpr std::uint32_t OFF_AFFINE = 640;
static constexpr std::uint32_t OFF_ADD = 704;
static constexpr std::uint32_t OFF_POOL = 768;
static constexpr std::uint32_t OFF_UOP = 832;
static constexpr std::uint32_t OFF_WEIGHT = 1024;
static constexpr std::uint32_t OFF_CONV_Q = 1088;
static constexpr std::uint32_t OFF_AFFINE_Q = 1408;
static constexpr std::uint32_t OFF_ADD_Q = 1472;
static constexpr std::uint32_t OFF_POOL_Q = 1536;

static int g_failures = 0;

static void set_byte(esp_int8::axi_vec_t& word, int lane, std::uint8_t value) {
    word.range(lane * 8 + 7, lane * 8) = value;
}

static std::uint8_t get_byte(const esp_int8::axi_vec_t& word, int lane) {
    return static_cast<std::uint8_t>(word.range(lane * 8 + 7, lane * 8));
}

static std::int8_t get_i8(const esp_int8::axi_vec_t& word, int lane) {
    return static_cast<std::int8_t>(get_byte(word, lane));
}

static void write_u8(esp_int8::axi_vec_t* mem, std::uint32_t byte_offset, std::uint8_t value) {
    set_byte(mem[byte_offset >> 5], static_cast<int>(byte_offset & 0x1fU), value);
}

static void write_i8(esp_int8::axi_vec_t* mem, std::uint32_t byte_offset, std::int8_t value) {
    write_u8(mem, byte_offset, static_cast<std::uint8_t>(value));
}

static void write_u16(esp_int8::axi_vec_t* mem, std::uint32_t byte_offset, std::uint16_t value) {
    write_u8(mem, byte_offset + 0, static_cast<std::uint8_t>(value & 0xffU));
    write_u8(mem, byte_offset + 1, static_cast<std::uint8_t>((value >> 8) & 0xffU));
}

static void write_u32(esp_int8::axi_vec_t* mem, std::uint32_t byte_offset, std::uint32_t value) {
    write_u8(mem, byte_offset + 0, static_cast<std::uint8_t>(value & 0xffU));
    write_u8(mem, byte_offset + 1, static_cast<std::uint8_t>((value >> 8) & 0xffU));
    write_u8(mem, byte_offset + 2, static_cast<std::uint8_t>((value >> 16) & 0xffU));
    write_u8(mem, byte_offset + 3, static_cast<std::uint8_t>((value >> 24) & 0xffU));
}

static void write_i32(esp_int8::axi_vec_t* mem, std::uint32_t byte_offset, std::int32_t value) {
    write_u32(mem, byte_offset, static_cast<std::uint32_t>(value));
}

static void write_header(esp_int8::axi_vec_t* mem) {
    write_u32(mem, 0, esp_int8::PARAM_BLOB_MAGIC);
    write_u32(mem, 4, esp_int8::PARAM_BLOB_VERSION);
    write_u32(mem, 8, esp_int8::TENSOR_DESC_COUNT);
    write_u32(mem, 12, 0);
    write_u32(mem, 16, 1);
    write_u32(mem, 20, 0);
    write_u32(mem, 24, 0);
    write_u32(mem, 28, 0);
    write_u32(mem, 32, 3);
    write_u32(mem, 36, 0);
    write_u32(mem, 40, OFF_TENSOR);
    write_u32(mem, 44, OFF_SCALE);
    write_u32(mem, 48, OFF_CONV);
    write_u32(mem, 52, OFF_AFFINE);
    write_u32(mem, 56, OFF_ADD);
    write_u32(mem, 60, OFF_POOL);
    write_u32(mem, 64, OFF_UOP);
    write_u32(mem, 68, OFF_WEIGHT);
    write_u32(mem, 72, OFF_CONV_Q);
    write_u32(mem, 76, OFF_AFFINE_Q);
    write_u32(mem, 80, OFF_ADD_Q);
    write_u32(mem, 84, OFF_POOL_Q);
}

static void write_tensor_desc(esp_int8::axi_vec_t* mem,
                              std::uint8_t tensor_id,
                              std::uint8_t bank_id,
                              std::uint32_t base_offset,
                              std::uint16_t h,
                              std::uint16_t w,
                              std::uint16_t c) {
    const std::uint32_t off = OFF_TENSOR + static_cast<std::uint32_t>(tensor_id) * 16U;
    write_u8(mem, off + 0, bank_id);
    write_u8(mem, off + 1, 1);
    write_u16(mem, off + 2, c);
    write_u32(mem, off + 4, base_offset);
    write_u16(mem, off + 8, h);
    write_u16(mem, off + 10, w);
    write_u16(mem, off + 12, c);
    write_u16(mem, off + 14, 0);
}

static void write_uop(esp_int8::axi_vec_t* mem,
                      std::uint32_t index,
                      std::uint8_t opcode,
                      std::uint8_t flags,
                      std::uint8_t src0,
                      std::uint8_t src1,
                      std::uint8_t dst,
                      std::uint8_t param_id,
                      std::uint8_t act_type,
                      std::uint16_t in_h,
                      std::uint16_t in_w,
                      std::uint16_t in_c,
                      std::uint16_t out_c,
                      std::uint8_t kernel,
                      std::uint8_t stride,
                      std::uint8_t dilation,
                      std::uint8_t padding) {
    const std::uint32_t off = OFF_UOP + index * 32U;
    write_u8(mem, off + 0, opcode);
    write_u8(mem, off + 1, flags);
    write_u8(mem, off + 2, src0);
    write_u8(mem, off + 3, src1);
    write_u8(mem, off + 4, dst);
    write_u8(mem, off + 5, param_id);
    write_u8(mem, off + 6, act_type);
    write_u8(mem, off + 7, 0);
    write_u16(mem, off + 8, in_h);
    write_u16(mem, off + 10, in_w);
    write_u16(mem, off + 12, in_c);
    write_u16(mem, off + 14, out_c);
    write_u8(mem, off + 16, kernel);
    write_u8(mem, off + 17, stride);
    write_u8(mem, off + 18, dilation);
    write_u8(mem, off + 19, padding);
    write_u16(mem, off + 20, 0);
    write_u16(mem, off + 22, out_c);
    write_u16(mem, off + 24, 0);
    write_u16(mem, off + 26, 0);
    write_u32(mem, off + 28, 0);
}

static void write_conv_qparam(esp_int8::axi_vec_t* mem) {
    for (int lane = 0; lane < 2; ++lane) {
        write_i32(mem, OFF_CONV_Q + lane * 4, 0);
        write_i32(mem, OFF_CONV_Q + 128 + lane * 4, 1);
        write_u8(mem, OFF_CONV_Q + 256 + lane, 0);
    }
}

static void write_weights(esp_int8::axi_vec_t* mem) {
    for (int k = 0; k < 9; ++k) {
        write_i8(mem, OFF_WEIGHT + k, 1);
    }
    write_i8(mem, OFF_WEIGHT + 9 + 4, 1);
}

static void build_param_blob(esp_int8::axi_vec_t* param) {
    for (int i = 0; i < 64; ++i) {
        param[i] = 0;
    }

    write_header(param);
    write_tensor_desc(param, esp_int8::TID_INPUT, esp_int8::BANK_FMEM0, 0x000000U, 3, 3, 1);
    write_tensor_desc(param, esp_int8::TID_OUT, esp_int8::BANK_FMEM0, 0x200000U, 3, 3, 2);

    write_u32(param, OFF_CONV + 0, OFF_WEIGHT);
    write_u32(param, OFF_CONV + 4, OFF_CONV_Q);
    write_u32(param, OFF_CONV + 8, OFF_CONV_Q);
    write_weights(param);
    write_conv_qparam(param);

    write_uop(param, 0, esp_int8::UOP_LOAD_FM, 0,
              esp_int8::TID_INVALID, esp_int8::TID_INVALID, esp_int8::TID_INPUT,
              0, esp_int8::ACT_NONE, 3, 3, 1, 1, 0, 0, 0, 0);
    write_uop(param, 1, esp_int8::UOP_CONV, 1U << esp_int8::UOP_FLAG_BIAS_EN,
              esp_int8::TID_INPUT, esp_int8::TID_INVALID, esp_int8::TID_OUT,
              0, esp_int8::ACT_NONE, 3, 3, 1, 2, 3, 1, 1, 1);
    write_uop(param, 2, esp_int8::UOP_END, 0,
              esp_int8::TID_INVALID, esp_int8::TID_INVALID, esp_int8::TID_INVALID,
              0, esp_int8::ACT_NONE, 0, 0, 0, 0, 0, 0, 0, 0);
}

int main() {
    static esp_int8::axi_vec_t frame_in[esp_int8::INPUT_FRAME_AXI_WORDS];
    static esp_int8::axi_vec_t frame_out[esp_int8::OUTPUT_FRAME_AXI_WORDS];
    static esp_int8::axi_vec_t param[64];
    volatile std::uint32_t dbg_status = 0;
    volatile std::uint32_t dbg_heartbeat = 0;
    volatile std::uint32_t dbg_act_words = 0;
    volatile std::uint32_t dbg_wgt_words = 0;
    volatile std::uint32_t dbg_psum_words = 0;
    volatile std::uint32_t dbg_out_words = 0;
    volatile std::uint32_t dbg_hw_version = 0;
    volatile std::uint32_t prof_uop_count = 0;
    volatile std::uint32_t prof_conv_count = 0;
    volatile std::uint32_t prof_win_read_ops = 0;
    volatile std::uint32_t prof_win_words = 0;
    volatile std::uint32_t prof_wgt_words = 0;
    volatile std::uint32_t prof_sa_mac_steps = 0;
    volatile std::uint32_t prof_psum_words = 0;
    volatile std::uint32_t prof_out_tiles = 0;
    volatile std::uint32_t prof_out_rmw_ops = 0;
    volatile std::uint32_t prof_model_cycles = 0;
    volatile std::uint32_t prof2_win_saved_reads = 0;
    volatile std::uint32_t prof2_win_actual_reads = 0;
    volatile std::uint32_t prof2_out_direct_words = 0;
    volatile std::uint32_t prof2_out_rmw_reads = 0;
    volatile std::uint32_t prof3_wgt_cycles = 0;
    volatile std::uint32_t prof3_win_cycles = 0;
    volatile std::uint32_t prof3_sa_cycles = 0;
    volatile std::uint32_t prof3_post_cycles = 0;
    volatile std::uint32_t prof3_write_cycles = 0;
    volatile std::uint32_t prof3_row_region_cycles = 0;

    for (int i = 0; i < esp_int8::INPUT_FRAME_AXI_WORDS; ++i) {
        frame_in[i] = 0;
    }
    for (int i = 0; i < esp_int8::OUTPUT_FRAME_AXI_WORDS; ++i) {
        frame_out[i] = 0;
    }
    build_param_blob(param);

    const std::int8_t input_values[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    for (int i = 0; i < 9; ++i) {
        write_i8(frame_in, static_cast<std::uint32_t>(i), input_values[i]);
    }

    espnet_encoder_int8_core(frame_in, frame_out, param,
                             esp_int8::MODE_INIT,
                             3,
                             dbg_status,
                             dbg_heartbeat,
                             dbg_act_words,
                             dbg_wgt_words,
                             dbg_psum_words,
                             dbg_out_words,
                             dbg_hw_version,
                             prof_uop_count,
                             prof_conv_count,
                             prof_win_read_ops,
                             prof_win_words,
                             prof_wgt_words,
                             prof_sa_mac_steps,
                             prof_psum_words,
                             prof_out_tiles,
                             prof_out_rmw_ops,
                             prof_model_cycles,
                             prof2_win_saved_reads,
                             prof2_win_actual_reads,
                             prof2_out_direct_words,
                             prof2_out_rmw_reads,
                             prof3_wgt_cycles,
                             prof3_win_cycles,
                             prof3_sa_cycles,
                             prof3_post_cycles,
                             prof3_write_cycles,
                             prof3_row_region_cycles);
    espnet_encoder_int8_core(frame_in, frame_out, param,
                             esp_int8::MODE_RUN,
                             3,
                             dbg_status,
                             dbg_heartbeat,
                             dbg_act_words,
                             dbg_wgt_words,
                             dbg_psum_words,
                             dbg_out_words,
                             dbg_hw_version,
                             prof_uop_count,
                             prof_conv_count,
                             prof_win_read_ops,
                             prof_win_words,
                             prof_wgt_words,
                             prof_sa_mac_steps,
                             prof_psum_words,
                             prof_out_tiles,
                             prof_out_rmw_ops,
                             prof_model_cycles,
                             prof2_win_saved_reads,
                             prof2_win_actual_reads,
                             prof2_out_direct_words,
                             prof2_out_rmw_reads,
                             prof3_wgt_cycles,
                             prof3_win_cycles,
                             prof3_sa_cycles,
                             prof3_post_cycles,
                             prof3_write_cycles,
                             prof3_row_region_cycles);

    const std::int8_t expected[18] = {
        12, 1, 21, 2, 16, 3,
        27, 4, 45, 5, 33, 6,
        24, 7, 39, 8, 28, 9,
    };

    for (int i = 0; i < 18; ++i) {
        const std::int8_t got = get_i8(frame_out[i >> 5], i & 0x1f);
        if (got != expected[i]) {
            std::printf("[FAIL] TOP_CONV byte=%d got=%d expected=%d\n",
                        i,
                        static_cast<int>(got),
                        static_cast<int>(expected[i]));
            ++g_failures;
        }
    }

    if (g_failures != 0) {
        std::printf("top_conv_dispatch_tb failed: %d failure(s)\n", g_failures);
        return 1;
    }

    std::printf("top_conv_dispatch_tb passed\n");
    return 0;
}
