#include "../include/npu_config.hpp"
#include "../include/npu_uop.hpp"

#include <cstdint>
#include <cstdio>

void espnet_encoder_int8_core(const esp_int8::axi_vec_t* gmem_frame_in,
                              esp_int8::axi_vec_t* gmem_frame_out,
                              const esp_int8::axi_vec_t* gmem_param,
                              std::uint32_t mode,
                              std::uint32_t uop_count);

static constexpr std::uint32_t OFF_TENSOR = 128;
static constexpr std::uint32_t OFF_SCALE = 512;
static constexpr std::uint32_t OFF_CONV = 576;
static constexpr std::uint32_t OFF_AFFINE = 640;
static constexpr std::uint32_t OFF_ADD = 704;
static constexpr std::uint32_t OFF_POOL = 768;
static constexpr std::uint32_t OFF_UOP = 832;
static constexpr std::uint32_t OFF_WEIGHT = 1024;
static constexpr std::uint32_t OFF_CONV_Q = 1024;
static constexpr std::uint32_t OFF_AFFINE_Q = 1088;
static constexpr std::uint32_t OFF_ADD_Q = 1152;
static constexpr std::uint32_t OFF_POOL_Q = 1216;

static void set_byte(esp_int8::axi_vec_t& word, int lane, std::uint8_t value) {
    word.range(lane * 8 + 7, lane * 8) = value;
}

static std::uint8_t get_byte(const esp_int8::axi_vec_t& word, int lane) {
    return static_cast<std::uint8_t>(word.range(lane * 8 + 7, lane * 8));
}

static void write_u8(esp_int8::axi_vec_t* mem, std::uint32_t byte_offset, std::uint8_t value) {
    set_byte(mem[byte_offset >> 5], static_cast<int>(byte_offset & 0x1fU), value);
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

static void write_header(esp_int8::axi_vec_t* mem) {
    write_u32(mem, 0, esp_int8::PARAM_BLOB_MAGIC);
    write_u32(mem, 4, esp_int8::PARAM_BLOB_VERSION);
    write_u32(mem, 8, 0);
    write_u32(mem, 12, 0);
    write_u32(mem, 16, 0);
    write_u32(mem, 20, 0);
    write_u32(mem, 24, 0);
    write_u32(mem, 28, 0);
    write_u32(mem, 32, 1);
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

static void write_end_uop(esp_int8::axi_vec_t* mem) {
    write_u8(mem, OFF_UOP + 0, esp_int8::UOP_END);
    write_u8(mem, OFF_UOP + 1, 0);
    write_u8(mem, OFF_UOP + 2, esp_int8::TID_INVALID);
    write_u8(mem, OFF_UOP + 3, esp_int8::TID_INVALID);
    write_u8(mem, OFF_UOP + 4, esp_int8::TID_INVALID);
    write_u8(mem, OFF_UOP + 5, 0);
    write_u8(mem, OFF_UOP + 6, esp_int8::ACT_NONE);
    write_u8(mem, OFF_UOP + 7, 0);
    write_u16(mem, OFF_UOP + 8, 0);
    write_u16(mem, OFF_UOP + 10, 0);
    write_u16(mem, OFF_UOP + 12, 0);
    write_u16(mem, OFF_UOP + 14, 0);
    write_u8(mem, OFF_UOP + 16, 0);
    write_u8(mem, OFF_UOP + 17, 0);
    write_u8(mem, OFF_UOP + 18, 0);
    write_u8(mem, OFF_UOP + 19, 0);
    write_u16(mem, OFF_UOP + 20, 0);
    write_u16(mem, OFF_UOP + 22, 0);
    write_u16(mem, OFF_UOP + 24, 0);
    write_u16(mem, OFF_UOP + 26, 0);
    write_u32(mem, OFF_UOP + 28, 0);
}

static void build_param_blob(esp_int8::axi_vec_t* param) {
    for (int i = 0; i < 64; ++i) {
        param[i] = 0;
    }
    write_header(param);
    write_end_uop(param);
}

int main() {
    static esp_int8::axi_vec_t frame_in[esp_int8::INPUT_FRAME_AXI_WORDS];
    static esp_int8::axi_vec_t frame_out[esp_int8::OUTPUT_FRAME_AXI_WORDS];
    static esp_int8::axi_vec_t param[64];

    for (int i = 0; i < esp_int8::INPUT_FRAME_AXI_WORDS; ++i) {
        frame_in[i] = 0;
    }
    for (int i = 0; i < esp_int8::OUTPUT_FRAME_AXI_WORDS; ++i) {
        frame_out[i] = 0;
    }
    build_param_blob(param);

    espnet_encoder_int8_core(frame_in, frame_out, param,
                             esp_int8::MODE_INIT,
                             1);
    espnet_encoder_int8_core(frame_in, frame_out, param,
                             esp_int8::MODE_RUN,
                             1);

    for (int i = 0; i < 16; ++i) {
        if (get_byte(frame_out[0], i) != 0) {
            std::printf("[FAIL] TOP_SMOKE byte=%d nonzero\n", i);
            return 1;
        }
    }

    std::printf("top_tb passed\n");
    return 0;
}
