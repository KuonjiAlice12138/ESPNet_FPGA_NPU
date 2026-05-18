#include "../include/npu_config.hpp"
#include "../include/npu_uop.hpp"

#include <cstdio>
#include <cstdint>

namespace esp_int8 {
void param_dma_init(const axi_vec_t* gmem_param);
bool param_dma_ready();
error_code_t param_dma_error();
bool param_dma_get_header(param_blob_header_t& header);
bool param_dma_get_tensor_desc(u8_t tensor_id, tensor_desc_t& desc);
bool param_dma_get_uop(u16_t uop_id, uop_t& uop);
bool param_dma_get_pool_qparam(u8_t param_id, pool_q_t& qparam);

void instruction_fetch_decode(const axi_vec_t* gmem_param, u32_t uop_count);
error_code_t if_dec_error();
u16_t if_dec_current_uop_id();
bool if_dec_get_current_uop(uop_t& uop);
}  // namespace esp_int8

static constexpr std::uint32_t OFF_TENSOR = 128;
static constexpr std::uint32_t OFF_SCALE = 512;
static constexpr std::uint32_t OFF_CONV = 576;
static constexpr std::uint32_t OFF_AFFINE = 640;
static constexpr std::uint32_t OFF_ADD = 704;
static constexpr std::uint32_t OFF_POOL = 768;
static constexpr std::uint32_t OFF_UOP = 832;
static constexpr std::uint32_t OFF_WEIGHT = 1024;
static constexpr std::uint32_t OFF_CONV_Q = 1088;
static constexpr std::uint32_t OFF_AFFINE_Q = 1152;
static constexpr std::uint32_t OFF_ADD_Q = 1216;
static constexpr std::uint32_t OFF_POOL_Q = 1280;

static int g_failures = 0;

static void expect(bool cond, const char* msg) {
    if (!cond) {
        std::printf("[FAIL] %s\n", msg);
        ++g_failures;
    }
}

static void write_u8(esp_int8::axi_vec_t* mem, std::uint32_t byte_offset, std::uint8_t value) {
    const std::uint32_t word_idx = byte_offset >> 5;
    const int lane = static_cast<int>(byte_offset & 0x1fU);
    mem[word_idx].range(lane * 8 + 7, lane * 8) = value;
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
    write_u32(mem, 8, esp_int8::TENSOR_DESC_COUNT);
    write_u32(mem, 12, 1);
    write_u32(mem, 16, 0);
    write_u32(mem, 20, 0);
    write_u32(mem, 24, 0);
    write_u32(mem, 28, 1);
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
    write_u16(mem, off + 2, 0);
    write_u32(mem, off + 4, base_offset);
    write_u16(mem, off + 8, h);
    write_u16(mem, off + 10, w);
    write_u16(mem, off + 12, c);
    write_u16(mem, off + 14, 0);
}

static void write_uop(esp_int8::axi_vec_t* mem,
                      std::uint32_t index,
                      std::uint8_t opcode,
                      std::uint8_t src0,
                      std::uint8_t src1,
                      std::uint8_t dst,
                      std::uint16_t h,
                      std::uint16_t w,
                      std::uint16_t c) {
    const std::uint32_t off = OFF_UOP + index * 32U;
    write_u8(mem, off + 0, opcode);
    write_u8(mem, off + 1, 0);
    write_u8(mem, off + 2, src0);
    write_u8(mem, off + 3, src1);
    write_u8(mem, off + 4, dst);
    write_u8(mem, off + 5, 0);
    write_u8(mem, off + 6, esp_int8::ACT_NONE);
    write_u8(mem, off + 7, 0);
    write_u16(mem, off + 8, h);
    write_u16(mem, off + 10, w);
    write_u16(mem, off + 12, c);
    write_u16(mem, off + 14, c);
    write_u8(mem, off + 16, 0);
    write_u8(mem, off + 17, 0);
    write_u8(mem, off + 18, 0);
    write_u8(mem, off + 19, 0);
    write_u16(mem, off + 20, 0);
    write_u16(mem, off + 22, c);
    write_u16(mem, off + 24, 0);
    write_u16(mem, off + 26, 0);
    write_u32(mem, off + 28, 0);
}

static void write_pool_desc(esp_int8::axi_vec_t* mem) {
    write_u32(mem, OFF_POOL + 0, OFF_POOL_Q);
    write_u32(mem, OFF_POOL + 4, 0);
    write_u32(mem, OFF_POOL + 8, 0);
    write_u32(mem, OFF_POOL + 12, 0);
}

static void write_pool_qparam(esp_int8::axi_vec_t* mem) {
    write_u8(mem, OFF_POOL_Q + 0, 3);
    write_u8(mem, OFF_POOL_Q + 1, 2);
    write_u8(mem, OFF_POOL_Q + 2, 0);
    write_u8(mem, OFF_POOL_Q + 3, esp_int8::ACT_NONE);
    write_u32(mem, OFF_POOL_Q + 4, 5);
    write_u32(mem, OFF_POOL_Q + 8, 6);
    write_u32(mem, OFF_POOL_Q + 12, 12345);
    write_u8(mem, OFF_POOL_Q + 16, 13);
}

static void build_blob(esp_int8::axi_vec_t* mem) {
    for (int i = 0; i < 64; ++i) {
        mem[i] = 0;
    }

    write_header(mem);
    write_tensor_desc(mem, esp_int8::TID_INPUT, esp_int8::BANK_FMEM0, 0x000000, 512, 1024, 3);
    write_tensor_desc(mem, esp_int8::TID_OUT, esp_int8::BANK_FMEM0, 0x200000, 64, 128, 2);
    write_u32(mem, OFF_SCALE + 0, 1);
    write_u8(mem, OFF_SCALE + 4, 0);
    write_pool_desc(mem);
    write_pool_qparam(mem);

    write_uop(mem, 0, esp_int8::UOP_LOAD_FM,
              esp_int8::TID_INVALID, esp_int8::TID_INVALID, esp_int8::TID_INPUT,
              512, 1024, 3);
    write_uop(mem, 1, esp_int8::UOP_STORE,
              esp_int8::TID_OUT, esp_int8::TID_INVALID, esp_int8::TID_INVALID,
              64, 128, 2);
    write_uop(mem, 2, esp_int8::UOP_END,
              esp_int8::TID_INVALID, esp_int8::TID_INVALID, esp_int8::TID_INVALID,
              0, 0, 0);
}

int main() {
    static esp_int8::axi_vec_t param[64];
    build_blob(param);

    esp_int8::param_dma_init(param);
    expect(esp_int8::param_dma_ready(), "param_dma ready after valid blob");
    expect(esp_int8::param_dma_error() == esp_int8::ERR_NONE, "param_dma error is none");

    esp_int8::param_blob_header_t header;
    expect(esp_int8::param_dma_get_header(header), "header available");
    expect(header.uop_count == 3, "uop_count loaded");
    expect(header.tensor_desc_count == esp_int8::TENSOR_DESC_COUNT, "tensor desc count loaded");
    expect(header.pool_desc_count == 1, "pool desc count loaded");

    esp_int8::tensor_desc_t input_desc;
    expect(esp_int8::param_dma_get_tensor_desc(esp_int8::TID_INPUT, input_desc), "input tensor desc available");
    expect(input_desc.h == 512 && input_desc.w == 1024 && input_desc.c == 3, "input tensor shape decoded");

    esp_int8::tensor_desc_t out_desc;
    expect(esp_int8::param_dma_get_tensor_desc(esp_int8::TID_OUT, out_desc), "output tensor desc available");
    expect(out_desc.h == 64 && out_desc.w == 128 && out_desc.c == 2, "output tensor shape decoded");

    esp_int8::uop_t uop;
    expect(esp_int8::param_dma_get_uop(0, uop), "uop 0 available");
    expect(uop.opcode == esp_int8::UOP_LOAD_FM && uop.dst_tensor == esp_int8::TID_INPUT, "uop 0 decoded as LOAD_FM");
    expect(esp_int8::param_dma_get_uop(1, uop), "uop 1 available");
    expect(uop.opcode == esp_int8::UOP_STORE && uop.src0_tensor == esp_int8::TID_OUT, "uop 1 decoded as STORE");

    esp_int8::pool_q_t pool_q;
    expect(esp_int8::param_dma_get_pool_qparam(0, pool_q), "pool qparam 0 available");
    expect(pool_q.kernel == 3 && pool_q.stride == 2, "pool qparam kernel/stride decoded");
    expect(pool_q.same_scale == 0 && pool_q.mult == 12345 && pool_q.shift == 13, "pool qparam scale fields decoded");

    esp_int8::instruction_fetch_decode(param, 3);
    expect(esp_int8::if_dec_error() == esp_int8::ERR_NONE, "if_dec accepts valid uop stream");
    expect(esp_int8::if_dec_current_uop_id() == 2, "if_dec stops at END");
    expect(esp_int8::if_dec_get_current_uop(uop), "if_dec current uop available");
    expect(uop.opcode == esp_int8::UOP_END, "if_dec current uop is END");

    esp_int8::instruction_fetch_decode(param, 2);
    expect(esp_int8::if_dec_error() == esp_int8::ERR_UOP_DECODE, "if_dec rejects mismatched uop_count");

    write_u8(param, OFF_UOP + 32, 0x7e);
    esp_int8::param_dma_init(param);
    esp_int8::instruction_fetch_decode(param, 3);
    expect(esp_int8::if_dec_error() == esp_int8::ERR_UNSUPPORTED_OPCODE, "if_dec rejects unsupported opcode");
    expect(esp_int8::if_dec_current_uop_id() == 1, "if_dec reports bad uop id");

    if (g_failures != 0) {
        std::printf("control_dma_tb failed: %d failure(s)\n", g_failures);
        return 1;
    }

    std::printf("control_dma_tb passed\n");
    return 0;
}
