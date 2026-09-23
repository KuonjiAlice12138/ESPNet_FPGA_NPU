#include "../include/npu_config.hpp"
#include "../include/npu_types.hpp"

#include <cstdio>

namespace esp_int8 {
bool on_chip_memory_read_packed_tile(const tensor_desc_t& desc,
                                     i32_t h,
                                     i32_t w,
                                     u16_t c_begin,
                                     u8_t valid_c,
                                     axi_vec_t& packed);
bool on_chip_memory_write_fmbuf_abs_word(u8_t bank_id,
                                         u32_t byte_offset,
                                         axi_vec_t packed);
}  // namespace esp_int8

namespace {

int g_failures = 0;

void expect(bool condition, const char* tag) {
    if (!condition) {
        std::printf("[FAIL] %s\n", tag);
        ++g_failures;
    }
}

esp_int8::axi_vec_t sequential_word(unsigned first) {
    esp_int8::axi_vec_t word = 0;
    for (int lane = 0; lane < esp_int8::AXI_WORD_BYTES; ++lane) {
        word.range(lane * 8 + 7, lane * 8) =
            first + static_cast<unsigned>(lane);
    }
    return word;
}

void expect_packed_range(const char* tag,
                         const esp_int8::axi_vec_t& packed,
                         unsigned first,
                         unsigned count) {
    for (int lane = 0; lane < esp_int8::AXI_WORD_BYTES; ++lane) {
        const unsigned got = packed.range(lane * 8 + 7, lane * 8).to_uint();
        const unsigned expected = static_cast<unsigned>(lane) < count
                                      ? first + static_cast<unsigned>(lane)
                                      : 0U;
        if (got != expected) {
            std::printf("[FAIL] %s lane=%d got=%u expected=%u\n",
                        tag, lane, got, expected);
            ++g_failures;
        }
    }
}

void run_read_case(const esp_int8::tensor_desc_t& desc,
                   const char* tag,
                   unsigned c_begin,
                   unsigned valid_c,
                   unsigned expected_first,
                   unsigned expected_count) {
    esp_int8::axi_vec_t packed = ~esp_int8::axi_vec_t(0);
    expect(esp_int8::on_chip_memory_read_packed_tile(
               desc, 0, 0, c_begin, valid_c, packed),
           tag);
    expect_packed_range(tag, packed, expected_first, expected_count);
}

}  // namespace

int main() {
    using namespace esp_int8;
    const u8_t bank = static_cast<u8_t>(static_cast<unsigned>(BANK_FMEM0));
    expect(on_chip_memory_write_fmbuf_abs_word(
               bank, static_cast<u32_t>(0), sequential_word(1)),
           "word0 setup");
    expect(on_chip_memory_write_fmbuf_abs_word(
               bank, static_cast<u32_t>(AXI_WORD_BYTES), sequential_word(33)),
           "word1 setup");

    tensor_desc_t desc = {};
    desc.bank_id = bank;
    desc.elem_bytes = 1;
    desc.reserved0 = 64;
    desc.base_offset = 0;
    desc.h = 1;
    desc.w = 1;
    desc.c = 64;

    run_read_case(desc, "aligned full word", 0, 32, 1, 32);
    run_read_case(desc, "same-word 31-byte tail", 1, 31, 2, 31);
    run_read_case(desc, "one-byte boundary", 31, 1, 32, 1);
    run_read_case(desc, "cross-word full read", 31, 32, 32, 32);
    run_read_case(desc, "second-word 31-byte tail", 33, 31, 34, 31);
    run_read_case(desc, "empty channel range", 64, 32, 0, 0);

    if (g_failures != 0) {
        std::printf("memory_packed_reader_tb failed: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("memory_packed_reader_tb passed\n");
    return 0;
}
