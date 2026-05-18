#include "../include/npu_config.hpp"
#include "../include/npu_uop.hpp"

#include <cstdint>
#include <cstdio>

namespace esp_int8 {
bool on_chip_memory_get_tensor_desc(u8_t tensor_id, tensor_desc_t& desc);
bool on_chip_memory_read_tile(u8_t tensor_id, i32_t h, i32_t w, u16_t c_begin, u8_t valid_c, i8_t tile[TM]);
bool on_chip_memory_write_tile(u8_t tensor_id, u16_t h, u16_t w, u16_t c_begin, u8_t valid_c, const i8_t tile[TM]);
}  // namespace esp_int8

static int g_failures = 0;

static std::int8_t to_i8(esp_int8::i8_t x) {
    return static_cast<std::int8_t>(x.to_int());
}

static void expect(bool cond, const char* msg) {
    if (!cond) {
        std::printf("[FAIL] %s\n", msg);
        ++g_failures;
    }
}

static void expect_lane(const char* tag, int lane, esp_int8::i8_t got, std::int8_t expected) {
    const std::int8_t got_i8 = to_i8(got);
    if (got_i8 != expected) {
        std::printf("[FAIL] %s lane=%d got=%d expected=%d\n",
                    tag,
                    lane,
                    static_cast<int>(got_i8),
                    static_cast<int>(expected));
        ++g_failures;
    }
}

static void clear_tile(esp_int8::i8_t tile[esp_int8::TM]) {
    for (int i = 0; i < esp_int8::TM; ++i) {
        tile[i] = 0;
    }
}

static void fill_seq(esp_int8::i8_t tile[esp_int8::TM], int base) {
    for (int i = 0; i < esp_int8::TM; ++i) {
        tile[i] = static_cast<esp_int8::i8_t>(base + i);
    }
}

static bool write_channel_range(std::uint8_t tensor_id,
                                std::uint16_t h,
                                std::uint16_t w,
                                std::uint16_t c_begin,
                                std::uint16_t count,
                                int base) {
    esp_int8::i8_t tile[esp_int8::TM];
    std::uint16_t done = 0;
    while (done < count) {
        clear_tile(tile);
        const std::uint16_t n = (count - done > esp_int8::TM) ? esp_int8::TM : static_cast<std::uint16_t>(count - done);
        for (std::uint16_t i = 0; i < n; ++i) {
            tile[i] = static_cast<esp_int8::i8_t>(base + done + i);
        }
        if (!esp_int8::on_chip_memory_write_tile(tensor_id, h, w, c_begin + done, static_cast<esp_int8::u8_t>(n), tile)) {
            return false;
        }
        done += n;
    }
    return true;
}

static void test_static_descs() {
    esp_int8::tensor_desc_t desc;
    expect(esp_int8::on_chip_memory_get_tensor_desc(esp_int8::TID_B2_CAT, desc), "B2 desc available");
    expect(desc.bank_id == esp_int8::BANK_FMEM0 && desc.h == 128 && desc.w == 256 && desc.c == 131,
           "B2 desc matches frozen tensor table");
    expect(esp_int8::on_chip_memory_get_tensor_desc(esp_int8::TID_POOL_TMP, desc), "POOL_TMP desc available");
    expect(desc.bank_id == esp_int8::BANK_BRAM_SCR0 && desc.h == 256 && desc.w == 512 && desc.c == 3,
           "POOL_TMP desc matches frozen tensor table");
    expect(esp_int8::on_chip_memory_get_tensor_desc(esp_int8::TID_POOL2, desc), "POOL2 desc available");
    expect(desc.bank_id == esp_int8::BANK_BRAM_SCR1 && desc.h == 128 && desc.w == 256 && desc.c == 3,
           "POOL2 desc matches frozen tensor table");
}

static void test_b1_concat_offsets() {
    esp_int8::i8_t tile[esp_int8::TM];
    esp_int8::i8_t rd[esp_int8::TM];
    fill_seq(tile, 10);
    expect(esp_int8::on_chip_memory_write_tile(esp_int8::TID_B1_CAT, 2, 3, 0, 16, tile),
           "B1 level1 slice write c[0:15]");

    clear_tile(tile);
    tile[0] = -3;
    tile[1] = -2;
    tile[2] = -1;
    expect(esp_int8::on_chip_memory_write_tile(esp_int8::TID_B1_CAT, 2, 3, 16, 3, tile),
           "B1 pool slice write c[16:18]");

    expect(esp_int8::on_chip_memory_read_tile(esp_int8::TID_B1_CAT, 2, 3, 0, 19, rd),
           "B1 concat read c[0:18]");
    for (int i = 0; i < 16; ++i) {
        expect_lane("B1_CONV_SLICE", i, rd[i], static_cast<std::int8_t>(10 + i));
    }
    expect_lane("B1_POOL_SLICE", 16, rd[16], -3);
    expect_lane("B1_POOL_SLICE", 17, rd[17], -2);
    expect_lane("B1_POOL_SLICE", 18, rd[18], -1);
    expect_lane("B1_TAIL_ZERO", 19, rd[19], 0);
}

static void test_b2_concat_and_cin_tail() {
    esp_int8::i8_t rd[esp_int8::TM];
    expect(write_channel_range(esp_int8::TID_B2_CAT, 5, 7, 0, 64, -64), "B2 block slice write c[0:63]");
    expect(write_channel_range(esp_int8::TID_B2_CAT, 5, 7, 64, 64, 1), "B2 level2 slice write c[64:127]");
    expect(write_channel_range(esp_int8::TID_B2_CAT, 5, 7, 128, 3, 90), "B2 pool slice write c[128:130]");

    expect(esp_int8::on_chip_memory_read_tile(esp_int8::TID_B2_CAT, 5, 7, 0, 32, rd), "B2 read c[0:31]");
    expect_lane("B2_C0", 0, rd[0], -64);
    expect_lane("B2_C31", 31, rd[31], -33);

    expect(esp_int8::on_chip_memory_read_tile(esp_int8::TID_B2_CAT, 5, 7, 64, 32, rd), "B2 read c[64:95]");
    expect_lane("B2_C64", 0, rd[0], 1);
    expect_lane("B2_C95", 31, rd[31], 32);

    expect(esp_int8::on_chip_memory_read_tile(esp_int8::TID_B2_CAT, 5, 7, 128, 32, rd), "B2 Cin=131 tail read");
    expect_lane("B2_C128", 0, rd[0], 90);
    expect_lane("B2_C129", 1, rd[1], 91);
    expect_lane("B2_C130", 2, rd[2], 92);
    for (int i = 3; i < esp_int8::TM; ++i) {
        expect_lane("B2_CIN_TAIL_ZERO", i, rd[i], 0);
    }
}

static void test_padding_and_oob() {
    esp_int8::i8_t tile[esp_int8::TM];
    fill_seq(tile, 1);
    expect(esp_int8::on_chip_memory_read_tile(esp_int8::TID_INPUT, -1, 0, 0, 3, tile),
           "padding read top row succeeds");
    for (int i = 0; i < 3; ++i) {
        expect_lane("PAD_TOP_ZERO", i, tile[i], 0);
    }

    expect(esp_int8::on_chip_memory_read_tile(esp_int8::TID_INPUT, 512, 0, 0, 3, tile),
           "padding read bottom row succeeds");
    for (int i = 0; i < 3; ++i) {
        expect_lane("PAD_BOTTOM_ZERO", i, tile[i], 0);
    }

    clear_tile(tile);
    expect(!esp_int8::on_chip_memory_write_tile(esp_int8::TID_B1_CAT, 256, 0, 0, 1, tile),
           "write rejects h out of range");
    expect(!esp_int8::on_chip_memory_write_tile(esp_int8::TID_B1_CAT, 0, 0, 18, 2, tile),
           "write rejects channel overrun");
}

static void test_scratch_banks() {
    esp_int8::i8_t tile[esp_int8::TM];
    esp_int8::i8_t rd[esp_int8::TM];
    clear_tile(tile);
    tile[0] = 11;
    tile[1] = 12;
    tile[2] = 13;
    expect(esp_int8::on_chip_memory_write_tile(esp_int8::TID_POOL_TMP, 255, 511, 0, 3, tile),
           "POOL_TMP scratch write last pixel");
    expect(esp_int8::on_chip_memory_read_tile(esp_int8::TID_POOL_TMP, 255, 511, 0, 3, rd),
           "POOL_TMP scratch read last pixel");
    expect_lane("POOL_TMP_C0", 0, rd[0], 11);
    expect_lane("POOL_TMP_C1", 1, rd[1], 12);
    expect_lane("POOL_TMP_C2", 2, rd[2], 13);

    tile[0] = -11;
    tile[1] = -12;
    tile[2] = -13;
    expect(esp_int8::on_chip_memory_write_tile(esp_int8::TID_POOL2, 127, 255, 0, 3, tile),
           "POOL2 scratch write last pixel");
    expect(esp_int8::on_chip_memory_read_tile(esp_int8::TID_POOL2, 127, 255, 0, 3, rd),
           "POOL2 scratch read last pixel");
    expect_lane("POOL2_C0", 0, rd[0], -11);
    expect_lane("POOL2_C1", 1, rd[1], -12);
    expect_lane("POOL2_C2", 2, rd[2], -13);
}

int main() {
    test_static_descs();
    test_b1_concat_offsets();
    test_b2_concat_and_cin_tail();
    test_padding_and_oob();
    test_scratch_banks();

    if (g_failures != 0) {
        std::printf("memory_tile_tb failed: %d failure(s)\n", g_failures);
        return 1;
    }

    std::printf("memory_tile_tb passed\n");
    return 0;
}
