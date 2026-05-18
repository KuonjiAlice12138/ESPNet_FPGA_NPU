#include "../include/npu_config.hpp"
#include "../include/npu_q.hpp"
#include "../include/npu_uop.hpp"

#include <cstdint>
#include <cstdio>

namespace esp_int8 {
bool on_chip_memory_read_tile(const tensor_desc_t& desc,
                              i32_t h,
                              i32_t w,
                              u16_t c_begin,
                              u8_t valid_c,
                              i8_t tile[TM]);
bool on_chip_memory_write_tile(const tensor_desc_t& desc,
                               u16_t h,
                               u16_t w,
                               u16_t c_begin,
                               u8_t valid_c,
                               const i8_t tile[TM]);
bool concat_writer(const tensor_desc_t& src,
                   const tensor_desc_t& dst,
                   u16_t c_offset,
                   u16_t valid_c);
bool avgpool_unit_checked(const tensor_desc_t& src,
                          const tensor_desc_t& dst,
                          const pool_q_t& qparam,
                          i8_t* fmbuf_base);
}  // namespace esp_int8

static int g_failures = 0;

static esp_int8::u8_t u8(unsigned value) {
    return esp_int8::u8_t(value);
}

static esp_int8::u16_t u16(unsigned value) {
    return esp_int8::u16_t(value);
}

static esp_int8::u32_t u32(unsigned value) {
    return esp_int8::u32_t(value);
}

static esp_int8::i8_t i8(int value) {
    return esp_int8::i8_t(value);
}

static int to_int(esp_int8::i8_t value) {
    return value.to_int();
}

static esp_int8::tensor_desc_t make_desc(unsigned bank,
                                         unsigned base,
                                         unsigned h,
                                         unsigned w,
                                         unsigned c,
                                         unsigned phys_c,
                                         unsigned c_offset) {
    esp_int8::tensor_desc_t desc;
    desc.bank_id = u8(bank);
    desc.elem_bytes = 1;
    desc.reserved0 = u16(phys_c);
    desc.base_offset = u32(base);
    desc.h = u16(h);
    desc.w = u16(w);
    desc.c = u16(c);
    desc.reserved1 = u16(c_offset);
    return desc;
}

static int pattern(int h, int w, int c) {
    return ((h * 31 + w * 17 + c * 7 + 19) % 121) - 60;
}

static bool write_pixel(const esp_int8::tensor_desc_t& desc, int h, int w, int c, int value) {
    esp_int8::i8_t tile[esp_int8::TM];
    for (int lane = 0; lane < esp_int8::TM; ++lane) {
        tile[lane] = 0;
    }
    tile[0] = i8(value);
    return esp_int8::on_chip_memory_write_tile(desc,
                                               u16(h),
                                               u16(w),
                                               u16(c),
                                               u8(1),
                                               tile);
}

static bool read_pixel(const esp_int8::tensor_desc_t& desc, int h, int w, int c, int& value) {
    esp_int8::i8_t tile[esp_int8::TM];
    if (!esp_int8::on_chip_memory_read_tile(desc,
                                            esp_int8::i32_t(h),
                                            esp_int8::i32_t(w),
                                            u16(c),
                                            u8(1),
                                            tile)) {
        return false;
    }
    value = to_int(tile[0]);
    return true;
}

static bool fill_tensor(const esp_int8::tensor_desc_t& desc, int value_offset) {
    for (int h = 0; h < static_cast<int>(desc.h.to_uint()); ++h) {
        for (int w = 0; w < static_cast<int>(desc.w.to_uint()); ++w) {
            for (int c = 0; c < static_cast<int>(desc.c.to_uint()); ++c) {
                if (!write_pixel(desc, h, w, c, pattern(h, w, c) + value_offset)) {
                    return false;
                }
            }
        }
    }
    return true;
}

static void fail_at(const char* tag, int h, int w, int c, int got, int expected) {
    if (g_failures < 40) {
        std::printf("[FAIL] %s h=%d w=%d c=%d got=%d expected=%d\n",
                    tag,
                    h,
                    w,
                    c,
                    got,
                    expected);
    }
    ++g_failures;
}

static void run_concat_copy_case() {
    const esp_int8::tensor_desc_t src =
        make_desc(esp_int8::BANK_FMEM0, 0x500000U, 2, 3, 3, 3, 0);
    const esp_int8::tensor_desc_t dst =
        make_desc(esp_int8::BANK_FMEM0, 0x501000U, 2, 3, 7, 7, 0);

    for (int h = 0; h < 2; ++h) {
        for (int w = 0; w < 3; ++w) {
            for (int c = 0; c < 7; ++c) {
                if (!write_pixel(dst, h, w, c, -99)) {
                    std::printf("[FAIL] CONCAT_COPY failed to initialize dst\n");
                    ++g_failures;
                    return;
                }
            }
        }
    }

    if (!fill_tensor(src, 0)) {
        std::printf("[FAIL] CONCAT_COPY failed to fill src\n");
        ++g_failures;
        return;
    }

    if (!esp_int8::concat_writer(src, dst, u16(4), u16(3))) {
        std::printf("[FAIL] CONCAT_COPY returned false\n");
        ++g_failures;
        return;
    }

    for (int h = 0; h < 2; ++h) {
        for (int w = 0; w < 3; ++w) {
            for (int c = 0; c < 7; ++c) {
                int got = 0;
                if (!read_pixel(dst, h, w, c, got)) {
                    std::printf("[FAIL] CONCAT_COPY failed to read dst\n");
                    ++g_failures;
                    return;
                }
                const int expected = (c < 4) ? -99 : pattern(h, w, c - 4);
                if (got != expected) {
                    fail_at("CONCAT_COPY", h, w, c, got, expected);
                }
            }
        }
    }
}

static void run_concat_view_case() {
    const esp_int8::tensor_desc_t src_view =
        make_desc(esp_int8::BANK_FMEM0, 0x502000U, 2, 2, 3, 8, 4);
    const esp_int8::tensor_desc_t dst_container =
        make_desc(esp_int8::BANK_FMEM0, 0x502000U, 2, 2, 8, 8, 0);

    if (!fill_tensor(src_view, 2)) {
        std::printf("[FAIL] CONCAT_VIEW failed to fill src view\n");
        ++g_failures;
        return;
    }

    if (!esp_int8::concat_writer(src_view, dst_container, u16(4), u16(3))) {
        std::printf("[FAIL] CONCAT_VIEW returned false\n");
        ++g_failures;
        return;
    }

    for (int h = 0; h < 2; ++h) {
        for (int w = 0; w < 2; ++w) {
            for (int c = 4; c < 7; ++c) {
                int got = 0;
                if (!read_pixel(dst_container, h, w, c, got)) {
                    std::printf("[FAIL] CONCAT_VIEW failed to read dst\n");
                    ++g_failures;
                    return;
                }
                const int expected = pattern(h, w, c - 4) + 2;
                if (got != expected) {
                    fail_at("CONCAT_VIEW", h, w, c, got, expected);
                }
            }
        }
    }
}

static int round_div9_ref(int x) {
    return (x >= 0) ? ((x + 4) / 9) : ((x - 4) / 9);
}

static int clamp_i8_ref(int x) {
    if (x > 127) {
        return 127;
    }
    if (x < -128) {
        return -128;
    }
    return x;
}

static int round_shift_ref(long long x, unsigned shift) {
    if (shift == 0U) {
        return static_cast<int>(x);
    }
    const long long bias = 1LL << (shift - 1U);
    return static_cast<int>((x >= 0) ? ((x + bias) >> shift) : ((x - bias) >> shift));
}

static int pool_ref_at(int oh, int ow, int c, bool requant) {
    int sum = 0;
    for (int kh = 0; kh < 3; ++kh) {
        for (int kw = 0; kw < 3; ++kw) {
            const int ih = oh * 2 + kh - 1;
            const int iw = ow * 2 + kw - 1;
            if (ih >= 0 && iw >= 0 && ih < 5 && iw < 5) {
                sum += pattern(ih, iw, c);
            }
        }
    }

    const int avg = round_div9_ref(sum);
    if (!requant) {
        return clamp_i8_ref(avg);
    }
    return clamp_i8_ref(round_shift_ref(static_cast<long long>(avg) * 3LL, 1));
}

static void run_pool_case(const char* tag, bool requant, unsigned src_base, unsigned dst_base) {
    const esp_int8::tensor_desc_t src =
        make_desc(esp_int8::BANK_FMEM0, src_base, 5, 5, 3, 3, 0);
    const esp_int8::tensor_desc_t dst =
        make_desc(esp_int8::BANK_FMEM0, dst_base, 3, 3, 3, 3, 0);

    if (!fill_tensor(src, 0)) {
        std::printf("[FAIL] %s failed to fill src\n", tag);
        ++g_failures;
        return;
    }

    esp_int8::pool_q_t qparam;
    qparam.kernel = u8(3);
    qparam.stride = u8(2);
    qparam.same_scale = u8(requant ? 0U : 1U);
    qparam.act_type = u8(esp_int8::ACT_NONE);
    qparam.src_scale_id = 0;
    qparam.dst_scale_id = 1;
    qparam.mult = esp_int8::i32_t(3);
    qparam.shift = u8(1);
    for (int i = 0; i < 11; ++i) {
        qparam.reserved[i] = 0;
    }

    if (!esp_int8::avgpool_unit_checked(src, dst, qparam, 0)) {
        std::printf("[FAIL] %s returned false\n", tag);
        ++g_failures;
        return;
    }

    for (int h = 0; h < 3; ++h) {
        for (int w = 0; w < 3; ++w) {
            for (int c = 0; c < 3; ++c) {
                int got = 0;
                if (!read_pixel(dst, h, w, c, got)) {
                    std::printf("[FAIL] %s failed to read dst\n", tag);
                    ++g_failures;
                    return;
                }
                const int expected = pool_ref_at(h, w, c, requant);
                if (got != expected) {
                    fail_at(tag, h, w, c, got, expected);
                }
            }
        }
    }
}

int main() {
    run_concat_copy_case();
    run_concat_view_case();
    run_pool_case("POOL_SAME_SCALE", false, 0x503000U, 0x504000U);
    run_pool_case("POOL_REQUANT", true, 0x505000U, 0x506000U);

    if (g_failures != 0) {
        std::printf("pool_concat_domain_tb failed: %d failure(s)\n", g_failures);
        return 1;
    }

    std::printf("pool_concat_domain_tb passed\n");
    return 0;
}
