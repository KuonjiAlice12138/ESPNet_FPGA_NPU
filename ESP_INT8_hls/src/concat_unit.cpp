#include "../include/npu_config.hpp"
#include "../include/npu_types.hpp"

namespace esp_int8 {

bool on_chip_memory_read_packed_tile(const tensor_desc_t& desc,
                                     i32_t h,
                                     i32_t w,
                                     u16_t c_begin,
                                     u8_t valid_c,
                                     act_vec_t& packed);
bool on_chip_memory_read_fmbuf_abs_word(u8_t bank_id,
                                        u32_t byte_offset,
                                        axi_vec_t& packed);
bool on_chip_memory_write_fmbuf_abs_word(u8_t bank_id,
                                         u32_t byte_offset,
                                         axi_vec_t packed);
bool on_chip_memory_read_pool2_abs_word(u32_t byte_offset,
                                        axi_vec_t& packed);
bool on_chip_memory_write_pool2_abs_word(u32_t byte_offset,
                                         axi_vec_t packed);

static u16_t desc_phys_c(const tensor_desc_t& desc) {
#pragma HLS INLINE
    return (desc.reserved0.to_uint() == 0U) ? desc.c : desc.reserved0;
}

static u16_t desc_c_offset(const tensor_desc_t& desc) {
#pragma HLS INLINE
    return desc.reserved1;
}

static u8_t concat_lanes(u16_t remaining_c) {
#pragma HLS INLINE
    const unsigned rem = remaining_c.to_uint();
    return static_cast<u8_t>((rem > static_cast<unsigned>(TM)) ? TM : rem);
}

static bool same_shape_2d(const tensor_desc_t& src, const tensor_desc_t& dst) {
#pragma HLS INLINE
    return src.h.to_uint() == dst.h.to_uint() && src.w.to_uint() == dst.w.to_uint();
}

static bool slice_already_in_place(const tensor_desc_t& src,
                                   const tensor_desc_t& dst,
                                   u16_t c_offset,
                                   u16_t valid_c) {
#pragma HLS INLINE
    const unsigned dst_begin = dst.reserved1.to_uint() + c_offset.to_uint();
    return src.bank_id.to_uint() == dst.bank_id.to_uint() &&
           src.base_offset.to_uint() == dst.base_offset.to_uint() &&
           same_shape_2d(src, dst) &&
           desc_phys_c(src).to_uint() == desc_phys_c(dst).to_uint() &&
           desc_c_offset(src).to_uint() == dst_begin &&
           src.c.to_uint() >= valid_c.to_uint() &&
           dst.c.to_uint() >= c_offset.to_uint() + valid_c.to_uint();
}

static u32_t concat_tensor_byte_offset(const tensor_desc_t& desc,
                                       u16_t h,
                                       u16_t w,
                                       u16_t c_begin) {
#pragma HLS INLINE
    return static_cast<u32_t>((static_cast<u32_t>(h) * desc.w + w) * desc_phys_c(desc) +
                              desc_c_offset(desc) + c_begin);
}

static bool concat_read_aligned_abs_word(const tensor_desc_t& desc,
                                         u32_t byte_offset,
                                         axi_vec_t& word) {
#pragma HLS INLINE
    if (desc.bank_id.to_uint() == static_cast<unsigned>(BANK_BRAM_SCR1)) {
        return on_chip_memory_read_pool2_abs_word(byte_offset, word);
    }
    return on_chip_memory_read_fmbuf_abs_word(desc.bank_id, byte_offset, word);
}

static bool concat_write_aligned_abs_word(const tensor_desc_t& desc,
                                          u32_t byte_offset,
                                          axi_vec_t word) {
#pragma HLS INLINE
    if (desc.bank_id.to_uint() == static_cast<unsigned>(BANK_BRAM_SCR1)) {
        return on_chip_memory_write_pool2_abs_word(byte_offset, word);
    }
    return on_chip_memory_write_fmbuf_abs_word(desc.bank_id, byte_offset, word);
}

static axi_vec_t concat_low_byte_mask(unsigned byte_count) {
#pragma HLS INLINE
    if (byte_count >= static_cast<unsigned>(AXI_WORD_BYTES)) {
        return ~static_cast<axi_vec_t>(0);
    }
    return static_cast<axi_vec_t>((static_cast<axi_vec_t>(1) << (byte_count * 8U)) - 1U);
}

static bool concat_write_slice_narrow(const tensor_desc_t& desc,
                                      u16_t h,
                                      u16_t w,
                                      u16_t c_begin,
                                      u8_t valid_c,
                                      act_vec_t packed) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off
    const unsigned lanes = valid_c.to_uint();
    if (lanes == 0U || lanes > static_cast<unsigned>(TM) ||
        h >= desc.h || w >= desc.w ||
        c_begin.to_uint() + lanes > desc.c.to_uint()) {
        return false;
    }

    const u32_t start_offset = desc.base_offset + concat_tensor_byte_offset(desc, h, w, c_begin);
    const u32_t word0_offset = start_offset & static_cast<u32_t>(~(AXI_WORD_BYTES - 1));
    const unsigned byte0 = start_offset.to_uint() & static_cast<unsigned>(AXI_WORD_BYTES - 1);

    if (byte0 == 0U && lanes == static_cast<unsigned>(AXI_WORD_BYTES)) {
        return concat_write_aligned_abs_word(desc, word0_offset, packed);
    }

    axi_vec_t word0 = 0;
    if (!concat_read_aligned_abs_word(desc, word0_offset, word0)) {
        return false;
    }
    const unsigned first_count =
        ((byte0 + lanes) <= static_cast<unsigned>(AXI_WORD_BYTES))
            ? lanes
            : (static_cast<unsigned>(AXI_WORD_BYTES) - byte0);
    const axi_vec_t mask0 = static_cast<axi_vec_t>(concat_low_byte_mask(first_count) << (byte0 * 8U));
    const axi_vec_t shifted0 = static_cast<axi_vec_t>(packed << (byte0 * 8U));
    word0 = static_cast<axi_vec_t>((word0 & ~mask0) | (shifted0 & mask0));
    if (!concat_write_aligned_abs_word(desc, word0_offset, word0)) {
        return false;
    }

    if (first_count < lanes) {
        const u32_t word1_offset = word0_offset + static_cast<u32_t>(AXI_WORD_BYTES);
        axi_vec_t word1 = 0;
        if (!concat_read_aligned_abs_word(desc, word1_offset, word1)) {
            return false;
        }
        const unsigned second_count = lanes - first_count;
        const axi_vec_t mask1 = concat_low_byte_mask(second_count);
        const axi_vec_t shifted1 = static_cast<axi_vec_t>(packed >> (first_count * 8U));
        word1 = static_cast<axi_vec_t>((word1 & ~mask1) | (shifted1 & mask1));
        if (!concat_write_aligned_abs_word(desc, word1_offset, word1)) {
            return false;
        }
    }
    return true;
}

bool concat_writer(const tensor_desc_t& src,
                   const tensor_desc_t& dst,
                   u16_t c_offset,
                   u16_t valid_c) {
#pragma HLS INLINE off
    if (!same_shape_2d(src, dst) ||
        valid_c.to_uint() == 0U ||
        src.c.to_uint() < valid_c.to_uint() ||
        dst.c.to_uint() < c_offset.to_uint() + valid_c.to_uint()) {
        return false;
    }

    if (slice_already_in_place(src, dst, c_offset, valid_c)) {
        return true;
    }

    const int h_count = static_cast<int>(dst.h.to_uint());
    const int w_count = static_cast<int>(dst.w.to_uint());
    const int c_blocks = static_cast<int>((valid_c.to_uint() + static_cast<unsigned>(TM) - 1U) /
                                          static_cast<unsigned>(TM));

    for (int h_i = 0; h_i < MAX_FM_H; ++h_i) {
        if (h_i >= h_count) {
            break;
        }
        const u16_t h = static_cast<u16_t>(h_i);
        for (int w_i = 0; w_i < MAX_FM_W; ++w_i) {
            if (w_i >= w_count) {
                break;
            }
            const u16_t w = static_cast<u16_t>(w_i);
            for (int c_blk = 0; c_blk < MAX_C_TILE_COUNT; ++c_blk) {
#pragma HLS PIPELINE off
                if (c_blk >= c_blocks) {
                    break;
                }
                const u16_t c = static_cast<u16_t>(c_blk * TM);
                const u16_t remaining = static_cast<u16_t>(valid_c - c);
                const u8_t lanes = concat_lanes(remaining);
                act_vec_t packed = 0;
                if (!on_chip_memory_read_packed_tile(src,
                                                     static_cast<i32_t>(h),
                                                     static_cast<i32_t>(w),
                                                     c,
                                                     lanes,
                                                     packed)) {
                    return false;
                }
                if (!concat_write_slice_narrow(dst,
                                               h,
                                               w,
                                               static_cast<u16_t>(c_offset + c),
                                               lanes,
                                               packed)) {
                    return false;
                }
            }
        }
    }

    return true;
}

}  // namespace esp_int8
