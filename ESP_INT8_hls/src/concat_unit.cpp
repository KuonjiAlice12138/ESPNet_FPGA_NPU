#include "../include/npu_config.hpp"
#include "../include/npu_types.hpp"

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
                i8_t tile[TM];
#pragma HLS ARRAY_PARTITION variable=tile complete dim=1
                if (!on_chip_memory_read_tile(src,
                                              static_cast<i32_t>(h),
                                              static_cast<i32_t>(w),
                                              c,
                                              lanes,
                                              tile)) {
                    return false;
                }
                if (!on_chip_memory_write_tile(dst,
                                               h,
                                               w,
                                               static_cast<u16_t>(c_offset + c),
                                               lanes,
                                               tile)) {
                    return false;
                }
            }
        }
    }

    return true;
}

}  // namespace esp_int8
