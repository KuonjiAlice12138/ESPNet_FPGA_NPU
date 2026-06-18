#include "../include/npu_config.hpp"
#include "../include/npu_uop.hpp"
#include "../include/npu_types.hpp"

namespace esp_int8 {

static axi_vec_t s_pool2_bram[BRAM_SCR1_AXI_WORDS];
static axi_vec_t s_fmbuf_uram[FMBUF_URAM_AXI_WORDS];
static axi_vec_t s_fmbuf_bram[FMBUF_BRAM_AXI_WORDS];

struct tensor_static_desc_t {
    u8_t bank_id;
    u32_t base_offset;
    u16_t h;
    u16_t w;
    u16_t c;
    u16_t phys_c;
    u16_t c_offset;
    u32_t byte_size;
};

static tensor_static_desc_t make_static_desc(unsigned bank_id,
                                             unsigned base,
                                             unsigned h,
                                             unsigned w,
                                             unsigned c,
                                             unsigned phys_c = 0,
                                             unsigned c_offset = 0) {
#pragma HLS INLINE
    tensor_static_desc_t desc;
    desc.bank_id = bank_id;
    desc.base_offset = base;
    desc.h = h;
    desc.w = w;
    desc.c = c;
    desc.phys_c = (phys_c == 0) ? c : phys_c;
    desc.c_offset = c_offset;
    desc.byte_size = static_cast<u32_t>(h) * static_cast<u32_t>(w) * static_cast<u32_t>(c);
    return desc;
}

static u16_t desc_phys_c(const tensor_desc_t& desc) {
#pragma HLS INLINE
    return (desc.reserved0.to_uint() == 0U) ? desc.c : desc.reserved0;
}

static u16_t desc_c_offset(const tensor_desc_t& desc) {
#pragma HLS INLINE
    return desc.reserved1;
}

static u8_t effective_lanes(u8_t valid_c) {
#pragma HLS INLINE
    const unsigned lanes = valid_c.to_uint();
    if (lanes == 0U || lanes > static_cast<unsigned>(TM)) {
        return TM;
    }
    return valid_c;
}

static bool get_static_tensor_desc(u8_t tensor_id, tensor_static_desc_t& desc) {
#pragma HLS INLINE
    switch (static_cast<unsigned>(tensor_id.to_uint())) {
        case TID_INPUT:
            desc = make_static_desc(BANK_FMEM0, 0x000000U, 512, 1024, 3);
            return true;
        case TID_POOL1:
            desc = make_static_desc(BANK_FMEM0, FMBUF_POOL1_BASE, 256, 512, 3);
            return true;
        case TID_B1_CAT:
        case TID_B1_ACT:
            desc = make_static_desc(BANK_FMEM1, 0x180000U, 256, 512, 19);
            return true;
        case TID_L20_CAT:
        case TID_L20_ACT:
            desc = make_static_desc(BANK_FMEM0,
                                    FMBUF_L20_BASE,
                                    128,
                                    256,
                                    64,
                                    FMBUF_L20_PHYS_C,
                                    FMBUF_L20_C_OFFSET);
            return true;
        case TID_L2B0_CAT:
        case TID_L2B0_ACT:
            desc = make_static_desc(BANK_FMEM0, 0x000000U, 128, 256, 64, 131, 0);
            return true;
        case TID_POOL2:
            desc = make_static_desc(BANK_BRAM_SCR1, 0x000000U, 128, 256, 3);
            return true;
        case TID_B2_CAT:
        case TID_B2_ACT:
            desc = make_static_desc(BANK_FMEM0, 0x000000U, 128, 256, 131);
            return true;
        case TID_L30_CAT:
        case TID_L30_ACT:
            desc = make_static_desc(BANK_FMEM1, FMBUF_L30_BASE, 64, 128, 128);
            return true;
        case TID_L3B0_CAT:
        case TID_L3B0_ACT:
            desc = make_static_desc(BANK_FMEM0, 0x000000U, 64, 128, 128, 256, 128);
            return true;
        case TID_B3_CAT:
        case TID_B3_ACT:
            desc = make_static_desc(BANK_FMEM0, 0x000000U, 64, 128, 256);
            return true;
        case TID_POOL_TMP:
            desc = make_static_desc(BANK_BRAM_SCR0, 0x000000U, 256, 512, 3);
            return true;
        default:
            desc = make_static_desc(BANK_FMEM0, 0, 0, 0, 0);
            return false;
    }
}

static bool bank_base(u8_t bank_id, u32_t& base) {
#pragma HLS INLINE
    switch (static_cast<unsigned>(bank_id.to_uint())) {
        case BANK_FMEM0:
        case BANK_FMEM1:
        case BANK_FMEM2:
            base = 0;
            return true;
        case BANK_BRAM_SCR0:
            base = FMBUF_POOL_TMP_BASE;
            return true;
        default:
            base = 0;
            return false;
    }
}

static bool resolve_phys_addr(u8_t bank_id, u32_t byte_offset, u32_t& phys_addr) {
#pragma HLS INLINE
    u32_t base = 0;
    if (!bank_base(bank_id, base)) {
        phys_addr = 0;
        return false;
    }
    phys_addr = base + byte_offset;
    return phys_addr.to_uint() < static_cast<unsigned>(FMBUF_BYTES);
}

static bool read_fmbuf_uram_word(u32_t word_idx, axi_vec_t& value) {
#pragma HLS INLINE
#pragma HLS BIND_STORAGE variable=s_fmbuf_uram type=ram_2p impl=uram
#pragma HLS RESET variable=s_fmbuf_uram off
    const unsigned idx = word_idx.to_uint();
    if (idx >= static_cast<unsigned>(FMBUF_URAM_AXI_WORDS)) {
        value = 0;
        return false;
    }
    value = s_fmbuf_uram[idx];
    return true;
}

static bool write_fmbuf_uram_word(u32_t word_idx, axi_vec_t value) {
#pragma HLS INLINE
    const unsigned idx = word_idx.to_uint();
    if (idx >= static_cast<unsigned>(FMBUF_URAM_AXI_WORDS)) {
        return false;
    }
    s_fmbuf_uram[idx] = value;
    return true;
}

static bool read_fmbuf_bram_word(u32_t word_idx, axi_vec_t& value) {
#pragma HLS INLINE
#pragma HLS BIND_STORAGE variable=s_fmbuf_bram type=ram_2p impl=bram
#pragma HLS RESET variable=s_fmbuf_bram off
    const unsigned idx = word_idx.to_uint();
    if (idx >= static_cast<unsigned>(FMBUF_BRAM_AXI_WORDS)) {
        value = 0;
        return false;
    }
    value = s_fmbuf_bram[idx];
    return true;
}

static bool write_fmbuf_bram_word(u32_t word_idx, axi_vec_t value) {
#pragma HLS INLINE
    const unsigned idx = word_idx.to_uint();
    if (idx >= static_cast<unsigned>(FMBUF_BRAM_AXI_WORDS)) {
        return false;
    }
    s_fmbuf_bram[idx] = value;
    return true;
}

static bool read_pool2_bram_word(u32_t word_idx, axi_vec_t& value) {
#pragma HLS INLINE
#pragma HLS BIND_STORAGE variable=s_pool2_bram type=ram_2p impl=bram
#pragma HLS RESET variable=s_pool2_bram off
    const unsigned idx = word_idx.to_uint();
    if (idx >= static_cast<unsigned>(BRAM_SCR1_AXI_WORDS)) {
        value = 0;
        return false;
    }
    value = s_pool2_bram[idx];
    return true;
}

static bool write_pool2_bram_word(u32_t word_idx, axi_vec_t value) {
#pragma HLS INLINE
    const unsigned idx = word_idx.to_uint();
    if (idx >= static_cast<unsigned>(BRAM_SCR1_AXI_WORDS)) {
        return false;
    }
    s_pool2_bram[idx] = value;
    return true;
}

static bool read_phys_word(u32_t phys_byte_addr, axi_vec_t& value) {
#pragma HLS INLINE
    const unsigned idx = phys_byte_addr.to_uint();
    if ((idx & (AXI_WORD_BYTES - 1)) != 0U || idx + AXI_WORD_BYTES > static_cast<unsigned>(FMBUF_BYTES)) {
        value = 0;
        return false;
    }

    const u32_t word_idx = phys_byte_addr >> 5;
    if (idx < static_cast<unsigned>(FMBUF_URAM_BYTES)) {
        return read_fmbuf_uram_word(word_idx, value);
    }
    return read_fmbuf_bram_word(word_idx - static_cast<u32_t>(FMBUF_URAM_AXI_WORDS), value);
}

static bool write_phys_word(u32_t phys_byte_addr, axi_vec_t value) {
#pragma HLS INLINE
    const unsigned idx = phys_byte_addr.to_uint();
    if ((idx & (AXI_WORD_BYTES - 1)) != 0U || idx + AXI_WORD_BYTES > static_cast<unsigned>(FMBUF_BYTES)) {
        return false;
    }

    const u32_t word_idx = phys_byte_addr >> 5;
    if (idx < static_cast<unsigned>(FMBUF_URAM_BYTES)) {
        return write_fmbuf_uram_word(word_idx, value);
    }
    return write_fmbuf_bram_word(word_idx - static_cast<u32_t>(FMBUF_URAM_AXI_WORDS), value);
}

static bool read_bank_word(u8_t bank_id, u32_t byte_offset, axi_vec_t& value) {
#pragma HLS INLINE
    if (bank_id.to_uint() == static_cast<unsigned>(BANK_BRAM_SCR1)) {
        const unsigned idx = byte_offset.to_uint();
        if ((idx & (AXI_WORD_BYTES - 1)) != 0U ||
            idx + AXI_WORD_BYTES > static_cast<unsigned>(BRAM_SCR1_BYTES)) {
            value = 0;
            return false;
        }
        return read_pool2_bram_word(static_cast<u32_t>(idx >> 5), value);
    }

    u32_t phys_addr = 0;
    if (!resolve_phys_addr(bank_id, byte_offset, phys_addr)) {
        value = 0;
        return false;
    }
    return read_phys_word(phys_addr, value);
}

static bool write_bank_word(u8_t bank_id, u32_t byte_offset, axi_vec_t value) {
#pragma HLS INLINE
    if (bank_id.to_uint() == static_cast<unsigned>(BANK_BRAM_SCR1)) {
        const unsigned idx = byte_offset.to_uint();
        if ((idx & (AXI_WORD_BYTES - 1)) != 0U ||
            idx + AXI_WORD_BYTES > static_cast<unsigned>(BRAM_SCR1_BYTES)) {
            return false;
        }
        return write_pool2_bram_word(static_cast<u32_t>(idx >> 5), value);
    }

    u32_t phys_addr = 0;
    if (!resolve_phys_addr(bank_id, byte_offset, phys_addr)) {
        return false;
    }
    return write_phys_word(phys_addr, value);
}

static axi_vec_t make_low_byte_mask(unsigned byte_count) {
#pragma HLS INLINE
    if (byte_count == 0U) {
        return 0;
    }
    if (byte_count >= static_cast<unsigned>(AXI_WORD_BYTES)) {
        return ~axi_vec_t(0);
    }
    const unsigned bit_count = byte_count * 8U;
    return static_cast<axi_vec_t>((static_cast<axi_vec_t>(1) << bit_count) - 1);
}

static bool read_tile_packed_word(u8_t bank_id,
                                  u32_t word0_offset,
                                  unsigned byte0,
                                  unsigned read_count,
                                  axi_vec_t& packed) {
#pragma HLS INLINE
    packed = 0;
    if (read_count == 0U) {
        return true;
    }

    const bool crosses_word = (byte0 + read_count) > static_cast<unsigned>(AXI_WORD_BYTES);
    if (!crosses_word) {
        axi_vec_t word = 0;
        if (!read_bank_word(bank_id, word0_offset, word)) {
            return false;
        }
        packed = static_cast<axi_vec_t>(word >> (byte0 * 8U));
    } else {
        axi_vec_t word0 = 0;
        axi_vec_t word1 = 0;
        if (!read_bank_word(bank_id, word0_offset, word0)) {
            return false;
        }
        if (!read_bank_word(bank_id, word0_offset + AXI_WORD_BYTES, word1)) {
            return false;
        }

        ap_uint<AXI_WORD_BITS * 2> pair = 0;
        pair.range(AXI_WORD_BITS - 1, 0) = word0;
        pair.range(AXI_WORD_BITS * 2 - 1, AXI_WORD_BITS) = word1;
        packed = static_cast<axi_vec_t>(pair >> (byte0 * 8U));
    }

    if (read_count < static_cast<unsigned>(AXI_WORD_BYTES)) {
        packed &= make_low_byte_mask(read_count);
    }
    return true;
}

static bool write_packed_full_word(u8_t bank_id,
                                   u32_t word_offset,
                                   axi_vec_t packed);

static bool write_packed_one_word(u8_t bank_id,
                                  u32_t word_offset,
                                  unsigned byte0,
                                  unsigned lane_count,
                                  axi_vec_t packed);

static bool write_packed_cross_word(u8_t bank_id,
                                    u32_t word0_offset,
                                    unsigned byte0,
                                    unsigned lane_count,
                                    axi_vec_t packed);

static bool write_packed_full_word(u8_t bank_id,
                                   u32_t word_offset,
                                   axi_vec_t packed) {
#pragma HLS INLINE
    return write_bank_word(bank_id, word_offset, packed);
}

static bool write_packed_one_word(u8_t bank_id,
                                  u32_t word_offset,
                                  unsigned byte0,
                                  unsigned lane_count,
                                  axi_vec_t packed) {
#pragma HLS INLINE
    axi_vec_t word = 0;
    if (!read_bank_word(bank_id, word_offset, word)) {
        return false;
    }

    const axi_vec_t mask = static_cast<axi_vec_t>(make_low_byte_mask(lane_count) << (byte0 * 8U));
    const axi_vec_t shifted = static_cast<axi_vec_t>(packed << (byte0 * 8U));
    word = static_cast<axi_vec_t>((word & ~mask) | (shifted & mask));
    return write_bank_word(bank_id, word_offset, word);
}

static bool write_packed_cross_word(u8_t bank_id,
                                    u32_t word0_offset,
                                    unsigned byte0,
                                    unsigned lane_count,
                                    axi_vec_t packed) {
#pragma HLS INLINE
    axi_vec_t word0 = 0;
    axi_vec_t word1 = 0;
    if (!read_bank_word(bank_id, word0_offset, word0)) {
        return false;
    }
    if (!read_bank_word(bank_id, word0_offset + AXI_WORD_BYTES, word1)) {
        return false;
    }

    const unsigned first_count = static_cast<unsigned>(AXI_WORD_BYTES) - byte0;
    const unsigned second_count = lane_count - first_count;

    const axi_vec_t mask0 = static_cast<axi_vec_t>(make_low_byte_mask(first_count) << (byte0 * 8U));
    const axi_vec_t shifted0 = static_cast<axi_vec_t>(packed << (byte0 * 8U));
    word0 = static_cast<axi_vec_t>((word0 & ~mask0) | (shifted0 & mask0));

    const axi_vec_t mask1 = make_low_byte_mask(second_count);
    const axi_vec_t shifted1 = static_cast<axi_vec_t>(packed >> (first_count * 8U));
    word1 = static_cast<axi_vec_t>((word1 & ~mask1) | (shifted1 & mask1));

    if (!write_bank_word(bank_id, word0_offset, word0)) {
        return false;
    }
    return write_bank_word(bank_id, word0_offset + AXI_WORD_BYTES, word1);
}

static u32_t tensor_elem_offset(const tensor_desc_t& desc, u16_t h, u16_t w, u16_t c) {
#pragma HLS INLINE
    return (static_cast<u32_t>(h) * static_cast<u32_t>(desc.w) + static_cast<u32_t>(w)) *
               static_cast<u32_t>(desc_phys_c(desc)) +
           static_cast<u32_t>(desc_c_offset(desc)) +
           static_cast<u32_t>(c);
}

bool on_chip_memory_read_packed_tile(const tensor_desc_t& desc,
                                     i32_t h,
                                     i32_t w,
                                     u16_t c_begin,
                                     u8_t valid_c,
                                     axi_vec_t& packed) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off
    packed = 0;
    const u8_t lanes = effective_lanes(valid_c);
    const unsigned lane_count = lanes.to_uint();
    const bool spatial_valid = (h >= 0 && w >= 0 &&
                                h < static_cast<i32_t>(desc.h) &&
                                w < static_cast<i32_t>(desc.w));
    if (!spatial_valid || c_begin >= desc.c) {
        return true;
    }

    const unsigned remaining_c = static_cast<unsigned>(desc.c.to_uint() - c_begin.to_uint());
    const unsigned read_count = (lane_count < remaining_c) ? lane_count : remaining_c;
    if (read_count == 0U) {
        return true;
    }

    const u32_t start_offset =
        desc.base_offset +
        tensor_elem_offset(desc, static_cast<u16_t>(h), static_cast<u16_t>(w), c_begin);
    const u32_t word0_offset = start_offset & static_cast<u32_t>(~(AXI_WORD_BYTES - 1));
    const unsigned byte0 = start_offset.to_uint() & 0x1fU;
    return read_tile_packed_word(desc.bank_id, word0_offset, byte0, read_count, packed);
}

bool on_chip_memory_read_packed_contiguous(const tensor_desc_t& desc,
                                           i32_t h,
                                           i32_t w,
                                           u16_t c_begin,
                                           u8_t valid_bytes,
                                           axi_vec_t& packed) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off
    packed = 0;
    const u8_t lanes = effective_lanes(valid_bytes);
    const unsigned lane_count = lanes.to_uint();
    const bool spatial_valid = (h >= 0 && w >= 0 &&
                                h < static_cast<i32_t>(desc.h) &&
                                w < static_cast<i32_t>(desc.w));
    if (!spatial_valid) {
        return true;
    }

    const unsigned phys_c = desc_phys_c(desc).to_uint();
    const unsigned c_begin_u = c_begin.to_uint();
    if (c_begin_u >= phys_c) {
        return true;
    }

    const unsigned w_u = static_cast<u16_t>(w).to_uint();
    const unsigned remaining_row_bytes =
        (desc.w.to_uint() - w_u) * phys_c - c_begin_u;
    const unsigned read_count =
        (lane_count < remaining_row_bytes) ? lane_count : remaining_row_bytes;
    if (read_count == 0U) {
        return true;
    }

    const u32_t start_offset =
        desc.base_offset +
        tensor_elem_offset(desc, static_cast<u16_t>(h), static_cast<u16_t>(w), c_begin);
    const u32_t word0_offset = start_offset & static_cast<u32_t>(~(AXI_WORD_BYTES - 1));
    const unsigned byte0 = start_offset.to_uint() & 0x1fU;
    return read_tile_packed_word(desc.bank_id, word0_offset, byte0, read_count, packed);
}

bool on_chip_memory_read_aligned_full_tile(const tensor_desc_t& desc,
                                           i32_t h,
                                           i32_t w,
                                           u16_t c_begin,
                                           axi_vec_t& packed) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off
    packed = 0;
    const bool spatial_valid = (h >= 0 && w >= 0 &&
                                h < static_cast<i32_t>(desc.h) &&
                                w < static_cast<i32_t>(desc.w));
    if (!spatial_valid || c_begin >= desc.c) {
        return true;
    }

    if (static_cast<unsigned>(desc.c.to_uint() - c_begin.to_uint()) <
        static_cast<unsigned>(AXI_WORD_BYTES)) {
        return true;
    }

    const u32_t start_offset =
        desc.base_offset +
        tensor_elem_offset(desc, static_cast<u16_t>(h), static_cast<u16_t>(w), c_begin);
    return read_bank_word(desc.bank_id, start_offset, packed);
}

bool on_chip_memory_write_packed_tile(const tensor_desc_t& desc,
                                      u16_t h,
                                      u16_t w,
                                      u16_t c_begin,
                                      u8_t valid_c,
                                      axi_vec_t packed) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off
    const u8_t lanes = effective_lanes(valid_c);
    const unsigned lane_count = lanes.to_uint();
    if (h >= desc.h || w >= desc.w || c_begin >= desc.c ||
        c_begin.to_uint() + lane_count > desc.c.to_uint()) {
        return false;
    }

    const u32_t start_offset = desc.base_offset + tensor_elem_offset(desc, h, w, c_begin);
    const u32_t word0_offset = start_offset & static_cast<u32_t>(~(AXI_WORD_BYTES - 1));
    const unsigned byte0 = start_offset.to_uint() & 0x1fU;
    const bool crosses_word = (byte0 + lane_count) > static_cast<unsigned>(AXI_WORD_BYTES);

    if (byte0 == 0U &&
        c_begin.to_uint() == 0U &&
        desc_phys_c(desc).to_uint() >= static_cast<unsigned>(AXI_WORD_BYTES) &&
        desc.c.to_uint() > lane_count) {
        return write_packed_full_word(desc.bank_id, word0_offset, packed);
    }
    if (byte0 == 0U &&
        desc_phys_c(desc).to_uint() >= static_cast<unsigned>(AXI_WORD_BYTES) &&
        c_begin.to_uint() == 0U &&
        lane_count == static_cast<unsigned>(desc.c.to_uint())) {
        return write_packed_full_word(desc.bank_id, word0_offset, packed);
    }
    if (byte0 == 0U && lane_count == static_cast<unsigned>(AXI_WORD_BYTES)) {
        return write_packed_full_word(desc.bank_id, word0_offset, packed);
    }
    if (!crosses_word) {
        return write_packed_one_word(desc.bank_id, word0_offset, byte0, lane_count, packed);
    }
    return write_packed_cross_word(desc.bank_id, word0_offset, byte0, lane_count, packed);
}

bool on_chip_memory_write_aligned_full_tile(const tensor_desc_t& desc,
                                            u16_t h,
                                            u16_t w,
                                            u16_t c_begin,
                                            axi_vec_t packed) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off
    if (h >= desc.h || w >= desc.w ||
        c_begin.to_uint() + static_cast<unsigned>(AXI_WORD_BYTES) > desc.c.to_uint()) {
        return false;
    }

    const u32_t start_offset = desc.base_offset + tensor_elem_offset(desc, h, w, c_begin);
    if ((start_offset.to_uint() & static_cast<unsigned>(AXI_WORD_BYTES - 1)) != 0U) {
        return false;
    }
    return write_bank_word(desc.bank_id, start_offset, packed);
}

bool on_chip_memory_write_fmbuf_abs_word(u8_t bank_id,
                                         u32_t byte_offset,
                                         axi_vec_t packed) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off
    if ((byte_offset.to_uint() & static_cast<unsigned>(AXI_WORD_BYTES - 1)) != 0U) {
        return false;
    }

    u32_t phys_addr = byte_offset;
    switch (static_cast<unsigned>(bank_id.to_uint())) {
        case BANK_FMEM0:
        case BANK_FMEM1:
        case BANK_FMEM2:
            phys_addr = byte_offset;
            break;
        case BANK_BRAM_SCR0:
            phys_addr = static_cast<u32_t>(FMBUF_POOL_TMP_BASE) + byte_offset;
            break;
        default:
            return false;
    }
    return write_phys_word(phys_addr, packed);
}

bool on_chip_memory_read_fmbuf_abs_word(u8_t bank_id,
                                        u32_t byte_offset,
                                        axi_vec_t& packed) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off
    if ((byte_offset.to_uint() & static_cast<unsigned>(AXI_WORD_BYTES - 1)) != 0U) {
        packed = 0;
        return false;
    }

    u32_t phys_addr = byte_offset;
    switch (static_cast<unsigned>(bank_id.to_uint())) {
        case BANK_FMEM0:
        case BANK_FMEM1:
        case BANK_FMEM2:
            phys_addr = byte_offset;
            break;
        case BANK_BRAM_SCR0:
            phys_addr = static_cast<u32_t>(FMBUF_POOL_TMP_BASE) + byte_offset;
            break;
        default:
            packed = 0;
            return false;
    }
    return read_phys_word(phys_addr, packed);
}

bool on_chip_memory_write_pool2_abs_word(u32_t byte_offset,
                                         axi_vec_t packed) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off
    const unsigned idx = byte_offset.to_uint();
    if ((idx & static_cast<unsigned>(AXI_WORD_BYTES - 1)) != 0U ||
        idx + AXI_WORD_BYTES > static_cast<unsigned>(BRAM_SCR1_BYTES)) {
        return false;
    }
    return write_pool2_bram_word(static_cast<u32_t>(idx >> 5), packed);
}

bool on_chip_memory_read_pool2_abs_word(u32_t byte_offset,
                                        axi_vec_t& packed) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off
    const unsigned idx = byte_offset.to_uint();
    if ((idx & static_cast<unsigned>(AXI_WORD_BYTES - 1)) != 0U ||
        idx + AXI_WORD_BYTES > static_cast<unsigned>(BRAM_SCR1_BYTES)) {
        packed = 0;
        return false;
    }
    return read_pool2_bram_word(static_cast<u32_t>(idx >> 5), packed);
}

bool on_chip_memory_write_axi_word(u8_t tensor_id, u32_t word_offset, axi_vec_t value) {
#pragma HLS INLINE off
    tensor_static_desc_t desc;
    const u32_t byte_offset = word_offset * AXI_WORD_BYTES;
    if (!get_static_tensor_desc(tensor_id, desc) || byte_offset + AXI_WORD_BYTES > desc.byte_size) {
        return false;
    }
    return write_bank_word(desc.bank_id, desc.base_offset + byte_offset, value);
}

}  // namespace esp_int8
