#include "../include/npu_config.hpp"
#include "../include/npu_uop.hpp"
#include "../include/npu_types.hpp"

namespace esp_int8 {

static axi_vec_t s_pool2_bram[BRAM_SCR1_AXI_WORDS];
static axi_vec_t s_fmbuf_uram[FMBUF_URAM_AXI_WORDS];
static axi_vec_t s_fmbuf_bram[FMBUF_BRAM_AXI_WORDS];

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

bool on_chip_memory_write_input_axi_word(u32_t word_offset, axi_vec_t value) {
#pragma HLS INLINE off
    const u32_t byte_offset = word_offset * AXI_WORD_BYTES;
    if (byte_offset + AXI_WORD_BYTES > static_cast<u32_t>(INPUT_FRAME_BYTES)) {
        return false;
    }
    return write_bank_word(static_cast<u8_t>(static_cast<unsigned>(BANK_FMEM0)), byte_offset, value);
}

}  // namespace esp_int8
