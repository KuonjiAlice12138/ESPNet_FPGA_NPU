#include "../include/npu_config.hpp"
#include "../include/npu_schedule.hpp"
#include "../include/npu_types.hpp"

namespace esp_int8 {

bool on_chip_memory_write_fmbuf_abs_word(u8_t bank_id,
                                         u32_t byte_offset,
                                         act_vec_t packed);
bool on_chip_memory_write_pool2_abs_word(u32_t byte_offset,
                                         act_vec_t packed);


static bool row_contiguous_write_plan_ok(const tensor_desc_t& dst,
                                         u16_t width,
                                         u16_t valid_c) {
#pragma HLS INLINE off
  const unsigned w_u = width.to_uint();
  const unsigned c_u = valid_c.to_uint();
  if (w_u == 0U || c_u == 0U || w_u > static_cast<unsigned>(ROW_CONTIG_MAX_W) ||
      c_u > static_cast<unsigned>(ROW_CONTIG_MAX_C)) {
    return false;
  }
  if (dst.w.to_uint() != w_u || dst.c.to_uint() != c_u ||
      tensor_phys_c(dst).to_uint() != c_u ||
      dst.reserved1.to_uint() != 0U) {
    return false;
  }
  const unsigned row_bytes = tensor_row_stride_bytes(dst).to_uint();
  if ((row_bytes & static_cast<unsigned>(AXI_WORD_BYTES - 1)) != 0U) {
    return false;
  }
  if ((dst.base_offset.to_uint() & static_cast<unsigned>(AXI_WORD_BYTES - 1)) != 0U) {
    return false;
  }
  return (row_bytes / static_cast<unsigned>(AXI_WORD_BYTES)) <=
         static_cast<unsigned>(ROW_CONTIG_MAX_WORDS);
}

static bool write_row_contiguous_abs_word(const tensor_desc_t& dst,
                                          u32_t byte_offset,
                                          axi_vec_t word) {
#pragma HLS INLINE
  if (dst.bank_id.to_uint() == static_cast<unsigned>(BANK_BRAM_SCR1)) {
    return on_chip_memory_write_pool2_abs_word(byte_offset, word);
  }
  return on_chip_memory_write_fmbuf_abs_word(dst.bank_id, byte_offset, word);
}

bool conv_store_write_row_contiguous_word(const tensor_desc_t& dst,
                                          u32_t abs_offset,
                                          const axi_vec_t& word) {
#pragma HLS INLINE off
  return write_row_contiguous_abs_word(dst, abs_offset, word);
}

bool conv_store_word_span_plan_ok(u8_t bank_id,
                                   u64_t first_offset,
                                   u32_t byte_step,
                                   u16_t word_count) {
#pragma HLS INLINE off
  static_assert((FMBUF_POOL_TMP_BASE & (AXI_WORD_BYTES - 1)) == 0,
                "Pool scratch base must be word aligned");
  static_assert((FMBUF_POOL2_ALIAS_BASE & (AXI_WORD_BYTES - 1)) == 0 &&
                    BRAM_SCR1_BYTES <= FMBUF_POOL2_ALIAS_BYTES &&
                    FMBUF_POOL2_ALIAS_BASE + BRAM_SCR1_BYTES <= FMBUF_BYTES,
                "Pool alias must fit the shared physical memory");
  const unsigned count = word_count.to_uint();
  if (count == 0U || count > static_cast<unsigned>(ROW_CONTIG_MAX_WORDS) ||
      first_offset.range(4, 0).to_uint() != 0U ||
      byte_step.to_uint() < static_cast<unsigned>(AXI_WORD_BYTES) ||
      (byte_step.to_uint() & static_cast<unsigned>(AXI_WORD_BYTES - 1)) != 0U) {
    return false;
  }
  unsigned byte_capacity = 0U;
  switch (bank_id.to_uint()) {
    case static_cast<unsigned>(BANK_FMEM0):
    case static_cast<unsigned>(BANK_FMEM1):
    case static_cast<unsigned>(BANK_FMEM2):
      byte_capacity = FMBUF_BYTES;
      break;
    case static_cast<unsigned>(BANK_BRAM_SCR0):
      byte_capacity = FMBUF_BYTES - FMBUF_POOL_TMP_BASE;
      break;
    case static_cast<unsigned>(BANK_BRAM_SCR1):
      byte_capacity = BRAM_SCR1_BYTES;
      break;
    default:
      return false;
  }
  if (first_offset >= static_cast<u64_t>(byte_capacity)) { return false; }
  // Widen before the last-address calculation; a wrapped 32-bit end is not valid.
  const ap_uint<9> last_word = static_cast<ap_uint<9>>(count - 1U);
  const u64_t end_offset = first_offset +
      static_cast<u64_t>(last_word * byte_step) + static_cast<u64_t>(AXI_WORD_BYTES);
  return end_offset <= static_cast<u64_t>(byte_capacity);
}

bool conv_store_row_contiguous_plan_ok(const tensor_desc_t& dst,
                                       u16_t width,
                                       u16_t valid_c) {
#pragma HLS INLINE off
  return row_contiguous_write_plan_ok(dst, width, valid_c);
}

}  // namespace esp_int8
