#include "../include/npu_config.hpp"
#include "../include/npu_schedule.hpp"
#include "../include/npu_types.hpp"

namespace esp_int8 {

bool on_chip_memory_write_fmbuf_abs_word(u8_t bank_id,
                                         u32_t byte_offset,
                                         act_vec_t packed);
bool on_chip_memory_write_pool2_abs_word(u32_t byte_offset,
                                         act_vec_t packed);
bool on_chip_memory_write_aligned_full_tile(const tensor_desc_t& desc,
                                            u16_t h,
                                            u16_t w,
                                            u16_t c_begin,
                                            axi_vec_t packed);

static u16_t store_tensor_phys_c(const tensor_desc_t& desc) {
#pragma HLS INLINE
  return (desc.reserved0.to_uint() == 0U) ? desc.c : desc.reserved0;
}

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
      store_tensor_phys_c(dst).to_uint() != c_u ||
      dst.reserved1.to_uint() != 0U) {
    return false;
  }
  const unsigned row_bytes = w_u * c_u;
  if ((row_bytes & static_cast<unsigned>(AXI_WORD_BYTES - 1)) != 0U) {
    return false;
  }
  if ((dst.base_offset.to_uint() & static_cast<unsigned>(AXI_WORD_BYTES - 1)) != 0U) {
    return false;
  }
  return (row_bytes / static_cast<unsigned>(AXI_WORD_BYTES)) <=
         static_cast<unsigned>(ROW_CONTIG_MAX_WORDS);
}

static void row_contiguous_set_byte(axi_vec_t row_words[ROW_CONTIG_MAX_WORDS],
                                    u32_t byte_idx,
                                    i8_t value) {
#pragma HLS INLINE
  const unsigned word_idx = byte_idx.to_uint() >> 5;
  const int lane = static_cast<int>(byte_idx.to_uint() & static_cast<unsigned>(AXI_WORD_BYTES - 1));
  u8_t raw = 0;
  raw.range(7, 0) = value.range(7, 0);
  axi_vec_t word = row_words[word_idx];
  word.range(lane * 8 + 7, lane * 8) = raw;
  row_words[word_idx] = word;
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

bool conv_store_write_aligned_tile(const tensor_desc_t& dst,
                                   u16_t h,
                                   u16_t w,
                                   u16_t c,
                                   const act_vec_t& word) {
#pragma HLS INLINE off
  return on_chip_memory_write_aligned_full_tile(dst, h, w, c, word);
}

bool conv_store_write_row_contiguous_word(const tensor_desc_t& dst,
                                          u32_t abs_offset,
                                          const axi_vec_t& word) {
#pragma HLS INLINE off
  return write_row_contiguous_abs_word(dst, abs_offset, word);
}

void conv_store_row_contiguous_set_byte(axi_vec_t row_words[ROW_CONTIG_MAX_WORDS],
                                        u32_t byte_idx,
                                        i8_t value) {
#pragma HLS INLINE
  row_contiguous_set_byte(row_words, byte_idx, value);
}

bool conv_store_row_contiguous_plan_ok(const tensor_desc_t& dst,
                                       u16_t width,
                                       u16_t valid_c) {
#pragma HLS INLINE off
  return row_contiguous_write_plan_ok(dst, width, valid_c);
}

}  // namespace esp_int8
