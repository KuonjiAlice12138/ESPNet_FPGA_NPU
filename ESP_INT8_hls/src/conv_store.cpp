#include "../include/npu_config.hpp"
#include "../include/npu_types.hpp"

namespace esp_int8 {

bool on_chip_memory_write_packed_tile(const tensor_desc_t& desc,
                                      u16_t h,
                                      u16_t w,
                                      u16_t c_begin,
                                      u8_t valid_c,
                                      act_vec_t packed);
bool on_chip_memory_write_aligned_row_word(const tensor_desc_t& desc,
                                           u16_t h,
                                           u32_t row_byte_offset,
                                           act_vec_t packed);

static u16_t ceil_div_u16(u16_t a, u16_t b) {
#pragma HLS INLINE
  return static_cast<u16_t>((a + b - 1) / b);
}

static u16_t conv_effective_stride(const conv_cfg_t& cfg) {
#pragma HLS INLINE
  return (cfg.stride == 0) ? static_cast<u16_t>(1) : static_cast<u16_t>(cfg.stride);
}

static u16_t conv_out_dim(u16_t in_size, u16_t stride) {
#pragma HLS INLINE
  return ceil_div_u16(in_size, stride);
}

template <int DST_LANE, int SRC_LANE, int COUNT>
static void copy_act_segment(act_vec_t& dst, const act_vec_t& src) {
#pragma HLS INLINE
  dst.range(DST_LANE * 8 + COUNT * 8 - 1, DST_LANE * 8) =
      src.range(SRC_LANE * 8 + COUNT * 8 - 1, SRC_LANE * 8);
}

static u16_t core_desc_phys_c(const tensor_desc_t& desc) {
#pragma HLS INLINE
  return (desc.reserved0.to_uint() == 0U) ? desc.c : desc.reserved0;
}
static bool store_compact_c12_row(const tensor_desc_t& dst,
                                  u16_t out_row,
                                  int out_w_i,
                                  act_vec_t row_buf[MAX_FM_W]) {
#pragma HLS INLINE off
  bool ok = true;
  for (int group = 0; group < MAX_FM_W / 8; ++group) {
#pragma HLS PIPELINE off
    const int pix = group * 8;
    if (pix >= out_w_i) {
      break;
    }
    act_vec_t p0 = row_buf[pix + 0];
    act_vec_t p1 = row_buf[pix + 1];
    act_vec_t p2 = row_buf[pix + 2];
    act_vec_t p3 = row_buf[pix + 3];
    act_vec_t p4 = row_buf[pix + 4];
    act_vec_t p5 = row_buf[pix + 5];
    act_vec_t p6 = row_buf[pix + 6];
    act_vec_t p7 = row_buf[pix + 7];

    act_vec_t w0 = 0;
    copy_act_segment<0, 0, 12>(w0, p0);
    copy_act_segment<12, 0, 12>(w0, p1);
    copy_act_segment<24, 0, 8>(w0, p2);

    act_vec_t w1 = 0;
    copy_act_segment<0, 8, 4>(w1, p2);
    copy_act_segment<4, 0, 12>(w1, p3);
    copy_act_segment<16, 0, 12>(w1, p4);
    copy_act_segment<28, 0, 4>(w1, p5);

    act_vec_t w2 = 0;
    copy_act_segment<0, 4, 8>(w2, p5);
    copy_act_segment<8, 0, 12>(w2, p6);
    copy_act_segment<20, 0, 12>(w2, p7);

    const u32_t byte_offset = static_cast<u32_t>(group) * static_cast<u32_t>(96);
    if (!on_chip_memory_write_aligned_row_word(dst, out_row, byte_offset, w0)) {
      ok = false;
    }
    if (!on_chip_memory_write_aligned_row_word(
            dst, out_row, byte_offset + static_cast<u32_t>(32), w1)) {
      ok = false;
    }
    if (!on_chip_memory_write_aligned_row_word(
            dst, out_row, byte_offset + static_cast<u32_t>(64), w2)) {
      ok = false;
    }
  }
  return ok;
}

static bool store_compact_c16_row(const tensor_desc_t& dst,
                                  u16_t out_row,
                                  int out_w_i,
                                  act_vec_t row_buf[MAX_FM_W]) {
#pragma HLS INLINE off
  bool ok = true;
  for (int group = 0; group < MAX_FM_W / 2; ++group) {
#pragma HLS PIPELINE off
    const int pix = group * 2;
    if (pix >= out_w_i) {
      break;
    }
    act_vec_t word = 0;
    copy_act_segment<0, 0, 16>(word, row_buf[pix + 0]);
    copy_act_segment<16, 0, 16>(word, row_buf[pix + 1]);
    const u32_t byte_offset = static_cast<u32_t>(group) * static_cast<u32_t>(32);
    if (!on_chip_memory_write_aligned_row_word(dst, out_row, byte_offset, word)) {
      ok = false;
    }
  }
  return ok;
}

static void write_compact_row_word(const tensor_desc_t& dst,
                                   u16_t out_row,
                                   u32_t byte_offset,
                                   act_vec_t word,
                                   bool& ok) {
#pragma HLS INLINE
  if (!on_chip_memory_write_aligned_row_word(dst, out_row, byte_offset, word)) {
    ok = false;
  }
}

static bool store_compact_c2_row(const tensor_desc_t& dst,
                                 u16_t out_row,
                                 int out_w_i,
                                 act_vec_t row_buf[MAX_FM_W]) {
#pragma HLS INLINE off
  bool ok = true;
  for (int group = 0; group < MAX_FM_W / 16; ++group) {
#pragma HLS PIPELINE off
    const int pix = group * 16;
    if (pix >= out_w_i) {
      break;
    }
    act_vec_t word = 0;
    copy_act_segment<0, 0, 2>(word, row_buf[pix + 0]);
    copy_act_segment<2, 0, 2>(word, row_buf[pix + 1]);
    copy_act_segment<4, 0, 2>(word, row_buf[pix + 2]);
    copy_act_segment<6, 0, 2>(word, row_buf[pix + 3]);
    copy_act_segment<8, 0, 2>(word, row_buf[pix + 4]);
    copy_act_segment<10, 0, 2>(word, row_buf[pix + 5]);
    copy_act_segment<12, 0, 2>(word, row_buf[pix + 6]);
    copy_act_segment<14, 0, 2>(word, row_buf[pix + 7]);
    copy_act_segment<16, 0, 2>(word, row_buf[pix + 8]);
    copy_act_segment<18, 0, 2>(word, row_buf[pix + 9]);
    copy_act_segment<20, 0, 2>(word, row_buf[pix + 10]);
    copy_act_segment<22, 0, 2>(word, row_buf[pix + 11]);
    copy_act_segment<24, 0, 2>(word, row_buf[pix + 12]);
    copy_act_segment<26, 0, 2>(word, row_buf[pix + 13]);
    copy_act_segment<28, 0, 2>(word, row_buf[pix + 14]);
    copy_act_segment<30, 0, 2>(word, row_buf[pix + 15]);
    const u32_t byte_offset = static_cast<u32_t>(group) * static_cast<u32_t>(32);
    write_compact_row_word(dst, out_row, byte_offset, word, ok);
  }
  return ok;
}

static bool store_compact_c25_row(const tensor_desc_t& dst,
                                  u16_t out_row,
                                  int out_w_i,
                                  act_vec_t row_buf[MAX_FM_W]) {
#pragma HLS INLINE off
  bool ok = true;
  for (int group = 0; group < MAX_FM_W / 32; ++group) {
#pragma HLS PIPELINE off
    const int pix = group * 32;
    if (pix >= out_w_i) {
      break;
    }
    const u32_t byte_offset = static_cast<u32_t>(group) * static_cast<u32_t>(800);

    act_vec_t w0 = 0;
    copy_act_segment<0, 0, 25>(w0, row_buf[pix + 0]);
    copy_act_segment<25, 0, 7>(w0, row_buf[pix + 1]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(0), w0, ok);
    act_vec_t w1 = 0;
    copy_act_segment<0, 7, 18>(w1, row_buf[pix + 1]);
    copy_act_segment<18, 0, 14>(w1, row_buf[pix + 2]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(32), w1, ok);
    act_vec_t w2 = 0;
    copy_act_segment<0, 14, 11>(w2, row_buf[pix + 2]);
    copy_act_segment<11, 0, 21>(w2, row_buf[pix + 3]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(64), w2, ok);
    act_vec_t w3 = 0;
    copy_act_segment<0, 21, 4>(w3, row_buf[pix + 3]);
    copy_act_segment<4, 0, 25>(w3, row_buf[pix + 4]);
    copy_act_segment<29, 0, 3>(w3, row_buf[pix + 5]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(96), w3, ok);
    act_vec_t w4 = 0;
    copy_act_segment<0, 3, 22>(w4, row_buf[pix + 5]);
    copy_act_segment<22, 0, 10>(w4, row_buf[pix + 6]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(128), w4, ok);
    act_vec_t w5 = 0;
    copy_act_segment<0, 10, 15>(w5, row_buf[pix + 6]);
    copy_act_segment<15, 0, 17>(w5, row_buf[pix + 7]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(160), w5, ok);
    act_vec_t w6 = 0;
    copy_act_segment<0, 17, 8>(w6, row_buf[pix + 7]);
    copy_act_segment<8, 0, 24>(w6, row_buf[pix + 8]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(192), w6, ok);
    act_vec_t w7 = 0;
    copy_act_segment<0, 24, 1>(w7, row_buf[pix + 8]);
    copy_act_segment<1, 0, 25>(w7, row_buf[pix + 9]);
    copy_act_segment<26, 0, 6>(w7, row_buf[pix + 10]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(224), w7, ok);
    act_vec_t w8 = 0;
    copy_act_segment<0, 6, 19>(w8, row_buf[pix + 10]);
    copy_act_segment<19, 0, 13>(w8, row_buf[pix + 11]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(256), w8, ok);
    act_vec_t w9 = 0;
    copy_act_segment<0, 13, 12>(w9, row_buf[pix + 11]);
    copy_act_segment<12, 0, 20>(w9, row_buf[pix + 12]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(288), w9, ok);
    act_vec_t w10 = 0;
    copy_act_segment<0, 20, 5>(w10, row_buf[pix + 12]);
    copy_act_segment<5, 0, 25>(w10, row_buf[pix + 13]);
    copy_act_segment<30, 0, 2>(w10, row_buf[pix + 14]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(320), w10, ok);
    act_vec_t w11 = 0;
    copy_act_segment<0, 2, 23>(w11, row_buf[pix + 14]);
    copy_act_segment<23, 0, 9>(w11, row_buf[pix + 15]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(352), w11, ok);
    act_vec_t w12 = 0;
    copy_act_segment<0, 9, 16>(w12, row_buf[pix + 15]);
    copy_act_segment<16, 0, 16>(w12, row_buf[pix + 16]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(384), w12, ok);
    act_vec_t w13 = 0;
    copy_act_segment<0, 16, 9>(w13, row_buf[pix + 16]);
    copy_act_segment<9, 0, 23>(w13, row_buf[pix + 17]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(416), w13, ok);
    act_vec_t w14 = 0;
    copy_act_segment<0, 23, 2>(w14, row_buf[pix + 17]);
    copy_act_segment<2, 0, 25>(w14, row_buf[pix + 18]);
    copy_act_segment<27, 0, 5>(w14, row_buf[pix + 19]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(448), w14, ok);
    act_vec_t w15 = 0;
    copy_act_segment<0, 5, 20>(w15, row_buf[pix + 19]);
    copy_act_segment<20, 0, 12>(w15, row_buf[pix + 20]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(480), w15, ok);
    act_vec_t w16 = 0;
    copy_act_segment<0, 12, 13>(w16, row_buf[pix + 20]);
    copy_act_segment<13, 0, 19>(w16, row_buf[pix + 21]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(512), w16, ok);
    act_vec_t w17 = 0;
    copy_act_segment<0, 19, 6>(w17, row_buf[pix + 21]);
    copy_act_segment<6, 0, 25>(w17, row_buf[pix + 22]);
    copy_act_segment<31, 0, 1>(w17, row_buf[pix + 23]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(544), w17, ok);
    act_vec_t w18 = 0;
    copy_act_segment<0, 1, 24>(w18, row_buf[pix + 23]);
    copy_act_segment<24, 0, 8>(w18, row_buf[pix + 24]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(576), w18, ok);
    act_vec_t w19 = 0;
    copy_act_segment<0, 8, 17>(w19, row_buf[pix + 24]);
    copy_act_segment<17, 0, 15>(w19, row_buf[pix + 25]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(608), w19, ok);
    act_vec_t w20 = 0;
    copy_act_segment<0, 15, 10>(w20, row_buf[pix + 25]);
    copy_act_segment<10, 0, 22>(w20, row_buf[pix + 26]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(640), w20, ok);
    act_vec_t w21 = 0;
    copy_act_segment<0, 22, 3>(w21, row_buf[pix + 26]);
    copy_act_segment<3, 0, 25>(w21, row_buf[pix + 27]);
    copy_act_segment<28, 0, 4>(w21, row_buf[pix + 28]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(672), w21, ok);
    act_vec_t w22 = 0;
    copy_act_segment<0, 4, 21>(w22, row_buf[pix + 28]);
    copy_act_segment<21, 0, 11>(w22, row_buf[pix + 29]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(704), w22, ok);
    act_vec_t w23 = 0;
    copy_act_segment<0, 11, 14>(w23, row_buf[pix + 29]);
    copy_act_segment<14, 0, 18>(w23, row_buf[pix + 30]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(736), w23, ok);
    act_vec_t w24 = 0;
    copy_act_segment<0, 18, 7>(w24, row_buf[pix + 30]);
    copy_act_segment<7, 0, 25>(w24, row_buf[pix + 31]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(768), w24, ok);
  }
  return ok;
}

static bool store_compact_c28_row(const tensor_desc_t& dst,
                                  u16_t out_row,
                                  int out_w_i,
                                  act_vec_t row_buf[MAX_FM_W]) {
#pragma HLS INLINE off
  bool ok = true;
  for (int group = 0; group < MAX_FM_W / 8; ++group) {
#pragma HLS PIPELINE off
    const int pix = group * 8;
    if (pix >= out_w_i) {
      break;
    }
    const u32_t byte_offset = static_cast<u32_t>(group) * static_cast<u32_t>(224);
    act_vec_t w0 = 0;
    copy_act_segment<0, 0, 28>(w0, row_buf[pix + 0]);
    copy_act_segment<28, 0, 4>(w0, row_buf[pix + 1]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(0), w0, ok);
    act_vec_t w1 = 0;
    copy_act_segment<0, 4, 24>(w1, row_buf[pix + 1]);
    copy_act_segment<24, 0, 8>(w1, row_buf[pix + 2]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(32), w1, ok);
    act_vec_t w2 = 0;
    copy_act_segment<0, 8, 20>(w2, row_buf[pix + 2]);
    copy_act_segment<20, 0, 12>(w2, row_buf[pix + 3]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(64), w2, ok);
    act_vec_t w3 = 0;
    copy_act_segment<0, 12, 16>(w3, row_buf[pix + 3]);
    copy_act_segment<16, 0, 16>(w3, row_buf[pix + 4]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(96), w3, ok);
    act_vec_t w4 = 0;
    copy_act_segment<0, 16, 12>(w4, row_buf[pix + 4]);
    copy_act_segment<12, 0, 20>(w4, row_buf[pix + 5]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(128), w4, ok);
    act_vec_t w5 = 0;
    copy_act_segment<0, 20, 8>(w5, row_buf[pix + 5]);
    copy_act_segment<8, 0, 24>(w5, row_buf[pix + 6]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(160), w5, ok);
    act_vec_t w6 = 0;
    copy_act_segment<0, 24, 4>(w6, row_buf[pix + 6]);
    copy_act_segment<4, 0, 28>(w6, row_buf[pix + 7]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(192), w6, ok);
  }
  return ok;
}

static bool store_c16_into_c19_row(const tensor_desc_t& dst,
                                   u16_t out_row,
                                   int out_w_i,
                                   act_vec_t row_buf[MAX_FM_W]) {
#pragma HLS INLINE off
  bool ok = true;
  for (int group = 0; group < MAX_FM_W / 32; ++group) {
#pragma HLS PIPELINE off
    const int pix = group * 32;
    if (pix >= out_w_i) {
      break;
    }
    const u32_t byte_offset = static_cast<u32_t>(group) * static_cast<u32_t>(608);

    act_vec_t w0 = 0;
    copy_act_segment<0, 0, 16>(w0, row_buf[pix + 0]);
    copy_act_segment<19, 0, 13>(w0, row_buf[pix + 1]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(0), w0, ok);
    act_vec_t w1 = 0;
    copy_act_segment<0, 13, 3>(w1, row_buf[pix + 1]);
    copy_act_segment<6, 0, 16>(w1, row_buf[pix + 2]);
    copy_act_segment<25, 0, 7>(w1, row_buf[pix + 3]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(32), w1, ok);
    act_vec_t w2 = 0;
    copy_act_segment<0, 7, 9>(w2, row_buf[pix + 3]);
    copy_act_segment<12, 0, 16>(w2, row_buf[pix + 4]);
    copy_act_segment<31, 0, 1>(w2, row_buf[pix + 5]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(64), w2, ok);
    act_vec_t w3 = 0;
    copy_act_segment<0, 1, 15>(w3, row_buf[pix + 5]);
    copy_act_segment<18, 0, 14>(w3, row_buf[pix + 6]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(96), w3, ok);
    act_vec_t w4 = 0;
    copy_act_segment<0, 14, 2>(w4, row_buf[pix + 6]);
    copy_act_segment<5, 0, 16>(w4, row_buf[pix + 7]);
    copy_act_segment<24, 0, 8>(w4, row_buf[pix + 8]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(128), w4, ok);
    act_vec_t w5 = 0;
    copy_act_segment<0, 8, 8>(w5, row_buf[pix + 8]);
    copy_act_segment<11, 0, 16>(w5, row_buf[pix + 9]);
    copy_act_segment<30, 0, 2>(w5, row_buf[pix + 10]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(160), w5, ok);
    act_vec_t w6 = 0;
    copy_act_segment<0, 2, 14>(w6, row_buf[pix + 10]);
    copy_act_segment<17, 0, 15>(w6, row_buf[pix + 11]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(192), w6, ok);
    act_vec_t w7 = 0;
    copy_act_segment<0, 15, 1>(w7, row_buf[pix + 11]);
    copy_act_segment<4, 0, 16>(w7, row_buf[pix + 12]);
    copy_act_segment<23, 0, 9>(w7, row_buf[pix + 13]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(224), w7, ok);
    act_vec_t w8 = 0;
    copy_act_segment<0, 9, 7>(w8, row_buf[pix + 13]);
    copy_act_segment<10, 0, 16>(w8, row_buf[pix + 14]);
    copy_act_segment<29, 0, 3>(w8, row_buf[pix + 15]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(256), w8, ok);
    act_vec_t w9 = 0;
    copy_act_segment<0, 3, 13>(w9, row_buf[pix + 15]);
    copy_act_segment<16, 0, 16>(w9, row_buf[pix + 16]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(288), w9, ok);
    act_vec_t w10 = 0;
    copy_act_segment<3, 0, 16>(w10, row_buf[pix + 17]);
    copy_act_segment<22, 0, 10>(w10, row_buf[pix + 18]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(320), w10, ok);
    act_vec_t w11 = 0;
    copy_act_segment<0, 10, 6>(w11, row_buf[pix + 18]);
    copy_act_segment<9, 0, 16>(w11, row_buf[pix + 19]);
    copy_act_segment<28, 0, 4>(w11, row_buf[pix + 20]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(352), w11, ok);
    act_vec_t w12 = 0;
    copy_act_segment<0, 4, 12>(w12, row_buf[pix + 20]);
    copy_act_segment<15, 0, 16>(w12, row_buf[pix + 21]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(384), w12, ok);
    act_vec_t w13 = 0;
    copy_act_segment<2, 0, 16>(w13, row_buf[pix + 22]);
    copy_act_segment<21, 0, 11>(w13, row_buf[pix + 23]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(416), w13, ok);
    act_vec_t w14 = 0;
    copy_act_segment<0, 11, 5>(w14, row_buf[pix + 23]);
    copy_act_segment<8, 0, 16>(w14, row_buf[pix + 24]);
    copy_act_segment<27, 0, 5>(w14, row_buf[pix + 25]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(448), w14, ok);
    act_vec_t w15 = 0;
    copy_act_segment<0, 5, 11>(w15, row_buf[pix + 25]);
    copy_act_segment<14, 0, 16>(w15, row_buf[pix + 26]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(480), w15, ok);
    act_vec_t w16 = 0;
    copy_act_segment<1, 0, 16>(w16, row_buf[pix + 27]);
    copy_act_segment<20, 0, 12>(w16, row_buf[pix + 28]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(512), w16, ok);
    act_vec_t w17 = 0;
    copy_act_segment<0, 12, 4>(w17, row_buf[pix + 28]);
    copy_act_segment<7, 0, 16>(w17, row_buf[pix + 29]);
    copy_act_segment<26, 0, 6>(w17, row_buf[pix + 30]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(544), w17, ok);
    act_vec_t w18 = 0;
    copy_act_segment<0, 6, 10>(w18, row_buf[pix + 30]);
    copy_act_segment<13, 0, 16>(w18, row_buf[pix + 31]);
    write_compact_row_word(dst, out_row, byte_offset + static_cast<u32_t>(576), w18, ok);
  }
  return ok;
}

bool store_conv_output_row(const tensor_desc_t& dst,
                                  u16_t out_row,
                                  u16_t c_offset,
                                  const conv_cfg_t& cfg,
                                  act_vec_t row_buf[MAX_FM_W]) {
#pragma HLS INLINE off
  const u16_t stride = conv_effective_stride(cfg);
  const u16_t out_w = conv_out_dim(cfg.in_w, stride);
  const int out_w_i = static_cast<int>(out_w.to_uint());
  const u8_t valid_c = static_cast<u8_t>(cfg.out_c.to_uint());
  bool ok = true;

  const bool compact_row =
      c_offset.to_uint() == 0U &&
      core_desc_phys_c(dst).to_uint() == dst.c.to_uint() &&
      dst.c.to_uint() == cfg.out_c.to_uint();
  if (compact_row && cfg.out_c.to_uint() == 12U &&
      (out_w_i & 7) == 0) {
    return store_compact_c12_row(dst, out_row, out_w_i, row_buf);
  }
  if (compact_row && cfg.out_c.to_uint() == 16U &&
      (out_w_i & 1) == 0) {
    return store_compact_c16_row(dst, out_row, out_w_i, row_buf);
  }
  if (compact_row && cfg.out_c.to_uint() == 25U &&
      (out_w_i & 31) == 0) {
    return store_compact_c25_row(dst, out_row, out_w_i, row_buf);
  }
  if (compact_row && cfg.out_c.to_uint() == 28U &&
      (out_w_i & 7) == 0) {
    return store_compact_c28_row(dst, out_row, out_w_i, row_buf);
  }
  if (compact_row && cfg.out_c.to_uint() == 2U &&
      (out_w_i & 15) == 0) {
    return store_compact_c2_row(dst, out_row, out_w_i, row_buf);
  }
  if (c_offset.to_uint() == 0U &&
      cfg.out_c.to_uint() == 16U &&
      dst.c.to_uint() == 19U &&
      core_desc_phys_c(dst).to_uint() == 19U &&
      (out_w_i & 31) == 0) {
    return store_c16_into_c19_row(dst, out_row, out_w_i, row_buf);
  }

  for (int ow_i = 0; ow_i < MAX_FM_W; ++ow_i) {
#pragma HLS PIPELINE off
    if (ow_i >= out_w_i) {
      break;
    }
    const act_vec_t packed = row_buf[ow_i];
    if (!on_chip_memory_write_packed_tile(dst,
                                          out_row,
                                          static_cast<u16_t>(ow_i),
                                          c_offset,
                                          valid_c,
                                          packed)) {
      ok = false;
    }
  }
  return ok;
}


}  // namespace esp_int8
