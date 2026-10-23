#include "../include/npu_config.hpp"
#include "../include/npu_ctrl.hpp"
#include "../include/npu_uop.hpp"
#include "../include/npu_schedule.hpp"
#include "../include/npu_q.hpp"
#include "top_call.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace esp_int8 {
bool param_dma_ready();
unsigned csim_last_uop();
unsigned csim_last_error();
bool on_chip_memory_read_fmbuf_abs_word(u8_t bank, u32_t offset, axi_vec_t& word);
bool on_chip_memory_write_fmbuf_abs_word(u8_t bank, u32_t offset, axi_vec_t word);
bool param_dma_get_affine_qparam(u8_t param_id, u8_t block, aff_q_t& qparam);
bool param_dma_get_fixed_exec_desc(u8_t id, fixed_exec_desc_t& desc);
bool resolve_tensor_read(u8_t tensor_id, tensor_desc_t& desc);
bool resolve_tensor_write(u8_t tensor_id, u16_t h, u16_t w, u16_t c, tensor_desc_t& desc);
void reset_scratch_state();
bool on_chip_memory_read_aligned_tensor_word(const tensor_desc_t& desc,
                                              u32_t offset, axi_vec_t& word);
bool on_chip_memory_write_pool_bram_word(u8_t bank, u32_t offset, axi_vec_t word);
bool conv_store_write_row_contiguous_word(const tensor_desc_t& dst, u32_t offset,
                                          const axi_vec_t& word);
bool conv_store_word_span_plan_ok(u8_t bank, u64_t first, u32_t step, u16_t count);
error_code_t vec_alu_run_fixed_issue(const npu_issue_t& issue, axi_vec_t* frame_out);
bool ppu_consume_block5_final_row(const block5_sched_desc_t& sched,
                                  const tensor_desc_t& s0, const tensor_desc_t& s1,
                                  const tensor_desc_t& s2, const tensor_desc_t& s3,
                                  const tensor_desc_t& residual, const tensor_desc_t& dst,
                                  bool has_residual, const add_q_t& add_qparam,
                                  u8_t act_type, u16_t local_row, u16_t abs_row,
                                  act_vec_t row_buf[MAX_FM_W]);
void ppu_consume_conv_stream(const row_consumer_desc_t& consumer,
                             const tensor_desc_t& dst,
                             const tensor_desc_t& add_other,
                             bool has_add_other,
                             const add_q_t& add_qparam,
                             const conv_cfg_t& cfg,
                             u16_t out_row,
                             hls::stream<act_vec_t>& conv_stream,
                             u8_t& status);
void upsample_fused_begin();
void upsample_fused_consume_logits_row(axi_vec_t* frame_out,
                                       u16_t row, u8_t valid_c,
                                       const act_vec_t row_buf[MAX_FM_W]);
}

#ifndef ESP_INT8_CSIM_CLASS_COUNT
#define ESP_INT8_CSIM_CLASS_COUNT 2
#endif

static bool read_binary(const char* path, std::vector<std::uint8_t>& bytes) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::printf("[FAIL] failed to open %s\n", path);
    return false;
  }
  bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  return true;
}

static bool write_binary(const char* path, const std::vector<std::uint8_t>& bytes) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    std::printf("[FAIL] failed to write %s\n", path);
    return false;
  }
  out.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  return static_cast<bool>(out);
}

static void set_byte(esp_int8::axi_vec_t& word, int lane, std::uint8_t value) {
  word.range(lane * 8 + 7, lane * 8) = value;
}

static std::uint8_t get_byte(const esp_int8::axi_vec_t& word, int lane) {
  return static_cast<std::uint8_t>(word.range(lane * 8 + 7, lane * 8).to_uint());
}

static void pack_bytes(const std::vector<std::uint8_t>& bytes,
                       esp_int8::axi_vec_t* words,
                       std::size_t word_count) {
  for (std::size_t i = 0; i < word_count; ++i) {
    words[i] = 0;
  }
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    set_byte(words[i / esp_int8::AXI_WORD_BYTES],
             static_cast<int>(i % esp_int8::AXI_WORD_BYTES),
             bytes[i]);
  }
}

static std::int8_t as_i8(std::uint8_t value) {
  return static_cast<std::int8_t>(value);
}

static bool check_bounded_int8_math() {
  using namespace esp_int8;
  const std::int32_t coeffs[] = {-2147483647 - 1, -65537, -1, 0, 1, 65537, 2147483647};
  const std::int32_t biases[] = {-2147483647 - 1, -1, 0, 1, 2147483647};
  const int edges[] = {-128, -127, -65, -1, 0, 1, 63, 126, 127};
  unsigned cases = 0;
  for (int value = -128; value <= 127; ++value) {
    for (std::int32_t mul : coeffs) {
      for (std::int32_t bias : biases) {
        for (unsigned shift = 0; shift <= 31; ++shift) {
          for (unsigned act = 0; act < 2; ++act) {
            const i8_t got = affine_i8_to_i8_bounded(static_cast<i8_t>(value),
                static_cast<i32_t>(mul), static_cast<i32_t>(bias),
                static_cast<u8_t>(shift), static_cast<u8_t>(act));
            const i8_t expected = affine_i8_to_i8(static_cast<i8_t>(value),
                static_cast<i32_t>(mul), static_cast<i32_t>(bias),
                static_cast<u8_t>(shift), static_cast<u8_t>(act));
            if (got != expected) {
              std::printf("[FAIL] bounded affine value=%d mul=%d bias=%d shift=%u act=%u\n",
                          value, mul, bias, shift, act);
              return false;
            }
            ++cases;
          }
        }
      }
    }
  }
  for (int lhs : edges) {
    for (int rhs : edges) {
      for (std::int32_t mul : coeffs) {
        for (unsigned shift = 0; shift <= 31; ++shift) {
          for (unsigned mode = 0; mode < 4; ++mode) {
            add_q_t q = add_q_t();
            q.mult = mul;
            q.shift = shift;
            q.requant_bypass = mode & 1U;
            q.act_type = (mode >> 1U) & 1U;
            if (add_i8_bounded(static_cast<i8_t>(lhs), static_cast<i8_t>(rhs), q) !=
                add_i8(static_cast<i8_t>(lhs), static_cast<i8_t>(rhs), q)) {
              std::printf("[FAIL] bounded add lhs=%d rhs=%d mul=%d shift=%u mode=%u\n",
                          lhs, rhs, mul, shift, mode);
              return false;
            }
            ++cases;
          }
        }
      }
    }
  }
  for (unsigned shift = 1; shift <= 31; ++shift) {
    const std::int64_t unit = std::int64_t(1) << shift;
    for (int multiple = -3; multiple <= 3; ++multiple) {
      for (int delta = -1; delta <= 1; ++delta) {
        const std::int64_t value = multiple * unit + unit / 2 + delta;
        const i32_t expected = round_shift(static_cast<i64_t>(value), static_cast<u8_t>(shift));
        if (round_shift_bounded<40>(static_cast<ap_int<40>>(value), static_cast<u8_t>(shift)) != expected ||
            round_shift_bounded<41>(static_cast<ap_int<41>>(value), static_cast<u8_t>(shift)) != expected) {
          std::printf("[FAIL] bounded rounding half-tie shift=%u multiple=%d delta=%d\n",
                      shift, multiple, delta);
          return false;
        }
        cases += 2;
      }
    }
  }
  std::printf("[PASS] bounded INT8 math: %u signed/half-tie/shift/INT32-wrap/add-boundary cases\n", cases);
  return true;
}

static bool check_vec_span_plans() {
  using namespace esp_int8;
  struct SpanCase { unsigned bank; std::uint64_t first; unsigned step; unsigned count; bool valid; };
  const SpanCase cases[] = {
    {BANK_FMEM0, 0, 32, 1, true},
    {BANK_FMEM1, FMBUF_BYTES - 32, 32, 1, true},
    {BANK_FMEM2, FMBUF_URAM_BYTES - 32, 32, 2, true},
    {BANK_FMEM0, FMBUF_BYTES - 32, 32, 2, false},
    {BANK_FMEM0, FMBUF_BYTES, 32, 1, false},
    {BANK_FMEM0, 0, 32, ROW_CONTIG_MAX_WORDS, true},
    {BANK_FMEM0, 0, 32, ROW_CONTIG_MAX_WORDS + 1, false},
    {BANK_FMEM0, 0, 32, 0, false},
    {BANK_FMEM0, 1, 32, 1, false},
    {BANK_FMEM0, 0, 0, 1, false},
    {BANK_FMEM0, 0, 16, 1, false},
    {BANK_FMEM0, 0, 33, 1, false},
    {BANK_FMEM0, 0xffffffe0ULL, 32, 2, false},
    {BANK_FMEM0, 0x100000000ULL, 32, 1, false},
    {BANK_FMEM0, 0, 0xffffffe0U, ROW_CONTIG_MAX_WORDS, false},
    {BANK_FMEM0, 0xffffffffffffffe0ULL, 32, 2, false},
    {BANK_BRAM_SCR0, 0, 32, 1, true},
    {BANK_BRAM_SCR0, FMBUF_BYTES - FMBUF_POOL_TMP_BASE - 32, 32, 1, true},
    {BANK_BRAM_SCR0, FMBUF_BYTES - FMBUF_POOL_TMP_BASE, 32, 1, false},
    {BANK_BRAM_SCR1, BRAM_SCR1_BYTES - 32, 32, 1, true},
    {BANK_BRAM_SCR1, BRAM_SCR1_BYTES - 32, 32, 2, false},
    {BANK_BRAM_SCR1, 0, 32, BRAM_SCR1_AXI_WORDS, true},
    {BANK_FMEM0, 0, 256, 30, true},
    {127, 0, 32, 1, false},
  };
  for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    const SpanCase& test = cases[i];
    const bool valid = conv_store_word_span_plan_ok(static_cast<u8_t>(test.bank),
        static_cast<u64_t>(test.first), static_cast<u32_t>(test.step), static_cast<u16_t>(test.count));
    if (valid != test.valid) {
      std::printf("[FAIL] Vec span validation case=%u got=%u expected=%u\n", i, valid, test.valid);
      return false;
    }
    if (valid) {
      tensor_desc_t dst = tensor_desc_t();
      dst.bank_id = static_cast<u8_t>(test.bank);
      axi_vec_t word = 0;
      for (unsigned n = 0; n < test.count; ++n) {
        if (!conv_store_write_row_contiguous_word(dst,
                static_cast<u32_t>(test.first + std::uint64_t(n) * test.step), word)) {
          std::printf("[FAIL] accepted Vec span rejected by checked writer case=%u word=%u\n", i, n);
          return false;
        }
      }
    }
  }
  std::printf("[PASS] Vec span: bank boundaries, URAM/BRAM crossing, stride, counts and overflow\n");
  return true;
}

static bool check_vec_alias_case(unsigned fixed_id, unsigned seed) {
  using namespace esp_int8;
  reset_scratch_state();
  fixed_exec_desc_t fixed;
  tensor_desc_t src[3];
  tensor_desc_t dst;
  if (!param_dma_get_fixed_exec_desc(static_cast<u8_t>(fixed_id), fixed) ||
      !resolve_tensor_read(fixed.src0_tensor, src[0]) ||
      !resolve_tensor_read(fixed.src1_tensor, src[1]) ||
      !resolve_tensor_write(fixed.dst_tensor, fixed.in_h, fixed.in_w, fixed.valid_c, dst)) {
    return false;
  }
  const bool compact =
      (fixed.flags.to_uint() & static_cast<unsigned>(FIXED_FLAG_ROW_CONTIGUOUS_STORE)) != 0U;
  const unsigned source_count = fixed.reserved0.to_uint() == static_cast<unsigned>(TID_INVALID) ? 2U : 3U;
  if (source_count == 3U && !resolve_tensor_read(static_cast<u8_t>(fixed.reserved0.to_uint()), src[2])) {
    return false;
  }
  const unsigned h = fixed.in_h.to_uint();
  const unsigned w = fixed.in_w.to_uint();
  const unsigned channels = fixed.valid_c.to_uint();
  for (unsigned s = 0; s < source_count; ++s) {
    const tensor_desc_t& source = src[s];
    const unsigned begin = source.base_offset.to_uint() + source.reserved1.to_uint();
    const unsigned end = begin + (h - 1U) * tensor_row_stride_bytes(source).to_uint() +
        (w - 1U) * tensor_phys_c(source).to_uint() + source.c.to_uint();
    for (unsigned offset = begin & ~31U; offset < ((end + 31U) & ~31U); offset += 32U) {
      axi_vec_t word = 0;
      for (unsigned lane = 0; lane < AXI_WORD_BYTES; ++lane) {
        set_byte(word, lane, static_cast<std::uint8_t>((offset + lane) * 37U + seed * 53U));
      }
      const bool ok = source.bank_id.to_uint() == static_cast<unsigned>(BANK_BRAM_SCR1)
          ? on_chip_memory_write_pool_bram_word(source.bank_id, offset, word)
          : on_chip_memory_write_fmbuf_abs_word(source.bank_id, offset, word);
      if (!ok) { return false; }
    }
  }

  // Snapshot all inputs before the destination overwrites either aliased source.
  std::vector<std::uint8_t> inputs[3];
  for (unsigned s = 0; s < source_count; ++s) {
    const tensor_desc_t& source = src[s];
    axi_vec_t cached = 0;
    unsigned cached_offset = ~0U;
    for (unsigned y = 0; y < h; ++y) {
      for (unsigned x = 0; x < w; ++x) {
        for (unsigned c = 0; c < source.c.to_uint(); ++c) {
          const unsigned offset = source.base_offset.to_uint() + source.reserved1.to_uint() +
              y * tensor_row_stride_bytes(source).to_uint() +
              x * tensor_phys_c(source).to_uint() + c;
          const unsigned aligned = offset & ~31U;
          if (cached_offset != aligned) {
            if (!on_chip_memory_read_aligned_tensor_word(source, aligned, cached)) { return false; }
            cached_offset = aligned;
          }
          inputs[s].push_back(get_byte(cached, offset & 31U));
        }
      }
    }
  }
  const unsigned dst_stride = tensor_row_stride_bytes(dst).to_uint();
  const unsigned dst_begin = dst.base_offset.to_uint() + dst.reserved1.to_uint();
  std::vector<std::uint8_t> expected(h * dst_stride + AXI_WORD_BYTES);
  for (unsigned offset = 0; offset < expected.size(); offset += AXI_WORD_BYTES) {
    axi_vec_t word = 0;
    if (!on_chip_memory_read_aligned_tensor_word(dst, dst_begin + offset, word)) { return false; }
    for (unsigned lane = 0; lane < AXI_WORD_BYTES; ++lane) {
      expected[offset + lane] = get_byte(word, lane);
    }
  }
  std::vector<aff_q_t> qparams((channels + TM - 1U) / TM);
  for (unsigned block = 0; block < qparams.size(); ++block) {
    if (!param_dma_get_affine_qparam(fixed.param_id, static_cast<u8_t>(block), qparams[block])) {
      return false;
    }
  }
  for (unsigned y = 0; y < h; ++y) {
    for (unsigned x = 0; x < w; ++x) {
      for (unsigned c = 0; c < channels; ++c) {
        unsigned source = 0;
        unsigned local = c;
        while (source < source_count && local >= src[source].c.to_uint()) {
          local -= src[source++].c.to_uint();
        }
        if (source >= source_count) { return false; }
        const i8_t value = as_i8(inputs[source][(y * w + x) * src[source].c.to_uint() + local]);
        const aff_q_t& qparam = qparams[c / TM];
        const i8_t result = affine_i8_to_i8(value, qparam.mul[c % TM], qparam.bias[c % TM],
                                            qparam.shift[c % TM], fixed.act_type);
        expected[y * dst_stride + x * tensor_phys_c(dst).to_uint() + c] =
            static_cast<std::uint8_t>(result.range(7, 0).to_uint());
      }
    }
    if (compact) {
      for (unsigned pad = w * channels; pad < dst_stride; ++pad) {
        expected[y * dst_stride + pad] = 0;
      }
    }
  }
  npu_issue_t issue = npu_issue_t();
  issue.kind = static_cast<u8_t>(static_cast<unsigned>(ISSUE_VEC_AFFINE));
  issue.fixed_desc_id = static_cast<u8_t>(fixed_id);
  if (vec_alu_run_fixed_issue(issue, nullptr) != ERR_NONE) { return false; }
  for (unsigned offset = 0; offset < expected.size(); offset += AXI_WORD_BYTES) {
    axi_vec_t word = 0;
    if (!on_chip_memory_read_aligned_tensor_word(dst, dst_begin + offset, word)) { return false; }
    for (unsigned lane = 0; lane < AXI_WORD_BYTES; ++lane) {
      if (get_byte(word, lane) != expected[offset + lane]) {
        std::printf("[FAIL] Vec alias fixed=%u seed=%u byte=%u got=%u expected=%u\n",
                    fixed_id, seed, offset + lane, get_byte(word, lane), expected[offset + lane]);
        return false;
      }
    }
  }
  return true;
}

static bool check_vec_bad_destinations(const std::vector<std::uint8_t>& original,
                                        esp_int8::axi_vec_t* param,
                                        const esp_int8::axi_vec_t* frame_in,
                                        esp_int8::axi_vec_t* frame_out) {
  using namespace esp_int8;
  const unsigned tensor_section = unsigned(original.at(40)) | (unsigned(original.at(41)) << 8) |
      (unsigned(original.at(42)) << 16) | (unsigned(original.at(43)) << 24);
  struct BadDest { unsigned fixed_id; unsigned tensor; unsigned bank; unsigned base; };
  const BadDest cases[] = {
    {5, 10, BANK_FMEM0, FMBUF_BYTES - 32U},
    {8, 16, BANK_FMEM0, FMBUF_BYTES - 32U},
    {5, 10, BANK_FMEM0, 0xffffffe0U},
    {8, 16, BANK_FMEM0, 0xffffffe0U},
    {5, 10, 127, 0},
    {8, 16, 127, 0},
  };
  const axi_vec_t canary = ~static_cast<axi_vec_t>(0);
  for (const auto& test : cases) {
    auto altered = original;
    const unsigned offset = tensor_section + test.tensor * 16U;
    altered.at(offset) = static_cast<std::uint8_t>(test.bank);
    for (unsigned byte = 0; byte < 4; ++byte) {
      altered.at(offset + 4U + byte) = static_cast<std::uint8_t>(test.base >> (byte * 8U));
    }
    pack_bytes(altered, param, 8192U);
    call_espnet_encoder_int8_core(frame_in, frame_out, param, MODE_INIT, UOP_COUNT_ENCODER);
    if (!param_dma_ready()) { return false; }
    reset_scratch_state();
    if (!on_chip_memory_write_fmbuf_abs_word(static_cast<u8_t>(static_cast<unsigned>(BANK_FMEM0)),
                                               0, canary)) { return false; }
    npu_issue_t issue = npu_issue_t();
    issue.kind = static_cast<u8_t>(static_cast<unsigned>(ISSUE_VEC_AFFINE));
    issue.fixed_desc_id = static_cast<u8_t>(test.fixed_id);
    const error_code_t error = vec_alu_run_fixed_issue(issue, nullptr);
    axi_vec_t got = 0;
    if (error != ERR_BANK_OVERFLOW ||
        !on_chip_memory_read_fmbuf_abs_word(static_cast<u8_t>(static_cast<unsigned>(BANK_FMEM0)), 0, got) ||
        got != canary) {
      std::printf("[FAIL] Vec bad destination fixed=%u bank=%u base=%08x error=%u\n",
                  test.fixed_id, test.bank, test.base, static_cast<unsigned>(error));
      return false;
    }
  }
  pack_bytes(original, param, 8192U);
  call_espnet_encoder_int8_core(frame_in, frame_out, param, MODE_INIT, UOP_COUNT_ENCODER);
  std::printf("[PASS] Vec U39/U71 reject invalid bank, frame overrun and wrapped base before commit\n");
  return param_dma_ready();
}

static bool check_block5_cache_row(bool l3, unsigned row, bool residual_enabled) {
  using namespace esp_int8;
  const unsigned counts[5] = {l3 ? 28U : 16U, l3 ? 25U : 12U, l3 ? 25U : 12U,
                              l3 ? 25U : 12U, l3 ? 25U : 12U};
  tensor_desc_t sources[4];
  for (unsigned source = 0; source < 4; ++source) {
    tensor_desc_t& desc = sources[source];
    desc = tensor_desc_t();
    desc.bank_id = static_cast<u8_t>(static_cast<unsigned>(BANK_FMEM0));
    desc.elem_bytes = 1;
    desc.h = 3;
    desc.w = 3;
    desc.c = counts[source];
    desc.reserved0 = TENSOR_ROW_ALIGN32 | counts[source];
    desc.reserved1 = 16;
    desc.base_offset = 0x24000U + source * 0x1000U;
    const unsigned stride = tensor_row_stride_bytes(desc).to_uint();
    std::vector<std::uint8_t> data(16U + 3U * stride + 31U, 0);
    for (unsigned y = 0; y < 3; ++y) {
      for (unsigned x = 0; x < 3; ++x) {
        for (unsigned c = 0; c < counts[source]; ++c) {
          data[16U + y * stride + x * counts[source] + c] =
              static_cast<std::uint8_t>(13U + source * 47U + y * 9U + x * 5U + c * 3U);
        }
      }
    }
    for (unsigned byte = 0; byte + AXI_WORD_BYTES <= data.size(); byte += AXI_WORD_BYTES) {
      axi_vec_t word = 0;
      for (unsigned lane = 0; lane < AXI_WORD_BYTES; ++lane) {
        set_byte(word, lane, data[byte + lane]);
      }
      if (!on_chip_memory_write_fmbuf_abs_word(desc.bank_id, desc.base_offset + byte, word)) {
        return false;
      }
    }
  }
  tensor_desc_t dst = tensor_desc_t();
  dst.bank_id = static_cast<u8_t>(static_cast<unsigned>(BANK_FMEM0));
  dst.elem_bytes = 1;
  dst.h = 3;
  dst.w = 3;
  dst.c = l3 ? 128U : 64U;
  dst.reserved0 = dst.c;
  dst.base_offset = 0x20000U;
  tensor_desc_t residual = dst;
  residual.base_offset = 0x22000U;
  for (unsigned byte = 0; byte < 3U * tensor_row_stride_bytes(residual).to_uint();
       byte += AXI_WORD_BYTES) {
    axi_vec_t word = 0;
    for (unsigned lane = 0; lane < AXI_WORD_BYTES; ++lane) {
      set_byte(word, lane, static_cast<std::uint8_t>((byte + lane) * 17U + 31U));
    }
    if (!on_chip_memory_write_fmbuf_abs_word(residual.bank_id, residual.base_offset + byte, word)) {
      return false;
    }
  }
  act_vec_t row_buf[MAX_FM_W];
  for (unsigned x = 0; x < 3; ++x) {
    row_buf[x] = 0;
    for (unsigned c = 0; c < counts[4]; ++c) {
      set_byte(row_buf[x], c, static_cast<std::uint8_t>(19U + x * 11U + c * 7U));
    }
  }
  block5_sched_desc_t sched = block5_sched_desc_t();
  sched.pattern = static_cast<u8_t>(l3 ? static_cast<unsigned>(BLOCK5_PATTERN_L3_C28_4C25)
                                       : static_cast<unsigned>(BLOCK5_PATTERN_L2_C16_4C12));
  sched.out_w = 3;
  sched.valid_c = dst.c;
  sched.affine_param_id = l3 ? 4 : 1;
  add_q_t add_qparam = add_q_t();
  add_qparam.requant_bypass = 1;
  const bool ok = ppu_consume_block5_final_row(sched, sources[0], sources[1], sources[2],
                                               sources[3], residual, dst, residual_enabled,
                                               add_qparam, static_cast<u8_t>(0),
                                               static_cast<u16_t>(row), static_cast<u16_t>(row),
                                               row_buf);
  if (row >= 3U) { return !ok; }
  if (!ok) { return false; }
  for (unsigned x = 0; x < 3; ++x) {
    for (unsigned tile = 0; tile < dst.c.to_uint() / TM; ++tile) {
      aff_q_t qparam;
      if (!param_dma_get_affine_qparam(sched.affine_param_id, static_cast<u8_t>(tile), qparam)) {
        return false;
      }
      const unsigned offset = row * tensor_row_stride_bytes(dst).to_uint() +
                               x * dst.c.to_uint() + tile * TM;
      axi_vec_t got = 0;
      if (!on_chip_memory_read_fmbuf_abs_word(dst.bank_id, dst.base_offset + offset, got)) {
        return false;
      }
      for (unsigned lane = 0; lane < TM; ++lane) {
        unsigned channel = tile * TM + lane;
        unsigned source = 0;
        while (channel >= counts[source]) { channel -= counts[source++]; }
        int value = source == 4U
            ? as_i8(static_cast<std::uint8_t>(19U + x * 11U + channel * 7U))
            : as_i8(static_cast<std::uint8_t>(13U + source * 47U + row * 9U + x * 5U + channel * 3U));
        if (residual_enabled) {
          value += as_i8(static_cast<std::uint8_t>((offset + lane) * 17U + 31U));
          value = value > 127 ? 127 : (value < -128 ? -128 : value);
        }
        const i8_t expected = affine_i8_to_i8(static_cast<i8_t>(value), qparam.mul[lane],
                                              qparam.bias[lane], qparam.shift[lane],
                                              static_cast<u8_t>(0));
        if (get_byte(got, lane) != expected.range(7, 0).to_uint()) {
          std::printf("[FAIL] BLOCK5 cache L%u row=%u x=%u tile=%u lane=%u\n",
                      l3 ? 3U : 2U, row, x, tile, lane);
          return false;
        }
      }
    }
  }
  return true;
}

static bool check_block5_cache_reload(const std::vector<std::uint8_t>& original,
                                       esp_int8::axi_vec_t* param,
                                       const esp_int8::axi_vec_t* frame_in,
                                       esp_int8::axi_vec_t* frame_out) {
  using namespace esp_int8;
  if (!check_block5_cache_row(false, 0, false) ||
      !check_block5_cache_row(false, 1, true)) { return false; }
  auto altered = original;
  const unsigned affine_section = unsigned(altered.at(76)) | (unsigned(altered.at(77)) << 8) |
      (unsigned(altered.at(78)) << 16) | (unsigned(altered.at(79)) << 24);
  const unsigned first_block = affine_section + 320U;  // Param 0 has one 320-byte block.
  for (unsigned block = 0; block < 2; ++block) {
    for (unsigned lane = 0; lane < TM; ++lane) {
      const unsigned start = first_block + block * 320U;
      for (unsigned byte = 0; byte < 4; ++byte) {
        altered.at(start + lane * 4U + byte) = byte == 0U ? 1U : 0U;
        altered.at(start + 128U + lane * 4U + byte) = byte == 0U ? 23U : 0U;
      }
      altered.at(start + 256U + lane) = 0;
    }
  }
  pack_bytes(altered, param, 8192U);
  call_espnet_encoder_int8_core(frame_in, frame_out, param, MODE_INIT, UOP_COUNT_ENCODER);
  if (!param_dma_ready() || !check_block5_cache_row(false, 2, true) ||
      !check_block5_cache_row(true, 1, false) || !check_block5_cache_row(false, 0, true) ||
      !check_block5_cache_row(false, 2, false) || !check_block5_cache_row(false, 3, true) ||
      !check_block5_cache_row(false, 1, true)) { return false; }
  pack_bytes(original, param, 8192U);
  call_espnet_encoder_int8_core(frame_in, frame_out, param, MODE_INIT, UOP_COUNT_ENCODER);
  if (!param_dma_ready() || !check_block5_cache_row(false, 2, false)) { return false; }
  std::printf("[PASS] BLOCK5 coefficients: contiguous/gapped rows, pattern change, failed row, INIT reload\n");
  return true;
}

static void bilinear_axis_map(int out_idx,
                              int in_size,
                              int& idx0,
                              int& idx1,
                              int& w0,
                              int& w1) {
  const int scale = esp_int8::UPSAMPLE_SCALE;
  const int clip_high_start = in_size * scale - (scale / 2);
  if (out_idx < (scale / 2)) {
    idx0 = 0;
    idx1 = 0;
    w0 = 16;
    w1 = 0;
    return;
  }
  if (out_idx >= clip_high_start) {
    idx0 = in_size - 1;
    idx1 = in_size - 1;
    w0 = 16;
    w1 = 0;
    return;
  }

  const int numer = 2 * out_idx - (scale - 1);
  idx0 = numer >> 4;
  idx1 = idx0 + 1;
  w1 = numer & 0x0f;
  w0 = 16 - w1;
}

static int logit_at(const std::vector<std::uint8_t>& logits_nhwc,
                    int row,
                    int col,
                    int class_id,
                    int class_count) {
  const std::size_t off =
      (static_cast<std::size_t>(row) * esp_int8::ENCODER_OUT_W +
       static_cast<std::size_t>(col)) *
      static_cast<std::size_t>(class_count);
  return static_cast<int>(as_i8(logits_nhwc[off + static_cast<std::size_t>(class_id)]));
}

static std::vector<std::uint8_t> make_fullres_mask_golden(
    const std::vector<std::uint8_t>& logits_nhwc,
    int class_count) {
  std::vector<std::uint8_t> mask(esp_int8::FULLRES_MASK_BYTES, 0);
  for (int y = 0; y < esp_int8::FULLRES_MASK_H; ++y) {
    int y0 = 0;
    int y1 = 0;
    int wy0 = 16;
    int wy1 = 0;
    bilinear_axis_map(y, esp_int8::ENCODER_OUT_H, y0, y1, wy0, wy1);
    for (int x = 0; x < esp_int8::FULLRES_MASK_W; ++x) {
      int x0 = 0;
      int x1 = 0;
      int wx0 = 16;
      int wx1 = 0;
      bilinear_axis_map(x, esp_int8::ENCODER_OUT_W, x0, x1, wx0, wx1);

      int best_value = -0x7fffffff;
      std::uint8_t best_class = 0;
      for (int class_id = 0; class_id < class_count; ++class_id) {
        const int top = logit_at(logits_nhwc, y0, x0, class_id, class_count) * wx0 +
                        logit_at(logits_nhwc, y0, x1, class_id, class_count) * wx1;
        const int bottom = logit_at(logits_nhwc, y1, x0, class_id, class_count) * wx0 +
                           logit_at(logits_nhwc, y1, x1, class_id, class_count) * wx1;
        const int interp = top * wy0 + bottom * wy1;
        if (interp > best_value) {
          best_value = interp;
          best_class = static_cast<std::uint8_t>(class_id);
        }
      }
      mask[static_cast<std::size_t>(y) * esp_int8::FULLRES_MASK_W +
           static_cast<std::size_t>(x)] = best_class;
    }
  }
  return mask;
}

static bool check_compact_packing() {
  using namespace esp_int8;
  const unsigned channels[] = {12, 16, 17, 25, 28};
  const unsigned layouts[] = {STORE_LAYOUT_COMPACT_C12, STORE_LAYOUT_COMPACT_C16,
                              STORE_LAYOUT_COMPACT_C17, STORE_LAYOUT_COMPACT_C25,
                              STORE_LAYOUT_COMPACT_C28};
  unsigned cases = 0;
  for (unsigned shape = 0; shape < 5; ++shape) {
    for (unsigned width = 1; width <= MAX_FM_W; ++width) {
      if (width > 32U && width != MAX_FM_W) { continue; }
      ++cases;
      tensor_desc_t dst = tensor_desc_t();
      dst.bank_id = static_cast<u8_t>(static_cast<unsigned>(BANK_FMEM0));
      dst.base_offset = 2U * AXI_WORD_BYTES;
      dst.elem_bytes = 1;
      dst.h = 2;
      dst.w = width;
      dst.c = channels[shape];
      dst.reserved0 = TENSOR_ROW_ALIGN32 | channels[shape];
      row_consumer_desc_t consumer = row_consumer_desc_t();
      consumer.mode = static_cast<u8_t>(static_cast<unsigned>(ROW_CONSUMER_NONE));
      consumer.valid_c = channels[shape];
      consumer.reserved0 = layouts[shape];
      conv_cfg_t cfg = conv_cfg_t();
      cfg.in_h = 2;
      cfg.in_w = width;
      cfg.out_c = channels[shape];
      cfg.stride = 1;
      hls::stream<act_vec_t> words;
      std::vector<std::uint8_t> expected(tensor_row_stride_bytes(dst).to_uint(), 0);
      for (unsigned col = 0; col < width; ++col) {
        act_vec_t word = 0;
        for (unsigned lane = 0; lane < AXI_WORD_BYTES; ++lane) {
          const std::uint8_t value = static_cast<std::uint8_t>(
              17U + col * 41U + lane * 13U + shape * 7U);
          set_byte(word, static_cast<int>(lane), value);
          if (lane < channels[shape]) {
            expected[col * channels[shape] + lane] = value;
          }
        }
        words.write(word);
      }
      const tensor_desc_t unused = tensor_desc_t();
      const add_q_t unused_q = add_q_t();
      u8_t status = 0;
      ppu_consume_conv_stream(consumer, dst, unused, false, unused_q,
                               cfg, static_cast<u16_t>(1), words, status);
      if (status.to_uint() != 1U || !words.empty()) {
        std::printf("[FAIL] compact stream C%u W%u status/drain\n", channels[shape], width);
        return false;
      }
      const u32_t row_base = dst.base_offset + tensor_row_stride_bytes(dst);
      for (unsigned offset = 0; offset < expected.size(); offset += AXI_WORD_BYTES) {
        axi_vec_t word = 0;
        if (!on_chip_memory_read_fmbuf_abs_word(dst.bank_id, row_base + static_cast<u32_t>(offset), word)) {
          return false;
        }
        for (unsigned lane = 0; lane < AXI_WORD_BYTES; ++lane) {
          if (get_byte(word, static_cast<int>(lane)) != expected[offset + lane]) {
            std::printf("[FAIL] compact C%u W%u byte=%u\n",
                        channels[shape], width, offset + lane);
            return false;
          }
        }
      }
      // Invalid layout still consumes every upstream token and reports failure.
      consumer.reserved0 = static_cast<unsigned>(STORE_LAYOUT_NONE);
      for (unsigned col = 0; col < width; ++col) { words.write(static_cast<act_vec_t>(0)); }
      ppu_consume_conv_stream(consumer, dst, unused, false, unused_q,
                               cfg, static_cast<u16_t>(1), words, status);
      if (status.to_uint() != 0U || !words.empty()) {
        std::printf("[FAIL] invalid compact layout did not drain\n");
        return false;
      }
      if (width == MAX_FM_W) {
        // Allow two commits before overflowing; the rest of the input must still drain.
        consumer.reserved0 = layouts[shape];
        dst.base_offset = FMBUF_BYTES - 2U * AXI_WORD_BYTES;
        for (unsigned col = 0; col < width; ++col) { words.write(static_cast<act_vec_t>(0)); }
        ppu_consume_conv_stream(consumer, dst, unused, false, unused_q,
                                 cfg, static_cast<u16_t>(0), words, status);
        if (status.to_uint() != 0U || !words.empty()) {
          std::printf("[FAIL] compact C%u write overflow did not drain\n", channels[shape]);
          return false;
        }
      }
    }
  }
  std::printf("[PASS] compact C12/C16/C17/C25/C28: %u shapes, all tails, padding and error drain\n", cases);
  return true;
}

static bool check_binary_interpolation() {
  using namespace esp_int8;
  static axi_vec_t output[OUTPUT_FRAME_AXI_WORDS];
  static act_vec_t row_buf[MAX_FM_W];
  for (unsigned pattern = 0; pattern < 3; ++pattern) {
    std::vector<std::uint8_t> logits(ENCODER_OUT_H * ENCODER_OUT_W * 2);
    for (unsigned pixel = 0; pixel < ENCODER_OUT_H * ENCODER_OUT_W; ++pixel) {
      if (pattern == 0U) {
        logits[pixel * 2] = (pixel & 1U) == 0U ? 128U : 127U;
        logits[pixel * 2 + 1] = (pixel & 1U) == 0U ? 127U : 128U;
      } else {
        logits[pixel * 2] = static_cast<std::uint8_t>(pixel * 53U + 17U);
        logits[pixel * 2 + 1] = pattern == 2U ? logits[pixel * 2]
            : static_cast<std::uint8_t>(pixel * 97U + 251U);
      }
    }
    const auto expected = make_fullres_mask_golden(logits, 2);
    for (auto& word : output) { word = ~static_cast<axi_vec_t>(0); }
    upsample_fused_begin();
    for (int row = 0; row < ENCODER_OUT_H; ++row) {
      for (int col = 0; col < ENCODER_OUT_W; ++col) {
        row_buf[col] = 0;
        const int offset = (row * ENCODER_OUT_W + col) * 2;
        set_byte(row_buf[col], 0, logits[offset]);
        set_byte(row_buf[col], 1, logits[offset + 1]);
      }
      upsample_fused_consume_logits_row(output, static_cast<u16_t>(row),
                                         static_cast<u8_t>(2), row_buf);
    }
    for (unsigned pixel = 0; pixel < expected.size(); ++pixel) {
      if (get_byte(output[pixel / AXI_WORD_BYTES], pixel % AXI_WORD_BYTES) != expected[pixel]) {
        std::printf("[FAIL] bilinear pattern=%u pixel=%u\n", pattern, pixel);
        return false;
      }
    }
  }
  std::printf("[PASS] binary bilinear: extrema/random/ties, edge clipping, cross-row packing and reset\n");
  return true;
}

static void count_labels(const std::vector<std::uint8_t>& bytes,
                         int& zeros,
                         int& ones,
                         int& others,
                         int& aa) {
  zeros = 0;
  ones = 0;
  others = 0;
  aa = 0;
  for (std::uint8_t value : bytes) {
    if (value == 0U) {
      ++zeros;
    } else if (value == 1U) {
      ++ones;
    } else {
      ++others;
      if (value == 0xaaU) {
        ++aa;
      }
    }
  }
}

int main() {
  static constexpr int MAX_ALLOWED_MASK_MISMATCHES = 0;
  static_assert(ESP_INT8_CSIM_CLASS_COUNT == 2, "H240 grayscale release is binary2");
  if (!check_bounded_int8_math() || !check_compact_packing() ||
      !check_binary_interpolation() || !check_vec_span_plans()) { return 1; }
  const char* artifact_dir = "D:/ESP_INT8/hw_artifacts/binary2_int8_h240w240_gray_r6_v4";
  if (const char* selected = std::getenv("ESP_INT8_CSIM_ARTIFACT_DIR")) {
    artifact_dir = selected;
  }
  std::printf("[TB] artifact_dir=%s classes=%d\n", artifact_dir, ESP_INT8_CSIM_CLASS_COUNT);
  const std::string param_path = std::string(artifact_dir) + "/PARAM.BIN";
  const std::string input_path = std::string(artifact_dir) + "/input_q.bin";
  const std::string golden_logits_path =
      std::string(artifact_dir) + "/golden_output_q.bin";
  const char* hls_output_path = "hls_output_mask.bin";

  std::vector<std::uint8_t> param_bytes;
  std::vector<std::uint8_t> input_bytes;
  std::vector<std::uint8_t> golden_logits;
  if (!read_binary(param_path.c_str(), param_bytes) ||
      !read_binary(input_path.c_str(), input_bytes) ||
      !read_binary(golden_logits_path.c_str(), golden_logits)) {
    return 1;
  }
  if (input_bytes.size() != static_cast<std::size_t>(esp_int8::INPUT_FRAME_BYTES) ||
      golden_logits.size() != static_cast<std::size_t>(
          esp_int8::ENCODER_OUT_H * esp_int8::ENCODER_OUT_W *
          ESP_INT8_CSIM_CLASS_COUNT)) {
    std::printf("[FAIL] unexpected artifact sizes in %s: input=%zu golden_logits=%zu\n",
                artifact_dir,
                input_bytes.size(),
                golden_logits.size());
    return 1;
  }
  const std::vector<std::uint8_t> golden_mask =
      make_fullres_mask_golden(golden_logits, ESP_INT8_CSIM_CLASS_COUNT);
  std::vector<std::uint8_t> published_mask;
  if (!read_binary((std::string(artifact_dir) + "/golden_fullres_mask.bin").c_str(),
                   published_mask) || published_mask != golden_mask) {
    std::printf("[FAIL] published mask is not the independent bilinear reference\n");
    return 1;
  }

  const std::size_t param_words =
      (param_bytes.size() + esp_int8::AXI_WORD_BYTES - 1U) / esp_int8::AXI_WORD_BYTES;
  static esp_int8::axi_vec_t frame_in[esp_int8::INPUT_FRAME_AXI_WORDS];
  static esp_int8::axi_vec_t frame_out[esp_int8::OUTPUT_FRAME_AXI_WORDS];
  static esp_int8::axi_vec_t param[8192];

  if (param_words > 8192U) {
    std::printf("[FAIL] param blob too large for top TB buffer: words=%zu\n", param_words);
    return 1;
  }

  pack_bytes(input_bytes, frame_in, esp_int8::INPUT_FRAME_AXI_WORDS);
  pack_bytes(param_bytes, param, 8192U);
  for (int i = 0; i < esp_int8::OUTPUT_FRAME_AXI_WORDS; ++i) {
    frame_out[i] = ~static_cast<esp_int8::axi_vec_t>(0);
    for (int lane = 0; lane < esp_int8::AXI_WORD_BYTES; ++lane) {
      set_byte(frame_out[i], lane, 0xaaU);
    }
  }

  call_espnet_encoder_int8_core(frame_in, frame_out, param,
                                esp_int8::MODE_INIT,
                                esp_int8::UOP_COUNT_ENCODER);
  std::printf("top golden diag: param_ready=%u input_bytes=%zu param_bytes=%zu golden_logits=%zu\n",
              esp_int8::param_dma_ready() ? 1U : 0U,
              input_bytes.size(),
              param_bytes.size(),
              golden_logits.size());
  if (!esp_int8::param_dma_ready()) {
    std::printf("[FAIL] current PARAM rejected by INIT\n");
    return 1;
  }
  call_espnet_encoder_int8_core(frame_in, frame_out, param,
                                esp_int8::MODE_RUN,
                                esp_int8::UOP_COUNT_ENCODER);
  std::printf("top golden diag: after MODE_RUN last_uop=%u last_error=%u\n",
              esp_int8::csim_last_uop(),
              esp_int8::csim_last_error());
  if (esp_int8::csim_last_error() != 0U) {
    std::printf("[FAIL] MODE_RUN returned an error\n");
    return 1;
  }

  std::vector<std::uint8_t> hls_output(golden_mask.size());
  for (std::size_t i = 0; i < hls_output.size(); ++i) {
    hls_output[i] = get_byte(frame_out[i / esp_int8::AXI_WORD_BYTES],
                             static_cast<int>(i % esp_int8::AXI_WORD_BYTES));
  }
  if (!write_binary(hls_output_path, hls_output)) {
    return 1;
  }

  std::vector<std::uint8_t> hls_lowres_logits;
  if (!read_binary("csim_u72_lowres_logits.bin", hls_lowres_logits) ||
      hls_lowres_logits.size() != golden_logits.size()) {
    std::printf("[FAIL] missing or malformed HLS lowres logits side dump\n");
    return 1;
  }
  const std::vector<std::uint8_t> upsample_reference_mask =
      make_fullres_mask_golden(hls_lowres_logits, ESP_INT8_CSIM_CLASS_COUNT);

  int golden_zeros = 0;
  int golden_ones = 0;
  int golden_others = 0;
  int golden_aa = 0;
  int output_zeros = 0;
  int output_ones = 0;
  int output_others = 0;
  int output_aa = 0;
  count_labels(golden_mask, golden_zeros, golden_ones, golden_others, golden_aa);
  count_labels(hls_output, output_zeros, output_ones, output_others, output_aa);
  std::printf("top golden diag: golden zero=%d one=%d other=%d\n",
              golden_zeros,
              golden_ones,
              golden_others);
  std::printf("top golden diag: output zero=%d one=%d other=%d aa=%d\n",
              output_zeros,
              output_ones,
              output_others,
              output_aa);

  int mismatches = 0;
  int lowres_mismatches = 0;
  int upsample_mismatches = 0;
  int invalid_labels = 0;
  for (std::size_t i = 0; i < golden_logits.size(); ++i) {
    if (hls_lowres_logits[i] != golden_logits[i]) {
      ++lowres_mismatches;
    }
  }
  for (std::size_t i = 0; i < golden_mask.size(); ++i) {
    const std::uint8_t got = hls_output[i];
    const std::uint8_t expected = golden_mask[i];
    if (got >= static_cast<std::uint8_t>(ESP_INT8_CSIM_CLASS_COUNT)) {
      ++invalid_labels;
    }
    if (got != expected) {
      if (mismatches < 32) {
        std::printf("[DIFF] fullres mask mismatch byte=%zu got=%u expected=%u\n",
                    i,
                    static_cast<unsigned>(got),
                    static_cast<unsigned>(expected));
      }
      ++mismatches;
    }
    if (got != upsample_reference_mask[i]) {
      ++upsample_mismatches;
    }
  }

  std::printf("wrote HLS output: %s bytes=%zu\n", hls_output_path, hls_output.size());
  std::printf("top fullres golden mask stats: mismatches=%d/%zu invalid_labels=%d\n",
              mismatches,
              golden_mask.size(),
              invalid_labels);
  std::printf("top replay logits stats: mismatches=%d/%zu\n",
              lowres_mismatches,
              golden_logits.size());
  std::printf("top upsample stats: mismatches=%d/%zu\n",
              upsample_mismatches,
              hls_output.size());
  if (lowres_mismatches != 0 ||
      upsample_mismatches != 0 ||
      mismatches > MAX_ALLOWED_MASK_MISMATCHES ||
      invalid_labels != 0) {
    std::printf("[FAIL] replay logits=%d fullres=%d upsample=%d invalid=%d\n",
                lowres_mismatches,
                mismatches,
                upsample_mismatches,
                invalid_labels);
    return 1;
  }

  for (unsigned seed = 1; seed <= 2; ++seed) {
    if (!check_vec_alias_case(5U, seed) || !check_vec_alias_case(8U, seed)) {
      std::printf("[FAIL] Vec compact/aligned staged commit alias protection\n");
      return 1;
    }
  }
  std::printf("[PASS] Vec U39/U71: aliased sources, row padding, end guard and repeat staging\n");
  if (!check_vec_bad_destinations(param_bytes, param, frame_in, frame_out)) { return 1; }

  std::vector<std::uint8_t> old_param_bytes;
  const auto read_u16 = [&](unsigned offset) {
    return unsigned(param_bytes.at(offset)) | (unsigned(param_bytes.at(offset + 1)) << 8);
  };
  const auto read_u32 = [&](unsigned offset) {
    return read_u16(offset) | (read_u16(offset + 2) << 16);
  };
  const unsigned conv_offset = read_u32(88);
  const unsigned schedule_offset = read_u32(92) + unsigned(param_bytes.at(conv_offset + 2)) * 128;
  struct BadField { unsigned offset; unsigned value; const char* name; };
  const BadField corruptions[] = {
    {schedule_offset, 4U | (3U << 8), "retired C3 mode"},
    {schedule_offset + 126, 2U | (48U << 2), "retired row reuse"},
    {schedule_offset + 8, 3U, "wrong schedule channels"},
    {schedule_offset + 12, 2U, "wrong K tiles"},
  };
  for (const auto& fault : corruptions) {
    auto corrupted = param_bytes;
    corrupted.at(fault.offset) = fault.value & 255U;
    corrupted.at(fault.offset + 1) = (fault.value >> 8) & 255U;
    pack_bytes(corrupted, param, 8192U);
    call_espnet_encoder_int8_core(frame_in, frame_out, param,
                                  esp_int8::MODE_INIT, esp_int8::UOP_COUNT_ENCODER);
    if (esp_int8::param_dma_ready()) {
      std::printf("[FAIL] INIT accepted invalid schedule: %s\n", fault.name);
      return 1;
    }
  }
  std::printf("[PASS] INIT rejected %zu malformed/retired schedules\n",
              sizeof(corruptions) / sizeof(corruptions[0]));

  if (!read_binary("D:/ESP_INT8/hw_artifacts/binary2_int8_h256w512_r2_v4/PARAM.BIN",
                   old_param_bytes)) {
    return 1;
  }
  pack_bytes(old_param_bytes, param, 8192U);
  call_espnet_encoder_int8_core(frame_in, frame_out, param,
                                esp_int8::MODE_INIT,
                                esp_int8::UOP_COUNT_ENCODER);
  if (esp_int8::param_dma_ready()) {
    std::printf("[FAIL] legacy H256W512 PARAM passed H240W240/C1 geometry gate\n");
    return 1;
  }

  pack_bytes(param_bytes, param, 8192U);
  call_espnet_encoder_int8_core(frame_in, frame_out, param,
                                esp_int8::MODE_INIT, esp_int8::UOP_COUNT_ENCODER);
  if (!esp_int8::param_dma_ready()) {
    std::printf("[FAIL] valid INIT did not recover after rejection\n");
    return 1;
  }
  if (!check_block5_cache_reload(param_bytes, param, frame_in, frame_out)) {
    std::printf("[FAIL] BLOCK5 coefficient cache invalidation\n");
    return 1;
  }
  std::printf("[PASS] replay logits/mask are bit-exact, final AXI word covered, old geometry rejected\n");

  std::printf("top_golden_sample_tb finished: mismatches=%d\n", mismatches);
  return 0;
}
