#include "../include/npu_config.hpp"
#include "../include/npu_uop.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

void espnet_encoder_int8_core(const esp_int8::axi_vec_t* gmem_frame_in,
                              esp_int8::axi_vec_t* gmem_frame_out,
                              const esp_int8::axi_vec_t* gmem_param,
                              std::uint32_t mode,
                              std::uint32_t uop_count);

namespace esp_int8 {
bool param_dma_ready();
bool param_dma_get_tensor_desc(u8_t tensor_id, tensor_desc_t& desc);
}

static bool read_binary(const char* path, std::vector<std::uint8_t>& bytes) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::printf("[DIAG][FAIL] failed to open %s\n", path);
    return false;
  }
  bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  return true;
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

static void print_desc(const char* name, esp_int8::u8_t tensor_id) {
  esp_int8::tensor_desc_t desc;
  if (!esp_int8::param_dma_get_tensor_desc(tensor_id, desc)) {
    std::printf("[DIAG] desc %-8s unavailable\n", name);
    return;
  }
  std::printf("[DIAG] desc %-8s bank=%u base=0x%08x h=%u w=%u c=%u phys=%u coff=%u\n",
              name,
              static_cast<unsigned>(desc.bank_id.to_uint()),
              static_cast<unsigned>(desc.base_offset.to_uint()),
              static_cast<unsigned>(desc.h.to_uint()),
              static_cast<unsigned>(desc.w.to_uint()),
              static_cast<unsigned>(desc.c.to_uint()),
              static_cast<unsigned>(desc.reserved0.to_uint()),
              static_cast<unsigned>(desc.reserved1.to_uint()));
}

int main() {
  const char* param_path = "D:/ESP_INT8/hw_artifacts/hw_constrained_qat_3ep_single/param_blob.bin";
  const char* input_path = "D:/ESP_INT8/hw_artifacts/hw_constrained_qat_3ep_single/input_q.bin";

  std::vector<std::uint8_t> param_bytes;
  std::vector<std::uint8_t> input_bytes;
  if (!read_binary(param_path, param_bytes) || !read_binary(input_path, input_bytes)) {
    return 1;
  }

  static esp_int8::axi_vec_t frame_in[esp_int8::INPUT_FRAME_AXI_WORDS];
  static esp_int8::axi_vec_t frame_out[esp_int8::OUTPUT_FRAME_AXI_WORDS];
  static esp_int8::axi_vec_t param[4096];

  pack_bytes(input_bytes, frame_in, esp_int8::INPUT_FRAME_AXI_WORDS);
  pack_bytes(param_bytes, param, 4096U);
  for (int i = 0; i < esp_int8::OUTPUT_FRAME_AXI_WORDS; ++i) {
    frame_out[i] = 0xaaaaaaaaaaaaaaaaULL;
  }

  espnet_encoder_int8_core(frame_in, frame_out, param, esp_int8::MODE_INIT, esp_int8::UOP_COUNT_ENCODER);
  std::printf("[DIAG] param_ready=%u param_bytes=%zu input_bytes=%zu out_words=%d\n",
              esp_int8::param_dma_ready() ? 1U : 0U,
              param_bytes.size(),
              input_bytes.size(),
              esp_int8::OUTPUT_FRAME_AXI_WORDS);
  print_desc("INPUT", esp_int8::TID_INPUT);
  print_desc("OUT", esp_int8::TID_OUT);

  espnet_encoder_int8_core(frame_in, frame_out, param, esp_int8::MODE_RUN, esp_int8::UOP_COUNT_ENCODER);

  std::uint64_t zero_count = 0;
  std::uint64_t one_count = 0;
  std::uint64_t other_count = 0;
  std::uint64_t aa_count = 0;
  for (int i = 0; i < esp_int8::OUTPUT_FRAME_BYTES; ++i) {
    const std::uint8_t v = get_byte(frame_out[i / esp_int8::AXI_WORD_BYTES],
                                    i % esp_int8::AXI_WORD_BYTES);
    if (v == 0U) {
      ++zero_count;
    } else if (v == 1U) {
      ++one_count;
    } else {
      ++other_count;
      if (v == 0xaaU) {
        ++aa_count;
      }
    }
  }

  std::printf("[DIAG] output zero=%llu one=%llu other=%llu aa=%llu\n",
              static_cast<unsigned long long>(zero_count),
              static_cast<unsigned long long>(one_count),
              static_cast<unsigned long long>(other_count),
              static_cast<unsigned long long>(aa_count));
  std::printf("[DIAG] first32=");
  for (int i = 0; i < 32; ++i) {
    std::printf("%u%s",
                static_cast<unsigned>(get_byte(frame_out[i / esp_int8::AXI_WORD_BYTES],
                                               i % esp_int8::AXI_WORD_BYTES)),
                (i == 31) ? "\n" : ",");
  }

  if (!esp_int8::param_dma_ready()) {
    return 2;
  }
  if (aa_count == static_cast<std::uint64_t>(esp_int8::OUTPUT_FRAME_BYTES)) {
    std::printf("[DIAG][FAIL] MODE_RUN did not touch output buffer\n");
    return 3;
  }
  return 0;
}
