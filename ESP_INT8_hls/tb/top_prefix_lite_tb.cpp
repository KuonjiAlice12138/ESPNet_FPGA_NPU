#include "../include/npu_config.hpp"
#include "../include/npu_uop.hpp"
#include "top_call.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

static bool read_binary(const char* path, std::vector<std::uint8_t>& bytes) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::printf("[FAIL] failed to open %s\n", path);
    return false;
  }
  bytes.assign(std::istreambuf_iterator<char>(in),
               std::istreambuf_iterator<char>());
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

int main() {
  const char* param_path =
      "D:/ESP_INT8/hw_artifacts/sched_v4_p7_0702/PARAM.BIN";
  const char* input_path =
      "D:/ESP_INT8/hw_artifacts/sched_v4_p7_0702/input_q.bin";

  std::vector<std::uint8_t> param_bytes;
  std::vector<std::uint8_t> input_bytes;
  if (!read_binary(param_path, param_bytes) ||
      !read_binary(input_path, input_bytes)) {
    return 1;
  }
  if (input_bytes.size() != static_cast<std::size_t>(esp_int8::INPUT_FRAME_BYTES)) {
    std::printf("[FAIL] unexpected input size: %zu\n", input_bytes.size());
    return 1;
  }

  const std::size_t param_words =
      (param_bytes.size() + esp_int8::AXI_WORD_BYTES - 1U) /
      esp_int8::AXI_WORD_BYTES;
  if (param_words > 8192U) {
    std::printf("[FAIL] param blob too large: words=%zu\n", param_words);
    return 1;
  }

  static esp_int8::axi_vec_t frame_in[esp_int8::INPUT_FRAME_AXI_WORDS];
  static esp_int8::axi_vec_t frame_out[esp_int8::OUTPUT_FRAME_AXI_WORDS];
  static esp_int8::axi_vec_t param[8192];

  pack_bytes(input_bytes, frame_in, esp_int8::INPUT_FRAME_AXI_WORDS);
  pack_bytes(param_bytes, param, 8192U);
  for (int i = 0; i < esp_int8::OUTPUT_FRAME_AXI_WORDS; ++i) {
    frame_out[i] = 0;
  }

#ifdef ESP_INT8_CSIM_MAX_UOP
  std::printf("top_prefix_lite_tb: ESP_INT8_CSIM_MAX_UOP=%d\n",
              ESP_INT8_CSIM_MAX_UOP);
#else
  std::printf("top_prefix_lite_tb: full MODE_RUN, no prefix limit\n");
#endif

  call_espnet_encoder_int8_core(frame_in,
                                frame_out,
                                param,
                                esp_int8::MODE_INIT,
                                esp_int8::UOP_COUNT_ENCODER);
  call_espnet_encoder_int8_core(frame_in,
                                frame_out,
                                param,
                                esp_int8::MODE_RUN,
                                esp_int8::UOP_COUNT_ENCODER);

#ifdef ESP_INT8_CSIM_DUMP_UPSAMPLE_INPUT
  {
    std::vector<std::uint8_t> lowres_logits(esp_int8::ENCODER_LOGITS_BYTES);
    for (std::size_t i = 0; i < lowres_logits.size(); ++i) {
      lowres_logits[i] =
          get_byte(frame_out[i / esp_int8::AXI_WORD_BYTES],
                   static_cast<int>(i % esp_int8::AXI_WORD_BYTES));
    }

    std::ofstream out("hls_output_lowres_logits.bin", std::ios::binary);
    if (!out) {
      std::printf("[FAIL] failed to write hls_output_lowres_logits.bin\n");
      return 1;
    }
    out.write(reinterpret_cast<const char*>(lowres_logits.data()),
              static_cast<std::streamsize>(lowres_logits.size()));
    if (!out) {
      std::printf("[FAIL] incomplete write hls_output_lowres_logits.bin\n");
      return 1;
    }
    std::printf("[CSIM-DUMP] wrote hls_output_lowres_logits.bin bytes=%zu\n",
                lowres_logits.size());
  }
#endif

  unsigned checksum = 0;
  int nonzero = 0;
  constexpr int kPreviewWords = 64;
  for (int i = 0; i < kPreviewWords * esp_int8::AXI_WORD_BYTES; ++i) {
    const std::uint8_t value = get_byte(frame_out[i / esp_int8::AXI_WORD_BYTES],
                                        i % esp_int8::AXI_WORD_BYTES);
    checksum = (checksum * 131U) + static_cast<unsigned>(value);
    if (value != 0U) {
      ++nonzero;
    }
  }

  std::printf("top_prefix_lite_tb passed: preview_words=%d nonzero=%d checksum=0x%08x\n",
              kPreviewWords,
              nonzero,
              checksum);
  return 0;
}
