#include "../include/npu_config.hpp"
#include "../include/npu_uop.hpp"
#include "top_call.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

namespace esp_int8 {
bool param_dma_ready();
unsigned csim_last_uop();
unsigned csim_last_error();
}  // namespace esp_int8

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

static std::uint8_t get_byte(const esp_int8::axi_vec_t& word, int lane) {
  return static_cast<std::uint8_t>(word.range(lane * 8 + 7, lane * 8).to_uint());
}

static void set_byte(esp_int8::axi_vec_t& word, int lane, std::uint8_t value) {
  word.range(lane * 8 + 7, lane * 8) = value;
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
  const char* param_path = "D:/ESP_INT8/hw_artifacts/binary2_int8_h256w512_r2_v4/PARAM.BIN";
  const char* input_path = "D:/ESP_INT8/hw_artifacts/binary2_int8_h256w512_r2_v4/input_q.bin";
  const char* golden_path = "D:/ESP_INT8/hw_artifacts/binary2_int8_h256w512_r2_v4/golden_output_q.bin";

  std::vector<std::uint8_t> param_bytes;
  std::vector<std::uint8_t> input_bytes;
  std::vector<std::uint8_t> golden;
  if (!read_binary(param_path, param_bytes) ||
      !read_binary(input_path, input_bytes) ||
      !read_binary(golden_path, golden)) {
    return 1;
  }
  if (input_bytes.size() != static_cast<std::size_t>(esp_int8::INPUT_FRAME_BYTES) ||
      golden.size() != static_cast<std::size_t>(esp_int8::ENCODER_LOGITS_BYTES)) {
    std::printf("[FAIL] unexpected artifact sizes input=%zu golden=%zu\n",
                input_bytes.size(),
                golden.size());
    return 1;
  }

  const std::size_t param_words =
      (param_bytes.size() + esp_int8::AXI_WORD_BYTES - 1U) / esp_int8::AXI_WORD_BYTES;
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

  std::printf("top lowres diag: param_ready=%u last_uop=%u last_error=%u\n",
              esp_int8::param_dma_ready() ? 1U : 0U,
              esp_int8::csim_last_uop(),
              esp_int8::csim_last_error());

  std::vector<std::uint8_t> hls_output(golden.size());
  int mismatches = 0;
  int max_diff = 0;
  int nonzero = 0;
  for (std::size_t i = 0; i < golden.size(); ++i) {
    const std::uint8_t got =
        get_byte(frame_out[i / esp_int8::AXI_WORD_BYTES],
                 static_cast<int>(i % esp_int8::AXI_WORD_BYTES));
    hls_output[i] = got;
    const std::uint8_t exp = golden[i];
    if (got != 0U) {
      ++nonzero;
    }
    const int diff = static_cast<int>(got) - static_cast<int>(exp);
    const int abs_diff = (diff < 0) ? -diff : diff;
    if (abs_diff > max_diff) {
      max_diff = abs_diff;
    }
    if (got != exp) {
      if (mismatches < 32) {
        std::printf("[DIFF] lowres byte=%zu got=%u expected=%u\n",
                    i,
                    static_cast<unsigned>(got),
                    static_cast<unsigned>(exp));
      }
      ++mismatches;
    }
  }
  if (!write_binary("hls_output_lowres_logits.bin", hls_output)) {
    return 1;
  }

  std::printf("top_lowres_logits_tb: mismatches=%d/%zu max_diff=%d nonzero=%d\n",
              mismatches,
              golden.size(),
              max_diff,
              nonzero);
  std::printf("wrote HLS lowres logits: hls_output_lowres_logits.bin bytes=%zu\n",
              hls_output.size());
  return (mismatches == 0 && esp_int8::csim_last_error() == 0U) ? 0 : 1;
}
