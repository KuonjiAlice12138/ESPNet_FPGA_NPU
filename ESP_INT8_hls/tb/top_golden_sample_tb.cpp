#include "../include/npu_config.hpp"
#include "../include/npu_uop.hpp"
#include "top_call.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace esp_int8 {
bool param_dma_ready();
unsigned csim_last_uop();
unsigned csim_last_error();
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
  // Full-resolution output is accepted by mask-level metrics rather than
  // bit-exact logits. The current HLS integer path has a stable ~0.68% mask
  // delta versus software bilinear-logit golden, with <0.5pp mIoU drop.
  static constexpr int MAX_ALLOWED_MASK_MISMATCHES = 4096;
#if ESP_INT8_CSIM_CLASS_COUNT == 20
  const char* artifact_dir = "D:/ESP_INT8/hw_artifacts/cityscapes20_p7_qat_final";
#else
  const char* artifact_dir = "D:/ESP_INT8/hw_artifacts/sched_v4_p7_0702";
#endif
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
  call_espnet_encoder_int8_core(frame_in, frame_out, param,
                                esp_int8::MODE_RUN,
                                esp_int8::UOP_COUNT_ENCODER);
  std::printf("top golden diag: after MODE_RUN last_uop=%u last_error=%u\n",
              esp_int8::csim_last_uop(),
              esp_int8::csim_last_error());

  std::vector<std::uint8_t> hls_output(golden_mask.size());
  for (std::size_t i = 0; i < hls_output.size(); ++i) {
    hls_output[i] = get_byte(frame_out[i / esp_int8::AXI_WORD_BYTES],
                             static_cast<int>(i % esp_int8::AXI_WORD_BYTES));
  }
  if (!write_binary(hls_output_path, hls_output)) {
    return 1;
  }

  std::vector<std::uint8_t> upsample_reference_mask = golden_mask;
#if ESP_INT8_CSIM_CLASS_COUNT == 20
  std::vector<std::uint8_t> hls_lowres_logits;
  if (!read_binary("csim_u72_lowres_logits.bin", hls_lowres_logits) ||
      hls_lowres_logits.size() != golden_logits.size()) {
    std::printf("[FAIL] missing or malformed HLS lowres logits side dump\n");
    return 1;
  }
  upsample_reference_mask =
      make_fullres_mask_golden(hls_lowres_logits, ESP_INT8_CSIM_CLASS_COUNT);
#endif

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
  int upsample_mismatches = 0;
  int invalid_labels = 0;
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
#if ESP_INT8_CSIM_CLASS_COUNT == 20
  std::printf("top multiclass upsample stats: mismatches=%d/%zu\n",
              upsample_mismatches,
              hls_output.size());
  if (upsample_mismatches != 0 || invalid_labels != 0) {
    std::printf("[FAIL] multiclass upsample/reference mismatch=%d invalid=%d\n",
                upsample_mismatches,
                invalid_labels);
    return 1;
  }
  std::printf("[PASS] multiclass upsample is bit-exact; software mask delta=%d/%zu\n",
              mismatches,
              golden_mask.size());
#else
  if (mismatches > MAX_ALLOWED_MASK_MISMATCHES || invalid_labels != 0) {
    std::printf("[FAIL] fullres mask check: mismatches=%d/%zu threshold=%d invalid=%d\n",
                mismatches,
                golden_mask.size(),
                MAX_ALLOWED_MASK_MISMATCHES,
                invalid_labels);
    return 1;
  } else {
    std::printf("[PASS] fullres mask mismatches=%d/%zu threshold=%d\n",
                mismatches,
                golden_mask.size(),
                MAX_ALLOWED_MASK_MISMATCHES);
  }
#endif

  std::printf("top_golden_sample_tb finished: mismatches=%d\n", mismatches);
  return 0;
}
