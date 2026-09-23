#include "../include/npu_q.hpp"
#include "../include/npu_schedule.hpp"
#include "../include/npu_ctrl.hpp"

#include <cstdint>
#include <cstdio>
#include <initializer_list>

namespace esp_int8 {
void post_process_conv_row_to_buffer(hls::stream<psum_half_vec_t>& psum_stream,
                                     act_vec_t row_buf[MAX_FM_W],
                                     const conv_cfg_t& cfg,
                                     const window_sched_desc_t& sched,
                                     u8_t act_type,
                                     const conv_q_t& qparam,
                                     volatile u8_t& prof_conv_post_state);
}

static int failures = 0;

static int accumulator(int pixel, int channel) {
  static const int values[] = {-4096, -257, -129, -5, -3, -1, 0, 1,
                                3, 5, 127, 129, 255, 4096, 1000000, -1000000};
  return values[(pixel * 7 + channel) % 16];
}

static int reference(int acc, int channel, bool relu, const esp_int8::conv_q_t& q) {
  std::int64_t scaled = (static_cast<std::int64_t>(acc) + q.bias[channel].to_int()) *
                        q.mult[channel].to_int();
  const unsigned shift = q.shift[channel].to_uint();
  if (shift != 0) {
    const std::int64_t half = std::int64_t(1) << (shift - 1);
    scaled = scaled >= 0 ? (scaled + half) >> shift : -((-scaled + half) >> shift);
  }
  if (scaled > 127) scaled = 127;
  if (scaled < -128) scaled = -128;
  return relu && scaled < 0 ? 0 : static_cast<int>(scaled);
}

static void run_case(int width, int channels, bool paired, bool relu, int stride) {
  using namespace esp_int8;
  conv_cfg_t cfg = {};
  cfg.in_h = 1;
  cfg.in_w = width * stride - (stride - 1);
  cfg.out_c = channels;
  cfg.stride = stride;
  window_sched_desc_t sched = {};
  sched.flags = paired ? WINDOW_SCHED_FLAG_PIXEL_PARALLEL_2 : 0;
  if (paired && width % 2) sched.flags |= WINDOW_SCHED_FLAG_ODD_TAIL;
  conv_q_t q = {};
  static const unsigned shifts[] = {0, 1, 2, 3, 6, 10, 31};
  for (int channel = 0; channel < TM; ++channel) {
    q.bias[channel] = channel - 9;
    q.mult[channel] = channel % 5 + 1;
    q.shift[channel] = shifts[channel % 7];
  }
  act_vec_t row[MAX_FM_W];
  const act_vec_t fence = ~act_vec_t(0);
  for (int x = 0; x < MAX_FM_W; ++x) row[x] = fence;
  hls::stream<psum_half_vec_t> psums;
  const int issues = paired ? (width + 1) / 2 : width;
  const int halves = paired || channels > 16 ? 2 : 1;
  for (int issue = 0; issue < issues; ++issue) {
    for (int half = 0; half < halves; ++half) {
      psum_half_vec_t word = 0;
      for (int lane = 0; lane < 16; ++lane) {
        const int pixel = paired ? issue * 2 + half : issue;
        const int channel = paired ? lane : half * 16 + lane;
        const i32_t acc = accumulator(pixel, channel);
        word.range(lane * 32 + 31, lane * 32) = acc.range(31, 0);
      }
      psums.write(word);
    }
  }
  const psum_half_vec_t sentinel = ~psum_half_vec_t(0);
  psums.write(sentinel);
  volatile u8_t state = PROF_CONV_POST_IDLE_OR_DONE;
  post_process_conv_row_to_buffer(psums, row, cfg, sched,
                                  static_cast<u8_t>(relu ? ACT_RELU : ACT_NONE), q, state);
  const u8_t final_state = state;
  if (psums.size() != 1 || psums.read() != sentinel ||
      final_state.to_uint() != static_cast<unsigned>(PROF_CONV_POST_IDLE_OR_DONE)) {
    std::printf("[FAIL] token/state contract w=%d c=%d paired=%d\n", width, channels, paired);
    ++failures;
  }
  for (int x = 0; x < MAX_FM_W; ++x) {
    if (x >= width) {
      if (row[x] != fence) {
        std::printf("[FAIL] write beyond row x=%d w=%d\n", x, width);
        ++failures;
      }
      continue;
    }
    for (int channel = 0; channel < TM; ++channel) {
      i8_t value;
      value.range(7, 0) = row[x].range(channel * 8 + 7, channel * 8);
      const int expected = channel < channels ? reference(accumulator(x, channel), channel, relu, q) : 0;
      if (value.to_int() != expected) {
        if (failures < 16) std::printf("[FAIL] w=%d c=%d pair=%d x=%d lane=%d got=%d expected=%d\n",
                                     width, channels, paired, x, channel, value.to_int(), expected);
        ++failures;
      }
    }
  }
}

int main() {
  int cases = 0;
  for (int relu = 0; relu < 2; ++relu) {
    for (int channels : {2, 12, 16, 19, 20, 25, 28, 32}) {
      for (int width : {1, 2, 3, 7}) {
        run_case(width, channels, false, relu != 0, 2);
        ++cases;
        if (channels <= 16) {
          run_case(width, channels, true, relu != 0, 1);
          ++cases;
        }
      }
    }
  }
  run_case(esp_int8::MAX_FM_W - 1, 12, true, false, 1);
  run_case(esp_int8::MAX_FM_W, 32, false, true, 1);
  std::printf("conv_post_tb: cases=%d mismatches=%d %s\n", cases + 2, failures,
              failures == 0 ? "PASS" : "FAIL");
  return failures == 0 ? 0 : 1;
}
