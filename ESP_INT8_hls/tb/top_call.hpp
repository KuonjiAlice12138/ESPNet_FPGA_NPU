#pragma once

#include "../include/npu_config.hpp"

#include <cstdint>

void espnet_encoder_int8_core(const esp_int8::axi_vec_t* gmem_frame_in,
                              esp_int8::axi_vec_t* gmem_frame_out,
                              const esp_int8::axi_vec_t* gmem_param,
                              std::uint32_t mode,
                              std::uint32_t uop_count,
                              volatile esp_int8::u8_t& prof_stage_id,
                              volatile ap_uint<1>& prof_active,
                              volatile esp_int8::u8_t& prof_pc,
                              volatile esp_int8::u8_t& prof_issue_kind,
                              volatile esp_int8::u8_t& prof_conv_win_state,
                              volatile esp_int8::u8_t& prof_conv_sa_state,
                              volatile esp_int8::u8_t& prof_conv_post_state);

inline void call_espnet_encoder_int8_core(const esp_int8::axi_vec_t* gmem_frame_in,
                                          esp_int8::axi_vec_t* gmem_frame_out,
                                          const esp_int8::axi_vec_t* gmem_param,
                                          std::uint32_t mode,
                                          std::uint32_t uop_count) {
  static volatile esp_int8::u8_t prof_stage_id = 0;
  static volatile ap_uint<1> prof_active = 0;
  static volatile esp_int8::u8_t prof_pc = 0;
  static volatile esp_int8::u8_t prof_issue_kind = 0;
  static volatile esp_int8::u8_t prof_conv_win_state = 0;
  static volatile esp_int8::u8_t prof_conv_sa_state = 0;
  static volatile esp_int8::u8_t prof_conv_post_state = 0;
  espnet_encoder_int8_core(gmem_frame_in,
                           gmem_frame_out,
                           gmem_param,
                           mode,
                           uop_count,
                           prof_stage_id,
                           prof_active,
                           prof_pc,
                           prof_issue_kind,
                           prof_conv_win_state,
                           prof_conv_sa_state,
                           prof_conv_post_state);
}
