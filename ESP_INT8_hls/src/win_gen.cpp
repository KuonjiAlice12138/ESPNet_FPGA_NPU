#include "../include/npu_config.hpp"
#include "../include/npu_schedule.hpp"
#include "../include/npu_types.hpp"

namespace esp_int8 {

bool on_chip_memory_read_packed_tile(const tensor_desc_t& desc,
                                     i32_t h,
                                     i32_t w,
                                     u16_t c_begin,
                                     u8_t valid_c,
                                     act_vec_t& packed);

bool on_chip_memory_read_aligned_full_tile(const tensor_desc_t& desc,
                                           i32_t h,
                                           i32_t w,
                                           u16_t c_begin,
                                           act_vec_t& packed);

bool param_dma_get_pack_cmd(u16_t cmd_idx, window_pack_cmd_t& cmd);

static act_vec_t read_tile_or_zero(const tensor_desc_t& desc,
                                   i32_t h,
                                   i32_t w,
                                   u16_t c_begin,
                                   u8_t valid_c) {
#pragma HLS INLINE
    act_vec_t packed = 0;
    const bool ok = on_chip_memory_read_packed_tile(desc, h, w, c_begin, valid_c, packed);
    return ok ? packed : act_vec_t(0);
}

static act_vec_t read_aligned_or_zero(const tensor_desc_t& desc,
                                      i32_t h,
                                      i32_t w,
                                      u16_t c_begin) {
#pragma HLS INLINE
    act_vec_t packed = 0;
    const bool ok = on_chip_memory_read_aligned_full_tile(desc, h, w, c_begin, packed);
    return ok ? packed : act_vec_t(0);
}

static u16_t ceil_div_u16(u16_t a, u16_t b) {
#pragma HLS INLINE
    return static_cast<u16_t>((a + b - 1) / b);
}

static u16_t conv_out_dim(u16_t in_size, u16_t stride) {
#pragma HLS INLINE
    return ceil_div_u16(in_size, stride);
}

static u16_t win_desc_phys_c(const tensor_desc_t& desc) {
#pragma HLS INLINE
    return (desc.reserved0.to_uint() == 0U) ? desc.c : desc.reserved0;
}

static bool win_can_read_aligned_1x1(const tensor_desc_t& desc, u16_t c_begin) {
#pragma HLS INLINE
    const unsigned phys_c = win_desc_phys_c(desc).to_uint();
    const unsigned start_c = desc.reserved1.to_uint() + c_begin.to_uint();
    const unsigned base_start = desc.base_offset.to_uint() + start_c;
    return phys_c >= static_cast<unsigned>(AXI_WORD_BYTES) &&
           ((phys_c & static_cast<unsigned>(AXI_WORD_BYTES - 1)) == 0U) &&
           c_begin.to_uint() + static_cast<unsigned>(AXI_WORD_BYTES) <= desc.c.to_uint() &&
           ((base_start & static_cast<unsigned>(AXI_WORD_BYTES - 1)) == 0U);
}

static act_vec_t low_byte_mask(int count) {
#pragma HLS INLINE
    if (count <= 0) {
        return 0;
    }
    if (count >= AXI_WORD_BYTES) {
        return ~act_vec_t(0);
    }
    return static_cast<act_vec_t>((static_cast<act_vec_t>(1) << (count * 8)) - 1);
}

static void insert_packed_segment(act_vec_t& word,
                                  int lane_offset,
                                  int count,
                                  act_vec_t segment) {
#pragma HLS INLINE
    const act_vec_t mask = static_cast<act_vec_t>(low_byte_mask(count) << (lane_offset * 8));
    const act_vec_t shifted = static_cast<act_vec_t>(segment << (lane_offset * 8));
    word = static_cast<act_vec_t>((word & ~mask) | (shifted & mask));
}

static void apply_window_pack_cmd(const act_vec_t spatial_word[9],
                                  const window_pack_cmd_t& cmd,
                                  act_vec_t& word) {
#pragma HLS INLINE
    const unsigned flags = cmd.flags.to_uint();
    if ((flags & static_cast<unsigned>(PACK_CMD_VALID)) == 0U || cmd.byte_count.to_uint() == 0U) {
        return;
    }
    if ((flags & static_cast<unsigned>(PACK_CMD_ZERO)) != 0U) {
        return;
    }

    const int spatial = static_cast<int>(cmd.spatial_id.to_uint());
    if (spatial < 0 || spatial >= 9) {
        return;
    }
    const int src_lane = static_cast<int>(cmd.src_c_begin.to_uint());
    const int dst_lane = static_cast<int>(cmd.dst_lane_begin.to_uint());
    const int count = static_cast<int>(cmd.byte_count.to_uint());
    const act_vec_t segment = static_cast<act_vec_t>(spatial_word[spatial] >> (src_lane * 8));
    insert_packed_segment(word, dst_lane, count, segment);
}

static void emit_scheduled_smallc_words(const act_vec_t spatial_word[9],
                                        const window_sched_desc_t& sched,
                                        hls::stream<act_vec_t>& act_stream) {
#pragma HLS INLINE off
    const int k_tiles_i = static_cast<int>(sched.k_tiles.to_uint());
    for (int kt = 0; kt < MAX_K_TILE_COUNT; ++kt) {
        if (kt >= k_tiles_i) {
            break;
        }
        act_vec_t word = 0;
        const int begin = static_cast<int>(sched.kt_cmd_base[kt].to_uint());
        const int end = static_cast<int>(sched.kt_cmd_base[kt + 1].to_uint());
        const int cmd_base = static_cast<int>(sched.cmd_base.to_uint());
        for (int ci = 0; ci < MAX_PACK_CMDS_PER_KT; ++ci) {
#pragma HLS PIPELINE off
            const int rel_idx = begin + ci;
            if (rel_idx < end) {
                window_pack_cmd_t cmd;
                if (param_dma_get_pack_cmd(static_cast<u16_t>(cmd_base + rel_idx), cmd)) {
                    apply_window_pack_cmd(spatial_word, cmd, word);
                }
            }
        }
        act_stream.write(word);
    }
}

static void scheduled_smallc_3x3_window_row(const tensor_desc_t& src_desc,
                                            const conv_exec_desc_t& conv_desc,
                                            const window_sched_desc_t& sched,
                                            hls::stream<act_vec_t>& act_stream,
                                            u16_t out_row) {
#pragma HLS INLINE off
    const u16_t stride =
        (conv_desc.stride.to_uint() == 0U) ? static_cast<u16_t>(1) : static_cast<u16_t>(conv_desc.stride.to_uint());
    const u16_t dilation =
        (conv_desc.dilation.to_uint() == 0U) ? static_cast<u16_t>(1) : static_cast<u16_t>(conv_desc.dilation.to_uint());
    const u16_t padding =
        (conv_desc.padding.to_uint() == 0U) ? dilation : static_cast<u16_t>(conv_desc.padding.to_uint());
    const u16_t out_w = conv_out_dim(conv_desc.in_w, stride);
    const int out_w_i = static_cast<int>(out_w.to_uint());
    const i32_t base_h =
        static_cast<i32_t>(out_row.to_uint() * stride.to_uint()) - static_cast<i32_t>(padding.to_uint());

    for (int ow_i = 0; ow_i < MAX_FM_W; ++ow_i) {
        if (ow_i >= out_w_i) {
            break;
        }
        const i32_t base_w =
            static_cast<i32_t>(ow_i * static_cast<int>(stride.to_uint())) - static_cast<i32_t>(padding.to_uint());

        act_vec_t spatial_word[9];
#pragma HLS ARRAY_PARTITION variable=spatial_word complete dim=1
        for (int sp = 0; sp < 9; ++sp) {
#pragma HLS PIPELINE II=10
            const int kh = sp / 3;
            const int kw = sp - kh * 3;
            const i32_t ih = base_h + static_cast<i32_t>(kh * static_cast<int>(dilation.to_uint()));
            const i32_t iw = base_w + static_cast<i32_t>(kw * static_cast<int>(dilation.to_uint()));
            spatial_word[sp] =
                read_tile_or_zero(src_desc,
                                  ih,
                                  iw,
                                  static_cast<u16_t>(0),
                                  static_cast<u8_t>(conv_desc.in_c.to_uint()));
        }
        emit_scheduled_smallc_words(spatial_word, sched, act_stream);
    }
}

static void scheduled_segment_3x3_window_row(const tensor_desc_t& src_desc,
                                             const conv_exec_desc_t& conv_desc,
                                             const window_sched_desc_t& sched,
                                             hls::stream<act_vec_t>& act_stream,
                                             u16_t out_row) {
#pragma HLS INLINE off
    const u16_t stride =
        (conv_desc.stride.to_uint() == 0U) ? static_cast<u16_t>(1) : static_cast<u16_t>(conv_desc.stride.to_uint());
    const u16_t dilation =
        (conv_desc.dilation.to_uint() == 0U) ? static_cast<u16_t>(1) : static_cast<u16_t>(conv_desc.dilation.to_uint());
    const u16_t padding =
        (conv_desc.padding.to_uint() == 0U) ? dilation : static_cast<u16_t>(conv_desc.padding.to_uint());
    const u16_t out_w = conv_out_dim(conv_desc.in_w, stride);
    const int out_w_i = static_cast<int>(out_w.to_uint());
    const int k_tiles_i = static_cast<int>(sched.k_tiles.to_uint());
    const i32_t base_h =
        static_cast<i32_t>(out_row.to_uint() * stride.to_uint()) - static_cast<i32_t>(padding.to_uint());

    for (int ow_i = 0; ow_i < MAX_FM_W; ++ow_i) {
        if (ow_i >= out_w_i) {
            break;
        }
        const i32_t base_w =
            static_cast<i32_t>(ow_i * static_cast<int>(stride.to_uint())) - static_cast<i32_t>(padding.to_uint());
        for (int kt = 0; kt < MAX_K_TILE_COUNT; ++kt) {
            if (kt >= k_tiles_i) {
                break;
            }
            act_vec_t word = 0;
            const int begin = static_cast<int>(sched.kt_cmd_base[kt].to_uint());
            const int end = static_cast<int>(sched.kt_cmd_base[kt + 1].to_uint());
            const int cmd_base = static_cast<int>(sched.cmd_base.to_uint());
            for (int ci = 0; ci < MAX_PACK_CMDS_PER_KT; ++ci) {
#pragma HLS PIPELINE off
                const int rel_idx = begin + ci;
                if (rel_idx < end) {
                    window_pack_cmd_t cmd;
                    if (param_dma_get_pack_cmd(static_cast<u16_t>(cmd_base + rel_idx), cmd)) {
                        const unsigned flags = cmd.flags.to_uint();
                        if ((flags & static_cast<unsigned>(PACK_CMD_VALID)) != 0U &&
                            (flags & static_cast<unsigned>(PACK_CMD_ZERO)) == 0U &&
                            cmd.byte_count.to_uint() != 0U) {
                            const int spatial = static_cast<int>(cmd.spatial_id.to_uint());
                            const int kh = spatial / 3;
                            const int kw = spatial - kh * 3;
                            const i32_t ih = base_h + static_cast<i32_t>(kh * static_cast<int>(dilation.to_uint()));
                            const i32_t iw = base_w + static_cast<i32_t>(kw * static_cast<int>(dilation.to_uint()));
                            act_vec_t segment = read_tile_or_zero(src_desc,
                                                                  ih,
                                                                  iw,
                                                                  static_cast<u16_t>(cmd.src_c_begin.to_uint()),
                                                                  static_cast<u8_t>(cmd.byte_count.to_uint()));
                            insert_packed_segment(word,
                                                  static_cast<int>(cmd.dst_lane_begin.to_uint()),
                                                  static_cast<int>(cmd.byte_count.to_uint()),
                                                  segment);
                        }
                    }
                }
            }
            act_stream.write(word);
        }
    }
}

static void scheduled_1x1_window_row(const tensor_desc_t& src_desc,
                                     const conv_exec_desc_t& conv_desc,
                                     const window_sched_desc_t& sched,
                                     hls::stream<act_vec_t>& act_stream,
                                     u16_t out_row,
                                     bool aligned_full_tile) {
#pragma HLS INLINE off
    const u16_t stride =
        (conv_desc.stride.to_uint() == 0U) ? static_cast<u16_t>(1) : static_cast<u16_t>(conv_desc.stride.to_uint());
    const u16_t out_w = conv_out_dim(conv_desc.in_w, stride);
    const int out_w_i = static_cast<int>(out_w.to_uint());
    const int k_tiles_i = static_cast<int>(sched.k_tiles.to_uint());
    const i32_t ih = static_cast<i32_t>(out_row.to_uint() * stride.to_uint());
    const unsigned in_c = conv_desc.in_c.to_uint();

    for (int ow_i = 0; ow_i < MAX_FM_W; ++ow_i) {
        if (ow_i >= out_w_i) {
            break;
        }
        const i32_t iw = static_cast<i32_t>(ow_i * static_cast<int>(stride.to_uint()));
        for (int kt = 0; kt < MAX_K_TILE_COUNT; ++kt) {
#pragma HLS PIPELINE off
            if (kt >= k_tiles_i) {
                break;
            }
            const unsigned c_begin = static_cast<unsigned>(kt) * static_cast<unsigned>(TK);
            act_vec_t word = 0;
            if (aligned_full_tile &&
                win_can_read_aligned_1x1(src_desc, static_cast<u16_t>(c_begin))) {
                word = read_aligned_or_zero(src_desc, ih, iw, static_cast<u16_t>(c_begin));
            } else {
                unsigned valid = (c_begin < in_c) ? (in_c - c_begin) : 0U;
                if (valid > static_cast<unsigned>(TK)) {
                    valid = static_cast<unsigned>(TK);
                }
                word = read_tile_or_zero(src_desc,
                                         ih,
                                         iw,
                                         static_cast<u16_t>(c_begin),
                                         static_cast<u8_t>(valid));
            }
            act_stream.write(word);
        }
    }
}

void scheduled_window_generator_row(const tensor_desc_t& src_desc,
                                    const conv_exec_desc_t& conv_desc,
                                    const window_sched_desc_t& sched,
                                    hls::stream<act_vec_t>& act_stream,
                                    u16_t out_row) {
#pragma HLS INLINE off
    const unsigned mode = sched.mode.to_uint();
    if (mode == static_cast<unsigned>(WIN_MODE_FIRST_C3) ||
        mode == static_cast<unsigned>(WIN_MODE_SMALLC_3X3_STAGED)) {
        scheduled_smallc_3x3_window_row(src_desc, conv_desc, sched, act_stream, out_row);
        return;
    }
    if (mode == static_cast<unsigned>(WIN_MODE_LARGEC_3X3_SEGMENT)) {
        scheduled_segment_3x3_window_row(src_desc, conv_desc, sched, act_stream, out_row);
        return;
    }
    if (mode == static_cast<unsigned>(WIN_MODE_1X1_ALIGNED)) {
        scheduled_1x1_window_row(src_desc, conv_desc, sched, act_stream, out_row, true);
        return;
    }
    if (mode == static_cast<unsigned>(WIN_MODE_1X1_PACKED)) {
        scheduled_1x1_window_row(src_desc, conv_desc, sched, act_stream, out_row, false);
        return;
    }
}

}  // namespace esp_int8
