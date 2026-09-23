#include "../include/npu_config.hpp"
#include "../include/npu_schedule.hpp"
#include "../include/npu_uop.hpp"

namespace esp_int8 {

static param_blob_header_t s_header;
static tensor_desc_t s_tensor_desc[MAX_TENSOR_DESC_COUNT];
static conv_qparam_t s_conv_qparam[MAX_CONV_PARAM_DESC_COUNT];
static affine_qparam_t s_affine_qparam[MAX_AFFINE_PARAM_DESC_COUNT][8];
static u8_t s_affine_qparam_count[MAX_AFFINE_PARAM_DESC_COUNT];
static add_qparam_t s_add_qparam[MAX_ADD_PARAM_DESC_COUNT];
static pool_qparam_t s_pool_qparam[MAX_POOL_PARAM_DESC_COUNT];
static conv_exec_desc_t s_conv_exec_desc[MAX_CONV_EXEC_DESC_COUNT];
static window_sched_desc_t s_window_sched_desc[MAX_WINDOW_SCHED_COUNT];
static row_consumer_desc_t s_row_consumer_desc[MAX_ROW_CONSUMER_DESC_COUNT];
static fixed_exec_desc_t s_fixed_exec_desc[MAX_FIXED_EXEC_DESC_COUNT];
static exec_plan_entry_t s_exec_plan[MAX_EXEC_PLAN_COUNT];
static block5_sched_desc_t s_block5_sched[MAX_BLOCK5_SCHED_COUNT];
static u32_t s_weight_bytes = 0;

static bool s_ready = false;

static const unsigned k_v3_affine_blocks[MAX_AFFINE_PARAM_DESC_COUNT] = {1, 2, 2, 5, 4, 4, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0};

static bool validate_deployment_geometry() {
#pragma HLS INLINE
    const unsigned input_idx = static_cast<unsigned>(TID_INPUT);
    const unsigned output_idx = static_cast<unsigned>(TID_OUT);
    if (s_header.tensor_desc_count.to_uint() <= output_idx) {
        return false;
    }

    const tensor_desc_t& input = s_tensor_desc[input_idx];
    const tensor_desc_t& output = s_tensor_desc[output_idx];
    const unsigned output_c = output.c.to_uint();
    return input.elem_bytes.to_uint() == 1U &&
           input.h.to_uint() == static_cast<unsigned>(INPUT_FRAME_H) &&
           input.w.to_uint() == static_cast<unsigned>(INPUT_FRAME_W) &&
           input.c.to_uint() == static_cast<unsigned>(INPUT_FRAME_C) &&
           output.elem_bytes.to_uint() == 1U &&
           output.h.to_uint() == static_cast<unsigned>(ENCODER_OUT_H) &&
           output.w.to_uint() == static_cast<unsigned>(ENCODER_OUT_W) &&
           (output_c == 2U || output_c == 20U);
}

static axi_vec_t* get_wbuf() {
#pragma HLS INLINE
    static axi_vec_t s_wbuf[WBUF_AXI_WORDS];
#pragma HLS BIND_STORAGE variable=s_wbuf type=ram_2p impl=bram
#pragma HLS RESET variable=s_wbuf off
    return s_wbuf;
}

static u8_t read_u8(const axi_vec_t* gmem_param, u32_t byte_offset) {
#pragma HLS INLINE
    const u32_t word_idx = byte_offset >> 5;
    const int lane = static_cast<int>((byte_offset & 0x1f).to_uint());
    const axi_vec_t word = gmem_param[word_idx];
    return word.range(lane * 8 + 7, lane * 8);
}

static u16_t read_u16_le(const axi_vec_t* gmem_param, u32_t byte_offset) {
#pragma HLS INLINE
    const u16_t b0 = read_u8(gmem_param, byte_offset);
    const u16_t b1 = read_u8(gmem_param, byte_offset + 1);
    return b0 | (b1 << 8);
}

static u32_t read_u32_le(const axi_vec_t* gmem_param, u32_t byte_offset) {
#pragma HLS INLINE
    const u32_t b0 = read_u8(gmem_param, byte_offset);
    const u32_t b1 = read_u8(gmem_param, byte_offset + 1);
    const u32_t b2 = read_u8(gmem_param, byte_offset + 2);
    const u32_t b3 = read_u8(gmem_param, byte_offset + 3);
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

static i32_t read_i32_le(const axi_vec_t* gmem_param, u32_t byte_offset) {
#pragma HLS INLINE
    i32_t value;
    value.range(31, 0) = read_u32_le(gmem_param, byte_offset);
    return value;
}

static i32_t word_i32_le(axi_vec_t word, int lane32) {
#pragma HLS INLINE
    i32_t value;
    value.range(31, 0) = word.range(lane32 * 32 + 31, lane32 * 32);
    return value;
}

static u8_t word_u8(axi_vec_t word, int lane8) {
#pragma HLS INLINE
    return word.range(lane8 * 8 + 7, lane8 * 8);
}

static bool is_aligned64(u32_t offset) {
#pragma HLS INLINE
    return (offset & (SECTION_ALIGNMENT_BYTES - 1)) == 0;
}

static bool param_blob_is_v4() {
#pragma HLS INLINE
    return s_header.version.to_uint() == PARAM_BLOB_VERSION_SCHED;
}

static u32_t v4_exec_entry_count(const param_blob_header_t& header) {
#pragma HLS INLINE
    return header.reserved0;
}

static u32_t v4_conv_exec_offset(const param_blob_header_t& header) {
#pragma HLS INLINE
    return header.reserved1[0];
}

static u32_t v4_window_sched_offset(const param_blob_header_t& header) {
#pragma HLS INLINE
    return header.reserved1[1];
}

static u32_t v4_window_pack_cmd_offset(const param_blob_header_t& header) {
#pragma HLS INLINE
    return header.reserved1[2];
}

static u32_t v4_row_consumer_offset(const param_blob_header_t& header) {
#pragma HLS INLINE
    return header.reserved1[3];
}

static u32_t v4_fixed_exec_offset(const param_blob_header_t& header) {
#pragma HLS INLINE
    return header.reserved1[4];
}

static u32_t v4_exec_plan_offset(const param_blob_header_t& header) {
#pragma HLS INLINE
    return header.reserved1[5];
}

static u32_t v4_window_sched_count(const param_blob_header_t& header) {
#pragma HLS INLINE
    return header.reserved1[6];
}

static u32_t v4_window_pack_cmd_count(const param_blob_header_t& header) {
#pragma HLS INLINE
    return header.reserved1[7];
}

static u32_t v4_row_consumer_count(const param_blob_header_t& header) {
#pragma HLS INLINE
    return header.reserved1[9];
}

static u32_t v4_block5_sched_offset(const param_blob_header_t& header) {
#pragma HLS INLINE
    return header.reserved1[8];
}

static u32_t v4_fixed_exec_count(const param_blob_header_t& header) {
#pragma HLS INLINE
    const u32_t fixed_offset = v4_fixed_exec_offset(header);
    const u32_t exec_offset = v4_exec_plan_offset(header);
    if (fixed_offset.to_uint() == 0U || exec_offset < fixed_offset) {
        return 0;
    }
    return (exec_offset - fixed_offset) / static_cast<u32_t>(FIXED_EXEC_DESC_BLOB_BYTES);
}

static void load_header(const axi_vec_t* gmem_param, param_blob_header_t& header) {
#pragma HLS INLINE
    header.magic = read_u32_le(gmem_param, 0);
    header.version = read_u32_le(gmem_param, 4);
    header.tensor_desc_count = read_u32_le(gmem_param, 8);
    header.scale_desc_count = read_u32_le(gmem_param, 12);
    header.conv_desc_count = read_u32_le(gmem_param, 16);
    header.affine_desc_count = read_u32_le(gmem_param, 20);
    header.add_desc_count = read_u32_le(gmem_param, 24);
    header.pool_desc_count = read_u32_le(gmem_param, 28);
    header.uop_count = read_u32_le(gmem_param, 32);
    header.reserved0 = read_u32_le(gmem_param, 36);
    header.tensor_desc_offset = read_u32_le(gmem_param, 40);
    header.scale_desc_offset = read_u32_le(gmem_param, 44);
    header.conv_desc_offset = read_u32_le(gmem_param, 48);
    header.affine_desc_offset = read_u32_le(gmem_param, 52);
    header.add_desc_offset = read_u32_le(gmem_param, 56);
    header.pool_desc_offset = read_u32_le(gmem_param, 60);
    header.uop_offset = read_u32_le(gmem_param, 64);
    header.weight_data_offset = read_u32_le(gmem_param, 68);
    header.conv_qparam_offset = read_u32_le(gmem_param, 72);
    header.affine_qparam_offset = read_u32_le(gmem_param, 76);
    header.add_qparam_offset = read_u32_le(gmem_param, 80);
    header.pool_qparam_offset = read_u32_le(gmem_param, 84);

    for (int i = 0; i < 10; ++i) {
#pragma HLS UNROLL
        header.reserved1[i] = read_u32_le(gmem_param, 88 + i * 4);
    }
}

static error_code_t validate_header(const param_blob_header_t& header) {
#pragma HLS INLINE
    if (header.magic != PARAM_BLOB_MAGIC) {
        return ERR_BAD_BLOB;
    }
    if (header.version.to_uint() != PARAM_BLOB_VERSION_SCHED) {
        return ERR_BAD_BLOB;
    }
    if (header.uop_count.to_uint() != static_cast<unsigned>(UOP_COUNT_ENCODER)) {
        return ERR_UOP_DECODE;
    }
    if (header.tensor_desc_count > MAX_TENSOR_DESC_COUNT ||
        header.scale_desc_count > SCALE_DESC_COUNT_MAX) {
        return ERR_TENSOR_DESC_RANGE;
    }
    if (header.conv_desc_count > MAX_CONV_EXEC_DESC_COUNT ||
        header.affine_desc_count > MAX_AFFINE_PARAM_DESC_COUNT ||
        header.add_desc_count > MAX_ADD_PARAM_DESC_COUNT ||
        header.pool_desc_count > MAX_POOL_PARAM_DESC_COUNT ||
        v4_window_sched_count(header) > MAX_WINDOW_SCHED_COUNT ||
        v4_window_pack_cmd_count(header) > MAX_WINDOW_PACK_CMD_COUNT ||
        v4_row_consumer_count(header) > MAX_ROW_CONSUMER_DESC_COUNT ||
        v4_fixed_exec_count(header) > MAX_FIXED_EXEC_DESC_COUNT ||
        v4_exec_entry_count(header) > MAX_EXEC_PLAN_COUNT) {
        return ERR_PARAM_DESC_RANGE;
    }
    if (!is_aligned64(v4_conv_exec_offset(header)) ||
        !is_aligned64(v4_window_sched_offset(header)) ||
        !is_aligned64(v4_window_pack_cmd_offset(header)) ||
        !is_aligned64(v4_row_consumer_offset(header)) ||
        !is_aligned64(v4_fixed_exec_offset(header)) ||
        !is_aligned64(v4_exec_plan_offset(header)) ||
        !is_aligned64(v4_block5_sched_offset(header))) {
        return ERR_PARAM_DESC_RANGE;
    }
    if (v4_fixed_exec_offset(header).to_uint() == 0U ||
        v4_exec_plan_offset(header) < v4_fixed_exec_offset(header) ||
        ((v4_exec_plan_offset(header) - v4_fixed_exec_offset(header)) %
         static_cast<u32_t>(FIXED_EXEC_DESC_BLOB_BYTES)) != 0) {
        return ERR_PARAM_DESC_RANGE;
    }
    if (!is_aligned64(header.tensor_desc_offset) ||
        !is_aligned64(header.scale_desc_offset) ||
        !is_aligned64(header.uop_offset) ||
        !is_aligned64(header.weight_data_offset) ||
        !is_aligned64(header.conv_qparam_offset) ||
        !is_aligned64(header.affine_qparam_offset) ||
        !is_aligned64(header.add_qparam_offset) ||
        !is_aligned64(header.pool_qparam_offset)) {
        return ERR_PARAM_DESC_RANGE;
    }
    return ERR_NONE;
}

static tensor_desc_t load_tensor_desc(const axi_vec_t* gmem_param, u32_t offset) {
#pragma HLS INLINE
    tensor_desc_t desc;
    desc.bank_id = read_u8(gmem_param, offset + 0);
    desc.elem_bytes = read_u8(gmem_param, offset + 1);
    desc.reserved0 = read_u16_le(gmem_param, offset + 2);
    desc.base_offset = read_u32_le(gmem_param, offset + 4);
    desc.h = read_u16_le(gmem_param, offset + 8);
    desc.w = read_u16_le(gmem_param, offset + 10);
    desc.c = read_u16_le(gmem_param, offset + 12);
    desc.reserved1 = read_u16_le(gmem_param, offset + 14);
    return desc;
}

static conv_qparam_t load_conv_qparam_wordwise(const axi_vec_t* gmem_param, u32_t offset) {
#pragma HLS INLINE off
    conv_qparam_t qparam;
    const u32_t word_base = offset >> 5;

    for (int word_idx = 0; word_idx < 4; ++word_idx) {
#pragma HLS PIPELINE off
        const axi_vec_t word = gmem_param[word_base + word_idx];
        for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
            qparam.bias[word_idx * 8 + lane] = word_i32_le(word, lane);
        }
    }
    for (int word_idx = 0; word_idx < 4; ++word_idx) {
#pragma HLS PIPELINE off
        const axi_vec_t word = gmem_param[word_base + 4 + word_idx];
        for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
            qparam.mult[word_idx * 8 + lane] = word_i32_le(word, lane);
        }
    }

    const axi_vec_t shift_word = gmem_param[word_base + 8];
    const axi_vec_t reserved_word = gmem_param[word_base + 9];
    for (int lane = 0; lane < 32; ++lane) {
#pragma HLS UNROLL
        qparam.shift[lane] = word_u8(shift_word, lane);
        qparam.reserved[lane] = word_u8(reserved_word, lane);
    }
    return qparam;
}

static affine_qparam_t load_affine_qparam_wordwise(const axi_vec_t* gmem_param, u32_t offset) {
#pragma HLS INLINE off
    affine_qparam_t qparam;
    const u32_t word_base = offset >> 5;

    for (int word_idx = 0; word_idx < 4; ++word_idx) {
#pragma HLS PIPELINE off
        const axi_vec_t word = gmem_param[word_base + word_idx];
        for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
            qparam.mul[word_idx * 8 + lane] = word_i32_le(word, lane);
        }
    }
    for (int word_idx = 0; word_idx < 4; ++word_idx) {
#pragma HLS PIPELINE off
        const axi_vec_t word = gmem_param[word_base + 4 + word_idx];
        for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
            qparam.bias[word_idx * 8 + lane] = word_i32_le(word, lane);
        }
    }

    const axi_vec_t shift_word = gmem_param[word_base + 8];
    const axi_vec_t reserved_word = gmem_param[word_base + 9];
    for (int lane = 0; lane < 32; ++lane) {
#pragma HLS UNROLL
        qparam.shift[lane] = word_u8(shift_word, lane);
        qparam.reserved[lane] = word_u8(reserved_word, lane);
    }
    return qparam;
}

static add_qparam_t load_add_qparam(const axi_vec_t* gmem_param, u32_t offset) {
#pragma HLS INLINE
    add_qparam_t qparam;
    qparam.mult = read_i32_le(gmem_param, offset + 0);
    qparam.shift = read_u8(gmem_param, offset + 4);
    qparam.act_type = read_u8(gmem_param, offset + 5);
    qparam.requant_bypass = read_u8(gmem_param, offset + 6);
    qparam.reserved0 = read_u8(gmem_param, offset + 7);
    qparam.src_scale_id_a = read_u32_le(gmem_param, offset + 8);
    qparam.src_scale_id_b = read_u32_le(gmem_param, offset + 12);
    qparam.dst_scale_id = read_u32_le(gmem_param, offset + 16);
    qparam.reserved1 = read_u32_le(gmem_param, offset + 20);
    return qparam;
}

static pool_qparam_t load_pool_qparam(const axi_vec_t* gmem_param, u32_t offset) {
#pragma HLS INLINE
    pool_qparam_t qparam;
    qparam.kernel = read_u8(gmem_param, offset + 0);
    qparam.stride = read_u8(gmem_param, offset + 1);
    qparam.same_scale = read_u8(gmem_param, offset + 2);
    qparam.act_type = read_u8(gmem_param, offset + 3);
    qparam.src_scale_id = read_u32_le(gmem_param, offset + 4);
    qparam.dst_scale_id = read_u32_le(gmem_param, offset + 8);
    qparam.mult = read_i32_le(gmem_param, offset + 12);
    qparam.shift = read_u8(gmem_param, offset + 16);
    for (int i = 0; i < 11; ++i) {
#pragma HLS UNROLL
        qparam.reserved[i] = read_u8(gmem_param, offset + 17 + i);
    }
    return qparam;
}

static window_sched_desc_t load_window_sched_desc(const axi_vec_t* gmem_param, u32_t offset) {
#pragma HLS INLINE
    window_sched_desc_t desc;
    desc.mode = read_u8(gmem_param, offset + 0);
    desc.kernel = read_u8(gmem_param, offset + 1);
    desc.stride = read_u8(gmem_param, offset + 2);
    desc.dilation = read_u8(gmem_param, offset + 3);
    desc.padding = read_u8(gmem_param, offset + 4);
    desc.cache_chunks = read_u8(gmem_param, offset + 5);
    desc.cache_col_slots = read_u8(gmem_param, offset + 6);
    desc.flags = read_u8(gmem_param, offset + 7);
    desc.in_c = read_u16_le(gmem_param, offset + 8);
    desc.out_w = read_u16_le(gmem_param, offset + 10);
    desc.k_tiles = read_u16_le(gmem_param, offset + 12);
    desc.cmd_base = read_u16_le(gmem_param, offset + 14);
    desc.cmd_count = read_u16_le(gmem_param, offset + 16);
    for (int i = 0; i < MAX_K_TILE_COUNT + 1; ++i) {
#pragma HLS PIPELINE off
        desc.kt_cmd_base[i] = read_u16_le(gmem_param, offset + 18 + i * 2);
    }
    desc.loader_class = read_u8(gmem_param, offset + 100);
    desc.loader_request_cols = read_u8(gmem_param, offset + 101);
    desc.loader_warmup_issues = read_u8(gmem_param, offset + 102);
    desc.loader_warmup_new_cols = read_u8(gmem_param, offset + 103);
    desc.loader_steady_new_cols = read_u8(gmem_param, offset + 104);
    desc.loader_words_per_col = read_u8(gmem_param, offset + 105);
    desc.loader_warmup_mask = read_u8(gmem_param, offset + 106);
    desc.loader_steady_mask = read_u8(gmem_param, offset + 107);
    for (int i = 0; i < WINDOW_LOADER_RESERVED_WORDS; ++i) {
#pragma HLS PIPELINE off
        desc.reserved[i] = read_u16_le(gmem_param, offset + 108 + i * 2);
    }
    return desc;
}

static bool window_row_reuse_contract_valid(
    const window_sched_desc_t& desc) {
#pragma HLS INLINE
    const unsigned word =
        desc.reserved[WINDOW_LOADER_ROW_REUSE_WORD].to_uint();
    const unsigned reuse_mode = word & WINDOW_ROW_REUSE_MODE_MASK;
    const unsigned packed_words =
        (word >> WINDOW_ROW_REUSE_WORDS_SHIFT) &
        WINDOW_ROW_REUSE_WORDS_MASK;
    if ((word >> WINDOW_ROW_REUSE_RESERVED_SHIFT) != 0U) {
        return false;
    }
    if (reuse_mode == static_cast<unsigned>(WINDOW_ROW_REUSE_NONE)) {
        return packed_words == 0U;
    }
    const bool common =
        reuse_mode == static_cast<unsigned>(WINDOW_ROW_REUSE_STRIDE2_KEEP1) &&
        desc.mode.to_uint() == static_cast<unsigned>(WIN_MODE_3X3_STAGED_C3) &&
        desc.in_c.to_uint() == 3U && desc.stride.to_uint() == 2U &&
        desc.kernel.to_uint() == 3U &&
        desc.dilation.to_uint() == 1U &&
        desc.loader_class.to_uint() ==
            static_cast<unsigned>(WIN_LOADER_3X3_NARROW) &&
        (desc.flags.to_uint() &
         static_cast<unsigned>(WINDOW_SCHED_FLAG_PIXEL_PARALLEL_2)) != 0U &&
        packed_words > 0U &&
        packed_words <= static_cast<unsigned>(WINGEN_NARROW_ROW_WORDS);
    return common;
}

static bool window_row_reuse_source_valid(const window_sched_desc_t& sched,
                                         const conv_exec_desc_t& conv,
                                         const tensor_desc_t& src) {
#pragma HLS INLINE
    const unsigned word = sched.reserved[WINDOW_LOADER_ROW_REUSE_WORD].to_uint();
    if (word == 0U) {
        return true;
    }
    const unsigned phys_c = src.reserved0.to_uint() == 0U
        ? src.c.to_uint() : src.reserved0.to_uint();
    const unsigned row_bytes = src.w.to_uint() * phys_c;
    const unsigned words = (word >> WINDOW_ROW_REUSE_WORDS_SHIFT) & WINDOW_ROW_REUSE_WORDS_MASK;
    const int out_w = (static_cast<int>(conv.in_w.to_uint()) +
                       2 * static_cast<int>(conv.padding.to_uint()) - 3) / 2 + 1;
    return src.h.to_uint() > 0U && src.w.to_uint() > 0U &&
           src.elem_bytes.to_uint() == 1U && src.c.to_uint() == 3U && phys_c == 3U &&
           src.reserved1.to_uint() == 0U &&
           (src.base_offset.to_uint() & (AXI_WORD_BYTES - 1U)) == 0U &&
           src.h == conv.in_h && src.w == conv.in_w &&
           (row_bytes & (AXI_WORD_BYTES - 1U)) == 0U && row_bytes / AXI_WORD_BYTES == words &&
           conv.out_c.to_uint() > 0U && conv.out_c.to_uint() <= 16U &&
           conv.in_c == sched.in_c && conv.kernel == sched.kernel &&
           conv.stride == sched.stride && conv.dilation == sched.dilation &&
           conv.padding == sched.padding && conv.k_tiles == sched.k_tiles &&
           sched.k_tiles.to_uint() == 1U && out_w > 0 &&
           sched.out_w.to_uint() == static_cast<unsigned>(out_w) &&
           ((sched.flags.to_uint() & static_cast<unsigned>(WINDOW_SCHED_FLAG_ODD_TAIL)) != 0U)
               == ((out_w & 1) != 0);
}

static conv_exec_desc_t load_conv_exec_desc(const axi_vec_t* gmem_param, u32_t offset) {
#pragma HLS INLINE
    conv_exec_desc_t desc;
    desc.param_id = read_u8(gmem_param, offset + 0);
    desc.qparam_id = read_u8(gmem_param, offset + 1);
    desc.window_sched_id = read_u8(gmem_param, offset + 2);
    desc.row_consumer_id = read_u8(gmem_param, offset + 3);
    desc.in_h = read_u16_le(gmem_param, offset + 4);
    desc.in_w = read_u16_le(gmem_param, offset + 6);
    desc.in_c = read_u16_le(gmem_param, offset + 8);
    desc.out_c = read_u16_le(gmem_param, offset + 10);
    desc.kernel = read_u8(gmem_param, offset + 12);
    desc.stride = read_u8(gmem_param, offset + 13);
    desc.dilation = read_u8(gmem_param, offset + 14);
    desc.padding = read_u8(gmem_param, offset + 15);
    desc.src_tensor = read_u8(gmem_param, offset + 16);
    desc.dst_tensor = read_u8(gmem_param, offset + 17);
    desc.dst_c_offset = read_u16_le(gmem_param, offset + 18);
    desc.valid_c = read_u16_le(gmem_param, offset + 20);
    desc.packed_weight_word_offset = read_u32_le(gmem_param, offset + 22);
    desc.k_tiles = read_u16_le(gmem_param, offset + 26);
    desc.weight_words = read_u16_le(gmem_param, offset + 28);
    desc.flags = read_u16_le(gmem_param, offset + 30);
    desc.reserved = read_u16_le(gmem_param, offset + 32);
    return desc;
}

static row_consumer_desc_t load_row_consumer_desc(const axi_vec_t* gmem_param, u32_t offset) {
#pragma HLS INLINE
    row_consumer_desc_t desc;
    desc.mode = read_u8(gmem_param, offset + 0);
    desc.add_other_tensor = read_u8(gmem_param, offset + 1);
    desc.store_dst_tensor = read_u8(gmem_param, offset + 2);
    desc.add_qparam_id = read_u8(gmem_param, offset + 3);
    desc.store_c_offset = read_u16_le(gmem_param, offset + 4);
    desc.valid_c = read_u16_le(gmem_param, offset + 6);
    desc.alias_tensor = read_u8(gmem_param, offset + 8);
    desc.affine_param_id = read_u8(gmem_param, offset + 9);
    desc.affine_block_base = read_u8(gmem_param, offset + 10);
    desc.act_type = read_u8(gmem_param, offset + 11);
    desc.reserved0 = read_u16_le(gmem_param, offset + 12);
    desc.reserved1 = read_u16_le(gmem_param, offset + 14);
    return desc;
}

static exec_plan_entry_t load_exec_plan_entry(const axi_vec_t* gmem_param, u32_t offset) {
#pragma HLS INLINE
    exec_plan_entry_t entry;
    entry.kind = read_u8(gmem_param, offset + 0);
    entry.desc_id = read_u8(gmem_param, offset + 1);
    entry.logical_uop_id = read_u8(gmem_param, offset + 2);
    entry.flags = read_u8(gmem_param, offset + 3);
    return entry;
}

static fixed_exec_desc_t load_fixed_exec_desc(const axi_vec_t* gmem_param, u32_t offset) {
#pragma HLS INLINE
    fixed_exec_desc_t desc;
    desc.kind = read_u8(gmem_param, offset + 0);
    desc.src0_tensor = read_u8(gmem_param, offset + 1);
    desc.src1_tensor = read_u8(gmem_param, offset + 2);
    desc.dst_tensor = read_u8(gmem_param, offset + 3);
    desc.param_id = read_u8(gmem_param, offset + 4);
    desc.add_param_id = read_u8(gmem_param, offset + 5);
    desc.act_type = read_u8(gmem_param, offset + 6);
    desc.flags = read_u8(gmem_param, offset + 7);
    desc.in_h = read_u16_le(gmem_param, offset + 8);
    desc.in_w = read_u16_le(gmem_param, offset + 10);
    desc.in_c = read_u16_le(gmem_param, offset + 12);
    desc.out_c = read_u16_le(gmem_param, offset + 14);
    desc.kernel = read_u8(gmem_param, offset + 16);
    desc.stride = read_u8(gmem_param, offset + 17);
    desc.dilation = read_u8(gmem_param, offset + 18);
    desc.padding = read_u8(gmem_param, offset + 19);
    desc.c_offset = read_u16_le(gmem_param, offset + 20);
    desc.valid_c = read_u16_le(gmem_param, offset + 22);
    desc.qparam_id = read_u16_le(gmem_param, offset + 24);
    desc.reserved0 = read_u16_le(gmem_param, offset + 26);
    desc.reserved1 = read_u32_le(gmem_param, offset + 28);
    return desc;
}

static block5_sched_desc_t load_block5_sched_desc(const axi_vec_t* gmem_param, u32_t offset) {
#pragma HLS INLINE
    block5_sched_desc_t desc;
    desc.pattern = read_u8(gmem_param, offset + 0);
    desc.branch_count = read_u8(gmem_param, offset + 1);
    desc.first_branch_conv_id = read_u8(gmem_param, offset + 2);
    desc.first_window_sched_id = read_u8(gmem_param, offset + 3);
    desc.first_conv_qparam_id = read_u8(gmem_param, offset + 4);
    desc.src_tensor = read_u8(gmem_param, offset + 5);
    desc.dst_tensor = read_u8(gmem_param, offset + 6);
    desc.add_tensor = read_u8(gmem_param, offset + 7);
    desc.chain_add_qparam_id0 = read_u8(gmem_param, offset + 8);
    desc.chain_add_qparam_id1 = read_u8(gmem_param, offset + 9);
    desc.chain_add_qparam_id2 = read_u8(gmem_param, offset + 10);
    desc.residual_add_qparam_id = read_u8(gmem_param, offset + 11);
    desc.affine_param_id = read_u8(gmem_param, offset + 12);
    desc.affine_block_count = read_u8(gmem_param, offset + 13);
    desc.finalizer_kind = read_u8(gmem_param, offset + 14);
    desc.scratch_region = read_u8(gmem_param, offset + 15);
    desc.out_h = read_u16_le(gmem_param, offset + 16);
    desc.out_w = read_u16_le(gmem_param, offset + 18);
    desc.row_group_h = read_u16_le(gmem_param, offset + 20);
    desc.valid_c = read_u16_le(gmem_param, offset + 22);
    desc.reserved0 = read_u16_le(gmem_param, offset + 24);
    desc.reserved1 = read_u16_le(gmem_param, offset + 26);
    desc.reserved2 = read_u32_le(gmem_param, offset + 28);
    return desc;
}

static bool load_weight_data(const axi_vec_t* gmem_param) {
#pragma HLS INLINE
    if (s_header.conv_qparam_offset < s_header.weight_data_offset) {
        s_weight_bytes = 0;
        return false;
    }

    s_weight_bytes = s_header.conv_qparam_offset - s_header.weight_data_offset;
    if (s_weight_bytes.to_uint() > static_cast<unsigned>(WBUF_BYTES)) {
        return false;
    }

    const u32_t src_word_base = s_header.weight_data_offset >> 5;
    const u32_t weight_words = (s_weight_bytes + static_cast<u32_t>(AXI_WORD_BYTES - 1)) >> 5;
    for (int i = 0; i < WBUF_AXI_WORDS; ++i) {
#pragma HLS PIPELINE off
        if (static_cast<unsigned>(i) >= weight_words.to_uint()) {
            break;
        }
        get_wbuf()[i] = gmem_param[src_word_base + i];
    }
    return true;
}

static bool read_wbuf_word(u32_t local_word_offset, wgt_vec_t& word) {
#pragma HLS INLINE
    const u32_t byte_offset = local_word_offset << 5;
    if (byte_offset >= s_weight_bytes) {
        word = 0;
        return false;
    }
    word = get_wbuf()[local_word_offset];
    return true;
}

void param_dma_init(const axi_vec_t* gmem_param) {
#pragma HLS INLINE off
// PARAM v4 control tables are shallow descriptor memories. Binding them to
// BRAM wastes one BRAM per decomposed struct field; LUTRAM keeps the ABI and
// read latency unchanged while recovering scarce block RAM for data buffers.
#pragma HLS BIND_STORAGE variable=s_conv_exec_desc type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE variable=s_window_sched_desc type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE variable=s_row_consumer_desc type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE variable=s_fixed_exec_desc type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE variable=s_exec_plan type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE variable=s_block5_sched type=ram_2p impl=lutram
#pragma HLS RESET variable=s_conv_exec_desc off
#pragma HLS RESET variable=s_window_sched_desc off
#pragma HLS RESET variable=s_row_consumer_desc off
#pragma HLS RESET variable=s_fixed_exec_desc off
#pragma HLS RESET variable=s_exec_plan off
#pragma HLS RESET variable=s_block5_sched off
    s_ready = false;
    s_weight_bytes = 0;

    load_header(gmem_param, s_header);
    const error_code_t header_error = validate_header(s_header);
    if (header_error != ERR_NONE) {
        return;
    }
    if (!load_weight_data(gmem_param)) {
        return;
    }

    for (int i = 0; i < MAX_TENSOR_DESC_COUNT; ++i) {
#pragma HLS PIPELINE off
        if (i < static_cast<int>(s_header.tensor_desc_count.to_uint())) {
            s_tensor_desc[i] = load_tensor_desc(gmem_param, s_header.tensor_desc_offset + i * 16);
        }
    }
    if (!validate_deployment_geometry()) {
        return;
    }
    for (int i = 0; i < MAX_CONV_EXEC_DESC_COUNT; ++i) {
#pragma HLS PIPELINE off
        if (i < static_cast<int>(s_header.conv_desc_count.to_uint())) {
            s_conv_exec_desc[i] =
                load_conv_exec_desc(gmem_param, v4_conv_exec_offset(s_header) + i * CONV_EXEC_DESC_BLOB_BYTES);
        }
    }
    for (int i = 0; i < MAX_WINDOW_SCHED_COUNT; ++i) {
#pragma HLS PIPELINE off
        if (i < static_cast<int>(v4_window_sched_count(s_header).to_uint())) {
            s_window_sched_desc[i] =
                load_window_sched_desc(gmem_param, v4_window_sched_offset(s_header) + i * WINDOW_SCHED_DESC_BLOB_BYTES);
            if (!window_row_reuse_contract_valid(s_window_sched_desc[i])) {
                return;
            }
        }
    }
    // Validate physical row layout once at INIT; keep it out of the row hot path.
    bool referenced[MAX_WINDOW_SCHED_COUNT] = {};
    for (int i = 0; i < MAX_CONV_EXEC_DESC_COUNT; ++i) {
#pragma HLS PIPELINE off
        if (static_cast<unsigned>(i) >= s_header.conv_desc_count.to_uint()) {
            break;
        }
        const conv_exec_desc_t& conv = s_conv_exec_desc[i];
        const unsigned sid = conv.window_sched_id.to_uint();
        const unsigned tid = conv.src_tensor.to_uint();
        if (sid >= v4_window_sched_count(s_header).to_uint()) {
            return;
        }
        referenced[sid] = true;
        if (s_window_sched_desc[sid].reserved[WINDOW_LOADER_ROW_REUSE_WORD].to_uint() != 0U) {
            // BLOCK5 NONE schedules use row-local scratch IDs, not tensor-table IDs.
            if (tid >= s_header.tensor_desc_count.to_uint() ||
                !window_row_reuse_source_valid(s_window_sched_desc[sid], conv, s_tensor_desc[tid])) {
                return;
            }
        }
    }
    for (int i = 0; i < MAX_WINDOW_SCHED_COUNT; ++i) {
#pragma HLS PIPELINE off
        if (static_cast<unsigned>(i) < v4_window_sched_count(s_header).to_uint() &&
            s_window_sched_desc[i].reserved[WINDOW_LOADER_ROW_REUSE_WORD].to_uint() != 0U &&
            !referenced[i]) {
            return;
        }
    }
    for (int i = 0; i < MAX_ROW_CONSUMER_DESC_COUNT; ++i) {
#pragma HLS PIPELINE off
        if (i < static_cast<int>(v4_row_consumer_count(s_header).to_uint())) {
            s_row_consumer_desc[i] =
                load_row_consumer_desc(gmem_param, v4_row_consumer_offset(s_header) + i * ROW_CONSUMER_DESC_BLOB_BYTES);
        }
    }
    for (int i = 0; i < MAX_FIXED_EXEC_DESC_COUNT; ++i) {
#pragma HLS PIPELINE off
        if (i < static_cast<int>(v4_fixed_exec_count(s_header).to_uint())) {
            s_fixed_exec_desc[i] =
                load_fixed_exec_desc(gmem_param, v4_fixed_exec_offset(s_header) + i * FIXED_EXEC_DESC_BLOB_BYTES);
        }
    }
    for (int i = 0; i < MAX_EXEC_PLAN_COUNT; ++i) {
#pragma HLS PIPELINE off
        if (i < static_cast<int>(v4_exec_entry_count(s_header).to_uint())) {
            s_exec_plan[i] =
                load_exec_plan_entry(gmem_param, v4_exec_plan_offset(s_header) + i * EXEC_PLAN_ENTRY_BLOB_BYTES);
        }
    }
    for (int i = 0; i < MAX_BLOCK5_SCHED_COUNT; ++i) {
#pragma HLS PIPELINE off
        s_block5_sched[i] =
            load_block5_sched_desc(gmem_param, v4_block5_sched_offset(s_header) + i * BLOCK5_SCHED_DESC_BLOB_BYTES);
    }
    for (int i = 0; i < MAX_CONV_PARAM_DESC_COUNT; ++i) {
#pragma HLS PIPELINE off
        if (i < static_cast<int>(s_header.conv_desc_count.to_uint())) {
            s_conv_qparam[i] =
                load_conv_qparam_wordwise(gmem_param, s_header.conv_qparam_offset + i * sizeof(conv_qparam_t));
        }
    }
    u32_t affine_cursor = s_header.affine_qparam_offset;
    for (int i = 0; i < MAX_AFFINE_PARAM_DESC_COUNT; ++i) {
        if (i < static_cast<int>(s_header.affine_desc_count.to_uint())) {
            const unsigned blocks = k_v3_affine_blocks[i];
            s_affine_qparam_count[i] = static_cast<u8_t>((blocks > 8U) ? 8U : blocks);
            for (int block = 0; block < 8; ++block) {
#pragma HLS PIPELINE off
                if (static_cast<unsigned>(block) < blocks && block < 8) {
                    s_affine_qparam[i][block] =
                        load_affine_qparam_wordwise(gmem_param, affine_cursor + block * sizeof(affine_qparam_t));
                }
            }
            affine_cursor += static_cast<u32_t>(blocks * sizeof(affine_qparam_t));
        }
    }
    for (int i = 0; i < MAX_ADD_PARAM_DESC_COUNT; ++i) {
#pragma HLS PIPELINE off
        if (i < static_cast<int>(s_header.add_desc_count.to_uint())) {
            s_add_qparam[i] =
                load_add_qparam(gmem_param, s_header.add_qparam_offset + i * sizeof(add_qparam_t));
        }
    }
    for (int i = 0; i < MAX_POOL_PARAM_DESC_COUNT; ++i) {
#pragma HLS PIPELINE off
        if (i < static_cast<int>(s_header.pool_desc_count.to_uint())) {
            s_pool_qparam[i] =
                load_pool_qparam(gmem_param, s_header.pool_qparam_offset + i * sizeof(pool_qparam_t));
        }
    }
    s_ready = true;
}

bool param_dma_ready() {
#pragma HLS INLINE
    return s_ready;
}

bool param_dma_is_schedule_blob() {
#pragma HLS INLINE
    return s_ready && param_blob_is_v4();
}

bool param_dma_get_tensor_desc(u8_t tensor_id, tensor_desc_t& desc) {
#pragma HLS INLINE
    const int idx = static_cast<int>(tensor_id.to_uint());
    if (!s_ready || idx < 0 || idx >= static_cast<int>(s_header.tensor_desc_count.to_uint())) {
        return false;
    }
    desc = s_tensor_desc[idx];
    return true;
}

bool param_dma_get_pool_qparam(u8_t param_id, pool_q_t& qparam) {
#pragma HLS INLINE
    const int idx = static_cast<int>(param_id.to_uint());
    if (!s_ready || idx < 0 || idx >= static_cast<int>(s_header.pool_desc_count.to_uint())) {
        return false;
    }
    qparam = s_pool_qparam[idx];
    return true;
}

bool param_dma_get_conv_qparam(u8_t param_id, conv_q_t& qparam) {
#pragma HLS INLINE
    const int idx = static_cast<int>(param_id.to_uint());
    if (!s_ready || idx < 0 || idx >= static_cast<int>(s_header.conv_desc_count.to_uint())) {
        return false;
    }
    qparam = s_conv_qparam[idx];
    return true;
}

bool param_dma_get_conv_exec_desc(u8_t id, conv_exec_desc_t& desc) {
#pragma HLS INLINE
    const int idx = static_cast<int>(id.to_uint());
    if (!s_ready || !param_blob_is_v4() ||
        idx < 0 || idx >= static_cast<int>(s_header.conv_desc_count.to_uint())) {
        return false;
    }
    desc = s_conv_exec_desc[idx];
    return true;
}

bool param_dma_get_window_sched(u8_t id, window_sched_desc_t& desc) {
#pragma HLS INLINE
    const int idx = static_cast<int>(id.to_uint());
    if (!s_ready || !param_blob_is_v4() ||
        idx < 0 || idx >= static_cast<int>(v4_window_sched_count(s_header).to_uint())) {
        return false;
    }
    desc = s_window_sched_desc[idx];
    return window_row_reuse_contract_valid(desc);
}

bool param_dma_get_row_consumer(u8_t id, row_consumer_desc_t& desc) {
#pragma HLS INLINE
    const int idx = static_cast<int>(id.to_uint());
    if (!s_ready || !param_blob_is_v4() ||
        idx < 0 || idx >= static_cast<int>(v4_row_consumer_count(s_header).to_uint())) {
        return false;
    }
    desc = s_row_consumer_desc[idx];
    return true;
}

bool param_dma_get_exec_entry(u8_t pc, exec_plan_entry_t& entry) {
#pragma HLS INLINE
    const int idx = static_cast<int>(pc.to_uint());
    if (!s_ready || !param_blob_is_v4() ||
        idx < 0 || idx >= static_cast<int>(v4_exec_entry_count(s_header).to_uint())) {
        return false;
    }
    entry = s_exec_plan[idx];
    return true;
}

bool param_dma_get_fixed_exec_desc(u8_t id, fixed_exec_desc_t& desc) {
#pragma HLS INLINE
    const int idx = static_cast<int>(id.to_uint());
    if (!s_ready || !param_blob_is_v4() ||
        idx < 0 || idx >= static_cast<int>(v4_fixed_exec_count(s_header).to_uint())) {
        return false;
    }
    desc = s_fixed_exec_desc[idx];
    return true;
}

bool param_dma_get_block5_sched(u8_t id, block5_sched_desc_t& desc) {
#pragma HLS INLINE
    const int idx = static_cast<int>(id.to_uint());
    if (!s_ready || !param_blob_is_v4() ||
        v4_block5_sched_offset(s_header).to_uint() == 0U ||
        idx < 0 || idx >= MAX_BLOCK5_SCHED_COUNT) {
        return false;
    }
    desc = s_block5_sched[idx];
    return desc.pattern.to_uint() != static_cast<unsigned>(BLOCK5_PATTERN_INVALID);
}

bool param_dma_get_packed_weight_vec(const conv_exec_desc_t& desc,
                                     u16_t tm,
                                     u16_t kt,
                                     wgt_vec_t& word) {
#pragma HLS INLINE off
    if (!s_ready || !param_blob_is_v4() ||
        tm >= desc.out_c ||
        kt >= desc.k_tiles) {
        word = 0;
        return false;
    }
    const u32_t local_word =
        desc.packed_weight_word_offset +
        static_cast<u32_t>(tm) * static_cast<u32_t>(desc.k_tiles) +
        static_cast<u32_t>(kt);
    return read_wbuf_word(local_word, word);
}

bool param_dma_get_affine_qparam(u8_t param_id, u8_t block_id, aff_q_t& qparam) {
#pragma HLS INLINE
    const int idx = static_cast<int>(param_id.to_uint());
    const int block = static_cast<int>(block_id.to_uint());
    if (!s_ready || idx < 0 || idx >= static_cast<int>(s_header.affine_desc_count.to_uint()) ||
        block < 0 || block >= static_cast<int>(s_affine_qparam_count[idx].to_uint())) {
        return false;
    }
    qparam = s_affine_qparam[idx][block];
    return true;
}

bool param_dma_get_add_qparam(u8_t param_id, add_q_t& qparam) {
#pragma HLS INLINE
    const int idx = static_cast<int>(param_id.to_uint());
    if (!s_ready || idx < 0 || idx >= static_cast<int>(s_header.add_desc_count.to_uint())) {
        return false;
    }
    qparam = s_add_qparam[idx];
    return true;
}

}  // namespace esp_int8
