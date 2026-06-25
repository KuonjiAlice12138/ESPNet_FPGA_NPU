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
static window_pack_cmd_t s_window_pack_cmd[MAX_WINDOW_PACK_CMD_COUNT];
static row_consumer_desc_t s_row_consumer_desc[MAX_ROW_CONSUMER_DESC_COUNT];
static exec_plan_entry_t s_exec_plan[MAX_EXEC_PLAN_COUNT];
static u32_t s_weight_bytes = 0;

static bool s_ready = false;

static const unsigned k_v3_affine_blocks[MAX_AFFINE_PARAM_DESC_COUNT] = {1, 2, 2, 5, 4, 4, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0};

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

static bool param_blob_is_v3() {
#pragma HLS INLINE
    return s_header.version.to_uint() == PARAM_BLOB_VERSION_SCHED;
}

static u32_t v3_exec_entry_count(const param_blob_header_t& header) {
#pragma HLS INLINE
    return header.reserved0;
}

static u32_t v3_conv_exec_offset(const param_blob_header_t& header) {
#pragma HLS INLINE
    return header.reserved1[0];
}

static u32_t v3_window_sched_offset(const param_blob_header_t& header) {
#pragma HLS INLINE
    return header.reserved1[1];
}

static u32_t v3_window_pack_cmd_offset(const param_blob_header_t& header) {
#pragma HLS INLINE
    return header.reserved1[2];
}

static u32_t v3_row_consumer_offset(const param_blob_header_t& header) {
#pragma HLS INLINE
    return header.reserved1[3];
}

static u32_t v3_exec_plan_offset(const param_blob_header_t& header) {
#pragma HLS INLINE
    return header.reserved1[5];
}

static u32_t v3_window_sched_count(const param_blob_header_t& header) {
#pragma HLS INLINE
    return header.reserved1[6];
}

static u32_t v3_window_pack_cmd_count(const param_blob_header_t& header) {
#pragma HLS INLINE
    return header.reserved1[7];
}

static u32_t v3_row_consumer_count(const param_blob_header_t& header) {
#pragma HLS INLINE
    return header.reserved1[9];
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
        v3_window_sched_count(header) > MAX_WINDOW_SCHED_COUNT ||
        v3_window_pack_cmd_count(header) > MAX_WINDOW_PACK_CMD_COUNT ||
        v3_row_consumer_count(header) > MAX_ROW_CONSUMER_DESC_COUNT ||
        v3_exec_entry_count(header) > MAX_EXEC_PLAN_COUNT) {
        return ERR_PARAM_DESC_RANGE;
    }
    if (!is_aligned64(v3_conv_exec_offset(header)) ||
        !is_aligned64(v3_window_sched_offset(header)) ||
        !is_aligned64(v3_window_pack_cmd_offset(header)) ||
        !is_aligned64(v3_row_consumer_offset(header)) ||
        !is_aligned64(v3_exec_plan_offset(header))) {
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

static window_pack_cmd_t load_window_pack_cmd(const axi_vec_t* gmem_param, u32_t offset) {
#pragma HLS INLINE
    window_pack_cmd_t cmd;
    cmd.spatial_id = read_u8(gmem_param, offset + 0);
    cmd.src_c_begin = read_u8(gmem_param, offset + 1);
    cmd.dst_lane_begin = read_u8(gmem_param, offset + 2);
    cmd.byte_count = read_u8(gmem_param, offset + 3);
    cmd.flags = read_u8(gmem_param, offset + 4);
    cmd.reserved0 = read_u8(gmem_param, offset + 5);
    cmd.reserved1 = read_u16_le(gmem_param, offset + 6);
    return cmd;
}

static window_sched_desc_t load_window_sched_desc(const axi_vec_t* gmem_param, u32_t offset) {
#pragma HLS INLINE
    window_sched_desc_t desc;
    desc.mode = read_u8(gmem_param, offset + 0);
    desc.kernel = read_u8(gmem_param, offset + 1);
    desc.stride = read_u8(gmem_param, offset + 2);
    desc.dilation = read_u8(gmem_param, offset + 3);
    desc.in_c = read_u16_le(gmem_param, offset + 4);
    desc.k_tiles = read_u16_le(gmem_param, offset + 6);
    desc.cmd_base = read_u16_le(gmem_param, offset + 8);
    desc.cmd_count = read_u16_le(gmem_param, offset + 10);
    for (int i = 0; i < MAX_K_TILE_COUNT + 1; ++i) {
#pragma HLS PIPELINE off
        desc.kt_cmd_base[i] = read_u16_le(gmem_param, offset + 12 + i * 2);
    }
    desc.flags = read_u16_le(gmem_param, offset + 94);
    return desc;
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
#pragma HLS BIND_STORAGE variable=s_conv_exec_desc type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=s_window_sched_desc type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=s_window_pack_cmd type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=s_row_consumer_desc type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=s_exec_plan type=ram_2p impl=bram
#pragma HLS RESET variable=s_conv_exec_desc off
#pragma HLS RESET variable=s_window_sched_desc off
#pragma HLS RESET variable=s_window_pack_cmd off
#pragma HLS RESET variable=s_row_consumer_desc off
#pragma HLS RESET variable=s_exec_plan off
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
    for (int i = 0; i < MAX_CONV_EXEC_DESC_COUNT; ++i) {
#pragma HLS PIPELINE off
        if (i < static_cast<int>(s_header.conv_desc_count.to_uint())) {
            s_conv_exec_desc[i] =
                load_conv_exec_desc(gmem_param, v3_conv_exec_offset(s_header) + i * CONV_EXEC_DESC_BLOB_BYTES);
        }
    }
    for (int i = 0; i < MAX_WINDOW_SCHED_COUNT; ++i) {
#pragma HLS PIPELINE off
        if (i < static_cast<int>(v3_window_sched_count(s_header).to_uint())) {
            s_window_sched_desc[i] =
                load_window_sched_desc(gmem_param, v3_window_sched_offset(s_header) + i * WINDOW_SCHED_DESC_BLOB_BYTES);
        }
    }
    for (int i = 0; i < MAX_WINDOW_PACK_CMD_COUNT; ++i) {
#pragma HLS PIPELINE off
        if (i < static_cast<int>(v3_window_pack_cmd_count(s_header).to_uint())) {
            s_window_pack_cmd[i] =
                load_window_pack_cmd(gmem_param, v3_window_pack_cmd_offset(s_header) + i * WINDOW_PACK_CMD_BLOB_BYTES);
        }
    }
    for (int i = 0; i < MAX_ROW_CONSUMER_DESC_COUNT; ++i) {
#pragma HLS PIPELINE off
        if (i < static_cast<int>(v3_row_consumer_count(s_header).to_uint())) {
            s_row_consumer_desc[i] =
                load_row_consumer_desc(gmem_param, v3_row_consumer_offset(s_header) + i * ROW_CONSUMER_DESC_BLOB_BYTES);
        }
    }
    for (int i = 0; i < MAX_EXEC_PLAN_COUNT; ++i) {
#pragma HLS PIPELINE off
        if (i < static_cast<int>(v3_exec_entry_count(s_header).to_uint())) {
            s_exec_plan[i] =
                load_exec_plan_entry(gmem_param, v3_exec_plan_offset(s_header) + i * EXEC_PLAN_ENTRY_BLOB_BYTES);
        }
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
    return s_ready && param_blob_is_v3();
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
    if (!s_ready || !param_blob_is_v3() ||
        idx < 0 || idx >= static_cast<int>(s_header.conv_desc_count.to_uint())) {
        return false;
    }
    desc = s_conv_exec_desc[idx];
    return true;
}

bool param_dma_get_window_sched(u8_t id, window_sched_desc_t& desc) {
#pragma HLS INLINE
    const int idx = static_cast<int>(id.to_uint());
    if (!s_ready || !param_blob_is_v3() ||
        idx < 0 || idx >= static_cast<int>(v3_window_sched_count(s_header).to_uint())) {
        return false;
    }
    desc = s_window_sched_desc[idx];
    return true;
}

bool param_dma_get_pack_cmd(u16_t cmd_idx, window_pack_cmd_t& cmd) {
#pragma HLS INLINE
    const int idx = static_cast<int>(cmd_idx.to_uint());
    if (!s_ready || !param_blob_is_v3() ||
        idx < 0 || idx >= static_cast<int>(v3_window_pack_cmd_count(s_header).to_uint())) {
        return false;
    }
    cmd = s_window_pack_cmd[idx];
    return true;
}

bool param_dma_get_row_consumer(u8_t id, row_consumer_desc_t& desc) {
#pragma HLS INLINE
    const int idx = static_cast<int>(id.to_uint());
    if (!s_ready || !param_blob_is_v3() ||
        idx < 0 || idx >= static_cast<int>(v3_row_consumer_count(s_header).to_uint())) {
        return false;
    }
    desc = s_row_consumer_desc[idx];
    return true;
}

bool param_dma_get_exec_entry(u8_t pc, exec_plan_entry_t& entry) {
#pragma HLS INLINE
    const int idx = static_cast<int>(pc.to_uint());
    if (!s_ready || !param_blob_is_v3() ||
        idx < 0 || idx >= static_cast<int>(v3_exec_entry_count(s_header).to_uint())) {
        return false;
    }
    entry = s_exec_plan[idx];
    return true;
}

bool param_dma_get_packed_weight_vec(const conv_exec_desc_t& desc,
                                     u16_t tm,
                                     u16_t kt,
                                     wgt_vec_t& word) {
#pragma HLS INLINE off
    if (!s_ready || !param_blob_is_v3() ||
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
