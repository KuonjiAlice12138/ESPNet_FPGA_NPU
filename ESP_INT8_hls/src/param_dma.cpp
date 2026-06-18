#include "../include/npu_config.hpp"
#include "../include/npu_uop.hpp"

namespace esp_int8 {

static param_blob_header_t s_header;
static tensor_desc_t s_tensor_desc[MAX_TENSOR_DESC_COUNT];
static conv_param_desc_t s_conv_desc[MAX_CONV_PARAM_DESC_COUNT];
static affine_param_desc_t s_affine_desc[MAX_AFFINE_PARAM_DESC_COUNT];
static add_param_desc_t s_add_desc[MAX_ADD_PARAM_DESC_COUNT];
static pool_param_desc_t s_pool_desc[MAX_POOL_PARAM_DESC_COUNT];
static conv_qparam_t s_conv_qparam[MAX_CONV_PARAM_DESC_COUNT];
static affine_qparam_t s_affine_qparam[MAX_AFFINE_PARAM_DESC_COUNT][8];
static u8_t s_affine_qparam_count[MAX_AFFINE_PARAM_DESC_COUNT];
static add_qparam_t s_add_qparam[MAX_ADD_PARAM_DESC_COUNT];
static pool_qparam_t s_pool_qparam[MAX_POOL_PARAM_DESC_COUNT];
static u32_t s_weight_bytes = 0;

static bool s_ready = false;

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

static bool is_aligned64(u32_t offset) {
#pragma HLS INLINE
    return (offset & (SECTION_ALIGNMENT_BYTES - 1)) == 0;
}

static u16_t effective_kernel(const conv_cfg_t& cfg) {
#pragma HLS INLINE
    return (cfg.kernel == 1) ? static_cast<u16_t>(1) : static_cast<u16_t>(3);
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
    if (header.magic != PARAM_BLOB_MAGIC || header.version != PARAM_BLOB_VERSION) {
        return ERR_BAD_BLOB;
    }
    if (header.uop_count.to_uint() != static_cast<unsigned>(UOP_COUNT_ENCODER)) {
        return ERR_UOP_DECODE;
    }
    if (header.tensor_desc_count > MAX_TENSOR_DESC_COUNT ||
        header.scale_desc_count > SCALE_DESC_COUNT_MAX) {
        return ERR_TENSOR_DESC_RANGE;
    }
    if (header.conv_desc_count > MAX_CONV_PARAM_DESC_COUNT ||
        header.affine_desc_count > MAX_AFFINE_PARAM_DESC_COUNT ||
        header.add_desc_count > MAX_ADD_PARAM_DESC_COUNT ||
        header.pool_desc_count > MAX_POOL_PARAM_DESC_COUNT) {
        return ERR_PARAM_DESC_RANGE;
    }
    if (!is_aligned64(header.tensor_desc_offset) ||
        !is_aligned64(header.scale_desc_offset) ||
        !is_aligned64(header.conv_desc_offset) ||
        !is_aligned64(header.affine_desc_offset) ||
        !is_aligned64(header.add_desc_offset) ||
        !is_aligned64(header.pool_desc_offset) ||
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

static conv_param_desc_t load_conv_desc(const axi_vec_t* gmem_param, u32_t offset) {
#pragma HLS INLINE
    conv_param_desc_t desc;
    desc.weight_offset = read_u32_le(gmem_param, offset + 0);
    desc.bias_offset = read_u32_le(gmem_param, offset + 4);
    desc.requant_offset = read_u32_le(gmem_param, offset + 8);
    desc.reserved = read_u32_le(gmem_param, offset + 12);
    return desc;
}

static affine_param_desc_t load_affine_desc(const axi_vec_t* gmem_param, u32_t offset) {
#pragma HLS INLINE
    affine_param_desc_t desc;
    desc.affine_offset = read_u32_le(gmem_param, offset + 0);
    desc.reserved0 = read_u32_le(gmem_param, offset + 4);
    desc.reserved1 = read_u32_le(gmem_param, offset + 8);
    desc.reserved2 = read_u32_le(gmem_param, offset + 12);
    return desc;
}

static add_param_desc_t load_add_desc(const axi_vec_t* gmem_param, u32_t offset) {
#pragma HLS INLINE
    add_param_desc_t desc;
    desc.add_offset = read_u32_le(gmem_param, offset + 0);
    desc.reserved0 = read_u32_le(gmem_param, offset + 4);
    desc.reserved1 = read_u32_le(gmem_param, offset + 8);
    desc.reserved2 = read_u32_le(gmem_param, offset + 12);
    return desc;
}

static pool_param_desc_t load_pool_desc(const axi_vec_t* gmem_param, u32_t offset) {
#pragma HLS INLINE
    pool_param_desc_t desc;
    desc.pool_offset = read_u32_le(gmem_param, offset + 0);
    desc.reserved0 = read_u32_le(gmem_param, offset + 4);
    desc.reserved1 = read_u32_le(gmem_param, offset + 8);
    desc.reserved2 = read_u32_le(gmem_param, offset + 12);
    return desc;
}

static conv_qparam_t load_conv_qparam(const axi_vec_t* gmem_param, u32_t offset) {
#pragma HLS INLINE
    conv_qparam_t qparam;
    for (int i = 0; i < 32; ++i) {
        qparam.bias[i] = read_i32_le(gmem_param, offset + i * 4);
        qparam.mult[i] = read_i32_le(gmem_param, offset + 128 + i * 4);
        qparam.shift[i] = read_u8(gmem_param, offset + 256 + i);
        qparam.reserved[i] = read_u8(gmem_param, offset + 288 + i);
    }
    return qparam;
}

static affine_qparam_t load_affine_qparam(const axi_vec_t* gmem_param, u32_t offset) {
#pragma HLS INLINE
    affine_qparam_t qparam;
    for (int i = 0; i < 32; ++i) {
        qparam.mul[i] = read_i32_le(gmem_param, offset + i * 4);
        qparam.bias[i] = read_i32_le(gmem_param, offset + 128 + i * 4);
        qparam.shift[i] = read_u8(gmem_param, offset + 256 + i);
        qparam.reserved[i] = read_u8(gmem_param, offset + 288 + i);
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

static bool read_wbuf_i8(u32_t local_offset, i8_t& value) {
#pragma HLS INLINE
    if (local_offset >= s_weight_bytes) {
        value = 0;
        return false;
    }

    const axi_vec_t word = get_wbuf()[local_offset >> 5];
    const unsigned lane = local_offset.to_uint() & 0x1fU;
    u8_t raw = word.range(lane * 8 + 7, lane * 8);
    value.range(7, 0) = raw;
    return true;
}

static void set_wgt_vec_lane_dynamic(wgt_vec_t& word, int lane, i8_t value) {
#pragma HLS INLINE
    u8_t raw = 0;
    raw.range(7, 0) = value.range(7, 0);
    const wgt_vec_t widened = static_cast<wgt_vec_t>(raw);
    word |= static_cast<wgt_vec_t>(widened << (lane * 8));
}

void param_dma_init(const axi_vec_t* gmem_param) {
#pragma HLS INLINE off
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
    for (int i = 0; i < MAX_CONV_PARAM_DESC_COUNT; ++i) {
#pragma HLS PIPELINE off
        if (i < static_cast<int>(s_header.conv_desc_count.to_uint())) {
            s_conv_desc[i] = load_conv_desc(gmem_param, s_header.conv_desc_offset + i * 16);
        }
    }
    for (int i = 0; i < MAX_AFFINE_PARAM_DESC_COUNT; ++i) {
#pragma HLS PIPELINE off
        if (i < static_cast<int>(s_header.affine_desc_count.to_uint())) {
            s_affine_desc[i] = load_affine_desc(gmem_param, s_header.affine_desc_offset + i * 16);
        }
    }
    for (int i = 0; i < MAX_ADD_PARAM_DESC_COUNT; ++i) {
#pragma HLS PIPELINE off
        if (i < static_cast<int>(s_header.add_desc_count.to_uint())) {
            s_add_desc[i] = load_add_desc(gmem_param, s_header.add_desc_offset + i * 16);
        }
    }
    for (int i = 0; i < MAX_POOL_PARAM_DESC_COUNT; ++i) {
#pragma HLS PIPELINE off
        if (i < static_cast<int>(s_header.pool_desc_count.to_uint())) {
            s_pool_desc[i] = load_pool_desc(gmem_param, s_header.pool_desc_offset + i * 16);
        }
    }
    for (int i = 0; i < MAX_CONV_PARAM_DESC_COUNT; ++i) {
#pragma HLS PIPELINE off
        if (i < static_cast<int>(s_header.conv_desc_count.to_uint())) {
            s_conv_qparam[i] = load_conv_qparam(gmem_param, s_conv_desc[i].requant_offset);
        }
    }
    for (int i = 0; i < MAX_AFFINE_PARAM_DESC_COUNT; ++i) {
        if (i < static_cast<int>(s_header.affine_desc_count.to_uint())) {
            const u32_t start = s_affine_desc[i].affine_offset;
            const u32_t end = (i + 1 < static_cast<int>(s_header.affine_desc_count.to_uint()))
                                  ? s_affine_desc[i + 1].affine_offset
                                  : s_header.add_qparam_offset;
            const unsigned byte_count = (end > start) ? static_cast<unsigned>((end - start).to_uint()) : 0U;
            const unsigned blocks = byte_count / sizeof(affine_qparam_t);
            const unsigned capped_blocks = (blocks > 8U) ? 8U : blocks;
            s_affine_qparam_count[i] = static_cast<u8_t>(capped_blocks);
            for (int block = 0; block < 8; ++block) {
#pragma HLS PIPELINE off
                if (static_cast<unsigned>(block) < capped_blocks) {
                    s_affine_qparam[i][block] =
                        load_affine_qparam(gmem_param, start + block * sizeof(affine_qparam_t));
                }
            }
        }
    }
    for (int i = 0; i < MAX_ADD_PARAM_DESC_COUNT; ++i) {
#pragma HLS PIPELINE off
        if (i < static_cast<int>(s_header.add_desc_count.to_uint())) {
            s_add_qparam[i] = load_add_qparam(gmem_param, s_add_desc[i].add_offset);
        }
    }
    for (int i = 0; i < MAX_POOL_PARAM_DESC_COUNT; ++i) {
#pragma HLS PIPELINE off
        if (i < static_cast<int>(s_header.pool_desc_count.to_uint())) {
            s_pool_qparam[i] = load_pool_qparam(gmem_param, s_pool_desc[i].pool_offset);
        }
    }
    s_ready = true;
}

bool param_dma_ready() {
#pragma HLS INLINE
    return s_ready;
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

bool param_dma_get_weight_vec(u8_t param_id,
                              u16_t oc,
                              u16_t kt,
                              const conv_cfg_t& cfg,
                               wgt_vec_t& word) {
#pragma HLS INLINE off
    const int idx = static_cast<int>(param_id.to_uint());
    if (!s_ready || idx < 0 || idx >= static_cast<int>(s_header.conv_desc_count.to_uint())) {
        word = 0;
        return false;
    }
    if (s_conv_desc[idx].weight_offset < s_header.weight_data_offset) {
        word = 0;
        return false;
    }

    const u16_t kernel = effective_kernel(cfg);
    const u16_t k_total = static_cast<u16_t>(cfg.in_c * kernel * kernel);
    const u32_t local_base = s_conv_desc[idx].weight_offset - s_header.weight_data_offset;
    const int in_c_i = static_cast<int>(cfg.in_c.to_uint());
    const int kernel_i = static_cast<int>(kernel.to_uint());
    const int k_idx_start = static_cast<int>(kt.to_uint()) * TK;
    int cin_i = k_idx_start;
    int spatial_i = 0;
    if (kernel_i != 1 && in_c_i != 0) {
        spatial_i = k_idx_start / in_c_i;
        cin_i = k_idx_start - spatial_i * in_c_i;
    }
    word = 0;

    for (int lane = 0; lane < TK; ++lane) {
#pragma HLS PIPELINE II=1
        const int k_idx_i = k_idx_start + lane;
        i8_t value = 0;
        bool valid = false;

        if (oc < cfg.out_c && k_idx_i < static_cast<int>(k_total.to_uint())) {
            u32_t weight_offset = local_base;
            if (kernel_i == 1) {
                weight_offset +=
                    static_cast<u32_t>(oc) * static_cast<u32_t>(cfg.in_c) +
                    static_cast<u32_t>(cin_i);
            } else {
                const int kh_i = spatial_i / kernel_i;
                const int kw_i = spatial_i - kh_i * kernel_i;
                weight_offset +=
                    ((static_cast<u32_t>(oc) * static_cast<u32_t>(cfg.in_c) +
                      static_cast<u32_t>(cin_i)) *
                         static_cast<u32_t>(kernel_i) +
                     static_cast<u32_t>(kh_i)) *
                        static_cast<u32_t>(kernel_i) +
                    static_cast<u32_t>(kw_i);
            }
            valid = read_wbuf_i8(weight_offset, value);
        }

        if (!valid) {
            value = 0;
        }
        set_wgt_vec_lane_dynamic(word, lane, value);

        if (kernel_i == 1) {
            ++cin_i;
        } else {
            ++cin_i;
            if (cin_i >= in_c_i) {
                cin_i = 0;
                ++spatial_i;
            }
        }
    }
    return true;
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
