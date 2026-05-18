#include "../include/npu_config.hpp"
#include "../include/npu_uop.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

namespace esp_int8 {
void param_dma_init(const axi_vec_t* gmem_param);
bool param_dma_ready();
error_code_t param_dma_error();
bool param_dma_get_header(param_blob_header_t& header);
bool param_dma_get_tensor_desc(u8_t tensor_id, tensor_desc_t& desc);
bool param_dma_get_uop(u16_t uop_id, uop_t& uop);
bool param_dma_get_pool_qparam(u8_t param_id, pool_q_t& qparam);
bool param_dma_get_conv_qparam(u8_t param_id, conv_q_t& qparam);
bool param_dma_get_affine_qparam(u8_t param_id, u8_t block_id, aff_q_t& qparam);
bool param_dma_get_add_qparam(u8_t param_id, add_q_t& qparam);
bool param_dma_get_weight_vec(u8_t param_id, u16_t oc, u16_t kt, const conv_cfg_t& cfg, wgt_vec_t& word);

void instruction_fetch_decode(const axi_vec_t* gmem_param, u32_t uop_count);
error_code_t if_dec_error();
u16_t if_dec_current_uop_id();
bool if_dec_get_current_uop(uop_t& uop);
}  // namespace esp_int8

static void set_byte(esp_int8::axi_vec_t& word, int lane, std::uint8_t value) {
    word.range(lane * 8 + 7, lane * 8) = value;
}

static bool expect(bool cond, const char* msg) {
    if (!cond) {
        std::printf("%s\n", msg);
        return false;
    }
    return true;
}

static bool check_tensor_desc(std::uint8_t tensor_id,
                              std::uint8_t bank,
                              std::uint32_t base,
                              std::uint16_t h,
                              std::uint16_t w,
                              std::uint16_t c,
                              std::uint16_t phys_c,
                              std::uint16_t c_offset) {
    esp_int8::tensor_desc_t desc;
    if (!esp_int8::param_dma_get_tensor_desc(static_cast<esp_int8::u8_t>(tensor_id), desc)) {
        std::printf("tensor desc %u unavailable\n", static_cast<unsigned>(tensor_id));
        return false;
    }
    if (desc.bank_id != bank || desc.base_offset != base ||
        desc.h != h || desc.w != w || desc.c != c ||
        desc.reserved0 != phys_c || desc.reserved1 != c_offset) {
        std::printf("tensor desc %u mismatch: bank=%u base=0x%x shape=%ux%ux%u phys_c=%u c_offset=%u\n",
                    static_cast<unsigned>(tensor_id),
                    static_cast<unsigned>(desc.bank_id),
                    static_cast<unsigned>(desc.base_offset),
                    static_cast<unsigned>(desc.h),
                    static_cast<unsigned>(desc.w),
                    static_cast<unsigned>(desc.c),
                    static_cast<unsigned>(desc.reserved0),
                    static_cast<unsigned>(desc.reserved1));
        return false;
    }
    return true;
}

static bool check_uop(std::uint16_t idx,
                      std::uint8_t opcode,
                      std::uint8_t src0,
                      std::uint8_t src1,
                      std::uint8_t dst,
                      std::uint8_t param_id,
                      std::uint16_t c_offset,
                      std::uint16_t valid_c) {
    esp_int8::uop_t uop;
    if (!esp_int8::param_dma_get_uop(static_cast<esp_int8::u16_t>(idx), uop)) {
        std::printf("uop %u unavailable\n", static_cast<unsigned>(idx));
        return false;
    }
    if (uop.opcode != opcode || uop.src0_tensor != src0 || uop.src1_tensor != src1 ||
        uop.dst_tensor != dst || uop.param_id != param_id ||
        uop.c_offset != c_offset || uop.valid_c != valid_c) {
        std::printf("uop %u mismatch: op=%u src0=%u src1=%u dst=%u param=%u c_offset=%u valid_c=%u\n",
                    static_cast<unsigned>(idx),
                    static_cast<unsigned>(uop.opcode),
                    static_cast<unsigned>(uop.src0_tensor),
                    static_cast<unsigned>(uop.src1_tensor),
                    static_cast<unsigned>(uop.dst_tensor),
                    static_cast<unsigned>(uop.param_id),
                    static_cast<unsigned>(uop.c_offset),
                    static_cast<unsigned>(uop.valid_c));
        return false;
    }
    return true;
}

static bool validate_tensor_contract() {
    bool ok = true;
    ok &= check_tensor_desc(esp_int8::TID_INPUT, esp_int8::BANK_FMEM0, 0x000000U, 512, 1024, 3, 3, 0);
    ok &= check_tensor_desc(esp_int8::TID_POOL1, esp_int8::BANK_FMEM0, 0x3E0000U, 256, 512, 3, 3, 0);
    ok &= check_tensor_desc(esp_int8::TID_L20_CAT, esp_int8::BANK_FMEM1, 0x418000U, 128, 256, 64, 64, 0);
    ok &= check_tensor_desc(esp_int8::TID_L2B0_CAT, esp_int8::BANK_FMEM0, 0x000000U, 128, 256, 64, 131, 0);
    ok &= check_tensor_desc(esp_int8::TID_L2B0_ACT, esp_int8::BANK_FMEM0, 0x000000U, 128, 256, 64, 131, 0);
    ok &= check_tensor_desc(esp_int8::TID_POOL2, esp_int8::BANK_BRAM_SCR1, 0x000000U, 128, 256, 3, 3, 0);
    ok &= check_tensor_desc(esp_int8::TID_L30_CAT, esp_int8::BANK_FMEM1, 0x418000U, 64, 128, 128, 128, 0);
    ok &= check_tensor_desc(esp_int8::TID_L3B0_CAT, esp_int8::BANK_FMEM0, 0x000000U, 64, 128, 128, 256, 128);
    ok &= check_tensor_desc(esp_int8::TID_L3B0_ACT, esp_int8::BANK_FMEM0, 0x000000U, 64, 128, 128, 256, 128);
    ok &= check_tensor_desc(esp_int8::TID_OUT, esp_int8::BANK_FMEM0, 0x200000U, 64, 128, 2, 2, 0);
    ok &= check_tensor_desc(esp_int8::TID_POOL_TMP, esp_int8::BANK_BRAM_SCR0, 0x000000U, 256, 512, 3, 3, 0);
    return ok;
}

static bool validate_key_uops() {
    bool ok = true;
    ok &= check_uop(0, esp_int8::UOP_LOAD_FM, esp_int8::TID_INVALID, esp_int8::TID_INVALID,
                    esp_int8::TID_INPUT, 0, 0, 3);
    ok &= check_uop(1, esp_int8::UOP_POOL, esp_int8::TID_INPUT, esp_int8::TID_INVALID,
                    esp_int8::TID_POOL1, 0, 0, 3);
    ok &= check_uop(2, esp_int8::UOP_CONV, esp_int8::TID_INPUT, esp_int8::TID_INVALID,
                    esp_int8::TID_B1_CAT, 0, 0, 16);
    ok &= check_uop(5, esp_int8::UOP_POOL, esp_int8::TID_INPUT, esp_int8::TID_INVALID,
                    esp_int8::TID_POOL_TMP, 1, 0, 3);
    ok &= check_uop(6, esp_int8::UOP_POOL, esp_int8::TID_POOL_TMP, esp_int8::TID_INVALID,
                    esp_int8::TID_POOL2, 2, 0, 3);
    ok &= check_uop(21, esp_int8::UOP_CONV, esp_int8::TID_L20_ACT, esp_int8::TID_INVALID,
                    esp_int8::LS_C1, 7, 0, 12);
    ok &= check_uop(34, esp_int8::UOP_ADD, esp_int8::TID_L2B0_CAT, esp_int8::TID_L20_ACT,
                    esp_int8::TID_L2B0_CAT, 6, 0, 64);
    ok &= check_uop(36, esp_int8::UOP_STORE, esp_int8::TID_L2B0_ACT, esp_int8::TID_INVALID,
                    esp_int8::TID_B2_CAT, 0, 0, 64);
    ok &= check_uop(37, esp_int8::UOP_STORE, esp_int8::TID_L20_ACT, esp_int8::TID_INVALID,
                    esp_int8::TID_B2_CAT, 0, 64, 64);
    ok &= check_uop(38, esp_int8::UOP_STORE, esp_int8::TID_POOL2, esp_int8::TID_INVALID,
                    esp_int8::TID_B2_CAT, 0, 128, 3);
    ok &= check_uop(54, esp_int8::UOP_CONV, esp_int8::TID_L30_ACT, esp_int8::TID_INVALID,
                    esp_int8::LS_C1, 19, 0, 25);
    ok &= check_uop(67, esp_int8::UOP_ADD, esp_int8::TID_L3B0_CAT, esp_int8::TID_L30_ACT,
                    esp_int8::TID_L3B0_CAT, 13, 0, 128);
    ok &= check_uop(69, esp_int8::UOP_STORE, esp_int8::TID_L30_ACT, esp_int8::TID_INVALID,
                    esp_int8::TID_B3_CAT, 0, 0, 128);
    ok &= check_uop(70, esp_int8::UOP_STORE, esp_int8::TID_L3B0_ACT, esp_int8::TID_INVALID,
                    esp_int8::TID_B3_CAT, 0, 128, 128);
    ok &= check_uop(72, esp_int8::UOP_CONV, esp_int8::TID_B3_ACT, esp_int8::TID_INVALID,
                    esp_int8::TID_OUT, 25, 0, 2);
    ok &= check_uop(73, esp_int8::UOP_STORE, esp_int8::TID_OUT, esp_int8::TID_INVALID,
                    esp_int8::TID_INVALID, 0, 0, 2);
    ok &= check_uop(74, esp_int8::UOP_END, esp_int8::TID_INVALID, esp_int8::TID_INVALID,
                    esp_int8::TID_INVALID, 0, 0, 0);
    return ok;
}

static bool validate_uop_ranges() {
    static const std::uint8_t conv_params[26] = {
        0, 1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24,
        25,
    };
    bool conv_seen[26] = {};
    bool ok = true;

    for (int i = 0; i < esp_int8::UOP_COUNT_ENCODER; ++i) {
        esp_int8::uop_t uop;
        if (!esp_int8::param_dma_get_uop(static_cast<esp_int8::u16_t>(i), uop)) {
            std::printf("uop %d unavailable during range check\n", i);
            return false;
        }
        const unsigned opcode = uop.opcode.to_uint();
        if (opcode == esp_int8::UOP_CONV) {
            ok &= expect(uop.out_c.to_uint() <= esp_int8::TM, "conv out_c exceeds TM");
            ok &= expect(uop.param_id.to_uint() < esp_int8::CONV_PARAM_DESC_COUNT, "conv param_id out of range");
            ok &= expect(uop.valid_c == uop.out_c, "conv valid_c must equal out_c");
            ok &= expect(uop.kernel == 1 || uop.kernel == 3, "conv kernel must be 1 or 3");
            ok &= expect(uop.dilation == 1 || uop.dilation == 2 || uop.dilation == 4 ||
                         uop.dilation == 8 || uop.dilation == 16, "conv dilation unsupported");
            conv_seen[uop.param_id.to_uint()] = true;
        } else if (opcode == esp_int8::UOP_ADD) {
            ok &= expect(uop.param_id.to_uint() < esp_int8::ADD_PARAM_DESC_COUNT, "add param_id out of range");
            ok &= expect(uop.valid_c.to_uint() != 0U, "add valid_c must be non-zero");
        } else if (opcode == esp_int8::UOP_AFFINE) {
            ok &= expect(uop.param_id.to_uint() < esp_int8::AFFINE_PARAM_DESC_COUNT, "affine param_id out of range");
            ok &= expect(uop.valid_c.to_uint() != 0U, "affine valid_c must be non-zero");
        } else if (opcode == esp_int8::UOP_POOL) {
            ok &= expect(uop.param_id.to_uint() < esp_int8::POOL_PARAM_DESC_COUNT, "pool param_id out of range");
            ok &= expect(uop.valid_c == 3, "pool valid_c must be 3");
        }
    }

    for (unsigned i = 0; i < sizeof(conv_params); ++i) {
        if (!conv_seen[conv_params[i]]) {
            std::printf("conv param %u not referenced by uop table\n", conv_params[i]);
            ok = false;
        }
    }
    return ok;
}

static bool validate_weight_buffer_access() {
    esp_int8::conv_cfg_t cfg;
    cfg.in_h = 512;
    cfg.in_w = 1024;
    cfg.in_c = 3;
    cfg.out_c = 16;
    cfg.kernel = 3;
    cfg.stride = 2;
    cfg.dilation = 1;
    cfg.bias_en = 1;

    esp_int8::wgt_vec_t word = 0;
    if (!esp_int8::param_dma_get_weight_vec(0, 0, 0, cfg, word)) {
        std::printf("weight vec for conv param 0 unavailable\n");
        return false;
    }

    cfg.in_h = 64;
    cfg.in_w = 128;
    cfg.in_c = 256;
    cfg.out_c = 2;
    cfg.kernel = 1;
    cfg.stride = 1;
    cfg.dilation = 1;
    if (!esp_int8::param_dma_get_weight_vec(25, 1, 7, cfg, word)) {
        std::printf("weight vec for classifier tail unavailable\n");
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    const char* blob_path = (argc > 1) ? argv[1] : "D:/ESP_INT8/hw_artifacts/hw_constrained_qat_3ep_single/param_blob.bin";

    std::ifstream in(blob_path, std::ios::binary);
    if (!in) {
        std::printf("failed to open blob: %s\n", blob_path);
        return 1;
    }

    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)), {});
    if (bytes.size() < esp_int8::PARAM_HEADER_BYTES) {
        std::printf("blob too small: %zu bytes\n", bytes.size());
        return 1;
    }

    const std::size_t word_count = (bytes.size() + esp_int8::AXI_WORD_BYTES - 1) / esp_int8::AXI_WORD_BYTES;
    std::vector<esp_int8::axi_vec_t> param(word_count);
    for (std::size_t i = 0; i < word_count; ++i) {
        param[i] = 0;
    }
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        set_byte(param[i / esp_int8::AXI_WORD_BYTES], static_cast<int>(i % esp_int8::AXI_WORD_BYTES), bytes[i]);
    }

    esp_int8::param_dma_init(param.data());
    if (!esp_int8::param_dma_ready()) {
        std::printf("param_dma_init failed, error=%u\n", static_cast<unsigned>(esp_int8::param_dma_error()));
        return 1;
    }

    esp_int8::param_blob_header_t header;
    if (!esp_int8::param_dma_get_header(header)) {
        std::printf("header unavailable\n");
        return 1;
    }
    if (header.uop_count != esp_int8::UOP_COUNT_ENCODER) {
        std::printf("unexpected uop_count=%u\n", static_cast<unsigned>(header.uop_count));
        return 1;
    }
    if (header.pool_desc_count != esp_int8::POOL_PARAM_DESC_COUNT) {
        std::printf("unexpected pool_desc_count=%u\n", static_cast<unsigned>(header.pool_desc_count));
        return 1;
    }
    if (header.conv_desc_count != esp_int8::CONV_PARAM_DESC_COUNT ||
        header.affine_desc_count != esp_int8::AFFINE_PARAM_DESC_COUNT ||
        header.add_desc_count != esp_int8::ADD_PARAM_DESC_COUNT) {
        std::printf("unexpected param counts: conv=%u affine=%u add=%u\n",
                    static_cast<unsigned>(header.conv_desc_count),
                    static_cast<unsigned>(header.affine_desc_count),
                    static_cast<unsigned>(header.add_desc_count));
        return 1;
    }
    if (!validate_tensor_contract()) {
        return 1;
    }
    if (!validate_key_uops()) {
        return 1;
    }
    if (!validate_uop_ranges()) {
        return 1;
    }

    esp_int8::instruction_fetch_decode(param.data(), header.uop_count);
    if (esp_int8::if_dec_error() != esp_int8::ERR_NONE) {
        std::printf("if_dec failed, error=%u, uop=%u\n",
                    static_cast<unsigned>(esp_int8::if_dec_error()),
                    static_cast<unsigned>(esp_int8::if_dec_current_uop_id()));
        return 1;
    }

    esp_int8::uop_t last;
    if (!esp_int8::param_dma_get_uop(esp_int8::UOP_COUNT_ENCODER - 1, last) || last.opcode != esp_int8::UOP_END) {
        std::printf("last uop is not END\n");
        return 1;
    }

    for (int i = 0; i < esp_int8::POOL_PARAM_DESC_COUNT; ++i) {
        esp_int8::pool_q_t pool_q;
        if (!esp_int8::param_dma_get_pool_qparam(static_cast<esp_int8::u8_t>(i), pool_q)) {
            std::printf("pool qparam %d unavailable\n", i);
            return 1;
        }
        const bool expected_same_scale = (i == 1);
        if (pool_q.kernel != 3 || pool_q.stride != 2 ||
            (pool_q.same_scale.to_uint() != (expected_same_scale ? 1U : 0U))) {
            std::printf("unexpected pool qparam %d: kernel=%u stride=%u same_scale=%u\n",
                        i,
                        static_cast<unsigned>(pool_q.kernel),
                        static_cast<unsigned>(pool_q.stride),
                        static_cast<unsigned>(pool_q.same_scale));
            return 1;
        }
    }

    esp_int8::conv_q_t conv_q;
    if (!esp_int8::param_dma_get_conv_qparam(0, conv_q)) {
        std::printf("conv qparam 0 unavailable\n");
        return 1;
    }
    if (!validate_weight_buffer_access()) {
        return 1;
    }

    esp_int8::aff_q_t aff_q;
    if (!esp_int8::param_dma_get_affine_qparam(6, 7, aff_q)) {
        std::printf("affine qparam param=6 block=7 unavailable\n");
        return 1;
    }

    for (int i = 0; i < esp_int8::ADD_PARAM_DESC_COUNT; ++i) {
        esp_int8::add_q_t add_q;
        if (!esp_int8::param_dma_get_add_qparam(static_cast<esp_int8::u8_t>(i), add_q)) {
            std::printf("add qparam %d unavailable\n", i);
            return 1;
        }
        if (add_q.requant_bypass.to_uint() != 1U) {
            std::printf("unexpected add qparam %d requant_bypass=%u\n",
                        i,
                        static_cast<unsigned>(add_q.requant_bypass));
            return 1;
        }
    }

    std::printf("blob_file_tb passed: bytes=%zu, uops=%u, scales=%u, conv=%u, affine=%u, add=%u, pools=%u\n",
                bytes.size(),
                static_cast<unsigned>(header.uop_count),
                static_cast<unsigned>(header.scale_desc_count),
                static_cast<unsigned>(header.conv_desc_count),
                static_cast<unsigned>(header.affine_desc_count),
                static_cast<unsigned>(header.add_desc_count),
                static_cast<unsigned>(header.pool_desc_count));
    return 0;
}
