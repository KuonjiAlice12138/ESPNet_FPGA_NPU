#include "../include/npu_config.hpp"
#include "../include/npu_schedule.hpp"
#include "../include/npu_uop.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

namespace esp_int8 {
void param_dma_init(const axi_vec_t* gmem_param);
bool param_dma_ready();
bool param_dma_get_tensor_desc(u8_t tensor_id, tensor_desc_t& desc);
bool param_dma_get_conv_exec_desc(u8_t id, conv_exec_desc_t& desc);
bool param_dma_get_window_sched(u8_t id, window_sched_desc_t& desc);
bool param_dma_get_row_consumer(u8_t id, row_consumer_desc_t& desc);
bool param_dma_get_exec_entry(u8_t pc, exec_plan_entry_t& entry);
bool param_dma_get_packed_weight_vec(const conv_exec_desc_t& desc, u16_t tm, u16_t kt, wgt_vec_t& word);
}  // namespace esp_int8

static void set_byte(esp_int8::axi_vec_t& word, int lane, std::uint8_t value) {
    word.range(lane * 8 + 7, lane * 8) = value;
}

static bool expect(bool cond, const char* msg) {
    if (!cond) {
        std::printf("FAIL: %s\n", msg);
        return false;
    }
    return true;
}

static bool load_blob(const char* path, std::vector<esp_int8::axi_vec_t>& param, std::size_t& byte_count) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::printf("failed to open blob: %s\n", path);
        return false;
    }

    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)), {});
    byte_count = bytes.size();
    if (bytes.size() < esp_int8::PARAM_HEADER_BYTES) {
        std::printf("blob too small: %zu bytes\n", bytes.size());
        return false;
    }

    const std::size_t word_count = (bytes.size() + esp_int8::AXI_WORD_BYTES - 1) / esp_int8::AXI_WORD_BYTES;
    param.assign(word_count, 0);
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        set_byte(param[i / esp_int8::AXI_WORD_BYTES], static_cast<int>(i % esp_int8::AXI_WORD_BYTES), bytes[i]);
    }
    return true;
}

static bool validate_v4_schedule_contract() {
    using namespace esp_int8;
    bool ok = true;

    tensor_desc_t input_desc;
    ok &= expect(param_dma_get_tensor_desc(TID_INPUT, input_desc), "input tensor desc loaded");
    ok &= expect(input_desc.h == 240 && input_desc.w == 240 && input_desc.c == 1,
                 "input tensor shape matches PARAM v4 contract");

    conv_exec_desc_t conv0;
    ok &= expect(param_dma_get_conv_exec_desc(0, conv0), "conv exec desc 0 loaded");
    ok &= expect(conv0.param_id == 0 && conv0.qparam_id == 0, "conv0 ids");
    ok &= expect(conv0.window_sched_id == 0, "conv0 window schedule id");
    ok &= expect(conv0.src_tensor == TID_INPUT && conv0.dst_tensor == TID_B1_CAT, "conv0 tensors");
    ok &= expect(conv0.in_h == 240 && conv0.in_w == 240 && conv0.in_c == 1, "conv0 input shape");
    ok &= expect(conv0.out_c == 16 && conv0.kernel == 3 && conv0.k_tiles == 1, "conv0 geometry");
    ok &= expect(conv0.weight_words == 16, "conv0 active-OC packed weight words");

    window_sched_desc_t sched0;
    ok &= expect(param_dma_get_window_sched(conv0.window_sched_id, sched0), "conv0 schedule loaded");
    ok &= expect(sched0.mode == WIN_MODE_3X3_STAGED_C1, "first layer staged C1 schedule mode");
    ok &= expect(sched0.in_c == 1 && sched0.k_tiles == 1, "first layer schedule shape");
    ok &= expect(sched0.stride == 2 && sched0.dilation == 1 && sched0.padding == 1, "first layer window geometry");
    ok &= expect(sched0.out_w == 120 && sched0.cache_chunks == 1, "first layer staged window descriptor");
    ok &= expect(sched0.cmd_count == 0 && sched0.cmd_base == 0, "staged C1 uses a static composer, not a runtime pack program");
    ok &= expect(sched0.kt_cmd_base[0] == 0 && sched0.kt_cmd_base[1] == 0, "retired kt command range is empty");
    ok &= expect(sched0.reserved[WINDOW_LOADER_ROW_REUSE_WORD] == (2U | (8U << 2)),
                 "C1 schedule selects raw packed-row KEEP1");

    window_sched_desc_t c12_sched;
    ok &= expect(param_dma_get_window_sched(2, c12_sched), "C12 schedule loaded");
    ok &= expect(c12_sched.mode == WIN_MODE_3X3_STAGED_C12, "C12 staged schedule mode");
    ok &= expect(c12_sched.in_c == 12 && c12_sched.k_tiles == 4, "C12 schedule shape");
    ok &= expect(c12_sched.cache_chunks == 1 && c12_sched.cache_col_slots == 4, "C12 cache descriptor");
    ok &= expect(c12_sched.cmd_count == 0, "C12 uses a static composer without runtime pack commands");

    window_sched_desc_t c25_sched;
    ok &= expect(param_dma_get_window_sched(9, c25_sched), "C25 schedule loaded");
    ok &= expect(c25_sched.mode == WIN_MODE_3X3_STAGED_C25, "C25 staged schedule mode");
    ok &= expect(c25_sched.in_c == 25 && c25_sched.k_tiles == 8, "C25 schedule shape");
    ok &= expect(c25_sched.cache_chunks == 1 && c25_sched.cache_col_slots == 4, "C25 cache descriptor");
    ok &= expect(c25_sched.cmd_count == 0, "C25 uses a static composer without runtime pack commands");

    row_consumer_desc_t row3;
    ok &= expect(param_dma_get_row_consumer(3, row3), "row consumer 3 loaded");
    ok &= expect(row3.mode == ROW_CONSUMER_NONE && row3.store_dst_tensor == 255,
                 "BLOCK5 branch is scratch-owned, not an old concat store slice");
    ok &= expect(row3.store_c_offset == 0 && row3.valid_c == 12, "BLOCK5 branch has twelve valid scratch channels");

    exec_plan_entry_t entry0;
    ok &= expect(param_dma_get_exec_entry(0, entry0), "exec entry 0 loaded");
    ok &= expect(entry0.kind == EXEC_POOL && entry0.desc_id == 0 && entry0.logical_uop_id == 1,
                 "exec entry 0 is pool1");

    exec_plan_entry_t entry1;
    ok &= expect(param_dma_get_exec_entry(1, entry1), "exec entry 1 loaded");
    ok &= expect(entry1.kind == EXEC_CONV && entry1.desc_id == 0 && entry1.logical_uop_id == 2,
                 "exec entry 1 is first conv");

    wgt_vec_t packed = 0;
    ok &= expect(param_dma_get_packed_weight_vec(conv0, 0, 0, packed), "packed weight word 0 loaded");

    wgt_vec_t invalid = 0;
    ok &= expect(!param_dma_get_packed_weight_vec(conv0, conv0.out_c, 0, invalid),
                 "packed weight getter rejects padded output channel");
    ok &= expect(invalid == 0, "invalid packed weight read returns zero");

    return ok;
}

int main() {
    const char* blob_path = "D:/ESP_INT8/hw_artifacts/binary2_int8_h240w240_gray_r6_v4/PARAM.BIN";
    std::vector<esp_int8::axi_vec_t> param;
    std::size_t byte_count = 0;
    if (!load_blob(blob_path, param, byte_count)) {
        return 1;
    }

    esp_int8::param_dma_init(param.data());
    if (!esp_int8::param_dma_ready()) {
        std::printf("param_dma_init rejected PARAM v4 blob\n");
        return 1;
    }
    if (!validate_v4_schedule_contract()) {
        return 1;
    }

    std::printf("blob_v4_tb passed: bytes=%zu\n", byte_count);
    return 0;
}
