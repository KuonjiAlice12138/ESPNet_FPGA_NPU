#include "../include/npu_types.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

void round4r_conv_row_handshake_probe(esp_int8::u8_t case_id,
                                      esp_int8::u8_t seed,
                                      esp_int8::u32_t& checksum);

int main() {
    const char* probe_case = std::getenv("ESP_INT8_R4R3_CASE");
    const bool run_c12 = probe_case != nullptr && std::strcmp(probe_case, "c12") == 0;
    const esp_int8::u8_t case_id = run_c12 ? esp_int8::u8_t(1) : esp_int8::u8_t(0);
    esp_int8::u32_t checksum = 0;
    round4r_conv_row_handshake_probe(case_id, esp_int8::u8_t(37), checksum);
    std::printf("[R4R3] case=%s checksum=0x%08x\n",
                run_c12 ? "c12" : "c3",
                checksum.to_uint());
    return 0;
}
