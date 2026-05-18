#include "../include/npu_types.hpp"

namespace esp_int8 {

void perf_irq_update(u32_t mode, u32_t uop_count) {
    (void)mode;
    (void)uop_count;
    // The current HLS top exposes completion through ap_ctrl_hs done/idle.
    // Detailed perf/IRQ registers are reserved for the AXI-Lite wrapper stage.
}

}  // namespace esp_int8
