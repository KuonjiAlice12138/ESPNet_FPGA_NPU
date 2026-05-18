#include <string.h>

#include "./app_config.h"
#include "drivers/sd_card.h"
#include "hal/int8_npu.h"
#include "xil_cache.h"
#include "xil_printf.h"
#include "xstatus.h"

static inline u64 read_cycle_counter(void)
{
    u64 val;
    asm volatile("mrs %0, CNTPCT_EL0" : "=r"(val));
    return val;
}

static u8 g_input[INT8_INPUT_BYTES] __attribute__((aligned(64)));
static u8 g_output[INT8_OUTPUT_BYTES] __attribute__((aligned(64)));
static u8 g_param[INT8_PARAM_HW_MAX_BYTES] __attribute__((aligned(64)));

typedef struct {
    u32 stop_after_uop;
    u32 dump_tensor_id;
    const char *tag;
    const char *out_file;
} DebugCase;

typedef struct {
    const char *tag;
    u32 stop_after;
    u64 cycles;
    u64 start_lo;
    u64 end_lo;
} PerfEntry;

static PerfEntry s_perf[32];
static u32 s_perf_count = 0U;

static void record_perf_start(const char *tag, u32 stop_after)
{
    if (s_perf_count >= 32U) return;
    s_perf[s_perf_count].tag = tag;
    s_perf[s_perf_count].stop_after = stop_after;
    s_perf[s_perf_count].cycles = 0ULL;
    s_perf[s_perf_count].start_lo = read_cycle_counter();
    s_perf_count++;
}

static void record_perf_end(void)
{
    if (s_perf_count == 0U) return;
    u32 idx = s_perf_count - 1U;
    u64 tEnd = read_cycle_counter();
    s_perf[idx].end_lo = tEnd;
    s_perf[idx].cycles = (s_perf[idx].end_lo >= s_perf[idx].start_lo)
        ? (s_perf[idx].end_lo - s_perf[idx].start_lo) : 0ULL;
}

static void print_perf_summary(void)
{
    u32 i;
    u64 total = 0ULL;
    xil_printf("\r\nAPP: ==== PERFORMANCE ====\r\n");
    xil_printf("APP: APU freq ~100MHz, 1 cycle ~10ns\r\n\r\n");
    xil_printf("APP: %-18s %6s %12s %10s\r\n", "Tag", "UOPs", "Cycles", "ms");
    for (i = 0U; i < s_perf_count; ++i) {
        if (s_perf[i].cycles == 0ULL) continue;
        u64 ms = s_perf[i].cycles / 100000ULL;
        total += s_perf[i].cycles;
        xil_printf("APP: %-18s %6u %12llu %10llu\r\n",
                   s_perf[i].tag, s_perf[i].stop_after,
                   s_perf[i].cycles, ms);
    }
    xil_printf("APP: --------------------------------\r\n");
    xil_printf("APP: FULL TOTAL %18llu cycles = %llu ms\r\n",
               total, total / 100000ULL);
}

static const DebugCase kDebugCases[] = {
    {0U, INT8_TID_INPUT, "DBG_U00_INPUT", "D00OUT.BIN"},
    {1U, INT8_TID_POOL1, "DBG_U01_POOL1", "D01OUT.BIN"},
    {2U, INT8_TID_INPUT, "DBG_U02_IN", "D02IN.BIN"},
    {2U, INT8_TID_B1_CAT, "DBG_U02_C1", "D02OUT.BIN"},
    {3U, INT8_TID_B1_CAT, "DBG_U03_B1CAT", "D03OUT.BIN"},
    {4U, INT8_TID_B1_ACT, "DBG_U04_B1", "D04OUT.BIN"},
    {20U, INT8_TID_L20_ACT, "DBG_U20_L20", "D20OUT.BIN"},
    {39U, INT8_TID_B2_ACT, "DBG_U39_B2", "D39OUT.BIN"},
    {53U, INT8_TID_L30_ACT, "DBG_U53_L30", "D53OUT.BIN"},
    {69U, INT8_TID_L3B0_ACT, "DBG_U69_L3B0", "D69OUT.BIN"},
    {72U, INT8_TID_OUT, "DBG_U72_OUT", "D72OUT.BIN"},
};

static void print_addr(const char *name, UINTPTR addr, u32 bytes)
{
    u64 a = (u64)addr;
    xil_printf("%s addr=0x%08x%08x bytes=%u\r\n", name, (u32)(a >> 32),
               (u32)a, bytes);
}

static int run_debug_case(Int8NpuContext *npu, const DebugCase *debug_case,
                          u32 uop_count)
{
    u32 timeout_polls = INT8_NPU_TIMEOUT_POLLS;
    if (debug_case->stop_after_uop >= INT8_NPU_LONG_DEBUG_START_UOP) {
        timeout_polls = INT8_NPU_DEBUG_LONG_TIMEOUT_POLLS;
    }

#if !INT8_APP_QUIET_PERF
    xil_printf("APP: debug case %s stop_after=%u tensor=%u timeout=%u\r\n",
               debug_case->tag, debug_case->stop_after_uop,
               debug_case->dump_tensor_id, timeout_polls);
#else
    (void)timeout_polls;
#endif

    memset(g_output, 0, sizeof(g_output));
    Xil_DCacheFlushRange((UINTPTR)g_output, INT8_OUTPUT_BYTES);

    record_perf_start(debug_case->tag, debug_case->stop_after_uop);

    if (int8_npu_run_debug(npu, (UINTPTR)g_input, (UINTPTR)g_output,
                           (UINTPTR)g_param, uop_count,
                           debug_case->stop_after_uop,
                           debug_case->dump_tensor_id,
                           INT8_NPU_DEBUG_DUMP_WORDS,
                           timeout_polls,
                           debug_case->tag) != XST_SUCCESS) {
        xil_printf("APP: debug case failed: %s\r\n", debug_case->tag);
        return XST_FAILURE;
    }

    record_perf_end();

    Xil_DCacheInvalidateRange((UINTPTR)g_output, INT8_OUTPUT_BYTES);
    if (SD_SaveMemoryToFile(debug_case->out_file, g_output,
                            INT8_OUTPUT_BYTES) != XST_SUCCESS) {
        xil_printf("APP: save %s failed\r\n", debug_case->out_file);
        return XST_FAILURE;
    }

#if !INT8_APP_QUIET_PERF
    xil_printf("APP: compare offline with %s on host\r\n", debug_case->out_file);
    xil_printf("APP: debug case done: %s\r\n", debug_case->tag);
#endif
    return XST_SUCCESS;
}

static int run_debug_bringup(Int8NpuContext *npu, u32 uop_count)
{
    u32 i;

#if !INT8_APP_QUIET_PERF
    xil_printf("APP: debug bring-up cases=%u\r\n",
               (u32)(sizeof(kDebugCases) / sizeof(kDebugCases[0])));
#else
    (void)(sizeof(kDebugCases));
#endif
    xil_printf("APP: PERF profiling start\r\n");
    for (i = 0U; i < (u32)(sizeof(kDebugCases) / sizeof(kDebugCases[0]));
         ++i) {
        if (run_debug_case(npu, &kDebugCases[i], uop_count) != XST_SUCCESS) {
            return XST_FAILURE;
        }
    }
    xil_printf("APP: PERF profiling done\r\n");
    print_perf_summary();
    return XST_SUCCESS;
}

int main(void)
{
    Int8NpuContext npu;
    Int8ParamBlobHeader header;
    u32 param_size = 0U;
    u32 input_size = 0U;

    xil_printf("\r\nESP INT8 app %s\r\n", INT8_APP_BUILD_TAG);
    print_addr("input", (UINTPTR)g_input, INT8_INPUT_BYTES);
    print_addr("output", (UINTPTR)g_output, INT8_OUTPUT_BYTES);
    print_addr("param", (UINTPTR)g_param, INT8_PARAM_HW_MAX_BYTES);

    if (SD_Init() != XST_SUCCESS) {
        xil_printf("APP: SD init failed\r\n");
        return XST_FAILURE;
    }
    if (SD_LoadFileToMemory(INT8_PARAM_FILE, (UINTPTR)g_param,
                            INT8_PARAM_HW_MAX_BYTES,
                            &param_size) != XST_SUCCESS) {
        xil_printf("APP: load param failed\r\n");
        return XST_FAILURE;
    }
    if (int8_npu_read_param_header(g_param, param_size, &header) !=
        XST_SUCCESS) {
        xil_printf("APP: read param header failed\r\n");
        return XST_FAILURE;
    }
    if (int8_npu_validate_param_header(&header, param_size) != XST_SUCCESS) {
        return XST_FAILURE;
    }

    if (SD_LoadFileToMemory(INT8_INPUT_FILE, (UINTPTR)g_input, INT8_INPUT_BYTES,
                            &input_size) != XST_SUCCESS) {
        xil_printf("APP: load input failed\r\n");
        return XST_FAILURE;
    }
    if (input_size != INT8_INPUT_BYTES) {
        xil_printf("APP: bad input size=%u expected=%u\r\n", input_size,
                   INT8_INPUT_BYTES);
        return XST_FAILURE;
    }

    memset(g_output, 0, sizeof(g_output));

    if (int8_npu_init(&npu) != XST_SUCCESS) {
        return XST_FAILURE;
    }

    Xil_DCacheFlushRange((UINTPTR)g_param, param_size);
    Xil_DCacheFlushRange((UINTPTR)g_input, INT8_INPUT_BYTES);
    Xil_DCacheFlushRange((UINTPTR)g_output, INT8_OUTPUT_BYTES);

    if (int8_npu_run_init(&npu, (UINTPTR)g_param, header.uop_count,
                          INT8_NPU_TIMEOUT_POLLS) != XST_SUCCESS) {
        xil_printf("APP: MODE_INIT failed\r\n");
        return XST_FAILURE;
    }
    xil_printf("APP: MODE_INIT done\r\n");

#if INT8_APP_ENABLE_DEBUG_BRINGUP
    if (run_debug_bringup(&npu, header.uop_count) != XST_SUCCESS) {
        xil_printf("APP: debug bring-up failed\r\n");
        return XST_FAILURE;
    }
#if !INT8_APP_RUN_FULL_AFTER_DEBUG
    xil_printf("APP: full MODE_RUN disabled after debug bring-up\r\n");
    return XST_SUCCESS;
#endif
#endif

    xil_printf("APP: full MODE_RUN timing start\r\n");
    record_perf_start("FULL_MODE_RUN", header.uop_count);
    if (int8_npu_run_infer(&npu, (UINTPTR)g_input, (UINTPTR)g_output,
                           (UINTPTR)g_param, header.uop_count,
                           INT8_NPU_TIMEOUT_POLLS) != XST_SUCCESS) {
        xil_printf("APP: MODE_RUN failed\r\n");
        return XST_FAILURE;
    }
    record_perf_end();
    Xil_DCacheInvalidateRange((UINTPTR)g_output, INT8_OUTPUT_BYTES);
    xil_printf("APP: MODE_RUN done\r\n");
    print_perf_summary();
    int8_npu_dump_profile_regs(&npu, "MODE_RUN");

    if (SD_SaveMemoryToFile(INT8_OUTPUT_FILE, g_output, INT8_OUTPUT_BYTES) !=
        XST_SUCCESS) {
        xil_printf("APP: save output failed\r\n");
        return XST_FAILURE;
    }

    xil_printf("APP: compare offline with %s on host\r\n", INT8_OUTPUT_FILE);
    xil_printf("APP: done\r\n");
    return XST_SUCCESS;
}
