#include <stdio.h>
#include <string.h>

#include "./app_config.h"
#include "drivers/sd_card.h"
#include "hal/int8_npu.h"
#include "sleep.h"
#include "xil_cache.h"
#include "xil_printf.h"
#include "xparameters.h"
#include "xstatus.h"

static inline u64 read_timer_counter(void)
{
    u64 val;
    asm volatile("mrs %0, CNTPCT_EL0" : "=r"(val));
    return val;
}

static inline u64 read_timer_frequency(void)
{
    u64 val;
    asm volatile("mrs %0, CNTFRQ_EL0" : "=r"(val));
    return val;
}

static u64 timer_ticks_to_ms(u64 ticks)
{
    u64 hz = read_timer_frequency();
    if (hz == 0ULL) {
        return 0ULL;
    }
    return (ticks * 1000ULL + (hz / 2ULL)) / hz;
}

static void print_timer_selftest(void)
{
    u64 cntfrq_hz = read_timer_frequency();
#ifdef XPAR_CPU_TIMESTAMP_CLK_FREQ
    u64 bsp_hz = (u64)XPAR_CPU_TIMESTAMP_CLK_FREQ;
#else
    u64 bsp_hz = 0ULL;
#endif
    u64 start;
    u64 end;
    u64 ticks;
    u64 expected;
    u64 err;
    u64 err_ppm;

    start = read_timer_counter();
    usleep((unsigned long)INT8_APP_TIMER_SELFTEST_US);
    end = read_timer_counter();

    ticks = (end >= start) ? (end - start) : 0ULL;
    expected = (cntfrq_hz * (u64)INT8_APP_TIMER_SELFTEST_US + 500000ULL) /
               1000000ULL;
    err = (ticks >= expected) ? (ticks - expected) : (expected - ticks);
    err_ppm = (expected > 0ULL) ? ((err * 1000000ULL) / expected) : 0ULL;

    xil_printf("APP: timer selftest wait_us=%u cntfrq=%llu Hz bsp=%llu Hz\r\n",
               INT8_APP_TIMER_SELFTEST_US, cntfrq_hz, bsp_hz);
    xil_printf("APP: timer selftest ticks=%llu expected=%llu err=%llu ppm %s\r\n",
               ticks, expected, err_ppm,
               (err_ppm <= (u64)INT8_APP_TIMER_SELFTEST_MAX_ERR_PPM) ?
                   "PASS" : "WARN");
}

static u8 g_input[INT8_INPUT_BYTES] __attribute__((aligned(64)));
static u8 g_output[INT8_OUTPUT_BYTES] __attribute__((aligned(64)));
static u8 g_param[INT8_PARAM_HW_MAX_BYTES] __attribute__((aligned(64)));

typedef struct {
    const char *tag;
    u32 stop_after;
    u64 cycles;
    u64 start_lo;
    u64 end_lo;
} PerfEntry;

static PerfEntry s_perf[32];
static u32 s_perf_count = 0U;

static void record_perf_value(const char *tag, u32 stop_after, u64 ticks)
{
    if (s_perf_count >= 32U) return;
    s_perf[s_perf_count].tag = tag;
    s_perf[s_perf_count].stop_after = stop_after;
    s_perf[s_perf_count].cycles = ticks;
    s_perf[s_perf_count].start_lo = 0ULL;
    s_perf[s_perf_count].end_lo = ticks;
    s_perf_count++;
}

static void print_perf_summary(void)
{
    u32 i;
    u64 total = 0ULL;
    u64 timer_hz = read_timer_frequency();
    xil_printf("\r\nAPP: ==== PERFORMANCE ====\r\n");
    xil_printf("APP: ARM timer=%llu Hz (CNTPCT_EL0), PL target=%llu Hz (%uns)\r\n\r\n",
               timer_hz, (u64)INT8_APP_PL_TARGET_HZ,
               INT8_APP_PL_TARGET_PERIOD_NS);
    xil_printf("APP: %-18s %6s %12s %10s\r\n", "Tag", "UOPs", "Ticks", "ms");
    for (i = 0U; i < s_perf_count; ++i) {
        if (s_perf[i].cycles == 0ULL) continue;
        u64 ms = timer_ticks_to_ms(s_perf[i].cycles);
        total += s_perf[i].cycles;
        xil_printf("APP: %-18s %6u %12llu %10llu\r\n",
                   s_perf[i].tag, s_perf[i].stop_after,
                   s_perf[i].cycles, ms);
    }
    xil_printf("APP: --------------------------------\r\n");
    xil_printf("APP: FULL TOTAL %18llu ticks = %llu ms\r\n",
               total, timer_ticks_to_ms(total));
}

static void print_addr(const char *name, UINTPTR addr, u32 bytes)
{
    u64 a = (u64)addr;
    xil_printf("%s addr=0x%08x%08x bytes=%u\r\n", name, (u32)(a >> 32),
               (u32)a, bytes);
}

static int make_val_filename(const char *prefix, u32 index, char *out,
                             u32 out_size)
{
    int n;

    if (prefix == NULL || out == NULL || out_size == 0U || index > 9999U) {
        return XST_FAILURE;
    }

    n = snprintf(out, out_size, "%s%04u.BIN", prefix, index);
    if (n < 0 || (u32)n >= out_size) {
        return XST_FAILURE;
    }
    return XST_SUCCESS;
}

static int run_full_infer_once(Int8NpuContext *npu, u32 uop_count,
                               u64 *cycles)
{
    u64 start;
    u64 end;

    memset(g_output, 0, sizeof(g_output));
    Xil_DCacheFlushRange((UINTPTR)g_input, INT8_INPUT_BYTES);
    Xil_DCacheFlushRange((UINTPTR)g_output, INT8_OUTPUT_BYTES);

    start = read_timer_counter();
    if (int8_npu_run_infer(npu, (UINTPTR)g_input, (UINTPTR)g_output,
                           (UINTPTR)g_param, uop_count,
                           INT8_NPU_TIMEOUT_POLLS) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    end = read_timer_counter();

    Xil_DCacheInvalidateRange((UINTPTR)g_output, INT8_OUTPUT_BYTES);
    if (cycles != NULL) {
        *cycles = (end >= start) ? (end - start) : 0ULL;
    }
    return XST_SUCCESS;
}

static int run_single_image(Int8NpuContext *npu, u32 uop_count)
{
    u32 input_size = 0U;
    u64 cycles = 0ULL;
    const char *input_file = INT8_APP_SINGLE_PERF_INPUT_FILE;

    if (!SD_FileExists(input_file)) {
#if INT8_APP_ENABLE_VAL_SET_TEST
        static char fallback_file[16];
        if (make_val_filename(INT8_APP_VAL_INPUT_PREFIX,
                              INT8_APP_SINGLE_PERF_FALLBACK_INDEX,
                              fallback_file, sizeof(fallback_file)) !=
            XST_SUCCESS) {
            xil_printf("APP: bad single-perf fallback filename\r\n");
            return XST_FAILURE;
        }
        input_file = fallback_file;
#endif
    }

    xil_printf("APP: single perf input=%s\r\n", input_file);
    if (SD_LoadFileToMemory(input_file, (UINTPTR)g_input, INT8_INPUT_BYTES,
                            &input_size) != XST_SUCCESS) {
        xil_printf("APP: load input failed\r\n");
        return XST_FAILURE;
    }
    if (input_size != INT8_INPUT_BYTES) {
        xil_printf("APP: bad input size=%u expected=%u\r\n", input_size,
                   INT8_INPUT_BYTES);
        return XST_FAILURE;
    }

    xil_printf("APP: single full MODE_RUN timing start\r\n");
    if (run_full_infer_once(npu, uop_count, &cycles) != XST_SUCCESS) {
        xil_printf("APP: MODE_RUN failed\r\n");
        return XST_FAILURE;
    }
    record_perf_value("FULL_MODE_RUN", uop_count, cycles);

    xil_printf("APP: single MODE_RUN done\r\n");
    print_perf_summary();

    if (SD_SaveMemoryToFile(INT8_APP_SINGLE_PERF_OUTPUT_FILE, g_output,
                            INT8_OUTPUT_BYTES) != XST_SUCCESS) {
        xil_printf("APP: save output failed\r\n");
        return XST_FAILURE;
    }

    xil_printf("APP: compare offline with %s on host\r\n",
               INT8_APP_SINGLE_PERF_OUTPUT_FILE);
    xil_printf("APP: done\r\n");
    return XST_SUCCESS;
}

static int run_val_set(Int8NpuContext *npu, u32 uop_count)
{
    char input_file[16];
    char output_file[16];
    u32 offset;
    u32 processed = 0U;
    u64 total_cycles = 0ULL;
    u64 min_cycles = ~0ULL;
    u64 max_cycles = 0ULL;

    xil_printf("APP: VALSET start max=%u start=%u\r\n",
               INT8_APP_VAL_MAX_IMAGES, INT8_APP_VAL_START_INDEX);
    xil_printf("APP: VALSET input=%s%%04u.BIN output=%s%%04u.BIN\r\n",
               INT8_APP_VAL_INPUT_PREFIX, INT8_APP_VAL_OUTPUT_PREFIX);

    for (offset = 0U; offset < INT8_APP_VAL_MAX_IMAGES; ++offset) {
        u32 index = INT8_APP_VAL_START_INDEX + offset;
        u32 input_size = 0U;
        u64 cycles = 0ULL;

        if (make_val_filename(INT8_APP_VAL_INPUT_PREFIX, index, input_file,
                              sizeof(input_file)) != XST_SUCCESS ||
            make_val_filename(INT8_APP_VAL_OUTPUT_PREFIX, index, output_file,
                              sizeof(output_file)) != XST_SUCCESS) {
            xil_printf("APP: bad val filename index=%u\r\n", index);
            return XST_FAILURE;
        }

        if (!SD_FileExists(input_file)) {
            if (processed == 0U) {
                xil_printf("APP: VALSET first input missing: %s\r\n",
                           input_file);
                return XST_FAILURE;
            }
            xil_printf("APP: VALSET stop at missing input %s\r\n",
                       input_file);
            break;
        }

        if (SD_LoadFileToMemory(input_file, (UINTPTR)g_input, INT8_INPUT_BYTES,
                                &input_size) != XST_SUCCESS) {
            xil_printf("APP: VALSET load failed: %s\r\n", input_file);
            return XST_FAILURE;
        }
        if (input_size != INT8_INPUT_BYTES) {
            xil_printf("APP: VALSET bad input %s size=%u expected=%u\r\n",
                       input_file, input_size, INT8_INPUT_BYTES);
            return XST_FAILURE;
        }

        if (run_full_infer_once(npu, uop_count, &cycles) != XST_SUCCESS) {
            xil_printf("APP: VALSET MODE_RUN failed index=%u file=%s\r\n",
                       index, input_file);
            return XST_FAILURE;
        }

        if (SD_SaveMemoryToFile(output_file, g_output, INT8_OUTPUT_BYTES) !=
            XST_SUCCESS) {
            xil_printf("APP: VALSET save failed: %s\r\n", output_file);
            return XST_FAILURE;
        }

        processed++;
        total_cycles += cycles;
        if (cycles < min_cycles) min_cycles = cycles;
        if (cycles > max_cycles) max_cycles = cycles;

        if (INT8_APP_VAL_PROGRESS_EVERY > 0U &&
            (processed == 1U ||
             (processed % INT8_APP_VAL_PROGRESS_EVERY) == 0U)) {
            xil_printf("APP: VALSET idx=%u cycles=%llu ms=%llu out=%s\r\n",
                       index, cycles, timer_ticks_to_ms(cycles), output_file);
        }
    }

    if (processed == 0U) {
        xil_printf("APP: VALSET no samples processed\r\n");
        return XST_FAILURE;
    }

    xil_printf("\r\nAPP: ==== VALSET SUMMARY ====\r\n");
    xil_printf("APP: samples=%u total_cycles=%llu total_ms=%llu\r\n",
               processed, total_cycles, timer_ticks_to_ms(total_cycles));
    xil_printf("APP: avg_cycles=%llu avg_ms=%llu min_ms=%llu max_ms=%llu\r\n",
               total_cycles / processed, timer_ticks_to_ms(total_cycles / processed),
               timer_ticks_to_ms(min_cycles), timer_ticks_to_ms(max_cycles));
    xil_printf("APP: run host eval_val_hw_masks_fullres on Oxxxx.BIN files\r\n");
    return XST_SUCCESS;
}

int main(void)
{
    Int8NpuContext npu;
    Int8ParamBlobHeader header;
    u32 param_size = 0U;

    xil_printf("\r\nESP INT8 app %s\r\n", INT8_APP_BUILD_TAG);
    print_addr("input", (UINTPTR)g_input, INT8_INPUT_BYTES);
    print_addr("output", (UINTPTR)g_output, INT8_OUTPUT_BYTES);
    print_addr("param", (UINTPTR)g_param, INT8_PARAM_HW_MAX_BYTES);
    print_timer_selftest();

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

#if INT8_APP_ENABLE_VAL_SET_TEST
#if INT8_APP_ENABLE_SINGLE_PERF_BEFORE_VAL
    if (run_single_image(&npu, header.uop_count) != XST_SUCCESS) {
        xil_printf("APP: single perf before valset failed\r\n");
        return XST_FAILURE;
    }
#endif
    return run_val_set(&npu, header.uop_count);
#else
    return run_single_image(&npu, header.uop_count);
#endif
}
