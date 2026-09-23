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

#if (INT8_APP_ENABLE_EXEC_PREFIX_PROFILE + INT8_APP_ENABLE_VAL_SET_TEST + \
     INT8_APP_ENABLE_DUAL_VAL_SET_TEST + INT8_APP_ENABLE_DUAL_SINGLE_TEST) > 1
#error "select only one prefix, validation-set, or dual-single test mode"
#endif

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

typedef struct {
    u32 prefix_index;
    u32 active_exec_count;
    u32 new_exec_index;
    Int8ExecPlanEntry new_exec;
    u64 arm_ticks;
    u64 rtl_total;
    u64 rtl_window;
    u64 stage[INT8_STAGE_COUNTER_STAGE_COUNT];
    u32 has_conv_profile;
    u64 conv_win[INT8_CONV_WIN_STATE_COUNT];
    u64 conv_sa[INT8_CONV_SA_STATE_COUNT];
    u64 conv_post[INT8_CONV_POST_STATE_COUNT];
    u32 conv_current;
    u32 current_stage;
    u32 status;
} ExecPrefixProfile;

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

static int make_exec_prefix_filename(u32 prefix_index, char *out,
                                     u32 out_size)
{
    int n;

    if (out == NULL || out_size == 0U || prefix_index > 99U) {
        return XST_FAILURE;
    }
    n = snprintf(out, out_size, "P%02u.BIN", prefix_index);
    if (n < 0 || (u32)n >= out_size) {
        return XST_FAILURE;
    }
    return XST_SUCCESS;
}

static void capture_exec_prefix_profile(ExecPrefixProfile *profile)
{
    u32 stage;
    u32 state;

    profile->rtl_total = stage_counter_read_total();
    profile->rtl_window = stage_counter_read_active();
    for (stage = 0U; stage < INT8_STAGE_COUNTER_STAGE_COUNT; ++stage) {
        profile->stage[stage] = stage_counter_read_stage(stage);
    }
    profile->has_conv_profile =
        (stage_counter_has_conv_profile() != 0) ? 1U : 0U;
    for (state = 0U; state < INT8_CONV_WIN_STATE_COUNT; ++state) {
        profile->conv_win[state] = stage_counter_read_conv_win(state);
    }
    for (state = 0U; state < INT8_CONV_SA_STATE_COUNT; ++state) {
        profile->conv_sa[state] = stage_counter_read_conv_sa(state);
    }
    for (state = 0U; state < INT8_CONV_POST_STATE_COUNT; ++state) {
        profile->conv_post[state] = stage_counter_read_conv_post(state);
    }
    profile->conv_current = stage_counter_read_conv_current();
    profile->current_stage = stage_counter_read_current();
    profile->status = stage_counter_read_status();
}

static void print_exec_prefix_metadata(const ExecPrefixProfile *profile)
{
    xil_printf("P%02u,%u,", profile->prefix_index,
               profile->active_exec_count);
    if (profile->active_exec_count == 0U) {
        xil_printf("NA,NA,NA");
    } else {
        xil_printf("%u,%u,%u", profile->new_exec_index,
                   profile->new_exec.logical_uop_id,
                   profile->new_exec.kind);
    }
}

static void print_csv_delta(u64 current, u64 previous)
{
    if (current >= previous) {
        xil_printf(",%llu", current - previous);
    } else {
        xil_printf(",-%llu", previous - current);
    }
}

static void print_exec_prefix_rows(const ExecPrefixProfile *current,
                                   const ExecPrefixProfile *previous)
{
    u32 stage;
    u32 state;

    xil_printf("PREFIX_CUM,");
    print_exec_prefix_metadata(current);
    xil_printf(",%llu,%llu,%llu", current->arm_ticks, current->rtl_total,
               current->rtl_window);
    for (stage = 0U; stage < INT8_STAGE_COUNTER_STAGE_COUNT; ++stage) {
        xil_printf(",%llu", current->stage[stage]);
    }
    xil_printf(",%u", current->has_conv_profile);
    for (state = 0U; state < INT8_CONV_WIN_STATE_COUNT; ++state) {
        xil_printf(",%llu", current->conv_win[state]);
    }
    for (state = 0U; state < INT8_CONV_SA_STATE_COUNT; ++state) {
        xil_printf(",%llu", current->conv_sa[state]);
    }
    for (state = 0U; state < INT8_CONV_POST_STATE_COUNT; ++state) {
        xil_printf(",%llu", current->conv_post[state]);
    }
    xil_printf(",0x%08x,0x%08x,0x%08x\r\n", current->conv_current,
               current->current_stage, current->status);

    xil_printf("PREFIX_DELTA,");
    print_exec_prefix_metadata(current);
    print_csv_delta(current->arm_ticks, previous->arm_ticks);
    print_csv_delta(current->rtl_total, previous->rtl_total);
    print_csv_delta(current->rtl_window, previous->rtl_window);
    for (stage = 0U; stage < INT8_STAGE_COUNTER_STAGE_COUNT; ++stage) {
        print_csv_delta(current->stage[stage], previous->stage[stage]);
    }
    xil_printf(",%u", current->has_conv_profile);
    for (state = 0U; state < INT8_CONV_WIN_STATE_COUNT; ++state) {
        print_csv_delta(current->conv_win[state], previous->conv_win[state]);
    }
    for (state = 0U; state < INT8_CONV_SA_STATE_COUNT; ++state) {
        print_csv_delta(current->conv_sa[state], previous->conv_sa[state]);
    }
    for (state = 0U; state < INT8_CONV_POST_STATE_COUNT; ++state) {
        print_csv_delta(current->conv_post[state], previous->conv_post[state]);
    }
    xil_printf(",0x%08x,0x%08x,0x%08x\r\n", current->conv_current,
               current->current_stage, current->status);
}

static int run_infer_once(Int8NpuContext *npu, u32 uop_count,
                          u64 *cycles, u32 dump_stage_csv)
{
    u64 start;
    u64 end;

    memset(g_output, 0, sizeof(g_output));
    Xil_DCacheFlushRange((UINTPTR)g_input, INT8_INPUT_BYTES);
    Xil_DCacheFlushRange((UINTPTR)g_output, INT8_OUTPUT_BYTES);

    stage_counter_clear();
    stage_counter_freeze(0U);
    stage_counter_enable(1U);

    start = read_timer_counter();
    if (int8_npu_run_infer(npu, (UINTPTR)g_input, (UINTPTR)g_output,
                           (UINTPTR)g_param, uop_count,
                           INT8_NPU_TIMEOUT_POLLS) != XST_SUCCESS) {
        stage_counter_freeze(1U);
        stage_counter_enable(0U);
        if (dump_stage_csv != 0U) {
            stage_counter_dump_csv();
        }
        return XST_FAILURE;
    }
    end = read_timer_counter();
    stage_counter_freeze(1U);
    stage_counter_enable(0U);

    Xil_DCacheInvalidateRange((UINTPTR)g_output, INT8_OUTPUT_BYTES);
    if (cycles != NULL) {
        *cycles = (end >= start) ? (end - start) : 0ULL;
    }
    if (dump_stage_csv != 0U) {
        stage_counter_dump_csv();
    }
    return XST_SUCCESS;
}

static int run_full_infer_once(Int8NpuContext *npu, u32 uop_count,
                               u64 *cycles, u32 dump_stage_csv)
{
    return run_infer_once(npu, uop_count, cycles, dump_stage_csv);
}

static int validate_output_classes(const char *profile_name, u32 expected_classes)
{
    u32 class_counts[INT8_APP_CLASS20_CLASS_COUNT];
    u32 invalid_count = 0U;
    u32 i;

    if (expected_classes == 0U ||
        expected_classes > INT8_APP_CLASS20_CLASS_COUNT) {
        xil_printf("APP: unsupported class count=%u\r\n", expected_classes);
        return XST_FAILURE;
    }
    memset(class_counts, 0, sizeof(class_counts));
    for (i = 0U; i < INT8_OUTPUT_BYTES; ++i) {
        const u32 class_id = (u32)g_output[i];
        if (class_id < expected_classes) {
            ++class_counts[class_id];
        } else {
            ++invalid_count;
        }
    }

    xil_printf("APP: output profile=%s classes=%u invalid=%u\r\n",
               profile_name, expected_classes,
               invalid_count);
    xil_printf("APP: output class histogram");
    for (i = 0U; i < expected_classes; ++i) {
        xil_printf(" c%u=%u", i, class_counts[i]);
    }
    xil_printf("\r\n");
    return (invalid_count == 0U) ? XST_SUCCESS : XST_FAILURE;
}

static int run_exec_prefix_profile(Int8NpuContext *npu)
{
    ExecPrefixProfile current;
    ExecPrefixProfile previous;
    Int8ParamBlobHeader header;
    Int8ExecPlanEntry entry;
    char param_file[16];
    u32 expected_param_size = 0U;
    u32 input_size = 0U;
    u32 param_size;
    u32 prefix_index;
    u32 performance_gate_failed = 0U;
    const u32 conv_profile_cap =
        (stage_counter_has_conv_profile() != 0) ? 1U : 0U;

    memset(&previous, 0, sizeof(previous));
    xil_printf("APP: conv profile capability=%u required=%u\r\n",
               conv_profile_cap, INT8_APP_REQUIRE_CONV_PROFILE);
    if (INT8_APP_REQUIRE_CONV_PROFILE && conv_profile_cap == 0U) {
        xil_printf("APP: incompatible stage counter; abort prefix profiling\r\n");
        return XST_FAILURE;
    }
    if (SD_LoadFileToMemory(INT8_APP_SINGLE_PERF_INPUT_FILE,
                            (UINTPTR)g_input, INT8_INPUT_BYTES,
                            &input_size) != XST_SUCCESS ||
        input_size != INT8_INPUT_BYTES) {
        xil_printf("APP: prefix input load failed, size=%u expected=%u\r\n",
                   input_size, INT8_INPUT_BYTES);
        return XST_FAILURE;
    }

    xil_printf("APP: exec-prefix profiling files=P00.BIN..P%02u.BIN\r\n",
               INT8_APP_EXEC_PREFIX_COUNT - 1U);
    xil_printf("EXEC_PREFIX_CYCLES_BEGIN\r\n");
    xil_printf("record,prefix,active_execs,new_exec_index,logical_uop,kind,"
               "arm_ticks,rtl_total,rtl_window,"
               "s0_idle,s1_param,s2_frame,s3_ctrl,s4_wgt,s5_conv,s6_ppu,"
               "s7_b5,s8_vec,s9_pool,s10_up,s11_store,s12_error,"
               "conv_cap,cw0_idle_done,cw1_active,"
               "cs0_idle_done,cs1_compute_wait_act,cs2_emit_wait,"
               "cp0_idle_done,cp1_requant_wait,cp2_write,"
               "conv_current,current,status\r\n");

    for (prefix_index = 0U;
         prefix_index < INT8_APP_EXEC_PREFIX_COUNT;
         ++prefix_index) {
        u32 i;

        if (make_exec_prefix_filename(prefix_index, param_file,
                                      sizeof(param_file)) != XST_SUCCESS) {
            xil_printf("APP: bad prefix filename index=%u\r\n",
                       prefix_index);
            return XST_FAILURE;
        }
        param_size = 0U;
        if (SD_LoadFileToMemory(param_file, (UINTPTR)g_param,
                                INT8_PARAM_HW_MAX_BYTES,
                                &param_size) != XST_SUCCESS) {
            xil_printf("APP: prefix param load failed: %s\r\n", param_file);
            return XST_FAILURE;
        }
        if (expected_param_size == 0U) {
            expected_param_size = param_size;
        } else if (param_size != expected_param_size) {
            xil_printf("APP: prefix param size mismatch: %s size=%u expected=%u\r\n",
                       param_file, param_size, expected_param_size);
            return XST_FAILURE;
        }
        if (int8_npu_read_param_header(g_param, param_size, &header) !=
                XST_SUCCESS ||
            int8_npu_validate_param_header(&header, param_size) !=
                XST_SUCCESS) {
            xil_printf("APP: prefix param validation failed: %s\r\n",
                       param_file);
            return XST_FAILURE;
        }
        if (header.exec_plan_count != INT8_APP_EXEC_PREFIX_COUNT) {
            xil_printf("APP: prefix exec count=%u expected=%u file=%s\r\n",
                       header.exec_plan_count, INT8_APP_EXEC_PREFIX_COUNT,
                       param_file);
            return XST_FAILURE;
        }

        for (i = 0U; i < prefix_index; ++i) {
            if (int8_npu_read_exec_plan_entry(g_param, param_size, &header,
                                              i, &entry) != XST_SUCCESS ||
                entry.kind == INT8_EXEC_KIND_END) {
                xil_printf("APP: early END before pc=%u in %s\r\n",
                           prefix_index, param_file);
                return XST_FAILURE;
            }
        }
        if (int8_npu_read_exec_plan_entry(g_param, param_size, &header,
                                          prefix_index, &entry) !=
                XST_SUCCESS ||
            entry.kind != INT8_EXEC_KIND_END) {
            xil_printf("APP: missing END at pc=%u in %s\r\n",
                       prefix_index, param_file);
            return XST_FAILURE;
        }

        memset(&current, 0, sizeof(current));
        current.prefix_index = prefix_index;
        current.active_exec_count = prefix_index;
        if (prefix_index > 0U) {
            current.new_exec_index = prefix_index - 1U;
            if (int8_npu_read_exec_plan_entry(
                    g_param, param_size, &header, current.new_exec_index,
                    &current.new_exec) != XST_SUCCESS ||
                current.new_exec.kind == INT8_EXEC_KIND_END) {
                xil_printf("APP: invalid new exec at pc=%u in %s\r\n",
                           current.new_exec_index, param_file);
                return XST_FAILURE;
            }
        }

        Xil_DCacheFlushRange((UINTPTR)g_param, param_size);
        xil_printf("APP: prefix P%02u active=%u init/run\r\n",
                   prefix_index, prefix_index);
        if (int8_npu_run_init(npu, (UINTPTR)g_param, header.uop_count,
                              INT8_NPU_TIMEOUT_POLLS) != XST_SUCCESS) {
            xil_printf("APP: prefix MODE_INIT failed: %s\r\n", param_file);
            return XST_FAILURE;
        }
        if (run_infer_once(npu, header.uop_count, &current.arm_ticks, 0U) !=
            XST_SUCCESS) {
            xil_printf("APP: prefix MODE_RUN failed: %s\r\n", param_file);
            return XST_FAILURE;
        }
        capture_exec_prefix_profile(&current);
        print_exec_prefix_rows(&current, &previous);
        if (current.stage[12] != 0ULL || current.current_stage != 0U) {
            xil_printf("APP: prefix terminal state failed: %s error=%llu current=%u\r\n",
                       param_file, current.stage[12],
                       current.current_stage);
            return XST_FAILURE;
        }
        previous = current;
    }
    xil_printf("EXEC_PREFIX_CYCLES_END\r\n");

    {
        const u64 reference =
            INT8_APP_EXEC_PREFIX_REFERENCE_FULL_RTL_CYCLES;
        const u64 target =
            INT8_APP_EXEC_PREFIX_TARGET_MAX_RTL_CYCLES;
        if (reference == 0ULL) {
            xil_printf("APP: prefix baseline cycles=%llu reference=UNSET\r\n",
                       previous.rtl_total);
        } else if (previous.rtl_total <= reference) {
            const u64 saved = reference - previous.rtl_total;
            const u64 improvement_ppm =
                (reference > 0ULL) ?
                    ((saved * 1000000ULL) / reference) : 0ULL;
            xil_printf("APP: prefix baseline cycles=%llu reference=%llu "
                       "saved=%llu improvement_ppm=%llu\r\n",
                       previous.rtl_total, reference, saved,
                       improvement_ppm);
        } else {
            const u64 regression = previous.rtl_total - reference;
            const u64 regression_ppm =
                (reference > 0ULL) ?
                    ((regression * 1000000ULL) / reference) : 0ULL;
            xil_printf("APP: prefix baseline cycles=%llu reference=%llu "
                       "regression=%llu regression_ppm=%llu\r\n",
                       previous.rtl_total, reference, regression,
                       regression_ppm);
        }
        if (target == 0ULL) {
            xil_printf("APP: prefix target cycles=%llu max=UNSET INFO\r\n",
                       previous.rtl_total);
        } else {
            xil_printf("APP: prefix target cycles=%llu max=%llu %s\r\n",
                       previous.rtl_total, target,
                       (previous.rtl_total <= target) ? "PASS" : "FAIL");
        }
        if (target != 0ULL && previous.rtl_total > target) {
            performance_gate_failed = 1U;
        }
    }

    if (validate_output_classes(INT8_APP_DEPLOYMENT_NAME,
                                INT8_APP_EXPECTED_CLASS_COUNT) != XST_SUCCESS) {
        xil_printf("APP: prefix final output class validation failed\r\n");
        return XST_FAILURE;
    }

    if (SD_SaveMemoryToFile(INT8_APP_SINGLE_PERF_OUTPUT_FILE, g_output,
                            INT8_OUTPUT_BYTES) != XST_SUCCESS) {
        xil_printf("APP: prefix final output save failed\r\n");
        return XST_FAILURE;
    }
    xil_printf("APP: prefix profiling done, final output=%s\r\n",
               INT8_APP_SINGLE_PERF_OUTPUT_FILE);
    return (performance_gate_failed == 0U) ? XST_SUCCESS : XST_FAILURE;
}

static int run_single_image(Int8NpuContext *npu, u32 uop_count,
                            const char *profile_name, u32 expected_classes,
                            const char *input_file, const char *output_file,
                            u64 *arm_ticks)
{
    u32 input_size = 0U;
    u64 cycles = 0ULL;
    const u32 conv_profile_cap =
        (stage_counter_has_conv_profile() != 0) ? 1U : 0U;

    s_perf_count = 0U;
    xil_printf("\r\nAPP: SINGLE_PROFILE_BEGIN profile=%s classes=%u\r\n",
               profile_name, expected_classes);
    xil_printf("APP: conv profile capability=%u required=%u\r\n",
               conv_profile_cap, INT8_APP_REQUIRE_CONV_PROFILE);
    if (INT8_APP_REQUIRE_CONV_PROFILE && conv_profile_cap == 0U) {
        xil_printf("APP: incompatible stage counter; abort single profiling\r\n");
        return XST_FAILURE;
    }
    if (conv_profile_cap == 0U) {
        xil_printf("APP: stage-only profiling on restored platform\r\n");
    }

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

    xil_printf("APP: single full MODE_RUN timing start profile=%s\r\n",
               profile_name);
    if (run_full_infer_once(npu, uop_count, &cycles, 1U) != XST_SUCCESS) {
        xil_printf("APP: MODE_RUN failed\r\n");
        return XST_FAILURE;
    }
    record_perf_value("FULL_MODE_RUN", uop_count, cycles);
    if (arm_ticks != NULL) {
        *arm_ticks = cycles;
    }

    xil_printf("APP: single MODE_RUN done\r\n");
    print_perf_summary();

    if (SD_SaveMemoryToFile(output_file, g_output,
                            INT8_OUTPUT_BYTES) != XST_SUCCESS) {
        xil_printf("APP: save output failed\r\n");
        return XST_FAILURE;
    }

    if (validate_output_classes(profile_name, expected_classes) != XST_SUCCESS) {
        xil_printf("APP: invalid class IDs; saved mask retained for host analysis\r\n");
        return XST_FAILURE;
    }
    if (stage_counter_read_stage(12U) != 0ULL ||
        stage_counter_read_current() == 12U) {
        xil_printf("APP: NPU ERROR stage; profile=%s saved=%s, timing rejected\r\n",
                   profile_name, output_file);
        return XST_FAILURE;
    }
    xil_printf("APP: compare offline with %s on host\r\n",
               output_file);
    xil_printf("APP: SINGLE_PROFILE_END profile=%s output=%s\r\n",
               profile_name, output_file);
    return XST_SUCCESS;
}

static int run_val_set(Int8NpuContext *npu, u32 uop_count,
                       const char *profile_name, const char *input_prefix,
                       const char *output_prefix, u32 max_images,
                       u32 start_index, u32 expected_classes)
{
    char input_file[16];
    char output_file[16];
    u32 offset;
    u32 processed = 0U;
    u64 total_cycles = 0ULL;
    u64 min_cycles = ~0ULL;
    u64 max_cycles = 0ULL;

    xil_printf("APP: VALSET start profile=%s classes=%u max=%u start=%u\r\n",
               profile_name, expected_classes, max_images, start_index);
    xil_printf("APP: VALSET input=%s%%04u.BIN output=%s%%04u.BIN\r\n",
               input_prefix, output_prefix);

    for (offset = 0U; offset < max_images; ++offset) {
        u32 index = start_index + offset;
        u32 input_size = 0U;
        u64 cycles = 0ULL;

        if (make_val_filename(input_prefix, index, input_file,
                              sizeof(input_file)) != XST_SUCCESS ||
            make_val_filename(output_prefix, index, output_file,
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

        if (run_full_infer_once(npu, uop_count, &cycles,
                                (processed == 0U) ? 1U : 0U) != XST_SUCCESS) {
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

    xil_printf("\r\nAPP: ==== VALSET SUMMARY %s ====\r\n", profile_name);
    xil_printf("APP: profile=%s samples=%u total_cycles=%llu total_ms=%llu\r\n",
               profile_name,
               processed, total_cycles, timer_ticks_to_ms(total_cycles));
    xil_printf("APP: avg_cycles=%llu avg_ms=%llu min_ms=%llu max_ms=%llu\r\n",
               total_cycles / processed, timer_ticks_to_ms(total_cycles / processed),
               timer_ticks_to_ms(min_cycles), timer_ticks_to_ms(max_cycles));
    xil_printf("APP: run host eval_val_hw_masks_fullres on %sxxxx.BIN files\r\n",
               output_prefix);
    return XST_SUCCESS;
}

static int load_param_and_init(Int8NpuContext *npu, const char *param_file,
                               Int8ParamBlobHeader *header)
{
    u32 param_size = 0U;

    memset(g_param, 0, sizeof(g_param));
    if (SD_LoadFileToMemory(param_file, (UINTPTR)g_param,
                            INT8_PARAM_HW_MAX_BYTES,
                            &param_size) != XST_SUCCESS) {
        xil_printf("APP: load param failed: %s\r\n", param_file);
        return XST_FAILURE;
    }
    if (int8_npu_read_param_header(g_param, param_size, header) !=
            XST_SUCCESS ||
        int8_npu_validate_param_header(header, param_size) != XST_SUCCESS) {
        xil_printf("APP: param validation failed: %s\r\n", param_file);
        return XST_FAILURE;
    }

#if INT8_APP_ENABLE_DUAL_SINGLE_TEST
    /* Do not accidentally time an old ABI or an early-END prefix blob. */
    if (header->version != INT8_PARAM_BLOB_VERSION_CURRENT ||
        header->exec_plan_count != INT8_EXPECTED_EXEC_PLAN_COUNT) {
        xil_printf("APP: dual single requires full PARAM v4: %s\r\n", param_file);
        return XST_FAILURE;
    }
    for (u32 i = 0U; i < header->exec_plan_count; ++i) {
        Int8ExecPlanEntry entry;
        if (int8_npu_read_exec_plan_entry(g_param, param_size, header, i,
                                         &entry) != XST_SUCCESS ||
            ((entry.kind == INT8_EXEC_KIND_END) !=
             (i + 1U == header->exec_plan_count))) {
            xil_printf("APP: incomplete exec plan: %s index=%u\r\n",
                       param_file, i);
            return XST_FAILURE;
        }
    }
#endif

    Xil_DCacheFlushRange((UINTPTR)g_param, param_size);
    Xil_DCacheFlushRange((UINTPTR)g_input, INT8_INPUT_BYTES);
    Xil_DCacheFlushRange((UINTPTR)g_output, INT8_OUTPUT_BYTES);
    if (int8_npu_run_init(npu, (UINTPTR)g_param, header->uop_count,
                          INT8_NPU_TIMEOUT_POLLS) != XST_SUCCESS) {
        xil_printf("APP: MODE_INIT failed: %s\r\n", param_file);
        return XST_FAILURE;
    }
    xil_printf("APP: MODE_INIT done profile_param=%s uops=%u version=%u exec=%u\r\n",
               param_file, header->uop_count, header->version,
               header->exec_plan_count);
    return XST_SUCCESS;
}

static int run_dual_single_image(Int8NpuContext *npu)
{
    Int8ParamBlobHeader header;
    u64 binary_ticks = 0ULL;
    u64 class20_ticks = 0ULL;
    u64 binary_rtl = 0ULL;
    u64 class20_rtl = 0ULL;
    int binary_status = XST_FAILURE;
    int class20_status = XST_FAILURE;

    if (load_param_and_init(npu, INT8_APP_BINARY_PARAM_FILE, &header) ==
        XST_SUCCESS) {
        binary_status = run_single_image(
            npu, header.uop_count, "binary2", INT8_APP_BINARY_CLASS_COUNT,
            INT8_APP_BINARY_SINGLE_INPUT_FILE,
            INT8_APP_BINARY_SINGLE_OUTPUT_FILE, &binary_ticks);
        binary_rtl = stage_counter_read_total();
    } else {
        xil_printf("APP: binary2 init failed; continue with cityscapes20\r\n");
    }

    if (load_param_and_init(npu, INT8_APP_CLASS20_PARAM_FILE, &header) ==
        XST_SUCCESS) {
        class20_status = run_single_image(
            npu, header.uop_count, "cityscapes20",
            INT8_APP_CLASS20_CLASS_COUNT,
            INT8_APP_CLASS20_SINGLE_INPUT_FILE,
            INT8_APP_CLASS20_SINGLE_OUTPUT_FILE, &class20_ticks);
        class20_rtl = stage_counter_read_total();
    } else {
        xil_printf("APP: cityscapes20 init failed\r\n");
    }

    xil_printf("DUAL_SINGLE_SUMMARY_BEGIN\r\n");
    xil_printf("profile,classes,result,arm_ticks,arm_ms,rtl_cycles,output\r\n");
    xil_printf("binary2,%u,%s,%llu,%llu,%llu,%s\r\n",
               INT8_APP_BINARY_CLASS_COUNT,
               (binary_status == XST_SUCCESS) ? "PASS" : "FAIL", binary_ticks,
               timer_ticks_to_ms(binary_ticks), binary_rtl,
               INT8_APP_BINARY_SINGLE_OUTPUT_FILE);
    xil_printf("cityscapes20,%u,%s,%llu,%llu,%llu,%s\r\n",
               INT8_APP_CLASS20_CLASS_COUNT,
               (class20_status == XST_SUCCESS) ? "PASS" : "FAIL",
               class20_ticks,
               timer_ticks_to_ms(class20_ticks), class20_rtl,
               INT8_APP_CLASS20_SINGLE_OUTPUT_FILE);
    xil_printf("DUAL_SINGLE_SUMMARY_END\r\n");
    xil_printf("APP: dual single done binary2=%s cityscapes20=%s\r\n",
               (binary_status == XST_SUCCESS) ? "PASS" : "FAIL",
               (class20_status == XST_SUCCESS) ? "PASS" : "FAIL");
    return (binary_status == XST_SUCCESS && class20_status == XST_SUCCESS) ?
               XST_SUCCESS : XST_FAILURE;
}

static int run_dual_val_set(Int8NpuContext *npu)
{
    Int8ParamBlobHeader header;

    if (load_param_and_init(npu, INT8_APP_BINARY_PARAM_FILE, &header) !=
        XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (run_val_set(npu, header.uop_count, "binary2",
                    INT8_APP_BINARY_INPUT_PREFIX,
                    INT8_APP_BINARY_OUTPUT_PREFIX,
                    INT8_APP_DUAL_VAL_MAX_IMAGES, 0U,
                    INT8_APP_BINARY_CLASS_COUNT) != XST_SUCCESS) {
        return XST_FAILURE;
    }

    if (load_param_and_init(npu, INT8_APP_CLASS20_PARAM_FILE, &header) !=
        XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (run_val_set(npu, header.uop_count, "cityscapes20",
                    INT8_APP_CLASS20_INPUT_PREFIX,
                    INT8_APP_CLASS20_OUTPUT_PREFIX,
                    INT8_APP_DUAL_VAL_MAX_IMAGES, 0U,
                    INT8_APP_CLASS20_CLASS_COUNT) != XST_SUCCESS) {
        return XST_FAILURE;
    }

    xil_printf("APP: dual VALSET done binary2=%u cityscapes20=%u\r\n",
               INT8_APP_DUAL_VAL_MAX_IMAGES, INT8_APP_DUAL_VAL_MAX_IMAGES);
    return XST_SUCCESS;
}

int main(void)
{
    Int8NpuContext npu;
#if !INT8_APP_ENABLE_EXEC_PREFIX_PROFILE && !INT8_APP_ENABLE_DUAL_VAL_SET_TEST && \
    !INT8_APP_ENABLE_DUAL_SINGLE_TEST
    Int8ParamBlobHeader header;
    u32 param_size = 0U;
#endif

    xil_printf("\r\nESP INT8 app %s\r\n", INT8_APP_BUILD_TAG);
    print_addr("input", (UINTPTR)g_input, INT8_INPUT_BYTES);
    print_addr("output", (UINTPTR)g_output, INT8_OUTPUT_BYTES);
    print_addr("param", (UINTPTR)g_param, INT8_PARAM_HW_MAX_BYTES);
    print_timer_selftest();

    if (SD_Init() != XST_SUCCESS) {
        xil_printf("APP: SD init failed\r\n");
        return XST_FAILURE;
    }

    if (int8_npu_init(&npu) != XST_SUCCESS) {
        return XST_FAILURE;
    }

#if INT8_APP_ENABLE_EXEC_PREFIX_PROFILE
    return run_exec_prefix_profile(&npu);
#elif INT8_APP_ENABLE_DUAL_VAL_SET_TEST
    return run_dual_val_set(&npu);
#elif INT8_APP_ENABLE_DUAL_SINGLE_TEST
    return run_dual_single_image(&npu);
#else
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
    if (run_single_image(&npu, header.uop_count, INT8_APP_DEPLOYMENT_NAME,
                          INT8_APP_EXPECTED_CLASS_COUNT,
                          INT8_APP_SINGLE_PERF_INPUT_FILE,
                          INT8_APP_SINGLE_PERF_OUTPUT_FILE, NULL) != XST_SUCCESS) {
        xil_printf("APP: single perf before valset failed\r\n");
        return XST_FAILURE;
    }
#endif
    return run_val_set(&npu, header.uop_count, "single", INT8_APP_VAL_INPUT_PREFIX,
                       INT8_APP_VAL_OUTPUT_PREFIX, INT8_APP_VAL_MAX_IMAGES,
                       INT8_APP_VAL_START_INDEX,
                       INT8_APP_EXPECTED_CLASS_COUNT);
#else
    return run_single_image(&npu, header.uop_count, INT8_APP_DEPLOYMENT_NAME,
                             INT8_APP_EXPECTED_CLASS_COUNT,
                             INT8_APP_SINGLE_PERF_INPUT_FILE,
                             INT8_APP_SINGLE_PERF_OUTPUT_FILE, NULL);
#endif
#endif
}
