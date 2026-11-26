#include "./int8_npu.h"

#include "../app_config.h"
#include "xespnet_encoder_int8_core_hw.h"
#include "xil_io.h"
#include "xil_printf.h"
#include "xparameters.h"
#include "xstatus.h"
#include <string.h>
#include "../app_log.h"
#define xil_printf app_log_printf

#if !defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_MODEL_PROFILE_ID_DATA) || \
    !defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_SRC_MODE_DATA) || \
    !defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_MASK_MODE_DATA) || \
    !defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_RUNTIME_ABI_ID_DATA) || \
    !defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_RUNTIME_STATUS_DATA) || \
    !defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_RUNTIME_CAPABILITY_DATA)
#error "Rebuild the platform/BSP from the unified INT8 IP; legacy AXI-Lite map is unsupported."
#endif

#if defined(XPAR_NPU_STAGE_COUNTER_0_S00_AXI_BASEADDR)
#define INT8_STAGE_COUNTER_BASEADDR XPAR_NPU_STAGE_COUNTER_0_S00_AXI_BASEADDR
#define INT8_STAGE_COUNTER_PRESENT 1
#elif defined(XPAR_NPU_STAGE_COUNTER_0_S_AXI_BASEADDR)
#define INT8_STAGE_COUNTER_BASEADDR XPAR_NPU_STAGE_COUNTER_0_S_AXI_BASEADDR
#define INT8_STAGE_COUNTER_PRESENT 1
#elif defined(XPAR_NPU_STAGE_COUNTER_0_BASEADDR)
#define INT8_STAGE_COUNTER_BASEADDR XPAR_NPU_STAGE_COUNTER_0_BASEADDR
#define INT8_STAGE_COUNTER_PRESENT 1
#elif defined(XPAR_NPU_STAGE_COUNTER_AXI_0_S00_AXI_BASEADDR)
#define INT8_STAGE_COUNTER_BASEADDR XPAR_NPU_STAGE_COUNTER_AXI_0_S00_AXI_BASEADDR
#define INT8_STAGE_COUNTER_PRESENT 1
#else
#define INT8_STAGE_COUNTER_BASEADDR 0U
#define INT8_STAGE_COUNTER_PRESENT 0
#endif

#define STAGE_COUNTER_CTRL_OFFSET 0x000U
#define STAGE_COUNTER_STATUS_OFFSET 0x004U
#define STAGE_COUNTER_CURRENT_OFFSET 0x008U
#define STAGE_COUNTER_TOTAL_LO_OFFSET 0x010U
#define STAGE_COUNTER_TOTAL_HI_OFFSET 0x014U
#define STAGE_COUNTER_ACTIVE_LO_OFFSET 0x018U
#define STAGE_COUNTER_ACTIVE_HI_OFFSET 0x01CU
#define STAGE_COUNTER_CAPABILITY_OFFSET 0x020U
#define STAGE_COUNTER_CONV_CURRENT_OFFSET 0x024U
#define STAGE_COUNTER_STAGE_BASE_OFFSET 0x040U
#define STAGE_COUNTER_CONV_WIN_BASE_OFFSET 0x100U
#define STAGE_COUNTER_CONV_SA_BASE_OFFSET 0x140U
#define STAGE_COUNTER_CONV_POST_BASE_OFFSET 0x180U
#define STAGE_COUNTER_STAGE_STRIDE 0x008U

#define STAGE_COUNTER_CAP_CONV_PROFILE 0x00000001U

#define STAGE_COUNTER_CTRL_ENABLE 0x00000001U
#define STAGE_COUNTER_CTRL_CLEAR 0x00000002U
#define STAGE_COUNTER_CTRL_FREEZE 0x00000004U

static u32 stage_counter_read32(u32 offset)
{
#if INT8_STAGE_COUNTER_PRESENT
    return Xil_In32((UINTPTR)INT8_STAGE_COUNTER_BASEADDR + (UINTPTR)offset);
#else
    (void)offset;
    return 0U;
#endif
}

static void stage_counter_write32(u32 offset, u32 value)
{
#if INT8_STAGE_COUNTER_PRESENT
    Xil_Out32((UINTPTR)INT8_STAGE_COUNTER_BASEADDR + (UINTPTR)offset, value);
#else
    (void)offset;
    (void)value;
#endif
}

static u64 stage_counter_read64(u32 lo_offset)
{
    u32 lo0;
    u32 hi0;
    u32 lo1;
    u32 hi1;
    do {
        hi0 = stage_counter_read32(lo_offset + 4U);
        lo0 = stage_counter_read32(lo_offset);
        hi1 = stage_counter_read32(lo_offset + 4U);
        lo1 = lo0;
    } while (hi0 != hi1);
    return (((u64)hi1) << 32) | (u64)lo1;
}

static const char *stage_counter_name(unsigned stage)
{
    switch (stage) {
    case 0U: return "IDLE";
    case 1U: return "PARAM_INIT";
    case 2U: return "FRAME_LOAD";
    case 3U: return "MAIN_CTRL";
    case 4U: return "CONV_WEIGHT_LOAD";
    case 5U: return "CONV_ROW_DATAPATH";
    case 6U: return "PPU_ROW_CONSUME";
    case 7U: return "PPU_BLOCK5_FINAL";
    case 8U: return "VEC_FIXED";
    case 9U: return "AVGPOOL";
    case 10U: return "UPSAMPLE_OUT";
    case 11U: return "FRAME_STORE";
    case 12U: return "ERROR";
    default: return "UNKNOWN";
    }
}

static u64 stage_counter_cycles_to_us(u64 cycles)
{
    return (cycles * 1000000ULL + (INT8_APP_PL_TARGET_HZ / 2ULL)) /
           INT8_APP_PL_TARGET_HZ;
}

void stage_counter_clear(void)
{
    stage_counter_write32(STAGE_COUNTER_CTRL_OFFSET, STAGE_COUNTER_CTRL_CLEAR);
}

void stage_counter_enable(unsigned enable)
{
    u32 ctrl = stage_counter_read32(STAGE_COUNTER_CTRL_OFFSET);
    if (enable != 0U) {
        ctrl |= STAGE_COUNTER_CTRL_ENABLE;
    } else {
        ctrl &= ~STAGE_COUNTER_CTRL_ENABLE;
    }
    stage_counter_write32(STAGE_COUNTER_CTRL_OFFSET, ctrl);
}

void stage_counter_freeze(unsigned freeze)
{
    u32 ctrl = stage_counter_read32(STAGE_COUNTER_CTRL_OFFSET);
    if (freeze != 0U) {
        ctrl |= STAGE_COUNTER_CTRL_FREEZE;
    } else {
        ctrl &= ~STAGE_COUNTER_CTRL_FREEZE;
    }
    stage_counter_write32(STAGE_COUNTER_CTRL_OFFSET, ctrl);
}

u64 stage_counter_read_total(void)
{
    return stage_counter_read64(STAGE_COUNTER_TOTAL_LO_OFFSET);
}

u64 stage_counter_read_active(void)
{
    return stage_counter_read64(STAGE_COUNTER_ACTIVE_LO_OFFSET);
}

u64 stage_counter_read_stage(unsigned stage)
{
    if (stage >= INT8_STAGE_COUNTER_STAGE_COUNT) {
        return 0ULL;
    }
    return stage_counter_read64(STAGE_COUNTER_STAGE_BASE_OFFSET +
                                stage * STAGE_COUNTER_STAGE_STRIDE);
}

u32 stage_counter_read_current(void)
{
    return stage_counter_read32(STAGE_COUNTER_CURRENT_OFFSET);
}

u32 stage_counter_read_status(void)
{
    return stage_counter_read32(STAGE_COUNTER_STATUS_OFFSET);
}

int stage_counter_has_conv_profile(void)
{
    return ((stage_counter_read32(STAGE_COUNTER_CAPABILITY_OFFSET) &
             STAGE_COUNTER_CAP_CONV_PROFILE) != 0U) ? 1 : 0;
}

static u64 stage_counter_read_conv_raw(u32 base, unsigned state,
                                       unsigned state_count)
{
    if (state == 0U || state >= state_count) {
        return 0ULL;
    }
    return stage_counter_read64(base + state * STAGE_COUNTER_STAGE_STRIDE);
}

static u64 stage_counter_derive_conv_idle(u32 base, unsigned state_count)
{
    const u64 conv_cycles = stage_counter_read_stage(5U);
    u64 active_cycles = 0ULL;
    unsigned state;

    for (state = 1U; state < state_count; ++state) {
        active_cycles += stage_counter_read_conv_raw(base, state,
                                                     state_count);
    }
    return (conv_cycles >= active_cycles) ?
               (conv_cycles - active_cycles) : 0ULL;
}

u64 stage_counter_read_conv_win(unsigned state)
{
    if (state >= INT8_CONV_WIN_STATE_COUNT ||
        !stage_counter_has_conv_profile()) {
        return 0ULL;
    }
    if (state == 0U) {
        return stage_counter_derive_conv_idle(
            STAGE_COUNTER_CONV_WIN_BASE_OFFSET,
            INT8_CONV_WIN_STATE_COUNT);
    }
    return stage_counter_read_conv_raw(STAGE_COUNTER_CONV_WIN_BASE_OFFSET,
                                       state, INT8_CONV_WIN_STATE_COUNT);
}

u64 stage_counter_read_conv_sa(unsigned state)
{
    if (state >= INT8_CONV_SA_STATE_COUNT ||
        !stage_counter_has_conv_profile()) {
        return 0ULL;
    }
    if (state == 0U) {
        return stage_counter_derive_conv_idle(
            STAGE_COUNTER_CONV_SA_BASE_OFFSET,
            INT8_CONV_SA_STATE_COUNT);
    }
    return stage_counter_read_conv_raw(STAGE_COUNTER_CONV_SA_BASE_OFFSET,
                                       state, INT8_CONV_SA_STATE_COUNT);
}

u64 stage_counter_read_conv_post(unsigned state)
{
    if (state >= INT8_CONV_POST_STATE_COUNT ||
        !stage_counter_has_conv_profile()) {
        return 0ULL;
    }
    if (state == 0U) {
        return stage_counter_derive_conv_idle(
            STAGE_COUNTER_CONV_POST_BASE_OFFSET,
            INT8_CONV_POST_STATE_COUNT);
    }
    return stage_counter_read_conv_raw(STAGE_COUNTER_CONV_POST_BASE_OFFSET,
                                       state, INT8_CONV_POST_STATE_COUNT);
}

u32 stage_counter_read_conv_current(void)
{
    if (!stage_counter_has_conv_profile()) {
        return 0U;
    }
    return stage_counter_read32(STAGE_COUNTER_CONV_CURRENT_OFFSET);
}

static const char *stage_counter_conv_win_name(unsigned state)
{
    switch (state) {
    case 0U: return "IDLE_OR_DONE";
    case 1U: return "ACTIVE";
    default: return "UNKNOWN";
    }
}

static const char *stage_counter_conv_sa_name(unsigned state)
{
    switch (state) {
    case 0U: return "IDLE_OR_DONE";
    case 1U: return "COMPUTE_OR_WAIT_ACT";
    case 2U: return "PSUM_EMIT_OR_WAIT";
    default: return "UNKNOWN";
    }
}

static const char *stage_counter_conv_post_name(unsigned state)
{
    switch (state) {
    case 0U: return "IDLE_OR_DONE";
    case 1U: return "STAGING_WORD_OR_WAIT_PSUM";
    case 2U: return "COMMIT_WORD_OR_WAIT_PSUM";
    default: return "UNKNOWN";
    }
}

static void stage_counter_print_conv_row(const char *owner, unsigned state,
                                         const char *name, u64 cycles,
                                         u64 conv_cycles)
{
    u64 pct_x100 = (conv_cycles > 0ULL) ?
                       ((cycles * 10000ULL) / conv_cycles) : 0ULL;
    xil_printf("%s,%u,%s,%llu,%llu.%02llu\r\n",
               owner, state, name, cycles,
               pct_x100 / 100ULL, pct_x100 % 100ULL);
}

void stage_counter_dump_conv_csv(void)
{
#if INT8_STAGE_COUNTER_PRESENT
    u64 conv_cycles;
    u64 win_sum = 0ULL;
    u64 sa_sum = 0ULL;
    u64 post_sum = 0ULL;
    unsigned state;

    if (!stage_counter_has_conv_profile()) {
        xil_printf("RTL_CONV_CYCLES_UNAVAILABLE: capability bit absent\r\n");
        return;
    }

    conv_cycles = stage_counter_read_stage(5U);
    xil_printf("APP: POST bins classify psum words (wait/requant/pack included); "
               "not phase times\r\n");
    xil_printf("RTL_CONV_CYCLES_BEGIN\r\n");
    xil_printf("owner,state_id,state_name,cycles,percent_conv\r\n");
    for (state = 0U; state < INT8_CONV_WIN_STATE_COUNT; ++state) {
        const u64 cycles = stage_counter_read_conv_win(state);
        win_sum += cycles;
        stage_counter_print_conv_row("WIN", state,
                                     stage_counter_conv_win_name(state),
                                     cycles, conv_cycles);
    }
    for (state = 0U; state < INT8_CONV_SA_STATE_COUNT; ++state) {
        const u64 cycles = stage_counter_read_conv_sa(state);
        sa_sum += cycles;
        stage_counter_print_conv_row("SA", state,
                                     stage_counter_conv_sa_name(state),
                                     cycles, conv_cycles);
    }
    for (state = 0U; state < INT8_CONV_POST_STATE_COUNT; ++state) {
        const u64 cycles = stage_counter_read_conv_post(state);
        post_sum += cycles;
        stage_counter_print_conv_row("POST", state,
                                     stage_counter_conv_post_name(state),
                                     cycles, conv_cycles);
    }
    xil_printf("RTL_CONV_CHECK,conv_stage=%llu,win_sum=%llu,sa_sum=%llu,"
               "post_sum=%llu,%s\r\n",
               conv_cycles, win_sum, sa_sum, post_sum,
               (win_sum == conv_cycles && sa_sum == conv_cycles &&
                post_sum == conv_cycles) ? "PASS" : "FAIL");
    xil_printf("RTL_CONV_CURRENT,0x%08x\r\n",
               stage_counter_read_conv_current());
    xil_printf("RTL_CONV_CYCLES_END\r\n");
#else
    xil_printf("RTL_CONV_CYCLES_UNAVAILABLE: npu_stage_counter base macro not found\r\n");
#endif
}

void stage_counter_dump_csv(void)
{
#if INT8_STAGE_COUNTER_PRESENT
    u64 window_cycles = stage_counter_read_active();
    unsigned stage;
    xil_printf("RTL_STAGE_CYCLES_BEGIN\r\n");
    xil_printf("stage_id,stage_name,cycles,us,percent_window\r\n");
    for (stage = 0U; stage < INT8_STAGE_COUNTER_STAGE_COUNT; ++stage) {
        u64 cycles = stage_counter_read_stage(stage);
        u64 us = stage_counter_cycles_to_us(cycles);
        u64 pct_x100 = (window_cycles > 0ULL) ? ((cycles * 10000ULL) / window_cycles) : 0ULL;
        xil_printf("%u,%s,%llu,%llu,%llu.%02llu\r\n",
                   stage,
                   stage_counter_name(stage),
                   cycles,
                   us,
                   pct_x100 / 100ULL,
                   pct_x100 % 100ULL);
    }
    xil_printf("RTL_STAGE_TOTAL,%llu window=%llu current=0x%08x status=0x%08x base=0x%08x\r\n",
               stage_counter_read_total(),
               window_cycles,
               stage_counter_read32(STAGE_COUNTER_CURRENT_OFFSET),
               stage_counter_read32(STAGE_COUNTER_STATUS_OFFSET),
               (u32)INT8_STAGE_COUNTER_BASEADDR);
    xil_printf("RTL_STAGE_CYCLES_END\r\n");
    stage_counter_dump_conv_csv();
#else
    xil_printf("RTL_STAGE_CYCLES_UNAVAILABLE: npu_stage_counter base macro not found\r\n");
#endif
}

static u32 read_u32_le(const u8 *p)
{
    return ((u32)p[0]) | ((u32)p[1] << 8) | ((u32)p[2] << 16) |
           ((u32)p[3] << 24);
}

u32 int8_npu_read_ap_ctrl(const Int8NpuContext *ctx)
{
    if (ctx == NULL) {
        return 0U;
    }
    return XEspnet_encoder_int8_core_ReadReg(
        ctx->ip.Control_BaseAddress,
        XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_AP_CTRL);
}

static void print_u64_arg(const char *name, u64 value)
{
    xil_printf("%s=0x%08x%08x ", name, (u32)(value >> 32), (u32)value);
}

void int8_npu_dump_regs(const Int8NpuContext *ctx, const char *tag)
{
    u32 ap_ctrl;
    u32 ier;
    u32 isr;
    u32 mode;
    u32 uop_count;
    u64 input_addr;
    u64 output_addr;
    u64 param_addr;

    if (ctx == NULL) {
        return;
    }

    ap_ctrl = int8_npu_read_ap_ctrl(ctx);
    ier = XEspnet_encoder_int8_core_InterruptGetEnabled(
        (XEspnet_encoder_int8_core *)&ctx->ip);
    isr = XEspnet_encoder_int8_core_InterruptGetStatus(
        (XEspnet_encoder_int8_core *)&ctx->ip);
    mode = XEspnet_encoder_int8_core_Get_mode(
        (XEspnet_encoder_int8_core *)&ctx->ip);
    uop_count = XEspnet_encoder_int8_core_Get_uop_count(
        (XEspnet_encoder_int8_core *)&ctx->ip);
    input_addr = XEspnet_encoder_int8_core_Get_gmem_frame_in(
        (XEspnet_encoder_int8_core *)&ctx->ip);
    output_addr = XEspnet_encoder_int8_core_Get_gmem_frame_out(
        (XEspnet_encoder_int8_core *)&ctx->ip);
    param_addr = XEspnet_encoder_int8_core_Get_gmem_param(
        (XEspnet_encoder_int8_core *)&ctx->ip);

    xil_printf("NPU: regs %s ap=0x%08x ier=0x%08x isr=0x%08x "
               "mode=0x%08x uops=0x%08x\r\n",
               tag, ap_ctrl, ier, isr, mode, uop_count);
    xil_printf("NPU: regs %s ", tag);
    print_u64_arg("in", input_addr);
    print_u64_arg("out", output_addr);
    print_u64_arg("param", param_addr);
    xil_printf("\r\n");
    xil_printf("NPU: runtime abi=%08x capability=%08x status=%08x\r\n",
        int8_npu_runtime_abi(ctx),int8_npu_runtime_capability(ctx),int8_npu_runtime_status(ctx));
}

int int8_npu_init(Int8NpuContext *ctx)
{
    int status;
    XEspnet_encoder_int8_core_Config cfg;

    if (ctx == NULL) {
        return XST_FAILURE;
    }

    memset(ctx,0,sizeof(*ctx));
    memset(&cfg,0,sizeof(cfg));
#ifdef SDT
    cfg.Name = (char *)"espnet_encoder_int8_0";
#else
    cfg.DeviceId = 0U;
#endif
    cfg.Control_BaseAddress = (u64)XPAR_XESPNET_ENCODER_INT8_CORE_0_BASEADDR;

    status = XEspnet_encoder_int8_core_CfgInitialize(&ctx->ip, &cfg);
    if (status != XST_SUCCESS) {
        xil_printf("NPU: driver initialize failed, status=%d\r\n", status);
        return status;
    }

    XEspnet_encoder_int8_core_DisableAutoRestart(&ctx->ip);
    XEspnet_encoder_int8_core_InterruptGlobalDisable(&ctx->ip);
    XEspnet_encoder_int8_core_InterruptDisable(&ctx->ip, 0x3U);
    XEspnet_encoder_int8_core_InterruptClear(&ctx->ip, 0x3U);

    xil_printf("NPU: control base=0x%08x\r\n",
               (u32)ctx->ip.Control_BaseAddress);
    int8_npu_dump_regs(ctx, "after_init");
    return XST_SUCCESS;
}

int int8_npu_read_param_header(const u8 *param_blob, u32 param_bytes,
                               Int8ParamBlobHeader *header)
{
    if (param_blob == NULL || header == NULL ||
        param_bytes < INT8_PARAM_HEADER_BYTES) {
        return XST_FAILURE;
    }

    header->magic = read_u32_le(param_blob + 0U);
    header->version = read_u32_le(param_blob + 4U);
    header->tensor_desc_count = read_u32_le(param_blob + 8U);
    header->scale_desc_count = read_u32_le(param_blob + 12U);
    header->conv_desc_count = read_u32_le(param_blob + 16U);
    header->affine_desc_count = read_u32_le(param_blob + 20U);
    header->add_desc_count = read_u32_le(param_blob + 24U);
    header->pool_desc_count = read_u32_le(param_blob + 28U);
    header->uop_count = read_u32_le(param_blob + 32U);
    header->exec_plan_count = read_u32_le(param_blob + 36U);
    header->tensor_desc_offset = read_u32_le(param_blob + 40U);
    header->scale_desc_offset = read_u32_le(param_blob + 44U);
    header->conv_desc_offset = read_u32_le(param_blob + 48U);
    header->affine_desc_offset = read_u32_le(param_blob + 52U);
    header->add_desc_offset = read_u32_le(param_blob + 56U);
    header->pool_desc_offset = read_u32_le(param_blob + 60U);
    header->uop_offset = read_u32_le(param_blob + 64U);
    header->weight_data_offset = read_u32_le(param_blob + 68U);
    header->conv_qparam_offset = read_u32_le(param_blob + 72U);
    header->affine_qparam_offset = read_u32_le(param_blob + 76U);
    header->add_qparam_offset = read_u32_le(param_blob + 80U);
    header->pool_qparam_offset = read_u32_le(param_blob + 84U);
    header->conv_exec_desc_offset = read_u32_le(param_blob + 88U);
    header->window_sched_offset = read_u32_le(param_blob + 92U);
    header->window_cmd_offset = read_u32_le(param_blob + 96U);
    header->row_consumer_offset = read_u32_le(param_blob + 100U);
    header->fixed_exec_offset = read_u32_le(param_blob + 104U);
    header->exec_plan_offset = read_u32_le(param_blob + 108U);
    header->window_sched_count = read_u32_le(param_blob + 112U);
    header->window_cmd_count = read_u32_le(param_blob + 116U);
    header->block5_sched_offset = read_u32_le(param_blob + 120U);
    header->row_consumer_count = read_u32_le(param_blob + 124U);

    return XST_SUCCESS;
}

int int8_npu_validate_param_header(const Int8ParamBlobHeader *header,
                                   u32 param_bytes)
{
    u32 exec_end;
    u32 uop_end;

    if (header == NULL) {
        return XST_FAILURE;
    }
    if (param_bytes > INT8_PARAM_HW_MAX_BYTES) {
        xil_printf("NPU: param blob exceeds HLS m_axi depth, bytes=%u max=%u\r\n",
                   param_bytes, INT8_PARAM_HW_MAX_BYTES);
        return XST_FAILURE;
    }
    if (header->magic != INT8_PARAM_BLOB_MAGIC) {
        xil_printf("NPU: bad param magic=0x%08x\r\n", header->magic);
        return XST_FAILURE;
    }
    if (header->version != INT8_PARAM_BLOB_VERSION_CURRENT) {
        xil_printf("NPU: bad param version=0x%08x\r\n", header->version);
        return XST_FAILURE;
    }
    if (header->uop_count != INT8_EXPECTED_UOP_COUNT) {
        xil_printf("NPU: unexpected uop_count=%u expected=%u\r\n",
                   header->uop_count, INT8_EXPECTED_UOP_COUNT);
        return XST_FAILURE;
    }
    {
        const u32 expected_exec_count = INT8_EXPECTED_EXEC_PLAN_COUNT;
        if (header->exec_plan_count != expected_exec_count ||
        header->exec_plan_count > INT8_MAX_EXEC_PLAN_COUNT) {
            xil_printf("NPU: invalid exec_plan_count=%u expected=%u max=%u\r\n",
                       header->exec_plan_count, expected_exec_count,
                       INT8_MAX_EXEC_PLAN_COUNT);
            return XST_FAILURE;
        }
    }
    if (header->uop_offset > param_bytes) {
        xil_printf("NPU: invalid uop_offset=%u bytes=%u\r\n",
                   header->uop_offset, param_bytes);
        return XST_FAILURE;
    }

    uop_end = header->uop_offset + header->uop_count * INT8_UOP_BYTES;
    if (uop_end < header->uop_offset || uop_end > param_bytes) {
        xil_printf("NPU: invalid uop range, off=%u count=%u bytes=%u\r\n",
                   header->uop_offset, header->uop_count, param_bytes);
        return XST_FAILURE;
    }
    if (header->exec_plan_offset > param_bytes) {
        xil_printf("NPU: invalid exec_plan_offset=%u bytes=%u\r\n",
                   header->exec_plan_offset, param_bytes);
        return XST_FAILURE;
    }
    exec_end = header->exec_plan_offset +
               header->exec_plan_count * INT8_EXEC_PLAN_ENTRY_BYTES;
    if (exec_end < header->exec_plan_offset || exec_end > param_bytes) {
        xil_printf("NPU: invalid exec plan range, off=%u count=%u bytes=%u\r\n",
                   header->exec_plan_offset, header->exec_plan_count,
                   param_bytes);
        return XST_FAILURE;
    }

    xil_printf("NPU: param ok, uops=%u exec=%u conv=%u affine=%u add=%u pool=%u\r\n",
               header->uop_count, header->exec_plan_count,
               header->conv_desc_count,
               header->affine_desc_count, header->add_desc_count,
               header->pool_desc_count);
    return XST_SUCCESS;
}


u32 int8_npu_runtime_abi(const Int8NpuContext *ctx) {
    return XEspnet_encoder_int8_core_ReadReg(ctx->ip.Control_BaseAddress,
        XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_RUNTIME_ABI_ID_DATA);
}
u32 int8_npu_runtime_capability(const Int8NpuContext *ctx) {
    return XEspnet_encoder_int8_core_ReadReg(ctx->ip.Control_BaseAddress,
        XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_RUNTIME_CAPABILITY_DATA);
}
u32 int8_npu_runtime_status(const Int8NpuContext *ctx) {
    return XEspnet_encoder_int8_core_ReadReg(ctx->ip.Control_BaseAddress,
        XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_RUNTIME_STATUS_DATA);
}
int int8_npu_select_runtime(Int8NpuContext *ctx,u32 profile,u32 classes,u32 source,u32 mask) {
    if (!ctx || ctx->blocked) return XST_FAILURE;
    ctx->selected=ctx->ready=0;
    if ((classes!=2U && classes!=20U) || !runtime_profile_lookup(profile,&ctx->profile) ||
        !runtime_io_lookup(profile,source,mask,&ctx->io)) return XST_FAILURE;
    ctx->classes=classes; ctx->selected=1;
    return XST_SUCCESS;
}
int int8_npu_validate_deployment_geometry(const Int8NpuContext *ctx,const u8 *param_blob,
                                           u32 param_bytes,const Int8ParamBlobHeader *header) {
    if (!ctx || !ctx->selected || !param_blob || !header ||
        header->tensor_desc_count!=19U || header->tensor_desc_offset>param_bytes ||
        19U*INT8_PARAM_TENSOR_DESC_BYTES>param_bytes-header->tensor_desc_offset) return XST_FAILURE;
    const u8 *input=param_blob+header->tensor_desc_offset;
    const u8 *logits=input+INT8_PARAM_LOGITS_TENSOR_ID*INT8_PARAM_TENSOR_DESC_BYTES;
    u32 in_hw=read_u32_le(input+8),out_hw=read_u32_le(logits+8);
    if (input[1]!=1U || logits[1]!=1U || (in_hw&65535U)!=ctx->profile.input_h ||
        (in_hw>>16)!=ctx->profile.input_w || (read_u32_le(input+12)&65535U)!=ctx->profile.input_c ||
        (out_hw&65535U)!=ctx->profile.logits_h || (out_hw>>16)!=ctx->profile.logits_w ||
        (read_u32_le(logits+12)&65535U)!=ctx->classes) {
        xil_printf("NPU: PARAM/profile geometry or class count mismatch\r\n"); return XST_FAILURE;
    }
    return XST_SUCCESS;
}

static int run_once(Int8NpuContext *ctx,u32 mode,UINTPTR input_addr,UINTPTR output_addr,
                    UINTPTR param_addr,u32 uop_count,u32 timeout_polls,const char *tag) {
    if (!ctx || ctx->blocked || !ctx->selected || (mode==INT8_NPU_MODE_RUN && !ctx->ready) ||
        !param_addr || uop_count!=INT8_EXPECTED_UOP_COUNT ||
        !XEspnet_encoder_int8_core_IsReady(&ctx->ip)) return XST_FAILURE;
    if (mode==INT8_NPU_MODE_INIT) ctx->ready=0;
    XEspnet_encoder_int8_core_Set_gmem_frame_in(&ctx->ip,(u64)input_addr);
    XEspnet_encoder_int8_core_Set_gmem_frame_out(&ctx->ip,(u64)output_addr);
    XEspnet_encoder_int8_core_Set_gmem_param(&ctx->ip,(u64)param_addr);
    XEspnet_encoder_int8_core_Set_mode(&ctx->ip,mode);
    XEspnet_encoder_int8_core_Set_uop_count(&ctx->ip,uop_count);
    XEspnet_encoder_int8_core_WriteReg(ctx->ip.Control_BaseAddress,
        XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_MODEL_PROFILE_ID_DATA,ctx->profile.profile_id);
    XEspnet_encoder_int8_core_WriteReg(ctx->ip.Control_BaseAddress,
        XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_SRC_MODE_DATA,ctx->io.src_mode);
    XEspnet_encoder_int8_core_WriteReg(ctx->ip.Control_BaseAddress,
        XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_MASK_MODE_DATA,ctx->io.mask_mode);
    XEspnet_encoder_int8_core_Start(&ctx->ip);
    for (u32 poll=0;poll<timeout_polls;++poll) {
        if (XEspnet_encoder_int8_core_IsDone(&ctx->ip)) {
            const u32 status=int8_npu_runtime_status(ctx);
            if (int8_npu_runtime_abi(ctx)!=RUNTIME_ABI_ID ||
                int8_npu_runtime_capability(ctx)!=RUNTIME_CAPABILITY ||
                status!=((ctx->profile.profile_id<<16)|1U)) {
                ctx->ready=0;
                xil_printf("NPU: %s runtime ABI/capability/status rejected\r\n",tag);
                int8_npu_dump_regs(ctx,tag); return XST_FAILURE;
            }
            ctx->ready=1; return XST_SUCCESS;
        }
    }
    ctx->blocked=1; ctx->ready=0;
    xil_printf("NPU: timeout in %s; reset the hardware before retrying\r\n",tag);
    int8_npu_dump_regs(ctx,tag);
    return XST_FAILURE;
}

int int8_npu_run_init(Int8NpuContext *ctx, UINTPTR param_addr, u32 uop_count,
                      u32 timeout_polls)
{
    return run_once(ctx, INT8_NPU_MODE_INIT, (UINTPTR)0U, (UINTPTR)0U,
                    param_addr, uop_count, timeout_polls, "MODE_INIT");
}

int int8_npu_run_infer(Int8NpuContext *ctx, UINTPTR input_addr,
                       UINTPTR output_addr, UINTPTR param_addr, u32 uop_count,
                       u32 timeout_polls)
{
    if (input_addr == (UINTPTR)0U || output_addr == (UINTPTR)0U ||
        param_addr == (UINTPTR)0U) {
        return XST_FAILURE;
    }
    return run_once(ctx, INT8_NPU_MODE_RUN, input_addr, output_addr, param_addr,
                    uop_count, timeout_polls, "MODE_RUN");
}
