#include "./int8_npu.h"

#include "../app_config.h"
#include "xespnet_encoder_int8_core_hw.h"
#include "xil_io.h"
#include "xil_printf.h"
#include "xparameters.h"
#include "xstatus.h"

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
#define STAGE_COUNTER_STAGE_BASE_OFFSET 0x040U
#define STAGE_COUNTER_STAGE_STRIDE 0x008U

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

#if defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_DBG_STATUS_DATA) && \
    defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_DBG_HEARTBEAT_DATA) && \
    defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_DBG_ACT_WORDS_DATA) && \
    defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_DBG_WGT_WORDS_DATA) && \
    defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_DBG_PSUM_WORDS_DATA) && \
    defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_DBG_OUT_WORDS_DATA) && \
    defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_DBG_HW_VERSION_DATA)
#define INT8_NPU_HAS_DEBUG_REGS 1
#else
#define INT8_NPU_HAS_DEBUG_REGS 0
#endif

#if INT8_NPU_HAS_DEBUG_REGS
static const char *debug_phase_name(u32 phase)
{
    switch (phase) {
    case 0U: return "IDLE";
    case 1U: return "INIT";
    case 2U: return "PARAM";
    case 3U: return "IF_DEC";
    case 4U: return "FM_LOAD";
    case 5U: return "UOP_FETCH";
    case 6U: return "POOL";
    case 7U: return "STORE";
    case 8U: return "ADD";
    case 9U: return "AFFINE";
    case 10U: return "CONV";
    case 11U: return "DUMP";
    case 12U: return "FM_STORE";
    case 13U: return "DONE";
    case 14U: return "ERROR";
    default: return "UNKNOWN";
    }
}

static const char *debug_opcode_name(u32 opcode)
{
    switch (opcode) {
    case 0U: return "NOP";
    case 1U: return "LOAD_FM";
    case 2U: return "CONV";
    case 3U: return "POOL";
    case 4U: return "ADD";
    case 5U: return "AFFINE";
    case 6U: return "STORE";
    case 15U: return "END";
    default: return "UNKNOWN";
    }
}
#endif

static void int8_npu_dump_debug_regs(const Int8NpuContext *ctx, const char *tag)
{
#if INT8_NPU_HAS_DEBUG_REGS
    u32 status;
    u32 heartbeat;
    u32 act_words;
    u32 wgt_words;
    u32 psum_words;
    u32 out_words;
    u32 hw_version;
    u32 phase;
    u32 opcode;
    u32 current_uop;
    u32 last_done_uop;
    u32 error_code;

    if (ctx == NULL) {
        return;
    }

    status = XEspnet_encoder_int8_core_ReadReg(
        ctx->ip.Control_BaseAddress,
        XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_DBG_STATUS_DATA);
    heartbeat = XEspnet_encoder_int8_core_ReadReg(
        ctx->ip.Control_BaseAddress,
        XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_DBG_HEARTBEAT_DATA);
    act_words = XEspnet_encoder_int8_core_ReadReg(
        ctx->ip.Control_BaseAddress,
        XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_DBG_ACT_WORDS_DATA);
    wgt_words = XEspnet_encoder_int8_core_ReadReg(
        ctx->ip.Control_BaseAddress,
        XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_DBG_WGT_WORDS_DATA);
    psum_words = XEspnet_encoder_int8_core_ReadReg(
        ctx->ip.Control_BaseAddress,
        XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_DBG_PSUM_WORDS_DATA);
    out_words = XEspnet_encoder_int8_core_ReadReg(
        ctx->ip.Control_BaseAddress,
        XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_DBG_OUT_WORDS_DATA);
    hw_version = XEspnet_encoder_int8_core_ReadReg(
        ctx->ip.Control_BaseAddress,
        XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_DBG_HW_VERSION_DATA);

    phase = status & 0xffU;
    opcode = (status >> 8) & 0xffU;
    current_uop = (status >> 16) & 0xffU;
    last_done_uop = (status >> 24) & 0xffU;
    error_code = (heartbeat >> 16) & 0xffffU;

    xil_printf("NPU: dbg %s status=0x%08x phase=%u/%s opcode=%u/%s "
               "cur=%u last=%u hb=%u err=%u\r\n",
               tag, status, phase, debug_phase_name(phase), opcode,
               debug_opcode_name(opcode), current_uop, last_done_uop,
               heartbeat & 0xffffU, error_code);
    xil_printf("NPU: dbg %s stream act=%u wgt=%u psum=%u out=%u hw=0x%08x\r\n",
               tag, act_words, wgt_words, psum_words, out_words, hw_version);
#else
    (void)ctx;
    (void)tag;
#endif
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
    int8_npu_dump_debug_regs(ctx, tag);
}

int int8_npu_init(Int8NpuContext *ctx)
{
    int status;
    XEspnet_encoder_int8_core_Config cfg;

    if (ctx == NULL) {
        return XST_FAILURE;
    }

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
    if (header->version != INT8_PARAM_BLOB_VERSION) {
        xil_printf("NPU: bad param version=0x%08x\r\n", header->version);
        return XST_FAILURE;
    }
    if (header->uop_count != INT8_EXPECTED_UOP_COUNT) {
        xil_printf("NPU: unexpected uop_count=%u expected=%u\r\n",
                   header->uop_count, INT8_EXPECTED_UOP_COUNT);
        return XST_FAILURE;
    }
    if (header->exec_plan_count != INT8_EXPECTED_EXEC_PLAN_COUNT ||
        header->exec_plan_count > INT8_MAX_EXEC_PLAN_COUNT) {
        xil_printf("NPU: invalid exec_plan_count=%u expected=%u max=%u\r\n",
                   header->exec_plan_count, INT8_EXPECTED_EXEC_PLAN_COUNT,
                   INT8_MAX_EXEC_PLAN_COUNT);
        return XST_FAILURE;
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

int int8_npu_read_exec_plan_entry(const u8 *param_blob, u32 param_bytes,
                                  const Int8ParamBlobHeader *header,
                                  u32 index, Int8ExecPlanEntry *entry)
{
    u32 offset;

    if (param_blob == NULL || header == NULL || entry == NULL ||
        index >= header->exec_plan_count) {
        return XST_FAILURE;
    }
    offset = header->exec_plan_offset + index * INT8_EXEC_PLAN_ENTRY_BYTES;
    if (offset < header->exec_plan_offset ||
        offset + INT8_EXEC_PLAN_ENTRY_BYTES > param_bytes) {
        return XST_FAILURE;
    }

    entry->kind = param_blob[offset + 0U];
    entry->desc_id = param_blob[offset + 1U];
    entry->logical_uop_id = param_blob[offset + 2U];
    entry->flags = param_blob[offset + 3U];
    return XST_SUCCESS;
}

static int wait_done(Int8NpuContext *ctx, u32 timeout_polls, const char *tag)
{
    u32 i;
#if INT8_NPU_LIVE_DEBUG_INTERVAL_POLLS > 0U
    u32 next_live_poll = INT8_NPU_LIVE_DEBUG_INTERVAL_POLLS;
#endif

    for (i = 0U; i < timeout_polls; ++i) {
        if (XEspnet_encoder_int8_core_IsDone(&ctx->ip)) {
#if !INT8_APP_QUIET_PERF
            xil_printf("NPU: done %s, ap_ctrl=0x%08x\r\n", tag,
                       int8_npu_read_ap_ctrl(ctx));
            int8_npu_dump_regs(ctx, tag);
#endif
            return XST_SUCCESS;
        }
#if INT8_NPU_LIVE_DEBUG_INTERVAL_POLLS > 0U && !INT8_APP_QUIET_PERF
        if (i == next_live_poll) {
            xil_printf("NPU: wait %s poll=%u ap_ctrl=0x%08x\r\n", tag, i,
                       int8_npu_read_ap_ctrl(ctx));
            int8_npu_dump_debug_regs(ctx, tag);
            if (next_live_poll <=
                timeout_polls - INT8_NPU_LIVE_DEBUG_INTERVAL_POLLS) {
                next_live_poll += INT8_NPU_LIVE_DEBUG_INTERVAL_POLLS;
            } else {
                next_live_poll = timeout_polls;
            }
        }
#endif
    }

    xil_printf("NPU: timeout in %s, ap_ctrl=0x%08x\r\n", tag,
               int8_npu_read_ap_ctrl(ctx));
    int8_npu_dump_regs(ctx, tag);
    return XST_FAILURE;
}

static int run_once(Int8NpuContext *ctx, u32 mode, UINTPTR input_addr,
                    UINTPTR output_addr, UINTPTR param_addr, u32 uop_count,
                    u32 timeout_polls, const char *tag)
{
    if (ctx == NULL) {
        return XST_FAILURE;
    }
    if (!XEspnet_encoder_int8_core_IsReady(&ctx->ip)) {
        xil_printf("NPU: not ready before %s, ap_ctrl=0x%08x\r\n", tag,
                   int8_npu_read_ap_ctrl(ctx));
        return XST_FAILURE;
    }

    XEspnet_encoder_int8_core_Set_gmem_frame_in(&ctx->ip, (u64)input_addr);
    XEspnet_encoder_int8_core_Set_gmem_frame_out(&ctx->ip, (u64)output_addr);
    XEspnet_encoder_int8_core_Set_gmem_param(&ctx->ip, (u64)param_addr);
    XEspnet_encoder_int8_core_Set_mode(&ctx->ip, mode);
    XEspnet_encoder_int8_core_Set_uop_count(&ctx->ip, uop_count);

#if !INT8_APP_QUIET_PERF
    xil_printf("NPU: start %s mode=0x%08x uops=0x%08x\r\n", tag, mode,
               uop_count);
    int8_npu_dump_regs(ctx, tag);
#endif
    XEspnet_encoder_int8_core_Start(&ctx->ip);

    return wait_done(ctx, timeout_polls, tag);
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
