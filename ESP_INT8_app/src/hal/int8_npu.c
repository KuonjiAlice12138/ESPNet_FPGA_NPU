#include "./int8_npu.h"

#include "../app_config.h"
#include "xespnet_encoder_int8_core_hw.h"
#include "xil_printf.h"
#include "xparameters.h"
#include "xstatus.h"

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

static void int8_npu_dump_debug_regs(const Int8NpuContext *ctx, const char *tag)
{
#if defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_DBG_STATUS_DATA) && \
    defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_DBG_HEARTBEAT_DATA) && \
    defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_DBG_ACT_WORDS_DATA) && \
    defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_DBG_WGT_WORDS_DATA) && \
    defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_DBG_PSUM_WORDS_DATA) && \
    defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_DBG_OUT_WORDS_DATA) && \
    defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_DBG_HW_VERSION_DATA)
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

void int8_npu_dump_profile_regs(const Int8NpuContext *ctx, const char *tag)
{
#if defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF_UOP_COUNT_DATA) && \
    defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF_CONV_COUNT_DATA) && \
    defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF_WIN_READ_OPS_DATA) && \
    defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF_WIN_WORDS_DATA) && \
    defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF_WGT_WORDS_DATA) && \
    defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF_SA_MAC_STEPS_DATA) && \
    defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF_PSUM_WORDS_DATA) && \
    defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF_OUT_TILES_DATA) && \
    defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF_OUT_RMW_OPS_DATA) && \
    defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF_MODEL_CYCLES_DATA)
    u32 uop_count;
    u32 conv_count;
    u32 win_read_ops;
    u32 win_words;
    u32 wgt_words;
    u32 sa_mac_steps;
    u32 psum_words;
    u32 out_tiles;
    u32 out_rmw_ops;
    u32 model_cycles;

    if (ctx == NULL) {
        return;
    }

    uop_count = XEspnet_encoder_int8_core_ReadReg(
        ctx->ip.Control_BaseAddress,
        XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF_UOP_COUNT_DATA);
    conv_count = XEspnet_encoder_int8_core_ReadReg(
        ctx->ip.Control_BaseAddress,
        XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF_CONV_COUNT_DATA);
    win_read_ops = XEspnet_encoder_int8_core_ReadReg(
        ctx->ip.Control_BaseAddress,
        XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF_WIN_READ_OPS_DATA);
    win_words = XEspnet_encoder_int8_core_ReadReg(
        ctx->ip.Control_BaseAddress,
        XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF_WIN_WORDS_DATA);
    wgt_words = XEspnet_encoder_int8_core_ReadReg(
        ctx->ip.Control_BaseAddress,
        XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF_WGT_WORDS_DATA);
    sa_mac_steps = XEspnet_encoder_int8_core_ReadReg(
        ctx->ip.Control_BaseAddress,
        XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF_SA_MAC_STEPS_DATA);
    psum_words = XEspnet_encoder_int8_core_ReadReg(
        ctx->ip.Control_BaseAddress,
        XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF_PSUM_WORDS_DATA);
    out_tiles = XEspnet_encoder_int8_core_ReadReg(
        ctx->ip.Control_BaseAddress,
        XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF_OUT_TILES_DATA);
    out_rmw_ops = XEspnet_encoder_int8_core_ReadReg(
        ctx->ip.Control_BaseAddress,
        XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF_OUT_RMW_OPS_DATA);
    model_cycles = XEspnet_encoder_int8_core_ReadReg(
        ctx->ip.Control_BaseAddress,
        XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF_MODEL_CYCLES_DATA);

    xil_printf("NPU: prof %s uop=%u conv=%u win_read=%u win_words=%u\r\n",
               tag, uop_count, conv_count, win_read_ops, win_words);
    xil_printf("NPU: prof %s wgt=%u sa_steps=%u psum=%u out_tiles=%u "
               "rmw_ops=%u model_cycles=%u\r\n",
               tag, wgt_words, sa_mac_steps, psum_words, out_tiles,
               out_rmw_ops, model_cycles);
#if defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF2_WIN_SAVED_READS_DATA) && \
    defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF2_WIN_ACTUAL_READS_DATA) && \
    defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF2_OUT_DIRECT_WORDS_DATA) && \
    defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF2_OUT_RMW_READS_DATA)
    {
        u32 win_saved_reads = XEspnet_encoder_int8_core_ReadReg(
            ctx->ip.Control_BaseAddress,
            XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF2_WIN_SAVED_READS_DATA);
        u32 win_actual_reads = XEspnet_encoder_int8_core_ReadReg(
            ctx->ip.Control_BaseAddress,
            XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF2_WIN_ACTUAL_READS_DATA);
        u32 out_direct_words = XEspnet_encoder_int8_core_ReadReg(
            ctx->ip.Control_BaseAddress,
            XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF2_OUT_DIRECT_WORDS_DATA);
        u32 out_rmw_reads = XEspnet_encoder_int8_core_ReadReg(
            ctx->ip.Control_BaseAddress,
            XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF2_OUT_RMW_READS_DATA);
        xil_printf("NPU: prof2 %s win_saved=%u win_actual=%u "
                   "direct_words=%u rmw_reads=%u\r\n",
                   tag, win_saved_reads, win_actual_reads,
                   out_direct_words, out_rmw_reads);
    }
#endif
#if defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF3_WGT_CYCLES_DATA) && \
    defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF3_WIN_CYCLES_DATA) && \
    defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF3_SA_CYCLES_DATA) && \
    defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF3_POST_CYCLES_DATA) && \
    defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF3_WRITE_CYCLES_DATA) && \
    defined(XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF3_ROW_REGION_CYCLES_DATA)
    {
        u32 wgt_cycles = XEspnet_encoder_int8_core_ReadReg(
            ctx->ip.Control_BaseAddress,
            XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF3_WGT_CYCLES_DATA);
        u32 win_cycles = XEspnet_encoder_int8_core_ReadReg(
            ctx->ip.Control_BaseAddress,
            XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF3_WIN_CYCLES_DATA);
        u32 sa_cycles = XEspnet_encoder_int8_core_ReadReg(
            ctx->ip.Control_BaseAddress,
            XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF3_SA_CYCLES_DATA);
        u32 post_cycles = XEspnet_encoder_int8_core_ReadReg(
            ctx->ip.Control_BaseAddress,
            XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF3_POST_CYCLES_DATA);
        u32 write_cycles = XEspnet_encoder_int8_core_ReadReg(
            ctx->ip.Control_BaseAddress,
            XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF3_WRITE_CYCLES_DATA);
        u32 row_region_cycles = XEspnet_encoder_int8_core_ReadReg(
            ctx->ip.Control_BaseAddress,
            XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_PROF3_ROW_REGION_CYCLES_DATA);
        xil_printf("NPU: prof3 %s wgt_cyc=%u win_cyc=%u sa_cyc=%u\r\n",
                   tag, wgt_cycles, win_cycles, sa_cycles);
        xil_printf("NPU: prof3 %s post_cyc=%u write_cyc=%u "
                   "row_region_cyc=%u\r\n",
                   tag, post_cycles, write_cycles, row_region_cycles);
    }
#endif
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

    return XST_SUCCESS;
}

int int8_npu_validate_param_header(const Int8ParamBlobHeader *header,
                                   u32 param_bytes)
{
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

    xil_printf("NPU: param ok, uops=%u conv=%u affine=%u add=%u pool=%u\r\n",
               header->uop_count, header->conv_desc_count,
               header->affine_desc_count, header->add_desc_count,
               header->pool_desc_count);
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

int int8_npu_run_debug(Int8NpuContext *ctx, UINTPTR input_addr,
                       UINTPTR output_addr, UINTPTR param_addr, u32 uop_count,
                       u32 stop_after_uop, u32 dump_tensor_id,
                       u32 dump_words, u32 timeout_polls, const char *tag)
{
    u32 mode;
    u32 debug_uop_count;

    if (input_addr == (UINTPTR)0U || output_addr == (UINTPTR)0U ||
        param_addr == (UINTPTR)0U || uop_count > 0xffffU ||
        stop_after_uop > 0xfffeU || dump_tensor_id > 0xffU ||
        dump_words > 0xffffU) {
        return XST_FAILURE;
    }

    mode = INT8_NPU_MODE_RUN | INT8_NPU_DEBUG_ENABLE_MASK |
           ((dump_tensor_id & 0xffU) << INT8_NPU_DEBUG_DUMP_TENSOR_SHIFT) |
           ((dump_words & 0xffffU) << INT8_NPU_DEBUG_DUMP_WORDS_SHIFT);
    debug_uop_count =
        (uop_count & 0xffffU) |
        (((stop_after_uop + 1U) & 0xffffU)
         << INT8_NPU_DEBUG_STOP_AFTER_SHIFT);

    return run_once(ctx, mode, input_addr, output_addr, param_addr,
                    debug_uop_count, timeout_polls, tag);
}
