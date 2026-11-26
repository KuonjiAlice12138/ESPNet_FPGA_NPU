"""MinGW-only host contracts for the unified UART app, parser, and real HAL.

Run: python -B tools/test_unified_app_contract.py -v
Only generated mock BSP files and host executables live in TemporaryDirectory.
The published SD artifacts are read-only; no Vitis/platform build is invoked.
The mock accelerator supplies MASK.BIN, not a simulated neural-network result.
"""

import csv
import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
import unittest
import zlib


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "ESP_INT8_app" / "src"
ARTIFACTS = ROOT / "hw_artifacts" / "unified_int8"
CATALOG_HEADER = (
    "key,profile,classes,source,mask,param,input,output,param_crc,input_crc,mask_crc"
)
HEADERS = {
    "xil_types.h": r"""
        #ifndef MOCK_XIL_TYPES_H
        #define MOCK_XIL_TYPES_H
        #include <stdint.h>
        #include <stddef.h>
        typedef unsigned char u8;
        typedef unsigned short u16;
        typedef unsigned int u32;
        typedef unsigned long long u64;
        typedef uintptr_t UINTPTR;
        #endif
    """,
    "xstatus.h": "#define XST_SUCCESS 0\n#define XST_FAILURE 1\n",
    "xparameters.h": r"""
        #define XPAR_XESPNET_ENCODER_INT8_CORE_0_BASEADDR 0xA0000000U
        #define XPAR_NPU_STAGE_COUNTER_0_S00_AXI_BASEADDR 0xA1000000U
    """,
    "xil_printf.h": "int xil_printf(const char *format, ...);\n",
    "sleep.h": "void usleep(unsigned long microseconds);\n",
    "xil_io.h": r"""
        #include "xil_types.h"
        u32 Xil_In32(UINTPTR address);
        void Xil_Out32(UINTPTR address, u32 value);
    """,
    "xil_cache.h": r"""
        #include "xil_types.h"
        void Xil_DCacheFlushRange(UINTPTR address, u32 bytes);
        void Xil_DCacheInvalidateRange(UINTPTR address, u32 bytes);
    """,
    # Deliberately shuffled, non-hardware offsets: the HAL must use these names.
    "xespnet_encoder_int8_core_hw.h": r"""
        #ifndef MOCK_NPU_HW_H
        #define MOCK_NPU_HW_H
        #define XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_AP_CTRL 0x000U
        #define XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_MODEL_PROFILE_ID_DATA 0x3B0U
        #define XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_SRC_MODE_DATA 0x124U
        #define XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_MASK_MODE_DATA 0x2A8U
        #define XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_RUNTIME_ABI_ID_DATA 0x184U
        #define XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_RUNTIME_STATUS_DATA 0x2FCU
        #define XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_RUNTIME_CAPABILITY_DATA 0x208U
        #endif
    """,
    "xespnet_encoder_int8_core.h": r"""
        #ifndef MOCK_NPU_DRIVER_H
        #define MOCK_NPU_DRIVER_H
        #include "xil_types.h"
        typedef struct { UINTPTR Control_BaseAddress; u32 IsReady; }
            XEspnet_encoder_int8_core;
        typedef struct { u16 DeviceId; char *Name; u64 Control_BaseAddress; }
            XEspnet_encoder_int8_core_Config;
        int XEspnet_encoder_int8_core_CfgInitialize(
            XEspnet_encoder_int8_core *ip, XEspnet_encoder_int8_core_Config *cfg);
        void XEspnet_encoder_int8_core_DisableAutoRestart(XEspnet_encoder_int8_core *ip);
        void XEspnet_encoder_int8_core_InterruptGlobalDisable(XEspnet_encoder_int8_core *ip);
        void XEspnet_encoder_int8_core_InterruptDisable(XEspnet_encoder_int8_core *ip, u32 mask);
        void XEspnet_encoder_int8_core_InterruptClear(XEspnet_encoder_int8_core *ip, u32 mask);
        u32 XEspnet_encoder_int8_core_InterruptGetEnabled(XEspnet_encoder_int8_core *ip);
        u32 XEspnet_encoder_int8_core_InterruptGetStatus(XEspnet_encoder_int8_core *ip);
        u32 XEspnet_encoder_int8_core_Get_mode(XEspnet_encoder_int8_core *ip);
        u32 XEspnet_encoder_int8_core_Get_uop_count(XEspnet_encoder_int8_core *ip);
        u64 XEspnet_encoder_int8_core_Get_gmem_frame_in(XEspnet_encoder_int8_core *ip);
        u64 XEspnet_encoder_int8_core_Get_gmem_frame_out(XEspnet_encoder_int8_core *ip);
        u64 XEspnet_encoder_int8_core_Get_gmem_param(XEspnet_encoder_int8_core *ip);
        void XEspnet_encoder_int8_core_Set_mode(XEspnet_encoder_int8_core *ip, u32 value);
        void XEspnet_encoder_int8_core_Set_uop_count(XEspnet_encoder_int8_core *ip, u32 value);
        void XEspnet_encoder_int8_core_Set_gmem_frame_in(XEspnet_encoder_int8_core *ip, u64 value);
        void XEspnet_encoder_int8_core_Set_gmem_frame_out(XEspnet_encoder_int8_core *ip, u64 value);
        void XEspnet_encoder_int8_core_Set_gmem_param(XEspnet_encoder_int8_core *ip, u64 value);
        void XEspnet_encoder_int8_core_Start(XEspnet_encoder_int8_core *ip);
        u32 XEspnet_encoder_int8_core_IsReady(XEspnet_encoder_int8_core *ip);
        u32 XEspnet_encoder_int8_core_IsDone(XEspnet_encoder_int8_core *ip);
        u32 XEspnet_encoder_int8_core_ReadReg(UINTPTR base, u32 offset);
        void XEspnet_encoder_int8_core_WriteReg(UINTPTR base, u32 offset, u32 value);
        #endif
    """,
    "mock_bsp.h": r"""
        #ifndef MOCK_BSP_H
        #define MOCK_BSP_H
        #include "xespnet_encoder_int8_core.h"
        typedef struct {
            const char *fault, *root, *catalog;
            char model_dir[9];
            int config_result;
            unsigned hardware_ready, done_after, current_polls, completion_disabled;
            unsigned starts, init_runs, infer_runs, writes, polls, bad_mmio;
            unsigned loads, saves, flushes, invalidates, protocol_errors;
            u32 mode, uops, profile, source, mask, abi, status, capability, control;
            u64 input, output, param, ticks;
        } MockBsp;
        extern MockBsp mock_bsp;
        void mock_reset(const char *fault);
        void mock_report(void);
        #endif
    """,
    "ff.h": r"""
        #include "xil_types.h"
        typedef unsigned UINT;
        typedef unsigned long FSIZE_t;
        typedef int FRESULT;
        typedef struct { int unused; } FATFS;
        typedef struct { FSIZE_t size; unsigned is_log; } FIL;
        #define FR_OK 0
        #define FR_EXIST 8
        #define FA_READ 1
        #define FA_WRITE 2
        #define FA_CREATE_NEW 4
        #define FA_CREATE_ALWAYS 8
        #define f_size(file) ((file)->size)
        FRESULT f_mount(FATFS *fs,const char *path,int immediate);
        FRESULT f_open(FIL *file,const char *path,int mode);
        FRESULT f_read(FIL *file,void *dst,UINT count,UINT *read);
        FRESULT f_write(FIL *file,const void *src,UINT count,UINT *written);
        FRESULT f_sync(FIL *file);
        FRESULT f_close(FIL *file);
        FRESULT f_mkdir(const char *path);
    """,
}


MOCK_BSP = r"""
    #include <stdarg.h>
    #include <stdio.h>
    #include <string.h>
    #include "mock_bsp.h"
    #include "xespnet_encoder_int8_core_hw.h"
    #include "xparameters.h"
    #include "xstatus.h"
    #include "xil_io.h"
    #include "xil_cache.h"
    #include "sleep.h"
    #include "runtime_catalog.h"
    #include "drivers/sd_card.h"
    MockBsp mock_bsp;
    static char log_data[4*1024*1024];
    static u32 log_used;
    static unsigned log_closed;

    void mock_reset(const char *fault) {
        memset(&mock_bsp, 0, sizeof(mock_bsp));
        mock_bsp.fault = fault;
        mock_bsp.hardware_ready = mock_bsp.done_after = 1;
        mock_bsp.abi = 0x11160001U;
        mock_bsp.capability = 0x7FFFU;
    }
    static int fault_is(const char *name) { return !strcmp(mock_bsp.fault, name); }
    static int phase_fault(const char *name) {
        char fault[64];
        (void)snprintf(fault, sizeof(fault), "%s_%s", name,
                       mock_bsp.mode == 1U ? "init" : "infer");
        return fault_is(fault);
    }
    int xil_printf(const char *format, ...) {
        int result; va_list args;
        va_start(args, format); result = vprintf(format, args); va_end(args);
        return result;
    }
    char inbyte(void) { int c = getchar(); return c == EOF ? (char)4 : (char)c; }
    u64 host_timer_counter(void) { return mock_bsp.ticks++; }
    void usleep(unsigned long microseconds) {
        if (!fault_is("timer")) mock_bsp.ticks += (33333000ULL * microseconds) / 1000000ULL;
    }
    int XEspnet_encoder_int8_core_CfgInitialize(
        XEspnet_encoder_int8_core *ip, XEspnet_encoder_int8_core_Config *cfg) {
        ip->Control_BaseAddress = (UINTPTR)cfg->Control_BaseAddress;
        ip->IsReady = 1U;
        return mock_bsp.config_result;
    }
    void XEspnet_encoder_int8_core_DisableAutoRestart(XEspnet_encoder_int8_core *ip) { (void)ip; }
    void XEspnet_encoder_int8_core_InterruptGlobalDisable(XEspnet_encoder_int8_core *ip) { (void)ip; }
    void XEspnet_encoder_int8_core_InterruptDisable(XEspnet_encoder_int8_core *ip, u32 mask) {
        (void)ip; (void)mask;
    }
    void XEspnet_encoder_int8_core_InterruptClear(XEspnet_encoder_int8_core *ip, u32 mask) {
        (void)ip; (void)mask;
    }
    u32 XEspnet_encoder_int8_core_InterruptGetEnabled(XEspnet_encoder_int8_core *ip) { (void)ip; return 0; }
    u32 XEspnet_encoder_int8_core_InterruptGetStatus(XEspnet_encoder_int8_core *ip) { (void)ip; return 0; }
    u32 XEspnet_encoder_int8_core_Get_mode(XEspnet_encoder_int8_core *ip) { (void)ip; return mock_bsp.mode; }
    u32 XEspnet_encoder_int8_core_Get_uop_count(XEspnet_encoder_int8_core *ip) { (void)ip; return mock_bsp.uops; }
    u64 XEspnet_encoder_int8_core_Get_gmem_frame_in(XEspnet_encoder_int8_core *ip) { (void)ip; return mock_bsp.input; }
    u64 XEspnet_encoder_int8_core_Get_gmem_frame_out(XEspnet_encoder_int8_core *ip) { (void)ip; return mock_bsp.output; }
    u64 XEspnet_encoder_int8_core_Get_gmem_param(XEspnet_encoder_int8_core *ip) { (void)ip; return mock_bsp.param; }
    void XEspnet_encoder_int8_core_Set_mode(XEspnet_encoder_int8_core *ip, u32 value) {
        (void)ip; ++mock_bsp.writes; mock_bsp.mode = value;
    }
    void XEspnet_encoder_int8_core_Set_uop_count(XEspnet_encoder_int8_core *ip, u32 value) {
        (void)ip; ++mock_bsp.writes; mock_bsp.uops = value;
    }
    void XEspnet_encoder_int8_core_Set_gmem_frame_in(XEspnet_encoder_int8_core *ip, u64 value) {
        (void)ip; ++mock_bsp.writes; mock_bsp.input = value;
    }
    void XEspnet_encoder_int8_core_Set_gmem_frame_out(XEspnet_encoder_int8_core *ip, u64 value) {
        (void)ip; ++mock_bsp.writes; mock_bsp.output = value;
    }
    void XEspnet_encoder_int8_core_Set_gmem_param(XEspnet_encoder_int8_core *ip, u64 value) {
        (void)ip; ++mock_bsp.writes; mock_bsp.param = value;
    }
    u32 XEspnet_encoder_int8_core_ReadReg(UINTPTR base, u32 offset) {
        if (base != XPAR_XESPNET_ENCODER_INT8_CORE_0_BASEADDR) ++mock_bsp.bad_mmio;
        switch (offset) {
        case XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_AP_CTRL: return 4U;
        case XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_MODEL_PROFILE_ID_DATA: return mock_bsp.profile;
        case XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_SRC_MODE_DATA: return mock_bsp.source;
        case XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_MASK_MODE_DATA: return mock_bsp.mask;
        case XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_RUNTIME_ABI_ID_DATA: return mock_bsp.abi;
        case XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_RUNTIME_STATUS_DATA: return mock_bsp.status;
        case XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_RUNTIME_CAPABILITY_DATA: return mock_bsp.capability;
        default: ++mock_bsp.bad_mmio; return 0U;
        }
    }
    void XEspnet_encoder_int8_core_WriteReg(UINTPTR base, u32 offset, u32 value) {
        ++mock_bsp.writes;
        if (base != XPAR_XESPNET_ENCODER_INT8_CORE_0_BASEADDR) ++mock_bsp.bad_mmio;
        switch (offset) {
        case XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_MODEL_PROFILE_ID_DATA: mock_bsp.profile = value; break;
        case XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_SRC_MODE_DATA: mock_bsp.source = value; break;
        case XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_MASK_MODE_DATA: mock_bsp.mask = value; break;
        default: ++mock_bsp.bad_mmio; break;
        }
    }
    static int read_file(const char *path, void *out, u32 capacity, u32 *size) {
        FILE *file = fopen(path, "rb");
        long length;
        if (!file) return XST_FAILURE;
        if (fseek(file, 0, SEEK_END) || (length = ftell(file)) < 0 ||
            (unsigned long)length > capacity || fseek(file, 0, SEEK_SET)) {
            fclose(file); return XST_FAILURE;
        }
        *size = (u32)length;
        if (fread(out, 1, *size, file) != *size) { fclose(file); return XST_FAILURE; }
        fclose(file); return XST_SUCCESS;
    }
    void XEspnet_encoder_int8_core_Start(XEspnet_encoder_int8_core *ip) {
        runtime_io_t io;
        (void)ip;
        ++mock_bsp.writes; ++mock_bsp.starts; mock_bsp.current_polls = 0;
        if (mock_bsp.mode == 1U) ++mock_bsp.init_runs;
        else if (mock_bsp.mode == 2U) ++mock_bsp.infer_runs;
        else ++mock_bsp.protocol_errors;
        mock_bsp.abi = 0x11160001U; mock_bsp.capability = 0x7FFFU;
        mock_bsp.status = (mock_bsp.profile << 16) | 1U;
        mock_bsp.completion_disabled = (unsigned)phase_fault("timeout");
        if (phase_fault("bad_abi")) mock_bsp.abi ^= 1U;
        if (phase_fault("bad_capability")) mock_bsp.capability ^= 1U;
        if (phase_fault("extra_capability")) mock_bsp.capability |= 0x8000U;
        if (phase_fault("bad_status")) mock_bsp.status = mock_bsp.profile << 16;
        if (phase_fault("wrong_profile")) mock_bsp.status ^= 1U << 16;
        if (phase_fault("error_status")) mock_bsp.status |= 0x200U;
        printf("\nHOST_TX,%u,%u,%u,%u\n", mock_bsp.mode, mock_bsp.profile,
               mock_bsp.source, mock_bsp.mask);
        if (mock_bsp.mode != 2U || mock_bsp.completion_disabled) return;
        if (!runtime_io_lookup(mock_bsp.profile, mock_bsp.source, mock_bsp.mask, &io)) {
            ++mock_bsp.protocol_errors; return;
        }
        if (mock_bsp.root) {
            char path[1024]; u32 size;
            (void)snprintf(path, sizeof(path), "%s/%s/MASK.BIN", mock_bsp.root, mock_bsp.model_dir);
            if (read_file(path, (void *)(UINTPTR)mock_bsp.output, io.mask_bytes, &size) ||
                size != io.mask_bytes) { ++mock_bsp.protocol_errors; return; }
        } else memset((void *)(UINTPTR)mock_bsp.output, 0, io.mask_bytes);
        if (fault_is("mask_crc")) ((u8 *)(UINTPTR)mock_bsp.output)[0] ^= 1U;
        if (fault_is("mask_id")) ((u8 *)(UINTPTR)mock_bsp.output)[0] = 255U;
    }
    u32 XEspnet_encoder_int8_core_IsReady(XEspnet_encoder_int8_core *ip) {
        (void)ip; return mock_bsp.hardware_ready;
    }
    u32 XEspnet_encoder_int8_core_IsDone(XEspnet_encoder_int8_core *ip) {
        (void)ip; ++mock_bsp.polls; ++mock_bsp.current_polls;
        if (mock_bsp.completion_disabled) return 0;
        return mock_bsp.done_after && mock_bsp.current_polls >= mock_bsp.done_after;
    }
    u32 Xil_In32(UINTPTR address) {
        UINTPTR base = XPAR_NPU_STAGE_COUNTER_0_S00_AXI_BASEADDR;
        if (address < base || address - base > 0x200U) { ++mock_bsp.bad_mmio; return 0; }
        switch ((u32)(address - base)) {
        case 0x000U: return mock_bsp.control;
        case 0x008U: return fault_is("counter_current_error") ? 12U : 0U;
        case 0x010U: return fault_is("counter_zero") ? 0U : 100U;
        case 0x018U: return fault_is("counter_sum") ? 99U : 100U;
        case 0x020U: return fault_is("no_conv_counter") ? 0U : 1U;
        case 0x068U: return 100U; /* Stage 5: convolution. */
        case 0x0A0U: return fault_is("counter_error") ? 1U : 0U; /* Stage 12. */
        case 0x108U: return fault_is("conv_sum") ? 101U : 70U;
        case 0x148U: return 60U;
        case 0x150U: return 30U;
        case 0x188U: return 40U;
        case 0x190U: return 50U;
        default: return 0U; /* All high words and unused stages are zero. */
        }
    }
    void Xil_Out32(UINTPTR address, u32 value) {
        if (address != XPAR_NPU_STAGE_COUNTER_0_S00_AXI_BASEADDR) ++mock_bsp.bad_mmio;
        mock_bsp.control = value;
    }
    void Xil_DCacheFlushRange(UINTPTR address, u32 bytes) {
        ++mock_bsp.flushes;
        if ((address & 63U) || !bytes) ++mock_bsp.protocol_errors;
    }
    void Xil_DCacheInvalidateRange(UINTPTR address, u32 bytes) {
        ++mock_bsp.invalidates;
        if ((address & 63U) || !bytes) ++mock_bsp.protocol_errors;
    }
    int SD_Init(void) { return fault_is("sd_init") ? XST_FAILURE : XST_SUCCESS; }
    const char *SD_GetMountPath(void) { return "mock:/"; }
    static int overlay_file(const char *filename, char *path, size_t capacity) {
        const char *slash = strrchr(mock_bsp.catalog, '/');
        FILE *file;
        if (!slash) slash = strrchr(mock_bsp.catalog, '\\');
        if (!slash) return 0;
        (void)snprintf(path, capacity, "%.*s/%s", (int)(slash-mock_bsp.catalog),
                       mock_bsp.catalog, filename);
        file = fopen(path, "rb");
        if (!file) return 0;
        fclose(file); return 1;
    }
    int SD_FileExists(const char *filename) {
        char path[1024]; FILE *file;
        if (overlay_file(filename, path, sizeof(path))) return 1;
        (void)snprintf(path, sizeof(path), "%s/%s", mock_bsp.root, filename);
        file = fopen(path, "rb");
        if (!file) return 0;
        fclose(file); return 1;
    }
    int SD_LoadFileToMemory(const char *filename, UINTPTR dst, u32 capacity, u32 *size) {
        char path[1024]; int result;
        ++mock_bsp.loads;
        if (!strcmp(filename, "RUNS.CSV") && mock_bsp.catalog)
            (void)snprintf(path, sizeof(path), "%s", mock_bsp.catalog);
        else if (overlay_file(filename, path, sizeof(path))) { /* Host-only SD overlay. */ }
        else if (strchr(filename, '/') && strlen(strchr(filename, '/')+1)==9 &&
                 (strchr(filename, '/')[1]=='I' || strchr(filename, '/')[1]=='T')) {
            const char *slash = strchr(filename, '/');
            (void)snprintf(path, sizeof(path), "%s/%.*s/%s", mock_bsp.root,
                           (int)(slash-filename), filename, slash[1]=='I'?"INPUTQ.BIN":"MASK.BIN");
        }
        else (void)snprintf(path, sizeof(path), "%s/%s", mock_bsp.root, filename);
        result = read_file(path, (void *)dst, capacity, size);
        if (result != XST_SUCCESS) return result;
        if (strstr(filename, "/PARAM.BIN")) {
            const char *slash = strchr(filename, '/');
            size_t length = (size_t)(slash - filename);
            if (length >= sizeof(mock_bsp.model_dir)) { ++mock_bsp.protocol_errors; return XST_FAILURE; }
            memcpy(mock_bsp.model_dir, filename, length); mock_bsp.model_dir[length] = 0;
            if (fault_is("param_crc")) ((u8 *)dst)[0] ^= 1U;
            if (fault_is("empty_param")) *size = 0;
            if (fault_is("short_param")) *size = 127;
            if (fault_is("param_io")) return XST_FAILURE;
        }
        if (strstr(filename, "/INPUTQ.BIN") || strstr(filename, "/I0000.BIN")) {
            if (fault_is("input_crc")) ((u8 *)dst)[0] ^= 1U;
            if (fault_is("short_input")) --*size;
            if (fault_is("long_input") && *size < capacity) ((u8 *)dst)[(*size)++] = 0;
            if (fault_is("input_io")) return XST_FAILURE;
        }
        return XST_SUCCESS;
    }
    int SD_SaveMemoryToFile(const char *filename, const void *src, u32 size) {
        if (fault_is("save_io") || (fault_is("save_late") && mock_bsp.saves==16)) return XST_FAILURE;
        ++mock_bsp.saves;
        printf("\nHOST_SAVE,%s,%u,%08X\n", filename, size,
               (unsigned)runtime_crc32((const unsigned char *)src, size));
        return XST_SUCCESS;
    }
    int SD_LogOpen(char *path,u32 capacity) {
        if (fault_is("log_open")) return XST_FAILURE;
        (void)snprintf(path,capacity,"LOGS/TEST0000.TXT");
        log_used=0; log_closed=0; return XST_SUCCESS;
    }
    int SD_LogAppend(const void *data,u32 size) {
        if (fault_is("log_write") || size>sizeof(log_data)-log_used) return XST_FAILURE;
        memcpy(log_data+log_used,data,size); log_used+=size; return XST_SUCCESS;
    }
    int SD_LogSync(void) {
        return fault_is("log_sync") || (fault_is("log_late") && mock_bsp.infer_runs>=17)?
            XST_FAILURE:XST_SUCCESS;
    }
    int SD_LogClose(void) { log_closed=1; return fault_is("log_close")?XST_FAILURE:XST_SUCCESS; }
    void mock_report(void) {
        printf("\nHOST_SUMMARY,init=%u,infer=%u,loads=%u,saves=%u,writes=%u,polls=%u,"
               "bad_mmio=%u,protocol_errors=%u,flushes=%u,invalidates=%u\n",
               mock_bsp.init_runs, mock_bsp.infer_runs, mock_bsp.loads, mock_bsp.saves,
               mock_bsp.writes, mock_bsp.polls, mock_bsp.bad_mmio, mock_bsp.protocol_errors,
               mock_bsp.flushes, mock_bsp.invalidates);
        if (fault_is("capture_log")) {
            printf("HOST_LOG_STATE,path=LOGS/TEST0000.TXT,closed=%u\nHOST_LOG_BEGIN\n",log_closed);
            fwrite(log_data,1,log_used,stdout); puts("HOST_LOG_END");
        }
    }
"""


CATALOG_HARNESS = r"""
    #include <stdio.h>
    #include "runtime_catalog.h"
    int main(int argc, char **argv) {
        if (argc == 2 && !strcmp(argv[1], "crc")) {
            const unsigned char binary[] = {0, 128, 255};
            printf("%08X,%08X,%08X\n", (unsigned)runtime_crc32(NULL, 0),
                   (unsigned)runtime_crc32((const unsigned char *)"123456789", 9),
                   (unsigned)runtime_crc32(binary, sizeof(binary)));
            return 0;
        }
        if (argc == 4 && !strcmp(argv[1], "uint")) {
            uint32_t value;
            if (runtime_parse_uint(argv[3], (unsigned)atoi(argv[2]), &value))
                printf("OK,%u\n", (unsigned)value);
            else puts("REJECT");
            return 0;
        }
        if (argc == 3 && !strcmp(argv[1], "parse")) {
            runtime_case_t c;
            if (!runtime_parse_case(argv[2], &c)) { puts("REJECT"); return 0; }
            printf("OK,%s,%u,%u,%u,%u,%u,%u,%08X,%08X,%08X,%s,%s,%s\n",
                   c.key, (unsigned)c.profile, (unsigned)c.classes, (unsigned)c.source,
                   (unsigned)c.mask, (unsigned)c.input_bytes, (unsigned)c.output_bytes,
                   (unsigned)c.param_crc, (unsigned)c.input_crc, (unsigned)c.mask_crc,
                   c.param, c.input, c.output);
            return 0;
        }
        if (argc == 3 && !strcmp(argv[1], "parseval")) {
            runtime_val_case_t c;
            if (!runtime_parse_val_case(argv[2], &c)) { puts("REJECT"); return 0; }
            printf("OK,%u,%u,%u,%u,%u,%u\n", (unsigned)c.run.profile,
                   (unsigned)c.run.classes, (unsigned)c.run.source, (unsigned)c.run.mask,
                   (unsigned)c.run.input_bytes, (unsigned)c.run.output_bytes);
            return 0;
        }
        if (argc == 2 && !strcmp(argv[1], "lookup")) {
            runtime_profile_t p; runtime_io_t io;
            if (RUNTIME_ABI_ID != 0x11160001U || RUNTIME_CAPABILITY != 0x7FFFU ||
                RUNTIME_PROFILE_COUNT != 8U || runtime_profile_lookup(8, &p) ||
                runtime_profile_lookup(0, NULL) || runtime_io_lookup(0, 0, 0, NULL)) return 1;
            for (unsigned id = 0; id < 8; ++id) {
                if (!runtime_profile_lookup(id, &p) || p.profile_id != id) return 2;
                for (unsigned s = 0; s < 8; ++s) for (unsigned m = 0; m < 3; ++m)
                    if (runtime_io_lookup(id, s, m, &io))
                        printf("%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u\n", id, s, m,
                               (unsigned)io.source_h, (unsigned)io.source_w, (unsigned)io.source_c,
                               (unsigned)io.input_bytes, (unsigned)io.roi_top, (unsigned)io.roi_left,
                               (unsigned)io.roi_h, (unsigned)io.roi_w, (unsigned)io.mask_bytes);
            }
            return 0;
        }
        return 2;
    }
"""


HAL_HARNESS = r"""
    #include <stdio.h>
    #include <string.h>
    #include "app_config.h"
    #include "hal/int8_npu.h"
    #include "mock_bsp.h"
    #include "xstatus.h"
    #define CHECK(condition) do { if (!(condition)) { \
        fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } } while (0)
    #define PARAM_BYTES 2976U
    static u8 param[4096], input[1572864], output[524288];
    static void put32(u8 *out, u32 value) {
        for (unsigned i = 0; i < 4; ++i) out[i] = (u8)(value >> (8U * i));
    }
    static void make_param(const Int8NpuContext *ctx) {
        u8 *tensor;
        memset(param, 0, sizeof(param));
        put32(param, 0x544E4945U); put32(param + 4, 4);
        put32(param + 8, 19); put32(param + 32, 75); put32(param + 36, 16);
        put32(param + 40, 128); put32(param + 64, 512); put32(param + 108, 2912);
        tensor = param + 128; tensor[1] = 1;
        put32(tensor + 8, ctx->profile.input_h | (ctx->profile.input_w << 16));
        put32(tensor + 12, ctx->profile.input_c);
        tensor += 17 * 16; tensor[1] = 1;
        put32(tensor + 8, ctx->profile.logits_h | (ctx->profile.logits_w << 16));
        put32(tensor + 12, ctx->classes);
        for (u32 i = 0; i < 16; ++i) {
            param[2912 + i * 4] = (u8)(i + 1);
            param[2912 + i * 4 + 1] = (u8)(i + 2);
            param[2912 + i * 4 + 2] = (u8)(i + 3);
            param[2912 + i * 4 + 3] = (u8)(i + 4);
        }
    }
    static int context(Int8NpuContext *ctx) {
        CHECK(int8_npu_init(ctx) == XST_SUCCESS);
        CHECK(!ctx->selected && !ctx->ready && !ctx->blocked);
        CHECK(int8_npu_select_runtime(ctx, 0, 2, 0, 0) == XST_SUCCESS);
        make_param(ctx); return 0;
    }
    static int header_decode(void) {
        Int8ParamBlobHeader h;
        for (u32 i = 0; i < 32; ++i) put32(param + i * 4, 0x12345600U + i);
        CHECK(int8_npu_read_param_header(param, 128, &h) == XST_SUCCESS);
        {
            const u32 fields[] = {h.magic, h.version, h.tensor_desc_count, h.scale_desc_count,
                h.conv_desc_count, h.affine_desc_count, h.add_desc_count, h.pool_desc_count,
                h.uop_count, h.exec_plan_count, h.tensor_desc_offset, h.scale_desc_offset,
                h.conv_desc_offset, h.affine_desc_offset, h.add_desc_offset, h.pool_desc_offset,
                h.uop_offset, h.weight_data_offset, h.conv_qparam_offset, h.affine_qparam_offset,
                h.add_qparam_offset, h.pool_qparam_offset, h.conv_exec_desc_offset,
                h.window_sched_offset, h.window_cmd_offset, h.row_consumer_offset,
                h.fixed_exec_offset, h.exec_plan_offset, h.window_sched_count,
                h.window_cmd_count, h.block5_sched_offset, h.row_consumer_count};
            for (u32 i = 0; i < 32; ++i) CHECK(fields[i] == 0x12345600U + i);
        }
        CHECK(int8_npu_read_param_header(NULL, 128, &h) != XST_SUCCESS);
        CHECK(int8_npu_read_param_header(param, 128, NULL) != XST_SUCCESS);
        CHECK(int8_npu_read_param_header(param, 127, &h) != XST_SUCCESS);
        return 0;
    }
    static int header_reject(void) {
        Int8NpuContext ctx; Int8ParamBlobHeader h, bad;
        CHECK(!context(&ctx));
        CHECK(int8_npu_read_param_header(param, PARAM_BYTES, &h) == XST_SUCCESS);
        CHECK(int8_npu_validate_param_header(&h, PARAM_BYTES) == XST_SUCCESS);
        CHECK(int8_npu_validate_param_header(NULL, PARAM_BYTES) != XST_SUCCESS);
        CHECK(int8_npu_validate_param_header(&h, 262145) != XST_SUCCESS);
        CHECK(int8_npu_validate_param_header(&h, 0) != XST_SUCCESS);
        CHECK(int8_npu_validate_param_header(&h, PARAM_BYTES - 1) != XST_SUCCESS);
        for (unsigned i = 0; i < 12; ++i) {
            bad = h;
            switch (i) {
            case 0: bad.magic ^= 1U; break;
            case 1: bad.version = 3; break;
            case 2: bad.uop_count = 74; break;
            case 3: bad.uop_count = 76; break;
            case 4: bad.exec_plan_count = 0; break;
            case 5: bad.exec_plan_count = 15; break;
            case 6: bad.exec_plan_count = 17; break;
            case 7: bad.exec_plan_count = 97; break;
            case 8: bad.uop_offset = PARAM_BYTES + 1; break;
            case 9: bad.uop_offset = PARAM_BYTES - 1; break;
            case 10: bad.exec_plan_offset = PARAM_BYTES + 1; break;
            default: bad.exec_plan_offset = PARAM_BYTES - 1; break;
            }
            CHECK(int8_npu_validate_param_header(&bad, PARAM_BYTES) != XST_SUCCESS);
        }
        return 0;
    }
    static int geometry(void) {
        Int8NpuContext ctx; Int8ParamBlobHeader h, bad;
        CHECK(!context(&ctx));
        for (u32 pid = 0; pid < 8; ++pid) for (u32 classes = 2; classes <= 20; classes += 18) {
            CHECK(int8_npu_select_runtime(&ctx, pid, classes, 0, 0) == XST_SUCCESS);
            make_param(&ctx);
            CHECK(int8_npu_read_param_header(param, PARAM_BYTES, &h) == XST_SUCCESS);
            CHECK(int8_npu_validate_deployment_geometry(&ctx, param, PARAM_BYTES, &h) == XST_SUCCESS);
            {
                const unsigned offsets[] = {129, 136, 138, 140, 401, 408, 410, 412};
                for (unsigned i = 0; i < sizeof(offsets) / sizeof(offsets[0]); ++i) {
                    param[offsets[i]] ^= 1U;
                    CHECK(int8_npu_validate_deployment_geometry(&ctx, param, PARAM_BYTES, &h) != XST_SUCCESS);
                    param[offsets[i]] ^= 1U;
                }
            }
        }
        CHECK(int8_npu_validate_deployment_geometry(NULL, param, PARAM_BYTES, &h) != XST_SUCCESS);
        CHECK(int8_npu_validate_deployment_geometry(&ctx, NULL, PARAM_BYTES, &h) != XST_SUCCESS);
        CHECK(int8_npu_validate_deployment_geometry(&ctx, param, PARAM_BYTES, NULL) != XST_SUCCESS);
        CHECK(int8_npu_validate_deployment_geometry(&ctx, param, 431, &h) != XST_SUCCESS);
        bad = h; bad.tensor_desc_count = 18;
        CHECK(int8_npu_validate_deployment_geometry(&ctx, param, PARAM_BYTES, &bad) != XST_SUCCESS);
        bad.tensor_desc_count = 20;
        CHECK(int8_npu_validate_deployment_geometry(&ctx, param, PARAM_BYTES, &bad) != XST_SUCCESS);
        bad = h; bad.tensor_desc_offset = PARAM_BYTES + 1;
        CHECK(int8_npu_validate_deployment_geometry(&ctx, param, PARAM_BYTES, &bad) != XST_SUCCESS);
        ctx.selected = 0;
        CHECK(int8_npu_validate_deployment_geometry(&ctx, param, PARAM_BYTES, &h) != XST_SUCCESS);
        return 0;
    }
    static int selection(void) {
        Int8NpuContext ctx;
        const u32 invalid[][4] = {{8,2,0,0}, {0xFFFFFFFFU,2,0,0}, {0,0,0,0},
            {0,1,0,0}, {0,19,0,0}, {0,21,0,0}, {0,2,7,0}, {0,2,4,0},
            {1,20,1,0}, {2,2,1,0}, {0,2,0,1}, {0,2,1,2}};
        CHECK(int8_npu_init(NULL) != XST_SUCCESS);
        CHECK(!context(&ctx));
        CHECK(int8_npu_select_runtime(NULL, 0, 2, 0, 0) != XST_SUCCESS);
        for (unsigned i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
            CHECK(int8_npu_select_runtime(&ctx, 0, 2, 0, 0) == XST_SUCCESS);
            ctx.ready = 1;
            CHECK(int8_npu_select_runtime(&ctx, invalid[i][0], invalid[i][1],
                                          invalid[i][2], invalid[i][3]) != XST_SUCCESS);
            CHECK(!ctx.selected && !ctx.ready && !ctx.blocked);
        }
        CHECK(!mock_bsp.writes && !mock_bsp.starts);
        mock_bsp.config_result = 19;
        CHECK(int8_npu_init(&ctx) == 19);
        return 0;
    }
    static int sequence(void) {
        Int8NpuContext ctx;
        const u32 cases[][4] = {{0,2,0,0}, {3,2,0,0}, {3,20,0,0}, {4,2,0,0},
            {0,2,2,1}, {7,20,0,0}, {1,20,6,1}, {0,2,0,0}};
        CHECK(!context(&ctx));
        for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
            unsigned writes = mock_bsp.writes;
            CHECK(int8_npu_select_runtime(&ctx, cases[i][0], cases[i][1], cases[i][2], cases[i][3]) == XST_SUCCESS);
            CHECK(ctx.selected && !ctx.ready && !ctx.blocked && ctx.classes == cases[i][1]);
            CHECK(int8_npu_run_infer(&ctx, (UINTPTR)input, (UINTPTR)output, (UINTPTR)param, 75, 3) != XST_SUCCESS);
            CHECK(mock_bsp.writes == writes);
            CHECK(int8_npu_run_init(&ctx, (UINTPTR)param, 75, 3) == XST_SUCCESS);
            CHECK(ctx.ready && mock_bsp.mode == 1 && !mock_bsp.input && !mock_bsp.output);
            CHECK(mock_bsp.profile == cases[i][0] && mock_bsp.source == cases[i][2] && mock_bsp.mask == cases[i][3]);
            CHECK(int8_npu_runtime_abi(&ctx) == 0x11160001U);
            CHECK(int8_npu_runtime_capability(&ctx) == 0x7FFFU);
            CHECK(int8_npu_runtime_status(&ctx) == ((cases[i][0] << 16) | 1U));
            CHECK(int8_npu_run_infer(&ctx, (UINTPTR)input, (UINTPTR)output, (UINTPTR)param, 75, 3) == XST_SUCCESS);
            CHECK(ctx.ready && mock_bsp.mode == 2 && mock_bsp.uops == 75);
            CHECK(mock_bsp.input == (UINTPTR)input && mock_bsp.output == (UINTPTR)output && mock_bsp.param == (UINTPTR)param);
        }
        CHECK(mock_bsp.init_runs == 8 && mock_bsp.infer_runs == 8);
        return 0;
    }
    static int arguments(void) {
        Int8NpuContext ctx; unsigned writes;
        CHECK(!context(&ctx)); writes = mock_bsp.writes;
        CHECK(int8_npu_run_init(NULL, (UINTPTR)param, 75, 3) != XST_SUCCESS);
        CHECK(int8_npu_run_init(&ctx, 0, 75, 3) != XST_SUCCESS);
        CHECK(int8_npu_run_init(&ctx, (UINTPTR)param, 74, 3) != XST_SUCCESS);
        ctx.selected = 0;
        CHECK(int8_npu_run_init(&ctx, (UINTPTR)param, 75, 3) != XST_SUCCESS);
        ctx.selected = 1; mock_bsp.hardware_ready = 0;
        CHECK(int8_npu_run_init(&ctx, (UINTPTR)param, 75, 3) != XST_SUCCESS);
        CHECK(mock_bsp.writes == writes); mock_bsp.hardware_ready = 1;
        CHECK(int8_npu_run_init(&ctx, (UINTPTR)param, 75, 3) == XST_SUCCESS);
        writes = mock_bsp.writes;
        CHECK(int8_npu_run_infer(NULL, (UINTPTR)input, (UINTPTR)output, (UINTPTR)param, 75, 3) != XST_SUCCESS);
        CHECK(int8_npu_run_infer(&ctx, 0, (UINTPTR)output, (UINTPTR)param, 75, 3) != XST_SUCCESS);
        CHECK(int8_npu_run_infer(&ctx, (UINTPTR)input, 0, (UINTPTR)param, 75, 3) != XST_SUCCESS);
        CHECK(int8_npu_run_infer(&ctx, (UINTPTR)input, (UINTPTR)output, 0, 75, 3) != XST_SUCCESS);
        CHECK(int8_npu_run_infer(&ctx, (UINTPTR)input, (UINTPTR)output, (UINTPTR)param, 76, 3) != XST_SUCCESS);
        mock_bsp.hardware_ready = 0;
        CHECK(int8_npu_run_infer(&ctx, (UINTPTR)input, (UINTPTR)output, (UINTPTR)param, 75, 3) != XST_SUCCESS);
        CHECK(mock_bsp.writes == writes);
        return 0;
    }
    static int registers(void) {
        Int8NpuContext ctx;
        CHECK(!context(&ctx));
        mock_bsp.abi = 0x12345678U; mock_bsp.status = 0x76543210U; mock_bsp.capability = 0xA5A5A5A5U;
        CHECK(int8_npu_runtime_abi(&ctx) == 0x12345678U);
        CHECK(int8_npu_runtime_status(&ctx) == 0x76543210U);
        CHECK(int8_npu_runtime_capability(&ctx) == 0xA5A5A5A5U);
        CHECK(!mock_bsp.writes);
        return 0;
    }
    static int completion(const char *fault, int timeout) {
        Int8NpuContext ctx; unsigned polls, writes;
        int infer = strstr(fault, "_infer") != NULL;
        CHECK(!context(&ctx));
        if (infer) CHECK(int8_npu_run_init(&ctx, (UINTPTR)param, 75, 3) == XST_SUCCESS);
        polls = mock_bsp.polls;
        if (infer) CHECK(int8_npu_run_infer(&ctx, (UINTPTR)input, (UINTPTR)output, (UINTPTR)param, 75, 3) != XST_SUCCESS);
        else CHECK(int8_npu_run_init(&ctx, (UINTPTR)param, 75, 3) != XST_SUCCESS);
        CHECK(!ctx.ready && ctx.blocked == (unsigned)timeout);
        CHECK(mock_bsp.polls - polls == (timeout ? 3U : 1U));
        CHECK(mock_bsp.starts == (infer ? 2U : 1U));
        writes = mock_bsp.writes; polls = mock_bsp.polls;
        mock_bsp.fault = "none"; mock_bsp.completion_disabled = 0;
        CHECK(int8_npu_run_infer(&ctx, (UINTPTR)input, (UINTPTR)output, (UINTPTR)param, 75, 3) != XST_SUCCESS);
        if (timeout) {
            CHECK(int8_npu_select_runtime(&ctx, 4, 2, 0, 0) != XST_SUCCESS);
            CHECK(int8_npu_run_init(&ctx, (UINTPTR)param, 75, 3) != XST_SUCCESS);
        }
        CHECK(mock_bsp.writes == writes && mock_bsp.polls == polls);
        return 0;
    }
    int main(int argc, char **argv) {
        int result;
        if (argc < 2) return 2;
        mock_reset(argc == 3 ? argv[2] : "none");
        if (!strcmp(argv[1], "header_decode")) result = header_decode();
        else if (!strcmp(argv[1], "header_reject")) result = header_reject();
        else if (!strcmp(argv[1], "geometry")) result = geometry();
        else if (!strcmp(argv[1], "selection")) result = selection();
        else if (!strcmp(argv[1], "sequence")) result = sequence();
        else if (!strcmp(argv[1], "arguments")) result = arguments();
        else if (!strcmp(argv[1], "registers")) result = registers();
        else if (argc == 3 && !strcmp(argv[1], "completion")) result = completion(argv[2], 0);
        else if (argc == 3 && !strcmp(argv[1], "timeout")) result = completion(argv[2], 1);
        else return 2;
        if (mock_bsp.bad_mmio || mock_bsp.protocol_errors) {
            fprintf(stderr, "bad MMIO=%u protocol errors=%u\n", mock_bsp.bad_mmio, mock_bsp.protocol_errors);
            return 1;
        }
        return result;
    }
"""


APP_RUNNER = r"""
    #include "mock_bsp.h"
    int esp_int8_app_main(void);
    int main(int argc, char **argv) {
        int result;
        if (argc != 4) return 2;
        mock_reset(argv[1]); mock_bsp.root = argv[2]; mock_bsp.catalog = argv[3];
        result = esp_int8_app_main(); mock_report(); return result;
    }
"""


SD_HARNESS = r"""
    #include <stdio.h>
    #include <string.h>
    #include <stdarg.h>
    #include "ff.h"
    #include "drivers/sd_card.h"
    static const char *fault;
    static unsigned closed, synced;
    int xil_printf(const char *format, ...) { (void)format; return 0; }
    FRESULT f_mount(FATFS *fs,const char *path,int immediate) {
        (void)fs; (void)path; (void)immediate; return FR_OK;
    }
    FRESULT f_mkdir(const char *path) { (void)path; return FR_OK; }
    FRESULT f_open(FIL *file,const char *path,int mode) {
        (void)path; (void)mode; file->size=4; return FR_OK;
    }
    FRESULT f_read(FIL *file,void *dst,UINT count,UINT *read) {
        (void)file; memset(dst,0,count); *read=count; return FR_OK;
    }
    FRESULT f_write(FIL *file,const void *src,UINT count,UINT *written) {
        (void)file; (void)src; *written=count;
        if (!strcmp(fault,"short")) --*written;
        return !strcmp(fault,"write");
    }
    FRESULT f_sync(FIL *file) { (void)file; ++synced; return !strcmp(fault,"sync"); }
    FRESULT f_close(FIL *file) { (void)file; ++closed; return !strcmp(fault,"close"); }
    int main(int argc,char **argv) {
        unsigned char bytes[4]; u32 size=0;
        if (argc!=3) return 2;
        fault=argv[2];
        if (SD_Init()) return 3;
        int status=!strcmp(argv[1],"read")?
            SD_LoadFileToMemory("CASE/I0000.BIN",(UINTPTR)bytes,sizeof(bytes),&size):
            SD_SaveMemoryToFile("CASE/O0000.BIN",bytes,sizeof(bytes));
        printf("status=%d,closed=%u,synced=%u,size=%u\n",status,closed,synced,size);
        return 0;
    }
"""


LOG_HARNESS = r"""
    #include <stdarg.h>
    #include <stdio.h>
    #include <string.h>
    #include "ff.h"
    #include "app_log.h"
    #include "drivers/sd_card.h"
    static const char *fault;
    static unsigned used,writes,syncs,closes,opened;
    static char recorded[80000];
    int xil_printf(const char *format,...) {
        va_list args; va_start(args,format); int n=vprintf(format,args); va_end(args); return n;
    }
    FRESULT f_mount(FATFS *fs,const char *path,int immediate) {
        (void)fs; (void)path; (void)immediate; return FR_OK;
    }
    FRESULT f_mkdir(const char *path) { (void)path; return !strcmp(fault,"mkdir")?1:FR_EXIST; }
    FRESULT f_open(FIL *file,const char *path,int mode) {
        if (mode!=(FA_CREATE_NEW|FA_WRITE)) return 1;
        if (!strcmp(fault,"open")) return 1;
        if (strstr(path,"TEST0000.TXT")) return FR_EXIST;
        file->is_log=1; file->size=0; ++opened; return FR_OK;
    }
    FRESULT f_read(FIL *file,void *dst,UINT count,UINT *read) {
        (void)file; (void)dst; (void)count; (void)read; return 1;
    }
    FRESULT f_write(FIL *file,const void *src,UINT count,UINT *written) {
        (void)file; ++writes; *written=count;
        if (!strcmp(fault,"short")) --*written;
        if (count>sizeof(recorded)-used) return 1;
        memcpy(recorded+used,src,*written); used+=*written;
        return !strcmp(fault,"write");
    }
    FRESULT f_sync(FIL *file) { (void)file; ++syncs; return !strcmp(fault,"sync"); }
    FRESULT f_close(FIL *file) { (void)file; ++closes; return !strcmp(fault,"close"); }
    int main(int argc,char **argv) {
        if (argc!=2) return 2;
        fault=argv[1];
        app_log_printf("boot before mount %llu %08X %%\r\n",4294967297ULL,0xABCDEF01U);
        int status=SD_Init() || app_log_open();
        unsigned before=writes;
        app_log_printf("inference diagnostic: no SD writes here\r\n");
        unsigned after=writes;
        if (!strcmp(fault,"format")) {
            char long_line[2000]; memset(long_line,'X',sizeof(long_line)-1); long_line[sizeof(long_line)-1]=0;
            app_log_printf("%s",long_line);
        }
        if (!strcmp(fault,"overflow")) for (unsigned i=0;i<3000;++i) app_log_printf("unflushed line %08u\r\n",i);
        if (app_log_flush()) status=1;
        if (app_log_close()) status=1;
        printf("\nHOST_LOG_STATUS,%d,deferred=%u,opened=%u,closed=%u,syncs=%u\n",
               status,before==after,opened,closes,syncs);
        puts("HOST_DATA_BEGIN"); fwrite(recorded,1,used,stdout); puts("HOST_DATA_END");
        return 0;
    }
"""


def catalog_rows():
    return list(csv.DictReader(io.StringIO((ARTIFACTS / "RUNS.CSV").read_text(encoding="ascii"))))


def native_shape(profile):
    return ((240, 240, 1), (240, 240, 3), (256, 512, 1), (256, 512, 3),
            (512, 512, 1), (512, 512, 3), (512, 1024, 1), (512, 1024, 3))[profile]


def io_shape(profile, source, mask):
    height, width, channels = native_shape(profile)
    if source:
        height, width = {1: (256, 512), 2: (512, 1024), 3: (512, 512),
                         4: (256, 512), 5: (512, 1024), 6: (512, 512)}[source]
    return height, width, channels, (min(height, width) ** 2 if mask else
                                     native_shape(profile)[0] * native_shape(profile)[1])


class CompiledHostCase(unittest.TestCase):
    """Compile unchanged production translation units, never pre-include fixes."""

    KIND = None

    @classmethod
    def setUpClass(cls):
        if cls.KIND is None:
            return
        cc = os.environ.get("ESP_INT8_HOST_CC") or shutil.which("gcc")
        if not cc and Path("D:/mingw64/bin/gcc.exe").is_file():
            cc = "D:/mingw64/bin/gcc.exe"
        if not cc:
            raise unittest.SkipTest("MinGW gcc unavailable; set ESP_INT8_HOST_CC")
        cls.env = dict(os.environ)
        cls.env["PATH"] = str(Path(cc).resolve().parent) + os.pathsep + cls.env.get("PATH", "")
        target = subprocess.run([cc, "-dumpmachine"], capture_output=True, text=True,
                                timeout=15, env=cls.env, check=False)
        if target.returncode or "mingw" not in target.stdout.lower():
            raise unittest.SkipTest(f"MinGW gcc required, got {target.stdout.strip()!r}")
        cls.cc = cc
        cls.temp = tempfile.TemporaryDirectory(prefix="esp_int8_app_contract_")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        for name, content in HEADERS.items():
            (cls.directory / name).write_text(textwrap.dedent(content), encoding="ascii")
        cls.flags = ["-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                     "-I", str(cls.directory), "-I", str(APP)]
        cls.diagnostics = []
        cls.binary = cls.directory / (cls.KIND + ".exe")
        if cls.KIND == "catalog":
            source = cls.generated("catalog.c", CATALOG_HARNESS)
            cls.build([str(source), "-o", str(cls.binary)])
            return
        if cls.KIND == "sd":
            source = cls.generated("sd_test.c", SD_HARNESS)
            cls.build([str(source), str(APP / "drivers/sd_card.c"), str(APP / "app_log.c"), "-o", str(cls.binary)])
            return
        if cls.KIND == "log":
            source = cls.generated("log_test.c", LOG_HARNESS)
            if not (APP / "app_log.c").is_file():
                cls.diagnostics.append("missing shared App log implementation")
                return
            cls.build([str(source), str(APP / "drivers/sd_card.c"), str(APP / "app_log.c"), "-o", str(cls.binary)])
            return
        objects = []
        if cls.KIND == "app":
            objects.append(cls.object(APP / "main.c", "main.o",
                                      ["-DESP_INT8_APP_HOST_TEST", "-Dmain=esp_int8_app_main"]))
        objects.append(cls.object(APP / "hal" / "int8_npu.c", "hal.o"))
        objects.append(cls.object(APP / "app_log.c", "app_log.o"))
        objects.append(cls.object(cls.generated("mock_bsp.c", MOCK_BSP), "mock_bsp.o"))
        source = cls.generated("runner.c", APP_RUNNER if cls.KIND == "app" else HAL_HARNESS)
        objects.append(cls.object(source, "runner.o"))
        if not cls.diagnostics:
            cls.build([*map(str, objects), "-o", str(cls.binary)])

    @classmethod
    def generated(cls, filename, content):
        path = cls.directory / filename
        path.write_text(textwrap.dedent(content), encoding="ascii")
        return path

    @classmethod
    def build(cls, arguments):
        command = [cls.cc, *cls.flags, *arguments]
        result = subprocess.run(command, cwd=cls.directory, capture_output=True, text=True,
                                timeout=60, env=cls.env, check=False)
        if result.returncode or result.stdout or result.stderr:
            cls.diagnostics.append(subprocess.list2cmdline(command) + "\n" +
                                   result.stdout + result.stderr)

    @classmethod
    def object(cls, source, name, flags=()):
        path = cls.directory / name
        cls.build([*flags, "-c", str(source), "-o", str(path)])
        return path

    def assert_compiled(self):
        self.assertFalse(self.diagnostics, "Strict C compilation failed:\n" + "\n".join(self.diagnostics))
        self.assertTrue(self.binary.is_file())

    def run_host(self, *arguments, stdin="", expected=0, timeout=45):
        if self.diagnostics:
            self.skipTest("Strict compilation blocked execution; see test_00_strict_c_compilation")
        result = subprocess.run([str(self.binary), *map(str, arguments)], cwd=self.directory,
                                input=stdin, capture_output=True, text=True, timeout=timeout,
                                env=self.env, check=False)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        self.assertEqual(result.stderr, "", result.stderr)
        return result.stdout


class LogHostTests(CompiledHostCase):
    KIND = "log"

    def test_00_strict_c_compilation(self):
        self.assert_compiled()

    def test_real_logger_preserves_startup_and_creates_new_short_file_without_overwrite(self):
        stdout = self.run_host("none")
        self.assertIn("HOST_LOG_STATUS,0,deferred=1,opened=1,closed=1", stdout)
        before, recorded = stdout.split("HOST_DATA_BEGIN\n",1)
        recorded = recorded.split("HOST_DATA_END",1)[0]
        for text in ("boot before mount 4294967297 ABCDEF01 %", "SD: mounted at 1:/",
                     "APP: UART log=1:/LOGS/TEST0001.TXT", "inference diagnostic: no SD writes here"):
            self.assertIn(text, before)
            self.assertIn(text, recorded)

    def test_real_logger_detects_card_format_and_buffer_errors_without_silent_truncation(self):
        for fault in ("mkdir", "open", "write", "short", "sync", "close", "format", "overflow"):
            with self.subTest(fault=fault):
                stdout = self.run_host(fault)
                self.assertIn("HOST_LOG_STATUS,1,deferred=1", stdout)
                self.assertIn("APP: log", stdout)


class SdHostTests(CompiledHostCase):
    KIND = "sd"

    def test_00_strict_c_compilation(self):
        self.assert_compiled()

    def test_real_driver_rejects_write_sync_short_and_close_errors(self):
        self.assertIn("status=0,closed=1,synced=1", self.run_host("write", "none"))
        for fault in ("write", "sync", "short", "close"):
            with self.subTest(fault=fault):
                self.assertIn("status=1,closed=1", self.run_host("write", fault))
        self.assertIn("status=0,closed=1,synced=0,size=4", self.run_host("read", "none"))
        self.assertIn("status=1,closed=1,synced=0,size=0", self.run_host("read", "close"))


class CatalogHostTests(CompiledHostCase):
    KIND = "catalog"

    def test_00_strict_c_compilation(self):
        self.assert_compiled()

    def test_crc32_matches_independent_vectors(self):
        expected = ",".join(f"{zlib.crc32(data):08X}" for data in (b"", b"123456789", b"\0\x80\xff"))
        self.assertEqual(self.run_host("crc").strip(), expected)

    def test_published_ten_cases_parse_with_real_header(self):
        rows = catalog_rows()
        self.assertEqual(len(rows), 10)
        for row in rows:
            with self.subTest(key=row["key"]):
                h, w, c, output = io_shape(int(row["profile"]), int(row["source"]), int(row["mask"]))
                expected = ["OK", row["key"], row["profile"], row["classes"], row["source"], row["mask"],
                            str(h * w * c), str(output), row["param_crc"], row["input_crc"], row["mask_crc"],
                            row["param"], row["input"], row["output"]]
                self.assertEqual(self.run_host("parse", ",".join(row.values())).strip(), ",".join(expected))

    def test_uint_parser_rejects_signs_whitespace_overflow_and_wrong_digits(self):
        for base, value, expected in ((10, "0", 0), (10, "4294967295", 4294967295),
                                      (16, "FFFFFFFF", 4294967295), (16, "ABCDEF01", 0xABCDEF01)):
            with self.subTest(base=base, value=value):
                self.assertEqual(self.run_host("uint", base, value).strip(), f"OK,{expected}")
        for base in (10, 16):
            invalid = ("", "+1", "-1", " 1", "1 ", "\t1", "1\r", "0x10", "abcdef", "1G", "1.0",
                       "4294967296" if base == 10 else "100000000", "9" * 100)
            for value in invalid:
                with self.subTest(base=base, value=value):
                    self.assertEqual(self.run_host("uint", base, value).strip(), "REJECT")

    def test_invalid_catalog_records_are_rejected(self):
        valid = list(catalog_rows()[0].values())
        mutations = {
            0: ("", "lower", "TOOLONG99", "../BAD", "BAD.KEY"),
            1: ("8", "4294967295", "-1", "1x", " 0"),
            2: ("0", "1", "3", "19", "21", "+2"),
            3: ("4", "7", "4294967295"),
            4: ("1", "2", "4294967295"),
            5: ("../P.BIN", "/P.BIN", "DIR/../P.BIN", "DIR/SUB/P.BIN", "DIR\\P.BIN",
                "DIR/p.bin", "LONGDIR99/P.BIN", "DIR/TOOLONG99.BIN", "DIR/P.LONG", "DIR/P", "DIR/.BIN"),
            8: ("", "-1", "0xAB", "abcdef01", "100000000", "ZZZZZZZZ"),
            9: ("", "-1"),
            10: ("", "100000000"),
        }
        for index, values in mutations.items():
            for value in values:
                fields = valid.copy(); fields[index] = value
                with self.subTest(field=index, value=value):
                    self.assertEqual(self.run_host("parse", ",".join(fields)).strip(), "REJECT")
        invalid = [",".join(valid[:-1]), ",".join(valid) + ",EXTRA", "A" * 256]
        for left, right in ((5, 6), (5, 7), (6, 7)):
            fields = valid.copy(); fields[right] = fields[left]; invalid.append(",".join(fields))
        for line in invalid:
            with self.subTest(line=line):
                self.assertEqual(self.run_host("parse", line).strip(), "REJECT")

    def test_paths_are_bound_to_key_and_fixed_role_names(self):
        valid = list(catalog_rows()[0].values())
        for index in (5, 6, 7):
            for path in ("OTHER/" + valid[index].split("/")[1], valid[0] + "/OTHER.BIN"):
                fields = valid.copy(); fields[index] = path
                with self.subTest(field=index, path=path):
                    self.assertEqual(self.run_host("parse", ",".join(fields)).strip(), "REJECT")

    def test_all_runtime_profiles_and_supported_io_modes(self):
        expected = []
        for profile in range(8):
            sources = [0, 1, 2, 3] if profile == 0 else [0, 4, 5, 6] if profile == 1 else [0]
            for source in sources:
                for mask in range(2 if source else 1):
                    h, w, c, output = io_shape(profile, source, mask)
                    top, left = 0, (w - min(h, w)) // 2 if source else 0
                    roi_h, roi_w = (min(h, w),) * 2 if source else (h, w)
                    expected.append(",".join(map(str, (profile, source, mask, h, w, c, h * w * c,
                                                       top, left, roi_h, roi_w, output))))
        self.assertEqual(self.run_host("lookup").splitlines(), expected)

    def test_validation_catalog_accepts_all_40_hardware_modes_and_rejects_wrong_paths(self):
        count = 0
        for profile in range(8):
            sources = [0, 1, 2, 3] if profile == 0 else [0, 4, 5, 6] if profile == 1 else [0]
            for classes in (2, 20):
                for source in sources:
                    for mask in range(2 if source else 1):
                        h, w, c, size = io_shape(profile, source, mask)
                        line = f"CASE,{profile},{classes},{source},{mask},CASE/PARAM.BIN,ABCDEF01,CASE/SAMPLES.CSV,12345678"
                        self.assertEqual(self.run_host("parseval", line).strip(),
                                         f"OK,{profile},{classes},{source},{mask},{h*w*c},{size}")
                        count += 1
        self.assertEqual(count, 40)
        valid = "CASE,0,2,0,0,CASE/PARAM.BIN,ABCDEF01,CASE/SAMPLES.CSV,12345678"
        for line in (valid.replace("CASE/SAMPLES.CSV", "OTHER/SAMPLES.CSV"),
                     valid.replace("CASE/SAMPLES.CSV", "../SAMPLES.CSV"),
                     valid+",EXTRA", valid.replace(",12345678", ",100000000")):
            self.assertEqual(self.run_host("parseval", line).strip(), "REJECT")


class HalHostTests(CompiledHostCase):
    KIND = "hal"

    def test_00_strict_c_compilation(self):
        self.assert_compiled()

    def test_param_header_decodes_all_little_endian_fields(self):
        self.run_host("header_decode")

    def test_param_header_rejects_version_counts_ranges_and_size(self):
        self.run_host("header_reject")

    def test_deployment_geometry_tracks_selected_profile_and_classes(self):
        self.run_host("geometry")

    def test_invalid_runtime_selections_clear_readiness_without_transactions(self):
        self.run_host("selection")

    def test_profile_switching_requires_init_and_programs_generated_registers(self):
        self.run_host("sequence")

    def test_invalid_run_arguments_and_unready_hardware_do_not_transmit(self):
        self.run_host("arguments")

    def test_runtime_api_reads_distinct_generated_registers(self):
        self.run_host("registers")

    def test_missing_generated_runtime_register_macros_are_compile_errors(self):
        names = ("MODEL_PROFILE_ID", "SRC_MODE", "MASK_MODE", "RUNTIME_ABI_ID",
                 "RUNTIME_STATUS", "RUNTIME_CAPABILITY")
        for name in names:
            macro = f"XESPNET_ENCODER_INT8_CORE_CONTROL_ADDR_{name}_DATA"
            with self.subTest(macro=macro), tempfile.TemporaryDirectory(dir=self.directory) as temp:
                directory = Path(temp)
                header = "\n".join(line for line in HEADERS["xespnet_encoder_int8_core_hw.h"].splitlines()
                                   if macro not in line)
                (directory / "xespnet_encoder_int8_core_hw.h").write_text(header, encoding="ascii")
                result = subprocess.run(
                    [self.cc, "-std=c11", "-Wall", "-Wextra", "-Werror",
                     "-I", str(directory), "-I", str(self.directory), "-I", str(APP),
                     "-c", str(APP / "hal" / "int8_npu.c"), "-o", str(directory / "hal.o")],
                    capture_output=True, text=True, timeout=30, env=self.env, check=False)
                self.assertNotEqual(result.returncode, 0, f"Missing {macro} compiled: guessed offset fallback?")
                self.assertIn(macro, result.stderr)

    def test_completion_checks_exact_abi_capability_and_profile_status(self):
        for phase in ("init", "infer"):
            for fault in ("bad_abi", "bad_capability", "extra_capability", "bad_status",
                          "wrong_profile", "error_status"):
                with self.subTest(phase=phase, fault=fault):
                    self.run_host("completion", f"{fault}_{phase}")

    def test_timeout_blocks_selection_init_and_inference_without_more_tx(self):
        for phase in ("init", "infer"):
            with self.subTest(phase=phase):
                self.run_host("timeout", f"timeout_{phase}")


class AppHostTests(CompiledHostCase):
    KIND = "app"

    def app(self, commands, fault="none", catalog=None, expected=0):
        with tempfile.TemporaryDirectory(prefix="sd_", dir=self.directory) as temp:
            path = ARTIFACTS / "RUNS.CSV"
            if catalog is not None:
                path = Path(temp) / "RUNS.CSV"
                path.write_bytes(catalog)
            stdout = self.run_host(fault, ARTIFACTS, path, stdin=commands, expected=expected)
        summary = [line for line in stdout.splitlines() if line.startswith("HOST_SUMMARY,")]
        self.assertEqual(len(summary), 1, stdout)
        counts = {key: int(value) for key, value in
                  (field.split("=") for field in summary[0].split(",")[1:])}
        self.assertEqual(counts["bad_mmio"], 0, stdout)
        self.assertEqual(counts["protocol_errors"], 0, stdout)
        return stdout, counts

    @staticmethod
    def catalog_bytes(rows):
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=CATALOG_HEADER.split(","), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
        return stream.getvalue().encode("ascii")

    def assert_successes(self, stdout, counts, keys):
        results = [line.split(",") for line in stdout.splitlines() if line.startswith("RESULT,")]
        self.assertEqual([row[1] for row in results], keys, stdout)
        self.assertTrue(all(row[2] == "PASS" for row in results), stdout)
        self.assertTrue(all("arm_ticks=" in row[3] and "arm_us=" in row[4] and
                            row[5] == "rtl_cycles=100" for row in results), stdout)
        for name in ("init", "infer", "saves", "invalidates"):
            self.assertEqual(counts[name], len(keys), stdout)
        self.assertEqual(counts["loads"], 1 + 2 * len(keys), stdout)
        self.assertEqual(counts["flushes"], 3 * len(keys), stdout)
        rows = {row["key"]: row for row in catalog_rows()}
        saves = [line for line in stdout.splitlines() if line.startswith("HOST_SAVE,")]
        self.assertEqual(saves, [f"HOST_SAVE,{rows[key]['output']},"
                                f"{io_shape(int(rows[key]['profile']), int(rows[key]['source']), int(rows[key]['mask']))[3]},"
                                f"{rows[key]['mask_crc']}" for key in keys])
        transactions = [line for line in stdout.splitlines() if line.startswith("HOST_TX,")]
        self.assertEqual(transactions, [f"HOST_TX,{mode},{rows[key]['profile']},"
                                       f"{rows[key]['source']},{rows[key]['mask']}"
                                       for key in keys for mode in (1, 2)])

    def assert_rejected(self, stdout, counts, init, infer):
        self.assertNotIn("RESULT,", stdout)
        self.assertEqual((counts["init"], counts["infer"], counts["saves"]), (init, infer, 0), stdout)

    def test_00_strict_c_compilation(self):
        self.assert_compiled()

    def test_app_creates_one_log_and_mirrors_hal_counter_and_result_output(self):
        stdout, counts = self.app("run G240B2\nquit\n", "capture_log")
        self.assertIn("APP: UART log=", stdout)
        self.assertIn("HOST_LOG_STATE,path=LOGS/TEST0000.TXT,closed=1", stdout)
        recorded = stdout.split("HOST_LOG_BEGIN\n", 1)[1].split("HOST_LOG_END", 1)[0]
        for text in ("ESP INT8 app", "APP: timer selftest", "NPU: param ok",
                     "RTL_STAGE_CYCLES_BEGIN", "RTL_CONV_CHECK", "RESULT,G240B2,PASS"):
            self.assertIn(text, recorded)
        self.assert_successes(stdout.split("HOST_LOG_BEGIN", 1)[0], counts, ["G240B2"])

    def test_log_open_and_initial_write_errors_prevent_hardware_start(self):
        for fault in ("log_open", "log_write", "log_sync", "log_close"):
            with self.subTest(fault=fault):
                stdout, counts = self.app("quit\n", fault, expected=1)
                self.assertIn("log", stdout)
                self.assertEqual((counts["init"], counts["infer"]), (0, 0))

    def test_uart_list_help_quit_and_eof_never_start_hardware(self):
        stdout, counts = self.app("help\r\nlist\r\nquit\r\n")
        self.assertIn("only registered RUNS.CSV entries", stdout)
        for row in catalog_rows():
            h, w, c, output = io_shape(int(row["profile"]), int(row["source"]), int(row["mask"]))
            self.assertIn(",".join([row["key"], row["profile"], row["classes"], row["source"], row["mask"],
                                    str(h * w * c), str(output)]) + "\n", stdout)
        self.assert_rejected(stdout, counts, 0, 0)
        self.assertEqual(counts["loads"], 1)
        for commands in ("", "\x04run G240B2\n"):
            with self.subTest(commands=commands):
                stdout, counts = self.app(commands)
                self.assert_rejected(stdout, counts, 0, 0)

    def test_uart_run_all_ten_registered_cases_with_profile_class_and_io_switching(self):
        keys = [row["key"] for row in catalog_rows()]
        stdout, counts = self.app("".join(f"run {key}\n" for key in keys) + "quit\n")
        self.assert_successes(stdout, counts, keys)

    def test_suite_default_and_repeat_boundaries_include_first_case_rerun(self):
        keys = [row["key"] for row in catalog_rows()]
        for command, repeats in (("suite 1", 1), ("suite", 3), ("suite 10", 10)):
            with self.subTest(command=command):
                stdout, counts = self.app(command + "\nquit\n")
                self.assertIn(f"SUITE,PASS,repeats={repeats},cases=10", stdout)
                self.assert_successes(stdout, counts, keys * repeats + keys[:1])

    def test_unknown_malformed_and_out_of_range_commands_do_not_transmit(self):
        commands = ("", "RUN G240B2", "boot", "run", "run UNKNOWN", "run ../BAD",
                    "run G240B2 extra", "list extra", "help extra", "quit extra",
                    "suite 0", "suite 11", "suite -1", "suite +1", "suite 1x",
                    "suite 4294967296", "suite 1 extra")
        stdout, counts = self.app("\n".join(commands) + "\nlist\nquit\n")
        self.assertIn("APP: unknown case", stdout)
        self.assertIn("APP: invalid command", stdout)
        self.assertIn("APP: invalid suite count", stdout)
        self.assertNotIn("SUITE,", stdout)
        self.assert_rejected(stdout, counts, 0, 0)
        self.assertEqual(counts["loads"], 1)

    def test_overlong_uart_command_is_drained_before_next_valid_command(self):
        stdout, counts = self.app("run G240B2" + " " * 100 + "\nrun G240B2\nquit\n")
        self.assertIn("APP: command too long", stdout)
        self.assert_successes(stdout, counts, ["G240B2"])

    def test_catalog_accepts_crlf_and_final_row_without_newline(self):
        raw = self.catalog_bytes(catalog_rows())
        for data in (raw.replace(b"\n", b"\r\n"), raw.rstrip(b"\n")):
            with self.subTest(crlf=b"\r" in data):
                stdout, counts = self.app("run G240B2\nquit\n", catalog=data)
                self.assert_successes(stdout, counts, ["G240B2"])

    def test_catalog_rejects_bad_header_nul_duplicates_blank_rows_and_overcapacity(self):
        raw = self.catalog_bytes(catalog_rows())
        first = raw.splitlines()[1]
        malformed = (b"", CATALOG_HEADER.encode("ascii"), CATALOG_HEADER.encode("ascii") + b"\n",
                     b"wrong\n" + first + b"\n", b"\xef\xbb\xbf" + raw,
                     raw + b"\0", raw + b"\n", raw + first + b"\n", b"X" * 8192)
        for index, data in enumerate(malformed):
            with self.subTest(index=index):
                stdout, counts = self.app("run G240B2\nquit\n", catalog=data, expected=1)
                self.assertIn("initialization/catalog rejected", stdout)
                self.assert_rejected(stdout, counts, 0, 0)
        rows = []
        for i in range(33):
            row = catalog_rows()[0].copy(); key = f"K{i:07d}"; row["key"] = key
            for role, filename in (("param", "PARAM.BIN"), ("input", "INPUTQ.BIN"), ("output", "OUT.BIN")):
                row[role] = f"{key}/{filename}"
            rows.append(row)
        stdout, counts = self.app("list\nquit\n", catalog=self.catalog_bytes(rows[:32]))
        self.assertIn("K0000031,0,2,0,0,57600,57600", stdout)
        self.assert_rejected(stdout, counts, 0, 0)
        stdout, counts = self.app("quit\n", catalog=self.catalog_bytes(rows), expected=1)
        self.assertIn("initialization/catalog rejected", stdout)
        self.assert_rejected(stdout, counts, 0, 0)

    def test_timer_and_sd_boot_errors_reject_before_transactions(self):
        for fault in ("timer", "sd_init"):
            with self.subTest(fault=fault):
                stdout, counts = self.app("run G240B2\nquit\n", fault, expected=1)
                self.assertIn("initialization/catalog rejected", stdout)
                self.assert_rejected(stdout, counts, 0, 0)
                self.assertEqual(counts["loads"], 0)

    def test_param_and_input_crc_and_sd_errors_reject_before_inference(self):
        for fault in ("param_crc", "empty_param", "param_io", "input_crc", "input_io"):
            with self.subTest(fault=fault):
                stdout, counts = self.app("run G240B2\nlist\nquit\n", fault)
                self.assertIn("APP: file size/CRC rejected", stdout)
                self.assertIn("key,profile,classes,source,mask,input_bytes,output_bytes", stdout)
                self.assert_rejected(stdout, counts, int(fault.startswith("input")), 0)

    def test_exact_input_size_is_checked_even_when_crc_matches(self):
        original = (ARTIFACTS / "G240B2" / "INPUTQ.BIN").read_bytes()
        for fault, data in (("short_input", original[:-1]), ("long_input", original + b"\0")):
            with self.subTest(fault=fault):
                rows = catalog_rows(); rows[0]["input_crc"] = f"{zlib.crc32(data):08X}"
                stdout, counts = self.app("run G240B2\nquit\n", fault, self.catalog_bytes(rows))
                self.assertIn("APP: file size/CRC rejected", stdout)
                self.assert_rejected(stdout, counts, 1, 0)

    def test_short_param_with_matching_crc_rejects_before_init(self):
        rows = catalog_rows()
        data = (ARTIFACTS / "G240B2" / "PARAM.BIN").read_bytes()[:127]
        rows[0]["param_crc"] = f"{zlib.crc32(data):08X}"
        stdout, counts = self.app("run G240B2\nquit\n", "short_param", self.catalog_bytes(rows))
        self.assertIn("APP: run FAIL", stdout)
        self.assertNotIn("APP: file size/CRC rejected", stdout)
        self.assert_rejected(stdout, counts, 0, 0)

    def test_invalid_mask_id_mask_crc_and_output_io_never_report_pass(self):
        for fault, message in (("mask_id", "invalid mask ID"), ("mask_crc", "mask CRC mismatch"),
                               ("save_io", "APP: run FAIL")):
            with self.subTest(fault=fault):
                stdout, counts = self.app("run G240B2\nquit\n", fault)
                self.assertIn(message, stdout)
                self.assert_rejected(stdout, counts, 1, 1)

    def test_missing_bad_and_error_stage_profiles_cannot_report_timing(self):
        for fault in ("no_conv_counter", "counter_zero", "counter_sum", "counter_error",
                      "counter_current_error", "conv_sum"):
            with self.subTest(fault=fault):
                stdout, counts = self.app("run G240B2\nquit\n", fault)
                self.assert_rejected(stdout, counts, 1, int(fault != "no_conv_counter"))
                self.assertIn("required counter unavailable" if fault == "no_conv_counter" else
                              "rejected timing", stdout)

    def test_runtime_completion_errors_never_save_or_report_success(self):
        for phase in ("init", "infer"):
            for fault in ("bad_abi", "bad_capability", "extra_capability", "bad_status",
                          "wrong_profile", "error_status"):
                with self.subTest(phase=phase, fault=fault):
                    stdout, counts = self.app("run G240B2\nquit\n", f"{fault}_{phase}")
                    self.assertIn("runtime ABI/capability/status rejected", stdout)
                    self.assert_rejected(stdout, counts, 1, int(phase == "infer"))

    def test_timeout_stops_uart_loop_and_all_further_transactions(self):
        for phase in ("init", "infer"):
            with self.subTest(phase=phase):
                stdout, counts = self.app("run G240B2\nrun R256B2\nsuite 1\nquit\n", f"timeout_{phase}")
                self.assertIn("no more transactions", stdout)
                self.assertNotIn("APP: case=R256B2", stdout)
                self.assertNotIn("SUITE,", stdout)
                self.assert_rejected(stdout, counts, 1, int(phase == "infer"))
                self.assertEqual(counts["polls"], 600000000 + int(phase == "infer"))

    def validation_fixture(self, directory, keys=("G240B2",), *, mutation=None):
        root = Path(directory)
        (root / "RUNS.CSV").write_bytes((ARTIFACTS / "RUNS.CSV").read_bytes())
        rows = {row["key"]: row for row in catalog_rows()}
        catalog = ["key,profile,classes,source,mask,param,param_crc,samples,samples_crc"]
        for key in keys:
            row = rows[key]
            folder = root / key
            folder.mkdir()
            data = "index,input_crc,target_crc\n" + "".join(
                f"{i},{row['input_crc']},{row['mask_crc']}\n" for i in range(500))
            if mutation:
                data = mutation(data)
            (folder / "SAMPLES.CSV").write_bytes(data.encode("ascii"))
            catalog.append(",".join([key, row["profile"], row["classes"], row["source"], row["mask"],
                                     row["param"], row["param_crc"], f"{key}/SAMPLES.CSV",
                                     f"{zlib.crc32(data.encode('ascii')):08X}"]))
        (root / "VALRUNS.CSV").write_text("\n".join(catalog)+"\n", encoding="ascii")
        return root / "RUNS.CSV"

    def test_validation_autoruns_500_per_case_init_once_and_dumps_first_only(self):
        with tempfile.TemporaryDirectory(dir=self.directory) as temp:
            path = self.validation_fixture(temp, ("G240B2", "R256B20", "G2S1M1"))
            stdout = self.run_host("none", ARTIFACTS, path, stdin="quit\n", timeout=90)
        self.assertIn("VALSUITE,PASS,cases=3,samples_per_case=500", stdout)
        self.assertEqual(stdout.count("RTL_STAGE_CYCLES_BEGIN"), 3)
        self.assertEqual(stdout.count("VAL_SAMPLE,"), 1500)
        self.assertIn("VAL_METRICS,G240B2,PA_ppm=1000000,mIoU_ppm=1000000", stdout)
        self.assertIn("HOST_SUMMARY,init=3,infer=1500", stdout)
        self.assertIn("HOST_SAVE,G2S1M1/O0499.BIN,", stdout)

    def test_validation_txt_contains_all_500_rows_and_final_accuracy_and_timing(self):
        with tempfile.TemporaryDirectory(dir=self.directory) as temp:
            path = self.validation_fixture(temp)
            stdout = self.run_host("capture_log", ARTIFACTS, path, stdin="quit\n", timeout=60)
        recorded = stdout.split("HOST_LOG_BEGIN\n",1)[1].split("HOST_LOG_END",1)[0]
        self.assertEqual(recorded.count("VAL_SAMPLE,"),500)
        self.assertEqual(recorded.count("RTL_STAGE_CYCLES_BEGIN"),1)
        for text in ("VAL_SAMPLE,G240B2,499,", "VAL_METRICS,G240B2,PA_ppm=1000000",
                     "VAL_SUMMARY,G240B2,PASS,samples=500", "VALSUITE,PASS,cases=1"):
            self.assertIn(text,recorded)

    def test_validation_rejects_missing_duplicate_and_reordered_samples_before_init(self):
        mutations = (lambda s: s.rsplit("\n", 2)[0]+"\n",
                     lambda s: s.replace("1,", "0,", 1),
                     lambda s: s.replace("0,", "1,", 1))
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(dir=self.directory) as temp:
                path = self.validation_fixture(temp, mutation=mutation)
                stdout = self.run_host("none", ARTIFACTS, path, stdin="quit\n", expected=1)
                self.assertIn("VALSUITE,FAIL", stdout)
                self.assertIn("HOST_SUMMARY,init=0,infer=0", stdout)

    def test_validation_runtime_io_and_class_errors_reject_full_dataset_success(self):
        for fault in ("input_crc", "error_status_infer", "mask_id", "save_late", "conv_sum", "log_late"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory(dir=self.directory) as temp:
                path = self.validation_fixture(temp)
                stdout = self.run_host(fault, ARTIFACTS, path, stdin="quit\n", expected=1)
                self.assertIn("VALSUITE,FAIL", stdout)
                self.assertNotIn("VAL_METRICS,", stdout)
                self.assertNotIn("VALSUITE,PASS", stdout)

    def test_validation_manifest_crc_cannot_be_silently_ignored(self):
        with tempfile.TemporaryDirectory(dir=self.directory) as temp:
            path = self.validation_fixture(temp)
            with (Path(temp)/"G240B2/SAMPLES.CSV").open("ab") as handle:
                handle.write(b"0,00000000,00000000\n")
            stdout = self.run_host("none", ARTIFACTS, path, stdin="quit\n", expected=1)
        self.assertIn("VALSUITE,FAIL", stdout)
        self.assertIn("HOST_SUMMARY,init=0,infer=0", stdout)

    def test_validation_metrics_use_gt_not_the_single_image_golden_crc(self):
        with tempfile.TemporaryDirectory(dir=self.directory) as temp:
            root = Path(temp)
            path = self.validation_fixture(root)
            target = bytearray((ARTIFACTS/"G240B2/MASK.BIN").read_bytes())
            target[0] ^= 1
            target[1] = 255
            (root/"G240B2/T0000.BIN").write_bytes(target)
            samples = root/"G240B2/SAMPLES.CSV"
            lines = samples.read_text().splitlines()
            fields = lines[1].split(",")
            fields[2] = f"{zlib.crc32(target):08X}"
            lines[1] = ",".join(fields)
            data = ("\n".join(lines)+"\n").encode("ascii")
            samples.write_bytes(data)
            catalog = root/"VALRUNS.CSV"
            lines = catalog.read_text().splitlines()
            fields = lines[1].split(",")
            fields[-1] = f"{zlib.crc32(data):08X}"
            lines[1] = ",".join(fields)
            catalog.write_text("\n".join(lines)+"\n", encoding="ascii")
            stdout = self.run_host("none", ARTIFACTS, path, stdin="quit\n", timeout=60)
        valid = 500*57600-1
        self.assertIn(f"VAL_METRICS,G240B2,PA_ppm={(valid-1)*1000000//valid}", stdout)
        self.assertIn(f"valid_pixels={valid}", stdout)
        self.assertIn("VALSUITE,PASS,cases=1,samples_per_case=500", stdout)


if __name__ == "__main__":
    unittest.main()
