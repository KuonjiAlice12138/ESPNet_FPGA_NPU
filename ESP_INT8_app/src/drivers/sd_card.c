#include "./sd_card.h"

#include <stdio.h>
#include <string.h>

#include "ff.h"
#include "xil_printf.h"
#include "xstatus.h"
#include "../app_log.h"
#define xil_printf app_log_printf

#define SD_PATH_BUF_BYTES 128U

static FATFS g_fatfs;
static int g_mounted = 0;
static char g_mount_path[8] = {0};
static FIL g_log_file;
static int g_log_open;

static int build_path(const char *filename, char *out_path, u32 out_size)
{
    int n;

    if (!g_mounted || filename == NULL || out_path == NULL || out_size == 0U) {
        return XST_FAILURE;
    }

    n = snprintf(out_path, out_size, "%s%s", g_mount_path, filename);
    if (n < 0 || (u32)n >= out_size) {
        xil_printf("SD: path too long: %s\r\n", filename);
        return XST_FAILURE;
    }

    return XST_SUCCESS;
}

int SD_Init(void)
{
    static const char *kMountCandidates[] = {"1:/", "0:/"};
    UINT i;

    if (g_mounted) {
        return XST_SUCCESS;
    }

    for (i = 0U; i < (UINT)(sizeof(kMountCandidates) / sizeof(kMountCandidates[0])); ++i) {
        FRESULT fr = f_mount(&g_fatfs, kMountCandidates[i], 1);
        if (fr == FR_OK) {
            strncpy(g_mount_path, kMountCandidates[i], sizeof(g_mount_path) - 1U);
            g_mount_path[sizeof(g_mount_path) - 1U] = '\0';
            g_mounted = 1;
            xil_printf("SD: mounted at %s\r\n", g_mount_path);
            return XST_SUCCESS;
        }
        xil_printf("SD: mount %s failed, fr=%d\r\n", kMountCandidates[i], fr);
    }

    return XST_FAILURE;
}

const char *SD_GetMountPath(void)
{
    return g_mount_path;
}

int SD_FileExists(const char *filename)
{
    char path[SD_PATH_BUF_BYTES];
    FIL file;
    FRESULT fr;

    if (build_path(filename, path, sizeof(path)) != XST_SUCCESS) {
        return 0;
    }

    fr = f_open(&file, path, FA_READ);
    if (fr != FR_OK) {
        return 0;
    }

    (void)f_close(&file);
    return 1;
}

int SD_LoadFileToMemory(const char *filename, UINTPTR dst, u32 max_bytes,
                         u32 *byte_size)
{
    char path[SD_PATH_BUF_BYTES];
    FIL file;
    FRESULT fr;
    UINT bytes_read = 0U;
    FSIZE_t file_size;

    if (byte_size != NULL) {
        *byte_size = 0U;
    }
    if (dst == (UINTPTR)0U || max_bytes == 0U) {
        return XST_FAILURE;
    }
    if (build_path(filename, path, sizeof(path)) != XST_SUCCESS) {
        return XST_FAILURE;
    }

    fr = f_open(&file, path, FA_READ);
    if (fr != FR_OK) {
        xil_printf("SD: open read failed %s, fr=%d\r\n", path, fr);
        return XST_FAILURE;
    }

    file_size = f_size(&file);
    if (file_size > (FSIZE_t)max_bytes) {
        xil_printf("SD: %s too large, size=%u max=%u\r\n", filename,
                   (u32)file_size, max_bytes);
        (void)f_close(&file);
        return XST_FAILURE;
    }

    fr = f_read(&file, (void *)dst, (UINT)file_size, &bytes_read);
    FRESULT close_result = f_close(&file);
    if (fr == FR_OK) fr = close_result;
    if (fr != FR_OK || bytes_read != (UINT)file_size) {
        xil_printf("SD: read failed %s, fr=%d read=%u size=%u\r\n", filename,
                   fr, (u32)bytes_read, (u32)file_size);
        return XST_FAILURE;
    }

    if (byte_size != NULL) {
        *byte_size = (u32)file_size;
    }
    xil_printf("SD: loaded %s, bytes=%u\r\n", filename, (u32)file_size);
    return XST_SUCCESS;
}

int SD_SaveMemoryToFile(const char *filename, const void *src, u32 byte_size)
{
    char path[SD_PATH_BUF_BYTES];
    FIL file;
    FRESULT fr;
    UINT bytes_written = 0U;

    if (src == NULL || byte_size == 0U) {
        return XST_FAILURE;
    }
    if (build_path(filename, path, sizeof(path)) != XST_SUCCESS) {
        return XST_FAILURE;
    }

    fr = f_open(&file, path, FA_CREATE_ALWAYS | FA_WRITE);
    if (fr != FR_OK) {
        xil_printf("SD: open write failed %s, fr=%d\r\n", path, fr);
        return XST_FAILURE;
    }

    fr = f_write(&file, src, (UINT)byte_size, &bytes_written);
    if (fr == FR_OK) {
        fr = f_sync(&file);
    }
    FRESULT close_result = f_close(&file);
    if (fr == FR_OK) fr = close_result;

    if (fr != FR_OK || bytes_written != (UINT)byte_size) {
        xil_printf("SD: write failed %s, fr=%d written=%u size=%u\r\n",
                   filename, fr, (u32)bytes_written, byte_size);
        return XST_FAILURE;
    }

    xil_printf("SD: saved %s, bytes=%u\r\n", filename, byte_size);
    return XST_SUCCESS;
}

int SD_LogOpen(char *relative_path, u32 capacity)
{
    char directory[SD_PATH_BUF_BYTES],path[SD_PATH_BUF_BYTES];
    if (!relative_path || capacity<18U || g_log_open ||
        build_path("LOGS",directory,sizeof(directory))!=XST_SUCCESS) return XST_FAILURE;
    FRESULT fr=f_mkdir(directory);
    if (fr!=FR_OK && fr!=FR_EXIST) return XST_FAILURE;
    for (unsigned i=0;i<10000U;++i) {
        (void)snprintf(relative_path,capacity,"LOGS/TEST%04u.TXT",i);
        if (build_path(relative_path,path,sizeof(path))!=XST_SUCCESS) return XST_FAILURE;
        fr=f_open(&g_log_file,path,FA_CREATE_NEW|FA_WRITE);
        if (fr==FR_EXIST) continue;
        if (fr!=FR_OK) return XST_FAILURE;
        g_log_open=1; return XST_SUCCESS;
    }
    return XST_FAILURE;
}

int SD_LogAppend(const void *data, u32 bytes)
{
    UINT written=0;
    if (!g_log_open || !data || !bytes) return XST_FAILURE;
    FRESULT fr=f_write(&g_log_file,data,(UINT)bytes,&written);
    return fr==FR_OK && written==bytes?XST_SUCCESS:XST_FAILURE;
}

int SD_LogSync(void)
{
    return g_log_open && f_sync(&g_log_file)==FR_OK?XST_SUCCESS:XST_FAILURE;
}

int SD_LogClose(void)
{
    if (!g_log_open) return XST_FAILURE;
    FRESULT fr=f_close(&g_log_file);
    g_log_open=0;
    return fr==FR_OK?XST_SUCCESS:XST_FAILURE;
}
