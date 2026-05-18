#include "./sd_card.h"

#include <stdio.h>
#include <string.h>

#include "ff.h"
#include "xil_printf.h"
#include "xstatus.h"

#define SD_PATH_BUF_BYTES 128U

static FATFS g_fatfs;
static int g_mounted = 0;
static char g_mount_path[8] = {0};

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
    (void)f_close(&file);
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
    (void)f_close(&file);

    if (fr != FR_OK || bytes_written != (UINT)byte_size) {
        xil_printf("SD: write failed %s, fr=%d written=%u size=%u\r\n",
                   filename, fr, (u32)bytes_written, byte_size);
        return XST_FAILURE;
    }

    xil_printf("SD: saved %s, bytes=%u\r\n", filename, byte_size);
    return XST_SUCCESS;
}
