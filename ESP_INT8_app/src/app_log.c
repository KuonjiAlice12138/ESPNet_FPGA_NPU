#include "app_log.h"
#include "app_config.h"
#include "drivers/sd_card.h"
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include "xil_printf.h"
#include "xstatus.h"

static char g_log_buffer[INT8_APP_LOG_BUFFER_BYTES] __attribute__((aligned(64)));
static char g_log_path[32];
static u32 g_log_used;
static unsigned g_log_started, g_log_active, g_log_failed;

static void log_failure(const char *reason)
{
    if (!g_log_failed)
        xil_printf("APP: log %s; test stopped, TXT log may be incomplete\r\n",reason);
    g_log_failed=1;
}

void app_log_printf(const char *format, ...)
{
    char line[INT8_APP_LOG_LINE_BYTES];
    va_list args,copy;
    va_start(args,format);
    va_copy(copy,args);
    int count=vsnprintf(line,sizeof(line),format,args);
    va_end(args);
    if (count<0) log_failure("format failed");
    else if ((unsigned)count>=sizeof(line)) {
        (void)vprintf(format,copy);
        log_failure("line too long");
    } else {
        xil_printf("%s",line);
        if (!g_log_failed) {
            if ((u32)count>sizeof(g_log_buffer)-g_log_used) log_failure("buffer overflow");
            else {
                memcpy(g_log_buffer+g_log_used,line,(size_t)count);
                g_log_used+=(u32)count;
            }
        }
    }
    va_end(copy);
    /* Printing only queues bytes: no FatFs call may occur inside MODE_RUN timing. */
}

int app_log_ok(void) { return !g_log_failed; }

int app_log_flush(void)
{
    if (!g_log_active || g_log_failed) return XST_FAILURE;
    if (g_log_used && SD_LogAppend(g_log_buffer,g_log_used)!=XST_SUCCESS) {
        log_failure("write failed"); return XST_FAILURE;
    }
    g_log_used=0;
    if (SD_LogSync()!=XST_SUCCESS) {
        log_failure("sync failed"); return XST_FAILURE;
    }
    return XST_SUCCESS;
}

int app_log_open(void)
{
    if (g_log_started || g_log_failed) return XST_FAILURE;
    g_log_started=1;
    if (SD_LogOpen(g_log_path,sizeof(g_log_path))!=XST_SUCCESS) {
        log_failure("open failed"); return XST_FAILURE;
    }
    g_log_active=1;
    app_log_printf("APP: UART log=%s%s\r\n",SD_GetMountPath(),g_log_path);
    return app_log_flush();
}

int app_log_close(void)
{
    int status=g_log_failed?XST_FAILURE:XST_SUCCESS;
    if (g_log_active) {
        if (app_log_flush()!=XST_SUCCESS) status=XST_FAILURE;
        if (SD_LogClose()!=XST_SUCCESS) {
            log_failure("close failed"); status=XST_FAILURE;
        }
        g_log_active=0;
    }
    return status;
}
