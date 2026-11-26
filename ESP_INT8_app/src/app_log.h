#ifndef ESP_INT8_APP_LOG_H
#define ESP_INT8_APP_LOG_H

void app_log_printf(const char *format, ...);
int app_log_open(void);
int app_log_flush(void);
int app_log_close(void);
int app_log_ok(void);

#endif
