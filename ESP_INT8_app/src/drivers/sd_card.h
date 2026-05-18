#ifndef ESP_INT8_SD_CARD_H
#define ESP_INT8_SD_CARD_H

#include "xil_types.h"

int SD_Init(void);
const char *SD_GetMountPath(void);
int SD_FileExists(const char *filename);
int SD_LoadFileToMemory(const char *filename, UINTPTR dst, u32 max_bytes,
                         u32 *byte_size);
int SD_SaveMemoryToFile(const char *filename, const void *src, u32 byte_size);

#endif
