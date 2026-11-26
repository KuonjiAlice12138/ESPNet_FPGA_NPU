#ifndef ESP_RUNTIME_CATALOG_H
#define ESP_RUNTIME_CATALOG_H

#include "runtime_profiles.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RUNTIME_CATALOG_HEADER "key,profile,classes,source,mask,param,input,output,param_crc,input_crc,mask_crc"
#define RUNTIME_MAX_CASES 32U
#define RUNTIME_LINE_BYTES 256U
#define RUNTIME_VAL_CATALOG_HEADER "key,profile,classes,source,mask,param,param_crc,samples,samples_crc"
#define RUNTIME_VAL_SAMPLE_HEADER "index,input_crc,target_crc"

typedef struct {
    char key[9], param[32], input[32], output[32];
    uint32_t profile, classes, source, mask, param_crc, input_crc, mask_crc;
    uint32_t input_bytes, output_bytes;
} runtime_case_t;

typedef struct {
    runtime_case_t run;
    char samples[32];
    uint32_t samples_crc;
} runtime_val_case_t;

typedef struct { uint32_t input_crc, target_crc; } runtime_val_sample_t;

static inline uint32_t runtime_crc32(const unsigned char *data, uint32_t bytes) {
    uint32_t crc=0xffffffffU;
    for (uint32_t i=0;i<bytes;++i) {
        crc^=data[i];
        for (unsigned bit=0;bit<8;++bit) crc=(crc>>1)^(0xedb88320U & (0U-(crc&1U)));
    }
    return ~crc;
}

static inline int runtime_short_component(const char *s, int extension) {
    unsigned base=0,ext=0; int dot=0;
    if (!s || !*s) return 0;
    for (;*s;++s) {
        if (*s=='.' && extension && !dot && base) { dot=1; continue; }
        if (!((*s>='A' && *s<='Z') || (*s>='0' && *s<='9') || *s=='_')) return 0;
        if (dot) ++ext; else ++base;
    }
    return base>0 && base<=8 && (extension ? dot && ext>0 && ext<=3 : !dot);
}

static inline int runtime_short_path(const char *s) {
    char dir[9]; const char *slash=strchr(s,'/');
    if (!slash || slash-s<1 || slash-s>8 || strchr(slash+1,'/')) return 0;
    memcpy(dir,s,(size_t)(slash-s)); dir[slash-s]=0;
    return runtime_short_component(dir,0) && runtime_short_component(slash+1,1);
}

static inline int runtime_parse_uint(const char *s, unsigned base, uint32_t *value) {
    char *end; unsigned long long number;
    if (!s || !*s || *s=='-' || *s=='+' || *s==' ') return 0;
    for (const char *p=s;*p;++p)
        if (!((*p>='0' && *p<='9') || (base==16 && *p>='A' && *p<='F'))) return 0;
    errno=0; number=strtoull(s,&end,base);
    if (errno || *end || number>0xffffffffULL) return 0;
    *value=(uint32_t)number; return 1;
}

static inline int runtime_parse_case(const char *line, runtime_case_t *item) {
    char copy[RUNTIME_LINE_BYTES],*field[11],*next; unsigned count=0;
    runtime_io_t io;
    if (!line || !item || strlen(line)>=sizeof(copy)) return 0;
    strcpy(copy,line); next=copy;
    for (;;) {
        if (count==11) return 0;
        field[count++]=next;
        next=strchr(next,','); if (!next) break;
        *next++=0;
    }
    if (count!=11 || !runtime_short_component(field[0],0) ||
        !runtime_short_path(field[5]) || !runtime_short_path(field[6]) ||
        !runtime_short_path(field[7]) || !strcmp(field[5],field[6]) ||
        !strcmp(field[5],field[7]) || !strcmp(field[6],field[7])) return 0;
    memset(item,0,sizeof(*item));
    if (!runtime_parse_uint(field[1],10,&item->profile) ||
        !runtime_parse_uint(field[2],10,&item->classes) ||
        !runtime_parse_uint(field[3],10,&item->source) ||
        !runtime_parse_uint(field[4],10,&item->mask) ||
        !runtime_parse_uint(field[8],16,&item->param_crc) ||
        !runtime_parse_uint(field[9],16,&item->input_crc) ||
        !runtime_parse_uint(field[10],16,&item->mask_crc) ||
        (item->classes!=2 && item->classes!=20) ||
        !runtime_io_lookup(item->profile,item->source,item->mask,&io)) return 0;
    strcpy(item->key,field[0]); strcpy(item->param,field[5]);
    strcpy(item->input,field[6]); strcpy(item->output,field[7]);
    {
        char prefix[16];
        strcpy(prefix,item->key); strcat(prefix,"/");
        size_t n=strlen(prefix);
        if (strncmp(item->param,prefix,n) || strcmp(item->param+n,"PARAM.BIN") ||
            strncmp(item->input,prefix,n) || strcmp(item->input+n,"INPUTQ.BIN") ||
            strncmp(item->output,prefix,n) || strcmp(item->output+n,"OUT.BIN")) return 0;
    }
    item->input_bytes=io.input_bytes; item->output_bytes=io.mask_bytes;
    return 1;
}

static inline int runtime_split_fields(char *line, char **fields, unsigned count) {
    for (unsigned i=0;i<count;++i) {
        fields[i]=line;
        char *comma=strchr(line,',');
        if (i+1==count) return !comma && *line;
        if (!comma || comma==line) return 0;
        *comma=0; line=comma+1;
    }
    return 0;
}

static inline int runtime_parse_val_case(const char *line,runtime_val_case_t *item) {
    char copy[RUNTIME_LINE_BYTES],normal[RUNTIME_LINE_BYTES],expected[32],*f[9];
    if (!line || !item || strlen(line)>=sizeof(copy)) return 0;
    strcpy(copy,line);
    if (!runtime_split_fields(copy,f,9) || !runtime_short_component(f[0],0)) return 0;
    int n=snprintf(normal,sizeof(normal),"%s,%s,%s,%s,%s,%s,%s/INPUTQ.BIN,%s/OUT.BIN,%s,0,0",
                   f[0],f[1],f[2],f[3],f[4],f[5],f[0],f[0],f[6]);
    if (n<0 || (size_t)n>=sizeof(normal) || !runtime_parse_case(normal,&item->run) ||
        !runtime_parse_uint(f[8],16,&item->samples_crc)) return 0;
    (void)snprintf(expected,sizeof(expected),"%s/SAMPLES.CSV",item->run.key);
    if (strcmp(f[7],expected)) return 0;
    strcpy(item->samples,expected); return 1;
}

static inline int runtime_parse_val_sample(char *line,uint32_t expected,runtime_val_sample_t *sample) {
    char *f[3]; uint32_t index;
    return runtime_split_fields(line,f,3) && runtime_parse_uint(f[0],10,&index) && index==expected &&
           runtime_parse_uint(f[1],16,&sample->input_crc) && runtime_parse_uint(f[2],16,&sample->target_crc);
}

#endif
