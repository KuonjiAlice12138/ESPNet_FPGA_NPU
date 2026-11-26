#include <stdio.h>
#include <string.h>
#include "app_config.h"
#include "runtime_catalog.h"
#include "drivers/sd_card.h"
#include "hal/int8_npu.h"
#include "sleep.h"
#include "xil_cache.h"
#include "xil_printf.h"
#include "xparameters.h"
#include "xstatus.h"
#include "app_log.h"
#define xil_printf app_log_printf

extern char inbyte(void);
static u8 g_input[INT8_INPUT_BYTES] __attribute__((aligned(64)));
static u8 g_output[INT8_OUTPUT_BYTES] __attribute__((aligned(64)));
static u8 g_param[INT8_PARAM_HW_MAX_BYTES] __attribute__((aligned(64)));
static char g_catalog[INT8_APP_CATALOG_BYTES];
static runtime_case_t g_cases[RUNTIME_MAX_CASES];
static unsigned g_case_count;
static runtime_val_case_t g_val_cases[INT8_APP_VAL_MAX_CASES];
static unsigned g_val_case_count;
static char g_val_index[INT8_APP_VAL_INDEX_BYTES];
static runtime_val_sample_t g_val_samples[INT8_APP_VAL_SAMPLES];
static u8 g_target[INT8_OUTPUT_BYTES] __attribute__((aligned(64)));
static u64 g_confusion[20][20];

static u64 timer_counter(void) {
#ifdef ESP_INT8_APP_HOST_TEST
    extern u64 host_timer_counter(void); return host_timer_counter();
#else
    u64 value; asm volatile("isb; mrs %0, CNTPCT_EL0" : "=r"(value) :: "memory"); return value;
#endif
}
static u64 timer_frequency(void) {
#ifdef ESP_INT8_APP_HOST_TEST
    return 33333000ULL;
#else
    u64 value; asm volatile("mrs %0, CNTFRQ_EL0" : "=r"(value)); return value;
#endif
}
static int timer_selftest(void) {
    u64 hz=timer_frequency(),start=timer_counter(),ticks,expected,error;
    usleep(INT8_APP_TIMER_SELFTEST_US); ticks=timer_counter()-start;
    expected=(hz*INT8_APP_TIMER_SELFTEST_US+500000ULL)/1000000ULL;
    error=ticks>expected?ticks-expected:expected-ticks;
    xil_printf("APP: timer selftest hz=%llu ticks=%llu expected=%llu\r\n",hz,ticks,expected);
    return expected && error*1000000ULL/expected<=INT8_APP_TIMER_SELFTEST_MAX_ERR_PPM;
}

static int load_catalog(void) {
    u32 size; char *line,*end;
    if (SD_LoadFileToMemory(INT8_APP_CATALOG_FILE,(UINTPTR)g_catalog,sizeof(g_catalog)-1,&size)!=XST_SUCCESS) return 0;
    if (!size || memchr(g_catalog,0,size)) return 0;
    g_catalog[size]=0; line=g_catalog; g_case_count=0;
    end=strchr(line,'\n'); if (!end) return 0;
    *end=0; if (end>line && end[-1]=='\r') end[-1]=0;
    if (strcmp(line,RUNTIME_CATALOG_HEADER)) return 0;
    line=end+1;
    while (*line) {
        end=strchr(line,'\n'); if (end) *end=0;
        size_t length=strlen(line); if (length && line[length-1]=='\r') line[length-1]=0;
        if (!*line || g_case_count>=RUNTIME_MAX_CASES || !runtime_parse_case(line,&g_cases[g_case_count])) return 0;
        for (unsigned i=0;i<g_case_count;++i)
            if (!strcmp(g_cases[i].key,g_cases[g_case_count].key) ||
                !strcmp(g_cases[i].output,g_cases[g_case_count].output)) return 0;
        ++g_case_count;
        if (!end) break;
        line=end+1;
    }
    return g_case_count>0;
}

static int load_checked(const char *path,u8 *buffer,u32 capacity,u32 exact,u32 crc,u32 *size) {
    if (SD_LoadFileToMemory(path,(UINTPTR)buffer,capacity,size)!=XST_SUCCESS ||
        !*size || (exact && *size!=exact) || runtime_crc32(buffer,*size)!=crc) {
        xil_printf("APP: file size/CRC rejected: %s\r\n",path); return 0;
    }
    return 1;
}

static int profile_valid(void) {
    u64 total=stage_counter_read_total(),sum=0,conv=stage_counter_read_stage(5),w=0,s=0,p=0;
    if (!total || stage_counter_read_stage(12) || (stage_counter_read_current()&255U)==12U) return 0;
    for (unsigned i=0;i<INT8_STAGE_COUNTER_STAGE_COUNT;++i) sum+=stage_counter_read_stage(i);
    for (unsigned i=0;i<INT8_CONV_WIN_STATE_COUNT;++i) w+=stage_counter_read_conv_win(i);
    for (unsigned i=0;i<INT8_CONV_SA_STATE_COUNT;++i) s+=stage_counter_read_conv_sa(i);
    for (unsigned i=0;i<INT8_CONV_POST_STATE_COUNT;++i) p+=stage_counter_read_conv_post(i);
    return sum==stage_counter_read_active() && sum<=total && conv && w==conv && s==conv && p==conv;
}

static int prepare_case(Int8NpuContext *npu,const runtime_case_t *item,Int8ParamBlobHeader *header) {
    u32 param_size;
    if (int8_npu_select_runtime(npu,item->profile,item->classes,item->source,item->mask)!=XST_SUCCESS) return 0;
    xil_printf("APP: case=%s encoder=%ux%ux%u source=%ux%ux%u roi=%u,%u,%u,%u output=%ux%u classes=%u ABI=%08x\r\n",
        item->key,npu->profile.input_h,npu->profile.input_w,npu->profile.input_c,
        npu->io.source_h,npu->io.source_w,npu->io.source_c,npu->io.roi_top,npu->io.roi_left,
        npu->io.roi_h,npu->io.roi_w,npu->io.mask_h,npu->io.mask_w,item->classes,(u32)RUNTIME_ABI_ID);
    xil_printf("APP: identity param=%s CRC=%08x\r\n",item->param,item->param_crc);
    memset(g_param,0,sizeof(g_param));
    if (!load_checked(item->param,g_param,sizeof(g_param),0,item->param_crc,&param_size) ||
        int8_npu_read_param_header(g_param,param_size,header)!=XST_SUCCESS ||
        int8_npu_validate_param_header(header,param_size)!=XST_SUCCESS ||
        int8_npu_validate_deployment_geometry(npu,g_param,param_size,header)!=XST_SUCCESS) return 0;
    Xil_DCacheFlushRange((UINTPTR)g_param,(param_size+31U)&~31U);
    if (app_log_flush()!=XST_SUCCESS) return 0;
    return int8_npu_run_init(npu,(UINTPTR)g_param,header->uop_count,INT8_NPU_TIMEOUT_POLLS)==XST_SUCCESS;
}

static int infer_frame(Int8NpuContext *npu,const runtime_case_t *item,u32 uops,unsigned dump,u64 *ticks,u64 *rtl) {
    u64 start;
    memset(g_output,255,item->output_bytes);
    Xil_DCacheFlushRange((UINTPTR)g_input,item->input_bytes);
    Xil_DCacheFlushRange((UINTPTR)g_output,item->output_bytes);
    if (!stage_counter_has_conv_profile()) { xil_printf("APP: required counter unavailable\r\n"); return 0; }
    if (app_log_flush()!=XST_SUCCESS) return 0;
    stage_counter_clear(); stage_counter_freeze(0U); stage_counter_enable(1U);
    start=timer_counter();
    int status=int8_npu_run_infer(npu,(UINTPTR)g_input,(UINTPTR)g_output,(UINTPTR)g_param,
                                uops,INT8_NPU_TIMEOUT_POLLS);
    *ticks=timer_counter()-start;
    stage_counter_freeze(1U); stage_counter_enable(0U); *rtl=stage_counter_read_total();
    if (dump || status!=XST_SUCCESS) stage_counter_dump_csv();
    if (status!=XST_SUCCESS || !profile_valid()) {
        xil_printf("APP: rejected timing (runtime/counter/timeout error)\r\n"); return 0;
    }
    Xil_DCacheInvalidateRange((UINTPTR)g_output,item->output_bytes);
    for (u32 i=0;i<item->output_bytes;++i) if (g_output[i]>=item->classes) {
        xil_printf("APP: invalid mask ID at %u; timing rejected\r\n",i); return 0;
    }
    return app_log_flush()==XST_SUCCESS;
}

static int run_case(Int8NpuContext *npu,const runtime_case_t *item,unsigned dump) {
    Int8ParamBlobHeader header; u32 size; u64 ticks,rtl;
    xil_printf("APP: input=%s CRC=%08x mask_CRC=%08x\r\n",item->input,item->input_crc,item->mask_crc);
    if (!prepare_case(npu,item,&header) ||
        !load_checked(item->input,g_input,sizeof(g_input),item->input_bytes,item->input_crc,&size) ||
        !infer_frame(npu,item,header.uop_count,dump,&ticks,&rtl)) return 0;
    if (runtime_crc32(g_output,item->output_bytes)!=item->mask_crc) {
        xil_printf("APP: mask CRC mismatch; timing rejected\r\n"); return 0;
    }
    if (SD_SaveMemoryToFile(item->output,g_output,item->output_bytes)!=XST_SUCCESS) return 0;
    xil_printf("RESULT,%s,PASS,arm_ticks=%llu,arm_us=%llu,rtl_cycles=%llu,output=%s\r\n",
        item->key,ticks,(ticks*1000000ULL+timer_frequency()/2)/timer_frequency(),rtl,item->output);
    return app_log_flush()==XST_SUCCESS;
}

static char *csv_line(char **cursor) {
    char *line=*cursor,*end;
    if (!*line) return NULL;
    end=strchr(line,'\n');
    if (end) { *end=0; *cursor=end+1; } else *cursor=line+strlen(line);
    size_t length=strlen(line);
    if (length && line[length-1]=='\r') line[length-1]=0;
    return line;
}

static int load_val_catalog(void) {
    u32 bytes; char *cursor=g_catalog,*line;
    g_val_case_count=0;
    if (SD_LoadFileToMemory(INT8_APP_VAL_CATALOG_FILE,(UINTPTR)g_catalog,sizeof(g_catalog)-1,&bytes)!=XST_SUCCESS ||
        !bytes || memchr(g_catalog,0,bytes)) return 0;
    g_catalog[bytes]=0;
    line=csv_line(&cursor);
    if (!line || strcmp(line,RUNTIME_VAL_CATALOG_HEADER)) return 0;
    while ((line=csv_line(&cursor))) {
        if (!*line || g_val_case_count>=INT8_APP_VAL_MAX_CASES ||
            !runtime_parse_val_case(line,&g_val_cases[g_val_case_count])) return 0;
        for (unsigned i=0;i<g_val_case_count;++i)
            if (!strcmp(g_val_cases[i].run.key,g_val_cases[g_val_case_count].run.key)) return 0;
        ++g_val_case_count;
    }
    return g_val_case_count>0;
}

static int load_val_samples(const runtime_val_case_t *item) {
    u32 bytes,count=0; char *cursor=g_val_index,*line;
    if (!load_checked(item->samples,(u8 *)g_val_index,sizeof(g_val_index)-1,0,item->samples_crc,&bytes) ||
        memchr(g_val_index,0,bytes)) return 0;
    g_val_index[bytes]=0; line=csv_line(&cursor);
    if (!line || strcmp(line,RUNTIME_VAL_SAMPLE_HEADER)) return 0;
    while ((line=csv_line(&cursor))) {
        if (count>=INT8_APP_VAL_SAMPLES || strlen(line)>=RUNTIME_LINE_BYTES ||
            !runtime_parse_val_sample(line,count,&g_val_samples[count])) return 0;
        ++count;
    }
    return count==INT8_APP_VAL_SAMPLES;
}

static int valid_target(const runtime_case_t *item) {
    for (u32 i=0;i<item->output_bytes;++i)
        if (g_target[i]!=255U && g_target[i]>=item->classes) {
            xil_printf("APP: invalid target ID at %u\r\n",i); return 0;
        }
    return 1;
}

static void validation_metrics(const runtime_case_t *item) {
    u64 valid=0,correct=0,iou_sum=0; unsigned iou_count=0;
    for (u32 c=0;c<item->classes;++c) {
        u64 support=0,predicted=0,tp=g_confusion[c][c];
        for (u32 k=0;k<item->classes;++k) { support+=g_confusion[c][k]; predicted+=g_confusion[k][c]; }
        valid+=support; correct+=tp;
        if (item->classes==20 && c==19) continue;
        u64 denominator=support+predicted-tp;
        if (denominator) { iou_sum+=tp*1000000ULL/denominator; ++iou_count; }
        xil_printf("VAL_CLASS,%s,%u,tp=%llu,support=%llu,predicted=%llu,PA_ppm=%lld,IoU_ppm=%lld\r\n",
            item->key,c,tp,support,predicted,support?(long long)(tp*1000000ULL/support):-1LL,
            denominator?(long long)(tp*1000000ULL/denominator):-1LL);
        for (u32 k=0;k<item->classes;++k)
            xil_printf("VAL_CONFUSION,%s,%u,%u,%llu\r\n",item->key,c,k,g_confusion[c][k]);
    }
    xil_printf("VAL_METRICS,%s,PA_ppm=%lld,mIoU_ppm=%lld,valid_pixels=%llu\r\n",
        item->key,valid?(long long)(correct*1000000ULL/valid):-1LL,
        iou_count?(long long)(iou_sum/iou_count):-1LL,valid);
}

static int run_val_case(Int8NpuContext *npu,const runtime_val_case_t *entry) {
    const runtime_case_t *item=&entry->run;
    Int8ParamBlobHeader header;
    u64 sum_ticks=0,sum_rtl=0,min_ticks=~0ULL,max_ticks=0;
    u32 completed=0,size; char input[32],target[32],output[32];
    memset(g_confusion,0,sizeof(g_confusion));
    xil_printf("VAL_BEGIN,%s,samples=%u,ignore_target=%u,param_CRC=%08x,samples_CRC=%08x\r\n",
        item->key,INT8_APP_VAL_SAMPLES,item->classes==20?19U:255U,item->param_crc,entry->samples_crc);
    if (!load_val_samples(entry) || !prepare_case(npu,item,&header)) goto failed;
    for (u32 i=0;i<INT8_APP_VAL_SAMPLES;++i) {
        u64 ticks,rtl;
        (void)snprintf(input,sizeof(input),"%.8s/I%04u.BIN",item->key,i);
        (void)snprintf(target,sizeof(target),"%.8s/T%04u.BIN",item->key,i);
        (void)snprintf(output,sizeof(output),"%.8s/O%04u.BIN",item->key,i);
        if (!load_checked(input,g_input,sizeof(g_input),item->input_bytes,g_val_samples[i].input_crc,&size) ||
            !load_checked(target,g_target,sizeof(g_target),item->output_bytes,g_val_samples[i].target_crc,&size) ||
            !valid_target(item) || !infer_frame(npu,item,header.uop_count,i==0,&ticks,&rtl) ||
            SD_SaveMemoryToFile(output,g_output,item->output_bytes)!=XST_SUCCESS) goto failed;
        for (u32 p=0;p<item->output_bytes;++p) {
            u32 t=g_target[p];
            if (t==255U || (item->classes==20 && t==19U)) continue;
            ++g_confusion[t][g_output[p]];
        }
        sum_ticks+=ticks; sum_rtl+=rtl;
        if (ticks<min_ticks) min_ticks=ticks;
        if (ticks>max_ticks) max_ticks=ticks;
        ++completed;
        xil_printf("VAL_SAMPLE,%s,%u,arm_ticks=%llu,rtl_cycles=%llu,mask_CRC=%08x\r\n",
            item->key,i,ticks,rtl,runtime_crc32(g_output,item->output_bytes));
        if (app_log_flush()!=XST_SUCCESS) goto failed;
    }
    {
        u64 valid=0;
        for (u32 t=0;t<item->classes;++t)
            for (u32 p=0;p<item->classes;++p) valid+=g_confusion[t][p];
        if (!valid) { xil_printf("APP: no valid GT pixels; accuracy rejected\r\n"); goto failed; }
    }
    validation_metrics(item);
    xil_printf("VAL_SUMMARY,%s,PASS,samples=%u,arm_hz=%llu,PL_hz=%llu,total_arm_ticks=%llu,total_rtl_cycles=%llu,avg_arm_us=%llu,min_arm_us=%llu,max_arm_us=%llu\r\n",
        item->key,completed,timer_frequency(),INT8_APP_PL_TARGET_HZ,sum_ticks,sum_rtl,
        sum_ticks*1000000ULL/(timer_frequency()*completed),min_ticks*1000000ULL/timer_frequency(),
        max_ticks*1000000ULL/timer_frequency());
    if (app_log_flush()!=XST_SUCCESS) goto failed;
    return 1;
failed:
    xil_printf("VAL_SUMMARY,%s,FAIL,completed=%u,failed_index=%u; partial results are not VAL500\r\n",
        item->key,completed,completed);
    return 0;
}

static int run_validation(Int8NpuContext *npu,const char *key) {
    unsigned completed=0;
    if (!load_val_catalog()) goto failed;
    xil_printf("APP: validation scope=VALRUNS.CSV; unregistered hardware modes are NOT validated\r\n");
    for (unsigned i=0;i<g_val_case_count;++i) {
        if (strcmp(key,"all") && strcmp(key,g_val_cases[i].run.key)) continue;
        if (!run_val_case(npu,&g_val_cases[i])) goto failed;
        ++completed;
    }
    if (!completed) goto failed;
    xil_printf("VALSUITE,PASS,cases=%u,samples_per_case=%u\r\n",completed,INT8_APP_VAL_SAMPLES);
    if (app_log_flush()!=XST_SUCCESS) goto failed;
    return 1;
failed:
    xil_printf("VALSUITE,FAIL,completed_cases=%u; stopped, no invalid timing accepted\r\n",completed);
    return 0;
}

static void list_cases(void) {
    xil_printf("key,profile,classes,source,mask,input_bytes,output_bytes\r\n");
    for (unsigned i=0;i<g_case_count;++i) {
        const runtime_case_t *c=&g_cases[i];
        xil_printf("%s,%u,%u,%u,%u,%u,%u\r\n",c->key,c->profile,c->classes,c->source,c->mask,c->input_bytes,c->output_bytes);
    }
}

static int read_command(char *out,unsigned capacity) {
    unsigned used=0,overflow=0;
    for (;;) {
        char c=inbyte();
        if (c=='\r') continue;
        if (c=='\n') { out[used]=0; return !overflow; }
        if ((unsigned char)c==4U) return -1;
        if (used+1<capacity && !overflow) out[used++]=c; else overflow=1;
    }
}

int main(void) {
    Int8NpuContext npu; char command[80],verb[16],arg[32],extra[2];
    int result=XST_FAILURE;
    xil_printf("ESP INT8 app %s\r\n",INT8_APP_BUILD_TAG);
    int timer_ok=timer_selftest();
    if (SD_Init()!=XST_SUCCESS) {
        xil_printf("APP: initialization/catalog rejected; SD unavailable, no TXT log\r\n"); return XST_FAILURE;
    }
    if (app_log_open()!=XST_SUCCESS) goto finished;
    if (!timer_ok || (SD_FileExists(INT8_APP_CATALOG_FILE) && !load_catalog()) ||
        (!g_case_count && !SD_FileExists(INT8_APP_VAL_CATALOG_FILE)) || int8_npu_init(&npu)!=XST_SUCCESS) {
        xil_printf("APP: initialization/catalog rejected\r\n"); goto finished;
    }
    if (INT8_APP_AUTORUN_VALSET && SD_FileExists(INT8_APP_VAL_CATALOG_FILE)) {
        xil_printf("APP: autorun full validation; every registered case has 500 samples\r\n");
        if (!run_validation(&npu,"all")) goto finished;
    } else xil_printf("APP: VALRUNS.CSV not loaded; 500-image validation has not run\r\n");
    xil_printf("APP: commands: list | run KEY | suite [1..10] | val all|KEY | help | quit\r\n");
    for (;;) {
        xil_printf("npu> ");
        if (app_log_flush()!=XST_SUCCESS) goto finished;
        int read=read_command(command,sizeof(command));
        if (read<0) break;
        if (!read) { xil_printf("APP: command too long\r\n"); continue; }
        int fields=sscanf(command,"%15s %31s %1s",verb,arg,extra);
        if (fields==1 && !strcmp(verb,"quit")) break;
        if (fields==1 && !strcmp(verb,"list")) { list_cases(); continue; }
        if (fields==1 && !strcmp(verb,"help")) {
            xil_printf("list | run KEY | suite [1..10] | val all|KEY | quit; only registered RUNS.CSV entries (val: VALRUNS.CSV)\r\n"); continue;
        }
        if (fields==2 && !strcmp(verb,"val")) {
            (void)run_validation(&npu,arg);
        } else if (fields==2 && !strcmp(verb,"run")) {
            unsigned i; for (i=0;i<g_case_count && strcmp(arg,g_cases[i].key);++i) {}
            if (i==g_case_count) xil_printf("APP: unknown case\r\n");
            else if (!run_case(&npu,&g_cases[i],1)) xil_printf("APP: run FAIL\r\n");
        } else if ((fields==1 || fields==2) && !strcmp(verb,"suite")) {
            uint32_t repeats=3; unsigned ok=1;
            if (fields==2 && (!runtime_parse_uint(arg,10,&repeats) || repeats<1 || repeats>10)) ok=0;
            if (!ok || !g_case_count) { xil_printf("APP: invalid suite count/catalog\r\n"); continue; }
            for (uint32_t r=0;r<repeats && ok;++r)
                for (unsigned i=0;i<g_case_count && ok;++i) ok=run_case(&npu,&g_cases[i],r==0);
            if (ok) ok=run_case(&npu,&g_cases[0],0);
            xil_printf("SUITE,%s,repeats=%u,cases=%u\r\n",ok?"PASS":"FAIL",repeats,g_case_count);
        } else xil_printf("APP: invalid command\r\n");
        if (!app_log_ok()) goto finished;
        if (npu.blocked) { xil_printf("APP: timeout: hardware reset required; no more transactions\r\n"); break; }
    }
    result=XST_SUCCESS;
finished:
    if (app_log_close()!=XST_SUCCESS) result=XST_FAILURE;
    return result;
}
