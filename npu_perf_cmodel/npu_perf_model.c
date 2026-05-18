#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    OP_LOAD_DDR = 0,
    OP_CONV,
    OP_POOL,
    OP_ADD,
    OP_AFFINE,
    OP_STORE_FM,
    OP_STORE_DDR
} op_type_t;

typedef struct {
    double freq_mhz;
    double ddr_eff;
    double conv1_eff;
    double conv3_eff;
    double vec_eff;
    double pool_eff;
    uint32_t ddr_bus_bytes_per_cycle;
    uint32_t vec_lanes;
    uint32_t param_blob_bytes;
} model_cfg_t;

typedef struct {
    const char *name;
    int stage;
    op_type_t op;
    uint32_t in_h;
    uint32_t in_w;
    uint32_t in_c;
    uint32_t out_h;
    uint32_t out_w;
    uint32_t out_c;
    uint32_t kernel;
    uint32_t stride;
    uint32_t dilation;
    uint32_t bytes;
    uint32_t valid_c;
} uop_desc_t;

typedef struct {
    const char *name;
    uint64_t cycles;
} stage_stat_t;

static uint64_t ceil_div_u64(uint64_t a, uint64_t b) {
    return (a + b - 1ULL) / b;
}

static uint64_t ceil_div_bytes_eff(uint64_t bytes, uint32_t bytes_per_cycle, double eff) {
    double denom = (double)bytes_per_cycle * eff;
    return (uint64_t)ceil((double)bytes / denom);
}

static uint64_t model_load_ddr_cycles(const model_cfg_t *cfg, uint32_t bytes) {
    return ceil_div_bytes_eff(bytes, cfg->ddr_bus_bytes_per_cycle, cfg->ddr_eff) + 96ULL;
}

static uint64_t model_store_ddr_cycles(const model_cfg_t *cfg, uint32_t bytes) {
    return ceil_div_bytes_eff(bytes, cfg->ddr_bus_bytes_per_cycle, cfg->ddr_eff) + 96ULL;
}

static double conv_eff_for_uop(const model_cfg_t *cfg, const uop_desc_t *u) {
    double eff = (u->kernel == 1U) ? cfg->conv1_eff : cfg->conv3_eff;
    if (u->kernel == 3U && u->stride == 2U) {
        eff -= 0.02;
    }
    if (u->kernel == 3U) {
        if (u->dilation >= 16U) {
            eff -= 0.03;
        } else if (u->dilation >= 8U) {
            eff -= 0.02;
        } else if (u->dilation >= 2U) {
            eff -= 0.01;
        }
    }
    if (eff < 0.72) {
        eff = 0.72;
    }
    return eff;
}

static uint64_t model_conv_cycles(const model_cfg_t *cfg, const uop_desc_t *u) {
    uint64_t spatial = (uint64_t)u->out_h * (uint64_t)u->out_w;
    uint64_t oc_tiles = ceil_div_u64(u->out_c, cfg->vec_lanes);
    uint64_t k_flat = (uint64_t)u->in_c * (uint64_t)u->kernel * (uint64_t)u->kernel;
    uint64_t k_tiles = ceil_div_u64(k_flat, cfg->vec_lanes);
    uint64_t cin_tiles = ceil_div_u64(u->in_c, cfg->vec_lanes);
    uint64_t ideal = spatial * oc_tiles * k_tiles;
    uint64_t warmup = (u->kernel == 3U)
        ? (uint64_t)(2U * u->dilation) * (uint64_t)u->in_w * cin_tiles
        : cin_tiles;
    uint64_t tail = 64ULL + 8ULL * k_tiles + 4ULL * oc_tiles;
    double eff = conv_eff_for_uop(cfg, u);
    return (uint64_t)ceil(((double)ideal + (double)(warmup * oc_tiles) + (double)tail) / eff);
}

static uint64_t model_pool_cycles(const model_cfg_t *cfg, const uop_desc_t *u) {
    uint64_t out_pixels = (uint64_t)u->out_h * (uint64_t)u->out_w;
    uint64_t warmup = 2ULL * (uint64_t)u->in_w + 64ULL;
    uint64_t tail = 24ULL;
    return (uint64_t)ceil(((double)out_pixels + (double)warmup + (double)tail) / cfg->pool_eff);
}

static uint64_t model_affine_cycles(const model_cfg_t *cfg, const uop_desc_t *u) {
    uint64_t vectors = (uint64_t)u->out_h * (uint64_t)u->out_w *
                       ceil_div_u64(u->out_c, cfg->vec_lanes);
    return (uint64_t)ceil(((double)vectors + 32.0) / cfg->vec_eff);
}

static uint64_t model_add_cycles(const model_cfg_t *cfg, const uop_desc_t *u) {
    uint64_t vectors = (uint64_t)u->out_h * (uint64_t)u->out_w *
                       ceil_div_u64(u->out_c, cfg->vec_lanes);
    return (uint64_t)ceil(((double)vectors + 24.0) / cfg->vec_eff);
}

static uint64_t model_store_fm_cycles(const model_cfg_t *cfg, const uop_desc_t *u) {
    uint32_t channels = u->valid_c ? u->valid_c : u->out_c;
    uint64_t vectors = (uint64_t)u->out_h * (uint64_t)u->out_w *
                       ceil_div_u64(channels, cfg->vec_lanes);
    return (uint64_t)ceil(((double)vectors + 16.0) / cfg->vec_eff);
}

static uint64_t uop_cycles(const model_cfg_t *cfg, const uop_desc_t *u) {
    switch (u->op) {
        case OP_LOAD_DDR:
            return model_load_ddr_cycles(cfg, u->bytes);
        case OP_CONV:
            return model_conv_cycles(cfg, u);
        case OP_POOL:
            return model_pool_cycles(cfg, u);
        case OP_ADD:
            return model_add_cycles(cfg, u);
        case OP_AFFINE:
            return model_affine_cycles(cfg, u);
        case OP_STORE_FM:
            return model_store_fm_cycles(cfg, u);
        case OP_STORE_DDR:
            return model_store_ddr_cycles(cfg, u->bytes);
        default:
            return 0ULL;
    }
}

static double cycles_to_ms(uint64_t cycles, double freq_mhz) {
    return (double)cycles / (freq_mhz * 1000.0);
}

static void parse_args(int argc, char **argv, model_cfg_t *cfg) {
    int i;
    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--freq-mhz") == 0 && i + 1 < argc) {
            cfg->freq_mhz = atof(argv[++i]);
        } else if (strcmp(argv[i], "--ddr-eff") == 0 && i + 1 < argc) {
            cfg->ddr_eff = atof(argv[++i]);
        } else if (strcmp(argv[i], "--conv1-eff") == 0 && i + 1 < argc) {
            cfg->conv1_eff = atof(argv[++i]);
        } else if (strcmp(argv[i], "--conv3-eff") == 0 && i + 1 < argc) {
            cfg->conv3_eff = atof(argv[++i]);
        } else if (strcmp(argv[i], "--vec-eff") == 0 && i + 1 < argc) {
            cfg->vec_eff = atof(argv[++i]);
        } else if (strcmp(argv[i], "--pool-eff") == 0 && i + 1 < argc) {
            cfg->pool_eff = atof(argv[++i]);
        }
    }
}

static void print_cfg(const model_cfg_t *cfg) {
    printf("=== NPU Performance Model Assumptions ===\n");
    printf("freq_mhz           : %.2f\n", cfg->freq_mhz);
    printf("ddr_eff            : %.3f\n", cfg->ddr_eff);
    printf("conv1_eff          : %.3f\n", cfg->conv1_eff);
    printf("conv3_eff          : %.3f\n", cfg->conv3_eff);
    printf("vec_eff            : %.3f\n", cfg->vec_eff);
    printf("pool_eff           : %.3f\n", cfg->pool_eff);
    printf("ddr_bus_bytes/cyc  : %u\n", cfg->ddr_bus_bytes_per_cycle);
    printf("vector_lanes       : %u\n", cfg->vec_lanes);
    printf("param_blob_bytes   : %u\n", cfg->param_blob_bytes);
    printf("\n");
}

int main(int argc, char **argv) {
    model_cfg_t cfg = {
        100.0,   /* freq_mhz */
        0.70,    /* ddr_eff */
        0.93,    /* conv1_eff */
        0.88,    /* conv3_eff */
        0.92,    /* vec_eff */
        0.90,    /* pool_eff */
        32U,     /* ddr_bus_bytes_per_cycle */
        32U,     /* vec_lanes */
        256U * 1024U /* param_blob_bytes */
    };

    static const char *stage_names[] = {
        "Stage0_Input_B1",
        "Stage1_Level2_0",
        "Stage2_Level2_B0",
        "Stage3_B2",
        "Stage4_Level3_0",
        "Stage5_Level3_B0",
        "Stage6_B3_Classifier"
    };

    static const uop_desc_t uops[] = {
        {"LOAD_INPUT",         0, OP_LOAD_DDR, 0,   0,   0,   0,   0,   0, 0, 0, 0, 512U*1024U*3U, 0},
        {"POOL_B1",            0, OP_POOL,     512, 1024, 3,   256, 512, 3, 3, 2, 1, 0, 3},
        {"LEVEL1_CONV",        0, OP_CONV,     512, 1024, 3,   256, 512, 16,3, 2, 1, 0, 16},
        {"STORE_B1_POOL",      0, OP_STORE_FM, 256, 512,  3,   256, 512, 3,  0, 0, 0, 0, 3},
        {"B1_AFFINE",          0, OP_AFFINE,   256, 512, 19,   256, 512, 19, 0, 0, 0, 0, 19},

        {"L20_C1",             1, OP_CONV,     256, 512,  19,  128, 256, 12, 3, 2, 1, 0, 12},
        {"L20_D1",             1, OP_CONV,     128, 256,  12,  128, 256, 16, 3, 1, 1, 0, 16},
        {"L20_D2",             1, OP_CONV,     128, 256,  12,  128, 256, 12, 3, 1, 2, 0, 12},
        {"STORE_L20_D2",       1, OP_STORE_FM, 128, 256,  12,  128, 256, 12, 0, 0, 0, 0, 12},
        {"L20_D4",             1, OP_CONV,     128, 256,  12,  128, 256, 12, 3, 1, 4, 0, 12},
        {"ADD_L20_1",          1, OP_ADD,      128, 256,  12,  128, 256, 12, 0, 0, 0, 0, 12},
        {"STORE_L20_ADD1",     1, OP_STORE_FM, 128, 256,  12,  128, 256, 12, 0, 0, 0, 0, 12},
        {"L20_D8",             1, OP_CONV,     128, 256,  12,  128, 256, 12, 3, 1, 8, 0, 12},
        {"ADD_L20_2",          1, OP_ADD,      128, 256,  12,  128, 256, 12, 0, 0, 0, 0, 12},
        {"STORE_L20_ADD2",     1, OP_STORE_FM, 128, 256,  12,  128, 256, 12, 0, 0, 0, 0, 12},
        {"L20_D16",            1, OP_CONV,     128, 256,  12,  128, 256, 12, 3, 1, 16,0, 12},
        {"ADD_L20_3",          1, OP_ADD,      128, 256,  12,  128, 256, 12, 0, 0, 0, 0, 12},
        {"STORE_L20_ADD3",     1, OP_STORE_FM, 128, 256,  12,  128, 256, 12, 0, 0, 0, 0, 12},
        {"L20_AFFINE",         1, OP_AFFINE,   128, 256,  64,  128, 256, 64, 0, 0, 0, 0, 64},

        {"L2B0_C1",            2, OP_CONV,     128, 256,  64,  128, 256, 12, 1, 1, 1, 0, 12},
        {"L2B0_D1",            2, OP_CONV,     128, 256,  12,  128, 256, 16, 3, 1, 1, 0, 16},
        {"L2B0_D2",            2, OP_CONV,     128, 256,  12,  128, 256, 12, 3, 1, 2, 0, 12},
        {"STORE_L2B0_D2",      2, OP_STORE_FM, 128, 256,  12,  128, 256, 12, 0, 0, 0, 0, 12},
        {"L2B0_D4",            2, OP_CONV,     128, 256,  12,  128, 256, 12, 3, 1, 4, 0, 12},
        {"ADD_L2B0_1",         2, OP_ADD,      128, 256,  12,  128, 256, 12, 0, 0, 0, 0, 12},
        {"STORE_L2B0_ADD1",    2, OP_STORE_FM, 128, 256,  12,  128, 256, 12, 0, 0, 0, 0, 12},
        {"L2B0_D8",            2, OP_CONV,     128, 256,  12,  128, 256, 12, 3, 1, 8, 0, 12},
        {"ADD_L2B0_2",         2, OP_ADD,      128, 256,  12,  128, 256, 12, 0, 0, 0, 0, 12},
        {"STORE_L2B0_ADD2",    2, OP_STORE_FM, 128, 256,  12,  128, 256, 12, 0, 0, 0, 0, 12},
        {"L2B0_D16",           2, OP_CONV,     128, 256,  12,  128, 256, 12, 3, 1, 16,0, 12},
        {"ADD_L2B0_3",         2, OP_ADD,      128, 256,  12,  128, 256, 12, 0, 0, 0, 0, 12},
        {"STORE_L2B0_ADD3",    2, OP_STORE_FM, 128, 256,  12,  128, 256, 12, 0, 0, 0, 0, 12},
        {"ADD_L2B0_RES",       2, OP_ADD,      128, 256,  64,  128, 256, 64, 0, 0, 0, 0, 64},
        {"L2B0_AFFINE",        2, OP_AFFINE,   128, 256,  64,  128, 256, 64, 0, 0, 0, 0, 64},

        {"POOL_B2_TMP",        3, OP_POOL,     512, 1024, 3,   256, 512, 3,  3, 2, 1, 0, 3},
        {"POOL_B2_OUT",        3, OP_POOL,     256, 512,  3,   128, 256, 3,  3, 2, 1, 0, 3},
        {"STORE_B2_FROM_B0",   3, OP_STORE_FM, 128, 256,  64,  128, 256, 64, 0, 0, 0, 0, 64},
        {"STORE_B2_FROM_L20",  3, OP_STORE_FM, 128, 256,  64,  128, 256, 64, 0, 0, 0, 0, 64},
        {"STORE_B2_FROM_POOL", 3, OP_STORE_FM, 128, 256,  3,   128, 256, 3,  0, 0, 0, 0, 3},
        {"B2_AFFINE",          3, OP_AFFINE,   128, 256,  131, 128, 256, 131,0, 0, 0, 0, 131},

        {"L30_C1",             4, OP_CONV,     128, 256,  131, 64,  128, 25, 3, 2, 1, 0, 25},
        {"L30_D1",             4, OP_CONV,     64,  128,  25,  64,  128, 28, 3, 1, 1, 0, 28},
        {"L30_D2",             4, OP_CONV,     64,  128,  25,  64,  128, 25, 3, 1, 2, 0, 25},
        {"STORE_L30_D2",       4, OP_STORE_FM, 64,  128,  25,  64,  128, 25, 0, 0, 0, 0, 25},
        {"L30_D4",             4, OP_CONV,     64,  128,  25,  64,  128, 25, 3, 1, 4, 0, 25},
        {"ADD_L30_1",          4, OP_ADD,      64,  128,  25,  64,  128, 25, 0, 0, 0, 0, 25},
        {"STORE_L30_ADD1",     4, OP_STORE_FM, 64,  128,  25,  64,  128, 25, 0, 0, 0, 0, 25},
        {"L30_D8",             4, OP_CONV,     64,  128,  25,  64,  128, 25, 3, 1, 8, 0, 25},
        {"ADD_L30_2",          4, OP_ADD,      64,  128,  25,  64,  128, 25, 0, 0, 0, 0, 25},
        {"STORE_L30_ADD2",     4, OP_STORE_FM, 64,  128,  25,  64,  128, 25, 0, 0, 0, 0, 25},
        {"L30_D16",            4, OP_CONV,     64,  128,  25,  64,  128, 25, 3, 1, 16,0, 25},
        {"ADD_L30_3",          4, OP_ADD,      64,  128,  25,  64,  128, 25, 0, 0, 0, 0, 25},
        {"STORE_L30_ADD3",     4, OP_STORE_FM, 64,  128,  25,  64,  128, 25, 0, 0, 0, 0, 25},
        {"L30_AFFINE",         4, OP_AFFINE,   64,  128,  128, 64,  128, 128,0, 0, 0, 0, 128},

        {"L3B0_C1",            5, OP_CONV,     64,  128,  128, 64,  128, 25, 1, 1, 1, 0, 25},
        {"L3B0_D1",            5, OP_CONV,     64,  128,  25,  64,  128, 28, 3, 1, 1, 0, 28},
        {"L3B0_D2",            5, OP_CONV,     64,  128,  25,  64,  128, 25, 3, 1, 2, 0, 25},
        {"STORE_L3B0_D2",      5, OP_STORE_FM, 64,  128,  25,  64,  128, 25, 0, 0, 0, 0, 25},
        {"L3B0_D4",            5, OP_CONV,     64,  128,  25,  64,  128, 25, 3, 1, 4, 0, 25},
        {"ADD_L3B0_1",         5, OP_ADD,      64,  128,  25,  64,  128, 25, 0, 0, 0, 0, 25},
        {"STORE_L3B0_ADD1",    5, OP_STORE_FM, 64,  128,  25,  64,  128, 25, 0, 0, 0, 0, 25},
        {"L3B0_D8",            5, OP_CONV,     64,  128,  25,  64,  128, 25, 3, 1, 8, 0, 25},
        {"ADD_L3B0_2",         5, OP_ADD,      64,  128,  25,  64,  128, 25, 0, 0, 0, 0, 25},
        {"STORE_L3B0_ADD2",    5, OP_STORE_FM, 64,  128,  25,  64,  128, 25, 0, 0, 0, 0, 25},
        {"L3B0_D16",           5, OP_CONV,     64,  128,  25,  64,  128, 25, 3, 1, 16,0, 25},
        {"ADD_L3B0_3",         5, OP_ADD,      64,  128,  25,  64,  128, 25, 0, 0, 0, 0, 25},
        {"STORE_L3B0_ADD3",    5, OP_STORE_FM, 64,  128,  25,  64,  128, 25, 0, 0, 0, 0, 25},
        {"ADD_L3B0_RES",       5, OP_ADD,      64,  128,  128, 64,  128, 128,0, 0, 0, 0, 128},
        {"L3B0_AFFINE",        5, OP_AFFINE,   64,  128,  128, 64,  128, 128,0, 0, 0, 0, 128},

        {"STORE_B3_FROM_L30",  6, OP_STORE_FM, 64,  128,  128, 64,  128, 128,0, 0, 0, 0, 128},
        {"STORE_B3_FROM_B0",   6, OP_STORE_FM, 64,  128,  128, 64,  128, 128,0, 0, 0, 0, 128},
        {"B3_AFFINE",          6, OP_AFFINE,   64,  128,  256, 64,  128, 256,0, 0, 0, 0, 256},
        {"CLASSIFIER",         6, OP_CONV,     64,  128,  256, 64,  128, 2,  1, 1, 1, 0, 2},
        {"STORE_OUTPUT",       6, OP_STORE_DDR,0,   0,    0,   0,   0,   0,  0, 0, 0, 64U*128U*2U, 0}
    };

    parse_args(argc, argv, &cfg);
    print_cfg(&cfg);

    stage_stat_t stage_stats[7];
    uint64_t total_run_cycles = 0ULL;
    uint64_t init_cycles = model_load_ddr_cycles(&cfg, cfg.param_blob_bytes) + 512ULL;
    size_t i;

    for (i = 0; i < 7; ++i) {
        stage_stats[i].name = stage_names[i];
        stage_stats[i].cycles = 0ULL;
    }

    printf("=== Per-UOP Cycles ===\n");
    for (i = 0; i < sizeof(uops) / sizeof(uops[0]); ++i) {
        uint64_t cyc = uop_cycles(&cfg, &uops[i]);
        total_run_cycles += cyc;
        stage_stats[uops[i].stage].cycles += cyc;
        printf("%-20s stage=%d cycles=%10llu ms=%8.3f\n",
               uops[i].name,
               uops[i].stage,
               (unsigned long long)cyc,
               cycles_to_ms(cyc, cfg.freq_mhz));
    }

    printf("\n=== Per-Stage Summary ===\n");
    for (i = 0; i < 7; ++i) {
        printf("%-20s cycles=%10llu ms=%8.3f share=%6.2f%%\n",
               stage_stats[i].name,
               (unsigned long long)stage_stats[i].cycles,
               cycles_to_ms(stage_stats[i].cycles, cfg.freq_mhz),
               100.0 * (double)stage_stats[i].cycles / (double)total_run_cycles);
    }

    printf("\n=== Final Summary ===\n");
    printf("INIT cycles        : %10llu\n", (unsigned long long)init_cycles);
    printf("INIT ms            : %10.3f\n", cycles_to_ms(init_cycles, cfg.freq_mhz));
    printf("RUN cycles         : %10llu\n", (unsigned long long)total_run_cycles);
    printf("RUN ms             : %10.3f\n", cycles_to_ms(total_run_cycles, cfg.freq_mhz));
    printf("INIT+RUN cycles    : %10llu\n", (unsigned long long)(init_cycles + total_run_cycles));
    printf("INIT+RUN ms        : %10.3f\n", cycles_to_ms(init_cycles + total_run_cycles, cfg.freq_mhz));

    if (cycles_to_ms(total_run_cycles, cfg.freq_mhz) < 100.0) {
        printf("RESULT             : PASS (RUN latency < 100 ms at %.2f MHz)\n", cfg.freq_mhz);
    } else {
        printf("RESULT             : FAIL (RUN latency >= 100 ms at %.2f MHz)\n", cfg.freq_mhz);
    }

    return 0;
}
