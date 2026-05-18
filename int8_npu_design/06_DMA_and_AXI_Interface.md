# DMA and AXI System Interface (总线互联与系统调度)

## 1. 最终冻结原则
本文件定义 **软件/导出器/HLS/RTL 之间唯一有效的二进制契约**。从本版开始：

1. 不再保留“逻辑寄存器表”和“HLS 自动寄存器表”并存的双重解释
2. 不再保留“uop 先定义结构体、之后再决定打包方式”的自由度
3. 不再保留“参数文件若干个、运行时再拼”的开放实现

首版冻结为：

- 单一 HLS 核心：`espnet_encoder_int8_core`
- 单一软件可见包装 IP：`espnet_encoder_int8_top`
- `m_axi` 直连 DDR
- `s_axi_control` 走固定寄存器映射
- `gmem_param` 指向单一 `param_blob.bin`

## 2. 顶层接口冻结

### 2.1 HLS 核心接口
HLS 核心函数冻结为：

```cpp
void espnet_encoder_int8_core(
    const ap_uint<256>* gmem_frame_in,
    ap_uint<256>* gmem_frame_out,
    const ap_uint<256>* gmem_param,
    uint32_t mode,
    uint32_t uop_count,
    volatile uint32_t& dbg_status,
    volatile uint32_t& dbg_heartbeat,
    volatile uint32_t& dbg_act_words,
    volatile uint32_t& dbg_wgt_words,
    volatile uint32_t& dbg_psum_words,
    volatile uint32_t& dbg_out_words,
    volatile uint32_t& dbg_hw_version);
```

推荐 pragma：

```cpp
#pragma HLS INTERFACE m_axi     port=gmem_frame_in  offset=slave bundle=gmem0 depth=49152 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi     port=gmem_frame_out offset=slave bundle=gmem1 depth=512   max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE m_axi     port=gmem_param     offset=slave bundle=gmem2 depth=4096  max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE s_axilite port=mode           bundle=control
#pragma HLS INTERFACE s_axilite port=uop_count      bundle=control
#pragma HLS INTERFACE s_axilite port=dbg_status     bundle=control
#pragma HLS INTERFACE s_axilite port=dbg_heartbeat  bundle=control
#pragma HLS INTERFACE s_axilite port=dbg_act_words  bundle=control
#pragma HLS INTERFACE s_axilite port=dbg_wgt_words  bundle=control
#pragma HLS INTERFACE s_axilite port=dbg_psum_words bundle=control
#pragma HLS INTERFACE s_axilite port=dbg_out_words  bundle=control
#pragma HLS INTERFACE s_axilite port=dbg_hw_version bundle=control
#pragma HLS INTERFACE s_axilite port=return         bundle=control
```

### 2.2 软件可见 IP 形态
软件侧最终对接的不是裸 HLS 自动寄存器，而是**薄包装后的固定寄存器接口**：

- 包装名：`espnet_encoder_int8_top`
- 核心名：`espnet_encoder_int8_core`
- 包装职责：
  - 对齐固定 AXI-Lite 寄存器偏移
  - 把 `FRAME_IN_BASE / FRAME_OUT_BASE / PARAM_BASE` 转成 HLS 核心 pointer base
  - 汇总 `done / idle / irq / perf`

结论：

- **固定寄存器表以本文件为准**
- **若 HLS 自动生成寄存器偏移不同，必须由 wrapper 对齐**

### 2.3 与 FP32 平台的关系
本版不再设计多路离散 DMA IP，也不再保留 FP32 时代的 `MAGIC_CONV / MAGIC_BR / MAGIC_ADD` 数据包协议。

补充说明：

- 当前板级平台虽然仍保留 `4` 个 AXI DMA IP，但本 `v1` top **不直接使用这 4 个外部 DMA**
- `gmem_frame_in / gmem_frame_out / gmem_param` 对应的是 HLS 自动生成的 `m_axi` master 口
- `frame_dma / param_dma` 是 top 内部 memory mover 逻辑，而不是板上独立 DMA 外设

## 3. 运行模式冻结
首版只保留两个模式：

1. `MODE_INIT = 1`
2. `MODE_RUN  = 2`

### 3.1 MODE_INIT
作用：

- 从 DDR 的 `param_blob.bin` 把全部静态参数加载到片上
- 包括权重、量化参数、descriptor 表和 `uop table`

### 3.2 MODE_RUN
作用：

- 读取一帧 `input_q`
- 执行整条 Encoder `uop table`
- 将最终 `1 x 64 x 128 x 2` 结果写回 DDR

运行时 CPU 不参与逐层控制。

## 4. AXI-Lite 寄存器映射冻结
软件可见寄存器表冻结如下（由 HLS `s_axilite` 自动生成，偏移为实际值）：

### 4.1 控制寄存器

| Offset | Name | Description |
|---|---|---|
| `0x00` | `ap_ctrl` | bit0=`ap_start`, bit1=`ap_done`, bit2=`ap_idle`, bit3=`ap_ready` |
| `0x04` | `GIE` | Global Interrupt Enable |
| `0x08` | `IER` | IP Interrupt Enable Register |
| `0x0c` | `ISR` | IP Interrupt Status Register |
| `0x10` | `gmem_frame_in[31:0]` | 输入帧 DDR 地址低 32 位 |
| `0x14` | `gmem_frame_in[63:32]` | 输入帧 DDR 地址高 32 位 |
| `0x1c` | `gmem_frame_out[31:0]` | 输出帧 DDR 地址低 32 位 |
| `0x20` | `gmem_frame_out[63:32]` | 输出帧 DDR 地址高 32 位 |
| `0x28` | `gmem_param[31:0]` | param_blob.bin DDR 地址低 32 位 |
| `0x2c` | `gmem_param[63:32]` | param_blob.bin DDR 地址高 32 位 |
| `0x34` | `mode` | `0=idle`, `1=init`, `2=run` |
| `0x3c` | `uop_count` | 本次 RUN 有效 uop 数 |

### 4.2 调试寄存器（`_DATA` 为数据，`_VLD` 为有效标志）

| Offset | Name | Description |
|---|---|---|
| `0x44` | `dbg_status_DATA` | phase[7:0] \| opcode[15:8] \| current_uop[23:16] \| last_done_uop[31:24] |
| `0x48` | `dbg_status_VLD` | dbg_status 有效标志 |
| `0x54` | `dbg_heartbeat_DATA` | error_code[31:16] \| heartbeat_counter[15:0] |
| `0x58` | `dbg_heartbeat_VLD` | dbg_heartbeat 有效标志 |
| `0x64` | `dbg_act_words_DATA` | 激活向量发射计数 |
| `0x68` | `dbg_act_words_VLD` | dbg_act_words 有效标志 |
| `0x74` | `dbg_wgt_words_DATA` | 权重向量发射计数 |
| `0x78` | `dbg_wgt_words_VLD` | dbg_wgt_words 有效标志 |
| `0x84` | `dbg_psum_words_DATA` | 部分和发射计数 |
| `0x88` | `dbg_psum_words_VLD` | dbg_psum_words 有效标志 |
| `0x94` | `dbg_out_words_DATA` | 输出元素计数 |
| `0x98` | `dbg_out_words_VLD` | dbg_out_words 有效标志 |
| `0xa4` | `dbg_hw_version_DATA` | 硬件版本号（当前= `0x20250512`） |
| `0xa8` | `dbg_hw_version_VLD` | dbg_hw_version 有效标志 |

注意：调试寄存器在对应操作（CONV/POOL等）**返回后**才更新。操作执行期间各 _VLD 标志为 0，不可作为 live progress indicator。

### 4.3 已删除/预留项

以下寄存器在当前版本中**不存在**（perf_irq.cpp 为 stub，性能计数器预留至 AXI-Lite wrapper 阶段）：

- ~~`PERF_CYCLE`~~ 预留
- ~~`PERF_DDR_RD`~~ 预留
- ~~`PERF_DDR_WR`~~ 预留
- ~~`PERF_STALL`~~ 预留
- ~~`STATUS`~~（已拆入 dbg_status / dbg_heartbeat）

规则：

1. `FRAME_IN_BASE / FRAME_OUT_BASE / PARAM_BASE` 必须 `64-byte` 对齐
2. `UOP_COUNT` 必须等于 `param_blob` 头里的 `uop_count`
3. `MODE_INIT` 时只使用 `PARAM_BASE`
4. `MODE_RUN` 时 `PARAM_BASE / FRAME_IN_BASE / FRAME_OUT_BASE` 都必须有效

## 5. uop 二进制格式冻结

### 5.1 opcode

```cpp
enum uop_opcode_t : uint8_t {
    UOP_NOP      = 0,
    UOP_LOAD_FM  = 1,
    UOP_CONV     = 2,
    UOP_POOL     = 3,
    UOP_ADD      = 4,
    UOP_AFFINE   = 5,
    UOP_STORE    = 6,
    UOP_END      = 15
};
```

### 5.2 tensor id 命名空间
`src0_tensor / src1_tensor / dst_tensor` 使用统一 `uint8_t` namespace：

- `0x00 ~ 0x3F`：全局 tensor id
- `0x80 ~ 0x8F`：局部 scratch id
- `0xFF`：无效输入

### 5.3 256-bit packed uop
每条 `uop` 固定为 **32 bytes = 256 bits**，小端打包，字段顺序冻结如下：

```cpp
struct uop_t {
    uint8_t  opcode;        // [7:0]
    uint8_t  flags;         // [15:8]
    uint8_t  src0_tensor;   // [23:16]
    uint8_t  src1_tensor;   // [31:24]

    uint8_t  dst_tensor;    // [39:32]
    uint8_t  param_id;      // [47:40], opcode-specific
    uint8_t  act_type;      // [55:48]
    uint8_t  reserved0;     // [63:56], must be 0

    uint16_t in_h;          // [79:64]
    uint16_t in_w;          // [95:80]
    uint16_t in_c;          // [111:96]
    uint16_t out_c;         // [127:112]

    uint8_t  kernel;        // [135:128]
    uint8_t  stride;        // [143:136]
    uint8_t  dilation;      // [151:144]
    uint8_t  padding;       // [159:152]

    uint16_t c_offset;      // [175:160]
    uint16_t valid_c;       // [191:176]

    uint16_t qparam_id;     // [207:192], v1 current encoder fixed to 0
    uint16_t reserved1;     // [223:208], must be 0

    uint32_t reserved2;     // [255:224], must be 0
};
```

冻结规则：

1. `sizeof(uop_t) == 32`
2. `reserved0/1/2` 必须写 0
3. 所有 `uop` 按 32-byte 对齐顺序连续存放
4. 对当前 `ESPNet_Encoder v1`，所有 `uop.qparam_id` 固定写 `0`

### 5.4 flags 位定义

| Bit | Name | Meaning |
|---|---|---|
| `0` | `bias_en` | conv 后使用 bias |
| `1` | `relu_en` | 输出经过 relu |
| `2` | `requant_bypass` | `ADD` 直接 clamp，不走乘法 requant |
| `3` | `concat_mode` | 当前写回是 concat 子区间 |
| `4` | `alias_enable` | `CAT -> ACT` 地址别名使能；当前 Encoder 的 7 个 `AFFINE` uop 固定为 `1` |
| `5` | `pool_same_scale` | pool 输入/输出 scale 相同 |
| `6` | `last_uop_of_stage` | 调试/perf 阶段标记 |
| `7` | `reserved` | must be 0 |

## 6. descriptor 与参数表冻结

### 6.1 tensor descriptor
每个全局 tensor 一条固定 descriptor，大小 **16 bytes**：

```cpp
struct tensor_desc_t {
    uint8_t  bank_id;
    uint8_t  elem_bytes;    // v1 fixed to 1
    uint16_t reserved0;     // physical_c_stride; 0 means logical c
    uint32_t base_offset;   // byte offset within bank
    uint16_t h;
    uint16_t w;
    uint16_t c;
    uint16_t reserved1;     // channel_offset within physical row
};
```

冻结规则：

1. `sizeof(tensor_desc_t) == 16`
2. `base_offset` 单位是 byte
3. `reserved0` 固定解释为 `physical_c_stride`
4. `reserved1` 固定解释为 `channel_offset`
5. compact tensor 导出为 `reserved0 = c, reserved1 = 0`
6. view tensor 通过 `physical_c_stride/channel_offset` 映射到共享物理容器
7. `bank_id` 仅允许：
   - `0` = `FMEM0`
   - `1` = `FMEM1`
   - `2` = `FMEM2`
   - `0x80` = `BRAM_SCR0`
   - `0x81` = `BRAM_SCR1`

当前硬件 memory contract 下，`FMEM0/1/2` 是逻辑 bank/view 语义，物理落点统一进入共享 `FMBUF`。`T_L2B0_CAT/ACT` 是 `T_B2_CAT/ACT c[0:63]` 的 view，`T_L3B0_CAT/ACT` 是 `T_B3_CAT/ACT c[128:255]` 的 view。离线导出器必须把这些 view 写进 `tensor_desc_table`，不能再导出三份独立大 FMEM。

### 6.2 scale descriptor
所有 activation scale 统一进入 `scale_desc_table`，每条 **8 bytes**：

```cpp
struct scale_desc_t {
    int32_t mult;
    uint8_t shift;
    uint8_t reserved[3];
};
```

说明：

- `mult/shift` 用于把某一路源数据 requant 到目标 tensor scale
- 对称量化已冻结 `zp = 0`，因此这里不再保留 `zero_point`

### 6.3 opcode-specific 参数 descriptor
`param_id` 是 **opcode-specific namespace**，不再共用一个模糊的通用描述表。

#### CONV

```cpp
struct conv_param_desc_t {
    uint32_t weight_offset;   // packed weight tiles
    uint32_t bias_offset;     // conv_qparam_t array
    uint32_t requant_offset;  // conv_qparam_t array
    uint32_t reserved;        // must be 0
};
```

#### AFFINE

```cpp
struct affine_param_desc_t {
    uint32_t affine_offset;   // affine_qparam_t array
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
};
```

#### ADD

```cpp
struct add_param_desc_t {
    uint32_t add_offset;      // add_qparam_t
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
};
```

#### POOL

```cpp
struct pool_param_desc_t {
    uint32_t pool_offset;     // pool_qparam_t
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
};
```

### 6.4 qparam 实体
真正的数值参数表冻结如下。

#### conv_qparam_t

```cpp
struct conv_qparam_t {
    int32_t bias[32];
    int32_t mult[32];
    uint8_t shift[32];
    uint8_t reserved[32];
};
```

#### affine_qparam_t

```cpp
struct affine_qparam_t {
    int32_t mul[32];
    int32_t bias[32];
    uint8_t shift[32];
    uint8_t reserved[32];
};
```

#### add_qparam_t

```cpp
struct add_qparam_t {
    int32_t mult;
    uint8_t shift;
    uint8_t act_type;
    uint8_t requant_bypass;
    uint8_t reserved0;
    uint32_t src_scale_id_a;
    uint32_t src_scale_id_b;
    uint32_t dst_scale_id;
    uint32_t reserved1;
};
```

#### pool_qparam_t

```cpp
struct pool_qparam_t {
    uint8_t kernel;          // v1 fixed to 3
    uint8_t stride;          // v1 fixed to 2
    uint8_t same_scale;      // 1 = output uses input scale
    uint8_t act_type;        // v1 fixed to ACT_NONE
    uint32_t src_scale_id;
    uint32_t dst_scale_id;
    int32_t mult;            // used when same_scale = 0
    uint8_t shift;
    uint8_t reserved[11];
};
```

## 7. param blob 内存镜像冻结
运行时硬件只读取 **一个** `param_blob.bin`。其内存镜像冻结如下：

### 7.1 头部
头部固定 **128 bytes**：

```cpp
struct param_blob_header_t {
    uint32_t magic;               // 'EINT' = 0x544E4945
    uint32_t version;             // v1 = 0x00010000

    uint32_t tensor_desc_count;
    uint32_t scale_desc_count;
    uint32_t conv_desc_count;
    uint32_t affine_desc_count;
    uint32_t add_desc_count;
    uint32_t pool_desc_count;

    uint32_t uop_count;
    uint32_t reserved0;

    uint32_t tensor_desc_offset;
    uint32_t scale_desc_offset;
    uint32_t conv_desc_offset;
    uint32_t affine_desc_offset;
    uint32_t add_desc_offset;
    uint32_t pool_desc_offset;

    uint32_t uop_offset;
    uint32_t weight_data_offset;
    uint32_t conv_qparam_offset;
    uint32_t affine_qparam_offset;
    uint32_t add_qparam_offset;
    uint32_t pool_qparam_offset;

    uint32_t reserved1[10];
};
```

冻结规则：

1. 所有 offset 相对于 `param_blob.bin` 起始地址
2. 所有 section 起始地址必须 `64-byte` 对齐
3. `magic/version` 不匹配时，`MODE_INIT` 必须报错

### 7.2 section 顺序
虽然头部提供 offset，但首版离线导出固定采用以下顺序：

1. `header`
2. `tensor_desc_table`
3. `scale_desc_table`
4. `conv_desc_table`
5. `affine_desc_table`
6. `add_desc_table`
7. `pool_desc_table`
8. `uop_table`
9. `weight_data`
10. `conv_qparam_data`
11. `affine_qparam_data`
12. `add_qparam_data`
13. `pool_qparam_data`

## 8. bring-up 相关冻结

### 8.1 中断
首版允许 polling，但 IP 必须保留一个 `irq` 输出。

### 8.2 错误码

| Code | Meaning |
|---|---|
| `0` | OK |
| `1` | invalid mode |
| `2` | bad blob magic/version |
| `3` | uop decode error |
| `4` | tensor desc out of range |
| `5` | param desc out of range |
| `6` | bank overflow |
| `7` | unsupported opcode |

### 8.3 首版 bring-up 顺序
首版点亮顺序冻结如下：

1. `MODE_INIT`
2. `MODE_RUN` 控制域空调度：解析完整 `75` 条 uop，只执行 `LOAD_FM/STORE_DDR`
3. `STORE/CONCAT` tile 路径：覆盖 B1/B2/B3 channel slice 与 view no-op
4. `POOL` 路径：覆盖 `POOL_B1 / POOL_B2_TMP / POOL_B2_OUT`
5. 单层 `CONV + POST`
6. `ADD / AFFINE`
7. 完整 Encoder `uop_table`

## 9. 验收标准
本文件对应的系统集成验收条件为：

1. CPU 侧只需两次任务提交：`INIT` 和 `RUN`
2. 单帧运行中不再发生逐层 AXI DMA 提交
3. 中间特征图不回 DDR
4. 输出与 fake-quant golden 一致
5. `param_blob.bin` 可被硬件一次性解析
6. 性能计数器可读出真实 `cycle / ddr_rd / ddr_wr / stall`
