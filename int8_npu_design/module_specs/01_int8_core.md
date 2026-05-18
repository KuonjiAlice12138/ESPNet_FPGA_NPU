# int8_core Spec

## 1. 对应关系

- 逻辑模块：`espnet_encoder_int8_core`
- 当前实现文件：`src/int8_core.cpp`

## 2. 来源整理
本文件由以下共享文档内容重组而成：

- `01_System_Architecture.md` 的顶层架构、数据流、资源与时延约束
- `06_DMA_and_AXI_Interface.md` 的 HLS 核心接口与运行模式
- `07_ESPNet_Encoder_Uop_Schedule.md` 的整网执行顺序约束

## 3. 模块职责
`espnet_encoder_int8_core` 是首版单一 HLS 核心，负责：

1. 接收 DDR 中的输入帧、参数 blob 和控制参数
2. 在 `MODE_INIT` 下完成整模参数预加载
3. 在 `MODE_RUN` 下执行完整 encoder `uop table`
4. 组织内部 dataflow 子模块，不允许 CPU 逐层 dispatch

它不是软件可见包装 IP。软件可见层仍是后续薄包装的 `espnet_encoder_int8_top`。

## 4. 冻结接口

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

推荐 pragma 冻结为：

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

## 5. 内部子模块边界
`espnet_encoder_int8_core` 内部固定连接以下逻辑模块：

1. `instruction_fetch_decode`
2. `frame_dma`
3. `param_dma`
4. `on_chip_memory`
5. `window_generator`
6. `systolic_array_core`
7. `post_process_unit`
8. `avgpool_unit`
9. `concat_writer`
10. `perf_counter_and_irq`

## 6. 运行模式

### 6.1 `MODE_INIT`

- 只使用 `PARAM_BASE`
- 从 `param_blob.bin` 加载全部静态参数到片上
- 包括权重、量化参数、descriptor 表和 `uop table`

### 6.2 `MODE_RUN`

- 读取一帧 `input_q`
- 执行完整 encoder 调度
- 只允许“输入一次读 DDR、输出一次写 DDR”
- 中间特征图必须全部驻留片上

## 7. 数据流冻结

- 数据流类型：`weight-stationary`
- 权重驻留：`WBUF`
- 特征图驻留：`FMBUF`
- 卷积输出：`INT32`
- 层间流转：`INT8`
- `avgpool / add / affine / concat` 不允许回到 CPU

## 8. 首版实现约束

1. 首版目标频率：`100 MHz`
2. 不再采用 FP32 多独立 IP + 多 DMA 模式
3. 中间特征图不得写回 DDR
4. 参数一次性预加载
5. 控制流由 PL 内部顺序执行，CPU 只提交 `INIT/RUN`

## 9. 模型覆盖范围
本模块只要求覆盖当前已冻结的 `ESPNet_Encoder` 路径：

1. `Level1`
2. `B1`
3. `Level2_0`
4. `Level2_Block0`
5. `B2`
6. `Level3_0`
7. `Level3_Block0`
8. `B3`
9. `Classifier`

## 10. 片上资源与时延约束

- 物理 `FMBUF = 0x598000`
- `FMBUF_URAM = 0x380000`
- `FMBUF_BRAM = 0x218000`
- `FMEM0/1/2` 只保留逻辑 bank/view 语义
- 全模型 INT8 权重约 `108 KB`
- 首版目标是 `100 MHz` 下达到接近 `50 ms` 级可行实现

## 11. 验收重点

1. `MODE_INIT` 可正确解析 `param_blob`
2. `MODE_RUN` 可闭环完成整网执行
3. 输出与 fake-quant golden 一致
4. 不退化成层间 DDR spill
5. 具备 `cycle / ddr_rd / ddr_wr / stall` 统计能力
