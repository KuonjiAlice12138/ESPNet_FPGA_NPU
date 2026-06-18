# ESPNet INT8 NPU 架构与 Demo 基线说明

> 更新时间：2026-06-15
> Demo 建议基线：`platform_full100_0615` / `INT8-BOARD-20260615-P6J-TC100-VAL`
> 说明：本文用于组会和 demo slides。性能时间统一按 A53 `CNTPCT_EL0` timestamp tick 换算，当前平台 `CNTFRQ_EL0 = 33,333,000 Hz`，不再使用旧的 `100MHz/10ns` app 换算口径。

## 1. 当前结论

当前 INT8 NPU 已完成 ESPNet Encoder 的板端闭环：PS 端从 SD 卡读取输入和参数，PL 端执行 `MODE_INIT -> MODE_RUN`，NPU 输出 `512x1024` full-resolution 二分类 mask，再由主机离线统计 PA/mIoU。

Demo 建议使用最新 `P6J` timing-clean 版本，而不是早期 `full125` 探索版。理由如下：

| 版本 | 定位 | 结果 |
|---|---|---|
| `platform_full100_0615` | P6J timing-clean demo 基线 | 100 MHz routed timing clean，单图真实延迟约 `1003 ms`，500 张验证集输出完整，PA/mIoU 有效 |
| `platform_full100_0527` | full-resolution 功能/精度旧基线 | 100 MHz routed timing clean，平均真实延迟约 `1254 ms` |
| `platform_full125_0603` | 125 MHz timing-clean 探索 | 能跑通单图，但真实 `MODE_RUN` 延迟退化到 `1.748 s`，不适合作为 demo 性能线 |

需要明确的是：`P6J` 也不是实时 demo。它的价值是证明“模型量化 + NPU 片上推理 + full-resolution mask 输出 + 验证集精度评估”完整闭环，并展示相对旧 `full100` 约 `20%` 的端到端延迟改善，而不是证明实时推理。

## 2. Demo 基线数据

### 2.1 P6J 板端闭环

| 项目 | 数据 |
|---|---|
| Platform | `platform_full100_0615` |
| App tag | `INT8-BOARD-20260615-P6J-TC100-VAL` |
| PL clock | `100 MHz` |
| 输出格式 | `512x1024` uint8 mask，`0=target, 1=background` |
| 输出文件 | `O0000.BIN` 到 `O0499.BIN`，每个 `524288 bytes` |
| Routed timing | WNS `+0.139 ns`，TNS `0` |
| 资源 | CLB LUTs `56.72%`，CLB `91.79%`，BRAM Tile `97.58%`，URAM `100%`，DSP `35.74%` |

计时统一按 A53 timestamp tick 换算。P6J 单图测试结果：

| 项目 | 数据 |
|---|---:|
| 单图 `MODE_RUN` | `33,434,687` A53 timer ticks |
| A53 timer 频率 | `33,333,000 Hz` |
| 单图真实 wall-time | 约 `1.003 s` |
| 等效 100 MHz PL cycles | 约 `100.3 M cycles` |
| 相对 `full100_0527` | 约 `20.0%` 延迟下降 |

旧 `full100_0527` 基线为 `41,788,175` A53 ticks，约 `1.254 s`，等效约 `125.37 M` 100MHz PL cycles。P6J 已压缩端到端周期，但距离 `100 ms` 级目标仍约 `10x`。

### 2.2 P6J 精度

板端 full-resolution mask 与软件侧 full-resolution baseline 对比：

| 评估方式 | PA | mIoU |
|---|---:|---:|
| software fullres bilinear logits argmax | `0.97854548` | `0.86874892` |
| software fullres nearest mask | `0.97718681` | `0.86281516` |
| board fullres mask (`full100`) | `0.97800421` | `0.86356491` |
| board fullres mask (`P6J`) | `0.97800421` | `0.86356491` |

这个结果说明 P6J 板端输出没有显著精度劣化。由于硬件输出是 uint8 mask，不再保留 full-resolution 两通道 logits，因此 demo 中建议强调“最终 mask 质量”和“验证集 PA/mIoU”，不要声称与 PyTorch float bilinear logits bit-exact。

### 2.3 full125 反例

`platform_full125_0603` 已加入计时器自检：

```text
cntfrq=33333000 Hz
bsp=33333000 Hz
100ms selftest error=9982 ppm PASS
FULL_MODE_RUN=58253035 ticks = 1748 ms
```

SD 输出 `MASK.BIN` 检查通过：大小 `524288 bytes`，仅包含 0/1 两类。但该版本真实延迟约 `1.748 s`，比 `full100` 更慢，因此不作为 demo 数据。

## 3. 顶层系统

板端系统由 PS app、DDR buffer、SD 文件和 PL NPU IP 组成。

```text
SD Card
  ├─ PARAM.BIN       tensor desc、qparam、weight、UOP count/header metadata
  ├─ INPUTQ.BIN      单图输入，或 I0000.BIN...I0499.BIN 验证集输入
  └─ MASK.BIN/Oxxxx  NPU 输出 full-resolution mask

PS App
  ├─ 挂载 SD 卡
  ├─ 读取 PARAM.BIN 和输入图像到 DDR buffer
  ├─ 通过 AXI-Lite 配置 NPU 控制寄存器
  ├─ 启动 MODE_INIT / MODE_RUN
  └─ 保存 full-resolution mask 到 SD 卡

PL NPU IP
  ├─ AXI-Lite control
  ├─ M_AXI gmem0: input frame
  ├─ M_AXI gmem1: output mask
  ├─ M_AXI gmem2: param blob
  └─ 片上 feature/weight/qparam buffer 与静态 ESPNet graph scheduler
```

顶层 IP 为 `espnet_encoder_int8_core`。正常运行流程只需要一次 `MODE_INIT` 加载参数，然后对每张图执行一次 `MODE_RUN`。

| 端口 | 作用 |
|---|---|
| `gmem_frame_in` | DDR 输入图像 buffer，M_AXI 读 |
| `gmem_frame_out` | DDR 输出 mask buffer，M_AXI 写 |
| `gmem_param` | DDR 参数 blob buffer，M_AXI 读 |
| `mode` | `MODE_INIT` 或 `MODE_RUN` |
| `uop_count` | 整网 UOP 数量，当前为 75 |

## 4. 顶层硬件模块

当前设计不再按早期 `if_dec.cpp` 取指译码器来组织顶层控制，而是采用“静态 ESPNet graph scheduler + 统一 UOP 描述格式”的实现。UOP 仍作为内部算子描述结构保留，但整网 75 个算子的顺序、shape、tensor id 和 param id 已固化在 HLS 顶层调度逻辑中；`PARAM.BIN` 主要提供权重、量化参数、tensor metadata 和一致性校验信息。

| 架构模块 | 主要职责 | 当前源码承载 |
|---|---|---|
| AXI/Control Shell | AXI-Lite 控制寄存器、三路 M_AXI 端口、`MODE_INIT/MODE_RUN` 分发 | `int8_core.cpp` |
| Parameter Manager | 解析 `PARAM.BIN`，加载 tensor desc、weight、conv/add/affine/pool qparam | `param_dma.cpp` |
| Static Graph Scheduler | 固化 ESPNet Encoder 的 75 条 UOP 等价调度，完成 opcode 分派和 conv-store fusion | `int8_core.cpp` |
| Frame DMA | 输入图像从 DDR 进入片上 feature buffer，输出 mask 写回 DDR | `frame_dma.cpp` |
| On-Chip Feature Memory | feature map BRAM/URAM 存储、packed word 读写、bank/region 映射 | `memory.cpp` |
| Scratch Manager | tensor id 到全局/局部 scratch 区域的解析，控制中间特征生命周期 | `scratch_mgr.cpp` |
| Convolution Engine | window 生成、weight stream、INT8 MAC 阵列、psum 后处理和行写回 | `win_gen.cpp`、`sa_core.cpp`、`conv_store.cpp`、`int8_core.cpp` |
| Non-Conv Operators | Pool、Add、Affine、Store/Concat 等非卷积算子 | `avgpool_unit.cpp`、`concat_unit.cpp`、`int8_core.cpp` |
| Full-Resolution Output | `64x128x2` logits 上采样并 argmax，输出 `512x1024` mask | `upsample_unit.cpp` |

早期 `if_dec.cpp` 的逻辑已被拆分和静态化：UOP 构造/选择进入 `StaticUop<ID>` 和 `build_p6_static_uop()`，opcode 分派进入 `run_p6_dispatch_uop()`，整网循环进入 `run_p6_static_graph()`。因此汇报时不建议再把 `IF/Decode Unit` 画成独立硬件模块。

已废弃的历史路径：

| 历史项 | 当前状态 |
|---|---|
| 独立 `ppu.cpp` | 已移除；功能拆入 conv post-process、Add/Affine/Pool/Store |
| 独立 `if_dec.cpp` | 已移除；动态取指/译码被静态 graph scheduler 替代 |
| pair2/dual path | 曾上板验证，但端到端退化，不作为 demo 主线 |
| profiling/debug 端口 | 只用于定位，不作为 demo 依赖 |

## 5. UOP 调度

当前硬件不是动态解释 `PARAM.BIN` 中的 UOP 序列，而是使用编译期固化的 ESPNet Encoder 静态调度表。UOP 仍是统一的内部算子描述格式，用来承载 opcode、输入/输出 tensor、shape、kernel/stride/dilation、param id、channel offset 等字段；顶层按 UOP id 顺序生成描述并分派到对应算子单元。

```text
MODE_INIT:
  param_dma_init(gmem_param)
  validate PARAM.BIN header / uop_count
  load tensor desc / qparam / weight into on-chip buffers

MODE_RUN:
  frame_dma_load(gmem_frame_in)
  for static uop_id = 1..72:
      build static UOP descriptor
      dispatch to CONV / POOL / ADD / AFFINE / STORE
      apply graph-level fusion when enabled
  final CONV/UPSAMPLE path writes full-resolution mask
```

当前 UOP 规模：

| 类型 | 数量 |
|---|---:|
| CONV | 26 |
| POOL | 3 |
| ADD | 14 |
| AFFINE | 7 |
| STORE/CONCAT | 23 |
| LOAD | 1 |
| END | 1 |
| 总计 | 75 |

## 6. 卷积数据流

卷积是 NPU 主负载。每个 `CONV` UOP 按输出行调度，形成 row-level stream datapath。

```text
for each output row:
  window_generator_row
      -> act_stream
  feed_cached_weights
      -> wgt_stream
  systolic_array_core_row
      -> psum_stream
  post_process_row_to_buffer
      -> row_buf
  store_conv_output_row(row_buf)
```

`TM=32` 表示最多 32 个输出通道并行；`TK=32` 表示 reduction 维度按 32-lane K tile 输入。对于 ESPNet 中大量小通道卷积，阵列 lane 填充率并不总是高，这也是当前性能不理想的根本原因之一。

当前 `win_gen` 已包含若干固定 shape fast path：

| Path | 目标 |
|---|---|
| first-layer C3 3x3 stride2 | 输入图像第一层 |
| small-C 3x3 stride1 reuse | `C=12/19/25/28` 等 ESPNet 小通道卷积 |
| C19/C131 stride2 | level2/level3 下采样入口 |
| 1x1 aligned full | 对齐通道块 1x1 卷积 |
| generic path | 未命中专用形状时兜底 |

## 7. Full-Resolution 输出

`full100` demo 版不再输出 `64x128x2` logits，而是在 PL 侧完成固定 8x 上采样和 argmax：

```text
64x128x2 INT8 logits
  -> bilinear upsample to 512x1024x2
  -> argmax over 2 classes
  -> 512x1024 uint8 mask
  -> M_AXI write to gmem_frame_out
```

这样 demo 时 PS 端只需要保存 `MASK.BIN` 或 `Oxxxx.BIN`，主机端可直接读取 full-resolution mask 计算 PA/mIoU 或显示分割结果。该路径牺牲了 logits 级可比对能力，但更适合展示完整分割输出。

## 8. Demo 讲法

组会或演示建议使用如下叙事：

1. 任务场景：目标-背景二分类分割，输入 `512x1024`，输出 full-resolution mask。
2. 模型侧：ESPNet Encoder 经过硬件约束 INT8 量化，导出 `PARAM.BIN` 和 INT8 输入。
3. 硬件侧：PS 只负责 SD/DDR 和寄存器控制，PL NPU 按静态 graph scheduler 执行与 75 条 UOP 对齐的算子序列。
4. 输出侧：NPU 直接输出 `512x1024` mask，不需要 PS 做上采样。
5. 正确性：P6J 版 500 张验证集输出完整，PA `0.9780`，mIoU `0.8636`。
6. 工程闭环：100 MHz routed timing clean，资源接近上限但实现可用。
7. 性能：P6J 单图真实延迟约 `1.003 s/image`，比旧 full100 基线约快 `20%`。
8. 局限：当前仍不具备实时 demo 性能。

不建议继续使用以下表述：

| 不建议 | 原因 |
|---|---|
| `417 ms full100` | 旧 app 把 A53 timer tick 错按 100 MHz 换算 |
| `329 ms PERF125` | 同样是旧换算，且该版本 routed timing 不 clean |
| `125 MHz fullres 性能更好` | 当前 full125 上板结果实际更慢 |
| `实时推理` | 当前可信数据不支持 |

## 9. 性能瓶颈与下一步

当前性能问题是架构级的，不是单个 `upsample` 或 Vivado strategy 能解决。

| 瓶颈 | 现象 |
|---|---|
| 小通道卷积阵列利用率低 | ESPNet 中大量小 `out_c/in_c` 卷积无法填满 32 lane |
| window generation 与片上 memory 路径重 | 多数固定 shape 仍受 row/window read/pack 和 memory back-pressure 限制 |
| 非卷积 UOP 数量多 | ADD/AFFINE/STORE/CONCAT 带来大量片上 memory 往返 |
| URAM 满载 | 不能通过简单复制 buffer 或多路数据流换性能 |
| 125 MHz 收敛代价高 | 为 timing 做出的结构改动可能显著增加总 tick |

如果目标是可演示速度，下一步不应继续在当前 `512x1024` fullres 版本上小修小补。更现实的路线是：

1. 做一个 `256x512` 或 `128x256` demo profile，先在模型侧评估精度，再重导出 hardware artifacts。
2. 若坚持 `512x1024` 输入，则需要架构级重构：更多并行阵列、算子融合、减少 memory 往返，而不是继续微调 HLS pragma。
3. 论文中可保留当前 full100 作为“完整片上推理闭环与精度验证”结果，但性能目标需要重新措辞，不能再用旧的 ms 表。

## 10. Slides 建议

| Slide | 内容 |
|---|---|
| 1 | 任务目标：ESPNet INT8 FPGA NPU full-resolution 分割输出 |
| 2 | PS/PL/SD/DDR 顶层系统 |
| 3 | `PARAM.BIN`、静态 graph scheduler、`MODE_INIT/MODE_RUN` |
| 4 | 顶层硬件模块与源码承载关系 |
| 5 | 片上 feature memory 与 256-bit packed word |
| 6 | Conv row-level stream datapath |
| 7 | `TM=32/TK=32`、K tile 和小通道利用率问题 |
| 8 | Full-resolution upsample + argmax 输出 |
| 9 | full100 demo 数据：timing clean、PA/mIoU、真实延迟 |
| 10 | 局限与后续：继续做架构级并行化、减少 memory pass，而不是只调 Vivado strategy |
