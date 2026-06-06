# ESPNet INT8 NPU 架构与 Demo 基线说明

> 更新时间：2026-06-03  
> Demo 建议基线：`platform_full100_0527` / `INT8-BOARD-20260527-FULL100-UPFULL-VAL`  
> 说明：本文用于组会和 demo slides。性能时间统一按 A53 `CNTPCT_EL0` timestamp tick 换算，当前平台 `CNTFRQ_EL0 = 33,333,000 Hz`，不再使用旧的 `100MHz/10ns` app 换算口径。

## 1. 当前结论

当前 INT8 NPU 已完成 ESPNet Encoder 的板端闭环：PS 端从 SD 卡读取输入和参数，PL 端执行 `MODE_INIT -> MODE_RUN`，NPU 输出 `512x1024` full-resolution 二分类 mask，再由主机离线统计 PA/mIoU。

Demo 建议使用 `full100` 版本，而不是后续 `full125` 探索版。理由如下：

| 版本 | 定位 | 结果 |
|---|---|---|
| `platform_full100_0527` | full-resolution 功能/精度 demo 基线 | 100 MHz routed timing clean，500 张验证集输出完整，PA/mIoU 有效 |
| `platform_full125_0603` | 125 MHz timing-clean 探索 | 能跑通单图，但真实 `MODE_RUN` 延迟退化到 `1.748 s`，不适合作为 demo 性能线 |

需要明确的是：`full100` 也不是实时 demo。它的价值是证明“模型量化 + NPU 片上推理 + full-resolution mask 输出 + 验证集精度评估”完整闭环，而不是证明实时推理。

## 2. Demo 基线数据

### 2.1 full100 板端闭环

| 项目 | 数据 |
|---|---|
| Platform | `platform_full100_0527` |
| App tag | `INT8-BOARD-20260527-FULL100-UPFULL-VAL` |
| PL clock | `100 MHz` |
| 输出格式 | `512x1024` uint8 mask，`0=target, 1=background` |
| 输出文件 | `O0000.BIN` 到 `O0499.BIN`，每个 `524288 bytes` |
| Routed timing | WNS `+0.073 ns`，TNS `0` |
| 资源 | LUT `73.35%`，FF `25.39%`，BRAM Tile `86.02%`，URAM `100%`，DSP `39.77%` |

旧 app 打印的 `avg_ms=417` 是错误换算；它实际记录的是 A53 timestamp tick。修正后：

| 项目 | 数据 |
|---|---:|
| 平均单图 `MODE_RUN` | `41,788,175` A53 timer ticks |
| A53 timer 频率 | `33,333,000 Hz` |
| 平均真实 wall-time | 约 `1.254 s` |
| 等效 100 MHz PL cycles | 约 `125.37 M cycles` |
| 500 张验证集总时间 | 约 `626.8 s` |

### 2.2 full100 精度

板端 full-resolution mask 与软件侧 full-resolution baseline 对比：

| 评估方式 | PA | mIoU |
|---|---:|---:|
| software fullres bilinear logits argmax | `0.97854548` | `0.86874892` |
| software fullres nearest mask | `0.97718681` | `0.86281516` |
| board fullres mask (`full100`) | `0.97800421` | `0.86356491` |

这个结果说明板端输出没有显著精度劣化。由于硬件输出是 uint8 mask，不再保留 full-resolution 两通道 logits，因此 demo 中建议强调“最终 mask 质量”和“验证集 PA/mIoU”，不要声称与 PyTorch float bilinear logits bit-exact。

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
  ├─ PARAM.BIN       UOP、tensor desc、qparam、weight
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
  └─ 片上 feature/weight/qparam/uop buffer
```

顶层 IP 为 `espnet_encoder_int8_core`。正常运行流程只需要一次 `MODE_INIT` 加载参数，然后对每张图执行一次 `MODE_RUN`。

| 端口 | 作用 |
|---|---|
| `gmem_frame_in` | DDR 输入图像 buffer，M_AXI 读 |
| `gmem_frame_out` | DDR 输出 mask buffer，M_AXI 写 |
| `gmem_param` | DDR 参数 blob buffer，M_AXI 读 |
| `mode` | `MODE_INIT` 或 `MODE_RUN` |
| `uop_count` | 整网 UOP 数量，当前为 75 |

## 4. 模块边界

当前 HLS 代码按功能模块划分如下。这里的模块基本对应 `src/*.cpp`，但卷积 post-process 和 row writeback 保留在顶层文件内，以避免 HLS dataflow feedback 和 process merging 风险。

| 文件 | 功能 |
|---|---|
| `int8_core.cpp` | 顶层控制、UOP 顺序调度、Conv/Add/Affine/Store 执行入口、卷积 row buffer、producer-store fusion |
| `param_dma.cpp` | 解析 `PARAM.BIN`，加载 UOP、tensor desc、qparam 和 weight buffer |
| `if_dec.cpp` | UOP fetch/decode 与基本合法性检查 |
| `frame_dma.cpp` | 输入 frame load；输出阶段调用 full-resolution upsample store |
| `memory.cpp` | 片上 feature memory bank 映射、packed tile read/write、aligned row write |
| `win_gen.cpp` | 卷积 activation window 生成 |
| `sa_core.cpp` | `TM=32/TK=32` INT8 MAC 阵列，输出 INT32 psum |
| `avgpool_unit.cpp` | C3 3x3 stride2 avgpool 和 pool requant |
| `concat_unit.cpp` | STORE/CONCAT channel slice copy |
| `upsample_unit.cpp` | `64x128x2` logits 到 `512x1024` mask 的 bilinear upsample + argmax |

已废弃的历史路径：

| 历史项 | 当前状态 |
|---|---|
| 独立 `ppu.cpp` | 已移除；功能拆入 conv post-process、Add/Affine/Pool/Store |
| pair2/dual path | 曾上板验证，但端到端退化，不作为 demo 主线 |
| profiling/debug 端口 | 只用于定位，不作为 demo 依赖 |

## 5. UOP 调度

硬件不是把整网展开成固定 RTL pipeline，而是顺序解释执行 `PARAM.BIN` 中的 UOP 序列。

```text
MODE_INIT:
  param_dma_init(gmem_param)
  instruction_fetch_decode(gmem_param, uop_count)

MODE_RUN:
  frame_dma_load(gmem_frame_in)
  for each uop:
      execute LOAD / CONV / POOL / ADD / AFFINE / STORE / END
  frame_dma_store(gmem_frame_out)
      -> upsample_logits_bilinear_argmax_store()
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
3. 硬件侧：PS 只负责 SD/DDR 和寄存器控制，PL NPU 执行 75 条 UOP。
4. 输出侧：NPU 直接输出 `512x1024` mask，不需要 PS 做上采样。
5. 正确性：500 张验证集输出完整，PA `0.9780`，mIoU `0.8636`。
6. 工程闭环：100 MHz routed timing clean，资源接近上限但实现可用。
7. 局限：真实平均延迟约 `1.254 s/image`，当前不具备实时 demo 性能。

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
| 3 | `PARAM.BIN`、UOP 序列、`MODE_INIT/MODE_RUN` |
| 4 | HLS 模块边界与 `.cpp` 文件对应关系 |
| 5 | 片上 feature memory 与 256-bit packed word |
| 6 | Conv row-level stream datapath |
| 7 | `TM=32/TK=32`、K tile 和小通道利用率问题 |
| 8 | Full-resolution upsample + argmax 输出 |
| 9 | full100 demo 数据：timing clean、PA/mIoU、真实延迟 |
| 10 | 局限与后续：降分辨率 demo profile 或架构级并行化 |

