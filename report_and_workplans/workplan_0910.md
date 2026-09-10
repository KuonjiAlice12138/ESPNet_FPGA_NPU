# Workplan 0910 - H256W512 NPU 向 Roofline 岭点收敛

日期：2026-09-10
当前基线：`P7-H256W512-R3R`，PARAM v4，75 条 UOP，16 条 EXEC，
单套 32x32 INT8 MAC 阵列，100 MHz。

> **执行规则：** 每一轮必须完成验证和实测后，才能进入下一轮。
> 性能判断以上板 RTL cycle 为准，不以 HLS 的最大延时估计为准。

**总体目标：** 在不复制计算数据通路、不破坏 100 MHz 可实现性的前提下，
使 H256W512 NPU 从当前的片上数据搬运受限工作点，逐步靠近计算能力与片上
数据供给能力相匹配的 Roofline 岭点。

**架构路线：** 保持当前由 PARAM 调度驱动的 P7 架构，围绕现有
`WinGen -> SA -> 卷积后处理`流水线改善各模块的服务速率匹配。先消除不必要的
部分和（psum）输出阻塞，再摊薄逐行启动/排空流水线的固定开销，最后减少 WinGen 重复
取数和串行尾部处理时间。

**工具与器件：** Vitis HLS 2025.1、Vivado 2025.1、Zynq UltraScale+
`xczu15eg-ffvb1156-2-i`、C++/HLS stream、PARAM v4 编译器及整数 replay 工具。

---

## 0. 核心结论

首次 H256W512 上板结果有效，并达到此前设定的 `<=150 ms` 目标：

- binary2：`13,893,492 PL cycles = 138.935 ms`；
- cityscapes20：`13,911,890 PL cycles = 139.119 ms`；
- 两个模型仅相差 `18,398 cycles`，即 `0.132%`；
- 两次运行均为 `ERROR=0`，输出类别范围有效，ARM 计时与 RTL 计数一致。

当前 NPU **不是计算受限**，也**不是外部 DDR 带宽受限**，而是明显的
**片上数据流/传输受限**。主要时间消耗来自 WinGen 数据供给、SA 部分和（psum）排出、
卷积后处理、行缓冲交接，以及每行重新填充和排空流水线。

因此，下一版应优先提高现有 32x32 阵列在时间维度上的有效工作比例。扩大或复制
SA 只会提高理论峰值，不能消除已经观测到的等待，还会重新引入早期 P7 版本中
出现过的布局布线失败风险。

## 1. 当前进度与发布基线

### 1.1 功能状态

- HLS 当前固定输入为 `H=256, W=512, C=3`，低分辨率 logits 为
  `H=32, W=64`，最终 mask 为 `H=256, W=512`。
- binary2 与 cityscapes20 使用相互独立且已对齐的 artifact：
  - `hw_artifacts/binary2_int8_h256w512_v4`；
  - `hw_artifacts/cityscapes20_int8_h256w512_v4`。
- PARAM 保持 v4；U71 与 U72 仍然分离；失败的 PARAM v5/U71 融合路径已退出。
- 两个模型的整网 CSim 中，logits 和 mask 均与各自整数 replay 保持 bit-exact。
- 紧凑特征存储空间为 `0x1F0000` 字节，其中 `0x1C0000` 为 URAM，
  `0x30000` 为 BRAM；U54/L3B0 C1 scratch 生命周期冲突已经修复。
- 上板输出类别直方图均非退化结果，且不存在非法类别 ID。PA/mIoU 仍需由主机端
  使用完整验证集单独评估，不能根据单张图的直方图推断。

### 1.2 综合与实现状态

| 项目 | 当前结果 |
|---|---:|
| HLS 目标周期 / 估计周期 | `10.000 ns / 7.300 ns` |
| 布线后工作频率 | `100 MHz` |
| 布线后 WNS / WHS | `+0.737 ns / +0.010 ns` |
| Routing error | `0` |
| CLB LUT / LUT as logic | `231,748 (67.91%) / 206,563 (60.53%)` |
| FF | `205,516 (30.11%)` |
| BRAM tile | `246 (33.06%)` |
| URAM | `56 (50.00%)` |
| DSP | `1,318 (37.36%)` |

这是一版可用且可实现的物理基线。当前 BRAM 和 DSP 仍可容纳小规模、有明确边界的
行缓冲或更宽的共享后处理通路，但资源和绕线余量不足以支持第二套 SA、第二套
WinGen 或大范围复制的 cache 层级。

## 2. 上板周期归因

### 2.1 顶层阶段

| 阶段 | binary2 cycles | binary2 占比 | city20 cycles | city20 占比 |
|---|---:|---:|---:|---:|
| 空闲/采样边界（IDLE） | `277` | `<0.01%` | `273` | `<0.01%` |
| 输入帧加载（FRAME_LOAD） | `95,708` | `0.68%` | `95,710` | `0.68%` |
| 主控制（MAIN_CTRL） | `932` | `<0.01%` | `932` | `<0.01%` |
| 卷积权重装载（CONV_WEIGHT_LOAD） | `34,610` | `0.24%` | `34,610` | `0.24%` |
| 卷积行数据通路（CONV_ROW_DATAPATH） | `9,139,308` | `65.78%` | `9,157,708` | `65.82%` |
| PPU 行消费（PPU_ROW_CONSUME） | `302,234` | `2.17%` | `302,234` | `2.17%` |
| BLOCK5 收尾（PPU_BLOCK5_FINAL） | `2,049,822` | `14.75%` | `2,049,822` | `14.73%` |
| 固定向量操作（VEC_FIXED） | `1,509,681` | `10.86%` | `1,509,681` | `10.85%` |
| 平均池化（AVGPOOL） | `760,920` | `5.47%` | `760,920` | `5.46%` |
| **总计** | **`13,893,492`** | **`100%`** | **`13,911,890`** | **`100%`** |

顶层 stage bin 彼此互斥，其和等于整次 MODE_RUN 周期。卷积内部的 WinGen、SA、
postprocess 三组计数含义不同：它们在同一个 `CONV_ROW_DATAPATH` 窗口内并行采样，
不能相互相加。

`status=0x00000004` 表示计数器采样完成后的 frozen 状态，不是 NPU 错误；两次运行
的硬件 `ERROR` stage 均为 0。`UPSAMPLE_OUT` 和 `FRAME_STORE` 为 0，是因为当前
融合输出路径的周期归入其所属 row-consumer 阶段，并不表示没有执行上采样或输出。

### 2.2 相对 H512W1024 版本的缩放

上一版成功的 H512W1024 binary 基线为 `51,319,833 cycles`。若所有开销严格按照
像素面积缩放，H256W512 应为 `12,829,958 cycles`；实际结果比该值多
`1,063,534 cycles`，即高出 `8.29%`。

| 阶段 | H512W1024 | 理想四分之一 | H256W512 实测 | 相对理想值 |
|---|---:|---:|---:|---:|
| Frame load | `382,743` | `95,686` | `95,708` | `1.000x` |
| Conv row datapath | `32,229,027` | `8,057,257` | `9,139,308` | `1.134x` |
| PPU row consume | `1,202,400` | `300,600` | `302,234` | `1.005x` |
| PPU BLOCK5 final | `8,256,942` | `2,064,236` | `2,049,822` | `0.993x` |
| Fixed vector operations | `6,155,931` | `1,538,983` | `1,509,681` | `0.981x` |
| AvgPool | `3,048,809` | `762,202` | `760,920` | `0.998x` |

除卷积外，主要逐像素阶段几乎都严格按面积缩放。相对理想 4 倍缩放损失的绝大部分
来自卷积。权重装载和控制开销不会按面积缩小 4 倍，但两者合计只比理想值多约
`24.6k cycles`，不是主要矛盾。

H256 版本整网共处理 1,312 个卷积输出行。当前调用图对每个输出行分别启动并排空
一次 WinGen/SA/post DATAFLOW 区域，随后再通过共享 BRAM 行缓冲把结果交给消费者。
与宽度相关的工作量缩小了 4 倍，但逐行固定开销只随高度缩小 2 倍，因此在
H256W512 下占比反而上升。

## 3. 当前 Roofline 工作点

### 3.1 理论峰值与实测计算率

单套 32x32 阵列的逻辑峰值为：

```text
1024 INT8 MAC/cycle x 100 MHz = 102.4 GMAC/s
                                    = 204.8 GOPS
```

| 指标 | binary2 | cityscapes20 |
|---|---:|---:|
| 有效卷积 MAC 数 | `336,435,200` | `345,872,384` |
| 正确的物理槽位填充率 | `70.05%` | `70.78%` |
| K-tile MAC 发射周期 | `468,992` | `477,184` |
| 发射周期 / Conv 窗口 | `5.13%` | `5.21%` |
| Conv 有效吞吐率 | `3.681 GMAC/s` | `3.777 GMAC/s` |
| Conv 峰值利用率 | `3.59%` | `3.69%` |
| 端到端有效吞吐率 | `2.422 GMAC/s` | `2.486 GMAC/s` |
| 端到端峰值利用率 | `2.36%` | `2.43%` |

在真正发射 K tile 的周期内，binary2 与 cityscapes20 分别完成约 `717` 和 `725`
个有效 MAC/cycle，与约 70% 的静态槽位填充率吻合。这说明当前阵列形状并非主要
问题。真正的问题在时间维度：平均约 19 个 Conv stage 周期中，只有 1 个周期在
发射有效 K tile。

当前离线工具报告的填充率为 `47.60%/48.94%`，原因是其分母仍使用输出像素数，
没有使用双像素调度后的 `issue_count_per_row`。Round 0 必须先修正该公式，才能把
它作为后续 gate。正确的物理 MAC 槽位总数分别为 `480,247,808` 和
`488,636,416`。

### 3.2 卷积内部 owner 计数证据

| 并发 owner 状态 | binary2 cycles | 归一化结果 |
|---|---:|---:|
| WinGen active | `6,198,124` | `13.22 cycles / K-tile issue` |
| SA compute or wait-act | `461,660` | 静态 K-tile 发射数的 `0.984x` |
| SA psum emit or wait | `5,879,920` | `32.26 cycles / emitted psum word` |
| Post requant or wait-psum | `820,224` | 恰好 `9 cycles / output issue` |
| Post row-buffer write | `5,340,976` | `33.87 cycles / output pixel` |

这些状态同时包含实际处理、阻塞等待和控制驻留时间，不能直接视为某个运算的纯
latency。但它们给出了相互一致的结论：MAC 发射次数符合静态调度，而阵列在 Conv
窗口的大部分时间用于排出部分和，或等待上下游模块。

根据 artifact 推导的逐行 DATAFLOW 理论下界，binary2 为 `673,344 cycles`，
cityscapes20 为 `683,584 cycles`；实测 Conv 时间分别是下界的 `13.57x` 和
`13.40x`。HLS 同时确认 MAC 的 K-tile 循环已经达到 II=1，因此没有证据表明乘法器
流水线本身是瓶颈。

### 3.3 是否属于 memory-bound

外部内存并不是当前限制：

- Frame load 仅占 MODE_RUN 的 `0.68%`；
- 实测输入搬运带宽约为 `411 MB/s`；
- 卷积权重在 MODE_INIT 中已经加载到片上 PARAM 存储，每层卷积的片上 weight-cache
  装载仅占 `0.24%`；
- 若只计 MODE_RUN 的外部输入和输出，binary2 与 cityscapes20 的运算强度至少为
  `642 MAC/byte` 和 `660 MAC/byte`。

因此，当前工作点应定义为：

> **远未达到 compute-bound，属于片上带宽/传输与同步受限。** 当前上限来自片上
> 特征存储、窗口生成、psum 排出、行缓冲服务速率以及逐行同步，而不是 DDR 带宽，
> 也不是 MAC 数量不足。

若要在严格意义上画出片上 Roofline 横坐标，还需要物理片上字节流量与 stall
计数器。在此之前，使用静态槽位填充率、K-tile 时间发射占比、Conv 峰值利用率和
owner 状态驻留时间作为向岭点收敛的代理指标。

## 4. 根因与优化优先级

1. **逐行填充/排空屏障。** `run_conv_rows_task()` 对每个输出行调用一次 row
   DATAFLOW engine，WinGen、SA 与 postprocess 全网需要重新启动 1,312 次。
2. **SA 与 postprocess 服务速率不匹配。** SA 每个 word 输出 16 个 INT32 psum，
   postprocess 每周期只处理 8 lane；同时必须先形成完整的 256-bit 行缓冲 word，
   才能交给后续消费者。
3. **WinGen 供数代价较高。** WinGen active 平均约为 13 cycles/K-tile word。
   当前实现的关键路径也落在 narrow paired WinGen loader，因此优化必须减少地址与
   条件选择网络，不能继续加宽动态选择逻辑。
4. **串行尾部仍然较长。** BLOCK5 final、fixed vector 与 AvgPool 合计
   `4,320,423 cycles`，占整网 `31.1%`。在 Conv 尚未明显下降前它们是次要瓶颈，
   但最终会把性能限制在约 100 ms 附近。

当前不应优先采用以下方案：

- 扩大或复制 SA，因为静态填充率已有约 70%，时间发射占比却只有约 5%；
- 恢复整网 segment capture/replay，因为此前 P7 实验会重置 WinGen cache、增加
  中间物化，并最终增加总周期；
- 新增第二套 WinGen、PPU、memory owner 或完整特征行复制；
- 恢复 U71 融合或 PARAM v5，因为该路径已经上板功能失败，且与当前瓶颈无关。

## 5. 全轮次约束

- 保持 PARAM v4、75 条 UOP、16 条 EXEC、U71/U72 分离及现有 tensor/量化语义。
- 只允许一套 WinGen、一套 32x32 SA、一套 weight buffer、一套 Conv postprocess、
  一套 PPU 和一个物理 memory owner。源码调用点和 csynth hierarchy 都必须证明
  数据通路为 singleton。
- 不得新增当前卷积的 DDR spill、特征图复制或第二套完整行 cache。
- 调度继续由 PARAM 描述符驱动；HLS 内层不得根据运行时
  `in_c/kernel/dilation` 组合重新理解 ESPNet 网络形状。
- 不得引入 runtime task array、动态 engine mux、递归 DATAFLOW 层级，或针对不同
  mode 复制同类算术 kernel。
- 新增缓冲只能使用 BRAM，不能增加 URAM；URAM 必须保持 `56`。
- binary2 与 cityscapes20 的 INT8 重量化、激活、BLOCK5 拼接、双线性上采样和
  argmax 行为必须保持 bit-exact。
- 维持当前 100 MHz 实现门槛：routing error 为 `0`，`WNS>=0`，`WHS>=0`。
- 实现后资源警戒线：LUT as logic `<=65%`、总 CLB LUT `<=72%`、FF `<=35%`、
  BRAM tile `<=40%`、URAM `=56`、DSP `<=42%`。
- 若单个模块在 scheduling/binding 停留超过 30 分钟，或总 csynth 时间超过当前
  基线 2 倍，必须检查调用图和动态调度结构，不再等待已知的搜索空间爆炸。

## 6. Round 0 - 修正性能分析模型

**涉及文件**

- 修改：`tools/analyze_sa_utilization.py`
- 仅在确有必要时修改：`tools/analyze_conv_cycle_profile.py`
- 输入：当前 PARAM v4 artifact 与上板 stage CSV

**修改步骤**

- [ ] 物理槽位应按 `out_h * issue_count_per_row * oc_tiles * k_tiles * TM *
  effective_k_lanes` 计算；双像素调度不得继续使用 `out_pixels` 作为分母。
- [ ] 同时报告有效 MAC、静态槽位填充率、K-tile 发射周期、时间发射占比、Conv
  峰值利用率和端到端峰值利用率。
- [ ] 增加相对 H512W1024 上板基线的缩放表，并利用现有 prefix PARAM 文件生成
  per-EXEC 表。
- [ ] 本轮不修改 RTL/HLS 状态接口。只有现有计数器无法区分两个候选优化方案时，
  才允许增加新的 datapath instrumentation。

**验收门槛**

- [ ] binary2 静态填充率为 `70.05%`，city20 为 `70.78%`。
- [ ] 有效 MAC 总数分别为 `336,435,200` 和 `345,872,384`。
- [ ] PARAM v4 中每层卷积的静态 stream token 期望值全部一致。

## 7. Round 1 - 将卷积后处理带宽匹配到 16-lane psum 接口

**涉及文件**

- 修改：`ESP_INT8_hls/src/conv_engine.cpp`
- 仅共享类型确需调整时修改：`ESP_INT8_hls/include/npu_types.hpp`
- 验证：Conv/post focused test，以及两个模型的整网测试

**修改步骤**

- [ ] 将当前 8-lane、四组循环改为一套共享 16-lane 重量化数据通路，直接消费
  `psum_half_vec_t`。
- [ ] 对双像素调度，每个 16-lane psum word 直接重量化并打包到对应像素；对非双
  像素的 32 通道调度，将两个 16-lane half 合成为一个 256-bit pixel word。
- [ ] 维持单条 512-bit psum stream 和单个 postprocess 实例，不得新增双 psum
  stream、1024-bit 总线，或分别实现 paired/unpaired 算术引擎。
- [ ] 初始阶段保持 FIFO 深度不变。只有 focused schedule 证明更深 FIFO 能降低
  producer blocking 时，才允许把 `psum_stream` 深度从 8 增至 16；仍须绑定
  LUTRAM，并拒绝任何 BRAM 复制。
- [ ] 删除被替代的 8-lane 打包 helper，并扫描全部 HLS 源码，确认不存在第二个
  postprocess 调用点。

**验收门槛**

- [ ] 两个模型整网 CSim 与 replay 均为 0 mismatch。
- [ ] SA K-tile 循环继续保持 II=1，psum emit 循环不得退化。
- [ ] postprocess 只有一个实例；顶层 DSP 最多增加 32，LUT 增幅不超过 3%。
- [ ] HLS 估计周期保持 `<=7.5 ns`，csynth 时间保持可控。
- [ ] 上板目标：Conv `<=8.3M`、总周期 `<=13.1M`，WinGen active 不得增加。

若后处理加宽只增加资源而未降低上板 Conv 周期，必须在进入 Round 2 前回退。该结果
将证明主导因素是逐行重启，而不是后处理算术宽度。

## 8. Round 2 - 有界多行持续 DATAFLOW

这是本轮最关键的架构升级，与此前失败的 segment overlap 不同：不捕获和 replay
activation，不按 segment 重置 cache，也不尝试并行访问存在冲突的全局特征存储。

**涉及文件**

- 修改：`ESP_INT8_hls/src/conv_engine.cpp`
- 仅在确需提供行循环入口时修改：
  `ESP_INT8_hls/src/win_gen.cpp`、`ESP_INT8_hls/src/sa_core.cpp`
- 扩展：`tools/check_p6_hls_structure.py`

**修改步骤**

- [ ] 使用编译期固定的两行 row group，每两个输出行只启动一次 DATAFLOW 区域，
  替代当前每行启动一次的方式。
- [ ] 组内严格保持三个持续运行的 process：WinGen producer、单套 SA
  consumer/producer、单套 Conv postprocess。三个 process 必须遵守同一个
  `active_rows` token 契约。
- [ ] 使用两个显式 BRAM row slot。postprocess 依次写入 row 0 和 row 1，现有
  consumer 在该有界 group 结束后读取。禁止 dynamic task array、额外输出复制，
  以及通过全局 feature memory 构成 DATAFLOW feedback。
- [ ] `scheduled_window_generator_row()`、`systolic_array_core_row()` 与 postprocess
  在各自 group process 内都只能保留一个调用点。必须检查 csynth hierarchy 的
  clone 数量，而不能依赖 allocation Tcl 强行限制。
- [ ] 最后一组为奇数行时使用 `active_rows=1`，stream token 的生产与消费仍必须
  静态平衡。
- [ ] 只有两行 group 同时通过 csynth 和上板门槛后，才允许评估四行 group。
  四行版本不得增加 engine clone 和 URAM，BRAM 最多增加 16 tile，且不能造成
  综合时间爆炸。

**验收门槛**

- [ ] CSim 无 stream deadlock，两个模型均为 0 mismatch。
- [ ] 各 owner clone count 均保持 1，且无 DATAFLOW feedback warning。
- [ ] 并发 owner 的 idle 占比从约 31%～33% 降至 `<=20%`。
- [ ] 上板目标：Conv `<=7.0M`、总周期 `<=11.8M`、K-tile 时间发射占比
  `>=6.7%`。
- [ ] 100 MHz 实现合法，资源不超过第 5 节警戒线。

若两行 group 无法带来可测量的 idle 降低，应立即停止，不得继续扩大 group，也不得
重新引入 segment 级 ping-pong。

## 9. Round 3 - 由 schedule 指定的 WinGen 垂直复用

只有 Round 2 已经消除主要逐行重启开销，且剩余 Conv owner profile 仍显示 WinGen
是最长 active producer 时，才进入本轮。

**涉及文件**

- 修改：`ESP_INT8_hls/src/win_gen.cpp`
- 若需要保留 schedule bit，修改：
  `ESP_INT8_hls/include/npu_schedule.hpp`、`tools/export_int8_hw_blob.py`
- replay/contracts 只更新描述符合法性检查，不改变数值行为

**修改步骤**

- [ ] 编译器只为能够严格证明相邻输出行可复用的 schedule 设置标志。初始范围为
  3x3、dilation=1：stride=1 可复用两个输入行，stride=2 可复用一个输入行；
  dilation 卷积继续走当前路径。
- [ ] 在有界 row group 内保持滚动 BRAM line window，只替换旧行并加载新增输入行；
  不得创建第二套完整 WinGen cache 或复制 memory reader。
- [ ] 在 row-group 入口根据 `window_schedule_id`/schedule flags 选择复用模式；
  内层循环不得根据运行时 channel/kernel/dilation 组合进行形状判断。
- [ ] 在 packed read 循环前寄存 row base、边界和 cache-slot 选择。当前 narrow paired
  地址条件网络已经位于物理关键路径，不得再叠加一层动态 mux。
- [ ] 除非 per-EXEC 上板结果证明 wide/unpaired 路径占主导，否则保持现有水平
  packed-word 复用和 wide/unpaired 路径不变。

**验收门槛**

- [ ] exporter 对每个描述符报告精确减少的物理行读取次数，并拒绝不支持的复用
  schedule。
- [ ] WinGen 仍只有一个实例；URAM 不变，新增 BRAM 不超过实现资源警戒线。
- [ ] HLS 估计周期保持 `<=7.5 ns`，narrow paired 循环维持当前 II。
- [ ] 上板目标：WinGen active `<=4.5M`、Conv `<=5.8M`、总周期 `<=10.6M`，
  Conv 峰值利用率 `>=5.5%`。

## 10. Round 4 - 压缩串行尾部

只有 Conv 已降至 `<=5.8M cycles` 才进入本轮。在此之前优化尾部不足以使 NPU
靠近卷积计算岭点。

**涉及文件**

- 修改：`ESP_INT8_hls/src/ppu.cpp`
- 仅在上板数据证明有必要时修改：`ESP_INT8_hls/src/vec_alu_engine.cpp`
- 初始保持：`ESP_INT8_hls/src/avgpool_unit.cpp` 不变

**修改步骤**

- [ ] 将单套共享 BLOCK5 affine/add-affine lane factor 从 4 提升到 8，同时保留
  单一 finalizer 和单一 store tail。
- [ ] 维持现有 16-lane fixed-vector half datapath，除非计数器已经分离其 memory
  wait 与 arithmetic；不得在没有实测依据时新增 32-lane 复制通路。
- [ ] 冻结 C3 group-32 AvgPool 路径；其 `760,920 cycles` 已正确按面积缩放，
  不是当前性能退化来源。
- [ ] 保证 BLOCK5 源数据顺序、残差量化和 compact store layout bit-exact。

**验收门槛**

- [ ] PPU BLOCK5 final `<=1.5M`，同时 Vec 和 Conv 增幅均不超过 1%。
- [ ] 两个模型总周期均 `<=10.0M cycles`，即 100 MHz 下 `<=100 ms`。
- [ ] 通过第 5 节全部实现、资源和时序门槛。

## 11. 条件性岭点架构升级

若 Round 4 已接近 100 ms，但 Conv 峰值利用率仍低于 10%，下一步不应继续做局部
unroll，而应采用编译器指导的片上物理 bank 分离，支持真正持续的生产者/消费者
并行：

1. 根据 tensor 生命周期，把卷积源和目的 tensor 分配到独立片上物理 bank，同时
   不增加 URAM 总量。
2. bank 选择由编译器写入 PARAM；HLS 数据通路只执行描述符，不在运行时推断
   tensor placement。
3. 当 bank 独立性已经证明时，postprocess 输出直接流向 row consumer，同时
   WinGen 从另一 bank 读取；同 bank 或存在 alias 的情况保留有界 row-buffer
   fallback。
4. 对通过 bank 独立性证明的描述符彻底移除 fallback，不得插入 activation
   capture/replay 或外部内存往返。

该升级必须经过单独设计审查后才能实施。其临时岭点判据为 Conv `<=3.3M cycles`、
K-tile 时间发射占比 `>=14%`、Conv 峰值利用率 `>=10%`，且 100 MHz 实现合法。
达到该状态后，片上存储服务能力和算术发射能力才足够接近，可以开展有意义的实测
Roofline 分析；此前不能宣称已达到岭点。

## 12. 验证顺序与停止规则

每一轮 HLS 性能改进都必须按以下顺序执行：

```text
源码与 legacy/deadlogic 扫描
-> 架构和 PARAM contract 检查
-> focused module CSim
-> 发布节点执行 binary2 + city20 整网 CSim
-> csynth hierarchy / II / resource / timing audit
-> Vivado placement / route / timing
-> 一次双模型上板 profiling
-> 更新实测性能模型后再进入下一轮
```

出现以下任一情况时，停止并回退当前轮次：

- logits 或 mask mismatch、非法类别 ID、非零 ERROR stage；
- 出现第二套 WinGen、SA、postprocess、PPU 或 memory owner；
- DATAFLOW token 数量不匹配、feedback warning 或 CSim deadlock；
- scheduling/binding 反复停滞，或 csynth 总时间超过基线 2 倍；
- URAM 增加、资源超过警戒线、发生 routing congestion，或布线后 setup/hold
  slack 为负；
- 总周期或 `Conv + 可见 PPU` 周期退化超过 1%；
- 只是改变 stage counter 归属，而多个相关 stage 的合计周期没有下降。

## 13. 最终验收标准

近期 H256W512 发布目标：

```text
binary2 与 city20 总周期  <= 10.0M cycles @ 100 MHz
Conv row datapath          <= 5.8M cycles
ERROR stage               = 0
整网 integer replay       = 0 mismatch
32x32 数据通路实例数      = csynth hierarchy 确认只有一套
布局布线与时序            = 合法，WNS >= 0，WHS >= 0
```

当前 `13.9M-cycle` 版本是一版功能正确、能够合法布局布线的基线，也证明了输入尺寸和
片上存储规模迁移已经成功，但它还不是岭点设计。后续改进必须提高 SA 在时间维度上的
有效发射占比；只提高理论 MAC 峰值，或仅把周期从一个 stage bin 转移到另一个 bin，
都不能视为真实性能进展。
