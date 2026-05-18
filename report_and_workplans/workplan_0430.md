# ESPNet_Encoder INT8 NPU 项目工作进展与计划文档 (更新日期: 2026-04-30，2026-05-01补充)

## 1. 与上一版相比已完成的进度

上一版 `workplan_0427` 的重点是：控制域、frame DMA、memory 基础闭环、带硬件约束 QAT/export 以及 INT8 baseline。本轮工作继续向 HLS 整网验证、综合实现和验收口径收口，主要完成四件事。

### 1.1 Top 级 C model 验证推进到完整 75 条 UOP

在现有硬件 `.cpp/.hpp` 框架不新增设计文件的前提下，继续验证完整 `espnet_encoder_int8_core`：

* 用户手动运行的 Vitis C 仿真返回 `0 errors`，对应入口为 `top_conv_dispatch_tb`，可证明卷积调度小闭环通过。
* 额外使用完整 `top_golden_sample_tb` 跑通 `MODE_INIT + MODE_RUN + 75` 条 UOP 的 top 级 C model。
* Top TB 已能写出 HLS 整数语义输出：`D:\ESP_INT8\ESP_INT8_hls\hls_output_q.bin`。

完整 top 输出与 PC fake-quant golden 不再强制要求逐 byte bit-exact。当前单图统计结果为：

| 项目 | 结果 |
|---|---:|
| logit byte mismatch | `15590 / 16384` |
| max abs diff | `32` |
| mean abs diff | `6.564758` |
| argmax mask mismatch | `72 / 8192` |
| valid-pixel mask mismatch | `65 / 3842` |

这说明 logits 层面存在较多整数差异，但最终 mask 翻转比例较低。

### 1.2 HLS 输出对 PA/mIoU 的影响已量化

今天新增了对比脚本：

* [compare_hls_top_output.py](/D:/ESP_INT8/tools/compare_hls_top_output.py)

该脚本读取 `golden_output_q.bin / target.npy / hls_output_q.bin`，统一统计 logit 差异、mask 差异和单图 PA/mIoU。

当前单图结果：

| 输出来源 | PA | mIoU |
|---|---:|---:|
| PC fake-quant golden | `0.97188964` | `0.92504997` |
| HLS C model | `0.96954711` | `0.91880367` |
| HLS - golden | `-0.00234253` | `-0.00624630` |

因此当前 HLS 整数实现虽然不是 bit-exact，但在该单图上的指标下降较小。后续板端 bit-level 对齐应优先以 HLS C model 的 `hls_output_q.bin` 为参考，而不是直接要求与 PyTorch fake-quant golden 逐 byte 一致。

### 1.3 Bit-exact 不一致原因已初步定位

今天对 PyTorch QAT/export 与 HLS 计算路径做了对照，当前不 bit-exact 的主要来源已经明确：

* PyTorch golden 是 fake-quant 模型的 FP32 前向结果再 `torch.round(tensor / scale)` 得到；HLS 是全整数 `int8 * int8 -> int32 -> mult/shift requant`。
* HLS `round_shift()` 对负数使用 `(x - bias) >> shift`，在二补码算术右移下会产生负向偏置，这与 PyTorch `torch.round` 的 ties-to-even 语义不一致。
* AvgPool 分支在 HLS 中会先对 INT8 求和并 `round_div9()`，再 requant；PyTorch fake-quant 路径中 AvgPool 本身仍是 FP32 平均，量化点不同。
* ADD 当前是 shared-scale bypass，concat/store/frame_dma/memory 主要是搬运路径，不是当前误差主因。

这意味着当前误差主要来自量化舍入语义和 pool 计算路径，而不是 uop 调度或片上 memory layout 的明显错误。

### 1.4 顶层综合与实现完成首次闭环

围绕 `espnet_encoder_int8_core` 顶层，修正了若干 Vitis HLS 综合问题，包括 `ap_uint` 与枚举/窄位宽类型的 ambiguous conversion、顶层 `m_axi` depth 与 burst/outstanding 配置、dataflow 区域组织，以及非推理关键路径的过度流水。

虽然 HLS C synthesis 曾给出约 `907k LUT / 265%` 的保守估计，但 Vivado RTL synthesis 与 place & route 后资源显著下降，最终实现通过：

| 阶段 | LUT | FF | DSP | BRAM | URAM | Timing |
|---|---:|---:|---:|---:|---:|---|
| RTL synthesis | `261256` | `111935` | `1196` | `1354` | `112` | met, `6.257ns` |
| Place & Route | `254028` | `111894` | `1196` | `1354` | `112` | met, `9.902ns` |

Route 后 `WNS=0.098ns`、`TNS=0`，setup/hold failing endpoints 均为 `0`；`440474` 条 routable nets 全部 routed，routing errors 为 `0`。这说明当前实现不是空壳，而是完整设计已在 `xczu15eg-ffvb1156-2-i` 上完成布线并满足 `100MHz` 约束。

## 2. 当前阶段判断

截至目前，HLS 验证已经从“子系统 C 仿真通过”推进到“完整 top C model 可运行、可导出结果、可计算 mask/指标差异，并完成顶层综合/实现闭环”的阶段。

当前可以认为：

* 控制域、memory contract、uop schedule 和主要 datapath 已经具备继续冻结前验证的基础。
* PC fake-quant golden 与 HLS 整数实现不 bit-exact 是可解释的，不应再作为唯一失败标准。
* 当前合理验收口径应分两层：板端输出优先 bit-level 对齐 HLS C model；算法效果优先看完整验证集 PA/mIoU 是否接近 INT8 software baseline。
* 顶层实现已经满足 `100MHz` 时序，后续可以进入 IP 导出、Vivado 集成和 PS 侧单图闭环。

仍需注意：目前 mask/PA/mIoU 统计只基于单张 golden sample，完整 val set 的硬件侧批量数据还没有导出和验证；同时资源余量并不宽松，route 后 `CLB LUT=74.43%`、`CLB=95.86%`、`BRAM Tile=90.99%`、`URAM=100%`，后续不宜再增加大 buffer 或大规模并行逻辑。

## 3. 下一步工作计划

下一步建议按以下顺序推进：

1. 冻结当前 HLS 设计边界；除非发现功能错误，不再新增 `.cpp/.hpp` 硬件模块。
2. 视时间成本决定是否跑 C/RTL cosim；若完整 top 过慢，可先做关键子系统或单图短路径验证。
3. 导出 HLS IP，进入 Vivado block design 集成，连接 AXI-Lite 控制口和三个 `m_axi` 数据口。
4. 开始 INT8 PS app：复用 FP32 SD 卡读写路线，补齐 `param_blob.bin / input_q.bin / output_q.bin` 的 DDR 搬运、cache 维护和寄存器控制。
5. 板端单图跑通后，用 HLS C model 输出作为 bit-level 参考，再用 mask/PA/mIoU 对比 INT8 software baseline。
6. 若后续资源或时序再次紧张，优先优化 memory/write tile 复用、BRAM/URAM 分配和非卷积算子资源共享。

## 4. 小结

本轮核心进展不是继续堆新模块，而是把整网验证标准和实现可行性同时收口：当前 HLS top 已能跑完整单图推理并输出可评估结果，bit-exact 不一致主要来自 PyTorch fake-quant 与 HLS 整数实现的舍入/AvgPool 语义差异；顶层实现也已在目标器件上通过 place & route 并满足 `100MHz`。后续重点应转向设计冻结、IP 导出、Vivado 系统集成和 PS 侧单图板端闭环。
