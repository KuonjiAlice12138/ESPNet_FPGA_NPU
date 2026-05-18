# ESPNet_Encoder INT8 NPU 项目工作进展与计划文档 (更新日期: 2026-05-06)

## 1. 与上一版相比已完成的进度

上一版 `workplan_0430` 的重点是：完整 top C model、综合实现闭环、bit-exact 验收口径收口，以及进入 Vivado/PS 侧上板准备。本轮工作主要围绕完整 C/RTL 仿真取舍、上板验证策略和 PS-PL 交互边界进一步明确。

### 1.1 完整 top C/RTL cosim 已尝试，结论是不再作为主验收门槛

本轮使用已有 HLS 综合结果复用 `ESP_INT8_hls` work directory 启动完整单图 `top_golden_sample_tb` C/RTL cosim，没有重新跑 implementation，也没有重新跑 C synthesis。

仿真实际进入 XSIM，并完成 `MODE_INIT` transaction；完整 `MODE_RUN` transaction 运行较长时间后退出失败。关键日志如下：

| 项目 | 结果 |
|---|---:|
| C 侧 golden 检查 | `passed` |
| C 侧 mask mismatch | `72 / 8192`，阈值 `128` |
| XSIM 状态 | `FATAL_ERROR` |
| cosim report | `Verilog Fail` |
| RTL 输出文件 | `16400` bytes，未完整写完 |
| C reference 输出文件 | `32792` bytes |
| XSIM 峰值内存 | 约 `10.7GB` |
| 崩溃位置 | `hls/sim/verilog/axivip/axi_slave_seq_lib.sv` |

当前失败点来自 XSIM/AXI VIP 内核异常，不是 testbench 主动报数值 mismatch，也没有证据表明 NPU 逻辑已经在某个 UOP 上功能失败。因此完整单图 top C/RTL cosim 的时间和工具成本已经超过其验证收益，不再建议作为继续推进前的硬性条件。

### 1.2 验证策略转向板端闭环

当前已有基础包括：

* Top 级 HLS C model 可跑完整 `MODE_INIT + MODE_RUN + 75` 条 UOP。
* 单图 HLS 输出与 golden 的 mask 差异在既定阈值内。
* 顶层综合和实现已经通过，`100MHz` 时序满足。
* 完整 top C/RTL cosim 的失败更像工具/AXI VIP 压力问题，而不是明确设计错误。

因此后续验证重心应从完整 RTL cosim 转到上板验证。C/RTL cosim 仍可保留，但只用于小规模子模块、短 UOP 序列或极小 tensor 的局部定位，不再要求完整单图通过。

### 1.3 PS-PL 交互边界已明确

当前 INT8 NPU top 是三路 `m_axi` 加 AXI-Lite 控制口，不是 FP32 版本外置 AXI DMA 搬运模式。PL 侧通过 `m_axi` 主动访问 DDR，PS 侧负责准备数据、写控制寄存器、启动 IP 和回收结果。

顶层接口为：

| 接口 | 方向 | 作用 |
|---|---|---|
| `gmem_frame_in` | PL `m_axi` read | 输入图像 DDR 地址 |
| `gmem_frame_out` | PL `m_axi` write | 输出 logits/feature DDR 地址 |
| `gmem_param` | PL `m_axi` read | `param_blob.bin` DDR 地址 |
| `mode` | AXI-Lite | `MODE_INIT=1` 或 `MODE_RUN=2` |
| `uop_count` | AXI-Lite | 当前整网为 `75` |
| `ap_ctrl` | AXI-Lite | `ap_start/ap_done/ap_idle/ap_ready` |

HLS 已生成基础 driver，关键寄存器偏移为：

| 偏移 | 寄存器 |
|---:|---|
| `0x00` | `ap_ctrl` |
| `0x10` | `gmem_frame_in[63:0]` |
| `0x1c` | `gmem_frame_out[63:0]` |
| `0x28` | `gmem_param[63:0]` |
| `0x34` | `mode` |
| `0x3c` | `uop_count` |

## 2. 当前阶段判断

当前项目已经从 HLS 代码编写/子系统验证阶段，进入上板系统集成阶段。继续投入完整 top C/RTL cosim 不划算；更有效的风险收敛方式是尽快建立真实 PS-PL-DDR-SD 闭环。

第一版板端验证不应追求功能一次性跑完整验证集，而应按如下顺序 bring-up：

1. AXI-Lite 寄存器读写、`ap_start/ap_done/ap_idle` 自检。
2. `MODE_INIT` 单独运行，确认 `param_blob.bin` 能被 PL 正确读取且 IP 正常 done。
3. 小规模输入或短 UOP 子图验证。
4. 单张完整图片运行，输出 `output_q.bin`。
5. 与 HLS C model 的 `hls_output_q.bin` 做 byte/mask 对比。
6. 稳定后扩展到多图和 PA/mIoU 统计。

需要注意：当前 top 没有把内部 `error_code/current_uop/perf_counter` 暴露成 AXI-Lite 可读寄存器。第一版 app 先依赖 `ap_done`、超时保护和输出比对，不在上板前提前改 debug 硬件边界；若整网不 done、输出全零或误差明显超出 HLS C model 参考，再切换到 debug build。

## 3. 下一步工作计划

### 3.1 Vivado 硬件平台

1. 将当前 `espnet_encoder_int8_core` HLS IP 导入 Vivado block design。
2. 连接 Zynq PS、DDR、AXI-Lite control、三路 `m_axi` 到 PS 高性能 DDR 访问通路。
3. 配置 clock/reset，目标仍按当前 HLS 实现的 `100MHz` 口径推进。
4. 可先不接 interrupt，第一版 app 使用轮询 `ap_done`；若后续需要降低 CPU 等待开销再接 IRQ。
5. 生成 bitstream 并导出 `.xsa`，作为 Vitis app platform 输入。

### 3.2 PS 侧 app

1. 复用 FP32 版本 SD/FatFs 代码，读取 `param_blob.bin` 和 `input_q.bin`。
2. 在 DDR 中准备三块 buffer：`param`、`input`、`output`，保证地址和长度满足 `m_axi` 访问需求。
3. 启动前对 `param/input/output` 做 cache flush，运行结束后对 `output` 做 invalidate。
4. 按 `MODE_INIT -> MODE_RUN` 两阶段控制 IP，`MODE_RUN` 设置 `uop_count=75`。
5. 将输出写回 SD 卡为 `output_q.bin`，并记录运行日志、寄存器状态和耗时。
6. 用 HLS C model 输出作为第一参考，先验 byte/mask，再验 PA/mIoU。

### 3.3 风险与备用方案

如果上板后出现不 done、DDR 访问异常或输出全零等问题，优先排查：

1. 三个 64-bit DDR 基地址是否写入正确。
2. AXI interconnect/HP/HPC 端口地址映射是否覆盖 DDR。
3. cache flush/invalidate 是否遗漏。
4. `param_blob.bin`、`input_q.bin` 文件大小和内容是否与当前 HLS/export 版本一致。
5. `MODE_INIT` 是否在 `MODE_RUN` 前成功执行，`uop_count` 是否为 `75`。

当前计划先尝试整网上板，不立即增加硬件 debug 口。若整网 baseline bitstream 无法跑通或结果误差过大，再制作 debug 版硬件，优先补以下最小能力：

1. `stop_after_uop`：允许执行到指定 UOP 后提前结束，便于二分定位。
2. `gmem_debug` 或状态寄存器：记录最后完成的 UOP、opcode 和错误码。
3. 可选 `dump_tensor_id/dump_words`：把指定中间 tensor 或局部切片导出到 DDR buffer，用于逐层比对。

## 4. 小结

今天的主要结论是：完整 top C/RTL cosim 已经尝试，但被 XSIM/AXI VIP 层面的高成本问题卡住，不适合作为继续推进的硬性门槛。当前更合理的路线是冻结 HLS 设计，优先尝试整网 baseline 上板，进入 Vivado 平台搭建和 PS 侧 app 开发，用真实板端数据流完成 `SD -> DDR -> PL NPU -> DDR -> SD` 的单图闭环；如果整网跑不通或误差过大，再制作 debug 版硬件做分段定位。
