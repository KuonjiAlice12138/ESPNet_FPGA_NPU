# ESPNet_Encoder INT8 NPU 项目工作进展与计划文档 (合并版，2026-05-06/0507)

## 1. 已完成进展

### 1.1 完整 C/RTL cosim 结论收口

已尝试完整单图 `top_golden_sample_tb` C/RTL cosim：仿真完成 `MODE_INIT`，在完整 `MODE_RUN` 运行较长时间后由 XSIM/AXI VIP 层报错退出。失败点位于 `hls/sim/verilog/axivip/axi_slave_seq_lib.sv`，不是 testbench 数值 mismatch，也没有证据表明 NPU 逻辑在某条 UOP 上功能失败。

因此当前结论保持不变：完整全分辨率 C/RTL cosim 不再作为继续推进的硬性门槛；C/RTL 用于短路径和小规模顶层用例，整网正确性主要依赖 HLS C model、短路径 RTL、以及后续板端闭环验证。

### 1.2 短路径 top 级 C/RTL 已补强

本轮已用小规模 tensor 验证主要顶层 RTL 数据路径：

| 验证项 | 覆盖重点 | 结果 |
|---|---|---|
| `u00_input` debug | `MODE_INIT/RUN`、frame load、input dump | C/RTL `PASS` |
| `top_conv_dispatch_tb` | conv datapath、win_gen、SA、PPU、store | C/RTL `PASS` |
| `top_dispatch_tb` | affine/add 顶层调度路径 | C/RTL `PASS` |
| `top_pool_store_dispatch_tb` | pool、scratch、store/concat、frame store | C/RTL `PASS` |

其中 `top_pool_store_dispatch_tb` 为新增 TB，不改变硬件 `.cpp/.hpp` 边界；其 C/RTL report 为 Verilog `Pass`，总执行时间 `107523` cycles。

### 1.3 整网 C model 与 artifact 已复核

复跑真实单张图片 `top_golden_sample_tb` C 仿真，当前 HLS C model 输出满足既定 mask 验收口径：

| 项目 | 结果 |
|---|---:|
| 输出字节数 | `16384` |
| score byte mismatch | `15590 / 16384` |
| mean abs diff | `6.564758` |
| max abs diff | `32` |
| argmax mask mismatch | `72 / 8192` |
| TB 阈值 | `128` |

复跑 `blob_file_tb` 后确认 `param_blob.bin` 与当前硬件解析逻辑一致：`129856` bytes，`75` 条 UOP，`55` 个 scale，`26` 个 conv desc，`7` 个 affine desc，`14` 个 add desc，`3` 个 pool desc。

### 1.4 子域 C 回归通过

以下 C 级回归均为 `CSim done with 0 errors`：`control_dma_tb`、`memory_tile_tb`、`pool_concat_domain_tb`、`conv_domain_tb`、`win_gen_tb`、`sa_core_tb`、`ppu_tb`、`frame_dma_tb`。

### 1.5 Vivado 硬件平台已完成

已在 Vivado block design 中完成 INT8 NPU IP 与 Zynq PS 的系统集成，自动连线后修正地址分配，完整硬件平台综合、实现和 bitstream 生成均已成功。最终资源仍处于高占用状态，但实现阶段已通过 timing 和 routing，可作为第一版 baseline 上板平台。

Vitis 中已基于导出的硬件平台创建 `ESP_INT8_platform`，并确认 app 侧使用的 NPU AXI-Lite 控制基地址为 `0xA0000000`。当前第一版上板验证继续采用轮询 `ap_done`，暂不依赖 interrupt。

### 1.6 PS app 已完成第一版 build

已创建 `ESP_INT8_app`，并参考 FP32 app 的目录和依赖管理方式完成第一版 PS 侧程序：

| 模块 | 当前功能 |
|---|---|
| `drivers/sd_card.c/.h` | SD/FatFs 挂载、bin 文件读取、`OUTQ.BIN` 写回 |
| `hal/int8_npu.c/.h` | HLS driver 封装、`MODE_INIT/MODE_RUN` 启动、超时轮询 |
| `main.c` | 读取 `PARAM.BIN/INPUTQ.BIN`，flush/invalidate cache，启动 NPU，保存并可选比对输出 |

build 过程中已修正 `xilffs` 依赖问题：INT8 app 回到与 FP32 app 一致的方式，在 `CMakeLists.txt` 中声明 `xilstandalone;xiltimer;xilffs`，并通过 platform/domain BSP 提供 `ff.h` 和 `libxilffs.a`。当前 app 已 build 成功，晚些可进入 JTAG/串口上板运行。

## 2. 当前阶段判断

当前项目已从 HLS 验证和平台搭建阶段推进到板端 bring-up 前夜。HLS 设计、Vivado baseline 硬件平台、Vitis platform 和第一版 PS app 均已闭合，下一步不再优先追加 C/RTL cosim，而是用真实板端 `SD -> DDR -> PL NPU -> DDR -> SD` 数据流验证。

PS-PL 交互边界已经明确：INT8 NPU 使用三路 `m_axi` 主动访问 DDR，并通过 AXI-Lite 控制；PS 侧负责从 SD 卡读取 `PARAM.BIN/INPUTQ.BIN`，准备 DDR buffer，写控制寄存器，启动 `MODE_INIT -> MODE_RUN`，回收并写出 `OUTQ.BIN`。

| 接口/寄存器 | 作用 |
|---|---|
| `gmem_frame_in` / `0x10` | 输入图像 DDR 地址 |
| `gmem_frame_out` / `0x1c` | 输出 logits DDR 地址 |
| `gmem_param` / `0x28` | `param_blob.bin` DDR 地址 |
| `mode` / `0x34` | `MODE_INIT=1`，`MODE_RUN=2` |
| `uop_count` / `0x3c` | 当前整网为 `75` |
| `ap_ctrl` / `0x00` | `ap_start/ap_done/ap_idle/ap_ready` |

上板前需要保证 SD 卡根目录文件与当前 artifact 对齐。考虑到 `xilffs` 可能未启用 LFN，板端实际文件名统一使用 8.3 短文件名：

| SD 文件名 | 源 artifact | 大小 |
|---|---|---:|
| `PARAM.BIN` | `param_blob.bin` | `129856` |
| `INPUTQ.BIN` | `input_q.bin` | `1572864` |
| `HLSREF.BIN` | `hls_output_q.bin`，推荐参考 | `16384` |
| `GOLDREF.BIN` | `golden_output_q.bin`，可选 | `16384` |

其中 `HLSREF.BIN` 更适合作为第一轮板端 byte/mask 对齐参考；`GOLDREF.BIN` 用于观察与 QAT/PyTorch baseline 的差异，不要求 bit-exact。

## 3. 下一步计划

1. 准备 FAT32 SD 卡，根目录放置 `PARAM.BIN`、`INPUTQ.BIN`，建议同时放 `HLSREF.BIN`。
2. 用 Vitis/JTAG 下载 bitstream 和运行 `ESP_INT8_app.elf`，串口观察 SD 挂载、参数解析、`MODE_INIT`、`MODE_RUN` 和输出保存日志。
3. 首轮只要求单张图片闭环跑通：生成 `OUTQ.BIN`，app 内部优先与 `HLSREF.BIN` 做 byte/mask 参考比对。
4. 若 `MODE_INIT` 或 `MODE_RUN` 不 done，优先排查 AXI-Lite 控制、DDR 地址、cache flush/invalidate 和 SD 文件版本。
5. 若能 done 但输出全零或误差明显超过 HLS C model 参考，再制作 debug 版硬件，使用 `stop_after_uop/dump_tensor` 做分段定位。
