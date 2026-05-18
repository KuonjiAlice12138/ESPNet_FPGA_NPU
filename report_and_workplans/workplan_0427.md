# ESPNet_Encoder INT8 NPU 项目工作进展与计划文档 (更新日期: 2026-04-27)

## 1. 与上一版相比已完成的进度

上一版 `workplan_0427` 的重点是：HLS component 与代码骨架落地、补入 INT8 PS app 工作、明确 INT8 不再照搬 FP32 多 AXI DMA 逐层调度。今天后续工作已经继续向前推进，主要完成了四条线。

### 1.1 HLS 控制域实现与验证

在不新增硬件设计 `.cpp/.hpp` 文件的前提下，已经在现有框架内补齐并验证了首批控制域模块：

* `param_dma.cpp`：完成 `param_blob` header、tensor desc、scale desc、uop 和各类 param desc 的基础解析与范围检查。
* `if_dec.cpp`：完成 uop stream 的 opcode、tensor id、param id、`END` 检查和错误码上报。
* `int8_core.cpp`：完成 `MODE_INIT / MODE_RUN` 控制入口与错误状态传播。
* 新增 testbench `control_dma_tb.cpp / blob_file_tb.cpp`，只作为验证文件，不改变硬件模块划分。

当前验证结果：

* Vitis HLS `C-sim` 已通过，`CSim done with 0 errors`
* 真实导出的 `param_blob.bin` 可被控制域 testbench 解析
* `blob_file_tb` 通过：`bytes=129856, uops=75, scales=55`

这说明 `param_blob -> param_dma -> if_dec` 的控制链路已经具备继续向 datapath 联调的基础。

### 1.2 frame_dma 与 on_chip_memory 最小闭环

按照“先小闭环、再整网”的实现计划，今天继续补齐了首个 datapath 子系统：

* `frame_dma.cpp`：实现 `gmem_frame_in -> T_INPUT` 和 `T_OUT -> gmem_frame_out` 的 `256-bit AXI` word 级搬运。
* `memory.cpp`：补入 `FMEM0/1/2` 的片上存储模型，采用 `256-bit word array + byte lane` 访问方式。
* 保留后续模块需要的 byte 语义访问接口，便于 `win_gen / ppu / pool / concat` 按 `NHWC INT8` tensor layout 读写。
* 新增 `tb/frame_dma_tb.cpp`，只作为 testbench，不改变硬件模块划分。

当前验证结果：

* `frame_dma_tb` 优先读取真实导出的 `input_q.bin`
* 验证 `input_q.bin -> T_INPUT` 的 pack/unpack 与片上 byte layout
* 通过临时 `T_INPUT` 前缀拷贝到 `T_OUT`，验证 `T_OUT -> gmem_frame_out`
* Vitis HLS `C-sim` 已通过：`frame_dma_tb passed: input_bytes=1572864 output_bytes=16384 source=input_q.bin`

这一步验证的是 DDR 与片上存储之间的数据搬运路径，不代表整网推理已经完成。

### 1.3 带硬件约束 QAT 与硬件 artifact 导出闭环

今天重新修复并跑通了 QAT/export 侧，使模型侧量化行为与当前硬件约束对齐：

* QAT 默认启用硬件约束：activation / weight zero-point 固定为 `0`
* ADD bypass、concat/store-copy 等域内共享 activation scale
* QAT 训练过程中持续冻结并重新施加这些约束
* 重新跑了 `3 epoch` 带约束 QAT
* 清理并覆盖了之前不满足硬件约束的错误输出目录
* 使用 `--strict-zp` 导出了硬件可用 `hw_artifacts`

当前有效目录：

* 模型侧 artifact：`D:\ESP_INT8\quantized_artifacts_hw_constrained_qat_3ep`
* 硬件侧 artifact：`D:\ESP_INT8\hw_artifacts\hw_constrained_qat_3ep_single`

验收状态：

* `manifest.json` 中 `hardware_constraint_report.passed = true`
* `nonzero_zero_points = []`
* `scale_mismatches = []`
* `export_manifest.json` 中 `warnings = []`
* `input_q.bin` 为单帧 NHWC INT8 输入，大小 `1572864` bytes
* `golden_output_q.bin` 为单帧 NHWC INT8 golden 输出，大小 `16384` bytes

### 1.4 INT8 baseline 精度已重新确认

带硬件约束 QAT 后，在完整 val set `500` 张图上重新评估得到当前 INT8 software baseline：

| 指标 | 当前结果 |
|---|---:|
| PA / pixel accuracy | `0.9818317506` |
| mIoU | `0.8882841695` |

该结果应作为后续 FPGA/NPU 硬件推理结果的主对照 baseline。硬件侧首先需要做到与单帧 `golden_output_q.bin` bit-level 对齐；完整验证集 `val_data` 批量导出可以等单帧闭环稳定后再补。

### 1.5 数据处理流程文档与一键脚本

已经在 `tools/` 下补充数据处理说明和自动化入口：

* `tools/int8_data_export_workflow.md`
* `tools/run_hw_constrained_qat_export.bat`

`.bat` 用于后续复现“带约束 QAT -> 硬件 artifact 导出 -> baseline 评估 -> blob 解析验证”的完整流程。按要求，创建后没有再次运行。

## 2. 当前阶段判断

截至今天，项目状态可以概括为：

* HLS 代码已经从空骨架推进到“顶层 ABI + 控制域解析 + 帧搬运最小闭环”可验证状态。
* QAT/export 链路已经从“后处理式对齐”修正为“训练时带硬件约束”的版本。
* 当前 INT8 baseline 精度和硬件输入/输出 golden 数据已经可作为后续 bit-level 验证基准。
* PS app 工作已经正式纳入路线：仍需 SD 卡读写、DDR buffer、cache flush/invalidate、AXI-Lite 控制寄存器配置和输出写回。

需要明确的是：当前还没有完成整网硬件推理。`frame_dma / memory` 只完成了最小帧搬运与片上 byte/word 访问基础，`win_gen / sa_core / ppu / avgpool / concat` 仍处于待实现或待完善状态，因此目前通过的是控制域、数据导出和帧搬运子系统，不是完整 encoder 的端到端 bit-exact。

## 3. 下一步工作计划

下一步建议按“先小闭环、再整网”的顺序推进：

1. 继续完善 `memory` 的 tile 级读写接口，使后续 `win_gen / ppu / pool / concat` 不各自定义 bank 地址规则。
2. 优先实现 `ppu` 中的 bit-exact `CONV_POST / ADD_POST / AFFINE_POST`，用导出的 qparam/scale 数据做模块级 testbench。
3. 再推进 `win_gen + sa_core` 的卷积主路径，实现 1x1、3x3、dilated conv 的首个可比较 tile。
4. 完成 `avgpool_unit / concat_unit` 后，串起局部 block，再逐步扩展到完整 `75` 条 uop。
5. HLS 单帧 bit-level 稳定后，再启动 INT8 PS app：复用 SD 卡读写习惯，新写 `int8_npu_ctrl`，完成板级 `INIT/RUN/out_q.bin` 闭环。
6. 最后补导完整 `val_data` 批量验证集，用于板上 PA/mIoU 与 software baseline 对比。

## 4. 小结

今天的实质进展是把 INT8 路线从“设计可写”推进到“控制域可验、量化数据可用、baseline 可引用、帧搬运路径可跑”。下一阶段的主风险已经不在 QAT/export，而在 HLS datapath 是否能严格复现当前 artifact 的 layout、scale、rounding 和 uop 行为。
