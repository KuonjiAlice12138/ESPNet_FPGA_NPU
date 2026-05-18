# ESPNet_Encoder INT8 NPU 项目工作进展与计划文档 (更新日期: 2026-04-23)

## 1. 今日工作进展 (Progress Overview)

截至 2026 年 4 月 23 日，INT8 路线已经从“文档级架构收敛”推进到“量化训练闭环 + 硬件对齐数据导出”阶段。今天的核心成果有两类：一类是 **INT8 NPU 规格文档落地**，另一类是 **对称 INT8 量化精度恢复与硬件 golden 数据导出**。

### 1.1 INT8 NPU 规格文档收敛完成
今天已经系统阅读并对照了以下内容：
* `D:\ESP_FP32\encoder_fp32\src\model\espnet_encoder.c`
* `D:\ESP_FP32\encoder_fp32\src\hal\npu_dispatch.c`
* `D:\ESP_FP32\encoder_fp32\src\drivers\dma_ctrl.c`
* `D:\ESP_FP32\report_and_workplans\*.md`
* `D:\ESPNet\Model_flattened_quantized.py`
* `D:\ESPNet\test_single_image_for_all.py`
* `D:\ESP_INT8\int8_npu_design\*.md`

并已将 `D:\ESP_INT8\int8_npu_design` 下的 INT8 设计文档细化为可直接指导 HLS 编码的规格版本，主要包括：
* `01_System_Architecture.md`
* `02_Systolic_Array_Engine.md`
* `03_Data_Formatter_and_Quantization.md`
* `04_On_Chip_BRAM_Controller.md`
* `05_Hardware_Pooling_and_Activation.md`
* `06_DMA_and_AXI_Interface.md`
* `07_ESPNet_Encoder_Uop_Schedule.md`

本次文档收敛明确冻结了以下关键约束：
* 顶层采用单一 HLS NPU IP，内部以 SAE / Formatter / Pool-Act / BRAM Ctrl / DMA-AXI 五类子模块构成。
* 中间数据流不允许反量化回 float；`add` 和 `concat` 前必须做 requant / scale 对齐。
* 明确了 `uop / tensor_desc / param_desc` 描述符和寄存器接口格式。
* 明确了片上权重、特征图和量化参数的驻留策略，以及 `ESPNet_Encoder` 的执行顺序。

### 1.2 量化模型结构修正
为保证 PC 上的数据流能被硬件逐节点复原，今天完成了量化模型结构修正：
* 将 [Model_flattened_quantized.py](/D:/ESPNet/Model_flattened_quantized.py:19) 中所有 `add/cat` 节点改为各自独立的 `FloatFunctional`。
* 消除了原先多处 `torch.cat` 共用/隐式复用 observer 的问题，避免不同节点量化参数互相污染。
* 保留了对硬件友好的执行语义：节点输出的量化域边界清晰，可以一一映射到后续硬件 `uop`。

### 1.3 量化训练与评估流程修正
今天对 [test_single_image_for_all.py](/D:/ESPNet/test_single_image_for_all.py:31) 做了关键修正，核心是把“observer 校准”和“真实 fake-quant/QAT”分开：
* 新增了真正的 `FakeQuantize` 版硬件 qconfig，激活采用对称 `INT8`，权重采用 per-channel symmetric `INT8`。
* 修正了 `prepare_quantized_model()`，使其支持三条明确路径：
  * PTQ + convert 后端量化 kernel
  * 对称 INT8 fake-quant 仿真
  * 对称 INT8 QAT + fake-quant 评估
* 新增 fake-quant 评估分支，避免把 PyTorch CPU `QUint8` 后端 kernel 误当成我们目标硬件的数据流参考。
* 修正了 QAT 训练设备放置逻辑，使 QAT 和 fake-quant 评估可以跑在 GPU 上。

### 1.4 真实对称 INT8 精度恢复结果
今天重新建立了“硬件目标一致”的精度评估口径，即：**对称 INT8 激活 + 对称 INT8 权重 + fake-quant 数据流仿真**。

对照结果如下：

| 路径 | 条件 | 结果 |
| :--- | :--- | :--- |
| FP32 基线 | 原始验证集 500 张 | `mIoU = 0.8312184227` |
| 对称 INT8 PTQ | fake-quant，无训练 | `mIoU = 0.6681403996` |
| 对称 INT8 QAT | 全训练集，3 epoch，`lr=5e-5`，16 calibration batches | `mIoU = 0.8664571434` |

额外指标：
* 对称 INT8 QAT 导出模型复测 `Overall Acc = 0.9787953516`
* `Per-Class IoU = [0.75560664, 0.97730765]`

结论：
* 单纯 PTQ 对当前网络掉点明显，无法满足项目目标。
* 经过少量轮数的对称 INT8 QAT 后，模型精度已经恢复到可接受范围，并超过当前 FP32 基线。
* 这说明当前 `ESPNet_Encoder` 在对称 INT8 数据流下是可训练、可恢复、可部署的。

### 1.5 硬件 bit 级联调工件导出完成
今天补齐了可用于硬件 bit-compare 的导出工具 [export_quantized_artifacts.py](/D:/ESPNet/export_quantized_artifacts.py:113)，支持导出 fake-quant 模型的完整硬件参考数据。

已生成目录：
* [quantized_artifacts_qat_symmetric_3ep](/D:/ESPNet/quantized_artifacts_qat_symmetric_3ep)

其中包括：
* `fake_quant_state_dict.pth`
* `manifest.json`
* `activation_qparams.json`
* 每个卷积层的：
  * `weight_int8.npy/txt`
  * `weight_scales.npy/txt`
  * `weight_zero_points.npy/txt`
  * `bias_fp32.npy/txt`
* `golden_sample/`：
  * 单样本输入量化整数值
  * 每个卷积节点输出整数值
  * 每个 `FloatFunctional add/cat` 节点输出整数值
  * 各节点 `scale / zero_point / quant_min / quant_max`
  * 最终模型输出 `model_output_fp32.npy`

这批导出数据已经具备两种用途：
* 作为 HLS C-Sim / RTL Co-Sim 的逐节点 golden reference
* 作为后续板级调试时的 bit-level 对照目标

### 1.6 资源与时序可实现性评估完成
今天进一步对照了 FP32 阶段已经落地的板级平台与当前 INT8 规格，补做了资源/时序可实现性分析，得出以下结论：

* 当前板级平台确实保留了 `4` 个 AXI DMA，但在 INT8 `v1` top 方案中，这 `4` 个 DMA **不直接参与默认数据流**。
* 当前 INT8 top 采用的是单 HLS 顶层 + `m_axi` master 接口模式，外部等效上不再是“多 DMA 调度网络”。
* 之前 FP32 阶段真正导出的平台 `PL0` 实际频率是 `100 MHz`，不是历史估算中频繁引用的 `200 MHz`。
* 按当前文档原始写法，`FMEM0/1/2 = 4.5 + 4.5 + 2.5 MB` 总计 `11.5 MB`，对 ZU15EG 的 URAM 容量过于激进，缺乏实现余量。
* 因此今天已经将 INT8 文档统一修正为：
  * **首版目标频率：`100 MHz`**
  * **存储策略：URAM + BRAM 混用**
  * **FMBUF 总预算：压到 `8 ~ 9 MB`**

这意味着项目的首版目标已经从“直接冲 `200 MHz`”修正为“先在 `100 MHz` 下做出 bit-exact、可实现、可上板的完整 INT8 单 IP 版本”。

### 1.7 Final Spec 冻结完成
在完成资源/时序修正后，今天又进一步把 `01~07` 文档收敛成真正可编码的 final spec，重点补死了此前仍会导致实现分叉的几处：

* `06_DMA_and_AXI_Interface.md` 现在已经冻结：
  * 软件可见 wrapper IP
  * 固定 AXI-Lite 寄存器表
  * `uop` 的 `256-bit` packed 格式
  * `tensor_desc / scale_desc / opcode-specific param_desc`
  * `param_blob.bin` 的唯一内存镜像格式
* `03_Data_Formatter_and_Quantization.md` 现在已经冻结：
  * `pool` 的 scale 语义
  * 当前网络全部 `ADD` 走 `requant_bypass = 1`
  * `AFFINE_POST` 的完整参数接口
* `04_On_Chip_BRAM_Controller.md` 现在已经冻结：
  * `FMEM0/1/2` 固定容量
  * `WBUF/QBUF` 固定 BRAM 实现
  * `CAT -> ACT` 物理 alias 默认开启
* `07_ESPNet_Encoder_Uop_Schedule.md` 现在已经冻结：
  * `T_POOL_TMP` 独立建模
  * 最终 tensor bank/base offset
  * opcode-specific `param_id`
  * 当前网络全量 `uop` 执行表

这意味着下一步开始写 HLS 时，接口、数据流、量化域、片上存储和整网调度都已经具备唯一解释，不再依赖“实现时再决定”。

---

## 2. 今日关键结论 (Key Decisions and Conclusions)

### 2.1 PC 参考模型必须以 fake-quant 为准
PyTorch eager quantization 的 CPU 后端 `quantized::conv2d` 依赖 `QUint8` 激活输入，这与我们目标硬件的“对称 `INT8` 激活”不一致。因此：
* 不能把 PyTorch 后端 kernel 的整数执行结果当作硬件参考。
* 必须以 fake-quant 仿真的对称 `INT8` 数据流作为硬件对齐基准。

### 2.2 硬件侧必须显式支持 add/cat 前的 scale 对齐
`Model_flattened_quantized.py` 的量化图已经按节点切开 observer，说明后续硬件中：
* `add` 前两路输入必须对齐到同一目标 scale
* `concat` 前各分支必须先 requant 到目标输出 tensor scale
* 不应在中间节点进行 float dequant

### 2.3 当前阶段可以从“规格设计”切换到“编码实现”
截至今天，INT8 方案已经具备：
* 可执行的硬件规格文档
* 可恢复的量化模型训练流程
* 可导出的硬件 golden 数据

因此下一阶段的重心应该转移到 HLS 模块编码和 bit-exact 联调，而不是继续做宽泛的架构讨论。

### 2.4 首版实现目标已经调整为 100MHz
今天补充的一个重要决策是：

* **`100 MHz` 是当前更现实的首版验收频率**
* `150 MHz` 作为优化版目标
* `200 MHz` 只保留为后续冲刺目标

原因很明确：

* 当前 INT8 top 比 FP32 阶段的单个算子 IP 大得多
* 片上存储、window generator、post process、concat writer 和微指令控制器都被合并进了一个顶层
* 若继续维持过满的 URAM 规划并强推 `200 MHz`，实现风险过高

---

## 3. 当前风险与注意事项 (Risks)

### 3.1 fake-quant 与硬件整数实现之间仍需严格逐节点核对
虽然训练与导出流程已经就位，但硬件端若要做到 bit 级一致，仍需严查以下问题：
* rounding mode 是否与 PyTorch fake-quant 一致
* `clamp` 饱和边界是否严格采用 `[-128, 127]`
* per-channel weight scale 的应用顺序是否正确
* `add/cat` 前的目标 scale 选择是否与导出 manifest 完全一致
* BN folding / bias 累加 / requant 的定点顺序是否与 golden 保持一致

### 3.2 当前 golden dump 仅覆盖单样本
今天导出的 `golden_sample/` 已可作为硬件 bring-up 的第一份 bit-level 对照，但它只覆盖单个验证样本。
后续如果某一层出现数据歧义，建议再补导 3 到 5 个固定样本，分别覆盖：
* 正常道路场景
* 前景比例较低场景
* 边界/遮挡较复杂场景

### 3.3 精度超越 FP32 不能直接视作“无风险”
当前 QAT 结果高于 FP32 基线，说明训练有效，但也意味着：
* 需要固定随机种子和导出版本
* 后续不能随意改量化图或 observer 布置
* 否则很容易出现训练结果与导出结果不一致的问题

### 3.4 单 IP 架构的实现风险高于多 IP 架构
虽然单 IP 更有利于吞吐率和控制收敛，但也带来了新的实现风险：

* 片上大 bank 到阵列核心之间的布线更长
* dataflow 子模块更多，顶层互连更复杂
* AXI `256-bit` pack/unpack、window generator 和 post process 都会形成非纯计算路径

因此后续开发必须把“功能正确”和“实现收敛”并列看待，不能只看理论算力。

---

## 4. 下一步工作计划 (Action Plan)

### [Phase A] 建立软件节点到硬件 uop 的一一映射
这是接下来最优先的任务。
* 将 `Model_flattened_quantized.py` 中的每个卷积 / add / cat / BN-ReLU 节点映射到 `07_ESPNet_Encoder_Uop_Schedule.md`
* 为每个 `uop` 补齐：
  * 输入 tensor id
  * 输出 tensor id
  * 目标 activation scale id
  * 对应权重文件名
  * golden_sample 对照文件路径

### [Phase B] 编写第一版 HLS INT8 数据通路
建议按最小闭环顺序实现：
1. `Quant/Dequant-Free` 的对称 INT8 Conv 主路径
2. per-channel weight scale + bias 处理
3. requant/saturate 单元
4. `add` 单元
5. `concat` 写回控制

第一批建议先落地：
* `B1`
* `level2_0`
* `level2 block 0`

因为这部分已经具备完整的量化参数和 golden dump，最适合做 bit-compare 起点。

### [Phase B.5] 以 100MHz 为首版收敛目标
在真正开始写 HLS 代码前，今天已经明确：

* 首版资源规划必须按 `100 MHz` 验收
* `FMBUF` 不能继续按 `11.5 MB` 静态预留
* `WBUF/QBUF/line buffer/FIFO` 必须优先压到 BRAM
* `URAM` 只保留给大块 FMBUF

也就是说，后续 HLS 编码不能再以“默认 200MHz + 全大 bank URAM”作为隐含前提。

### [Phase C] 建立逐节点 bit-exact 验证流
* 为 HLS C-Sim 构建自动测试脚本
* 输入：
  * `weight_int8`
  * `activation_qparams`
  * `golden_sample` 整数输入
* 输出：
  * 每层输出与 `golden_sample/*/output_int.txt` 自动逐元素比对
* 第一阶段目标不是系统跑通，而是：
  * 单层 bit-exact
  * 单 block bit-exact
  * 再扩展到整个 Encoder

### [Phase D] 完成完整 Encoder 的端到端调度闭环
当 `uop schedule + HLS + golden compare` 跑通之后，再进入系统级目标：
* 完整支持 `ESPNet_Encoder` INT8 推理
* 减少中间特征图 DDR 往返
* 将端到端延迟压缩到 `50 ms` 级别

---

## 5. 今日交付物 (Deliverables)

今天形成的可复用交付物如下：
* `D:\ESP_INT8\int8_npu_design\01~07` 全套规格文档
* [test_single_image_for_all.py](/D:/ESPNet/test_single_image_for_all.py:31) 中修正后的 fake-quant / QAT 流程
* [export_quantized_artifacts.py](/D:/ESPNet/export_quantized_artifacts.py:202) 中支持 golden dump 的导出工具
* [quantized_artifacts_qat_symmetric_3ep](/D:/ESPNet/quantized_artifacts_qat_symmetric_3ep) 导出的对称 INT8 QAT 模型权重与量化参数
* 今天更新后的：
  * [01_System_Architecture.md](/D:/ESP_INT8/int8_npu_design/01_System_Architecture.md)
  * [02_Systolic_Array_Engine.md](/D:/ESP_INT8/int8_npu_design/02_Systolic_Array_Engine.md)
  * [03_Data_Formatter_and_Quantization.md](/D:/ESP_INT8/int8_npu_design/03_Data_Formatter_and_Quantization.md)
  * [04_On_Chip_BRAM_Controller.md](/D:/ESP_INT8/int8_npu_design/04_On_Chip_BRAM_Controller.md)
  * [05_Hardware_Pooling_and_Activation.md](/D:/ESP_INT8/int8_npu_design/05_Hardware_Pooling_and_Activation.md)
  * [06_DMA_and_AXI_Interface.md](/D:/ESP_INT8/int8_npu_design/06_DMA_and_AXI_Interface.md)
  * [07_ESPNet_Encoder_Uop_Schedule.md](/D:/ESP_INT8/int8_npu_design/07_ESPNet_Encoder_Uop_Schedule.md)
* 本日报告 [workplan_0423.md](/D:/ESP_INT8/report_and_workplans/workplan_0423.md)

codex resume 019dba04-b4bb-7490-9b93-f8f1122f09e1
