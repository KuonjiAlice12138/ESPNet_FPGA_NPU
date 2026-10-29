# workplan_1023：高分辨率 INT8 输入的轻量片上裁剪缩放

## 0. 结论与边界

**最终主方案：高分辨率 INPUTQ.BIN → 中心正方形裁剪 → 半像素中心最近邻字节搬运 → 原 encoder。**
以下尺寸均按 HxW：输入支持 `512x1024x1`、`256x512x1` 的**已量化有符号 INT8**，
encoder 仍接收 `240x240x1` INT8。解码、灰度、归一化和输入量化全部留在 Python/PS 端；
**本轮 FPGA 不增加量化 LUT、RGB 转灰、浮点归一化或 PNG 解码。**
取消的仅是新增输入预处理中的量化；原卷积/PPU内部的重量化与激活保持不变。

```text
Python/PS：图像 → 灰度/归一化/当前输入量化 → 高分辨率 INPUTQ
FPGA：高分辨率 INT8 INPUTQ → 裁剪/最近邻/打包 → 240x240 INT8 → 原 encoder
```

已完成调研、三组500张实验及两种尺寸的量化次序等价核查。Round 1已导出高分辨率INT8参考文件，
并实现输入DMA的裁剪/选样/打包及独立source-mode控制。Round 2的App源码适配和完整top CSim
已通过：直通、两种高分源、错误mode拒绝及切回直通均满足严格功能门槛。
**首轮综合于2026-10-24完成，高分源选样II=3、增量LUT/FF超预算，未进入实现；
Round 2R已修复到实际II=1、输入调度15377cycles，完整CSim通过。2026-10-25用户确认接受
增量LUT超预算681，冻结本轮HLS，直接进入100MHz实现，不再为该预算差额返修。
实现与`platform_h240_resize_1025`导出、App build及mode 2首轮上板已完成：512x1024 INT8输入
得到的240x240 mask与整数replay完全一致，PL实测49.879630ms。其他mode的板级测试及连续稳定性待补。**

基线为 `H240W240-STABLE-20261023` / `9516a54`，上板耗时 49.8544ms。
不重训、不改 QAT、模型拓扑、PARAM v4、SA/POST/PPU 或输出插值。实验与工具均放在
`D:/ESPNet/resolution_h240w240_gray/input_resize_1023/`，不放入 NPU 仓库的 tools。
本轮与尚未实施的 `workplan_1021` Conv 重构分开验收，不同时混入数据流大改。

**视野限制必须明确：** 推荐方案延续当前模型的中心视野，并不是整帧分割。
中心裁剪保留原图面积 50%，但在这 500 张图中只保留约 **40.00% 的车辆像素**，不是车辆实例数。
若应用要求整幅宽图的每个像素都被分割，本方案不能直接满足；不能把两侧未推理区域填成背景。
RGB888、PNG/JPEG 解码和真实相机域偏移未纳入本轮验收，后续路线单独见第7节。

## 1. 调研与算法取舍

这里的“压缩”指空间重采样，不是图像编码；不引入卷积、池化或可学习预处理网络。

| 方法 | 每个输出像素的主要工作 | 本轮判断 |
|---|---|---|
| 半像素最近邻 | 选一个源像素，地址加法/比较 | 默认方案，资源与状态最少 |
| 传统 floor 最近邻 | 选一个源像素，但采样相位不同 | 不能混用，512 输入有明显精度损失 |
| Q4/Q8 双线性 | 四点采样，分离插值约三个小乘法，需源行缓存 | 离线质量参照，不进入本轮INT8搬运路径 |
| AREA 面积重采样 | 汇聚多个覆盖像素，支持抗混叠，但控制/缓存更复杂 | 质量参照，不作为片上默认路径 |
| Bicubic/Lanczos/学习式缩放 | 更多采样点、乘加或模型参数 | 不符合本次低开销目标，不实现 |

Q4/Q8 仅指插值系数的4/8位小数，不是 W4A4；模型始终W8A8。
双线性/AREA与输入量化一般不能交换顺序，不得直接把这些历史实验改写为“对INPUTQ做插值”的精度。

[OpenCV 文档](https://docs.opencv.org/4.13.0/da/d54/group__imgproc__transform.html)
区分 nearest、nearest-exact、linear、area 等模式；最近邻并非只有一种坐标约定。
双线性的半像素坐标可核对
[OpenCV resize 源码](https://raw.githubusercontent.com/opencv/opencv/4.13.0/modules/imgproc/src/resize.cpp)。
[AMD Resize 文档](https://docs.amd.com/r/2023.2-English/Vitis_Libraries/vision/api-reference.html_2_87)
也提供 NN/双线性/AREA，其历史资源表显示面积法更重；表中不同器件/版本的数据不能当作本项目估计。
因此不直接引入通用 4K resize IP，而针对两个固定输入规格实现坐标发生器。

## 2. 实验设置与结果

### 2.1 可重复的比较方法

- 使用当前部署的 ReLU、W8A8、`p=q=1` 灰度模型，以及独立灰度 cache 的完整 500 张验证集。
  固定导出状态 `quantized_artifacts_binary2_int8_hw/fake_quant_state_dict.pth`，不更新 observer。
- 原始 Cityscapes 图为 `1024x2048`，先在主机生成两种待输入尺寸，再评估片上候选算法。
  主实验用 AREA 生成源图；第二组改用 LINEAR 检查对上游缩放方式的依赖，**不是要求 PL 先做 AREA**。
- 三组分别包含 21、7、11 个配置，共 19500 次“样本-配置”推理；每组均覆盖同样 500 张图，
  不是 19500 张独立图片。运行耗时约 238.1s、107.8s、143.2s，不是 FPGA 处理延时。
- 所有候选先在**相同的原始中心区域、相同的 240x240 GT** 上比较；宽视野预测通过 logits
  坐标映射投回此区域，而不是缩放 argmax mask。另存各自视野指标，但不跨视野直接排名。
  标签始终最近邻采样，车辆 trainId 13..18→0，其余有效类→1，255 忽略；letterbox 为240x120内容
  加上下各60行补边，padding标签忽略，图像补mean附近的灰度86。
- 开跑前确认原始 INPUTQ、1800 个 logits、57600 个 mask 字节均与已部署 golden 完全相同；
  推理常量缓存与原 QAT forward 相同。11 项数值/几何测试通过；权重与 cache 的原始 hash 未变化。
- 上述历史实验先对Gray8采样再量化。对本轮选定的最近邻，固定逐像素量化Q与选样N满足
  `N(Q(image)) = Q(N(image))`，饱和和舍入也不破坏该等式。两尺寸参考图均已核对0/57600字节差异，
  因此最近邻的500张精度结论继续适用；无需为移动量化位置重跑QAT或整组推理。

### 2.2 主实验：同一中心区域的全验证集指标

单位为百分比；两个分辨率栏依次为 `512x1024 / 256x512`。

| 几何与采样 | PA | mIoU | 车辆 IoU |
|---|---:|---:|---:|
| 原始大图中心裁剪、LINEAR 基线 | 96.9112 | 79.9629 | 63.1876 |
| 中心裁剪、LINEAR | 97.0047 / 96.9504 | 80.3140 / 79.5903 | 63.7902 / 62.3927 |
| **中心裁剪、半像素最近邻** | **96.9294 / 96.9701** | **80.0396 / 80.0694** | **63.3218 / 63.3361** |
| 中心裁剪、floor 最近邻 | 96.7191 / 96.8419 | 79.0426 / 79.3942 | 61.5479 / 62.1189 |
| 中心裁剪、Q4 双线性 | 97.0319 / 96.9447 | 80.4531 / 79.5452 | 64.0399 / 62.3083 |
| 中心裁剪、Q8 双线性 | 97.0159 / 96.9489 | 80.3965 / 79.5880 | 63.9438 / 62.3898 |
| 中心裁剪、AREA | 97.0146 / 96.9418 | 80.1877 / 79.5301 | 63.5245 / 62.2810 |
| 整帧拉伸、LINEAR | 96.2769 / 96.1736 | 73.9621 / 71.9502 | 51.8025 / 47.8660 |
| 等比例缩放加上下补边、LINEAR | 94.1755 / 94.5925 | 70.1014 / 71.1584 | 46.3359 / 48.0084 |

工作门槛预设为相对同源中心裁剪 LINEAR：mIoU 下降不超过 0.5pp、车辆 IoU 不超过 1.0pp、
PA 不超过 0.2pp。半像素最近邻在两种输入下均通过；512 的 mIoU 差为 -0.2744pp，
车辆 IoU 差 -0.4685pp。它并非最精确，但不值得为约 0.4pp 的额外 mIoU 默认加入 Q4 插值硬件。
整帧拉伸和 letterbox 在相同区域均明显退化，不能只看整体 PA 较高就接受。

### 2.3 上游与视野敏感性

源图改用 LINEAR 生成后，512 的上述指标不变；256 的结果如下：

| 256 输入的中心裁剪方法 | PA | mIoU | 车辆 IoU |
|---|---:|---:|---:|
| LINEAR | 97.0047 | 80.0253 | 63.2085 |
| **半像素最近邻** | **96.8835** | **79.8287** | **62.9481** |
| Q4 双线性 | 96.9972 | 79.9542 | 63.0735 |

最近邻仍通过点估计门槛；其 mIoU 相对原始已部署基线仅下降 0.1342pp。
按图片配对 bootstrap，512 相对同源 LINEAR 的 mIoU 差 95% 区间为 [-0.4726,-0.0705]pp；
256/LINEAR 源图为 [-0.6202,+0.2186]pp。因此只称**验证集工作门槛通过**，不宣称零损失或
统计上已证明所有输入都不劣于 0.5pp；真实相机输入仍需补测。

还评估了宽度为高度 9/8、5/4、3/2 的中心矩形 ROI，再轻度拉伸至正方形：

| ROI 宽/高 | 原图面积覆盖 | 车辆像素覆盖 | 最近邻 mIoU：512 / 256 |
|---|---:|---:|---:|
| **1（默认）** | **50.00%** | **40.00%** | **80.0396 / 80.0694** |
| 9/8 | 56.25% | 46.44% | 79.7729 / 80.0330 |
| 5/4 | 62.50% | 53.30% | 79.3635 / 79.4739 |
| 3/2 | 75.00% | 67.66% | 77.9711 / 78.0844 |

扩大 ROI 能增加覆盖，却不能形成两种输入统一、稳健优于默认方案的配置；9/8 在 512 上
相对同源 LINEAR 已低约 0.54pp。本轮保持训练时的正方形视野，不增加这些 runtime 模式。
若整帧覆盖成为硬约束，应另做全视野几何下的模型适配/QAT，而非偷偷扩大本方案的精度结论。

### 2.4 Round 2整网验收发现的参考差异

首次CSim中两种输入选样和mask均0 mismatch，但512源的logits在`[22,8,1]`为39，旧研究参考为40。
独立PARAM整数replay与HLS的1800个logits全部一致。审计26层卷积发现qparam一致，仅卷积19
`level3_blocks.0`的OIHW权重`[24,86,0,0]`不同：研究helper把26.5按远离零舍入成27，
原导出器`torch.round`按偶数舍入后实际存入PARAM的是26。不是输入DMA、布局或跨帧状态错误。
保留旧研究结果；正式验收在`selected_int8_reference`生成独立整数replay的L/M，不读取CSim输出
来制作golden，也不改HLS、QAT/checkpoint、PARAM或输入字节。两张参考的最终mask与旧研究结果相同。
第2节500张指标属于原研究模型比较，不冒充当前PARAM的完整500张上板精度；以后重训时须统一权重舍入。

## 3. 选定算法的精确定义

源高 `S=512或256`、源宽 `2S`，源数据为紧密排列的C1 INT8 INPUTQ，行stride为`2S bytes`。
ROI 为 `(x0=S/2,y0=0,w=S,h=S)`。对目标坐标 `d=0..239`：

```text
t(d) = floor(((2*d+1)*S)/480)
dst_q[y,x] = src_q[t(y), S/2+t(x)]
```

不得换成 `floor(d*S/240)`，后者就是本次测出退化的不同采样相位。
INT8字节原样保留补码，禁止再次量化、减128、重标定或截断到更低位宽。源文件无新增数据头，
shape/dtype/量化契约写入主机manifest并由mode选择，不塞进PARAM的模型tensor尺寸。

Python/PS离线输入量化必须沿用当前规则：mean=85.97366333007812、std=48.423927307128906、
normalization_divisor=255、input_scale=0.000107360385300126、zero_point=0，以及原舍入/饱和规则。
禁止按高分辨率图片重新校准scale或做逐图归一化。现有量化辅助函数可在主机复用，不能生成HLS量化表。
**Gray8和INT8文件可能长度相同，不能仅凭BIN后缀或文件大小判定格式。**

纵向坐标使用加法/比较 DDA，不在像素循环内做乘除法；横向等价规律也列在下表，
Round 2R改为直接连接固定字节位置，不再逐像素运行横向DDA：

| S | 初始 t | 初始 error | 每步基本增量 | error 每步增加 | 阈值 |
|---|---:|---:|---:|---:|---:|
| 512 | 1 | 32 | 2 | 64 | 480 |
| 256 | 0 | 256 | 1 | 32 | 480 |

每取一个输出后，error 加增量；若 >=480 则减480，并给 t 的基本增量再加1。
两规格均为 15 相位周期。ROI/stride 为常量组合，地址不需要通用乘法器。
横向每15个输出分别取自16/32个源字节：256源取`0..6,8..15`，512源取
`1,3,5,7,9,11,13,16,18,20,22,24,26,28,30`；源块和输出块连续衔接。
NPU 输入仍紧密排列为 57600 bytes/1800 个 256-bit word；240-byte 行末的 16-byte carry
必须带到下一行，不能误填充成 256-byte 行。

## 4. 预期硬件开销与延时预算

DDR 中保留完整的高分辨率INT8输入，但PL只对240个被选中的源行突发读取中心ROI，不扫描全部元素。

| 源INPUTQ规格 | 文件bytes | 选中行的ROI bytes | 256-bit读数 | 片上写数 |
|---|---:|---:|---:|---:|
| 512x1024 | 524288 | 122880 | 3840 | 1800 |
| 256x512 | 131072 | 61440 | 1920 | 1800 |

这是已推导并由参考地址序列确认的访问量，不是测得的 DDR 带宽。只需约两个源 word 的窗口、
输出 pack 寄存器和浅 FIFO，不需两行缓存、整帧缓存、卷积、池化或插值乘法器。
相对原Gray8预案，还取消了量化ROM及其多路复制；FPGA只有地址、选样与打包逻辑。
首轮以新增 DSP/BRAM/URAM=0，LUT<=1500、FF<=2000 为**预算**，实际以同级综合和实现为准。

选择 **4 个输出像素/clk**，不是无依据地扩大并行度。100MHz 下，忽略存储/控制的计算下界为：

| 并行像素数 | 57600 个输出的下界 | 假设替换原 FRAME_LOAD 后的整网下界 |
|---|---:|---:|
| 1 | 0.576ms | 50.29003ms |
| 2 | 0.288ms | 50.00203ms |
| **4** | **0.144ms** | **49.85803ms** |

Round 2R保留4字节输出组，但每15像素块用4个固定service slot，连续4块输出3/4/4/4组；
总计15360次service迭代、14400个有效组，不是连续每拍都输出4像素。只有实际II=1且无停顿时，
循环主体才约0.1536ms，不能把迭代数当成已测cycles；AXI、填充与排空继续计入下述预算。

旧 FRAME_LOAD=14037 cycles/0.14037ms，整网只有0.1456ms的 <50ms 裕度。
新输入阶段的工作预算为 **<=24000 cycles/0.24ms**；其余 stage 不变时整网约49.95403ms。
这只是门槛预算，不是性能承诺；必须计入 AXI 启动、cache refill、pack、写入及排空。
若外部实际是必须逐元素接收的流，不能跳过DDR行，本表不适用。PS端图像解码/量化与SD I/O
不在MODE_RUN计时内；评估系统端到端延时应另计，不能把这些工作消失在性能报表中。

## 5. 下一轮实施步骤

### Round 1：固化接口、参考数据和最小输入通路

- [x] 在 `D:/ESPNet/resolution_h240w240_gray/input_resize_1023/input_resize_1023.py` 增加
  `--export-prequant-reference`，输出独立 `selected_int8_reference/`：`I512Q.BIN`为524288 bytes，
  `I256Q.BIN`为131072 bytes；均先按当前输入契约量化高分辨率灰度数据。manifest保存
  `dtype=int8/layout=HWC/channels=1`、shape/stride、源样本、qparam、PARAM及文件hash；SD名满足8.3。
  不调用高分辨率encoder、不重编模型或修改compiler的240x240 tensor/layout，不覆盖旧参考文件。
- [x] 逐字核对新文件裁剪选样结果与原 `selected_reference/Q512.BIN/Q256.BIN` 完全一致；
  Q继续引用原目录；正式L/M改由PARAM整数replay生成，修复第2.4节发现的研究权重舍入差异，
  保存到`selected_int8_reference`，不覆盖旧研究参考、不复制整套artifact。
  补测试覆盖-128/-1/0/127、256个灰度量化值、坐标/打包边界以及拒绝错误输入契约。
  dtype/scale等错误由主机导出/部署检查拒绝；FPGA只验mode和边界，不能凭字节值识别Gray8/INT8。
- [x] 在原 `frame_dma.cpp` 内实现唯一预处理 owner，不新建 engine CPP。支持
  `src_mode=0`：原240x240 INT8直通；`1`：256x512 INT8选样；`2`：512x1024 INT8选样。
  复用原 gmem 输入、唯一 RAM writer；输出统一为1800 word。仅输入侧使用简单前向 DAG。
- [x] 按4元素小组生成坐标、选择缓存中的原始INT8字节并打包；不加入量化查找表或像素数值转换。
  DDA/源规格常量放入原 `include/npu_config.hpp`，不新增量化header。使用窄位宽索引，避免大
  complete-partition 数组的动态写；DDR 读取为选定行内连续 burst，不做逐像素随机 AXI 访问。
  不引入 Conv/PPU 跨行常驻 DATAFLOW、反馈队列或中间 DDR 写回。
- [x] `int8_core.cpp` 新增独立的 source-mode AXI-Lite 控制，帧开始锁存；外部输入口最大 depth
  改为16384 word。**不能把 INPUT_FRAME_BYTES/模型输入尺寸改成原图大小**，否则会扩大 FM 存储。
  不支持的 mode 在启动固定token通路前报错；DATAFLOW 子任务保持 void，不能提前退出导致消费者等待。
  FRAME_LOAD stage 覆盖整个新通路，保留全部原 counter，不重复计算预处理和输入写入时间。

### Round 2：App/TB 对齐与一次完整功能验收

- [x] 修改App的 `app_config.h`、`main.c`、`hal/int8_npu.h/.c`：最大输入缓冲扩为524288 bytes，
  按mode读取实际长度，设置新寄存器并保持64-byte对齐。更新平台/BSP后用生成驱动，不手写旧寄存器offset。
  模型PARAM继续用原P2G，输出仍57600 bytes。高分辨率文件选I512Q/I256Q，禁止误用G512/G256。
  上板准备时在主机核验INT8输入manifest，App不新增JSON解析；MODE_RUN计时只包含PL选样和原推理，
  不包含PS预处理/SD读取。
  默认一次启动依次测试mode 0/1/2，输出O2G/O256Q/O512Q及分mode统计；源码已适配，
  新平台/BSP生成与实际App编译仍待手动综合/导出之后。缺少`src_mode`生成接口时必须编译失败，不能手写offset。
- [x] 扩展 `tb/frame_dma_tb.cpp`、`tb/top_golden_sample_tb.cpp`，分别指定原PARAM目录和新参考目录，
  不复制整套artifact；读新I512Q/I256Q输入，先逐字检查选样后的Q512/Q256，再检查对应L/M。
  新源图的golden不能错误地使用旧单图golden；先用单元TB覆盖符号极值、跨word/行及末word。
  在一次top CSim中覆盖旧直通、两种源尺寸、连续切换/重入以及错误mode；预处理输出、
  logits、mask要求分别0/57600、0/1800、0/57600 mismatch，合法模式last_error=0。
  两个高分源的L/M取`selected_int8_reference`，不能继续使用旧研究L512；Q仍取原`selected_reference`。
  最终一次top CSim约6分52秒，mode 0→1→2→非法mode→0全部通过；每次合法执行input/logits/mask
  分别0/57600、0/1800、0/57600 mismatch，last_error=0；非法mode 3/0xffffffff在DMA前拒绝且不改写guard。
- [x] 清除本轮输入预处理中的试验性双线性/其他裁剪路径及临时helper，保留仍需要的旧INT8直通；
  不保留输入侧量化/灰度/RGB预处理分支，不清除网络内部重量化、主机离线量化工具、input DMA writer、
  encoder原有Pool或输出双线性。
  单datapath、唯一input writer、无输入数值运算/大数组的源码结构检查通过；全CPP语法/未引用静态函数扫描通过。
  DSP/RAM/II、实际端口冲突与搜索时间必须继续由综合报告确认，不把源码预算当成已综合资源。

### Round 2R：首轮综合未过门槛，先修输入DMA

2026-10-24首轮综合正常结束，HLS约11m25s、峰值2.892GB，未出现搜索失控。
报告目录为`ESP_INT8_hls/ESP_INT8_hls/hls/syn/report`；同级比较如下：

| 指标 | R8稳定基线 | 本轮输入选样版本 | 判定 |
|---|---:|---:|---|
| 顶层LUT | 343789 | 347505（+3716，+1.08%） | 超新增1500预算 |
| 顶层FF | 178344 | 180907（+2563，+1.44%） | 超新增2000预算 |
| DSP / BRAM18K / URAM | 1269 / 661 / 28 | 1269 / 661 / 28 | 无新增，符合预算 |
| HLS估计周期 | 7.300ns | 7.300ns | 非物理时序验收 |
| 高分源4像素组选样II | 目标1 | 实际3 | 吞吐门槛未过 |
| 高分源FRAME_LOAD | 预算<=24000cycles | 调度估计43216..43217cycles | 延时门槛未过 |

原生mode 0的直通循环仍为II=1；43217不是原生输入的实测时间。高分源循环的14400次迭代
是固定工作量，实际II=3对应43203cycles，不能用顶层巨大动态max latency来掩盖该局部问题。
若其余stage保持R8实测不变，以43217替换原14037cycles，整网条件估计约50.1462ms；
这不是上板测量，AXI停顿还可能增加延时。暂不因仅多约0.3ms而放弃原<50ms目标。

根因证据：`hls_compile.log:1627..1631`明确指出`frame_dma_select_pack`内
`source_words`在行初始化的`low`读取（源码65行）与条件refill的`high`读取（85行）之间
存在携带依赖；HLS把外层行循环和内层group循环展平后，II=1/2均失败，最终II=3。
这不是DDR未推断burst：两种输入读取均已推断256-bit连续burst，writer实际II=1。
选样模块LUT=2754、FF=1967，是新增逻辑的主要热点；其主动态移位器裁窄到312bit后仍需1217 LUT，
另三个64bit移位器各179 LUT。整个输入DMA为3830 LUT、2977 FF，无DSP或RAM。

生成RTL核对Conv/SA/PPU consumer仍各一份，SA=1024 DSP，没有本轮新增datapath克隆；
三个AXI master和全部profiling接口保留，新增`src_mode`正确映射到AXI-Lite的0x44，旧offset不变。
报告审计无硬性blocker，但这不代表预算通过。日志末尾通用“loop constraints satisfied”
也不能覆盖选样报告中的实际II=3。R8物理CLB已用95.27%，不应直接把额外LUT交给实现碰运气。

下一轮仅修改输入DMA及必要TB，不改模型、PARAM、Conv/SA/PPU、counter或主cfg：

1. 重构行初始化与源word refill的读取顺序，优先使用单一FIFO读取入口/简单状态机，或明确分离
   读取与选样，避免展平后多个读取点形成跨行依赖。保持单向浅FIFO、唯一input RAM writer，
   不用虚假`DEPENDENCE inter false`消除真实FIFO顺序，不新增整行/整帧缓存。
2. 缩小字节选择网络，利用固定两种规格及15相位规律，把宽动态移位和不必要的通用坐标逻辑
   改成有界小范围选择。保持半像素最近邻逐字节一致、240-byte行的跨行pack carry和精确token数；
   不以复制两个完整sampler、降低并行度或增加输入量化来换取II。
3. 清理替代后的旧helper，先跑输入DMA轻量CSim；最后一次top CSim检查三mode、重入、非法mode及
   input/logits/mask全部0 mismatch，再交用户手动综合。复查高分源实际II=1、FRAME_LOAD调度预算
   <=24000cycles、增量LUT<=1500/FF<=2000、DSP/RAM不增加；若报告未满足，不直接推进实现。

2026-10-25交付检查：上述修复及旧横向DDA/宽窗口清理已完成；74项契约检查、P7结构检查、
14个CPP的综合模式语法/未引用static函数扫描通过。DMA轻量CSim与最终top CSim通过，后者约5m57s；
三种输入及重入的input/logits/mask分别0/57600、0/1800、0/57600 mismatch，非法mode guard通过。
生产cfg、模型/PARAM、Conv/SA/PPU及接口未改；本轮未运行综合，II/资源/时序仍须用新报告确认。

#### Round 2R综合复查：吞吐恢复，LUT预算尚未完全通过

2026-10-25 15:24本轮报告确认使用新15像素固定映射和15360次service循环；综合约11min，
峰值3.907GB，无搜索失控。与首轮输入扩展版本比较：

| 指标 | 首轮 | Round 2R | 门槛结果 |
|---|---:|---:|---|
| 高分源选样实际II | 3 | **1** | 通过 |
| FRAME_LOAD调度max cycles | 43217 | **15377** | 低于24000，减少64.42% |
| 选样模块LUT / FF | 2754 / 1967 | **1219 / 729** | 明显下降 |
| 输入DMA总LUT / FF | 3830 / 2977 | **2295 / 1739** | 无新增DSP/RAM |
| 顶层LUT | 347505 | **345970** | 对R8增2181，仍超1500预算681 |
| 顶层FF | 180907 | **179669** | 对R8增1325，低于2000预算 |
| DSP / BRAM18K / URAM | 1269 / 661 / 28 | **不变** | 通过 |

选样局部估计周期从7.077降至4.611ns，顶层仍为7.300ns；原生直通和input writer均实际II=1。
AXI burst、`src_mode=0x44`、全部profiling端口及Conv/SA/PPU单实例均保留，SA仍1024 DSP；
报告审计0 blocker，无新增II/clock violation，原有Conv DATAFLOW提示不视作本轮退化。
以15377替换R8原输入stage的14037cycles，其余stage不变时整网条件估计为49.8678ms；
不包含未建模的AXI停顿，不作为上板结果或<50ms保证。

**结论：性能/FF/存储门槛通过，仅增量LUT预算未过。2026-10-25用户明确接受该差额，
批准按当前Round 2R版本直接进入100MHz实现；本轮LUT预算作为已批准例外，不再阻塞实现。**
剩余热点仍只在输入侧：reader为853 LUT，其中mux估计405 LUT；选样的128bit字节对齐移位
估计423 LUT。生成RTL已将位移量约束为0/8/16/24，Vivado可能进一步简化，因此这些是HLS估计，
不能当作已经验证的物理LUT或不可布线证据；本次预算放宽仅适用于当前版本，不取消物理验收。
以下小修暂停，仅在实现报告显示有必要时再评估：显式固定拼接替代该对齐移位、减少保留整256bit源word/半word选择的状态开销；
若仍需压缩，再评估共享reader的原生/高分辨率burst控制，必须保留连续burst、精确token数、II=1
及现有服务迭代数。不改模型、PARAM、Conv/SA/PPU或counter，不用降低并行度换资源。

### Round 3：实现与上板门槛

2026-10-25实现验收：100MHz，WNS=+0.667ns、WHS=+0.010ns，TNS/THS=0，391870条可绕线网络
全部合法布线，错误0。平台及export中的XSA/bitstream与最新Vivado输出SHA256一致；已安装输入DMA
及选样RTL与本轮csynth逐字节一致，顶层差异仅为package添加的仿真deadlock检测代码。
本轮App仅运行`src_mode=2`，读取`I512Q.BIN`（524288 bytes）、原`P2G.BIN`（143040 bytes），
生成`O512Q.BIN`（57600 bytes），比较`selected_int8_reference/M512.BIN`，不能与原生输入golden混用。
首轮mode 2的上板结果如下；其他mode及连续/跨图验收仍未完成，不能把本次单图成功标记为整轮结束。

- [x] 当前平台完成综合/package、100MHz实现及App手动build；新寄存器接口与实际mode 2运行一致。
- [x] mode 2首张图mask逐字节正确、ERROR=0，FRAME_LOAD<=24000且本次PL/ARM均<50ms；冻结的推理stage无周期回退。
- [ ] 三种mode各连续运行至少3次，逐字比较各自参考，记录每次PL/ARM/ERROR与mode切换；尚不宣称稳定<50ms。
- [ ] 补真实上游离线量化输入与跨图验证。若异常先查量化契约、坐标、打包或AXI停顿；不改输入量化边界，
  不擅自开启INT8双线性，不缩小SA或删除counter。

#### 首轮上板：mode 2正确，新增输入通路仅增加约25us

App=`INT8-BOARD-20261025-P7-H240-RESIZE-H512-B2`，平台=`platform_h240_resize_1025`。
实际加载`P2G.BIN`=143040 bytes、`I512Q.BIN`=524288 bytes，完整PARAM v4/75 UOP/16 EXEC，
`src_mode=2`。片上对中心512x512区域选样到240x240，再运行原网络；不包括整幅宽图两侧。

| Stage | 本次cycles | 本次ms | 相对R8原生输入cycles变化 |
|---|---:|---:|---:|
| FRAME_LOAD（裁剪/选样/打包/输入写入） | 16524 | 0.16524 | +2487 |
| Conv row datapath | 3418908 | 34.18908 | 0 |
| PPU row（含输出上采样） | 289114 | 2.89114 | 0 |
| BLOCK5 final | 658246 | 6.58246 | 0 |
| Vec | 433904 | 4.33904 | 0 |
| Pool | 135617 | 1.35617 | 0 |
| IDLE / MainCtrl / weight load合计 | 35650 | 0.35650 | +36（均为IDLE） |
| **TOTAL** | **4987963** | **49.87963** | **+2523** |

13个stage求和与total/window完全一致，ERROR=0；WIN/SA/POST各自求和均为3418908，且各状态值与R8
完全相同。三者在Conv内重叠，不能相加或将等待计数当成纯计算周期。PARAM_INIT=0是因为MODE_INIT
不在本次MODE_RUN窗口；计时同样不含PS解码/灰度/量化、SD I/O及cache维护，不能作为系统端到端延时。

ARM为`1662645/33333000=49.879848798ms`，与PL仅差0.219us；终端50ms是整数舍入，实际两者均<50ms。
相对R8的49.854400ms，整网仅增加25.23us（0.0506%），其中输入stage增24.87us、IDLE增0.36us。
这是**新输入通路相对旧直通的增量**，不是单独最近邻算术核延时。输入stage仅占整网0.3313%，
低于24000cycles预算7476cycles；相对HLS调度15377多1147cycles，现有counter不足以进一步细分差额。
全网<50ms裕度仅12037cycles=120.37us，仍须连续运行确认，不能以单次结果保证长期稳定。

实读SD核验：P2G/I512Q的大小和SHA256均与`RESIZEQ.json`一致；O512Q为57600 bytes，
与`selected_int8_reference/M512.BIN` **0/57600 mismatch**，invalid=0、c0=4145、c1=53455，
与UART一致。输出SHA256=`c2f7e2ccae286da5b3ad654849694198f8c914f220ecb05a5e149b944b521331`。
与原生输入golden相差122个像素是预处理不同的预期结果，不能误作错误；本次没有板级logits导出，
也没有重做500张验证集PA/mIoU，结论限于本图mode 2的mask与部署整数参考一致。

**下一步：冻结已通过的输入DMA，不再为已接受的681 LUT预算差额返修。** 先完成上述重复/跨mode验收；
若后续需要更多<50ms裕度，应回到`workplan_1021`的Conv数据流方向，而不是继续压缩仅占0.33%的输入stage。

## 6. 交付文件与复现

实验目录的 `README.md` 包含三组完整复现命令；已有结果默认拒绝覆盖。
`results_area500.json/.csv`、`results_linear500.json/.csv`、`results_partial500.json/.csv`
保存指标与逐图confusion，bootstrap可重算。`test_input_resize_1023.py` 当前16项通过，包含有符号
字节、量化/选样次序、错误输入契约及golden语义替换拒绝测试。

现有 `selected_reference/RESIZE.json` 是原Gray8研究参考，保存ROI、DDA及**离线**量化LUT；同目录：
`G512.BIN/G256.BIN`为原始Gray8帧，`Q*.BIN`为量化输入，`L*.BIN`为HWC logits，`M*.BIN`为mask。
两规格DDA与直接公式、LUT与原量化行为均0 mismatch；文件长度及SHA256已复核。
主方案的`selected_int8_reference/I512Q.BIN`、`I256Q.BIN`及`RESIZEQ.json`已导出；manifest
引用旧Q，并绑定新L/M的类型、长度和hash，`golden_authority=PARAM_V4_INTEGER_REPLAY`；
不复制原PARAM。两尺寸经真实输入DMA与片上memory writer逐字核对，均
0/57600 mismatch；符号极值、跨word/行、切回直通、写入边界及FIFO排空检查通过。
复现：在实验目录运行`python -B input_resize_1023.py --export-prequant-reference`；部署前运行
`python -B input_resize_1023.py --validate-prequant-manifest selected_int8_reference/RESIZEQ.json`。
验证入口为`ESP_INT8_hls/run_csim_input_resize.tcl`，默认一次完整top CSim；设置环境变量
`ESP_INT8_INPUT_RESIZE_TEST=unit`可只跑输入单元。生产cfg和原模型产物未改；App源码支持三种
source mode，当前只选择mode 2；新平台、App build及该mode首轮上板已完成。
CSim的最大stream深度3840来自顺序C执行，不代表硬件FIFO深度；源码为16/4。
Round 2R综合确认高分源实际II=1、FRAME_LOAD调度max=15377cycles；增量LUT超原预算681已获接受，
物理时序与mode 2单图板级验收已通过，输入stage实测16524cycles；有界FIFO跨mode/连续帧稳定性仍待补测，
不将4像素/拍的峰值当成全程每拍均有效。
旧Gray8研究目录不部署为FPGA的raw输入模式。
模型导出state SHA256=`04f2dc7bd072ce601da1a4f0e997fef95926ebe8c0b49644eba860e0e73d2403`；
PARAM SHA256仍为`9e380cebe305c4945642c4a334f75a7a2ba94eefa1f2ccc7f2964f8d3bf9d878`。

## 7. 后续可选路线：像素预处理与PNG解码分开

**本轮不实施、不综合这些分支，也不与主方案共占资源。** 若需要接收相机/原始图像，优先考虑
`PS解码PNG → DDR中已解码RGB888 → PL选样/灰度/归一化量化 → encoder`。这会增加颜色变换及
量化表，必须另立接口与精度验收；原生1024x2048 RGB的精度不能直接套用本轮两种灰度输入结果。
现有PNG直接灰度解码与普通BGR2GRAY初查并非逐字相同，需先对齐解码/色彩处理/舍入行为。

在DDR中RGB已解码、仍只突发读取选中ROI行的假设下，512x1024 RGB要读368640 bytes，
1024x2048 RGB要读737280 bytes；256-bit/100MHz满带宽下仅读取下界就为0.1152/0.2304ms，
尚不含计算、仲裁和排空。原尺寸RGB已接近0.24ms输入阶段预算，不能承诺整网仍<50ms。

若要连PNG文件解码也放到PL，则新增的是独立解码子系统，不是缩放模块：需要处理
[Deflate、PNG滤波与文件结构](https://www.w3.org/TR/png-3/#10Compression)。Deflate历史窗口最大32KiB，
还需RGB行缓存和其他状态。仅按原图2097152个像素计算，100MHz下若每拍分别输出1个或4个RGB像素，
输出全帧分别约20.97/5.24ms；这是条件吞吐估计，不是已测解码性能。当前资源/时序紧张，
暂不把PNG解码器塞进NPU，优先保留PS解码，之后单独评估是否值得硬件化。

**Round 1/2及Round 2R完整CSim功能验收完成；Round 2R综合通过输入延时/FF/存储门槛，增量LUT超预算681已获用户接受，可进入实现。
物理时序、平台/BSP、App build及mode 2首张图mask/<50ms已验收；多mode、连续与跨图门槛仍待完成。
第7节RGB/PNG路线仍仅为后续讨论，不进入当前输入通路。**
