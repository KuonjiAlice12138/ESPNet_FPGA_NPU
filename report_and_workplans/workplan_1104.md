# workplan_1104：中心区域推理与512x512高分辨率mask重建

## 0. 目标与基线

**目标：保持240x240灰度INT8 encoder不变，片上将高分辨率中心区域下采样后推理，再直接由30x30 logits重建512x512 mask，与同一高分辨率区域的真实标注计算PA/mIoU。**
尺寸均为HxW。2026-11-08已完成Round 0离线验收，以及获准后的Round 1/2源码改进、清理和最终top CSim。
encoder/模型/PARAM保持基线，默认输出现为512x512；**综合、100MHz实现及首张mode 3上板通过**。
本图mask与整数golden为0/262144 mismatch，PL延时54.12012ms；后续仍需重复帧/mode切换及板级验证集验收。
代码归档分支：`feature/h240-mask512`；初版标签：`H240-MASK512-INITIAL-20261110`。
执行时使用`executing-plans`，逐轮核对gate。

```text
512x512 INT8 / center_crop(512x1024 INT8) -> 512x512 ROI
512x512 ROI -> half-pixel NN -> 240x240 INT8 -> encoder -> 30x30x2 INT8 logits
30x30x2 INT8 logits -> bilinear + argmax -> 512x512 UINT8 mask -> DDR
```

512x512正方形输入无需再次空间裁剪；宽图输入仍只保留中心区域。上采样不会恢复被裁掉的两侧，
也不会重新获得下采样丢失的细节；本轮验证的是**低分辨率推理在高分辨率像素网格上的分割效果**。

基线为`feature/h240-resize`、`f40a820`、标签`H240-RESIZE-INITIAL-20261029`。
已上板的512x1024输入、240x240输出：4987963 PL cycles，100MHz下49.879630ms，
FRAME_LOAD=16524 cycles，ERROR=0，mask与整数参考0/57600 mismatch。该结果是单图，不是500张精度验收。
旧`workplan_1023`的PA/mIoU是在240x240 GT上得到，**不得直接作为本轮512x512指标的基准**。

## 1. 全轮次约束与接口契约

- 冻结模型、ReLU、W8A8量化、PARAM v4、75 UOP/16 EXEC及权重/输入量化参数；先不训练或QAT。
  PARAM继续使用`hw_artifacts/binary2_int8_h240w240_gray_r6_v4`，SHA256应保持
  `9e380cebe305c4945642c4a334f75a7a2ba94eefa1f2ccc7f2964f8d3bf9d878`。
- encoder输入始终240x240x1，输出始终30x30x2；SA保持32x32，Conv/WinGen/PPU/Vec/Pool结构不重构，
  不增加U71融合、整帧中间访存或新的可学习decoder；输入解码、灰度和量化仍在Python/PS端。
- 新增`INPUT_SRC_H512W512=3`：源数据为连续C1有符号INT8，262144 bytes、行跨度512 bytes。
  对应ROI=(0,0,512,512)。保留mode 0/1/2原有输入语义，mode 2的ROI仍为(256,0,512,512)。
  无自动识别尺寸；错误mode必须在DMA前拒绝。
- 正方形与宽图复用同一512->240选样器，仅改变reader的行跨度和ROI偏移；不得克隆输入DATAFLOW。
  最大输入容量仍为524288 bytes，不能把新增正方形输入误当成宽图的前262144 bytes。
- 第一版IP的输出**编译期固定512x512**；保留240输出的编译配置用于回归，单次综合只存在一套输出引擎。
  暂不新增运行时输出尺寸寄存器或任意分辨率resize。旧240输出App不能直接驱动新IP。
- 独立定义encoder几何与mask几何。`UPSAMPLE_SCALE=8`代表原30->240关系，不能改成17或用它推导512。
  新输出262144 bytes、8192个256-bit AXI word；同步修改输出m_axi/ap_memory depth、TB及App容量。
- 保留`upsample_unit.cpp`为输出重建owner，保留PPU逐logits行调用和现有profiling接口；
  不新增第二套upsample cpp、512x512 logits缓存或高分辨率特征图RAM。不用ALLOCATION/Tcl掩盖复制。
- 不自动运行csynth、实现、build或写SD；由用户手动推进。代码改完清理dead/legacy logic，再统一验证。

## 2. 高分辨率评估与golden

### 2.1 必须比较同一视野

主指标：`512x512预测mask`与**原输入所覆盖区域的512x512真实标签**逐像素比较，而非240x240标签。
若已有与源图严格配准的512标签，原样使用；不能先缩成240再放大作为“高分辨率GT”。
若从Cityscapes原1024x2048标注生成，保留中心1024x1024，再按半像素最近邻取到512x512，
记录原始ROI和采样规则；图像生成的缩放方法也必须写入manifest。512宽图中心ROI的标签须与此视野一致。

车辆trainId 13..18映射为0，其余有效类为1，255忽略；已映射的二分类标签不得重复按trainId解释。
从原标注PNG读取，检查样本ID、路径及hash，不能使用不匹配的旧`city.p`/240标签缓存。
先累加500张图的有效像素confusion，再计算总体PA、两类各自PA/IoU及两类mIoU，不平均逐图mIoU。
额外记录有效像素数、前景占比及视野覆盖；宽图未推理的两侧不能填背景参加评分。

若要与Cityscapes**原始标注分辨率**比较，另报中心1024x1024 ROI指标：将实际512 mask最近邻映射回该ROI，
与未缩小的GT比较。它与512网格评分分栏报告，仍不代表整幅1024x2048图的精度。

### 2.2 同一logits的方案对照

| 方案 | 输出方式 | 用途 |
|---|---|---|
| A | 原30->240双线性+argmax，再将类别mask最近邻放大至512 | 旧硬件结果的高分辨率评分基线 |
| B | **30->512双线性插值logits，再argmax** | 本轮推荐片上方案 |
| C | 对同一INT8 logits做FP64双线性参考，再argmax | 核查B的坐标/整数实现，不是新模型 |

A/B/C使用完全相同的源INT8、240输入、PARAM和30x30 logits，全部对同一512 GT评分。
禁止对类别ID做双线性插值；也不先生成240 mask再对其做双线性。“logits插值后分类”与“类别图放大”不等价。
默认精度gate：B相对A的PA不降超过0.2pp、mIoU不降超过0.5pp、车辆IoU不降超过1.0pp；
这些是验收阈值，不是已达到的结果。不通过则先复核坐标/GT与同源性，再讨论方案，不能擅自重训或修改基线。

部署golden必须来自实际PARAM整数replay，不从HLS dump反向生成，也不复用研究forward的旧权重舍入行为。
模型侧工作放在`D:/ESPNet/resolution_h240w240_gray/output_restore_1104/`；输入研究目录保持只读。
新manifest分别声明`source_shape`、`roi_xywh`、`encoder_shape`、`logits_shape`、`mask_shape`、
坐标/边界/tie规则、上游预处理、样本/GT/PARAM/hash；不覆盖旧240 golden。

## 3. 输出插值的硬件方案

采用像素中心对齐，即`align_corners=False`的几何关系。对输出坐标d=0..511：

```text
u(d) = (d + 0.5) * 30 / 512 - 0.5 = (30*d - 241) / 512
n = 30*d - 241
n < 0             : i0=i1=0,  w0=512, w1=0
n >= 29*512       : i0=i1=29, w0=512, w1=0
其他              : i0=n>>9, i1=i0+1, w1=n&511, w0=512-w1
```

横纵方向相同。这里9位小数权重对30->512是**精确有理坐标**，不是额外的模型量化；
不在插值中间重新量化、截断或饱和。当前30->240对应`u=(2*d-7)/16`，两配置可共用同一源码算术结构，
编译期选择分母/初值/步长，不同时综合两套kernel。512配置用加法、移位和掩码，不引入通用除法器。

继续使用二分类差值`D=logit1-logit0`：四角差值按双线性权重累加，结果>0输出1，否则输出0，平局选0。
差值范围[-255,255]，权重每轴和为512，最终绝对值不超过255*512*512=66846720；
乘法和加法中间量显式扩展到有符号32位，避免ap_int窄位宽表达式截断。无需除以512^2即可比较符号。

逐行消费规则：维护`next_output_y`，仅当其需要的下邻行`y1`已到达时输出，保留上一行和当前PPU行。
首个logits行输出顶部9行，末行输出剩余26行，中间每对logits行对应17或18行；总计必须正好512行。
不能继续沿用旧“每次8行、首尾4行”的控制。`upsample_fused_begin()`必须重置所有坐标、有效位和pack状态。

横向保留四角缓存和增量权重，避免每个输出像素重复读四次行RAM；内部段宽为17/18，边界段为9，
不能保留旧8像素数组/64-bit段包假设。按32个UINT8标签打包为一个AXI word，单一写回owner顺序输出8192 word。
每行512 bytes对应16 word，帧末无padding。保留小段寄存器即可，不增加512像素整行/整帧输出缓冲。

## 4. 分轮执行计划

### Round 0：冻结样本、离线评分与整数golden

- [x] 在上述模型目录新增`output_restore_1104.py`及对应测试，复用现有输入量化和整数replay。
  从已量化512x1024源图逐行提取中心512x512，生成正方形源；两种存储形式经选样后的57600 bytes必须一致。
  固定模型/PARAM/hash，不把网络输入分辨率改为512，也不更新observer。
- [x] `tools/eval_single_hw_outputs.py`的logits重建入口增加显式`mask_shape`，默认保持旧行为；
  `tools/eval_val_hw_masks_fullres.py`解耦预测/GT尺寸与encoder几何。先写坐标、边界、平局、符号极值、标签映射测试。
  `geometry_contract.py`仍描述240 encoder；不能伪造512的encoder geometry或非整数`target_scale`。
- [x] 同一组500张图分别计算A/B/C的512指标及可选原始ROI指标，记录新口径基线，输出逐图confusion供复核。
  B与C在边界/平局处用精确分数参考裁决，不能用浮点误差静默改变标签。
- [x] 导出少量部署参考：512方形INT8输入262144 bytes、240选样57600 bytes、HWC logits1800 bytes、
  512 mask262144 bytes及512 GT/hash；不复制/重导PARAM。精度gate和同源性通过后才进入Round 1。
  编译器/导出器不重编卷积schedule或改变blob；如需接入新输出manifest，只增加独立mask几何信息，
  原模型geometry仍为240输入/30 logits。

#### Round 0验收：2026-11-08，500张离线实验通过

全部使用实际PARAM解包的权重/qparam和现有exec-plan整数replay；仅对数组级卷积/Pool/Add加速，
不重新量化权重、不更新observer。卷积用FP64精确累加INT8乘积后做整数重量化，不是FP32模型推理。
原生、H256、H512三组已冻结参考分别0/1800 logits mismatch；H512保留此前修正后的PARAM权重舍入语义。

主评分直接读取原始标注，中心1024x1024 ROI半像素最近邻到512 GT，累计500张confusion后计算：

| 方案（同一512 GT） | PA / % | mIoU / % | 车辆IoU / % | 背景IoU / % |
|---|---:|---:|---:|---:|
| A：旧240 mask最近邻放大至512 | 96.7269 | 78.9627 | 61.3780 | 96.5473 |
| **B：30 logits直接插值至512再分类** | **96.7389** | **79.0182** | **61.4765** | **96.5598** |
| C：独立FP64插值参考 | 96.7389 | 79.0182 | 61.4765 | 96.5598 |

B的车辆/背景PA为79.4895%/97.9473%，有效像素113546154；相对A的PA、mIoU、车辆IoU分别
增加0.01195、0.05549、0.09847pp，全部通过预设gate。B/C在500张共131072000个输出像素上0 mismatch，
两种源存储布局选入240 encoder的字节也全部一致。它证明直接logits上采样可行，**不是大幅恢复细节**；
不得将79.0182%与旧240 GT上的80.0396%当作同口径精度下降。

另按实际512 mask最近邻放回原始1024x1024中心ROI，与未缩小标注比较：

| 方案（原始中心ROI GT） | PA / % | mIoU / % | 车辆IoU / % |
|---|---:|---:|---:|
| A | 96.7654 | 79.1627 | 61.7380 |
| B / C | **96.7774** | **79.2189** | **61.8378** |

此口径有效像素454333619，仍只覆盖原宽图面积50%、约40.00%的车辆像素，不是整帧分割。
整轮离线执行约77.1s，不是FPGA推理时间。92项仓库回归和9项研究/产物测试通过；
还完成真实高分GT读取的评估器端到端核对，旧240默认行为保留。
发现并修复Python replay中non-bypass ADD将标量传入tensor-only舍入函数的潜在bug，复用现有NumPy定点算术；
不改变HLS或模型数值契约。

数据位于`output_restore_1104/results_500.json`、`validation_logits.npz`及`selected_reference/RESTORE.json`，
保留500张逐图confusion、样本/GT/hash和全部低分辨率logits。6个参考为`I512S.BIN`、`Q240.BIN`、
`L30.BIN`、`M512.BIN`、`T512.BIN`、`M240.BIN`；源图、512 mask/GT各262144 bytes，logits为HWC的1800 bytes。
16个冻结文件hash未变化，未复制或重导PARAM，未修改HLS/App、未跑CSim/综合或新训练。
新参考中的mode 3/512输出只表示未来接口，**不能交给当前240输出IP直接上板**。
复现与只读核验命令见研究目录README；Round 0门槛已满足，后续源码验证情况见Round 2末尾。

### Round 1：最小HLS扩展

- [x] 修改`ESP_INT8_hls/include/npu_config.hpp`、`src/frame_dma.cpp`及`src/int8_core.cpp`：
  增加source mode 3，区分源pitch与512采样比例；复用现有15像素固定选样、FIFO深度16/4和唯一输入writer。
  更新合法mode检查、输出常量/断言及gmem_frame_out/ap_memory depth=8192，模型RAM大小不变。
- [x] 仅在`src/upsample_unit.cpp`内实现第3节输出器，保持已有begin/consume入口及PPU调用归属。
  240配置走相同编译期结构，原mask须逐字一致；512配置清除8x专用段长度、权重和行发射假设。
- [x] 核查`src/ppu.cpp`、`src/conv_engine.cpp`仍只有原逐行调用，不新增Conv/SA实例或并行consumer，
  不新增反馈DATAFLOW、全帧logits DMA或DDR中间mask。正常top不调用两步mask放大基线A。
- [x] 同步`ESP_INT8_app/src/app_config.h`、`main.c`、`hal/int8_npu.c/.h`的mode 3、输出H/W=512、
  容量262144、cache维护/类别检查/SD保存长度及encoder/输出分别打印；更新App contract测试，
  不要求旧平台运行新配置。平台绑定、build与SD准备仍留在Round 3，不在本轮自动执行。

### Round 2：清理、结构检查与最终CSim

- [x] 修改`tb/frame_dma_tb.cpp`、`tb/top_golden_sample_tb.cpp`、相关`tools/test_*contract.py`及structure checker，
  加入mode 3、输出8192 word与encoder/输出几何独立检查；验证无效mode=4/0xffffffff在DMA前被拒绝。
- [x] 单元测试覆盖：512方图/宽图ROI等价，signed INT8极值，512输出首尾行、17/18段界、全平局、
  正负差值和多帧reset；guard区不得被改写，恰好262144个有效标签、每行512像素、无FIFO残留。
  240编译配置与原参考逐字一致。只做局部单元验证，不逐层重复跑整网。
- [x] 全面静态扫描移除不再可达的8x helper、旧固定输出长度和重复store路径，保留当前DMA输入写入helper。
  最后只开启一次完整top CSim：mode 3同源输入，240选样0/57600 mismatch、logits0/1800、mask0/262144、
  invalid=0、ERROR=0，并保留原debug/guard能力。支持的其他source布局由输入单元及独立golden覆盖。

综合前交付核验：93项Python回归、P7结构检查、14个CPP的综合语法/unused-static检查及App语法检查通过；
输入DMA四种mode、signed极值、真实方/宽图参考及回切native通过，FIFO排空、输入末尾guard不变。
两种输出编译配置各通过7组插值模式、逐行就绪/首尾/段界/平局/中断帧重置/输出guard检查；
240回归与冻结M240为0/57600 mismatch。最后仅运行一次mode 3完整top CSim，得到
`input=0/57600 logits=0/1800 mask=0/262144 invalid=0 last_error=0`，8192个输出word及两端guard通过，
Vitis报告0 errors（含局部检查及编译约2分08秒）。日志和dump位于
`ESP_INT8_hls/hls_work_output_restore_full/solution1/csim/`。
旧8x专用helper/64-bit段包已删除；权重端口限制为Q9/Q4有效位宽，竖向插值使用两次差值乘法，
模型计算owner、PARAM hash和主`hls_config.cfg`未变化。App源码已适配，但现有旧平台不能运行新配置；
Round 1/2未自动执行csynth/实现、build或SD写入；后续手动综合验收见下。

### Round 3：手动综合/实现、App与高分辨率上板

- [x] 用户手动csynth后核查：一套SA=1024 DSP、一套upsample/输出writer、一套输入DMA；
  无新增搜索爆炸或循环复制，RAM绑定未退化，输出轴/标签循环达到实际II=1目标；不能只看pragma。
  未满足功能/结构gate不得用Vivado“可能优化回来”代替确认。
- [x] 通过综合后由用户手动package/100MHz实现；检查合法布线、WNS/WHS均非负，再导出新平台。
  保留旧release，平台/BSP/bitstream及App必须成套，不能只替换SD数据。
- [x] 将Round 1更新的App绑定到新平台，核对`app.yaml`、component、README、boot/launch配置与实际bitstream；
  重新核查输出容量/维护长度均为262144，encoder打印仍为240x240x1；PARAM仍为P2G.BIN。
  示例SD名`I512S.BIN`/`O512S.BIN`，均262144 bytes；
  宽图输入仍为`I512Q.BIN`，新高分输出使用`O512H.BIN`，不能覆盖旧57600 bytes的O512Q基线。
- [x] 用户手动build/准备SD/首张mode 3上板：512 mask与整数golden逐字一致。
- [ ] 连续3次检查reset/时延稳定性，并核对新IP的mode 2与mode 3结果一致。
  记录ARM原始ticks、PL cycles、FRAME_LOAD与PPU_ROW_CONSUME；输出插值仍计入现有PPU行阶段，
  不能把stage 10为0误判成没有上采样。随后经用户确认再准备500张板级验证，按第2节真实GT评分。

综合验收（2026-11-08 23:38）：本轮约11分26秒、峰值3.067GB，无搜索失控或阻断错误。
相对已成功实现的resize基线，LUT 345970->345891（-79）、FF 179669->179365（-304）、
DSP 1269->1268（-1），BRAM18K=661、URAM=28不变，全部满足本轮增量预算。
SA仍为单实例1024 DSP；单一输出重建器合计2068 LUT/1675 FF/4 DSP，无新增BRAM/URAM，
不是额外叠加4个DSP。输出像素循环实际II=1、段宽9..18；输入选样实际II=1、调度15377cycles不变。
估计周期7.300ns（136.99MHz），P7结构/克隆门槛和报告阻断检查通过，可推进package及100MHz实现。
LUT估计仍为器件101%，但低于已实现基线；旧Conv DATAFLOW吞吐告警仍存在，不等于本轮新增错误。
段/行控制与AXI等待尚有开销，不能由像素II=1推导整帧无停顿；<=55ms与合法布线仍须实际验证。

实现与平台核验（2026-11-09）：100MHz下post-route phys-opt WNS=+0.383ns、WHS=+0.010ns，
TNS/THS=0，391722条可绕线net全部完成、routing errors=0，bitgen成功。
`platform_1109`的XSA及bitstream与该实现输出hash一致，新增输出器/输入reader RTL与最新HLS一致；
App component、YAML、CMake均选该平台，boot配方已去掉旧平台路径，tag更新为`20261109-P7-H240-MASK512-B2`。
模式3单图、262144 bytes输入/输出、完整PARAM及全部profiling保持开启；24项App/profiling测试、
新BSP下三个C源文件语法检查通过。未自动build/生成BOOT.BIN，无现成新launch配置；SD未插入且未访问。

### 首轮上板：mode 3正确，512输出整网54.12012ms

2026-11-09，App=`INT8-BOARD-20261109-P7-H240-MASK512-B2`，平台=`platform_1109`。
用户完成build与上板，运行完整PARAM v4/75 UOP/16 EXEC；`I512S.BIN`为262144 bytes，
encoder仍为240x240x1，输出`O512S.BIN`为512x512、262144 bytes。
启动打印的input bytes=524288是最大DDR缓冲容量，实际本次SD读取/DMA源规格为mode 3的262144 bytes。

**实读SD核验通过。** P2G/I512S大小及hash与冻结manifest一致；O512S与
`output_restore_1104/selected_reference/M512.BIN` **0/262144 mismatch**，invalid=0，
全图c0=18853、c1=243291，与UART一致。输出SHA256为
`01d5811c52b921e545fc304758df407ce72bef494e556b2c44ea59075fd7ab02`。
无板级logits dump，不能把mask相同表述为本轮已独立核对1800个板级logits。
SD文件时间仍显示2010年，属于板端文件系统时钟未校准，不用mtime作为本轮输出身份依据。

样本`frankfurt_000000_000294`：重新读取原始GT PNG、核对hash，取原图中心1024x1024 ROI，
半像素最近邻到512后映射车辆=0、背景=1、255忽略，与冻结T512 **0 mismatch**。
主评分含229900个有效像素，忽略32244像素；以下均为**同一张图、同一512 GT**，不是500张验证集成绩。

| 输出方式 | PA / % | mIoU / % | 车辆PA / % | 车辆IoU / % | 背景PA / % | 背景IoU / % |
|---|---:|---:|---:|---:|---:|---:|
| 旧240 mask最近邻放大至512（A） | 97.6446 | 86.1459 | 85.8851 | 74.8245 | 98.6882 | 97.4673 |
| **本轮板级512 mask（B）** | **97.6581** | **86.2129** | **85.9384** | **74.9442** | **98.6981** | **97.4817** |

当前B的confusion（行=GT、列=预测）为`[[16104,2635],[2749,208412]]`。
相对A的PA/mIoU/车辆IoU增加0.01348/0.06702/0.11966pp，无精度下降。
另将实际512 mask最近邻映回原始1024x1024 ROI，与未缩小标注比较：
PA=97.7178%、mIoU=86.5168%、车辆IoU=75.4884%，有效像素919728。
它仍不是整幅1024x2048宽图评分，也不能将本图结果与Round 0的500张汇总指标直接作增减比较。

| Stage | 本轮cycles | 本轮ms | 占比 / % | 相对旧resize基线cycles |
|---|---:|---:|---:|---:|
| FRAME_LOAD | 16373 | 0.16373 | 0.30 | -151 |
| Conv row datapath | 3418908 | 34.18908 | 63.17 | 0 |
| PPU row（含最终输出插值/打包/写出） | 713320 | 7.13320 | 13.18 | +424206 |
| BLOCK5 final | 658246 | 6.58246 | 12.16 | 0 |
| Vec | 433904 | 4.33904 | 8.02 | 0 |
| Pool | 135617 | 1.35617 | 2.51 | 0 |
| IDLE/MainCtrl/权重加载合计 | 35644 | 0.35644 | 0.66 | -6 |
| **TOTAL** | **5412012** | **54.12012** | **100.00** | **+424049** |

13个stage之和与total/window完全相同，ERROR=0。WIN/SA/POST各自求和均为3418908，
各状态计数与旧基线全部一致，不能把这三组重叠计数相加。输出插值仍归入PPU row，stage 10/11为0不代表漏算。
ARM原始`1803995 / 33333000`为54.120391204ms，比PL仅多0.271us；UART的54ms为整数舍入。
MODE_INIT、SD I/O、PS预处理、cache维护及打印不在此推理窗口内，不能称为输入原始图片到保存mask的系统端到端延时。

对旧mode 2/240输出的49.87963ms，本轮总增4.24049ms（8.5014%）；增量几乎全部来自输出消费。
两轮输入mode不同（旧宽图、新方图），但ROI选入encoder的字节一致；严格同mode 2的性能A/B仍待补测。
首轮**<=55ms通过**，余量87988cycles=0.87988ms；输出stage增424206cycles，
**超过400000软预算24206cycles=0.24206ms（6.05%）**，不得写成所有性能门槛均通过。
像素循环II=1不等于整帧每拍一像素：仍有段/行切换、pack提交和AXI等待；现有计数不能继续精确拆分这些开销。

下一步先冻结已正确的数值/输入路径，做3次重复与mode 2/3切换，比较同一M512并记录计数；
随后由用户确认是否准备500张板级测试。暂不因24206cycles的小幅超预算返修整个架构。
若继续追求<50ms，当前至少还需省412012cycles，不能通过缩小SA、降低精度或恢复旧240 mask来达成；
应另按输出器局部控制开销/Conv数据流制定优化，不把本次功能验收当作已达成<50ms。

## 5. 资源、性能与最终gate

输出从57600增至262144 bytes，为原来的4.55倍；不代表整网计算扩大4.55倍。encoder计算量不变，
增加的204544 bytes对应6392个AXI word，在256-bit/100MHz每拍一word的理想条件下仅写出下界增63.92us。
若插值算术每拍一个像素，额外像素服务时间约2.04544ms；实际还包含段切换、pack、AXI和PPU背压，
**不能把DDR下界当总开销，也不承诺新版本仍<50ms**。

本轮工程目标：固定同源mode 2比较，除FRAME_LOAD和包含最终上采样的PPU行阶段外，其余stage周期不退化；
输出消费相对旧版增量争取<=400000 cycles，
整网先争取<=55ms@100MHz。功能正确后再依据报告和板级计数判断，不为维持50ms删counter或降低精度。
建议资源审查预算为相对resize基线LUT增<=1500、FF增<=2000、DSP增<=8，BRAM/URAM不新增；
这是设计预算而非已验证结果，超出先分析热点再由用户确认，不擅自缩小阵列或改模型。

最终通过需同时满足：高分辨率离线精度gate；真实PARAM整数golden和HLS/板级mask逐字一致；
512输出完整/连续帧无越界及ERROR；没有低/高输出路径并存导致的datapath复制；
100MHz合法实现；App/SD/平台尺寸一致；报告明确区分512网格、原始中心ROI与旧240网格的指标。
**本轮不以“512输出像素更多”宣称模型细节恢复或精度提升，必须由真实高分辨率GT实验确认。**
