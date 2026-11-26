# workplan_1116：P7 Unified INT8 多模式统一硬件

## 0. 目标与发布起点

目标是**一份 bitstream、一个 NPU IP、一个 App，通过运行时配置切换模型和输入/输出路径**，而不是把几个现有 IP 并排放置，也不是运行前更换编译宏。
支持 RGB/gray、H512W1024/H512W512/H256W512/H240W240，以及原生卷积和中心裁剪/下采样至 240 的轻量路径；继续支持二分类和 20-class。
模型仍为 ESPNet p=q=1、ReLU、对称 W8A8。训练权重、量化参数和 PARAM 随模型加载，不能通过改输入尺寸把同一模型视为各分辨率均已适配。

开发/最终发布分支选择 **`feature/p7-unified-int8`**，从 `feature/h512w512-gray-int8@a103115` 创建。
起点对应 `H512W512-GRAY-INT8-BASELINE-STABLE-20261116`，保留所有原稳定分支/tag，不移动或覆盖历史发布点。
选择原因：该版本具有最新原生 512 灰度、PARAM v4 综合修复和 QAT10 板级闭环，计算核心又继承 H240 R8；比从旧 master 重做后续性能修复更合适。
本轮采用**按模块移植稳定差异**，不直接合并完整分支历史。按以下门槛逐轮实施；综合/实现与 App build 均由用户手动执行。

## 1. 已核对的全部分支及稳定能力

已刷新远程引用；下表使用分支最后稳定提交，不采用中间失败实验或旧 tag 替代最终状态。

| 分支 | 稳定提交 | 可继承能力 | 板级 MODE_RUN 参考 |
|---|---|---|---|
| `master` | `bdb83c6` | 原生 512x1024 RGB，二分类/20-class；大尺寸重叠存储及保护逻辑 | 二分类 512.29815 ms；20-class 513.93493 ms |
| `feature/p7-int4` | `bdb83c6` | 与 master 指向同一提交；**不是另一份成功的 INT4 实现** | 同上 |
| `feature/h256w512` | `a9c5a43` | 原生 256x512 RGB，二分类/20-class；改进后的 C3/C19/C131 WinGen、RGB AvgPool | 约 120.46315/120.64715 ms |
| `feature/h240w240-gray-int8` | `9516a54` | 原生 240x240 gray 二分类，R8 稳定计算通路 | 49.85440 ms |
| `feature/h240-resize` | `f40a820` | gray 宽图中心方形裁剪、半像素最近邻至 240；240 mask | 512x1024 源图 49.87963 ms |
| `feature/h240-mask512` | `ab47bd4` | 增加 512x512 源图；30x30 logits 直接重建 512 mask | 54.12012 ms |
| `feature/h512w512-gray-int8` | `a103115` | 原生 512x512 gray，64x64 logits；QAT10 新模型 | 222.28515 ms |

这些数值来自不同阶段的独立硬件，不是统一版的性能承诺。功能依据还包括各分支的 `workplan_0803/0817/0910/1011/1023/1104/1110`、HLS 源码、导出器和实现报告。
合并前 H512 单图 mask 与整数 golden 为 0/262144 mismatch；PA=98.9952%、mIoU=93.6241%。当时的 500 张软件整数参考 mIoU=86.9902%；统一版实际 500 张板级结果见第8.1节，不能与单图指标混用。

### 1.1 不能直接拼接的部分

- `npu_config.hpp` 的输入尺寸、通道、logits 尺寸、类别容量、存储分区，以及部分循环边界仍为编译期常量；App 改文件名不能使它们运行时可变。
- `param_dma.cpp` 只接受当前编译几何；`memory.cpp/scratch_mgr.cpp` 还有 PARAM 外的隐式地址，不能只改 tensor descriptor。
- 灰度版本删除了 RGB 的 C3/C19/C131 调度路径和 C3 AvgPool，且将类别容量缩为 2；必须恢复真实运算能力，不是仅恢复 enum。
- resize 分支的 `src_mode` 已可运行时选择源图，但 encoder 仍固定 240，512 mask 仍通过编译宏选择；尚非完整的运行时多模式硬件。
- 灰度版本还移除了旧大图存储的部分覆盖保护。恢复 master 的重叠地址时，必须连同实际依赖的保护一起审计，不能误当 dead logic。

## 2. 最终功能与运行时契约

### 2.1 模式矩阵

以下各行均提供 C=1 gray 和 C=3 RGB 两种硬件能力；每次运行只选择其中一种，不在 PL 中把量化 RGB 平均成 gray。

| 片外输入 HxW | native 路径的 encoder / mask | lite 路径的裁剪 / encoder | lite 可选 mask |
|---|---|---|---|
| 512x1024 | 512x1024 / 512x1024 | 中心 512x512 / 240x240 | 240x240 或 ROI 512x512 |
| 512x512 | 512x512 / 512x512 | 完整正方形 / 240x240 | 240x240 或 ROI 512x512 |
| 256x512 | 256x512 / 256x512 | 中心 256x256 / 240x240 | 240x240 或 ROI 256x256 |
| 240x240 | 240x240 / 240x240 | 无需另建计算模式，直接使用原生 240 | 240x240 |

宽图 lite 输出只覆盖中心 ROI，**不能拉伸为完整宽图 mask 后声称保留原视野**。App 和 manifest 必须给出 ROI 坐标，评估时使用相同区域的 GT。
256x256 ROI mask 和 RGB lite 是本次新增能力，不能标注为已有上板验证。暂不扩展到任意尺寸、RGB/gray 自动转换、原始 PNG/JPEG 解码或自动检测输入尺寸。

输入继续使用离线预处理、归一化及量化后的 **NHWC INT8 INPUTQ**。native 不做输入 resize；lite 仅对已量化的同通道图像做中心裁剪与半像素最近邻选样。
输入 scale、通道顺序和预处理必须属于所加载模型。相同原图经过不同模型量化，不保证 INPUTQ 可互换。
lite 的方形边长 S=min(H,W)，ROI 起点为 ((H-S)/2,(W-S)/2)，输出轴坐标 d 对应 ROI 内 floor((d+0.5)*S/240)，再夹取到 [0,S-1]。
输出插值按半像素坐标 (d+0.5)*L/O-0.5，分别处理两轴和边界；固定点系数需由共享契约确定，不能让 C++ 与 Python 各自浮点取整。
输出保持“插值 logits 后分类”，不改成对低分辨率 argmax mask 最近邻放大；输出 uint8 类别号，支持 2/20 类。

### 2.2 有限配置，不做任意网络形状解释器

- 增加 `model_profile_id`：枚举上述四种 encoder 几何乘 C1/C3，共 8 种。类别数由 PARAM 的 logits descriptor 校验为 2 或 20，不复制模型专用硬件。
- 恢复 `src_mode`：0 为 native，源图必须与已加载 encoder 一致；1/2/3 沿用旧 gray 的 256x512/512x1024/512x512 到 240；4/5/6 对应 RGB 的相同 lite 源图。lite 只允许同通道的 240 encoder。
- 增加 `mask_mode`：encoder 分辨率或源图中心 ROI 分辨率；native 只允许原生输出。输入与输出几何分别校验，禁止把源图尺寸误当中间层尺寸。
- 增加可读 `runtime_abi_id/capability/status`。固定 profile 表包含几何、存储布局、有效容量和输出坐标方案；HLS/App/Python 必须由同一契约生成或做自动一致性校验。
- `MODE_INIT` 先清除 ready，再锁存 model profile，校验 PARAM v4、输入/logits、类别、schedule、tensor 地址及容量。失败不得继续沿用上一模型的 ready 状态。
- `MODE_RUN` 锁存 source/mask 配置，先验证再读 DDR；更换 encoder、通道或模型必须重新 INIT，运行中修改寄存器不生效。错误状态不能因 ap_done 到来被 App 判为成功。
- 保留 PARAM v4 的记录布局及既有算术，当前模型仍预期 75 UOP/16 EXEC。不重启 PARAM v5/U71 融合或 R5A 常驻任务流；旧 v4 文件也必须重新编译并验证布局兼容，不能只凭版本号放行。

## 3. 全轮次约束与首要风险

### 3.1 存储可行性必须先于硬件改造

当前 H512 实现已占 **108/112 URAM**。当前导出器布局直接扩大后的结果如下，不能把灰度 512 的成功简单外推到最大模式。
URAM 数为该布局 prefix 按 256-bit 有效字宽估算的下界，尚未计入 banking/深度取整；其他缓存也需另算。

| encoder HxWxC | 当前布局 FMBUF bytes | URAM prefix bytes | 估算 URAM 下界 |
|---|---:|---:|---:|
| 240x240x1 / x3 | 0xCF080 / 0xE5180 | 0xC8000 / 0xD0000 | 25 / 26 |
| 256x512x1 / x3 | 0x1C0000 / 0x1F0000 | 0x1B0000 / 0x1C0000 | 54 / 56 |
| 512x512x1 / x3 | 0x378000 / 0x3E0000 | 0x358000 / 0x380000 | 107 / 112 |
| 512x1024x1 / x3 | 0x6E8000 / 0x7B8000 | 0x6A8000 / 0x6F8000 | **213 / 223，超限** |

master 已在旧架构布局下支持最大 RGB：FMBUF=0x598000，URAM prefix=0x380000，其余 0x218000 由 BRAM 承担。
其可行性来自生命周期重叠，并非有更多片上 RAM。其中 B2 残差覆盖前的 64 行备份复用了空闲 scratch，不能只拷地址、删除备份保护。
统一版应采用**一个物理 FMBUF，按 profile 选择经过证明的地址/生命周期方案**，而不是把 8 份内存相加，或把所有独立缓冲都扩大到各自最大值。
如更改某区域的 URAM/BRAM 归属，必须检查所有专用读写路径，不能假定修改通用 accessor 后 URAM-only fast path 自动正确。

### 3.2 不可破坏的结构

- 保持 32x32 MAC 阵列、既有 K 分块和 POST16 数值语义，不以缩阵列、提高热循环 II 或新增 DDR 中间特征图换取合并成功。
- 顶层单 Conv gateway，Conv 内单 SA 调用及既有行级 DATAFLOW；WinGen、PPU、Vec、Pool、输出写回各自保留单一 owner。允许少量 mode adapter，禁止按 profile 实例化整套 engine。
- WinGen 仍只通过 `scheduled_window_generator_row()` 执行 PARAM schedule。恢复 RGB mode 不等于恢复运行时依据 in_c/kernel/dilation 猜网络的旧 dispatch。
- 不新增 C++ 模块文件，不恢复独立 BLOCK5 finalizer、旧 concat engine 或大 task-sequence mux；不靠 ALLOCATION/Tcl 限制弥补源码中的多调用入口。
- 配置使用小型 POD 字段，入口锁存后下发本阶段需要的值。禁止含转换运算符的控制对象、跨动态对象指针、巨型结构穿过 DATAFLOW，避免复现 disaggregate/未知指针和调度搜索问题。
- 保留现有 stage/conv profiling 接口及计数口径；WIN/SA/POST 是重叠且含等待的窗口，不能相加解释总时延。
- 每帧重置缓存有效位、行消费状态、输出 pack/DDA、scratch 保护状态和错误标志；不为小模式逐帧清零最大 FMBUF。

### 3.3 最大容量与边界

最大源图 512x1024x3：输入 1572864 bytes / 49152 AXI words；最大 mask 512x1024：524288 bytes / 16384 words；PARAM 保持 262144 bytes / 8192 words。
源图最大宽 1024 不等于所有内部行缓冲都需 1024；卷积最大输出行宽为 512，紧凑 C131 行为 256x131=33536 bytes，即 **1048 words**，logits 最大宽为 128。
`conv_store.cpp` 当前 10-bit `last_word`、Vec 的 10-bit 行提交计数等必须按 1048-word 边界重审，不能出现 1023 截断。
240xC1 的行长 240、240xC3 的行长 720 都不是 32-byte 对齐；保留跨行余字、phase=16 和局部补零逻辑，不能沿用原生 512 的整行对齐假设。
采用按阶段计算的容量上限及静态断言，分别审计 AXI depth、紧凑行、tile 数、packed C1/C3 cache 和计数位宽，不全局盲目放大 typedef。

## 4. Round 0：冻结契约并证明最大模式存储可行

**改动范围：**先改 `tools/geometry_contract.py`、`export_int8_hw_blob.py` 及必要的现有 contract test；不先改 HLS 通路。

1. 建立唯一 profile 表和布局版本，覆盖 8 种 encoder 几何、2/20 类、native/lite 源图、mask/ROI 语义；模型清单与硬件 capability 分开。复用现有导出器，不再新增一套编译器或 replay。
2. 审计 `build_memory_layout()` 与 master 的 `memory.cpp/scratch_mgr.cpp/vec_alu_engine.cpp`。对 tensor、残差、BLOCK5 scratch、池化临时区、紧凑行和 B2 备份建立逐 EXEC/逐行生命周期及实际读写覆盖检查。
3. 设计每个 profile 的有限地址表，共享同一物理 RAM。优先借鉴已成功的大图重叠布局，将备份复用纳入证明；同时检查主 RAM、WBUF、行缓存、端口/banking 取整后的 BRAM/URAM 总量及保护搬运开销。
4. 在 exporter/replay 中先落实同一 profile 地址/隐式 workspace 规则，为后续 CSim 提供可生成的新布局 PARAM 及整数参考；不等到 App 阶段才修编译器。确认切 profile 后上一帧内容不会在有效位未更新时被读取。冻结旧模型 checkpoint/qparam/input/golden hash，不重新训练掩盖硬件问题。

### 4.1 统一布局契约与后续 HLS 接入边界

Python 契约位于 `geometry_contract.py`，`runtime_abi_id=0x11160001`，layout=`p7-unified-layout-v1`。
profile 0/1=240 gray/RGB，2/3=256x512 gray/RGB，4/5=512x512 gray/RGB，6/7=512x1024 gray/RGB；source/mask ID 沿用第2节。
原生 INPUTQ 仍紧凑连续；统一布局中的中间 tensor 和 Pool 每行显式补齐32字节，覆盖 RGB240 的 C19/C131、C3 Pool 非对齐情况。

| 项目 | 统一布局约定 |
|---|---|
| 物理 FMBUF | 固定 `0x530000`，即5.1875 MiB，所有 profile 共用 |
| URAM / BRAM 分界 | 固定 `0x380000`；BRAM 尾部 `0x1B0000` |
| L20 C1及分支 workspace | 复用已结束生命周期的输入区；B1 tensor 从 INPUTQ 末尾开始，避免输入仍活跃时覆盖 |
| L20/L30 residual | 各自按阶段安排，不能继续把 B1/L20/L30 全绑定到一个旧 residual_base |
| L2B0/L3 C1 scratch | 位于 Pool2 有效区域之后，按阶段复用；WinGen/store 必须允许这些地址处于 BRAM |
| B2残差保护 | 复用已结束的高位 C1/workspace，32或64行的2次幂槽；逐行验证复制先于覆盖、槽位释放先于复用 |
| BLOCK5 row group | 上限32行，仍使用原有行级融合PPU；新PARAM仅改group大小，不能把scratch按旧64行再次展开 |

最大RGB候选的Pool1=`0x418000`、L2B0 C1/backup=`0x430000`、L2B0 branch workspace=`0x4B0000`。
**不要直接沿用master的`0x3E0000` Pool基址**：在新布局中它会与仍活跃的L20区域冲突，负向测试必须拒绝。
最大模式B2备份峰值63行，容量64行；gray/RGB的额外备份读写分别129024/124928 words，约1.290/1.249ms的II=1访存工作量。
32行分组相对64行最多增加3072个branch weight words；这些只是新增工作量，未包括cache warmup/控制，不构成总性能不退化的证明。

固定RAM的映射预算为112 URAM、864 BRAM18K；另外保留原生512**整机**实测457 BRAM18K及128块新adapter预留，合计估算1449/1488。
该计算重复保留了旧Pool RAM，属于带余量的预综合预算，但128块预留尚未经综合验证，不能把1449写成已测资源或保证后续一定可布线。
若实际adapter、端口映射或逻辑复制突破预算，仍应停在资源门槛，不得据此自动批准实现。

`export_int8_hw_blob.py --audit-runtime-memory` 为只读检查入口，覆盖8几何乘2类别的存储/逆序行保护；不生成临时报告或产物。
`recompile_runtime_param()` 从冻结PARAM构造新布局，仅改tensor、group及window reserved[9]的传输协议标注位；权重、qparam及数值调度字段不变，不支持改变模型几何或类别。
现存四组有效PARAM/INPUTQ/整数logits的SHA固定在 `test_unified_runtime_contract.py`；H256两组缺少现存fullres mask文件，后续须从冻结logits按统一规则重建。
统一HLS、默认导出、App与默认TB已使用上述契约；旧产物只作冻结来源，不能直接用于新IP。
实际RAM映射、clone数量、II与时序仍须用户手动综合后核对，源码与CSim通过不等于实现通过。

**Gate：**最大 RGB/gray 的地址、生命周期和端口冲突检查通过；预计总量不超过 112 URAM、1488 BRAM18K、3528 DSP，且保护逻辑有可接受的周期预算。
master 的资源成功仅是可行性参考，不代替新布局证明。若不通过，停在本轮重新设计存储，不能静默删掉最大 native 模式、增加 DDR spill 或改成多 bitstream 发布。
本轮输出契约、可复现测试和简要结论即可，不新增大量一次性脚本、临时报告或逐项实施日志。

## 5. Round 1：统一原生运行时及 RGB/gray 计算通路

**主要文件：**`npu_config.hpp/npu_types.hpp/npu_ctrl.hpp/npu_schedule.hpp`，`int8_core.cpp/param_dma.cpp/memory.cpp/scratch_mgr.cpp`，
`win_gen.cpp/avgpool_unit.cpp/conv_store.cpp/ppu.cpp/vec_alu_engine.cpp`；仅在确有接口需要时改 `main_ctrl.cpp/conv_engine.cpp`。

1. 将几何拆成“综合容量上限”和“锁存的活动 profile”，落实 INIT/RUN 协议、错误返回和容量校验。`param_dma` 按 descriptor 和 profile 精确校验布局，不恢复旧 cfg-based 调度或多套 PARAM parser。
2. 接入 Round 0 的地址表，统一专用/通用访存及 scratch getter，恢复确有生命周期依赖的备份保护。所有模型共享同一静态数组 owner；不通过任意 pointer cast 选择不同存储对象。
   分别接入 L20/L2B0/L30/L3B0 的 C1/branch 基址，不恢复统一旧scratch地址的假设；group按PARAM的32行推进、halo按绝对行坐标取数，尾组不足32行时限界。
3. 从 `a9c5a43` 移植 C3/C19/C131 schedule 执行和 C3 inner-fast AvgPool，保留当前 C1/C17/C129 及公共 C12/C25/C28/C64/C128 路径；同步恢复 exporter 的 RGB schedule/store plan 生成及 replay 支持。相同缓存和 loader 合并，禁止 C1/C3 各带一套完整 WinGen+SA。
4. 恢复最多 32 通道的分类容量，实际运算只处理有效 2/20 类。校正 1048-word 提交边界及所有活动行计数；从 `ab47bd4` 恢复 240 非对齐输入行所需处理，不回退 H512 计算优化。

**Gate：**8 种原生 profile 的编译/地址/边界测试通过，RGB 和 gray 的 WinGen token 顺序、AvgPool 舍入/饱和与各自整数参考一致；错误 PARAM/profile 在 DMA 前拒绝。
静态调用图仍为单 Conv、单 SA、单存储 owner，没有按分辨率复制整套计算通路。此时不要求每个小步骤跑 fullres CSim。

RGB 合并检查曾定位到 C19 输出错误：原灰度 compact packer 仅处理1/4字节尾段，恢复C19时漏掉3字节尾段。
现由公共packer按“已有余字节+本次字节数”更新group与carry，不增加独立RGB写回路径；C12/16/17/19/25/28须一并回归。

## 6. Round 2：统一输入裁剪和多尺寸/多类别 mask 输出

**主要文件：**`frame_dma.cpp/upsample_unit.cpp/ppu.cpp/int8_core.cpp`、对应 header 及 `frame_dma_tb.cpp/ppu_tb.cpp/top_golden_sample_tb.cpp`。

1. 移植 `ab47bd4` 的有界 read/select-pack/write 输入流水：native 直接读；lite 读取中心 ROI 所需行，用半像素最近邻 DDA 选样到 240。保留有限 FIFO 和背压传播，不引入整帧输入缓存。
2. 保留 gray 高效选样；为 RGB 实现同像素三通道一致选择及连续字打包，处理 AXI word/行边界余字。不能把旧 gray byte selector 当作 RGB pixel selector，也不在逐像素热循环增加通用除法。
3. 将当前编译期 mask 尺寸改为有限运行时方案。输出 owner 共用行缓存、坐标发生器和 DDR pack/write；binary 保留差分插值/符号判断快路径，20-class 恢复分组插值后 argmax，不复制整套上采样/写回 engine。
4. 分别处理 X/Y 坐标：原生矩形固定 8 倍、30->240、30->256、30->512。固定点位宽、边界夹取及 tie-break 保持一致；二分类同分归 0，多分类取最低类别号。不产生完整高分辨率 logits 中间帧。

**Gate：**源选择、ROI、通道顺序、跨字打包和输出坐标与独立参考逐字一致；最大/最小源图及 240 非对齐边界覆盖。
对新输入/输出有限 FIFO 给出有界 token/背压安全检查，尤其连续两帧与模式切换；必要的局部 RTL 验证须先与用户确认，不自动启动综合。不能以 CSim 无界 stream 成功代替有限 FIFO 不死锁的证据。
本轮结束做**第一个完整 CSim 检查点**：最大 RGB 20-class、原生 gray512，以及 gray/RGB lite 的关键跨行与 ROI 输出组合。只做这组必要代表用例，不每改一个 helper 就重跑全网。

测试入口为 `tools/test_unified_runtime_contract.py`，常规运行只做契约和真实C++组件检查；设置
`ESP_INT8_FULL_RUNTIME_CASES`后，使用vendor `ap_int/hls::stream`和整套HLS源码进行host CSim，不启动csynth。
0/1/2/3为冻结gray240、RGB256二分类/20-class、gray512；4为相同gray240输入经合成512源图选择后输出512 mask。
5/6是最大RGB20与RGB lite的**结构测试夹具**：保留已接受RGB20权重/qparam，只重编几何并与独立整数replay比较，不作为已训练模型发布或精度结论。
host CSim可能打印49152的stream峰值，这是DATAFLOW在C中顺序执行产生的无界stream深度，不是硬件FIFO需求；输入FIFO仍为16/4，不能据此扩成整帧缓存。

## 7. Round 3：编译产物、评估和 App 运行时选择闭环

**主要文件：**`tools/geometry_contract.py/export_int8_hw_blob.py/hw_param_replay.py/export_val_hw_dataset.py/eval_single_hw_outputs.py`，
`ESP_INT8_app/src/main.c/app_config.h/hal/int8_npu.[ch]`，现有 TB/contract test 和两侧 README。模型侧必要导出变更仍放 `D:/ESPNet`，不复制训练工具到此仓库。

1. 按统一布局重导出可发布模型的 PARAM/INPUTQ/golden；冻结已接受的权重及 qparam。manifest 记录 runtime ABI、profile、layout、类别映射、量化/预处理、ROI、mask 插值和文件 hash；同为 v4 不意味着旧地址自动兼容。
2. replay 从 manifest/descriptor 获取活动几何和 workspace，不再依赖当前 H512 全局常量。独立 golden 与 replay 同时检查 logits 和 mask；GT 变换按明确的半像素/裁剪规则执行，不能用普通 `INTER_NEAREST` 名称猜测等价，也不能从文件大小猜模式。
3. App 改成一个 ELF 的 `list/run/suite` 运行时入口，可通过 UART 选择已登记模型、source/mask 模式及 SD 文件；串口命令只接受有效组合。固定分配最大 DDR buffer，但加载、cache flush/invalidate 和保存长度均按活动 profile，禁止按最大长度越界读写。
4. INIT 及 RUN 超时、ERROR stage、输出长度/类别错误均拒绝计时结果；打印 encoder/source/ROI/output 和参数身份。保留 ARM 实读时钟及 RTL profiling，不手写猜测新的 AXI-Lite offset，使用新 IP 生成的接口定义。
5. 提供一次运行完整模式矩阵的 suite：每次更换模型执行 INIT，重复帧及 A->B->A 验证真实状态切换。SD 保持 8.3 文件名，通过清单绑定文件身份，不要求重新 build App 切模式。

**模型可用性边界：**现有可用方向为 RGB512x1024 二分类/20-class、RGB256x512 二分类/20-class、gray240 二分类、gray512 QAT10 二分类；必须先核实原始 checkpoint/量化参数是否仍可重现。
当前本地保留产物以 H256 RGB、H240 gray 和新 H512 gray 为主；Git 不保存已忽略的全部大模型/产物，历史板测记录不等于当前文件仍在。
RGB240、RGB512 方形及其他 gray 几何/gray20 的**适配模型不应假定已经存在**。无模型时可用结构/整数测试证明硬件接口，但不能声称精度已通过；实际发布模型清单另列缺项，不自动训练或直接迁移 H240 权重冒充原生512适配。

**Gate：**对所有拟发布模型做**第二个、最终完整 CSim 检查点**，同进程覆盖通道、几何、类别和 native/lite/output 切换；新 PARAM 的 logits/mask 对整数 golden 均为 0 mismatch，非法组合稳定拒绝，正常 ERROR=0。
所有代码改完再统一扫描 dead/legacy logic；只清除确无顶层调用或协议依赖的旧路径，不删 RGB adapter、DMA 输入写 helper、scratch 覆盖保护或仍需使用的验证接口。
通过后交用户手动 csynth，不自动启动综合或实现。报告核对实际 clone count、RAM 归属/容量、II、数组端口及 profiling 输出是否完整，不能只看 pragma 存在。

**综合前交付边界（2026-11-17）：**发布目录为 `hw_artifacts/unified_int8`，`RUNS.CSV`登记4个冻结native模型与6个真实gray lite/ROI输出用例；`MODELS.JSN`明确其余缺失模型，不以结构夹具冒充已训练模型。
默认TB同进程运行10个用例并返回首模型，共11次，logits与mask均0 mismatch；最大RGB20和RGB lite另以结构夹具完成全网整数对齐，不作精度结论。
App为单ELF的UART `list/run/suite`，检查文件CRC、ABI/status/capability、计数闭合、类别及mask身份，超时后禁止继续交易；主程序、HAL与parser的34项host测试通过。常规回归178项通过，另2项full-network入口单独执行；14个HLS CPP的host语法及引用/owner扫描通过，未保留无引用helper或ALLOCATION限制。
配套说明见两侧README；App旧平台路径只作占位，必须在新IP实现并导出统一XSA后重选platform/BSP再手动build，不能沿用旧ELF或BOOT。未自动启动csynth/实现/App build；RTL有限FIFO调度和物理资源仍属于后续报告/板级门槛。

### 7.1 首次统一版csynth检查（2026-11-17）

15:54完成的默认顶层综合与当前源码/统一接口一致，耗时约11分42秒；无HLS ERROR、无214-366函数签名复制警告，46项流水结果均为II=1。
顶层XML资源：LUT=375190（109.94%）、FF=203096、DSP=1349、BRAM18K=1432/1488、URAM=112/112；主FMBUF为112 URAM加770 BRAM18K，未复制整帧RAM。顶层总BRAM低于1449的预估，但只剩56块，URAM无余量。
沿生成RTL层级核对，Conv/WinGen/SA/POST/PPU consumer/Vec/Pool/输出上采样均为1个实例；SA单体1024 DSP，阵列仍为32x32。WinGen=53075 LUT、POST=22462、PPU consumer=42107、BLOCK5 finalizer=21267、Vec=35894，是后续物理失败时的定位起点，不把父/子资源重复相加。
估计周期7.300ns，clock uncertainty=2.70ns，恰好用满10ns的HLS预算，**不是已获得2.7ns物理布线裕度**；不采用顶层极大max latency估计推理延时。
输入DATAFLOW仍有2条214-113及1条200-471，位置是传参时的`channels==3`/`side==512`表达式；RTL保留read/select-pack/write及有限FIFO，未退化成整帧缓存或串行大函数。另有200-1449/1450吞吐警告，不能据此宣称已通过RTL背压/死锁验证。
**决策：允许一次受控实现可行性尝试，不直接认定可发布。** HLS LUT比原生512估计343155增长约9.34%，但历史上HLS估计与Vivado物理用量有差异，不仅凭109.94%就强制回退。先看Vivado综合实际资源，再核对place后的CLB/control sets、拥塞及setup/hold；若物理超限或拥塞失败，按热点/存储端口定位修复，不缩SA、不降低热循环II、不新增DDR中间图。

### 7.2 首次实现失败与修复轮（2026-11-18）

本节保留试验依据；四bank路线已因下述第二次实现失败撤销，当前修复要求以7.3为准，不再执行四bank gate。

11月17日实现失败：53条信号未布通、29处node overlap；place后LUT=248507（72.82%）、FF=191865、BRAM=592/744、URAM=112/112，但CLB=42046/42660（98.56%）、control sets=2087。place WNS=+0.509ns，route日志WNS降至-2.483ns、TNS=-2229.642ns；不能因LUT低于容量就判定可实现。
失败DCP的只读分析确认，最差路径从主FMBUF URAM到PPU字合并/BLOCK5读取寄存器，11～15级逻辑，最差已提取路径slack=-2.475ns；前10个overlap热点中9个涉及该URAM。未合法布线的延迟仅用于定位，不作为完成实现的Fmax。生成RTL将114688x256主存映射为单个深RAM，`cascade_height=1`，深度方向的输出选择和跨模块扇出是本轮优先修复对象，不是SA副本问题。

1. **存储局部化：**仅在`memory.cpp`将URAM物理存储改为4个28672-word浅bank，按`bank=word_index&3`、`row=word_index>>2`交织，bank维完整分区，各bank保持2P URAM、关闭整帧reset。逻辑FMBUF仍为`0x530000`、URAM分界仍为`0x380000`，总URAM理论仍为112；所有通用/专用读写及input DMA走同一映射，不新增旁路或DDR中间图，不修改PARAM/manifest/布局ABI。
2. **减少无效控制容量：**按既有ESPNet p=q=1契约，将tensor/conv/affine/add/pool/consumer/fixed/EXEC缓存上限收紧至19/26/7/14/3/26/10/16，Python导出器同步限制fixed/EXEC。10包含v4对齐槽；8条BLOCK5物理记录属于既有ABI，保持不动。alias改用每帧清除的有效位，移除64-bit epoch和tag比较；保留别名、B2覆盖备份及全部profile保护。此项减少无效RAM/选择器及控制扇出，不删运行能力。
3. **验证与下一门槛：**先测全URAM地址的bank双射、URAM/BRAM跨界span、逻辑bank别名和跨帧失效，再做发布用例同进程全网host CSim及最大RGB20/RGB lite结构夹具；logits/mask必须0 mismatch。完成引用/owner/deadlogic检查后交用户手动csynth，不自动综合。新报告必须确认4个浅URAM bank合计112、无额外整帧RAM、无Conv/SA复制、热循环II保持1，并对照本轮LUT/FF/BRAM及关键调用延迟确认未回退；旧`on_chip_memory_read_main_uram_word`为latency=1、interval=1，重点核对，不能只看MAC II。银行化只针对已定位热点，是否真正改善route/setup仍须后续实现验证，不提前宣称闭环。

本轮交付验证：180项常规回归通过；默认TB同进程11次发布用例、最大RGB20与RGB lite两项全网结构用例的logits/mask均0 mismatch。全地址映射及alias失效回归、source owner和静态引用检查通过；未运行csynth/实现，未修改冻结产物、App寄存器协议或默认cfg。

**修复轮csynth评估（11月18日22:37）：**与当前源码和默认cfg一致，约11分56秒完成，0 ERROR、无214-366，46项流水结果均II=1。XML顶层BRAM18K由1432降至1322（-110，余166），URAM=112、DSP=1349不变；LUT由375190增至381024（+1.55%，summary表为381030），FF由203096增至207683（+2.26%），因此不是LUT/FF资源目标全部通过。主FMBUF实际为4个28672x256、15-bit地址的URAM bank，各28块；BRAM尾仍为单份55296x256、770 BRAM18K，没有缩减特征存储或复制整帧RAM。主URAM读latency/interval仍为1/1；生成RTL可达实例核对Conv/WinGen/SA/POST/PPU consumer/BLOCK5 final/Vec/Pool/输出均各1个，SA仍1024 DSP。浅bank结构门槛通过，但总估计周期仍7.300ns，不能当作物理关键路径已修好，也不采用巨大的max latency推断整网耗时。
**决策：允许一次受控实现，不再凭HLS估计盲修。** 先看Vivado综合和place/phys-opt的实际CLB、control sets及URAM→PPU路径；若CLB仍接近上版98.56%、路径/拥塞未改善，则暂停长时间route并继续定位。原有输入DATAFLOW/吞吐警告仍在；201-303局部绑定警告不等于主RAM回退，最终Memory表及RTL已核对为4个2P URAM、各1-cycle。只有合法布线且100MHz setup/hold通过，才能进入统一版板测；本轮未运行综合/实现、未改HLS源码。

### 7.3 第二次实现失败与资源修复（2026-11-19）

四bank版实际恶化：850条信号未布通、536处overlap，route约4小时14分；WNS=-3.823ns、TNS=-5166.730ns。place后LUT=264560（比首版+16053）、FF=193539、CLB=42472/42660（99.56%）、control sets=2241；物理BRAM仍592 tiles，URAM仍112。**HLS少110 BRAM18K没有转化为物理BRAM节省，不能再拿它抵消LUT/packing回退。** 失败DCP最差路径仍为URAM→BLOCK5 composer（约13.52ns、10级逻辑）；task builder占24361 FF/10399 LUT，affine参数地址扇出最高20736，Conv issue控制扇出超过5000。

1. **撤销无效banking：**`memory.cpp`恢复单份114688x256 URAM及单读写字总线，删除四bank常量、选择器和对应结构gate；保留已收紧的合法描述符上限、alias有效位、B2保护及所有运行模式。不是恢复整版旧代码，也不改变布局或PARAM。
2. **合并冷配置通路：**`conv_engine.cpp`只保留`build_conv_task_from_issue()`，统一加载Conv descriptor/WinGen schedule/qparam及构造cfg；normal/BLOCK5只处理必要的源、目标、consumer和scratch差异，共用task提交。删除两套返回整份task的builder、重复previous-branch descriptor、未使用result.conv_desc及不可达preadd分支；不把配置放进DATAFLOW，不改MainCtrl single-tail/SA/WinGen/PPU算术，不新增CPP或ALLOCATION限制。
3. **消除系数存储地址大扇出：**`param_dma.cpp`将7x8的affine struct缓存改为8组320-bit BRAM字、每组32条；按固定前缀0/1/3/5/10/14/18存放26个实际block。每字保存4通道的mul/bias/shift/reserved位，保持PARAM的320-byte记录及getter数值不变；bank维静态完整分区，2P BRAM、关闭reset，读出并行解包。禁止串行逐通道读取。按端口/宽度映射预计72～144 BRAM18K，须由新报告确认，不能忽略取整或以增加DDR访问替代。

验证先跑source回归和全地址/跨帧、系数逐位/非法ID测试，再统一跑发布矩阵及最大RGB20/RGB lite全网host CSim，logits/mask必须0 mismatch；完成dead/legacy扫描后交用户手动csynth。新报告重点要求task builder无双份配置返回、affine缓存8个固定BRAM bank、主存仅一份、SA仍32x32/1024 DSP，热II及主URAM读1/1不退化；资源应向首版375190 LUT/203096 FF以下收敛，BRAM18K不超1488。下一次实现前先审查CLB/control sets和参数地址扇出，不再仅凭HLS LUT略超的历史经验放行。

本轮代码及旧路径清理已完成：182项常规回归通过，发布矩阵11次与最大RGB结构用例2次均为logits/mask 0 mismatch，包含跨模型返回、全部affine系数/reserved逐位校验和非法block拒绝；341个static helper引用检查无孤立项。默认cfg、MainCtrl、SA及PARAM/INPUTQ/golden未改；未运行csynth/实现。以上仅为源码及功能验证，不提前认定LUT/FF/BRAM、读延迟或物理时序已达标，交用户手动综合后再核对。

**资源修复csynth复核（11月19日22:14）：**与当前源码一致，约10分44秒完成，0 ERROR、无214-366，46项流水均II=1。XML资源为LUT=359233、FF=193400、BRAM18K=1338/1488、URAM=112/112、DSP=1347；比四bank失败版减少21791 LUT/14283 FF，增加16 BRAM18K，且比首版375190 LUT/203096 FF也更低，资源目标通过。生成RTL确认单task builder（13217 LUT/15068 FF），两套旧builder及两个独立mode adapter实例均不存在；Conv/WinGen/SA/POST/PPU/Vec/Pool仍各1个。affine缓存确为8个BRAM bank、各18 BRAM18K/1-cycle，合计144；字宽312是未使用高位reserved经优化后的结果，非算术变更。主FMBUF恢复单份URAM/BRAM，主URAM读latency/interval=1/1，SA仍1024 DSP。
**决策：可进入受控实现，先看综合/place。** 本轮是有效的HLS资源收敛，但LUT估计仍为器件的105.26%、URAM仍100%，7.300ns估计周期也未变化，不能提前宣称物理时序通过。重点核对CLB/control sets、旧affine地址20736扇出及URAM→PPU路径是否改善；若仍接近上版99.56% CLB或高拥塞，不再直接等待数小时route。另记录性能边界：packed-span helper由1～3-cycle变成固定2-cycle，task配置最大56-cycle；不能仅因MAC II不变就承诺整网周期完全不变，后续用板级stage计数核对。此轮只检查报告，未修改HLS或启动综合/实现。

## 8. Round 4：单平台实现与板级验收

**11月20日实现里程碑：**统一版100MHz实现通过，post-route phys-opt报告WNS=+0.075ns、TNS=0、WHS=+0.010ns，setup/hold均无失败端点。place资源LUT=235588、FF=176722、CLB=41630/42660（97.59%）、BRAM=664/744 tiles、URAM=112/112、DSP=1376；物理余量仍紧。已导出`platform_unified_1120`，下一步按同一平台逐模式500张验证：每模式只打印首图详细counter，逐图检查错误及文件身份，最后汇总PA/IoU和延时。当前10个发布case仍是单图产物，不能当成已备齐500张，也不能把缺失训练模型的其余硬件模式标为已验证；完整数据集及SD准备另行完成，App build仍由用户手动执行。

1. 用户手动综合/实现；只有一套 NPU 数据通路和 RAM，输出源选择无多驱动，BD 的控制接口、AXI depth、复位/时钟和地址空间重新核对。统一运行时接口需要重新 package IP、导出一个新平台，不沿用旧 XSA/ELF 配新寄存器。
2. 100MHz 下必须合法布线、setup/hold 通过；BRAM/URAM/DSP 不超器件容量。当前 H512 物理 total LUT=223151、URAM=108、CLB=95.19%，布局余量很紧，不能忽略控制集和路由压力。
3. HLS LUT 估计略超 341280 不单独判死刑，历史 H512 估计 343155 而物理可实现；同样不能据此允许无限增长。以共享层级、物理资源、拥塞及时序综合决策，资源失败时先定位新增 mode mux/复制和 RAM 端口，不降低性能掩盖问题。
4. 用同一 bitstream/ELF 做模式 suite，每种拟发布模式至少 3 次，并覆盖 C1<->C3、最小<->最大、2<->20 类、native<->lite、240/ROI 输出切换。先检查 mask 逐字及计数闭合，再记录延时；直方图一致不能代替逐像素一致。
5. 以各自稳定版作参考，固定模型及相同工作量比较；约 5% 以上额外延时必须解释并决定是否接受，不能用不同模型/ROI 的精度和耗时混算。统一目标是功能收敛且不显著退化，不承诺所有 native 模式都低于 50ms。

### 8.1 统一版500张全发布矩阵板测（2026-11-22复核）

`platform_unified_1120`与`INT8-BOARD-20261120-P7-UNIFIED-VAL500-TXT`完成10个发布case，各500张，共5000次推理；同一bitstream/ELF按清单切换模型和输入/输出模式。
SD日志为`LOGS/TEST0000.TXT`，最终`VALSUITE,PASS`；10组均有完整逐图计时、混淆矩阵、逐类PA/IoU、总体PA/mIoU及首图stage/conv计数，无ERROR/超时/保存失败。

下表为独立重读全部5000份mask与GT后的全验证集结果；精度单位为%，时间为100MHz RTL周期换算的每图均值，覆盖MODE_RUN，不含SD读写、CPU评估及终端打印。
宽图lite均只评估中心方形ROI；“输出512”不等于恢复完整512x1024视野。20-class忽略void类19/255，mIoU为19个有效类别的平均；二分类忽略GT=255。

| case | 输入/计算路径/输出 HxW | PA | mIoU | 平均ms |
|---|---|---:|---:|---:|
| G240B2 | gray240x240，native | 96.9112 | 79.9627 | 50.5602 |
| R256B2 | RGB256x512，native二分类 | 96.9799 | 82.3058 | 114.8452 |
| R256B20 | RGB256x512，native20-class | 84.6296 | 41.4913 | 140.8212 |
| G512B2 | gray512x512，native QAT10 | 98.2372 | 86.9902 | 225.4113 |
| G2S1M0 | gray256x512，中心裁剪->240，输出240 | 96.7935 | 79.0911 | 50.8332 |
| G2S1M1 | gray256x512，中心裁剪->240，输出256 | 96.8010 | 79.1299 | 52.1627 |
| G2S2M0 | gray512x1024，中心裁剪->240，输出240 | 96.7424 | 79.0119 | 50.9099 |
| G2S2M1 | gray512x1024，中心裁剪->240，输出512 | 96.7389 | 79.0182 | 65.2417 |
| G2S3M0 | gray512x512，下采样->240，输出240 | 96.7424 | 79.0119 | 50.9099 |
| G2S3M1 | gray512x512，下采样->240，输出512 | 96.7389 | 79.0182 | 65.2417 |

**正确性依据：**5000份mask的数量、顺序、长度、类别及日志CRC全部一致，GT身份与发布清单SHA一致；主机重新计算的混淆矩阵、总体/逐类指标及500张计时汇总均与App吻合。10个case各自冻结单图golden逐像素0 mismatch；每case首图stage总和及WIN/SA/POST各自计数窗口闭合，ERROR=0。ARM与RTL均值仅相差约0.27～2.03us。宽图512源与预裁剪512正方形源的500张输出逐字相同，分别覆盖240和512输出。

原生gray512和正方形lite输出512另用既有500张整数replay logits、核对模型/输入/GT身份后重建mask，各131072000像素均0 mismatch；结合宽图/正方形输出逐字相同，覆盖G512B2、G2S3M1、G2S2M1的全部500张软件对齐。未据此声称其他case的全部500张均已逐像素对齐软件推理。RGB二分类/20-class相对旧QAT记录的mIoU分别低约0.748/0.222个百分点；本次GT采样/整数部署口径与旧QAT记录不能直接等同，当前证据未显示统一硬件引入数值错误，也不将这两个差值直接归因为硬件故障。

**性能结论：**功能矩阵通过，但不能认定全部性能门槛通过。相对第1节独立稳定版，gray240/gray512/lite输出240约慢1.4%～2.1%，RGB二分类约快4.7%；RGB20由120.6472ms增至140.8212ms（+16.72%），lite输出512由54.1201ms增至65.2417ms（+20.55%），超过约5%退化预算。首图RGB二分类与20-class仅Conv相差18400周期，PPU行消费/输出相差2579200周期，是新增类别时间的主要来源；lite512相对lite240主要增加1433176周期于同一PPU阶段。优先检查共享输出上采样/分类路径的运行时控制和调度，不将此退化归因于SA吞吐，也不通过再次训练掩盖。源码当前20-class插值以4类为一组串行处理，与此热点一致；具体修补需另行规划后执行。

本次确认的是10个有模型的发布case，不是全部40种硬件组合；其余缺适配模型的模式仍不作精度/板级结论。原始日志、5000份板级mask及独立统计已归档到`hw_artifacts/unified_val500_1122/results_TEST0000`，`audit.json`保留逐类指标、混淆矩阵、计时和校验结果；未改SD结果或执行综合/build。

最终交付为本分支的一套源码、运行时契约、编译/评估工具、模型可用清单、App/SD 清单及精简板级结果。
PA/mIoU 同时注明模型、GT 视野、输出分辨率及单图/500张口径；只有实际验证的模型和模式进入发布清单，其余明确标为尚缺适配模型或验证。
全部发布门槛通过后再按用户指令提交/push，并创建统一版稳定 tag；不覆盖原分支稳定 tag，不提交 build、平台、hw_artifacts、SD 大文件、临时目录或本机 IDE 配置。

## 9. 最终检查清单

- 一个 IP/bitstream/ELF；App 运行时切换，不依赖重新 build、编译宏或多个常驻 NPU 副本。
- 四种源图几何、C1/C3、native/lite 和正确 ROI mask 语义均有明确能力表；类别能力与实际训练模型 availability 分开。
- 最大模式存储及别名安全有证明，1048-word 和 240 非对齐边界有测试；没有新增隐式 DDR 中间访存。
- WinGen schedule 主入口、单 Conv/SA、PPU/Vec/store owner 与有限 DATAFLOW 保持，接口和状态切换无遗留依赖。
- PARAM/INPUTQ/golden/replay/GT/App 同契约，正常结果逐字一致、ERROR=0；时序/资源/模式切换经同一平台板级确认。
