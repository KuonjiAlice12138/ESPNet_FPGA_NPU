# Workplan 0922 - H240W240 灰度二分类 FP32 训练

## 0. 本轮目标

在不加载任何已有 checkpoint 的前提下，随机初始化并训练一套 `p=1, q=1` 的
`Model_flattened.ESPNet_Encoder` 二分类 FP32 模型，输入改为真实单通道灰度图，几何
改为原图中心正方形裁剪后缩放至 `240x240`。本轮只判断新输入形态的精度上限，不做
INT8 QAT、硬件 artifact 导出、HLS 修改或 decoder 扩展。

当前 H256W512 RGB FP32 参照：PA `0.97143`、mIoU `0.83027`，车辆类 IoU
`0.69106`、背景类 IoU `0.96948`。

## 1. 固定设计

- 数据仍使用当前 Cityscapes train/val 列表和 binary2 标签映射：车辆 trainId
  `13..18 -> 0`，其余有效类别 `-> 1`，void `255` 保持 ignore。
- 每张图先取中心正方形：`side=min(H,W)`，裁剪起点使用整数下取整；图像使用双线性
  缩放至 `240x240`，标签使用最近邻缩放。
- 图像使用 OpenCV `BGR2GRAY` 转换；训练缓存单独版本化并保存一个灰度 mean/std，
  禁止复用 RGB cache。
- Dataset 输出 `[1, 240, 240]` FP32 tensor；训练和验证几何完全一致。训练只保留
  水平翻转，不叠加会改变中心视野的二次随机裁剪。
- 模型保持原 encoder 拓扑、卷积核、dilation、`p=q=1` 和二分类 classifier。
  首层输入改为 `1` 通道；两条输入投影仍来自灰度输入，因此拼接通道从 `16+3`、
  `128+3` 分别改为 `16+1`、`128+1`。对应 BatchNorm 与下采样卷积输入通道同步改为
  `17` 和 `129`。
- 训练从 PyTorch 默认随机初始化开始，禁止加载、折叠或迁移任何旧 checkpoint。
- 输出仍为 `30x30` 双通道 logits，验证时使用
  `bilinear(align_corners=False)` 上采样到 `240x240` 后 argmax。

## 2. 实现边界

在 `D:/ESPNet/train_cityscapes20` 的现有训练环境上做参数化扩展：

- `Model_flattened.py`：为 encoder 增加 `input_channels`，只允许 `1` 或 `3`，并由该值
  推导输入投影拼接维度。
- `DataSet.py`：增加显式 `image_mode`，灰度模式用 `cv2.IMREAD_GRAYSCALE`，不得在模型
  前把灰度复制成三通道。
- `Transforms.py`：增加中心正方形裁剪，并使 Normalize/ToTensor 同时支持单通道和
  三通道。
- `loadData.py`：cache metadata 加入 mode/version；灰度模式独立计算标量 mean/std。
- `main.py`：增加 `--image-mode gray|rgb` 与中心裁剪开关，模型按 mode 构造；训练日志
  写明几何、通道数、裁剪方式和随机初始化状态。
- 新增单元测试覆盖裁剪坐标、图像/标签插值、单通道 tensor shape、模型通道尺寸、
  cache 不混用和一次前向/反向传播。

不修改根目录量化模型、QAT、exporter、compiler、HLS 或现有 H256W512 产物。

## 3. 训练设置与验收

- epoch：`20`
- optimizer：Adam，初始学习率 `5e-4`，weight decay `5e-4`
- scheduler：第 `10` 和 `15` epoch 将学习率乘 `0.5`
- loss：CrossEntropy，ignore index `255`；使用现有 binary2 class weights
- batch size：以显存探测后的最大稳定值为准，但不改变梯度语义
- seed：固定并记录；每 epoch 记录 train/val loss、PA、两类 IoU、mIoU 和耗时
- 保存 `checkpoint_last.pth` 与 mIoU 最优的 `model_best.pth`，最终单独复评最优模型。

结果分级：

- 可接受：相对 H256W512 基准，mIoU 下降不超过 `2` 个百分点且车辆 IoU 下降不超过
  `3` 个百分点。
- 中度退化：mIoU 下降 `2..5` 个百分点，或车辆 IoU 下降 `3..7` 个百分点；先检查
  中心裁剪造成的类别覆盖变化及训练是否尚未收敛，再决定是否延长训练。
- 严重退化：mIoU 下降超过 `5` 个百分点，或车辆 IoU 下降超过 `7` 个百分点。

## 4. Decoder 决策门槛

本轮不直接加入反卷积。若出现严重退化，先用 RGB/灰度和 `30x30` logits 的对照评估
区分颜色信息损失、中心视野损失与低分辨率边界损失。只有证据表明主要瓶颈来自
`1/8` logits 时，才规划轻量可训练 decoder。

decoder 优先方案是利用 `1/4` 浅层特征的单次跳连融合，再逐级双线性上采样和小卷积
细化；不优先采用裸 `ConvTranspose2d`，因为它容易产生棋盘伪影，并会给后续 NPU 增加
新算子、特征缓存和显著访存压力。decoder 是否进入硬件必须另立计划，以精度收益、
MAC/访存增量和 HLS 可实现性共同验收。

## 5. 执行顺序

1. 编写失败测试，锁定灰度、中心裁剪、模型通道和 cache contract。
2. 修改训练侧代码并跑全部训练环境测试及一个 mini-batch 前向/反向 smoke test。
3. 清理独立输出目录，固定随机种子，启动 20 epoch FP32 训练。
4. 训练结束后加载 `model_best.pth` 复评 500 张验证集，输出 PA、mIoU、两类 IoU。
5. 按第 3 节分级；只有严重退化才进入第 4 节的结构性实验。

## 6. 100 epoch 实验结果（2026-09-22）

epoch 0 从随机权重开始，epoch 20 后保留模型与 Adam 状态继续到总计 100 epoch；续训
阶段以当前 `1.25e-4` 为基准，并在全局 epoch `40/70/90` 依次减半。未加载其他模型：

- 输出目录：`D:/ESPNet/resolution_h240w240_gray/fp32_binary2_scratch20`
- cache：`D:/ESPNet/resolution_h240w240_gray/cache/binary2_gray_center_stats_v2.pkl`
- 训练集/验证集：`2975/500`；batch size `8`；seed `20260922`
- 模型参数量：`109315`；输入 `[N,1,240,240]`；logits `[N,2,30,30]`
- 100 epoch 的逐轮训练与验证累计约 `7219.2 s`；全局最佳出现在 epoch 60。

| 指标 | H256W512 RGB 基准 | 灰度 20 epoch | 灰度 100 epoch | 100 epoch 相对基准 |
|---|---:|---:|---:|---:|
| PA | 0.97143 | 0.96249 | 0.96595 | -0.00548 |
| mIoU | 0.83027 | 0.77381 | 0.78672 | -0.04355 |
| 车辆 IoU | 0.69106 | 0.58725 | 0.60939 | -0.08167 |
| 背景 IoU | 0.96948 | 0.96038 | 0.96404 | -0.00544 |

灰度 cache 的 mean/std 为 `85.9737/48.4239`，有效像素直方图为
`[176723898, 2562200441]`，标签没有塌缩或错映射。20 到 100 epoch 使 mIoU 提升
`1.29` 个百分点、车辆 IoU 提升 `2.21` 个百分点，但仍未恢复原水平。epoch 60 后训练
loss 从约 `0.0527` 继续降到 `0.0447`，验证 mIoU 却未再超过 `0.78672`，且验证 loss
总体上升，说明后段已进入过拟合而不是训练不足。

按原门槛，mIoU 下降 `4.36` 个百分点属于中度退化，但车辆 IoU 下降 `8.17` 个百分点
仍属于严重退化。损失高度集中在稀疏车辆类，继续单纯增加 epoch 没有价值。

## 7. 后续精度恢复顺序

当前不能把差距直接归因于双线性上采样。两版 encoder 都是 output stride 8；`30x30`
与 `32x64` 的每个 logits 单元都覆盖约 `8x8` 输入像素。真正同时变化的是 RGB 到灰度、
横向视野、验证样本分布以及从头训练，因此裸反卷积不是有证据支持的第一选择。

下一轮按以下顺序执行：

1. 先用现有 RGB checkpoint 在相同 `240x240` 中心裁剪验证集上做只评估对照；必要时再做
   短周期 RGB 适配训练。该结果用于拆分几何/裁剪损失与灰度信息损失。
2. 在不改变推理图的前提下，优先试验面向稀疏车辆和边界的训练损失，例如
   `weighted CE + soft Dice`，或只在训练期使用的边界辅助损失。该方案不增加 NPU 算子、
   参数或访存，是首选精度恢复手段。
3. 若同几何 RGB 对照明显高于灰度模型，则主要瓶颈是颜色信息；应考虑灰度对比度增强、
   从灰度派生的低成本梯度特征，或重新评估必须单通道的约束，decoder 无法补回颜色。
4. 只有上述训练侧方案仍不能恢复且误差集中在小目标边界时，才加入轻量 decoder：从
   `1/4` 分辨率浅层特征引入一次 skip fusion，以双线性上采样加小卷积逐级细化。
5. 不优先采用裸 `ConvTranspose2d`；它容易产生棋盘伪影，并给后续 NPU 增加新算子、
   中间特征缓存和带宽压力。

结论：100 epoch 已排除“训练轮数不足”作为主要原因。可以开始考虑 decoder，但应先完成
同几何 RGB 对照与零推理开销的 loss 改进，以免用硬件代价掩盖灰度信息损失。

## 8. Loss-only 精度恢复实验（2026-09-22）

先执行不改变推理图的 `weighted CE + vehicle soft Dice`：

- source：100 epoch 全局最佳 `model_best.pth`（epoch 60，mIoU `0.78672`）
- CE：保持现有 class weights、ignore 255 和 `30x30` logits 监督
- Dice：只优化 class 0 车辆；将 logits 双线性上采样到 `240x240` 后计算 soft Dice；
  ignore 区域不参与
- 总损失：`weighted_CE + 0.25 * vehicle_soft_dice`
- fine-tune：20 epoch，Adam `2e-5`，weight decay `5e-4`，epoch `10/15` 各衰减一半
- 输出：`D:/ESPNet/resolution_h240w240_gray/fp32_binary2_dice025_ft20`

验收以 source checkpoint 为基准，而不是只与微调过程首轮比较：mIoU 和车辆 IoU 必须同时
不退化；车辆 IoU 提升 `>=1` 个百分点才视为值得保留。若无收益，不继续堆叠 Dice 权重，
转入同几何 RGB 对照；若有收益但仍明显低于 RGB 基准，再评估边界辅助损失或轻量 decoder。

20 epoch 微调已完成，最终加载并复评验证集 mIoU 最优的 epoch 8 checkpoint：

| 指标 | 灰度 100 epoch source | Dice 微调最佳 | 相对 source | H256W512 RGB 基准 | 相对 RGB 基准 |
|---|---:|---:|---:|---:|---:|
| PA | 0.96595 | 0.97028 | +0.00433 | 0.97143 | -0.00115 |
| mIoU | 0.78672 | 0.80554 | +0.01883 | 0.83027 | -0.02473 |
| 车辆 IoU | 0.60939 | 0.64248 | +0.03309 | 0.69106 | -0.04858 |
| 背景 IoU | 0.96404 | 0.96860 | +0.00456 | 0.96948 | -0.00088 |

- 输出目录：`D:/ESPNet/resolution_h240w240_gray/fp32_binary2_dice025_ft20`
- 最佳模型：`model_best.pth`；完整结果：`summary.json`；逐 epoch 数据：`train_log.csv`
- epoch 8 后训练 loss 仍缓慢下降，但验证 mIoU 未再超过 `0.80554`；继续训练到 epoch 19
  只在 `0.80268..0.80425` 区间波动，因此不能用最后权重替代最佳 checkpoint。
- 相对 source，PA、mIoU、车辆 IoU 同时提升，且车辆 IoU 提升 `3.31` 个百分点，明显超过
  预设的 `1` 个百分点 gate。该改进无需增加推理参数、算子或访存，确定保留。
- 相对原 RGB 基准，mIoU 差距由 `4.36` 缩小到 `2.47` 个百分点，车辆 IoU 差距由
  `8.17` 缩小到 `4.86` 个百分点；精度退化已从车辆类严重退化收敛为中度退化。

结论：训练目标与类别不平衡是此前精度损失的重要来源，当前实验已经证明无需 decoder
也能恢复大部分差距。下一步先做相同 `240x240` 中心裁剪的 RGB 对照，分离灰度与几何损失；
若仍需继续提高灰度精度，再增加仅训练期生效的轻量边界辅助损失，并保持当前 Dice 项。
在上述零推理开销方案完成前，不修改模型推理拓扑，也不引入反卷积 decoder。
