# frame_dma Spec

## 1. 对应关系

- 逻辑模块：`frame_dma`
- 当前实现文件：`src/frame_dma.cpp`

## 2. 来源整理
本文件重组自：

- `01_System_Architecture.md`
- `04_On_Chip_BRAM_Controller.md`
- `06_DMA_and_AXI_Interface.md`

## 3. 模块职责
`frame_dma` 负责运行期帧数据的 DDR <-> 片上搬运：

1. `MODE_RUN` 时读取输入帧
2. `MODE_RUN` 结束时写回最终输出
3. 在 `256-bit` AXI 向量与片上 byte 语义缓存之间做 pack/unpack

## 4. 不负责的事情

- 不负责 `param_blob` 加载
- 不负责中间特征图层间搬运
- 不负责卷积或后处理计算

## 5. 冻结接口语义

- `gmem_frame_in`：输入帧 DDR 基址
- `gmem_frame_out`：输出结果 DDR 基址

规则：

1. 地址必须 `64-byte` 对齐
2. 只允许整帧输入和整帧输出走 DDR
3. 中间 tensor 不得默认借助本模块回写 DDR

## 6. 数据格式

- DDR 侧统一按 `ap_uint<256>` 访问
- FMBUF 内部仍按稠密 `NHWC INT8` 布局

即：

- 总线层：宽向量搬运
- 片上逻辑层：byte 地址和 `tensor_desc_t` 解释

## 7. 读路径约束

### 7.1 输入帧读取

- 首次 `LOAD_FM` 时把 `input` 读入 `FMEM`
- `input` 必须在 B2 构建前保留，因为后续还要走两次 `avgpool`

### 7.2 不允许的实现退化

- 不允许每层重新从 DDR 读 activation
- 不允许因为局部 scratch 不够而把中间结果写回 DDR 再读出

## 8. 写路径约束

- 仅最终 `classifier` 输出默认写回 DDR
- 输出 shape 冻结为 `1 x 64 x 128 x 2`
- 输出前应已满足当前量化域约束

## 9. 与片上存储的关系
`frame_dma` 的片上目标落点必须与 `tensor_desc_t` 和 `FMEM0/1/2` 规划一致。

它负责搬运，不负责生命周期决策；真正的 tensor 落点和覆盖策略由 `on_chip_memory` 统一控制。

## 10. 性能计数
本模块是 `ddr_rd / ddr_wr` 计数的主要来源之一，必须可被 `perf_counter_and_irq` 统计。

## 11. 验收重点

1. 输入帧能正确读入片上
2. 输出结果能正确写回 DDR
3. pack/unpack 与 `NHWC INT8` 语义一致
4. 不引入任何层间 DDR spill
