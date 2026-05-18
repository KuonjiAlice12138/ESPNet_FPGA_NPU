# concat_unit Spec

## 1. 对应关系

- 逻辑模块：`concat_writer`
- 当前实现文件：`src/concat_unit.cpp`

## 2. 来源整理
本文件重组自：

- `05_Hardware_Pooling_and_Activation.md` 的 concat 内容
- `07_ESPNet_Encoder_Uop_Schedule.md` 的 concat/scale 共享约束

## 3. 模块职责
`concat_writer` 负责当前网络中所有 concat 的目标地址写入和通道切片路由。

它不是算术模块，不做数值变换；只做：

1. 目标 tensor 选择
2. `channel_offset` 计算
3. 通道子区间写回

## 4. 设计原则冻结

1. concat 不允许“读两路再复制一遍”
2. 各 producer 必须直接写入目标 tensor 不同 channel 区间
3. 进入同一 concat 目标 tensor 的所有分支必须预先对齐到同一 scale

## 5. 地址公式冻结

```cpp
dst_addr =
    dst_base +
    ((h * dst_w + w) * dst_c_total) +
    channel_offset +
    c_local;
```

## 6. 当前网络固定偏移

| Node | Offsets |
|---|---|
| B1 | `0`, `16` |
| Level2_0 cat | `0`, `16`, `28`, `40`, `52` |
| Level2_Block0 cat | `0`, `16`, `28`, `40`, `52` |
| B2 | `0`, `64`, `128` |
| Level3_0 cat | `0`, `28`, `53`, `78`, `103` |
| Level3_Block0 cat | `0`, `28`, `53`, `78`, `103` |
| B3 | `0`, `128` |

## 7. 共享 scale 约束

1. concat 本身不做 requant
2. 所有分支在写入目标 tensor 前必须已经 requant 到目标 scale

## 8. 存储落点

- 小写入队列/FIFO：BRAM
- concat 目标 tensor：`FMBUF`
- 不允许 concat 结果回写 DDR

## 9. 验收重点

1. B1/B2/B3 concat 不覆盖
2. fixed offset 与现有 FP32 工程一致
3. 只做路由，不引入额外数值变化
