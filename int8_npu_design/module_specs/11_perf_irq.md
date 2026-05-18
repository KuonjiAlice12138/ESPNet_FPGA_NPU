# perf_irq Spec

## 1. 对应关系

- 逻辑模块：`perf_counter_and_irq`
- 当前实现文件：`src/perf_irq.cpp`

## 2. 来源整理
本文件重组自：

- `06_DMA_and_AXI_Interface.md` 的寄存器、状态、中断与验收要求

## 3. 模块职责
`perf_counter_and_irq` 负责：

1. 维护运行期状态与错误码
2. 记录性能计数器
3. 提供 `done / idle / irq` 信号
4. 暴露 `current_uop_id`

## 4. 需要维护的状态

- `done`
- `idle`
- `error_code`
- `current_uop_id`

## 5. 需要统计的性能计数

1. `PERF_CYCLE`
2. `PERF_DDR_RD`
3. `PERF_DDR_WR`
4. `PERF_STALL`

## 6. AXI-Lite 可见语义
本模块与包装 IP 一起完成以下寄存器语义：

- `CTRL`
- `IRQ_ENABLE`
- `STATUS`
- `PERF_*`

其中包装 IP 负责对齐固定偏移，本模块负责维护底层计数和状态值。

## 7. 中断约束

- 首版允许 polling
- 但必须保留一个 `irq` 输出
- `done_irq_en` 生效后，完成时可触发中断

## 8. 错误码冻结

| Code | Meaning |
|---|---|
| `0` | OK |
| `1` | invalid mode |
| `2` | bad blob magic/version |
| `3` | uop decode error |
| `4` | tensor desc out of range |
| `5` | param desc out of range |
| `6` | bank overflow |
| `7` | unsupported opcode |

## 9. 与其他模块的边界

- `instruction_fetch_decode` 提供 `current_uop_id` 和 decode 相关错误
- `frame_dma` 提供 `ddr_rd / ddr_wr` 统计来源
- `int8_core` 提供模式切换边界

## 10. 验收重点

1. 完成后 `done` 有效
2. polling 和 irq 两种模式都可支持
3. `cycle / ddr_rd / ddr_wr / stall` 可读
4. 出错时 `STATUS` 能定位到 `current_uop_id`
