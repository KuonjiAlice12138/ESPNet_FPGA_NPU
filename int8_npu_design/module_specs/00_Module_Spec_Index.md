# ESPNet INT8 NPU Module-Aligned Spec Index

## 1. 文档目的
本目录把当前 `ESPNet_Encoder` INT8 NPU 的 spec 重新整理为**一文档对应一个硬件模块**的形式。

重整理原则：

1. 不改变已经冻结的功能边界
2. 不改变已有接口、数据结构、变量命名和行为约束
3. 不删除原 `01~07` 文档中的系统设计信息
4. 原 `01~07` 文档继续保留，作为来源追溯和共享约束文档

## 2. 为什么需要这套文档
原始 `01~07` 文档在项目早期基本遵循“一个模块一份说明”的思路，但随着后续讨论加入：

- `concat / add / activation / pool` 共文档
- `DMA / AXI / uop / param blob / perf / irq` 共文档
- 整网 `uop schedule` 与 scale domain 的共享约束

原文档逐渐演变为“按功能域归档”的形式，不再严格与 `.cpp` 模块一一对应。

本目录的目标是恢复“看一个模块就能看到该模块完整 spec”的阅读方式。

## 3. 当前模块与文档映射

| 逻辑模块 | 当前实现文件 | 模块 spec |
|---|---|---|
| `espnet_encoder_int8_core` | `src/int8_core.cpp` | `01_int8_core.md` |
| `instruction_fetch_decode` | `src/if_dec.cpp` | `02_if_dec.md` |
| `frame_dma` | `src/frame_dma.cpp` | `03_frame_dma.md` |
| `param_dma` | `src/param_dma.cpp` | `04_param_dma.md` |
| `on_chip_memory` | `src/memory.cpp` | `05_memory.md` |
| `window_generator` | `src/win_gen.cpp` | `06_win_gen.md` |
| `systolic_array_core` | `src/sa_core.cpp` | `07_sa_core.md` |
| `post_process_unit` | `src/ppu.cpp` | `08_ppu.md` |
| `avgpool_unit` | `src/avgpool_unit.cpp` | `09_avgpool_unit.md` |
| `concat_writer` | `src/concat_unit.cpp` | `10_concat_unit.md` |
| `perf_counter_and_irq` | `src/perf_irq.cpp` | `11_perf_irq.md` |

## 4. 仍然保留的共享文档
以下文档不直接对应某一个 `.cpp` 文件，但仍然保留，作为共享约束来源：

- `01_System_Architecture.md`
- `06_DMA_and_AXI_Interface.md`
- `07_ESPNet_Encoder_Uop_Schedule.md`

它们的作用分别是：

- `01_System_Architecture.md`：整体架构、资源预算、顶层约束、文档导航
- `06_DMA_and_AXI_Interface.md`：二进制契约、寄存器表、descriptor/uop/param blob 冻结格式
- `07_ESPNet_Encoder_Uop_Schedule.md`：整网执行顺序、tensor id、param_id、scale domain 共享约束

## 5. 阅读建议

1. 先读 `01_System_Architecture.md`
2. 再按本目录读取对应模块 spec
3. 涉及打包格式、寄存器、uop 时回看 `06_DMA_and_AXI_Interface.md`
4. 涉及执行顺序和整网共享语义时回看 `07_ESPNet_Encoder_Uop_Schedule.md`
