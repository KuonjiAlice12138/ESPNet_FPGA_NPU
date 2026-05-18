# memory Spec

## 1. 对应关系

- 逻辑模块：`on_chip_memory`
- 当前实现文件：`src/memory.cpp`

## 2. 来源整理
本文件主要重组自：

- `04_On_Chip_BRAM_Controller.md`

## 3. 模块职责
`on_chip_memory` 是单 IP NPU 的片上存储调度中心，负责：

1. 管理全局特征图 bank
2. 管理权重和量化参数 bank
3. 为卷积、池化、后处理和 concat 提供统一片上读写语义
4. 实施 tensor 生命周期复用

## 4. 存储层次冻结

### 4.1 总体原则

- `URAM`：大块 `FMBUF`
- `BRAM`：`WBUF / QBUF / line buffer / window buffer / scratch / FIFO`
- `LUTRAM/寄存器`：极小局部状态

### 4.2 物理 FMBUF 与逻辑 view

当前综合结果表明，`0x940000` byte 的三大静态 FMEM bank 无法在目标 `ZU15EG` 上实现。因此 memory 模块冻结为一个共享物理 FMBUF，而不是三个独立物理大数组。

| Region | Fixed Size | Impl | Usage |
|---|---:|---|---|
| `FMBUF_URAM` | `0x380000` | URAM | 输入、B2/B3 主容器的低地址主体 |
| `FMBUF_BRAM` | `0x218000` | BRAM | B1、L2/L3 局部 scratch、pool scratch、高地址尾部 |

`FMBUF` 物理总预算固定为 `0x598000` bytes。`FMEM0/1/2` 不再表示三块独立物理 RAM，而是逻辑 bank/view 语义。

### 4.3 关键 tensor 物理布局

| Tensor | Physical Base | Logical Shape | Physical Stride / Offset |
|---|---:|---|---|
| `T_INPUT` | `0x000000` | `512 x 1024 x 3` | compact |
| `T_B1_CAT/ACT` | `0x180000` | `256 x 512 x 19` | compact |
| `T_POOL1` | `0x3E0000` | `256 x 512 x 3` | compact |
| `T_POOL_TMP` | `0x440000` | `256 x 512 x 3` | compact scratch |
| `T_POOL2` | `BRAM_SCR1` | `128 x 256 x 3` | independent 96 KB BRAM scratch |
| `T_L20_CAT/ACT` | `0x000000` | `128 x 256 x 64` | view into B2, `phys_c=131, c_offset=64` |
| `T_L2B0_CAT/ACT` | `0x000000` | `128 x 256 x 64` | view into B2, `phys_c=131, c_offset=0` |
| `T_B2_CAT/ACT` | `0x000000` | `128 x 256 x 131` | compact B2 container |
| `T_L30_CAT/ACT` | `0x418000` | `64 x 128 x 128` | compact, reuses L20 region |
| `T_L3B0_CAT/ACT` | `0x000000` | `64 x 128 x 128` | view into B3, `phys_c=256, c_offset=128` |
| `T_B3_CAT/ACT` | `0x000000` | `64 x 128 x 256` | compact B3 container |
| `T_OUT` | `0x200000` | `64 x 128 x 2` | compact, after B3 read window |

### 4.4 参数 bank

| Bank | Fixed Size | Usage |
|---|---:|---|
| `WBUF` | `128 KB` | 全部 INT8 权重 |
| `QBUF0` | `64 KB` | conv/add/pool qparam |
| `QBUF1` | `64 KB` | affine / scale / uop / tensor desc |

### 4.5 局部缓存

- line buffer：BRAM
- window buffer：BRAM
- stream FIFO：BRAM
- add scratch：BRAM
- `BRAM_SCR0/1`：保留为逻辑 scratch bank id；实现上允许映射到 FMBUF 的生命周期空闲区域

## 5. 生命周期复用冻结

### 5.1 原则

1. 输入帧在 B2 前必须保留
2. B1 后 level1 原始输出可释放
3. B2 完成后可释放 input 占用
4. stage3 只保留 `b2_act / level3_xxx / b3_act / output`

### 5.2 固定复用顺序

| Time | Low FMBUF `0x000000...` | High FMBUF `0x418000...` |
|---|---|---|
| Load / B1 | input, B1, pool1 | pool tmp only |
| Early pool2 | input, B1, pool2 scratch | pool tmp |
| Level2_0 | L20 view at B2 `c[64:127]` | LS_C1 high scratch |
| Level2_B0 / B2 | B2 container and L2B0/L20 views | L2 block scratch |
| Level3_0 | B2 container, low scratch after first conv | L30 compact + LS_C1 scratch |
| Level3_B0 / B3 | B3 container and L3B0 view | L30 compact |
| Classifier | B3 container, output | free |

## 6. 地址规则

### 6.1 稠密 NHWC

默认 compact tensor：

`addr = base + ((h * W + w) * C + c)`

view tensor：

`addr = base + ((h * W + w) * physical_c_stride + channel_offset + c)`

`tensor_desc_t.reserved0` 固定解释为 `physical_c_stride`，为 `0` 时等于逻辑 `C`。`tensor_desc_t.reserved1` 固定解释为 `channel_offset`。

### 6.2 concat 写入

`dst_c = c_offset + c_local`

对于 `T_L2B0_ACT -> T_B2_CAT c[0:63]` 和 `T_L3B0_ACT -> T_B3_CAT c[128:255]`，源 tensor 已经是目标容器 view，`STORE/CONCAT` 可以退化为 no-op 或 metadata check。其他 concat 仍按 256-bit tile/word 路径完成片上复制，不允许经 DDR。

## 7. 读写仲裁

### 7.1 必需端口语义

1. `conv read`
2. `pool read`
3. `postproc / concat write`
4. `residual read`

### 7.2 首版仲裁原则

- `conv read` 优先级最高
- `postproc write` 可与 `conv read` 重叠
- 不要求 `pool read` 与 `conv read` 在同 bank 同拍并发

## 8. 冻结接口语义

- `tensor_desc_t` 解释逻辑 bank / base_offset / h / w / c
- `window_generator` 通过本模块读取窗口输入
- `post_process / concat / pool` 通过本模块回写

## 9. 不允许的实现偏差

1. 因 bank 不够退化成大规模 DDR spill
2. concat 结果先回 DDR 再读出
3. 为简化设计放弃生命周期复用
4. 把 view tensor 当 compact tensor 逐 byte 复制，导致 B2/B3 容器重复占用片上大容量

## 10. 必测场景

1. B1 concat 与 B2 concat 不覆盖
2. `Cin=131` 尾 tile 读取
3. `dilation=16` 行缓存访问
4. stride=2 / same padding
5. 生命周期覆盖后不污染下一阶段数据
6. `T_L2B0_ACT` 写入后可从 `T_B2_CAT c[0:63]` 读出
7. `T_L3B0_ACT` 写入后可从 `T_B3_CAT c[128:255]` 读出
