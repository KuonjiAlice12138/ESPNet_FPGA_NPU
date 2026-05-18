# ESPNet_Encoder INT8 UOP Schedule

## 1. 文档目的
本文档把当前 `ESPNet_Encoder` 的软件拓扑，冻结成 **唯一有效的硬件执行表**。从本版开始：

1. tensor id 固定
2. bank/base offset 固定
3. `CAT -> ACT` 物理 alias 固定开启
4. `POOL_TMP` 单独建模，不再复用 `T_POOL1`
5. `param_id` 改为 **opcode-specific namespace**
6. 当前网络所有 `uop.qparam_id` 固定为 `0`

## 2. 全局 tensor id 与物理映射冻结

### 2.1 规则

1. `CAT -> ACT` 对应同一物理地址窗口
2. `FMEM0/1/2` 只保留逻辑 bank/view 语义，物理容量以共享 `FMBUF` 为准
3. `BRAM_SCR0/1` 允许被全局小 tensor 使用
4. 下表中的 `Base Offset` 是最终 tensor descriptor 应写入的值
5. `tensor_desc.reserved0/reserved1` 分别固定为 `physical_c_stride/channel_offset`

### 2.2 tensor 表

`Base Offset` 是共享物理 FMBUF 内的地址。`Phys View` 为 compact 时，地址规则为 `base + ((h * W + w) * C + c)`；非 compact 时，`reserved0=physical_c_stride`，`reserved1=channel_offset`。

| Tensor ID | Name | Bank/View | Base Offset | Shape | Phys View |
|---|---|---|---:|---|---|
| `0` | `T_INPUT` | `FMEM0` | `0x000000` | `512 x 1024 x 3` | compact |
| `1` | `T_POOL1` | `FMEM0` | `0x3E0000` | `256 x 512 x 3` | compact |
| `2` | `T_B1_CAT` | `FMEM1` | `0x180000` | `256 x 512 x 19` | compact |
| `3` | `T_B1_ACT` | `FMEM1` | `0x180000` | `256 x 512 x 19` | compact |
| `4` | `T_L20_CAT` | `FMEM0 view` | `0x000000` | `128 x 256 x 64` | `phys_c=131, c_offset=64` |
| `5` | `T_L20_ACT` | `FMEM0 view` | `0x000000` | `128 x 256 x 64` | `phys_c=131, c_offset=64` |
| `6` | `T_L2B0_CAT` | `FMEM0 view` | `0x000000` | `128 x 256 x 64` | `phys_c=131, c_offset=0` |
| `7` | `T_L2B0_ACT` | `FMEM0 view` | `0x000000` | `128 x 256 x 64` | `phys_c=131, c_offset=0` |
| `8` | `T_POOL2` | `BRAM_SCR1` | `0x000000` | `128 x 256 x 3` | independent 96 KB BRAM scratch |
| `9` | `T_B2_CAT` | `FMEM0` | `0x000000` | `128 x 256 x 131` | compact B2 container |
| `10` | `T_B2_ACT` | `FMEM0` | `0x000000` | `128 x 256 x 131` | compact B2 container |
| `11` | `T_L30_CAT` | `FMEM1` | `0x418000` | `64 x 128 x 128` | compact |
| `12` | `T_L30_ACT` | `FMEM1` | `0x418000` | `64 x 128 x 128` | compact |
| `13` | `T_L3B0_CAT` | `FMEM0 view` | `0x000000` | `64 x 128 x 128` | `phys_c=256, c_offset=128` |
| `14` | `T_L3B0_ACT` | `FMEM0 view` | `0x000000` | `64 x 128 x 128` | `phys_c=256, c_offset=128` |
| `15` | `T_B3_CAT` | `FMEM0` | `0x000000` | `64 x 128 x 256` | compact B3 container |
| `16` | `T_B3_ACT` | `FMEM0` | `0x000000` | `64 x 128 x 256` | compact B3 container |
| `17` | `T_OUT` | `FMEM0` | `0x200000` | `64 x 128 x 2` | compact |
| `18` | `T_POOL_TMP` | `BRAM_SCR0` | `0x000000` | `256 x 512 x 3` | compact scratch at `0x440000` |

说明：

1. `T_B1_CAT` 与 `T_B1_ACT` 同址
2. `T_L20_CAT` 与 `T_L20_ACT` 同址，且二者都是 `T_B2` 容器的 `c[64:127]` view
3. `T_L2B0_CAT` 与 `T_L2B0_ACT` 同址，且二者都是 `T_B2` 容器的 `c[0:63]` view
4. `T_B2_CAT` 与 `T_B2_ACT` 同址
5. `T_L30_CAT` 与 `T_L30_ACT` 同址
6. `T_L3B0_CAT` 与 `T_L3B0_ACT` 同址，且二者都是 `T_B3` 容器的 `c[128:255]` view
7. `T_B3_CAT` 与 `T_B3_ACT` 同址

## 3. 局部 scratch id 冻结
以下 scratch 不进入全局 tensor table，只在 block 内部使用：

| ID | Scratch | Usage |
|---|---|---|
| `0x80` | `LS_C1` | 当前 block 的 `c1` 输出 tile |
| `0x81` | `LS_A` | branch 累积 add scratch A |
| `0x82` | `LS_B` | branch 累积 add scratch B |
| `0x83` | `LS_TMP` | 当前 branch 临时输出 |

冻结规则：

1. `LS_C1` 使用 BRAM tile buffer
2. `LS_A / LS_B / LS_TMP` 使用 BRAM scratch
3. scratch 不允许溢出到 DDR

## 4. opcode-specific param_id 冻结

### 4.1 CONV param_id

| Param ID | Logical Op |
|---|---|
| `0` | `LEVEL1_CONV` |
| `1` | `LEVEL2_0_C1` |
| `2` | `LEVEL2_0_D1` |
| `3` | `LEVEL2_0_D2` |
| `4` | `LEVEL2_0_D4` |
| `5` | `LEVEL2_0_D8` |
| `6` | `LEVEL2_0_D16` |
| `7` | `LEVEL2_B0_C1` |
| `8` | `LEVEL2_B0_D1` |
| `9` | `LEVEL2_B0_D2` |
| `10` | `LEVEL2_B0_D4` |
| `11` | `LEVEL2_B0_D8` |
| `12` | `LEVEL2_B0_D16` |
| `13` | `LEVEL3_0_C1` |
| `14` | `LEVEL3_0_D1` |
| `15` | `LEVEL3_0_D2` |
| `16` | `LEVEL3_0_D4` |
| `17` | `LEVEL3_0_D8` |
| `18` | `LEVEL3_0_D16` |
| `19` | `LEVEL3_B0_C1` |
| `20` | `LEVEL3_B0_D1` |
| `21` | `LEVEL3_B0_D2` |
| `22` | `LEVEL3_B0_D4` |
| `23` | `LEVEL3_B0_D8` |
| `24` | `LEVEL3_B0_D16` |
| `25` | `CLASSIFIER` |

### 4.2 AFFINE param_id

| Param ID | Logical Op |
|---|---|
| `0` | `B1_AFFINE` |
| `1` | `LEVEL2_0_AFFINE` |
| `2` | `LEVEL2_B0_AFFINE` |
| `3` | `B2_AFFINE` |
| `4` | `LEVEL3_0_AFFINE` |
| `5` | `LEVEL3_B0_AFFINE` |
| `6` | `B3_AFFINE` |

### 4.3 ADD param_id

| Param ID | Logical Op |
|---|---|
| `0` | `L20_ADD_1` |
| `1` | `L20_ADD_2` |
| `2` | `L20_ADD_3` |
| `3` | `L2B0_ADD_1` |
| `4` | `L2B0_ADD_2` |
| `5` | `L2B0_ADD_3` |
| `6` | `L2B0_RES_ADD` |
| `7` | `L30_ADD_1` |
| `8` | `L30_ADD_2` |
| `9` | `L30_ADD_3` |
| `10` | `L3B0_ADD_1` |
| `11` | `L3B0_ADD_2` |
| `12` | `L3B0_ADD_3` |
| `13` | `L3B0_RES_ADD` |

### 4.4 POOL param_id

| Param ID | Logical Op |
|---|---|
| `0` | `POOL_B1` |
| `1` | `POOL_B2_TMP` |
| `2` | `POOL_B2_OUT` |

## 5. scale domain 冻结

### 5.1 中间层没有 float dequant

- `QuantStub` 只在输入
- `DeQuantStub` 只在最终输出之后
- `UOP 1 ~ 72` 全部发生在 INT8/INT32 域

### 5.2 add 规则
当前 Encoder 的 14 个 `ADD` uop 全部冻结为：

1. `requant_bypass = 1`
2. `src_scale_a = src_scale_b = dst_scale`
3. `param_id` 仍必须有效，用于 bit-exact 验证和导出器一致性

### 5.3 concat 规则
以下 concat 目标 tensor 都只允许一个目标 scale：

- `T_B1_CAT`
- `T_L20_CAT`
- `T_L2B0_CAT`
- `T_B2_CAT`
- `T_L30_CAT`
- `T_L3B0_CAT`
- `T_B3_CAT`

所有 producer 必须在写入这些 tensor 前已经完成 requant。

为避免 B2/B3 前重复保存大 tensor，`T_L2B0_CAT/ACT` 固定为 `T_B2_CAT/ACT c[0:63]` 的 view，`T_L3B0_CAT/ACT` 固定为 `T_B3_CAT/ACT c[128:255]` 的 view。对应的 concat `STORE` 只做依赖和边界确认，不再复制整张 2 MB 级 feature map。

### 5.4 pool 规则
固定 3 条：

1. `T_POOL1` 输出 scale = `scale(T_B1_CAT)`
2. `T_POOL_TMP` 输出 scale = `scale(T_INPUT)`
3. `T_POOL2` 输出 scale = `scale(T_B2_CAT)`

## 6. RUN 阶段执行顺序冻结

### 6.1 Stage 0: 输入、B1 与提前 pool2

| UOP | Opcode | Src | Dst | Param | Note |
|---|---|---|---|---|---|
| `0` | `LOAD_FM` | DDR | `T_INPUT` | `-` | 载入输入帧 |
| `1` | `POOL` | `T_INPUT` | `T_POOL1` | `POOL:0` | `3x3 s2`，输出 scale = `T_B1_CAT` |
| `2` | `CONV` | `T_INPUT` | `T_B1_CAT` | `CONV:0` | conv post 内完成 requant/relu，写 `c[0:15]` |
| `3` | `STORE` | `T_POOL1` | `T_B1_CAT` | `-` | `concat_mode=1`，写 `c[16:18]` |
| `4` | `AFFINE` | `T_B1_CAT` | `T_B1_ACT` | `AFFINE:0` | `alias_enable=1` |
| `5` | `POOL` | `T_INPUT` | `T_POOL_TMP` | `POOL:1` | 第一次 `avgpool`，保留输入 scale；执行后输入仍可被释放 |
| `6` | `POOL` | `T_POOL_TMP` | `T_POOL2` | `POOL:2` | 第二次 `avgpool`，输出 scale = `T_B2_CAT`；覆盖已失效的 `T_POOL1` 区域 |

### 6.2 Stage 1: Level2_0

| UOP | Opcode | Src | Dst | Param | Note |
|---|---|---|---|---|---|
| `7` | `CONV` | `T_B1_ACT` | `LS_C1` | `CONV:1` | `3x3 s2 19->12` |
| `8` | `CONV` | `LS_C1` | `T_L20_CAT` | `CONV:2` | 写 `c[0:15]` |
| `9` | `CONV` | `LS_C1` | `LS_A` | `CONV:3` | `D2` 分支 |
| `10` | `STORE` | `LS_A` | `T_L20_CAT` | `-` | `concat_mode=1`，写 `c[16:27]` |
| `11` | `CONV` | `LS_C1` | `LS_TMP` | `CONV:4` | `D4` 分支 |
| `12` | `ADD` | `LS_A + LS_TMP` | `LS_B` | `ADD:0` | `requant_bypass=1` |
| `13` | `STORE` | `LS_B` | `T_L20_CAT` | `-` | `concat_mode=1`，写 `c[28:39]` |
| `14` | `CONV` | `LS_C1` | `LS_TMP` | `CONV:5` | `D8` 分支 |
| `15` | `ADD` | `LS_B + LS_TMP` | `LS_A` | `ADD:1` | `requant_bypass=1` |
| `16` | `STORE` | `LS_A` | `T_L20_CAT` | `-` | `concat_mode=1`，写 `c[40:51]` |
| `17` | `CONV` | `LS_C1` | `LS_TMP` | `CONV:6` | `D16` 分支 |
| `18` | `ADD` | `LS_A + LS_TMP` | `LS_B` | `ADD:2` | `requant_bypass=1` |
| `19` | `STORE` | `LS_B` | `T_L20_CAT` | `-` | `concat_mode=1`，写 `c[52:63]` |
| `20` | `AFFINE` | `T_L20_CAT` | `T_L20_ACT` | `AFFINE:1` | `alias_enable=1` |

### 6.3 Stage 2: Level2_Block0

| UOP | Opcode | Src | Dst | Param | Note |
|---|---|---|---|---|---|
| `21` | `CONV` | `T_L20_ACT` | `LS_C1` | `CONV:7` | `1x1 64->12` |
| `22` | `CONV` | `LS_C1` | `T_L2B0_CAT` | `CONV:8` | 写入 B2 容器 view 的 `c[0:15]` |
| `23` | `CONV` | `LS_C1` | `LS_A` | `CONV:9` | `D2` 分支 |
| `24` | `STORE` | `LS_A` | `T_L2B0_CAT` | `-` | `concat_mode=1`，写入 B2 view 的 `c[16:27]` |
| `25` | `CONV` | `LS_C1` | `LS_TMP` | `CONV:10` | `D4` 分支 |
| `26` | `ADD` | `LS_A + LS_TMP` | `LS_B` | `ADD:3` | `requant_bypass=1` |
| `27` | `STORE` | `LS_B` | `T_L2B0_CAT` | `-` | `concat_mode=1`，写入 B2 view 的 `c[28:39]` |
| `28` | `CONV` | `LS_C1` | `LS_TMP` | `CONV:11` | `D8` 分支 |
| `29` | `ADD` | `LS_B + LS_TMP` | `LS_A` | `ADD:4` | `requant_bypass=1` |
| `30` | `STORE` | `LS_A` | `T_L2B0_CAT` | `-` | `concat_mode=1`，写入 B2 view 的 `c[40:51]` |
| `31` | `CONV` | `LS_C1` | `LS_TMP` | `CONV:12` | `D16` 分支 |
| `32` | `ADD` | `LS_A + LS_TMP` | `LS_B` | `ADD:5` | `requant_bypass=1` |
| `33` | `STORE` | `LS_B` | `T_L2B0_CAT` | `-` | `concat_mode=1`，写入 B2 view 的 `c[52:63]` |
| `34` | `ADD` | `T_L2B0_CAT + T_L20_ACT` | `T_L2B0_CAT` | `ADD:6` | residual add，`requant_bypass=1`，目标仍为 B2 view |
| `35` | `AFFINE` | `T_L2B0_CAT` | `T_L2B0_ACT` | `AFFINE:2` | `alias_enable=1` |

### 6.4 Stage 3: B2

| UOP | Opcode | Src | Dst | Param | Note |
|---|---|---|---|---|---|
| `36` | `STORE` | `T_L2B0_ACT` | `T_B2_CAT` | `-` | `concat_mode=1`，写 `c[0:63]`；源已是 B2 view，可 no-op |
| `37` | `STORE` | `T_L20_ACT` | `T_B2_CAT` | `-` | `concat_mode=1`，写 `c[64:127]` |
| `38` | `STORE` | `T_POOL2` | `T_B2_CAT` | `-` | `concat_mode=1`，写 `c[128:130]` |
| `39` | `AFFINE` | `T_B2_CAT` | `T_B2_ACT` | `AFFINE:3` | `alias_enable=1` |

### 6.5 Stage 4: Level3_0

| UOP | Opcode | Src | Dst | Param | Note |
|---|---|---|---|---|---|
| `40` | `CONV` | `T_B2_ACT` | `LS_C1` | `CONV:13` | `3x3 s2 131->25` |
| `41` | `CONV` | `LS_C1` | `T_L30_CAT` | `CONV:14` | 写 `c[0:27]` |
| `42` | `CONV` | `LS_C1` | `LS_A` | `CONV:15` | `D2` 分支 |
| `43` | `STORE` | `LS_A` | `T_L30_CAT` | `-` | `concat_mode=1`，写 `c[28:52]` |
| `44` | `CONV` | `LS_C1` | `LS_TMP` | `CONV:16` | `D4` 分支 |
| `45` | `ADD` | `LS_A + LS_TMP` | `LS_B` | `ADD:7` | `requant_bypass=1` |
| `46` | `STORE` | `LS_B` | `T_L30_CAT` | `-` | `concat_mode=1`，写 `c[53:77]` |
| `47` | `CONV` | `LS_C1` | `LS_TMP` | `CONV:17` | `D8` 分支 |
| `48` | `ADD` | `LS_B + LS_TMP` | `LS_A` | `ADD:8` | `requant_bypass=1` |
| `49` | `STORE` | `LS_A` | `T_L30_CAT` | `-` | `concat_mode=1`，写 `c[78:102]` |
| `50` | `CONV` | `LS_C1` | `LS_TMP` | `CONV:18` | `D16` 分支 |
| `51` | `ADD` | `LS_A + LS_TMP` | `LS_B` | `ADD:9` | `requant_bypass=1` |
| `52` | `STORE` | `LS_B` | `T_L30_CAT` | `-` | `concat_mode=1`，写 `c[103:127]` |
| `53` | `AFFINE` | `T_L30_CAT` | `T_L30_ACT` | `AFFINE:4` | `alias_enable=1` |

### 6.6 Stage 5: Level3_Block0

| UOP | Opcode | Src | Dst | Param | Note |
|---|---|---|---|---|---|
| `54` | `CONV` | `T_L30_ACT` | `LS_C1` | `CONV:19` | `1x1 128->25` |
| `55` | `CONV` | `LS_C1` | `T_L3B0_CAT` | `CONV:20` | 写入 B3 容器 view 的 `c[128:155]` |
| `56` | `CONV` | `LS_C1` | `LS_A` | `CONV:21` | `D2` 分支 |
| `57` | `STORE` | `LS_A` | `T_L3B0_CAT` | `-` | `concat_mode=1`，写入 B3 view 的 local `c[28:52]` |
| `58` | `CONV` | `LS_C1` | `LS_TMP` | `CONV:22` | `D4` 分支 |
| `59` | `ADD` | `LS_A + LS_TMP` | `LS_B` | `ADD:10` | `requant_bypass=1` |
| `60` | `STORE` | `LS_B` | `T_L3B0_CAT` | `-` | `concat_mode=1`，写入 B3 view 的 local `c[53:77]` |
| `61` | `CONV` | `LS_C1` | `LS_TMP` | `CONV:23` | `D8` 分支 |
| `62` | `ADD` | `LS_B + LS_TMP` | `LS_A` | `ADD:11` | `requant_bypass=1` |
| `63` | `STORE` | `LS_A` | `T_L3B0_CAT` | `-` | `concat_mode=1`，写入 B3 view 的 local `c[78:102]` |
| `64` | `CONV` | `LS_C1` | `LS_TMP` | `CONV:24` | `D16` 分支 |
| `65` | `ADD` | `LS_A + LS_TMP` | `LS_B` | `ADD:12` | `requant_bypass=1` |
| `66` | `STORE` | `LS_B` | `T_L3B0_CAT` | `-` | `concat_mode=1`，写入 B3 view 的 local `c[103:127]` |
| `67` | `ADD` | `T_L3B0_CAT + T_L30_ACT` | `T_L3B0_CAT` | `ADD:13` | residual add，`requant_bypass=1`，目标仍为 B3 view |
| `68` | `AFFINE` | `T_L3B0_CAT` | `T_L3B0_ACT` | `AFFINE:5` | `alias_enable=1` |

### 6.7 Stage 6: B3 + Classifier

| UOP | Opcode | Src | Dst | Param | Note |
|---|---|---|---|---|---|
| `69` | `STORE` | `T_L30_ACT` | `T_B3_CAT` | `-` | `concat_mode=1`，写 `c[0:127]` |
| `70` | `STORE` | `T_L3B0_ACT` | `T_B3_CAT` | `-` | `concat_mode=1`，写 `c[128:255]`；源已是 B3 view，可 no-op |
| `71` | `AFFINE` | `T_B3_CAT` | `T_B3_ACT` | `AFFINE:6` | `alias_enable=1` |
| `72` | `CONV` | `T_B3_ACT` | `T_OUT` | `CONV:25` | `CLASSIFIER` |
| `73` | `STORE` | `T_OUT` | DDR | `-` | 写回输出 |
| `74` | `END` | `-` | `-` | `-` | 结束 |

## 7. 最终实现约束

### 7.1 alias 规则
`CAT -> ACT` 物理 alias **不是可选优化**，而是首版 final spec 的默认实现：

1. `T_B1_CAT` 与 `T_B1_ACT` 同址
2. `T_L20_CAT` 与 `T_L20_ACT` 同址
3. `T_L2B0_CAT` 与 `T_L2B0_ACT` 同址
4. `T_B2_CAT` 与 `T_B2_ACT` 同址
5. `T_L30_CAT` 与 `T_L30_ACT` 同址
6. `T_L3B0_CAT` 与 `T_L3B0_ACT` 同址
7. `T_B3_CAT` 与 `T_B3_ACT` 同址

### 7.2 不允许的实现偏差
以下做法均视为偏离 final spec：

1. 把 `T_POOL_TMP` 再次并回 `T_POOL1`
2. 把任一 `ADD` 改成运行时 requant add
3. 把 `STORE/CONCAT` 改成先搬运到 DDR 再拼接
4. 关闭 `CAT -> ACT` alias 后又不同步修改 tensor descriptor
5. 改动本表中的 tensor id、param_id、channel offset 或 bank/base 映射

## 8. 编码顺序
实现顺序仍建议按以下节奏推进：

1. `UOP 0 ~ 6`：输入、B1、提前 pool2
2. `UOP 7 ~ 20`：Level2_0
3. `UOP 21 ~ 35`：Level2_Block0
4. `UOP 36 ~ 39`：B2 concat/affine
5. `UOP 40 ~ 74`：Level3、B3、Classifier

但无论编码顺序如何，最终行为都必须以本表为准。

## 9. 导出器一致性检查
当前唯一有效导出器为 `D:\ESP_INT8\tools\export_int8_hw_blob.py`。导出的 `param_blob.bin` 必须满足：

1. `uop_count = 75`
2. `tensor_desc_count = 19`
3. `scale_desc_count = 55`
4. `uop[5] = POOL T_INPUT -> T_POOL_TMP`
5. `uop[6] = POOL T_POOL_TMP -> T_POOL2`
6. `uop[36] = STORE T_L2B0_ACT -> T_B2_CAT, c_offset=0`
7. `uop[37] = STORE T_L20_ACT -> T_B2_CAT, c_offset=64`
8. `uop[38] = STORE T_POOL2 -> T_B2_CAT, c_offset=128`
9. `uop[70] = STORE T_L3B0_ACT -> T_B3_CAT, c_offset=128`
