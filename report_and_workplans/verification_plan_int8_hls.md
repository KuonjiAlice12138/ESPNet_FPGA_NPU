# INT8 HLS Verification Plan

## 1. Current Decision

从 2026-05-10 起，真实尺寸 C/RTL co-sim 不再作为本项目验证主线，也不再作为 package、Vivado system 实现或上板前的阻塞门槛。

原因很明确：完整 `m_axi` 顶层 co-sim 容易被 AXI VIP/wrapper 成本干扰；`COSIM_LITE` 虽然去掉了 AXI VIP，但真实尺寸 U02 第一层卷积运行到约 `1.86%` 已耗费过长时间。继续尝试用 C/RTL co-sim 覆盖真实 UOP 序列，时间收益比不可接受。

当前默认 `hls_config.cfg` 已恢复为最终上板 AXI 配置，即 `m_axi + s_axilite` 顶层，不再是 `COSIM_LITE` 临时配置。

## 2. Verification Roles

| 路径 | 后续定位 |
| --- | --- |
| 板端 debug app | 主验证路径，覆盖真实 PS/PL/DDR/SD 闭环 |
| C sim | 保留，检查功能流程和输出数据生成 |
| 模块级小规模 TB | 保留，必要时验证局部算法或循环边界 |
| 真实尺寸 C/RTL co-sim | 放弃，不再作为验收门槛 |
| 完整 `m_axi` top co-sim | 放弃，不再用于真实数据路径定位 |

## 3. Board Debug Sequence

后续用同一套 stop-after UOP 分段在板端推进验证边界，并通过 `dbg_status/dbg_heartbeat/dbg_*_words` 定位卡点：

| Case | 目标 |
| --- | --- |
| `DBG_U00_INPUT` | 验证 input load |
| `DBG_U01_POOL1` | 验证初始 pool/input 路径 |
| `DBG_U02_C1` | 验证第一层真实卷积 |
| `DBG_U04_B1` | 验证 B1 concat/affine 输出 |
| `DBG_U20_L20` | 覆盖 Level2 输出 |
| `DBG_U34_L2_RES` | 覆盖 Level2 残差 add |
| `DBG_U39_B2` | 覆盖多路拼接融合到 B2 |
| `DBG_U53_L30` | 覆盖 Level3 前半 |
| `DBG_U67_L3_RES` | 覆盖 Level3 残差 add |
| `DBG_U70_B3CAT` | 覆盖深层多尺度拼接 |
| `DBG_U72_OUT` | 覆盖最终输出卷积 |

## 4. Debug Interpretation

若板端 timeout，优先按以下信息定位：

| 信息 | 判断依据 |
| --- | --- |
| `phase` | 当前卡在 init、decode、load、conv、dump 或 store |
| `current_uop / last_done_uop` | 判断 UOP 边界是否推进 |
| `opcode` | 判断当前算子类型 |
| `dbg_act_words` | 判断 `win_gen`/activation 生产是否推进 |
| `dbg_wgt_words` | 判断 weight 读取是否推进 |
| `dbg_psum_words` | 判断 SA 是否产生 psum |
| `dbg_out_words` | 判断 PPU/writeback 是否推进 |

若 U02 卡住，优先依据 `act/wgt/psum/out` 哪个计数停止来决定回到 `win_gen`、`param_dma/weight`、`sa_core`、`ppu` 或 memory writeback 进行定向修复。

## 5. Pass Criteria

最低通过标准调整为：

- 板端 `MODE_INIT` 稳定通过。
- 板端至少推进到 `DBG_U02_C1`，并能明确第一层卷积是否完成。
- 若 U02 通过，继续推进到 U04、U20、U34、U39、U53、U67、U70、U72。
- 每个通过的 debug case 需要保存输出 bin，并在 PC 端离线比对。
- 完整 `MODE_RUN` 只在分段 debug 足够推进后恢复。

## 6. Next Step

下一步直接进入板端验证：确认 app 使用最新版 platform/header，运行 debug bring-up，记录串口输出和生成的 `DxxOUT.BIN` 文件。若出现 timeout，依据 debug counter 定位并回到对应 HLS 模块修补。
