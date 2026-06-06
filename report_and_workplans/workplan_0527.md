# ESPNet_Encoder INT8 NPU 工作进展与计划 (2026-05-27)

## 1. 今日目标

今天从 `PERF125-NOPROF` 单路 stream 基线回退出发，放弃此前综合时间不可控的性能实验分支，转向补齐 full-resolution 输出能力。目标是让 NPU 不再只输出 `64x128` encoder mask/logits，而是在 PL 侧完成 `2-channel logits bilinear upsample -> argmax`，直接输出 `512x1024` 的最终二分类 mask。

## 2. 已完成工作

### 2.1 HLS 侧 full-resolution upsample 接入

已在 HLS 代码中加入 full-resolution mask 输出路径：

1. 新增 `upsample_unit.cpp`，实现从 `64x128x2` INT8 logits 到 `512x1024` mask 的固定 8x bilinear upsample + argmax。
2. `frame_dma_store` 改为调用上采样输出路径，最终写出 `512*1024` bytes mask。
3. 顶层 `m_axi` output depth 和配置常量同步扩展到 `16384` 个 256-bit word。
4. 保持主 NPU conv/pool/add/affine/store 数据流边界不变，避免重新引入 pair2/dual path 或 profiling/debug 逻辑。

验证结果：

| 项目 | 结果 |
|---|---|
| HLS CSim | `top_tb` 通过，`0 errors` |
| 上采样坐标/权重 | 已用随机 INT logits 对齐 PyTorch `bilinear, align_corners=False -> argmax` 行为 |
| HLS synthesis | 通过，无 `HLS 214-475` / `HLS 200-975` |

### 2.2 实现与平台导出

当前 full-resolution 版本以 `100 MHz` 作为 timing-clean 平台导出，对外命名为：

| 项目 | 结果 |
|---|---|
| platform | `platform_full100_0527` |
| app tag | `INT8-BOARD-20260527-FULL100-UPFULL-VAL` |
| PL clock | `100 MHz` |
| routed timing | WNS `+0.073 ns`，TNS `0`，hold/pulse width 无失败端点 |
| 资源 | LUT `73.35%`，FF `25.39%`，BRAM Tile `86.02%`，URAM `100%`，DSP `39.77%` |

判断：这一版实现不是 125 MHz 性能版，而是 full-resolution 功能闭环的 100 MHz timing-clean 验收版。最紧路径仍在 `s_uram/window/memory` 相关布线，route delay 占比接近 90% 以上；新增 upsample 本身不是 critical path。

### 2.3 PS app 与验证集测试

已将 Vitis app 对齐到新平台：

1. `vitis-comp.json`、`app.yaml`、CMake cache 均指向 `platform_full100_0527`。
2. `launch.json` 已改为使用新平台 bitstream、FSBL 和 `psu_init.tcl`。
3. app 输出配置改为 full-resolution mask：`MASK.BIN` / `Oxxxx.BIN`，单文件 `512*1024` bytes。
4. app 切换到完整验证集模式，输入 `I0000.BIN` 到 `I0499.BIN`，输出 `O0000.BIN` 到 `O0499.BIN`。

上板验证集结果：

```text
samples=500
total_cycles=20894087956
total_ms=208940
avg_cycles=41788175
avg_ms=417
min_ms=417
max_ms=417
```

SD 卡输出检查：

| 项目 | 结果 |
|---|---|
| 输出文件 | `O0000.BIN` 到 `O0499.BIN` |
| 文件数量 | `500` |
| 单文件大小 | `524288` bytes |
| 尺寸异常 | `0` |

## 3. 精度结果

已使用 `tools/eval_val_hw_masks_fullres.py` 对板端 full-resolution mask 进行评估。脚本也已修正导入路径初始化问题，后续可直接运行，不再需要手动设置 `PYTHONPATH`。

板端 full-resolution mask 指标：

| 指标 | 结果 |
|---|---|
| PA | `0.97800421` |
| mIoU | `0.86356491` |
| per-class acc | `[0.81670988, 0.99223832]` |
| per-class IoU | `[0.75068569, 0.97644412]` |

与软件侧 full-resolution baseline 对比：

| 评估方式 | PA | mIoU |
|---|---:|---:|
| software fullres bilinear logits argmax | `0.97854548` | `0.86874892` |
| software fullres nearest mask | `0.97718681` | `0.86281516` |
| board fullres mask | `0.97800421` | `0.86356491` |

判断：板端 full-resolution 输出格式和整体精度有效。当前 mIoU 比软件 float bilinear-logits baseline 低约 `0.00518`，但略高于 nearest-mask baseline。若后续要严格证明上采样单元 bit-exact，需要导出一版与硬件 fixed-point INT8 upsample 完全一致的 golden，而不能只和 PyTorch float bilinear 结果比较。

## 4. 当前判断

1. full-resolution 输出链路已经闭合：NPU 可直接输出 `512x1024` 二分类 mask，并完成 500 张验证集上板评估。
2. `platform_full100_0527` 是当前 full-resolution 功能验收平台，时序 clean，但频率只有 `100 MHz`。
3. 端到端推理时间从此前低分辨率 mask 输出约 `399 ms` 增加到约 `417 ms`，新增 full-resolution upsample 和更大 output DMA 写回带来约 `18 ms` 级开销。
4. 当前性能瓶颈仍主要在原 NPU 主数据流和 `s_uram/window/memory` 布线，不在 upsample 算法本身。
5. URAM 已满，后续优化不能依赖新增大规模片上 buffer。

## 5. 路线修正与下一步计划

当前板端 full-resolution 结果已经没有显著精度下降，后续不再把 full-resolution upsample 的 bit-exact 作为主线目标。下一阶段目标调整为两条并行路线：

1. 频率目标：从当前 `100 MHz timing-clean` 向 `125 MHz clean`、再向 `150 MHz` 收敛。
2. 周期目标：在不牺牲现有精度和功能闭环的前提下，减少 `MODE_RUN` 的真实端到端周期数。

已开始第一步低风险周期优化：`upsample_unit.cpp` 增加低分辨率 logits 行缓存，避免 512 个 full-resolution 输出行重复读取相同 `64x128x2` logits 行。该优化不新增大 buffer，不改变上采样数学行为；`top_tb` CSim 已通过，`0 errors`。下一步需要跑 HLS synthesis，检查 `frame_dma_store/upsample` latency 是否下降，以及是否引入新的 II、资源或 timing 风险。

后续计划：

1. 先用当前补丁跑 HLS synthesis，重点审查 `upsample_logits_bilinear_argmax_store`、`frame_dma_store`、top LUT/BRAM/URAM、estimated clock 和 audit warning。
2. 若综合健康，推进一版 `P5A-UPFULL-CYC` 实现，上板确认 full-val 平均延迟是否低于 `417 ms`。
3. 150 MHz 不能直接硬冲：当前 100 MHz routed WNS 只有 `+0.073 ns`，最差路径仍集中在 `s_uram/window/memory`，route delay 占比接近 90% 以上。下一步频率收敛应先做到 `125 MHz timing-clean`，再评估 `150 MHz`。
4. 频率优化重点不在 upsample，而在 `s_uram` 访问路径、`window_generator_row -> URAM ADDR` 路径、`s_bram -> ADD/STORE tile` 路径和 DSP pipeline warning。
5. 周期优化重点继续放在不复制主数据流的局部优化：upsample 行复用、non-conv 固定 shape 专用路径、轻量 uop fusion、window path 固定 shape 分支。
6. 后续报告中必须区分 APU timer tick、PL cycle 和真实墙钟时间，避免把频率提升误判为数据流周期下降。

## 6. 近期频率收敛尝试与经验修正

### 6.1 已尝试路线

在 full-resolution 功能闭环后，连续尝试了几轮 125/150 MHz 方向的性能收敛：

1. 150 MHz 方向曾将 HLS 目标周期压到约 `6.56 ns`，但综合长时间卡在 `emit_window_row_1x1_fast` / window 相关循环调度，属于 HLS scheduling search 爆炸，不适合作为当前主线。
2. 回退后尝试 8 ns HLS package 并跑完整 Vivado 实现，布线完成且无 routing error，但 routed timing 未过。
3. 8 ns 实现报告显示 `clk_pl_0=125 MHz`，WNS `-1.368 ns`，TNS 约 `-28819 ns`，失败端点约 `45011`。
4. 资源层面 CLB 使用率约 `96.3%`，BRAM Tile 约 `86.0%`，URAM `100%`，说明时序压力来自高资源密度和长距离布线，而不是单个简单组合表达式。

### 6.2 报告暴露的主要瓶颈

1. worst 10 setup path 主要集中在 `window_generator_row`，尤其是 `emit_window_row_3x3_c19_stride2_fast` 和 `emit_window_row_3x3_smallc_stride1_reuse`。
2. 典型最差路径为 `s_bram_U/RAMB36E2 -> spatial_word regs`，data path 约 `9.18 ns`，其中 route delay 约 `5.9 ns`，占比超过 60%。
3. methodology 采样中还大量出现 `systolic_array_core_row`、`execute_affine_uop`、`execute_add_uop` 的 TIMING-16，说明 BRAM/URAM 到 window/SA/non-conv 逻辑的长路径是系统性问题。
4. 单纯更换 Vivado strategy 很难稳定修复 `-1.3 ns` 级别、数万端点的 setup 失败；需要先在 HLS 结构上减少长路径和大扇出。

### 6.3 本轮 HLS 改进原则

本轮不再做会显著扩大调度搜索空间的优化，不新增第二套 stream/SA，不复制大规模片上 buffer。优先做三类低搜索复杂度修补：

1. `win_gen`：将 c19 stride2 内层路径从“读 row segment -> 拆成 `spatial_word[9]` -> 再拼流”改为“row segment 直接拼 `act_stream` word”，减少 fully-partitioned spatial array 和大 mux 带来的 BRAM 到寄存器长路径。
2. `sa_core`：将 MAC lane 内的 `oc < out_c`、`k_idx < k_total` 判断下沉为 tile 级 `valid_tm/valid_tk`，降低 DSP 前级比较和控制扇出。
3. `memory`：对主 feature-map URAM/BRAM 显式使用 `latency=2`，尝试在 RAM 输出侧增加时序余量，代价是需要综合后重新检查周期和 II。

### 6.4 下一步验证门槛

1. 先跑 full top CSim，必须保持 `0 errors`。
2. 再跑 8 ns HLS synthesis，重点检查是否重新出现综合时间爆炸、`HLS 214-475`、`HLS 200-975`、异常 II 退化或资源异常增长。
3. 若 HLS 报告健康，再 package 并跑 125 MHz Vivado 实现；只有 routed timing clean 才进入上板验证。
4. 若 8 ns 仍无法收敛，则保留本轮结构修补，降到 10 ns 形成稳定版；不要继续在不可控综合分支上叠加优化。
