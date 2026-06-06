# P6 HLS 架构升级阶段复盘与下一步计划 (2026-06-06)

## 1. 当前结论

最近几轮 P6 方向的改动说明：**固定 stage 调度本身不是问题，真正导致综合时间不可控的是“固定 stage 调度 + 旧版通用 UOP/operator/memory 框架混用”**。当 `static_espnet_scheduler` 仍然通过大块 dispatch 函数调用旧 `execute_conv_uop / pool / add / affine / concat` 路径时，HLS 需要在一个巨大的控制/存储访问图里同时分析旧全局 memory、scratch、URAM/BRAM 读写、window 生成和 SA 数据流，综合时间会迅速失控。

最新一次能在约 15min 内综合和 package 的版本，说明经过拆分和降复杂度后可综合性已恢复；但它仍不是完整 P6。报告中 `static_espnet_scheduler` 仍包含 75 次 UOP 循环和 `execute_static_uop_dispatch`，只能视为“静态 UOP 兼容版”，不能作为 100ms 级 P6 架构验收。

## 2. 最近尝试的成功经验

| 方向 | 结果 | 经验 |
|---|---|---|
| 回到 100MHz 目标 | 正确 | P6 当前优先压总 cycles，不应继续用 125/150MHz 作为主要矛盾 |
| 去掉 profiling/debug 主线依赖 | 正确 | 主线 HLS 代码必须先保证干净、可综合、可实现 |
| 拆出 `conv_store.cpp` / `scratch_mgr.cpp` / `upsample_unit.cpp` | 有价值 | 能降低 `int8_core.cpp` 的结构混乱度，也便于后续替换旧 operator |
| 保留顶层 AXI 接口和 `PARAM.BIN` | 正确 | 不增加 PS/app/platform 侧不必要风险 |
| 当前兼容版 csynth/package 通过 | 有价值但不是终点 | 最新 HLS：target `10ns`，estimated `8.756ns`，BRAM `1439/1488`，DSP `1228/3528`，URAM `112/112`，LUT HLS 估计仍高 |
| 使用 audit 脚本检查综合报告 | 必须保留 | 能及时发现 `HLS 214-475`、`HLS 200-975`、II 退化、stream depth 等风险 |

## 3. 最近失败或低效尝试

| 尝试 | 现象 | 根因判断 |
|---|---|---|
| 直接把 `static_espnet_scheduler` 改成大块固定 stage 调用 | 多次卡在 `static_espnet_scheduler` 或其下游模块调度/绑定阶段 | stage 函数过大，且仍间接包含旧 generic operator、memory access、scratch alias，HLS 搜索空间爆炸 |
| 在旧 memory 框架上修 `read_phys_word/write_fmbuf_*` | 出现语法/namespace 错误，修复后仍容易卡在 URAM 相关 warning 附近 | 旧 memory 层抽象太动态，URAM/BRAM 分支、packed cross-word RMW 和 alias 描述符让调度器难以收敛 |
| 试图一次性合并 P6A/B/C/D/E | CSim 或 csynth 代价不可控，定位困难 | 结构跨度太大，无法区分功能错误、综合器搜索爆炸、资源爆炸三类问题 |
| 继续在兼容 UOP dispatch 中局部 patch | 能综合，但性能意义有限 | 没有消除逐 UOP memory pass，本质瓶颈仍在 |
| 保留复杂动态 `win_gen` 与 shape-specific 新路径并存 | 综合时间和 LUT 风险上升 | HLS 会同时分析大量不可达或低频分支，dead/generic path 不能依赖工具自动消除 |

## 4. 当前代码状态

当前主线代码的可综合性比前几轮好，但架构上仍处于过渡状态：

- `static_espnet_scheduler` 仍然是 `for uop_idx < UOP_COUNT_ENCODER` 的静态 UOP 循环。
- `execute_static_uop_dispatch` 仍是主资源消耗模块。
- `execute_conv_uop / execute_pool_uop / execute_add_uop` 等旧 operator 仍在性能路径内。
- `memory.cpp` 仍保留全局 feature memory、scratch/alias/tensor desc 等旧式访问模型。
- `win_gen.cpp` 仍包含多种 generic / shape-specific 混合路径。
- `upsample_unit.cpp` 已独立，是后续保留模块。

因此当前版本可以作为“可综合安全点”，但不能宣称完成 P6。

## 5. 后续执行原则

为避免再次陷入死循环，下一轮必须遵守以下规则：

1. 不再写“巨大 monolithic static scheduler”。每个 stage/block 必须是小而确定的 HLS 函数。
2. 不在同一轮同时改调度、memory、SA、line buffer、upsample。每轮只交付一个可综合结构变化。
3. 新 P6 性能路径不能调用旧 `execute_*_uop`。可以保留旧路径做 fallback，但必须通过编译开关隔离，不能进入默认综合主路径。
4. 每个 P6 子模块先做轻量 CSim 或 prefix CSim，再做 top csynth。
5. 若 csynth 超过 fullres100 经验时间 2-3 倍且卡在同一 module，无需等到 9h，直接中断并拆小模块。
6. 所有 shape-specific fast path 必须删除或隔离对应 generic dead path，不能依赖 HLS 自动裁剪。
7. 对 URAM 大数组只允许 stage/block boundary 访问；block 内 row/branch/line buffer 尽量用局部 BRAM/LUTRAM。
8. 每次综合后必须记录：综合时间、top estimated clock、BRAM/DSP/LUT/URAM、是否仍有 75-UOP dispatch、主要 high-risk warning。

## 6. 下一步改进方案

### P6A-rerun: 干净静态调度骨架

目标不是立刻提速，而是建立一个不会综合爆炸的 P6 主路径。

要做：

1. 保留当前可综合 UOP 兼容版作为 fallback。
2. 新增默认 P6 path，但只包含固定 stage 函数壳：
   - `run_stem_stage()`
   - `run_level2_stage()`
   - `run_level3_stage()`
   - `run_classifier_stage()`
   - `run_fullres_output_stage()`
3. 初始每个 stage 内只调用少量已验证安全的小函数，不能把旧 `execute_static_uop_dispatch` 搬进去。
4. top 报告中必须不再出现 `static_espnet_scheduler` 下的 75-trip UOP loop，才算 P6A 骨架通过。

验收：

- CSim 先可输出 deterministic mask，允许与 full100 不 bit-exact。
- csynth 在 30-45min 内完成。
- HLS 报告无 `HLS 214-475` / `HLS 200-975`。

### P6B: 只替换最高收益的 memory/window 路径

目标是先证明结构性 cycles 下降，不追完整融合。

要做：

1. 第一层 C3 和 level2 输入改为固定 line-buffer path。
2. 不再从 `on_chip_memory_read_packed_contiguous` 重复取 3x3 window。
3. level2 输出仍可先落 stage boundary memory，避免一次性做 block fusion。

验收：

- `win_gen` 报告中高频路径不再依赖旧 generic packed-tile RMW。
- top synthesis 时间仍可控。
- 上板或计数器能看到 window/memory 访问下降。

### P6C: 局部 block fusion

目标是先融合一个 level2 ESP block，而不是全网一次性融合。

要做：

1. 选一个 level2 block 固定实现 `branch conv -> requant -> concat/add -> block output`。
2. branch 输出使用局部 row buffer，不写 scratch tensor。
3. 该 block 的 `STORE/CONCAT/ADD` 不再作为独立 pass。

验收：

- 单 block CSim 通过。
- 该 block 的 HLS 子模块可独立综合。
- 再接入 top。

### P6D/P6E 暂缓

原生 `2x16x32` SA 和最终 timing tuning 暂缓到 P6B/C 可控后再做。当前最重要的是先删除旧 UOP memory pass，而不是继续改 SA。

## 7. 下一次工作入口

下一次不要继续在当前 `execute_static_uop_dispatch` 上局部优化。建议从 `int8_core.cpp` 建立 P6 clean path 开始，同时用宏隔离：

```cpp
#if ESP_INT8_USE_P6_STATIC_PATH
  err = run_p6_static_graph(gmem_frame_out);
#else
  err = static_espnet_scheduler(gmem_frame_out);
#endif
```

默认综合先指向 P6 clean path；旧路径只作为 fallback/reference。只有当 HLS 报告确认默认路径不再综合 75-UOP dispatch 后，才继续做 P6B。

