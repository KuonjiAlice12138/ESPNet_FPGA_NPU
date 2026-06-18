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

本节记录的是 2026-06-06 早期过渡版状态；后文第 8-12 节已经更新为当前 P6-only 状态。

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

下一次不要继续在当前 `execute_static_uop_dispatch` 上局部优化。当前代码已经转向 P6-only：`core_mode_run()` 直接调用 `run_p6_static_graph()`，旧 `static_espnet_scheduler` 不再作为 fallback/reference 保留。只有当 HLS 报告确认默认路径不再综合 75-UOP dispatch 后，才继续做 P6B/C 的融合与 memory pass 压缩。

## 8. 2026-06-06 本轮代码落实记录

本轮已先在 git 建立 checkpoint：`f850d57`，tag `p6-pre-clean-rewrite-20260606`，并已推送到远程仓库。

已落实到 HLS 代码的 P6 改动：

- `npu_config.hpp` 已不再提供 `ESP_INT8_USE_P6_STATIC_PATH` fallback 开关，P6 静态图是唯一 MODE_RUN 路径。
- `core_mode_run()` 直接进入 `run_p6_static_graph()`，不再包含 `static_espnet_scheduler()` 兼容分支。
- 新增固定 stage/block 调度粒度：`run_stem_stage()`、`run_level2_entry_stage()`、`run_level2_block0_stage()`、`run_level2_merge_stage()`、`run_level3_entry_stage()`、`run_level3_block0_stage()`、`run_level3_merge_stage()`、`run_classifier_stage()`、`run_fullres_output_stage()`。
- 默认 P6 path 显式调用固定 UOP 模板，不再使用 `for uop_idx < 75` 的 dispatch loop。
- 已知 `conv -> store` 相邻对改为静态 alias pair，先覆盖 `9->10`、`23->24`、`42->43`、`56->57`，减少独立 STORE pass。
- `window_generator_row()` 只保留 ESPNet 当前固定形状的 fast path；旧 generic window fallback 已从 HLS 源码移除。
- 新增 `tools/check_p6_hls_structure.py`，用于防止代码重新退回旧 75-UOP dispatch、旧 generic window fallback 或旧 UOP table 缓存。

已完成验证：

- `python tools/check_p6_hls_structure.py` 通过。
- `hls_config_csim_prefix_lite.cfg`，`ESP_INT8_CSIM_MAX_UOP=4` 前缀 CSim 通过，0 errors。
- 完整 top CSim 已确认完成编译/链接生成 `csim.exe`，但整网仿真运行时间较长，未作为本轮阻塞项。

下一步验收重点：

- 由 GUI 或命令行运行 top csynth，必须确认报告中不再出现 `static_espnet_scheduler` 的 75-trip loop。
- 若 csynth 再次卡住，应定位到具体 P6 stage/block 或具体 memory/helper 函数，不再回到旧 dispatch 兼容路径。
- 若综合可控，再推进更深的 P6C：把 level2 block 内 add-store pair 和 branch concat 写回继续融合，减少 scratch/global memory pass。

## 9. 2026-06-06 综合卡点补充记录

本轮 top csynth 已经越过 `run_p6_static_graph()` 和 `core_mode_run()`，说明 P6 clean path 的宏隔离生效，默认路径没有再卡在旧 `static_espnet_scheduler`。新的卡点出现在 `scratch_mgr.cpp` 的 `invalidate_global_aliases()`：该函数在 reset 路径中对 `s_global_alias_valid[MAX_TENSOR_DESC_COUNT]` 和 `s_global_alias_desc[MAX_TENSOR_DESC_COUNT]` 做循环清零，并带 `PIPELINE II=1`。HLS 在该小函数 scheduling 阶段长时间无进展。

根因判断：

- 这是 reset/alias 管理逻辑的综合风格问题，不是 P6 stage 调度本身的问题。
- 对 `tensor_desc_t` 结构体数组做循环清零会让 HLS 构造不必要的结构体写网络。
- 对 invalidation 这种控制逻辑强行 `II=1` 没有性能收益，反而扩大调度搜索空间。

修复原则：

- global alias 失效改为 epoch/tag 机制：reset 只递增全局 epoch，不再清空 64 项结构体数组。
- 单个 alias 写入时记录当前 epoch，读取时比较 epoch 判断是否有效。
- scratch 局部 4 项 valid 清除保留，但不再重置 `tensor_desc_t` 内容；valid bit 已足够定义有效性。
- 新增结构检查规则，禁止默认 P6 代码重新出现 loop-style global alias invalidation。

已落实修补：

- `scratch_mgr.cpp` 删除 `invalidate_global_aliases()`，改为 `s_global_alias_epoch_tag[] + s_global_alias_epoch`。
- `reset_scratch_state()` 不再清空 64 项 global alias 描述符；scratch 局部只清 4 个 valid bit。
- 审查所有 `PIPELINE II=1`：未发现第二个 reset/invalidator 类强制流水问题；MODE_INIT header 小循环已改为 `PIPELINE off`。

## 10. 2026-06-06 P6-only dead logic 清理记录

本轮按 P6-only 原则继续清理 HLS 源码，避免旧路径在前端分析、综合调度或人工报告审查中继续干扰。

已删除：

- `int8_core.cpp` 中的 `ESP_INT8_USE_P6_STATIC_PATH` fallback、`fetch_static_uop()`、`execute_static_uop_dispatch()`、`static_espnet_scheduler()`。
- `win_gen.cpp` 中旧 generic `emit_window_row_3x3_fast()` / `emit_window_row_1x1_fast()` fallback；P6 只保留当前 ESPNet 固定形状 fast path。
- `param_dma.cpp` 中旧 `s_uop_table[]`、`load_uop()`、`param_dma_get_uop()` 和 UOP table 加载循环；P6 只校验 `uop_count == 75`。
- `frame_dma.cpp` 中未被顶层调用的空 `frame_dma_store()`。

当前约束：

- 顶层 AXI/app ABI 不变，仍使用 `MODE_INIT/MODE_RUN`、`uop_count=75`、同一版 param blob。
- P6 静态 stage 顺序不变，不回退到旧 UOP dispatch。
- `tools/check_p6_hls_structure.py` 已升级为 P6-only 结构门禁；综合前必须通过。

## 11. 2026-06-06 硬件克隆问题修复记录

最新 csynth 报告显示 P6 固定模板调度虽然删除了旧 scheduler，但引入了更严重的问题：`run_p6_static_uop<N>` / `P6OpRunner<opcode>` 被 HLS 展开成大量独立子模块，导致 `core_mode_run` 资源估计爆炸到 DSP `25144`、LUT `7486607`，这不是实现阶段能压回来的正常超估。

根因判断：

- 固定 schedule 的方向是对的，但不能用 UOP ID 模板函数承载算子执行。
- `StaticUop<ID>::make()` 只能用于构造静态描述，不应决定硬件 datapath 特化。
- conv/add/pool/affine/store 必须从同一个 non-template dispatch 入口调用，保证 HLS 复用一套算子硬件。

已落实修补：

- 删除 `P6OpRunner`、`run_p6_static_uop<ID>`、`P6_RUN/P6_RUN_PAIR` 模板调度宏。
- `run_p6_static_graph()` 改为单一 `uop_id=1..72` 循环，按固定顺序构造静态 `uop_t`。
- 新增 `build_p6_static_uop()` 只负责生成描述，不承载算子计算。
- 新增 `run_p6_dispatch_uop()` 作为共享 non-template datapath 入口。
- 保留 `9->10`、`23->24`、`42->43`、`56->57` 的 conv/store pair 融合，但改为运行时 `uop_t` pair，不再通过模板实例化。
- `tools/check_p6_hls_structure.py` 新增硬件克隆门禁：禁止 `P6OpRunner`、`run_p6_static_uop`、`run_p6_conv_store_pair<...>`、`P6_RUN` 等 token。

已完成验证：

- 结构检查通过：`python tools/check_p6_hls_structure.py`。
- 轻量前缀 CSim 通过：`hls_config_csim_prefix_lite.cfg`，`ESP_INT8_CSIM_MAX_UOP=4`，0 errors。

下一版 csynth 验收重点：

- 报告目录中不应再生成成批 `run_p6_static_uop_*_csynth.rpt`。
- `core_mode_run` 不应再出现 DSP/LUT 数量级爆炸；若仍超资源，应定位是否由 `build_p6_static_uop()` switch 或共享 dispatch 未被复用导致。
- 若 HLS 仍复制共享 dispatch，需要进一步使用单 call-site 调度或 `ALLOCATION instances=... limit=1` 约束，而不是回退到模板 stage 调度。

## 12. 2026-06-08 P6 route congestion 修补记录

最新 Vivado 实现失败点不是 raw LUT/DSP 不够，而是 route 阶段 global congestion level 6。placed 后层次资源显示拥塞压力集中在 `execute_conv_stream_row_region`，其中 `act_stream_U` 被实现成 `fifo_w256_d32` 并消耗约 4.5k LUT，`psum_stream_U` 也落入 LUTRAM。

已落实低风险修补：

- `act_stream`、`wgt_stream`、`psum_stream` 显式 `BIND_STORAGE type=fifo impl=bram`，保持 P6 单 SA 数据流和函数边界不变。
- `act_stream` 深度从 32 提升到 64，避免 256-bit FIFO 有效容量贴近 32-word burst 边界。
- `tools/check_p6_hls_structure.py` 新增门禁，防止宽 FIFO 重新回退到 LUTRAM/CLB。

下一版 csynth/impl 验收重点：

- HLS BRAM 增量应保持可控，不能挤占 full-resolution mask 与 feature buffer 的主体存储。
- Vivado 层次报告中 `act_stream_U` 不应再以大 LUT FIFO 形式出现。
- 若 route 仍失败，下一步再处理 `systolic_array_core_row` 的 `weight_buf` LUTRAM/SRL 局部拥塞；不应回退到模板调度或旧 UOP fallback。
