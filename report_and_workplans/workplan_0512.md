# ESPNet_Encoder INT8 NPU 工作进展与计划 (2026-05-11/12/13)

## 1. 已完成进展

本轮从 DBG12-ROWSTREAM 行级流架构基线出发，经 9 轮迭代实现全网络闭环和性能基线采集。

### 1.1 架构迭代 (DBG12→DBG14)

| 版本 | 改动 | 结果 |
|---|---|---|
| DBG12 | 行级 DATAFLOW，writeback 移出 DATAFLOW | 214-475 消除，资源超标 |
| DBG13 | 清理 volatile/标量输出，P0 权重缓存，P1 win_gen 拆分 | 综合通过 |
| DBG14 | `wgt_stream` depth 32→2305 | **全网络通过** |

关键工具链升级：`audit_hls_reports.py`（BLOCKER/ HIGH-RISK 自动审查）、`dbg_hw_version` 寄存器（版本指纹）。

### 1.2 Root-Cause: wgt_stream FIFO 容量边界

9 轮迭代中前 8 轮 U02 均 timeout，根因是 `fifo_w256_d32_A` 实际容量 31（depth-1）不足容纳 `feed_cached_weights` 的 32 词 burst。生产者阻塞在 FIFO 满，消费者等 `start_out` 才启动——结构性死锁。修复：depth=2305（`MAX_K_TILE_COUNT×TM+1`），增量 ~15 BRAM36。

### 1.3 全网络验证 (DBG14-FIFOFIX)

11 个 checkpoint (U00→U72) 全部通过。**D72OUT.BIN 与 CSim golden 0/16384 byte 差异**。硬件通路（AXI/DDR/SD/URAM/BRAM）全部正常，数据计算正确。

### 1.4 性能 Profiling (DBG15-PERF)

各 UOP 独立计时（含 MODE_INIT + MODE_RUN + dump）：

| 阶段 | UOP | ms | 占比 |
|---|---|---|---|
| U00 input | 0 | 2 | — |
| U01 pool | 1 | 50 | — |
| U02 第一层 3×3 conv | 2 | 119 | 1.4% |
| U03-U04 store+affine | 3-4 | 28 | — |
| **U07-U20 L2_0** | **7-20** | **~2900** | **34%** |
| **U21-U35 L2_B0** | **21-35** | **~2200** | **26%** |
| U36-U39 B2 concat | 36-39 | ~30 | — |
| **U40-U53 L3_0** | **40-53** | **~2100** | **25%** |
| U54-U69 L3_B0 | 54-69 | ~1100 | 13% |
| U70-U72 classifier | 70-72 | ~12 | — |
| **整网单次** | **0-72** | **8521** | **100%** |

**瓶颈：L2_0 + L2_B0 + L3_0 占 85%（~7200ms）**。这些层走 win_gen 通用 `load_window_line/load_window_vector` 路径（逐像素 `can_merge_lane` + `map_k_index`，PIPELINE off），而第一层走专用快速路径仅 119ms。

### 1.5 第一层快速路径 vs 通用路径对比

| 指标 | 第一层 3×3 | 通用 3×3 |
|---|---|---|
| 路径 | `emit_first_layer_3x3_window_row` | `load_window_line` + `can_merge_lane` |
| 每像素 BRAM 读 | 9 次 | ~27 次（含 merge 分析） |
| 每像素延迟 | ~0.9μs | ~10.6μs |
| 差距 | 基准 | **慢 12×** |

## 2. 性能收敛路线图 (目标 <50ms)

| 优先级 | 事项 | 原理 | 预估收益 |
|---|---|---|---|
| **P0** | L2_0/L3_0 3×3 conv 加专用快速路径（同第一层） | 跳过 `can_merge_lane` 动态分析，每像素固定 9 次 BRAM 读 | **10-15×** |
| **P0** | 1×1 conv 专用快速路径 | 直接连续读 256-bit 输入通道，不拆 lane | **5-8×** |
| P1 | `on_chip_memory_read_tile` 合并双读 (II=6→1) | 两次 `read_bank_word` 合并为一次 64B 读 | 2-3× |
| P1 | `on_chip_memory_write_tile` 减 RMW (II=13→1) | 写对齐到 256-bit 边界，消除跨字 RMW | 1.5-2× |
| P2 | 权重缓存加载 II=16→1 | `s_wbuf` 分区消除端口限制 | <1%（仅首行） |

**预估路径：** P0 两条实现后 L2/L3 时间从 ~7200ms → ~500ms。P1 再压到 ~150ms。进一步优化 SA 核 MAC 流水线和权重布局可达 <50ms。所有优化不增加 BRAM/URAM。

## 3. 下一步计划

1. 实现 P0：给 `in_c=19(kernel=3)` 和 `in_c=131(kernel=3)` 的层次加 `emit_window_row_3x3_fast` 专用路径，固定 9 次 BRAM 读/像素。
2. 实现 P0：给所有 1×1 conv 加 `emit_window_row_1x1_fast` 路径，连续 256-bit 读。
3. 重新综合→实现→上板验证 U02/U20/U39/U72 延迟。
4. 根据结果决定是否继续 P1。
5. 最终目标：整网 <50ms @ 100MHz，不增加 BRAM/URAM。

## 4. 版本基线备份

HLS 源码和 app 代码已备份至 `D:\ESP_INT8\backups\DBG14-FIFOFIX_20260513\`。
