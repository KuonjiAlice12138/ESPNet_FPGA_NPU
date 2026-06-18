#!/usr/bin/env python3
"""Lightweight structural checks for the P6-only HLS main path.

This script intentionally avoids parsing C++.  It catches the regression that
caused repeated P6 attempts to fall back into the old 75-UOP dispatch path or
to keep synthesis-heavy reset/dead fallback logic in the build.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
INT8_CORE = ROOT / "ESP_INT8_hls" / "src" / "int8_core.cpp"
CONFIG = ROOT / "ESP_INT8_hls" / "include" / "npu_config.hpp"
WIN_GEN = ROOT / "ESP_INT8_hls" / "src" / "win_gen.cpp"
SCRATCH_MGR = ROOT / "ESP_INT8_hls" / "src" / "scratch_mgr.cpp"
PARAM_DMA = ROOT / "ESP_INT8_hls" / "src" / "param_dma.cpp"
FRAME_DMA = ROOT / "ESP_INT8_hls" / "src" / "frame_dma.cpp"
SA_CORE = ROOT / "ESP_INT8_hls" / "src" / "sa_core.cpp"
AVGPOOL_UNIT = ROOT / "ESP_INT8_hls" / "src" / "avgpool_unit.cpp"
CONV_STORE = ROOT / "ESP_INT8_hls" / "src" / "conv_store.cpp"
CONCAT_UNIT = ROOT / "ESP_INT8_hls" / "src" / "concat_unit.cpp"
MEMORY = ROOT / "ESP_INT8_hls" / "src" / "memory.cpp"


def fail(msg: str) -> int:
    print(f"[P6-CHECK] FAIL: {msg}")
    return 1


def main() -> int:
    core = INT8_CORE.read_text(encoding="utf-8")
    config = CONFIG.read_text(encoding="utf-8")
    win_gen = WIN_GEN.read_text(encoding="utf-8")
    scratch_mgr = SCRATCH_MGR.read_text(encoding="utf-8")
    param_dma = PARAM_DMA.read_text(encoding="utf-8")
    frame_dma = FRAME_DMA.read_text(encoding="utf-8")
    sa_core = SA_CORE.read_text(encoding="utf-8")
    avgpool_unit = AVGPOOL_UNIT.read_text(encoding="utf-8")
    conv_store = CONV_STORE.read_text(encoding="utf-8")
    concat_unit = CONCAT_UNIT.read_text(encoding="utf-8")
    memory = MEMORY.read_text(encoding="utf-8")

    if "ESP_INT8_USE_P6_STATIC_PATH" in config or "ESP_INT8_USE_P6_STATIC_PATH" in core:
        return fail("P6 is still behind a compile-time fallback switch")
    if "run_p6_static_graph" not in core:
        return fail("int8_core.cpp does not define run_p6_static_graph")

    mode_run = re.search(
        r"static\s+error_code_t\s+core_mode_run\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}",
        core,
        flags=re.DOTALL,
    )
    if not mode_run:
        return fail("core_mode_run body not found")

    body = mode_run.group("body")
    if "run_p6_static_graph" not in body:
        return fail("core_mode_run does not call run_p6_static_graph")
    if "static_espnet_scheduler" in body:
        return fail("core_mode_run can still call the old static scheduler")

    p6_func = re.search(
        r"static\s+error_code_t\s+run_p6_static_graph\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}",
        core,
        flags=re.DOTALL,
    )
    if not p6_func:
        return fail("run_p6_static_graph body not found")

    p6_body = p6_func.group("body")
    banned = [
        "static_espnet_scheduler",
        "execute_static_uop_dispatch",
        "fetch_static_uop",
        "param_dma_get_uop",
        "execute_conv_uop",
        "execute_pool_uop",
        "execute_add_uop",
        "execute_affine_uop",
        "execute_store_uop",
    ]
    for token in banned:
        if token in core:
            return fail(f"int8_core.cpp still contains legacy path token: {token}")

    cloned_datapath_tokens = [
        "P6OpRunner",
        "run_p6_static_uop",
        "run_p6_conv_store_pair<",
        "#define P6_RUN",
        "#define P6_RUN_PAIR",
        "StaticUop<IDX>::opcode",
    ]
    for token in cloned_datapath_tokens:
        if token in core:
            return fail(f"int8_core.cpp can still clone datapaths through template dispatch: {token}")

    conv_call_sites = len(re.findall(r"\brun_p6_conv_op\s*\(", core))
    if conv_call_sites != 2:
        return fail(
            "run_p6_conv_op must have exactly one shared call site "
            f"(definition + dispatch call expected, found {conv_call_sites})"
        )

    required_shared_path = [
        "build_p6_static_uop",
        "run_p6_dispatch_uop",
        "run_p6_static_graph",
    ]
    for token in required_shared_path:
        if token not in core:
            return fail(f"missing P6 shared-scheduler component: {token}")

    if "#pragma HLS STREAM variable=act_stream depth=64" not in core:
        return fail("act_stream must keep depth=64 to avoid 32-word FIFO burst-margin risk")

    fifo_bindings = [
        "#pragma HLS BIND_STORAGE variable=act_stream type=fifo impl=bram",
        "#pragma HLS BIND_STORAGE variable=wgt_stream type=fifo impl=bram",
        "#pragma HLS BIND_STORAGE variable=psum_stream type=fifo impl=bram",
    ]
    for token in fifo_bindings:
        if token not in core:
            return fail(f"wide dataflow FIFO is not BRAM-backed: {token}")

    if "ESP_INT8_USE_P6_STATIC_PATH" in win_gen:
        return fail("window_generator_row still has old P6/non-P6 fallback switch")
    if "emit_window_row_3x3_c19_stride2_fast" not in win_gen:
        return fail("P6 window path lost C19 stride-2 fast path")
    if "emit_window_row_3x3_c131_stride2_fast" not in win_gen:
        return fail("P6 window path lost C131 stride-2 fast path")
    for token in ["emit_window_row_3x3_fast", "emit_window_row_1x1_fast"]:
        if token in win_gen:
            return fail(f"win_gen.cpp still contains generic fallback: {token}")

    if "static void invalidate_global_aliases()" in scratch_mgr:
        return fail("scratch_mgr still has loop-style global alias invalidation")
    if "s_global_alias_desc[i] = tensor_desc_t();" in scratch_mgr:
        return fail("scratch_mgr clears tensor_desc_t array in a loop")
    if re.search(r"invalidate_global_aliases[\s\S]*?#pragma\s+HLS\s+PIPELINE\s+II=1", scratch_mgr):
        return fail("global alias invalidation still forces a pipelined reset loop")

    for token in ["s_uop_table", "load_uop(", "param_dma_get_uop("]:
        if token in param_dma:
            return fail(f"param_dma.cpp still contains old UOP table logic: {token}")
    if "frame_dma_store(" in frame_dma:
        return fail("frame_dma.cpp still contains unused frame_dma_store")

    if "weight_buf[MAX_SA_K_TILES][TM][TK]" in sa_core:
        return fail("sa_core.cpp still uses fully-partitioned 3D i8 weight_buf")
    if "#pragma HLS ARRAY_PARTITION variable=weight_buf complete dim=3" in sa_core:
        return fail("sa_core.cpp still partitions weight_buf into TK-wide tiny memories")
    if "#pragma HLS BIND_STORAGE variable=weight_buf type=ram_1p impl=bram" not in sa_core:
        return fail("sa_core.cpp must keep SA weight_buf BRAM-backed; LUTRAM/SRL worsened routing")
    if "impl=lutram" in sa_core:
        return fail("sa_core.cpp still maps SA internals to LUTRAM")
    if "impl=srl" in sa_core:
        return fail("sa_core.cpp still maps internal SA FIFOs to SRL")
    for token in [
        "wgt_stream_g0",
        "wgt_stream_g1",
        "act_stream_g0",
        "act_stream_g1",
        "psum_stream_g0",
        "psum_stream_g1",
        "broadcast_act_stream",
        "split_weight_stream",
        "systolic_array_core_row_group",
    ]:
        if token in sa_core:
            return fail(f"sa_core.cpp still contains group-local stream/control fanout path: {token}")
    if "#pragma HLS DATAFLOW" in sa_core:
        return fail("sa_core.cpp must not create nested SA dataflow; top row-region dataflow is sufficient")
    for token in ["SA_SEGMENT_TM", "mac_tile_segment", "mac_tile_all_lanes"]:
        if token not in sa_core:
            return fail(f"sa_core.cpp lost physical MAC segmentation component: {token}")

    if "write_c3_avg_pixel" in avgpool_unit:
        return fail("avgpool_unit.cpp still contains pixel-level C3 writeback")
    if "avgpool_c3_group32_inner_fast_pack_write" not in avgpool_unit:
        return fail("avgpool_unit.cpp lost fixed-layout C3 group writer")
    if "append_c3_pixel_word" in avgpool_unit:
        return fail("avgpool_unit.cpp still uses dynamic C3 append packing")
    if "on_chip_memory_write_c3_packed" in avgpool_unit + memory:
        return fail("C3 avgpool write still depends on descriptor-based C3 writer")
    if "on_chip_memory_write_c3_abs" in avgpool_unit + memory:
        return fail("C3 avgpool write still uses byte-level RMW writer")
    if "write_aligned_abs_word_narrow" not in avgpool_unit:
        return fail("C3 avgpool write must use narrow bank-local absolute-word writer")
    if "on_chip_memory_write_fmbuf_abs_word" not in avgpool_unit + conv_store + memory:
        return fail("hot-path aligned writes must expose fmbuf-only absolute-word writer")
    if "on_chip_memory_write_pool2_abs_word" not in avgpool_unit + conv_store + memory:
        return fail("hot-path aligned writes must expose pool2-only absolute-word writer")
    if "on_chip_memory_write_aligned_row_word" in conv_store + memory:
        return fail("compact row store still depends on descriptor-based aligned row writer")
    if "write_compact_row_word" not in conv_store:
        return fail("compact row store lost its narrow row-word writer")
    if "on_chip_memory_write_packed_tile(dst" in conv_store:
        return fail("conv_store fallback still calls descriptor-based packed writer in the hot module")
    if "on_chip_memory_write_packed_tile(" in core:
        return fail("P6 core add/affine path still calls descriptor-based packed writer")
    if "on_chip_memory_write_packed_tile(" in concat_unit:
        return fail("P6 concat/store path still calls descriptor-based packed writer")
    if "write_packed_cross_word(" not in memory:
        return fail("memory fallback writer is missing; low-frequency unaligned paths still need coverage")
    if ("write_packed_cross_word(" in conv_store or
            "write_packed_cross_word(" in avgpool_unit or
            "write_packed_cross_word(" in core or
            "write_packed_cross_word(" in concat_unit):
        return fail("P6 hot writer modules must not call cross-word fallback directly")
    if "on_chip_memory_write_aligned_full_tile" not in memory or "on_chip_memory_write_aligned_full_tile" not in core:
        return fail("P6 add/affine full-tile write must keep aligned full-tile memory fast path")
    if "can_use_aligned_full_tile" not in core:
        return fail("aligned full-tile path must be guarded by descriptor/channel alignment")
    if "p6_write_tensor_slice_narrow" not in core:
        return fail("P6 add/affine tail writes must use narrow bank-local RMW")
    if "concat_write_slice_narrow" not in concat_unit:
        return fail("P6 concat/store tail writes must use narrow bank-local RMW")
    if "emit_smallc_c19_words(" in win_gen:
        return fail("C19 window path still has direct stream-write emitter instead of staged emitter")
    if "emit_smallc_c19_words_staged" not in win_gen:
        return fail("C19 window path lost staged stream emitter")

    print("[P6-CHECK] PASS: P6-only path has no legacy scheduler/fallback dead logic")
    return 0


if __name__ == "__main__":
    sys.exit(main())
