#!/usr/bin/env python3
"""Structural checks for the P7 MainCtrl HLS path.

This scanner is intentionally conservative and text-based. It enforces the
source-level architecture required before entering HLS synthesis:

  top wrapper -> main_ctrl_run -> execute_issue -> executor gateways

It also keeps the earlier P7 hard bans around PARAM v1, old WinGen dispatch,
generic packed writers, and cold RMW fallback.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HLS_DIR = ROOT / "ESP_INT8_hls"
SRC_DIR = HLS_DIR / "src"
INC_DIR = HLS_DIR / "include"
TB_DIR = HLS_DIR / "tb"

INT8_CORE = SRC_DIR / "int8_core.cpp"
MAIN_CTRL = SRC_DIR / "main_ctrl.cpp"
WIN_GEN = SRC_DIR / "win_gen.cpp"
PARAM_DMA = SRC_DIR / "param_dma.cpp"
CONV_STORE = SRC_DIR / "conv_store.cpp"
PPU = SRC_DIR / "ppu.cpp"
MEMORY = SRC_DIR / "memory.cpp"
AVGPOOL = SRC_DIR / "avgpool_unit.cpp"
SA_CORE = SRC_DIR / "sa_core.cpp"
UPSAMPLE = SRC_DIR / "upsample_unit.cpp"
SCRATCH = SRC_DIR / "scratch_mgr.cpp"
CONFIG = INC_DIR / "npu_config.hpp"
SCHEDULE = INC_DIR / "npu_schedule.hpp"
CTRL = INC_DIR / "npu_ctrl.hpp"
VEC_ALU = SRC_DIR / "vec_alu_engine.cpp"
CONV_ENGINE = SRC_DIR / "conv_engine.cpp"
APP_SRC = ROOT / "ESP_INT8_app" / "src"
APP_MAIN = APP_SRC / "main.c"
APP_NPU_C = APP_SRC / "hal" / "int8_npu.c"
APP_NPU_H = APP_SRC / "hal" / "int8_npu.h"


def fail(msg: str) -> int:
    print(f"[MC-CHECK] FAIL: {msg}")
    return 1


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def function_body(source: str, name: str) -> str | None:
    match = re.search(
        rf"(?:static\s+)?[\w:<>\s*&]+\s+{re.escape(name)}\s*\([^;]*?\)\s*\{{",
        source,
        flags=re.DOTALL,
    )
    if not match:
        return None
    open_pos = source.find("{", match.start())
    depth = 0
    for pos in range(open_pos, len(source)):
        ch = source[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[open_pos + 1 : pos]
    return None


def count_call_sites(source: str, token: str) -> int:
    return len(re.findall(rf"\b{re.escape(token)}\s*\(", source))


def call_site_lines(source: str, token: str) -> list[int]:
    lines: list[int] = []
    pattern = re.compile(rf"\b{re.escape(token)}\s*\(")
    for idx, line in enumerate(source.splitlines(), start=1):
        if not pattern.search(line):
            continue
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        if stripped.startswith("static ") or stripped.startswith("error_code_t "):
            continue
        if stripped.startswith("void ") or stripped.startswith("bool "):
            continue
        lines.append(idx)
    return lines


def require(haystack: str, token: str, msg: str) -> int:
    if token not in haystack:
        return fail(msg)
    return 0


def check_configs() -> int:
    for cfg_path in sorted(HLS_DIR.glob("hls_config*.cfg")):
        cfg = read(cfg_path)
        banned_refs = (
            "syn.file=src/concat_unit.cpp",
            "syn.file=src/if_dec.cpp",
            "syn.file=src/store_engine.cpp",
            "syn.file=src/upsample_engine.cpp",
            "syn.file=include/npu_engines.hpp",
            "tb.file=tb/blob_file_tb.cpp",
            "tb.file=tb/control_dma_tb.cpp",
            "tb.file=tb/top_dispatch_tb.cpp",
            "tb.file=tb/top_conv_dispatch_tb.cpp",
            "tb.file=tb/top_pool_store_dispatch_tb.cpp",
            "tb.file=tb/top_tb.cpp",
        )
        for token in banned_refs:
            if token in cfg:
                return fail(f"{cfg_path.name} references obsolete path: {token}")
        for line in cfg.splitlines():
            if line.startswith("syn.file=") or line.startswith("tb.file="):
                rel = line.split("=", 1)[1].strip().removeprefix("./")
                if not (HLS_DIR / rel).exists():
                    return fail(f"{cfg_path.name} references missing file: {rel}")

    for cfg_path in sorted(HLS_DIR.glob("hls_config*.cfg")):
        cfg = read(cfg_path)
        if "syn.file=src/int8_core.cpp" not in cfg:
            continue
        for token in (
            "syn.file=src/main_ctrl.cpp",
            "syn.file=src/vec_alu_engine.cpp",
            "syn.file=src/conv_engine.cpp",
            "syn.file=src/ppu.cpp",
            "syn.file=src/conv_store.cpp",
            "syn.file=include/npu_ctrl.hpp",
        ):
            if token not in cfg:
                return fail(f"{cfg_path.name} missing executor source entry: {token}")
        if "syn.file=src/row_consumer.cpp" in cfg:
            return fail(f"{cfg_path.name} still synthesizes row_consumer.cpp; PPU-1 must own conv-row postprocess")

    top_cfg = read(HLS_DIR / "hls_config.cfg")
    for token in (
        "syn.file=src/main_ctrl.cpp",
        "syn.file=src/vec_alu_engine.cpp",
        "syn.file=src/conv_engine.cpp",
        "syn.file=src/ppu.cpp",
        "syn.file=src/conv_store.cpp",
        "syn.file=include/npu_ctrl.hpp",
    ):
        if token not in top_cfg:
            return fail(f"hls_config.cfg missing MainCtrl source entry: {token}")
    if "ESP_INT8_ENABLE_COLD_RMW_FALLBACK" in top_cfg:
        return fail("hls_config.cfg enables cold RMW fallback")
    return 0


def check_no_invalid_allocation_pragmas() -> int:
    for path in sorted(SRC_DIR.glob("*.cpp")) + sorted(INC_DIR.glob("*.hpp")):
        source = read(path)
        if "#pragma HLS ALLOCATION" in source:
            return fail(
                f"allocation pragma remains in {path.relative_to(ROOT)}; "
                "fix call graph instead of forcing allocation"
            )
    return 0


def check_mainctrl_structure(core: str, main_ctrl: str, ctrl: str, vec_alu: str, conv_engine: str, ppu: str) -> int:
    forbidden_hot_tokens = (
        "MAX_FM_W",
        "MAX_K_TILE_COUNT",
        "TK",
        "TM",
        "AXI_WORD_BYTES",
        "row_buf",
        "weight_buf",
        "act_stream",
        "psum_stream",
        "window_pack_cmd",
    )
    for token in forbidden_hot_tokens:
        if token in main_ctrl:
            return fail(f"main_ctrl.cpp contains hot datapath token: {token}")
    for token in (
        "scheduled_window_generator_row",
        "systolic_array_core_row",
        "post_process_conv_iteration",
        "store_conv_output_row",
        "conv_store_write_conv_row",
    ):
        if token in main_ctrl:
            return fail(f"main_ctrl.cpp directly calls datapath/store function: {token}")
    for pattern in (
        r"for\s*\([^)]*MAX_FM_W",
        r"for\s*\([^)]*MAX_K_TILE_COUNT",
        r"for\s*\([^)]*AXI_WORD_BYTES",
        r"for\s*\([^)]*\bTM\b",
        r"for\s*\([^)]*\bTK\b",
    ):
        if re.search(pattern, main_ctrl):
            return fail(f"main_ctrl.cpp contains forbidden hot loop pattern: {pattern}")

    required_ctrl = (
        "enum npu_engine_t",
        "enum npu_issue_kind_t",
        "struct npu_issue_t",
        "main_ctrl_run",
        "conv_engine_exec",
        "vec_alu_engine_exec",
        "pool_engine_exec",
    )
    for token in required_ctrl:
        if token not in ctrl:
            return fail(f"npu_ctrl.hpp missing {token}")

    for token in ("main_ctrl_run", "execute_issue", "build_block5_issue_step", "build_fixed_issue"):
        if token not in main_ctrl:
            return fail(f"main_ctrl.cpp missing {token}")

    for token in (
        "NPU_ENGINE_UPSAMPLE",
        "ISSUE_UPSAMPLE_ROW",
        "upsample_engine_exec",
        "struct main_ctrl_ctx_t",
    ):
        if token in ctrl or token in main_ctrl or token in core:
            return fail(f"dead standalone-upsample/control placeholder remains: {token}")

    mode_run = function_body(core, "core_mode_run")
    if not mode_run:
        return fail("core_mode_run body not found")
    if "main_ctrl_run(gmem_frame_out," not in mode_run:
        return fail("core_mode_run must enter scheduled execution through main_ctrl_run")
    for token in ("PROF_STAGE_FRAME_LOAD", "PROF_STAGE_MAIN_CTRL", "prof_stage_id", "prof_pc", "prof_issue_kind"):
        if token not in mode_run:
            return fail(f"core_mode_run missing stage-counter token: {token}")
    if "run_scheduled_graph(" in mode_run:
        return fail("core_mode_run still calls legacy run_scheduled_graph")
    for token in ("param_dma_is_schedule_blob()", "reset_scratch_state()", "frame_dma_load"):
        if token not in mode_run:
            return fail(f"core_mode_run missing required top-level step: {token}")

    exec_body = function_body(main_ctrl, "execute_issue")
    if not exec_body:
        return fail("execute_issue body not found")
    expected_calls = {
        "conv_engine_exec": 1,
        "vec_alu_engine_exec": 1,
        "pool_engine_exec": 1,
    }
    for name, expected in expected_calls.items():
        actual = count_call_sites(exec_body, name)
        if actual != expected:
            return fail(f"execute_issue must call {name} exactly once, got {actual}")

    issue_call_lines = call_site_lines(main_ctrl, "execute_issue")
    print("[MC-CHECK] AUDIT: execute_issue call sites:")
    for line_no in issue_call_lines:
        print(f"  main_ctrl.cpp:{line_no}")
    if not issue_call_lines:
        return fail("main_ctrl.cpp has no execute_issue caller")

    combined_decl = "\n".join([core, main_ctrl, ctrl, vec_alu, conv_engine])
    total_expected = {
        "conv_engine_exec": 3,
        "vec_alu_engine_exec": 3,
        "pool_engine_exec": 3,
    }
    for name, expected in total_expected.items():
        actual = count_call_sites(combined_decl, name)
        if actual != expected:
            return fail(
                f"{name} must appear only as prototype + definition + execute_issue call, got {actual}"
            )

    sa_calls = call_site_lines(conv_engine, "systolic_array_core_row")
    if len(sa_calls) != 1:
        return fail(f"Round3 requires one SA call site in conv_engine.cpp, got {sa_calls}")
    win_calls = call_site_lines(conv_engine, "scheduled_window_generator_row")
    if len(win_calls) != 1:
        return fail(
            "Round3 requires one full-row WinGen call site, "
            f"got {win_calls}"
        )

    ctrl_body = function_body(main_ctrl, "main_ctrl_run")
    if not ctrl_body:
        return fail("main_ctrl_run body not found")
    for token in (
        "prepare_new_entry",
        "build_block5_issue_step",
        "execute_issue",
        "complete_entry",
    ):
        if token not in ctrl_body:
            return fail(f"main_ctrl_run missing {token}")
    for token in (
        "param_dma_get_exec_entry",
        "build_fixed_issue",
    ):
        if token not in main_ctrl:
            return fail(f"main_ctrl.cpp missing helper-side control token: {token}")
    if "#pragma HLS UNROLL" in ctrl_body and "#pragma HLS UNROLL off" not in ctrl_body:
        return fail("main_ctrl_run must not unroll the exec-plan loop")

    block_body = function_body(main_ctrl, "build_block5_issue_step")
    if not block_body:
        return fail("build_block5_issue_step body not found")
    for token in (
        "block5_sched_desc_t",
        "param_dma_get_block5_sched",
        "ISSUE_CONV_BLOCK5_BRANCH",
    ):
        if token not in block_body:
            return fail(f"build_block5_issue_step missing {token}")
    if "ISSUE_VEC_BLOCK5_FINAL" in block_body or "NPU_ENGINE_VEC" in block_body:
        return fail("PPU-2 BLOCK5 FSM must not emit standalone Vec finalizer issue")
    for banned in ("run_conv_issue_once", "run_conv_rows_task", "shared_conv_row_engine"):
        if banned in block_body:
            return fail(f"BLOCK5 micro-state directly calls conv datapath: {banned}")
    for token in ("BLOCK5_ROW_BLOCK_ROWS", "sched.branch_count", "ISSUE_CONV_BLOCK5_BRANCH"):
        if token not in block_body:
            return fail(f"BLOCK5 micro-state missing weight-reload audit anchor: {token}")
    print("[MC-CHECK] AUDIT: BLOCK5 weight reload risk:")
    print("  row_group_count = ceil(sched.out_h / BLOCK5_ROW_BLOCK_ROWS)")
    print("  branch_count = sched.branch_count")
    print("  estimated branch issue count = row_group_count * branch_count")
    return 0


def check_stage_counter_interfaces(core: str, ctrl: str, main_ctrl: str, conv_engine: str) -> int:
    required_stages = (
        "PROF_STAGE_IDLE = 0",
        "PROF_STAGE_PARAM_INIT = 1",
        "PROF_STAGE_FRAME_LOAD = 2",
        "PROF_STAGE_MAIN_CTRL = 3",
        "PROF_STAGE_CONV_WEIGHT_LOAD = 4",
        "PROF_STAGE_CONV_ROW_DATAPATH = 5",
        "PROF_STAGE_PPU_ROW_CONSUME = 6",
        "PROF_STAGE_PPU_BLOCK5_FINAL = 7",
        "PROF_STAGE_VEC_FIXED = 8",
        "PROF_STAGE_AVGPOOL = 9",
        "PROF_STAGE_UPSAMPLE_OUT = 10",
        "PROF_STAGE_FRAME_STORE = 11",
        "PROF_STAGE_ERROR = 12",
        "PROF_STAGE_COUNT = 13",
    )
    for token in required_stages:
        if token not in ctrl:
            return fail(f"npu_ctrl.hpp missing stable profile stage enum token: {token}")
    for token in ("prof_stage_id", "prof_active", "prof_pc", "prof_issue_kind"):
        if token not in core:
            return fail(f"top signature missing profile pin: {token}")
        expected_iface = "ap_vld" if token == "prof_stage_id" else "ap_none"
        if f"#pragma HLS INTERFACE {expected_iface} port={token}" not in core:
            return fail(f"profile pin must be {expected_iface}, not AXI-Lite: {token}")
        if f"s_axilite port={token}" in core:
            return fail(f"profile pin incorrectly exposed as AXI-Lite: {token}")
    for token in (
        "PROF_STAGE_CONV_WEIGHT_LOAD",
        "PROF_STAGE_CONV_ROW_DATAPATH",
        "PROF_STAGE_PPU_ROW_CONSUME",
        "PROF_STAGE_PPU_BLOCK5_FINAL",
    ):
        if token not in conv_engine:
            return fail(f"conv_engine.cpp missing stage token: {token}")
    for token in ("prof_pc = ctx.pc", "prof_issue_kind = issue.kind"):
        if token not in main_ctrl:
            return fail(f"main_ctrl.cpp missing issue profile assignment: {token}")
    hot_loop_patterns = (
        r"for\s*\([^)]*ow",
        r"for\s*\([^)]*kt",
        r"for\s*\([^)]*lane",
        r"for\s*\([^)]*pixel",
        r"for\s*\([^)]*byte",
        r"for\s*\([^)]*kh",
        r"for\s*\([^)]*kw",
    )
    for source_name, source in (("conv_engine.cpp", conv_engine), ("main_ctrl.cpp", main_ctrl)):
        for match in re.finditer(r"npu_profile_set_stage\s*\(", source):
            prefix = source[: match.start()]
            line_start = prefix.rfind("\n")
            nearby = source[max(0, line_start - 240) : match.start()]
            if any(re.search(pattern, nearby) for pattern in hot_loop_patterns):
                return fail(f"profile stage update appears near hot loop in {source_name}")
    return 0


def check_app_stage_counter() -> int:
    if not APP_MAIN.exists() or not APP_NPU_C.exists() or not APP_NPU_H.exists():
        return fail("app source files for stage counter check are missing")
    app_main = read(APP_MAIN)
    app_npu_c = read(APP_NPU_C)
    app_npu_h = read(APP_NPU_H)
    for token in (
        "stage_counter_clear",
        "stage_counter_enable",
        "stage_counter_freeze",
        "stage_counter_read_total",
        "stage_counter_read_active",
        "stage_counter_read_stage",
        "stage_counter_dump_csv",
    ):
        if token not in app_npu_h:
            return fail(f"int8_npu.h missing stage counter API: {token}")
        if token not in app_npu_c:
            return fail(f"int8_npu.c missing stage counter implementation: {token}")
    for token in (
        "RTL_STAGE_CYCLES_BEGIN",
        "RTL_STAGE_CYCLES_END",
        "STAGE_COUNTER_STAGE_BASE_OFFSET",
    ):
        if token not in app_npu_c:
            return fail(f"int8_npu.c missing stage counter output/register token: {token}")
    for token in (
        "stage_counter_clear();",
        "stage_counter_enable(1U);",
        "stage_counter_freeze(1U);",
        "stage_counter_dump_csv();",
    ):
        if token not in app_main:
            return fail(f"main.c MODE_RUN path missing stage counter call: {token}")
    return 0


def check_executor_gateways(
    core: str,
    conv_engine: str,
    vec_alu: str,
    conv_store: str,
    ppu: str,
) -> int:
    for token in (
        "conv_engine_exec",
        "run_conv_issue_once",
        "run_conv_rows_task",
        "run_segment_overlap_row",
        "overlap_compute_and_consume_segment",
        "compute_conv_datapath_owner",
        "build_normal_conv_task",
        "build_block5_branch_task",
        "load_conv_weight_buffer",
        "scheduled_window_generator_row",
        "systolic_array_core_row",
        "s_shared_conv_row_buf",
        "s_shared_weight_buf",
    ):
        if token in core:
            return fail(f"conv datapath token remains in int8_core.cpp after conv split: {token}")

    for token in (
        "conv_engine_exec",
        "run_conv_issue_once",
        "run_conv_rows_task",
        "shared_conv_row_engine",
        "generate_conv_window_row",
        "consume_compact_conv_row",
        "replay_conv_row_to_stream",
        "build_normal_conv_task",
        "build_block5_branch_task",
        "load_conv_weight_buffer",
    ):
        if token not in conv_engine:
            return fail(f"conv_engine.cpp missing {token}")

    conv_issue_body = function_body(conv_engine, "run_conv_issue_once")
    if conv_issue_body is None:
        return fail("run_conv_issue_once body not found")
    if count_call_sites(conv_issue_body, "run_conv_rows_task") != 1:
        return fail("run_conv_issue_once must call run_conv_rows_task exactly once")
    if count_call_sites(conv_engine, "run_conv_rows_task") != 2:
        return fail("run_conv_rows_task should appear only as definition plus run_conv_issue_once call")

    conv_engine_body = function_body(conv_engine, "conv_engine_exec")
    if conv_engine_body is None:
        return fail("conv_engine_exec body not found")
    if count_call_sites(conv_engine_body, "run_conv_issue_once") != 1:
        return fail("conv_engine_exec must be the only caller of run_conv_issue_once")
    if count_call_sites(conv_engine, "run_conv_issue_once") != 2:
        return fail("run_conv_issue_once should appear only as definition plus conv_engine_exec call")
    for token in ("ISSUE_CONV_NORMAL", "ISSUE_CONV_BLOCK5_BRANCH", "alias_tensor_to_slice"):
        if token not in conv_engine_body:
            return fail(f"conv_engine_exec missing {token}")

    conv_task_body = function_body(conv_engine, "run_conv_rows_task")
    if conv_task_body is None:
        return fail("run_conv_rows_task body not found")
    if count_call_sites(conv_task_body, "shared_conv_row_engine") != 1:
        return fail("Round0722 baseline must have one whole-row Conv call site")
    if count_call_sites(conv_task_body, "consume_compact_conv_row") != 1:
        return fail("Round0722 baseline must have one compact PPU call site per row")

    dataflow_body = function_body(conv_engine, "shared_conv_row_engine") or ""
    if "#pragma HLS DATAFLOW" not in dataflow_body:
        return fail("Round0722 whole-row Conv owner is not a DATAFLOW region")
    for token in (
        "generate_conv_window_row",
        "systolic_array_core_row",
        "post_process_conv_row_to_buffer",
    ):
        if count_call_sites(dataflow_body, token) != 1:
            return fail(f"Round0722 whole-row Conv owner must call {token} exactly once")
    if re.search(r"\b(if|for|while|switch)\s*\(", dataflow_body):
        return fail("Round0722 whole-row Conv owner contains control flow")

    compact_body = function_body(conv_engine, "consume_compact_conv_row") or ""
    if "#pragma HLS DATAFLOW" not in compact_body:
        return fail("Round0722 compact PPU owner is not a DATAFLOW region")
    for token in ("replay_conv_row_to_stream", "ppu_consume_conv_stream"):
        if count_call_sites(compact_body, token) != 1:
            return fail(f"Round0722 compact PPU owner must call {token} exactly once")
    if re.search(r"\b(if|for|while|switch)\s*\(", compact_body):
        return fail("Round0722 compact PPU owner contains control flow")

    generate_body = function_body(conv_engine, "generate_conv_window_row") or ""
    if count_call_sites(generate_body, "scheduled_window_generator_row") != 1:
        return fail("Round0722 WinGen gateway must call the scheduled entry exactly once")
    for token in ("issue_begin", "issue_count", "row_begin"):
        if token in generate_body:
            return fail(f"Round0722 whole-row WinGen gateway retains segment token {token}")

    for legacy in (
        "overlap_compute_and_consume_segment",
        "stage_segment_for_consumer",
        "consume_conv_segment_task",
        "s_segment_out_buf0",
        "s_segment_out_buf1",
        "s_segment_act_buf0",
        "s_segment_act_buf1",
        "capture_window_segment",
        "prefetch_window_segment",
        "produce_conv_iteration_acts",
        "compute_consume_segment_dataflow",
        "run_streamed_segment_row",
        "make_segment_cfg",
        "make_segment_sched",
    ):
        if legacy in conv_engine:
            return fail(f"Round0722 legacy overlap/materialization token remains: {legacy}")
    if "ppu_consume_conv_row" in conv_engine:
        return fail("Round0722 must not restore the obsolete generic Conv PPU wrapper")
    if count_call_sites(conv_engine, "ppu_consume_conv_stream") != 2:
        return fail("Round0722 compact PPU must have one declaration and one physical call site")
    if count_call_sites(conv_engine, "scheduled_window_generator_row") != 2:
        return fail("Round0722 WinGen must have one declaration and one physical call site")
    if "scheduled_window_generator_segment" in conv_engine:
        return fail("Round0722 obsolete segment WinGen gateway remains in conv_engine.cpp")
    if not re.search(
        r"ppu_consume_conv_stream\s*\([^;{]*hls::stream<act_vec_t>\s*&\s*conv_stream",
        ppu,
        flags=re.DOTALL,
    ):
        return fail("Round0722 PPU must retain one stream input gateway")
    for token in ("sa_cfg_stream", "sa_sched_flags_stream"):
        if token not in dataflow_body:
            return fail(f"Round0722 DATAFLOW control stream missing from Conv datapath owner: {token}")

    post_body = function_body(conv_engine, "post_process_conv_row_to_buffer")
    if post_body is None:
        return fail("post_process_conv_row_to_buffer body not found")
    if "set_act_vec_i8_dynamic" in post_body:
        return fail("Round0722 postprocess still performs dynamic packed-word RMW")
    for token in ("POST_LANES_PER_CYCLE", "out_lanes", "POST_REQUANT_GROUPS"):
        if token not in post_body:
            return fail(f"Round0722 shared lane postprocess structure missing: {token}")

    if re.search(r"static\s+[^;\n]*\bs_compact_", ppu):
        return fail("Round0722 compact PPU still carries segment-persistent static packet state")

    for token in (
        "run_scheduled_conv_op",
        "run_scheduled_fixed_op",
        "run_scheduled_graph",
        "run_scheduled_block5_op",
        "run_block5_branch_loop",
        "run_block5_row_group",
    ):
        if token in core:
            return fail(f"legacy top scheduler/helper remains in int8_core.cpp: {token}")

    if "run_scheduled_affine_family_op" in core or "run_scheduled_affine_family_op" in vec_alu:
        return fail("legacy run_scheduled_affine_family_op remains after Vec ALU split")
    if "execute_shared_vec_stage" in core or "execute_shared_vec_stage" in vec_alu:
        return fail("legacy execute_shared_vec_stage remains; vector arithmetic must use the canonical vec datapath")
    if function_body(core, "vec_alu_run_fixed_issue") is not None:
        return fail("vec_alu_run_fixed_issue must be owned by vec_alu_engine.cpp, not int8_core.cpp")
    if function_body(vec_alu, "vec_alu_run_fixed_issue") is None:
        return fail("vec_alu_engine.cpp missing vec_alu_run_fixed_issue")
    for token in (
        "compose_block5_l2_to_final",
        "compose_block5_l3_to_final",
        "apply_block5_affine_tile",
        "vec_alu_run_block5_final_issue",
    ):
        if function_body(core, token) is not None:
            return fail(f"{token} must be owned by vec_alu_engine.cpp, not int8_core.cpp")
        if function_body(vec_alu, token) is not None:
            return fail(f"standalone Vec BLOCK5 finalizer remains after PPU-2: {token}")
    if "run_fixed_aligned_affine_family_op" in core or "run_fixed_aligned_affine_family_op" in vec_alu:
        return fail("run_fixed_aligned_affine_family_op remains; split affine-only and add-affine paths")
    if function_body(vec_alu, "run_fixed_affine_common") is None:
        return fail("vec_alu_engine.cpp missing Round5 common fixed-affine path")
    for legacy_runner in (
        "run_fixed_aligned_affine_only_op",
        "run_fixed_row_contiguous_datapath",
        "run_fixed_row_contiguous_op",
        "run_b2_c131_row_contiguous_backup_op",
    ):
        if function_body(vec_alu, legacy_runner) is not None:
            return fail(f"Round5 duplicate fixed-affine runner remains: {legacy_runner}")
    if function_body(vec_alu, "run_fixed_aligned_add_affine_op") is not None:
        return fail("PPU-3 narrowing failed: standalone fixed add-affine path remains in Vec")

    vec_body = function_body(vec_alu, "vec_alu_engine_exec")
    if vec_body is None:
        return fail("vec_alu_engine_exec body not found in vec_alu_engine.cpp")
    for token in ("ISSUE_VEC_BLOCK5_FINAL", "vec_alu_run_block5_final_issue"):
        if token in vec_body or token in vec_alu:
            return fail(f"standalone Vec BLOCK5 finalizer token remains after PPU-2: {token}")
    if "vec_alu_run_fixed_issue" not in vec_body:
        return fail("vec_alu_engine_exec missing vec_alu_run_fixed_issue")
    if "ISSUE_VEC_ADD_AFFINE" in vec_body or "ISSUE_VEC_B2_BACKUP" in vec_body:
        return fail("PPU-3 narrowing failed: Vec engine still accepts legacy fixed issue kinds")

    if function_body(vec_alu, "vec_alu_apply_affine_block") is None:
        return fail("vec_alu_engine.cpp missing active affine-block primitive")
    for token in ("vec_alu_apply_add_only", "vec_alu_apply_affine_only",
                  "vec_alu_apply_add_affine_block"):
        if function_body(vec_alu, token) is not None:
            return fail(f"dead Vec primitive remains after PPU-5 cleanup: {token}")
    for token in ("vec_alu_apply_affine_full_tile", "vec_alu_apply_add_affine_full_tile"):
        if function_body(vec_alu, token) is not None:
            return fail(f"obsolete full-tile Vec primitive remains: {token}")
    for token in (
        "ppu_consume_conv_stream",
        "ppu_transform_conv_word",
        "ppu_write_compact_word",
        "ppu_apply_c19_affine_word",
        "ppu_apply_block_affine_word",
        "ppu_apply_block_add_affine_word",
        "ppu_init_block5_source_cursors",
        "ppu_read_block5_source_words",
        "ppu_compose_block5_l2_words",
        "ppu_compose_block5_l3_words",
        "ppu_finalize_block5_emit_tiles",
        "ppu_finalize_block5_static_row",
    ):
        if function_body(ppu, token) is None:
            return fail(f"PPU-5 local datapath helper missing: {token}")
    for helper in (
        "ppu_apply_block_affine_word",
        "ppu_apply_block_add_affine_word",
    ):
        helper_body = function_body(ppu, helper) or ""
        if "#pragma HLS UNROLL factor=4" not in helper_body:
            return fail(f"Round0717 {helper} must retain the routable factor=4 arithmetic baseline")
        if "#pragma HLS UNROLL factor=8" in helper_body:
            return fail(f"Round0717 {helper} still contains the rejected factor=8 sprint")
    for token in (
        "ppu_block5_segment_width",
        "ppu_read_block5_scratch_segment",
        "ppu_block5_compose_tile",
        "ppu_finalize_block5_row",
        "ppu_block5_rowbuf_segment_to_lanes",
        "ppu_read_compact_segment_to_lanes",
        "ppu_read_rowbuf_segment_to_lanes",
        "ppu_block5_l2_tile0_to_word",
        "ppu_block5_l2_tile1_to_word",
        "ppu_block5_l3_tile0_to_word",
        "ppu_block5_l3_tile1_to_word",
        "ppu_block5_l3_tile2_to_word",
        "ppu_block5_l3_tile3_to_word",
        "ppu_read_block5_compact_bytes",
        "ppu_read_block5_l2_source_words",
        "ppu_read_block5_l3_source_words",
    ):
        if function_body(ppu, token) is not None:
            return fail(f"PPU BLOCK5 runtime composer must not remain: {token}")
    for token in (
        "ppu_block5_compact_cursor_t",
        "ppu_init_block5_compact_cursor",
        "ppu_init_block5_source_cursors",
        "ppu_read_block5_compact_cursor",
        "ppu_read_block5_source_words",
    ):
        if token not in ppu:
            return fail(f"Round0717 BLOCK5 rolling source cache missing: {token}")
    init_cursor_body = function_body(ppu, "ppu_init_block5_compact_cursor") or ""
    if count_call_sites(init_cursor_body, "ppu_read_abs_word") != 1:
        return fail("Round0717 cursor init must prefetch exactly one first word")
    if "cursor.valid = true" not in init_cursor_body:
        return fail("Round0717 cursor init must leave the first word valid")
    init_sources_body = function_body(ppu, "ppu_init_block5_source_cursors") or ""
    if count_call_sites(init_sources_body, "ppu_init_block5_compact_cursor") != 4:
        return fail("Round0717 unified source init must initialize exactly four cursors")
    cursor_body = function_body(ppu, "ppu_read_block5_compact_cursor") or ""
    if count_call_sites(cursor_body, "ppu_read_abs_word") != 1:
        return fail("Round0717 hot cursor must have one physical read call site")
    for legacy_token in ("read_count", "read_idx < 2", "need_current_word"):
        if legacy_token in cursor_body:
            return fail(f"Round0717 hot cursor still permits the old two-read path: {legacy_token}")
    source_body = function_body(ppu, "ppu_read_block5_source_words") or ""
    read_count = count_call_sites(source_body, "ppu_read_block5_compact_cursor")
    if read_count != 4:
        return fail(f"Round0717 unified source reader must consume exactly four cursors, got {read_count}")
    if "local_row" in source_body or "u16_t ow" in source_body:
        return fail("Round0717 unified source reader must use sequential cursors, not per-pixel addresses")
    for helper in (
        "ppu_compose_block5_l2_words",
        "ppu_compose_block5_l3_words",
    ):
        helper_body = function_body(ppu, helper) or ""
        if "i8_t lanes" in helper_body or "ppu_pack_lanes" in helper_body:
            return fail(f"Round0717 {helper} must use fixed bit slices, not a lane array")
        if ".range(" not in helper_body:
            return fail(f"Round0717 {helper} is missing fixed-range tile composition")
    block5_row_body = function_body(ppu, "ppu_finalize_block5_static_row") or ""
    if block5_row_body.count("ppu_block5_compact_cursor_t") != 4:
        return fail("Round0717 BLOCK5 row owner must instantiate exactly four source cursors")
    if count_call_sites(block5_row_body, "ppu_init_block5_source_cursors") != 1:
        return fail("Round0717 BLOCK5 row owner must have one unified cursor-init call site")
    if count_call_sites(block5_row_body, "ppu_read_block5_source_words") != 1:
        return fail("Round0717 BLOCK5 row owner must have one unified source-reader call site")
    for helper in (
        "ppu_compose_block5_l2_words",
        "ppu_compose_block5_l3_words",
    ):
        if count_call_sites(block5_row_body, helper) != 1:
            return fail(f"Round0717 BLOCK5 row owner must have one {helper} call site")
    if "vec_alu_apply_" in ppu:
        return fail("PPU-5 failed: ppu.cpp still calls or declares vec_alu_apply_*")
    fixed_affine_body = function_body(vec_alu, "run_fixed_affine_common") or ""
    if re.search(r"\bvec_alu_apply_affine_only\s*\(", fixed_affine_body):
        return fail("common fixed-affine path still calls generic vec_alu_apply_affine_only")
    if "vec_alu_apply_affine_block" not in fixed_affine_body:
        return fail("common fixed-affine path must use vec_alu_apply_affine_block")
    combined_vec_users = "\n".join([core, conv_engine, ppu, vec_alu])
    affine_block_lines = call_site_lines(combined_vec_users, "vec_alu_apply_affine_block")
    print("[MC-CHECK] AUDIT: Vec ALU arithmetic call-site audit:")
    print(f"  vec_alu_apply_affine_block users: {len(affine_block_lines)} call sites {affine_block_lines}")
    if len(affine_block_lines) != 1:
        return fail("Round5 requires one source call site for the fixed affine datapath")
    for token in (
        "vec_affine_group_buffer_t",
        "vec_append_compact_group",
        "vec_pack_compact_groups",
        "FIXED_FLAG_CBLOCK_MAJOR",
    ):
        if token not in vec_alu:
            return fail(f"Round0717 Vec packetizer/loop-mode structure missing: {token}")
    if "for (int lane = 0; lane < TM; ++lane)" in fixed_affine_body:
        return fail("Round0717 C131 path still appends one byte per cycle")
    if "packed_lane * 8U" in fixed_affine_body:
        return fail("Round0717 C131 path still uses a dynamic 256-bit destination range")
    if "s_shared_row_contig_words[packed_word_idx] = packed_row_word" in fixed_affine_body:
        return fail("Round0717 legacy byte-packed row word remains")
    if "cblock_major" not in fixed_affine_body:
        return fail("Round0717 fixed-affine path does not dispatch the compiled loop mode")
    for token in (
        "vec_alu_load_affine_resident_bank",
        "i32_t qmul[TM][VEC_AFF_RESIDENT_BLOCKS]",
        "const int c_blk = cblock_major ? outer : inner",
        "const int h_i = cblock_major ? middle : row_h_i",
        "const int w_i = cblock_major ? inner : middle",
    ):
        if token not in fixed_affine_body:
            return fail(f"Round4A resident-qparam/traversal structure missing: {token}")
    outer_loop = fixed_affine_body.find("for (int outer")
    if outer_loop < 0:
        return fail("Round4A fixed-affine hot loop is missing")
    if "param_dma_get_affine_qparam" in fixed_affine_body[outer_loop:]:
        return fail("Round4A affine qparam load remains inside the tile hot loop")
    affine_apply = function_body(vec_alu, "vec_alu_apply_affine_block") or ""
    affine_half = function_body(vec_alu, "vec_alu_apply_affine_half") or ""
    if "#pragma HLS INLINE" not in affine_apply:
        return fail("Round4A affine block must inline into its unique fixed-Vec caller")
    if "#pragma HLS INLINE off" in affine_apply:
        return fail("Round4A affine block retains per-tile function handshake overhead")
    if "for (int pair = 0; pair < VEC_AFF_HALF_COUNT; ++pair)" not in affine_apply:
        return fail("Round4A affine block is not scheduled as two 16-lane halves")
    if "#pragma HLS PIPELINE II=1" not in affine_apply:
        return fail("Round4A affine half loop is not pipelined at II=1")
    if "vec_alu_apply_affine_half" not in affine_apply:
        return fail("Round4A affine block does not use the 16-lane half primitive")
    if "vec_affine_i8_to_i8_narrow" not in affine_half:
        return fail("Round4A affine half still carries the generic 64-bit requant datapath")
    if "vec_alu_apply_affine_group" in vec_alu:
        return fail("Round4A legacy 8-lane affine group helper remains")
    if re.search(r"valid_c[^\n]*(?:==|!=)[^\n]*(?:131|256)", fixed_affine_body):
        return fail("Round0717 HLS hot path infers fixed loop mode from valid_c")
    if "conv_store_row_contiguous_set_byte" in vec_alu:
        return fail("Round5 row-contiguous path still uses the legacy random byte setter")

    if "consume_conv_output_row" in core:
        return fail("legacy consume_conv_output_row remains in int8_core.cpp")
    for token in (
        "row_consumer_engine_consume",
        "add_other_row_to_buffer",
        "shared_row_affine_engine",
        "cat_other_row_to_buffer",
        "finalize_block_add_affine_row",
        "upsample_engine_consume_logits_row",
    ):
        if function_body(core, token) is not None:
            return fail(f"{token} must be owned by ppu.cpp, not int8_core.cpp")

    if "row_consumer_engine_consume" in conv_engine or "row_consumer_engine_consume" in ppu:
        return fail("legacy row_consumer_engine_consume remains in active PPU-1 path")
    ppu_body = function_body(ppu, "ppu_consume_conv_stream")
    if ppu_body is None:
        return fail("ppu_consume_conv_stream body not found in ppu.cpp")
    if count_call_sites(ppu, "ppu_consume_conv_stream") != 1:
        return fail("ppu_consume_conv_stream must have exactly one definition in ppu.cpp")
    if count_call_sites(conv_engine, "ppu_consume_conv_stream") != 2:
        return fail("ppu_consume_conv_stream should be declared and called only from conv_engine.cpp")
    if count_call_sites(ppu, "ppu_consume_upsample_row") != 1:
        return fail("ppu_consume_upsample_row must be defined once in ppu.cpp")
    if count_call_sites(conv_engine, "ppu_consume_upsample_row") != 2:
        return fail("ppu_consume_upsample_row should be declared and called only from conv_engine.cpp")
    banned_ppu_splits = (
        "consume_store_only",
        "consume_affine_store",
        "consume_add_affine_store",
        "consume_block_add_affine_store",
        "finalize_block_add_affine_row",
    )
    for token in banned_ppu_splits:
        if function_body(ppu, token) is not None or token in ppu_body:
            return fail(
                f"ppu.cpp reintroduced unsupported generic row-consumer path {token}"
            )
    for token in (
        "ppu_preadd_row",
        "ppu_preadd_segment",
        "ppu_consume_upsample_row",
        "ppu_consume_conv_stream",
        "ppu_transform_conv_word",
        "ppu_write_compact_word",
    ):
        if function_body(ppu, token) is None:
            return fail(f"ppu.cpp missing {token}")
    segment_body = function_body(ppu, "ppu_layout_to_compact_shape") or ""
    for token in ("STORE_LAYOUT_COMPACT_C16", "STORE_LAYOUT_COMPACT_C28"):
        if token not in segment_body:
            return fail(f"PPU common compact-store tail missing bounded BLOCK5 layout: {token}")
    for token in (
        "ppu_consume_block5_final_row",
        "ppu_finalize_block5_emit_tiles",
        "ppu_finalize_block5_static_row",
        "ppu_init_block5_source_cursors",
        "ppu_read_block5_source_words",
        "ppu_compose_block5_l2_words",
        "ppu_compose_block5_l3_words",
        "ppu_apply_block_affine_word",
        "ppu_apply_block_add_affine_word",
    ):
        if function_body(ppu, token) is None:
            return fail(f"PPU-5 fused BLOCK5 finalizer missing {token}")
    for token in (
        "compose_block5_l2_to_final",
        "compose_block5_l3_to_final",
        "ppu_read_block5_l2_tile",
        "ppu_read_block5_l3_tile",
        "ppu_apply_block5_affine_tile",
        "ppu_copy_block5_compact_segment",
        "vec_alu_run_block5_final_issue",
    ):
        if token in ppu:
            return fail(f"legacy standalone BLOCK5 finalizer token must not remain after PPU-5: {token}")
    emit_lines = call_site_lines(ppu, "ppu_finalize_block5_emit_word")
    print("[MC-CHECK] AUDIT: PPU BLOCK5 emit tail call-site audit:")
    print(f"  ppu_finalize_block5_emit_word users: {len(emit_lines)} call sites {emit_lines}")
    if len(emit_lines) != 1:
        return fail(
            "Round5 requires exactly one BLOCK5 arithmetic/store emit call site"
        )

    for token in (
        "conv_store_write_aligned_tile",
        "conv_store_write_row_contiguous_word",
        "conv_store_row_contiguous_plan_ok",
    ):
        if token not in conv_store:
            return fail(f"conv_store.cpp missing gateway {token}")
    for low_level in (
        "store_conv_output_row",
        "conv_store_write_conv_row",
        "row_contiguous_set_byte",
        "write_row_contiguous_abs_word",
        "on_chip_memory_write_aligned_full_tile",
        "row_contiguous_write_plan_ok",
    ):
        if re.search(rf"\b{re.escape(low_level)}\s*\(", core):
            return fail(f"int8_core.cpp directly calls low-level store API: {low_level}")
    for token in (
        "conv_store_write_aligned_tile",
        "conv_store_write_row_contiguous_word",
        "conv_store_row_contiguous_plan_ok",
    ):
        if token not in core and token not in ppu and token not in vec_alu:
            return fail(f"store gateway no longer used by top/PPU/vec path: {token}")
    if "conv_store_write_conv_row" in conv_store or "store_conv_output_row" in conv_store:
        return fail("conv_store.cpp still owns advanced conv-row writer in PPU-1")
    store_dispatch_body = conv_store
    for token in (
        "store_compact_c12_row",
        "store_compact_c16_row",
        "store_compact_c19_row",
        "store_compact_c25_row",
        "store_compact_c28_row",
        "store_compact_c2_row",
        "store_c16_into_c19_row",
        "store_compact_strided_row",
    ):
        if token in store_dispatch_body:
            return fail(f"conv_store.cpp still contains compact/high-level writer token: {token}")

    if "ROW_CONSUMER_UPSAMPLE_OUT" not in conv_task_body:
        return fail("Conv row scheduler no longer dispatches ROW_CONSUMER_UPSAMPLE_OUT")
    upsample_forward_body = function_body(ppu, "ppu_consume_upsample_row") or ""
    if "upsample_fused_consume_logits_row" not in upsample_forward_body:
        return fail("PPU upsample row-buffer wrapper no longer calls fused fullres upsample")
    print("[MC-CHECK] AUDIT: PPU conv-row streaming preserved: yes")
    print("[MC-CHECK] AUDIT: PPU upsample row-buffer path preserved: yes")
    print("[MC-CHECK] AUDIT: extra logits memory pass introduced: no")
    for name in ("pool_engine_exec",):
        if function_body(core, name) is None:
            return fail(f"{name} body not found")

    return 0


def check_p7_bans(sources: dict[str, str]) -> int:
    core = sources["core"]
    conv_engine = sources["conv_engine"]
    ppu = sources["ppu"]
    vec_alu = sources["vec_alu"]
    win_gen = sources["win_gen"]
    param_dma = sources["param_dma"]
    schedule = sources["schedule"]
    conv_store = sources["conv_store"]
    memory = sources["memory"]
    avgpool = sources["avgpool"]
    sa_core = sources["sa_core"]
    upsample = sources["upsample"]
    scratch = sources["scratch"]
    tb_text = sources["tb"]
    combined_hot = "\n".join([core, conv_engine, ppu, win_gen, param_dma, schedule, sources["config"], tb_text])

    required_tokens = (
        (schedule, "struct conv_exec_desc_t", "missing conv_exec_desc_t"),
        (schedule, "struct window_sched_desc_t", "missing window_sched_desc_t"),
        (schedule, "struct row_consumer_desc_t", "missing row_consumer_desc_t"),
        (schedule, "struct exec_plan_entry_t", "missing exec_plan_entry_t"),
        (schedule, "struct block5_sched_desc_t", "missing block5_sched_desc_t"),
        (sources["config"], "PARAM_BLOB_VERSION_SCHED", "missing PARAM v4 version constant"),
        (sources["config"], "MAX_BLOCK5_SCHED_COUNT", "missing BLOCK5 schedule capacity"),
        (param_dma, "param_dma_get_packed_weight_vec", "missing packed weight getter"),
        (param_dma, "param_dma_get_exec_entry", "missing exec-plan getter"),
        (param_dma, "param_dma_get_row_consumer", "missing row-consumer getter"),
        (param_dma, "param_dma_get_block5_sched", "missing BLOCK5 schedule getter"),
        (win_gen, "scheduled_window_generator_row", "missing scheduled full-row WinGen entry"),
        (win_gen, "WIN_MODE_3X3_STAGED_C131", "missing staged C131 WinGen coverage"),
        (ppu, "STORE_LAYOUT_", "PPU does not use scheduled store layout"),
        (core, "param_dma_is_schedule_blob", "top path does not require schedule blob"),
        (conv_engine, "shared_conv_row_engine", "missing shared whole-row Conv datapath owner"),
        (ppu, "ppu_transform_conv_word", "missing compact PPU word transform"),
        (vec_alu, "vec_alu_apply_affine_block", "missing active affine-block Vec primitive"),
    )
    for haystack, token, msg in required_tokens:
        if token not in haystack:
            return fail(msg)

    # PARAM v4 descriptor tables are shallow control memories. Keeping them in
    # BRAM burns one block RAM per decomposed field and has repeatedly pushed
    # top-level BRAM over the device limit. Data buffers and weight buffers are
    # intentionally excluded from this gate.
    for table in (
        "s_conv_exec_desc",
        "s_window_sched_desc",
        "s_row_consumer_desc",
        "s_fixed_exec_desc",
        "s_exec_plan",
        "s_block5_sched",
    ):
        if re.search(rf"BIND_STORAGE\s+variable={table}\s+type=ram_2p\s+impl=bram", param_dma):
            return fail(f"PARAM v4 control table must not be BRAM-bound: {table}")
        if not re.search(rf"BIND_STORAGE\s+variable={table}\s+type=ram_2p\s+impl=lutram", param_dma):
            return fail(f"PARAM v4 control table must be LUTRAM-bound: {table}")

    banned_tokens = (
        "param_dma_get_weight_vec",
        "param_dma_get_uop",
        "if_dec_",
        "static_espnet_scheduler",
        "run_p6_static_graph",
        "run_p6_dispatch_uop",
        "run_p6_add_op",
        "execute_static_uop_dispatch",
        "execute_conv_uop",
        "execute_add_uop",
        "emit_smallc_generic",
        "emit_smallc_c12",
        "emit_smallc_c19",
        "emit_smallc_c25",
        "emit_window_row_",
        "concat_writer(",
        "get_static_tensor_desc",
        "on_chip_memory_write_axi_word",
        "ESP_INT8_CSIM_DUMP_LEVEL3_SPLIT",
        "s_shared_conv_task_seq",
        "CONV_TASK_SEQUENCE_MAX",
        "run_conv_task_sequence",
        "ESP_INT8_ENABLE_COLD_RMW_FALLBACK",
        "write_tensor_slice_narrow_checked",
        "scheduled_window_generator_segment",
    )
    for token in banned_tokens:
        if token in combined_hot:
            return fail(f"obsolete P7-excluded token is still present: {token}")
    for token in ("STORE_LAYOUT_NARROW_FIXED", "STORE_LAYOUT_COLD_RMW_FALLBACK"):
        if token in conv_store:
            return fail(f"conv_store.cpp still has hot-path legacy store layout case: {token}")

    for token in (
        "WIN_MODE_SMALLC_3X3_STAGED",
        "WIN_MODE_LARGEC_3X3_SEGMENT",
        "WIN_MODE_FIRST_C3",
        "scheduled_smallc_3x3_window_row",
        "scheduled_segment_3x3_window_row",
        "WIN_MODE_3X3_CACHED",
        "param_dma_get_pack_cmd",
        "scheduled_3x3_cached_window_row",
        "apply_cached_window_pack_cmd",
        "build_staged_generic_word",
        "copy_dynamic_staged_cache_segment",
        "insert_packed_segment",
    ):
        if token in combined_hot:
            return fail(f"obsolete WinGen token is still present: {token}")
    if re.search(r"(?<!scheduled_)window_generator_row\s*\(", combined_hot):
        return fail("obsolete cfg-based window_generator_row is still present")

    win_body = function_body(win_gen, "scheduled_window_generator_row")
    if win_body is None:
        return fail("scheduled full-row WinGen entry is missing")
    if "sched.mode" not in win_body:
        return fail("scheduled_window_generator_row must dispatch by PARAM schedule mode")
    for pattern in (r"cfg\.in_c\s*==", r"conv_desc\.in_c\s*==", r"cfg\.kernel\s*==", r"conv_desc\.kernel\s*=="):
        if re.search(pattern, win_body):
            return fail("scheduled_window_generator_row still dispatches by runtime shape")

    for token in (
        "WIN_MODE_3X3_STAGED_C3",
        "WIN_MODE_3X3_STAGED_C12",
        "WIN_MODE_3X3_STAGED_C19",
        "WIN_MODE_3X3_STAGED_C25",
        "WIN_MODE_3X3_STAGED_C28",
        "WIN_MODE_3X3_STAGED_C64",
        "WIN_MODE_3X3_STAGED_C128",
        "WIN_MODE_3X3_STAGED_C131",
    ):
        if token not in schedule or token not in win_gen:
            return fail(f"staged WinGen mode missing from ABI/dispatch: {token}")

    if (SRC_DIR / "concat_unit.cpp").exists():
        return fail("legacy concat_unit.cpp still exists in src")
    if "on_chip_memory_read_packed_contiguous" in memory:
        return fail("unused PARAM-v1 packed-contiguous reader remains in memory.cpp")
    for token in ("on_chip_memory_write_packed_tile", "write_packed_cross_word", "write_packed_one_word"):
        if token in memory:
            return fail(f"unused PARAM-v1 packed writer remains in memory.cpp: {token}")
    if "on_chip_memory_read_packed_contiguous" in avgpool or "on_chip_memory_read_packed_tile" in avgpool:
        return fail("avgpool hot path must not use generic packed memory readers")
    for token in (
        "avgpool_pixel_c3_generic_pack",
        "avgpool_pixel_c3_inner_fast_pack",
        "avgpool_c3_group32_generic_pack_write",
        "avgpool_c3_group32_inner_fast_pack_write",
    ):
        if token in avgpool:
            return fail(f"avgpool reintroduced duplicated C3 pack/write path: {token}")
    if count_call_sites(avgpool, "avgpool_c3_group32_pack_write") != 2:
        return fail("avgpool must have exactly one C3 group32 pack/write call site plus its definition")
    if "SA_ACTIVE_TM = 32" not in sa_core or "SA_OUTPUT_TM = 16" not in sa_core:
        return fail("SA core must keep the P7 32-lane MAC / 16-lane output grouping")
    if "cyclic factor=SA_ACTIVE_TM" not in sa_core:
        return fail("SA core must partition arrays by the active 32-lane MAC group")
    if "emit_fullres_rows(" in upsample and upsample.count("emit_fullres_rows(") > 2:
        return fail("upsample fullres emitter has multiple specialized call sites")
    for token in ("resolve_block5_scratch_descs", "copy_src1_row_to_b2_backup", "read_b2_backup_src1_tile"):
        if token not in scratch:
            return fail(f"scratch_mgr.cpp missing storage helper: {token}")

    risky_enum_cast = re.compile(
        r"static_cast<\s*(?:u8_t|u16_t|u32_t)\s*>\(\s*"
        r"(?:BANK_|TID_|LS_|UOP_|ACT_|ERR_|POST_|ROW_CONSUMER_|STORE_LAYOUT_|EXEC_)[A-Za-z0-9_]*\s*\)"
    )
    for path in sorted(SRC_DIR.glob("*.cpp")) + sorted(INC_DIR.glob("*.hpp")):
        match = risky_enum_cast.search(read(path))
        if match:
            return fail(f"ambiguous enum-to-ap_uint static_cast in {path.relative_to(ROOT)}: {match.group(0)}")
    return 0


def check_csynth_hierarchy(report_root: Path) -> int:
    reports = []
    for name in ("csynth.rpt", "csynth_design_size.rpt"):
        path = report_root / name
        if path.exists():
            reports.append(path.read_text(encoding="utf-8", errors="ignore"))
    if not reports:
        return fail(f"no csynth reports found under {report_root}")
    text = "\n".join(reports)
    singleton_bases = (
        "main_ctrl_run",
        "execute_issue",
        "conv_engine_exec",
        "vec_alu_engine_exec",
        "run_conv_issue_once",
        "run_conv_rows_task",
        "shared_conv_row_engine",
        "consume_compact_conv_row",
        "scheduled_window_generator_row",
        "systolic_array_core_row",
        "ppu_consume_conv_stream",
        "post_process_conv_row_to_buffer",
        "store_conv_output_row",
        "shared_add_affine_vec_stage",
        "shared_affine_vec_engine",
        "shared_add_vec_engine",
    )
    for base in singleton_bases:
        names = sorted(set(re.findall(re.escape(base) + r"(?:_[0-9]+)?", text)))
        suffixed = [name for name in names if name != base]
        if suffixed:
            return fail(f"csynth hierarchy cloned {base}: {names}")
    hot_call_tokens = (
        "vec_alu_apply_",
        "ppu_write_compact_word",
        "shared_conv_row_engine",
        "ppu_consume_conv_stream",
        "systolic_array_core_row",
    )
    for line in text.splitlines():
        if re.search(r"\(\d+\s+calls?\)", line) and any(token in line for token in hot_call_tokens):
            print(f"[MC-CHECK] AUDIT: csynth hot call marker: {line.strip()}")
    return 0


def check_round2_wingen_structure(win_gen: str, exporter: str) -> int:
    banned = (
        "find_cached_col",
        "select_replacement_slot",
        "update_3x3_column_cache",
        "WINGEN_CACHE_COL_SLOTS][MAX_3X3_CACHE_CHUNKS",
        "scheduled_narrow_3x3_window_row",
        "scheduled_wide_3x3_window_row",
        "prepare_narrow_issue_windows",
        "update_wide_direct_cache",
        "load_narrow_cache_column",
        "load_wide_cache_column",
        "s_narrow_col_tag",
        "s_wide_col_tag",
    )
    for token in banned:
        if token in win_gen:
            return fail(f"Round2 WinGen legacy path remains: {token}")
    required = (
        "configure_window_row",
        "window_row_loader",
        "window_row_assembler",
        "scheduled_3x3_window_row_pipeline",
        "window_column_meta_t",
        "window_load_word_t",
        "narrow_3x3_window_t",
        "s_narrow_row_bank0",
        "s_narrow_row_bank1",
        "s_narrow_row_bank2",
        "window_row_loader_narrow_reuse",
        "window_row_assembler_read_narrow_window",
    )
    for token in required:
        if token not in win_gen:
            return fail(f"Round2 WinGen structure missing: {token}")

    loader_body = function_body(win_gen, "window_row_loader") or ""
    load_column_body = function_body(win_gen, "window_loader_emit_column") or ""
    assembler_body = function_body(win_gen, "window_row_assembler") or ""
    store_body = function_body(win_gen, "window_assembler_store_wide_word") or ""
    pipeline_body = function_body(win_gen, "scheduled_3x3_window_row_pipeline") or ""
    if "window_loader_emit_column" not in loader_body:
        return fail("Round2 loader does not own the compiled column-read sequence")
    if "read_packed_tile_from_row_cached" not in load_column_body:
        return fail("Round2 loader bypasses the bounded packed-word reader")
    if "s_wide_cache" in loader_body or "s_wide_cache" in load_column_body:
        return fail("Round4B loader illegally accesses assembler-owned wide cache")
    if "window_assembler_store_wide_word" not in assembler_body:
        return fail("Round4B assembler does not own wide-cache updates")
    if "s_wide_cache" not in store_body:
        return fail("Round4B wide-cache store gateway is missing")
    if "s_narrow_row_bank" in assembler_body or "s_narrow_row_bank" in store_body:
        return fail("Round4B narrow row bank has more than one DATAFLOW owner")
    if "on_chip_memory_read" in assembler_body or "on_chip_memory_read" in store_body:
        return fail("Round2 assembler performs an illegal FMBUF read")
    for token in (
        "#pragma HLS DATAFLOW",
        "#pragma HLS STREAM variable=column_meta_stream depth=16",
        "#pragma HLS STREAM variable=load_word_stream depth=16",
        "configure_window_row",
        "window_row_loader",
        "window_row_assembler",
    ):
        if token not in pipeline_body:
            return fail(f"Round2 bounded DATAFLOW region missing: {token}")

    emit_body = function_body(win_gen, "emit_narrow_words_for_mode")
    if "s_narrow_cache_row" in emit_body:
        return fail("narrow packer still reads the BRAM cache directly instead of the 9-word snapshot")
    if "4 * dilation" not in exporter:
        return fail("exporter does not compile dilation-aware cache_col_slots")
    for token in (
        "compile_window_loader_contract",
        "compile_window_row_reuse_word",
        "loader_warmup_mask",
        "loader_steady_mask",
    ):
        if token not in exporter:
            return fail(f"exporter is missing the Round2 loader contract: {token}")
    if "ARRAY_PARTITION variable=s_wide_cache.data complete dim=3" not in win_gen:
        return fail("C131 wide cache is not banked by channel chunk")
    return 0


def check_round0727_supply_structure(
    win_gen: str,
    memory: str,
    conv_engine: str,
) -> int:
    """Gate the bounded reader and Round4B single-owner narrow row reuse."""
    pipeline_body = (
        function_body(win_gen, "scheduled_3x3_window_row_pipeline") or ""
    )
    for token in (
        "packed_word_cursor_t",
        "s_packed_word_cursor",
        "reset_packed_word_cursors",
        "read_packed_tile_from_row_cached",
        "s_narrow_row_bank0",
        "s_narrow_row_bank1",
        "s_narrow_row_bank2",
        "narrow_reuse_prepare_rows",
        "read_packed_tile_from_reuse_row",
        "window_row_loader_narrow_reuse",
        "window_loader_emit_narrow_reuse_window",
        "window_row_assembler_read_narrow_window",
    ):
        if token not in win_gen:
            return fail(f"Round0727/Round4B supply structure missing: {token}")

    for legacy in (
        "on_chip_memory_read_packed_tile_from_row",
        "update_narrow_direct_cache",
        "stage_narrow_3x3_window_select",
        "s_narrow_cache_row",
        "window_loader_emit_narrow_run",
        "window_loader_emit_narrow_run_row",
    ):
        if legacy in win_gen or legacy in memory:
            return fail(f"Round0727/Round4B obsolete helper remains: {legacy}")
    if re.search(r"\bstage_narrow_3x3_window\s*\(", win_gen):
        return fail("Round0727 obsolete single-window stager remains")
    if "stage_narrow_3x3_window_pair" in win_gen:
        return fail("Round0727 dual-read paired stager would replicate narrow cache BRAM")

    narrow_loader = (
        function_body(win_gen, "window_row_loader_narrow_paired") or ""
    )
    reuse_loader = (
        function_body(win_gen, "window_row_loader_narrow_reuse") or ""
    )
    narrow_update = (
        function_body(win_gen, "window_loader_update_narrow_run") or ""
    )
    narrow_emit = (
        function_body(win_gen, "window_loader_emit_narrow_cached_window") or ""
    )
    reuse_prepare = function_body(win_gen, "narrow_reuse_prepare_rows") or ""
    reuse_load = (
        function_body(win_gen, "narrow_reuse_load_source_row") or ""
    )
    reuse_read = (
        function_body(win_gen, "read_packed_tile_from_reuse_row") or ""
    )
    bank_read = function_body(win_gen, "narrow_row_bank_read") or ""
    bank_lane_read = (
        function_body(win_gen, "narrow_row_bank_lane_read") or ""
    )
    reuse_emit = (
        function_body(win_gen, "window_loader_emit_narrow_reuse_window") or ""
    )
    assembler = function_body(win_gen, "window_row_assembler") or ""
    if not all(
        (
            narrow_loader,
            reuse_loader,
            narrow_update,
            narrow_emit,
            reuse_prepare,
            reuse_load,
            reuse_read,
            bank_read,
            bank_lane_read,
            reuse_emit,
            assembler,
        )
    ):
        return fail("Round0727/Round4B narrow loader hierarchy is incomplete")

    for forbidden in (
        "window_loader_emit_column",
        "MAX_3X3_CACHE_CHUNKS",
        "column_meta_stream",
    ):
        if forbidden in narrow_loader or forbidden in reuse_loader:
            return fail(f"Round0727/Round4B narrow loader retains generic logic: {forbidden}")
    if "window_loader_update_narrow_run" not in narrow_loader:
        return fail("Round0727 narrow loader does not update its compact horizontal ring")
    if "window_loader_emit_narrow_cached_window" not in narrow_loader:
        return fail("Round0727 narrow loader does not emit from its compact horizontal ring")
    if "narrow_reuse_prepare_rows" not in reuse_loader:
        return fail("Round4B reuse loader does not prepare the compiled three-row window")
    if "window_loader_emit_narrow_reuse_window" not in reuse_loader:
        return fail("Round4B reuse loader does not emit from the packed-row cache")
    if "read_packed_tile_from_reuse_row" not in reuse_emit:
        return fail("Round4B reuse emitter bypasses the packed-row cache reader")
    if "narrow_row_bank_lane_read" not in reuse_read:
        return fail("Round4B packed-row reader bypasses the striped row-bank gateway")
    for token in (
        "s_narrow_row_bank0",
        "s_narrow_row_bank1",
        "s_narrow_row_bank2",
    ):
        if token not in bank_lane_read:
            return fail(f"Round4B row-bank gateway is missing: {token}")
    if "on_chip_memory_read" in reuse_read or "read_packed_tile_from_row_cached" in reuse_read:
        return fail("Round4B hot reuse reader still accesses global feature memory")
    if "narrow_reuse_load_source_row" not in reuse_prepare:
        return fail("Round4B row preparation bypasses its single preload gateway")
    if "on_chip_memory_read_aligned_tensor_word" not in reuse_load:
        return fail("Round4B row preload bypasses the aligned feature-memory gateway")
    if "s_narrow_row_bank" in assembler:
        return fail("Round4B assembler illegally shares ownership of the narrow row banks")

    cached_read = function_body(win_gen, "read_packed_tile_from_row_cached") or ""
    aligned_cache = function_body(win_gen, "read_aligned_word_cached") or ""
    if "read_aligned_word_cached" not in cached_read:
        return fail("Round0727 packed reader bypasses the bounded aligned-word cache")
    if "on_chip_memory_read_aligned_tensor_word" not in aligned_cache:
        return fail("Round0727 aligned-word cache bypasses the single memory owner")
    loader_body = function_body(win_gen, "window_loader_emit_column") or ""
    if "on_chip_memory_read_packed_tile_from_row" in loader_body:
        return fail("Round0727 loader still performs uncached packed-tile reads")
    if "read_packed_tile_from_row_cached" not in loader_body:
        return fail("Round0727 loader does not use the bounded packed-word cursor")

    top_loader = function_body(win_gen, "window_row_loader") or ""
    for token in (
        "window_row_loader_narrow_reuse",
        "window_row_loader_narrow_paired",
        "window_row_loader_narrow_unpaired",
        "BIND_STORAGE variable=s_narrow_row_bank0",
        "BIND_STORAGE variable=s_narrow_row_bank1",
        "BIND_STORAGE variable=s_narrow_row_bank2",
    ):
        if token not in top_loader:
            return fail(f"Round4B single narrow-bank owner is missing: {token}")

    for bank in (
        "s_narrow_row_bank0",
        "s_narrow_row_bank1",
        "s_narrow_row_bank2",
    ):
        if not re.search(
            rf"static\s+u64_t\s+{bank}\s*"
            r"\[\s*WINGEN_NARROW_WORD_LANES\s*\]",
            win_gen,
        ):
            return fail(
                f"Round4B {bank} is not split into 64-bit physical stripes"
            )
        if (
            f"ARRAY_PARTITION variable={bank} complete dim=1"
            not in top_loader
        ):
            return fail(f"Round4B {bank} stripe dimension is not partitioned")
        if (
            f"BIND_STORAGE variable={bank} type=ram_1p impl=bram"
            not in top_loader
        ):
            return fail(f"Round4B {bank} is not bound to single-port BRAM")
        if (
            f"BIND_STORAGE variable={bank} type=ram_2p"
            in top_loader
        ):
            return fail(f"Round4B {bank} still uses the BRAM-expensive dual-port binding")
    if "#pragma HLS STREAM variable=assembler_cfg_stream depth=3" not in pipeline_body:
        return fail("Round4B assembler config FIFO must satisfy the HLS depth=3 guidance")
    if (
        "#pragma HLS BIND_STORAGE variable=assembler_cfg_stream "
        "type=fifo impl=lutram"
        not in pipeline_body
    ):
        return fail("Round4B depth=3 assembler config FIFO must be forced to LUTRAM")

    if function_body(memory, "on_chip_memory_read_aligned_tensor_word") is None:
        return fail("Round0727 memory.cpp is missing the aligned tensor-word gateway")

    post_body = function_body(conv_engine, "post_process_conv_row_to_buffer") or ""
    if not re.search(r"POST_LANES_PER_CYCLE\s*=\s*8\s*;", post_body):
        return fail("Round0727 postprocess is not fixed at 8 lanes/cycle")
    return 0


def check_round4a_ppu_structure(ppu: str, conv_engine: str) -> int:
    """Gate the fixed-rate postprocess-to-PPU stream consumer."""
    stream_consumer = function_body(ppu, "ppu_consume_conv_stream") or ""
    transform = function_body(ppu, "ppu_transform_conv_word") or ""
    writer = function_body(ppu, "ppu_write_compact_word") or ""
    if not stream_consumer or not transform or not writer:
        return fail("Round3 streamed compact consumer structure is incomplete")
    for legacy in (
        "ppu_consume_conv_segment",
        "ppu_consume_compact_segment_readonly",
        "ppu_cat_other_row_to_buffer",
        "ppu_apply_row_affine",
        "ppu_pack_compact_group",
        "ppu_store_compact_row_core",
        "ppu_store_compact_layout_segment",
    ):
        if function_body(ppu, legacy) is not None:
            return fail(f"legacy multi-pass compact consumer remains: {legacy}")
    for token in ("ppu_apply_add_word", "ppu_cat_tail_word", "ppu_apply_c19_affine_word"):
        if token not in transform:
            return fail(f"single-pass segment transform missing {token}")
    for token in ("carry_bytes", "packed_groups", "ppu_transform_conv_word"):
        if token not in stream_consumer:
            return fail(f"single-pass compact packetizer missing {token}")
    for token in ("conv_stream.read()", "status ="):
        if token not in stream_consumer:
            return fail(f"fixed-rate streamed PPU contract missing {token}")
    if "return false" in stream_consumer or "return true" in stream_consumer:
        return fail("streamed PPU may not return before draining its fixed-rate input")
    for dynamic_shift in ("byte_base * 8U", "carry_bytes.to_uint() * 8U"):
        if dynamic_shift in stream_consumer:
            return fail(f"compact packetizer retains timing-critical dynamic shift: {dynamic_shift}")
    if "on_chip_memory_write_pool2_abs_word" not in writer or "on_chip_memory_write_fmbuf_abs_word" not in writer:
        return fail("compact writer does not own both physical destination gateways")
    dataflow = function_body(conv_engine, "consume_compact_conv_row") or ""
    if count_call_sites(dataflow, "ppu_consume_conv_stream") != 1:
        return fail("compact PPU DATAFLOW must have one streamed PPU call site")

    pool_read = function_body(ppu, "ppu_read_abs_word") or ""
    compact_read = function_body(ppu, "ppu_read_block5_compact_cursor") or ""
    if not pool_read or not compact_read:
        return fail("Round0717 rolling BLOCK5 absolute-read gateway is missing")
    if count_call_sites(compact_read, "ppu_read_abs_word") != 1:
        return fail("Round0717 rolling compact reader must have one absolute-read call site")
    if "ppu_block5_read_abs_word" in ppu:
        return fail("Round4A-R legacy BLOCK5 absolute-read wrapper remains")
    for legacy_wrapper in ("ppu_store_compact_row", "ppu_store_block5_scratch_row"):
        if function_body(ppu, legacy_wrapper) is not None:
            return fail(f"Round4A duplicate compact-store wrapper remains: {legacy_wrapper}")
    for qparam_owner in ("ppu_finalize_block5_static_row",):
        body = function_body(ppu, qparam_owner) or ""
        loop_pos = body.find("for (int ow_i")
        if loop_pos < 0:
            return fail(f"Round4A qparam owner missing row loop: {qparam_owner}")
        if "param_dma_get_affine_qparam" in body[loop_pos:]:
            return fail(f"Round4A qparam load remains inside row hot loop: {qparam_owner}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", default="", help="Optional HLS syn/report directory for post-csynth clone gate")
    parser.add_argument(
        "--round4ar-only",
        action="store_true",
        help="Run only the source-level Round4A-R owner/packer gate",
    )
    args = parser.parse_args()

    obsolete_store_engine = SRC_DIR / "store_engine.cpp"
    if obsolete_store_engine.exists():
        return fail("obsolete src/store_engine.cpp still exists; conv_store.cpp must own writeback gateways")

    for required in (INT8_CORE, MAIN_CTRL, CTRL, VEC_ALU, CONV_ENGINE, PPU, CONV_STORE):
        if not required.exists():
            return fail(f"required MainCtrl file missing: {required.relative_to(ROOT)}")
    if check_configs() != 0:
        return 1
    if check_no_invalid_allocation_pragmas() != 0:
        return 1

    sources = {
        "core": read(INT8_CORE),
        "main_ctrl": read(MAIN_CTRL),
        "ctrl": read(CTRL),
        "conv_engine": read(CONV_ENGINE),
        "ppu": read(PPU),
        "vec_alu": read(VEC_ALU),
        "win_gen": read(WIN_GEN),
        "param_dma": read(PARAM_DMA),
        "conv_store": read(CONV_STORE),
        "memory": read(MEMORY),
        "avgpool": read(AVGPOOL),
        "sa_core": read(SA_CORE),
        "upsample": read(UPSAMPLE),
        "scratch": read(SCRATCH),
        "config": read(CONFIG),
        "schedule": read(SCHEDULE),
        "tb": "\n".join(read(path) for path in TB_DIR.glob("*.cpp")),
    }

    if args.round4ar_only:
        return check_round4a_ppu_structure(sources["ppu"], sources["conv_engine"])

    if check_mainctrl_structure(
        sources["core"],
        sources["main_ctrl"],
        sources["ctrl"],
        sources["vec_alu"],
        sources["conv_engine"],
        sources["ppu"],
    ) != 0:
        return 1
    if check_stage_counter_interfaces(
        sources["core"],
        sources["ctrl"],
        sources["main_ctrl"],
        sources["conv_engine"],
    ) != 0:
        return 1
    if check_app_stage_counter() != 0:
        return 1
    if check_executor_gateways(
        sources["core"],
        sources["conv_engine"],
        sources["vec_alu"],
        sources["conv_store"],
        sources["ppu"],
    ) != 0:
        return 1
    if check_round2_wingen_structure(sources["win_gen"], read(ROOT / "tools" / "export_int8_hw_blob.py")) != 0:
        return 1
    if check_round0727_supply_structure(
        sources["win_gen"],
        sources["memory"],
        sources["conv_engine"],
    ) != 0:
        return 1
    if check_round4a_ppu_structure(sources["ppu"], sources["conv_engine"]) != 0:
        return 1
    if check_p7_bans(sources) != 0:
        return 1
    if args.report_root:
        if check_csynth_hierarchy(Path(args.report_root)) != 0:
            return 1

    print("[MC-CHECK] PASS: MainCtrl/executor source structure and P7 hard bans satisfied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
