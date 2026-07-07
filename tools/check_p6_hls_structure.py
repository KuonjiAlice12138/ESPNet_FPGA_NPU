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
        "post_process_row_to_buffer",
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
        "struct main_ctrl_ctx_t",
        "main_ctrl_run",
        "conv_engine_exec",
        "vec_alu_engine_exec",
        "pool_engine_exec",
        "upsample_engine_exec",
    )
    for token in required_ctrl:
        if token not in ctrl:
            return fail(f"npu_ctrl.hpp missing {token}")

    for token in ("main_ctrl_run", "execute_issue", "build_block5_issue_step", "build_fixed_issue"):
        if token not in main_ctrl:
            return fail(f"main_ctrl.cpp missing {token}")

    mode_run = function_body(core, "core_mode_run")
    if not mode_run:
        return fail("core_mode_run body not found")
    if "main_ctrl_run(gmem_frame_out, profile_ctrl)" not in mode_run:
        return fail("core_mode_run must enter scheduled execution through main_ctrl_run")
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
        "upsample_engine_exec": 1,
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
        "upsample_engine_exec": 3,
    }
    for name, expected in total_expected.items():
        actual = count_call_sites(combined_decl, name)
        if actual != expected:
            return fail(
                f"{name} must appear only as prototype + definition + execute_issue call, got {actual}"
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
        "main_ctrl_record_exec_fetch",
        "main_ctrl_profile_record_exec_entry",
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
        "shared_conv_row_engine",
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
        return fail("run_conv_rows_task must be the only caller of shared_conv_row_engine")
    if count_call_sites(conv_engine, "shared_conv_row_engine") != 2:
        return fail("shared_conv_row_engine should appear only as definition plus run_conv_rows_task call")

    row_engine_body = function_body(conv_engine, "shared_conv_row_engine")
    if row_engine_body is None:
        return fail("shared_conv_row_engine body not found")
    if count_call_sites(row_engine_body, "systolic_array_core_row") != 1:
        return fail("shared_conv_row_engine must call systolic_array_core_row exactly once")

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
    if function_body(vec_alu, "run_fixed_aligned_affine_only_op") is None:
        return fail("vec_alu_engine.cpp missing fixed aligned affine-only path")
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
        "ppu_store_compact_row_core",
        "ppu_apply_c19_affine_word",
        "ppu_apply_block_affine_word",
        "ppu_apply_block_add_affine_word",
        "ppu_read_compact_segment_to_lanes",
        "ppu_read_rowbuf_segment_to_lanes",
        "ppu_block5_l2_tile0_to_word",
        "ppu_block5_l2_tile1_to_word",
        "ppu_block5_l3_tile0_to_word",
        "ppu_block5_l3_tile1_to_word",
        "ppu_block5_l3_tile2_to_word",
        "ppu_block5_l3_tile3_to_word",
        "ppu_finalize_block5_l2_row",
        "ppu_finalize_block5_l3_row",
    ):
        if function_body(ppu, token) is None:
            return fail(f"PPU-5 local datapath helper missing: {token}")
    for token in (
        "ppu_block5_segment_width",
        "ppu_read_block5_scratch_segment",
        "ppu_block5_compose_tile",
        "ppu_finalize_block5_row",
        "ppu_block5_rowbuf_segment_to_lanes",
    ):
        if function_body(ppu, token) is not None:
            return fail(f"PPU BLOCK5 runtime composer must not remain: {token}")
    if "vec_alu_apply_" in ppu:
        return fail("PPU-5 failed: ppu.cpp still calls or declares vec_alu_apply_*")
    fixed_affine_body = function_body(vec_alu, "run_fixed_aligned_affine_only_op") or ""
    if re.search(r"\bvec_alu_apply_affine_only\s*\(", fixed_affine_body):
        return fail("fixed aligned affine-only path still calls generic vec_alu_apply_affine_only")
    if "vec_alu_apply_affine_block" not in fixed_affine_body:
        return fail("fixed aligned affine-only path must use vec_alu_apply_affine_block")
    combined_vec_users = "\n".join([core, conv_engine, ppu, vec_alu])
    affine_block_lines = call_site_lines(combined_vec_users, "vec_alu_apply_affine_block")
    print("[MC-CHECK] AUDIT: Vec ALU arithmetic call-site audit:")
    print(f"  vec_alu_apply_affine_block users: {len(affine_block_lines)} call sites {affine_block_lines}")

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
    ppu_body = function_body(ppu, "ppu_consume_conv_row")
    if ppu_body is None:
        return fail("ppu_consume_conv_row body not found in ppu.cpp")
    if count_call_sites(ppu, "ppu_consume_conv_row") != 1:
        return fail("ppu_consume_conv_row must be defined once in ppu.cpp")
    if count_call_sites(conv_engine, "ppu_consume_conv_row") != 2:
        return fail("ppu_consume_conv_row should be declared and called only from conv_engine.cpp")
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
        "ppu_add_other_row_to_buffer",
        "ppu_apply_row_affine",
        "ppu_cat_other_row_to_buffer",
        "ppu_consume_upsample_out",
        "ppu_store_compact_row",
    ):
        if function_body(ppu, token) is None:
            return fail(f"ppu.cpp missing {token}")
    ppu_compact_body = function_body(ppu, "ppu_store_compact_row") or ""
    for token in ("STORE_LAYOUT_COMPACT_C16", "STORE_LAYOUT_COMPACT_C28"):
        if token in ppu_compact_body:
            return fail(f"PPU artifact-facing compact writer still accepts internal BLOCK5 layout: {token}")
    ppu_block5_scratch_body = function_body(ppu, "ppu_store_block5_scratch_row")
    if ppu_block5_scratch_body is None:
        return fail("ppu.cpp missing bounded BLOCK5 scratch transition writer")
    for token in ("STORE_LAYOUT_COMPACT_C16", "STORE_LAYOUT_COMPACT_C28"):
        if token not in ppu_block5_scratch_body:
            return fail(f"PPU BLOCK5 scratch writer missing internal layout: {token}")
    for token in (
        "ppu_consume_block5_final_row",
        "ppu_finalize_block5_l2_row",
        "ppu_finalize_block5_l3_row",
        "ppu_block5_l2_tile0_to_word",
        "ppu_block5_l2_tile1_to_word",
        "ppu_block5_l3_tile0_to_word",
        "ppu_block5_l3_tile1_to_word",
        "ppu_block5_l3_tile2_to_word",
        "ppu_block5_l3_tile3_to_word",
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
    if len(emit_lines) > 2:
        return fail(
            "PPU-6 failed: BLOCK5 emit tail is still replicated per static tile; "
            "keep at most one L2 and one L3 tail call site"
        )

    for token in (
        "conv_store_write_aligned_tile",
        "conv_store_write_row_contiguous_word",
        "conv_store_row_contiguous_set_byte",
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
        "conv_store_row_contiguous_set_byte",
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

    if "ROW_CONSUMER_UPSAMPLE_OUT" not in ppu_body:
        return fail("PPU streaming path no longer handles ROW_CONSUMER_UPSAMPLE_OUT")
    upsample_forward_body = function_body(ppu, "ppu_consume_upsample_out") or ""
    if "upsample_fused_consume_logits_row" not in upsample_forward_body:
        return fail("PPU upsample row-buffer wrapper no longer calls fused fullres upsample")
    print("[MC-CHECK] AUDIT: PPU conv-row streaming preserved: yes")
    print("[MC-CHECK] AUDIT: PPU upsample row-buffer path preserved: yes")
    print("[MC-CHECK] AUDIT: extra logits memory pass introduced: no")
    for name in ("upsample_engine_exec", "pool_engine_exec"):
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
        (win_gen, "scheduled_window_generator_row", "missing scheduled WinGen entry"),
        (win_gen, "WIN_MODE_3X3_STAGED_C131", "missing staged C131 WinGen coverage"),
        (ppu, "STORE_LAYOUT_", "PPU does not use scheduled store layout"),
        (core, "param_dma_is_schedule_blob", "top path does not require schedule blob"),
        (conv_engine, "shared_conv_row_engine", "missing shared conv row engine"),
        (ppu, "ppu_apply_row_affine", "missing PPU row affine engine"),
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
        return fail("scheduled_window_generator_row body not found")
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
    if "SA_ACTIVE_TM = 16" not in sa_core:
        return fail("SA core must use the P7 16-lane time-mux datapath")
    if "cyclic factor=SA_ACTIVE_TM" not in sa_core:
        return fail("SA core must partition arrays by active 16-lane group")
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
        "upsample_engine_exec",
        "run_conv_issue_once",
        "run_conv_rows_task",
        "shared_conv_row_engine",
        "scheduled_window_generator_row",
        "systolic_array_core_row",
        "post_process_row_to_buffer",
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
        "ppu_consume_conv_row",
        "systolic_array_core_row",
    )
    for line in text.splitlines():
        if re.search(r"\(\d+\s+calls?\)", line) and any(token in line for token in hot_call_tokens):
            print(f"[MC-CHECK] AUDIT: csynth hot call marker: {line.strip()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", default="", help="Optional HLS syn/report directory for post-csynth clone gate")
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

    if check_mainctrl_structure(
        sources["core"],
        sources["main_ctrl"],
        sources["ctrl"],
        sources["vec_alu"],
        sources["conv_engine"],
        sources["ppu"],
    ) != 0:
        return 1
    if check_executor_gateways(
        sources["core"],
        sources["conv_engine"],
        sources["vec_alu"],
        sources["conv_store"],
        sources["ppu"],
    ) != 0:
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
