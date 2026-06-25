#!/usr/bin/env python3
"""Structural checks for the current P7/S4 schedule-executor HLS path.

This is intentionally a lightweight source/config scanner. It is not a C++
parser; it catches regressions where the top path silently falls back to the
old PARAM-v1/UOP/WinGen implementation.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
HLS_DIR = ROOT / "ESP_INT8_hls"
SRC_DIR = HLS_DIR / "src"
INC_DIR = HLS_DIR / "include"
TB_DIR = HLS_DIR / "tb"

INT8_CORE = SRC_DIR / "int8_core.cpp"
WIN_GEN = SRC_DIR / "win_gen.cpp"
PARAM_DMA = SRC_DIR / "param_dma.cpp"
CONV_STORE = SRC_DIR / "conv_store.cpp"
MEMORY = SRC_DIR / "memory.cpp"
CONFIG = INC_DIR / "npu_config.hpp"
SCHEDULE = INC_DIR / "npu_schedule.hpp"


def fail(msg: str) -> int:
    print(f"[S4-CHECK] FAIL: {msg}")
    return 1


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def function_body(source: str, name: str) -> str | None:
    match = re.search(
        rf"static\s+[\w:<>]+\s+{name}\s*\([^)]*\)\s*\{{(?P<body>.*?)\n\}}",
        source,
        flags=re.DOTALL,
    )
    if match:
        return match.group("body")
    return None


def check_configs() -> int:
    for cfg_path in sorted(HLS_DIR.glob("hls_config*.cfg")):
        cfg = read(cfg_path)
        banned_refs = [
            "syn.file=src/concat_unit.cpp",
            "syn.file=src/if_dec.cpp",
            "tb.file=tb/blob_file_tb.cpp",
            "tb.file=tb/control_dma_tb.cpp",
            "tb.file=tb/top_dispatch_tb.cpp",
            "tb.file=tb/top_conv_dispatch_tb.cpp",
            "tb.file=tb/top_pool_store_dispatch_tb.cpp",
            "tb.file=tb/top_tb.cpp",
        ]
        for token in banned_refs:
            if token in cfg:
                return fail(f"{cfg_path.name} still references obsolete S4-excluded path: {token}")
        for line in cfg.splitlines():
            if line.startswith("syn.file=") or line.startswith("tb.file="):
                rel = line.split("=", 1)[1].strip().removeprefix("./")
                if not (HLS_DIR / rel).exists():
                    return fail(f"{cfg_path.name} references missing file: {rel}")
    return 0


def main() -> int:
    if check_configs() != 0:
        return 1

    core = read(INT8_CORE)
    win_gen = read(WIN_GEN)
    param_dma = read(PARAM_DMA)
    conv_store = read(CONV_STORE)
    memory = read(MEMORY)
    avgpool = read(SRC_DIR / "avgpool_unit.cpp")
    sa_core = read(SRC_DIR / "sa_core.cpp")
    upsample = read(SRC_DIR / "upsample_unit.cpp")
    config = read(CONFIG)
    schedule = read(SCHEDULE)
    tb_text = "\n".join(read(path) for path in TB_DIR.glob("*.cpp"))

    required_tokens = [
        (schedule, "struct conv_exec_desc_t", "missing conv_exec_desc_t"),
        (schedule, "struct window_sched_desc_t", "missing window_sched_desc_t"),
        (schedule, "struct row_consumer_desc_t", "missing row_consumer_desc_t"),
        (schedule, "struct exec_plan_entry_t", "missing exec_plan_entry_t"),
        (config, "PARAM_BLOB_VERSION_SCHED", "missing PARAM v3 version constant"),
        (param_dma, "param_dma_get_packed_weight_vec", "missing packed weight getter"),
        (param_dma, "param_dma_get_exec_entry", "missing exec-plan getter"),
        (param_dma, "param_dma_get_row_consumer", "missing row-consumer getter"),
        (win_gen, "scheduled_window_generator_row", "missing scheduled WinGen entry"),
        (schedule, "enum store_layout_mode_t", "missing scheduled store layout enum"),
        (conv_store, "STORE_LAYOUT_", "conv_store does not use scheduled store layout"),
        (core, "run_scheduled_graph", "missing scheduled graph executor"),
        (core, "run_scheduled_conv_op", "missing scheduled conv executor"),
        (core, "param_dma_is_schedule_blob", "top path does not require schedule blob"),
    ]
    for haystack, token, msg in required_tokens:
        if token not in haystack:
            return fail(msg)

    banned_source_tokens = [
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
        "PARAM_BLOB_VERSION =",
        "get_static_tensor_desc",
        "on_chip_memory_write_axi_word",
        "ESP_INT8_CSIM_DUMP_LEVEL3_SPLIT",
    ]
    combined_hot = "\n".join([core, win_gen, param_dma, config, schedule, tb_text])
    for token in banned_source_tokens:
        if token in combined_hot:
            return fail(f"obsolete S4-excluded token is still present: {token}")
    if re.search(r"(?<!scheduled_)window_generator_row\s*\(", combined_hot):
        return fail("obsolete cfg-based window_generator_row is still present")

    if (SRC_DIR / "concat_unit.cpp").exists():
        return fail("legacy concat_unit.cpp still exists in src; S4 must not expose concat_writer")

    mode_run = function_body(core, "core_mode_run")
    if not mode_run:
        return fail("core_mode_run body not found")
    if "run_scheduled_graph(gmem_frame_out)" not in mode_run:
        return fail("core_mode_run does not call run_scheduled_graph")
    if "param_dma_is_schedule_blob()" not in mode_run:
        return fail("core_mode_run does not reject non-schedule PARAM blobs")

    win_entry = re.search(
        r"void\s+scheduled_window_generator_row\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}",
        win_gen,
        flags=re.DOTALL,
    )
    if not win_entry:
        return fail("scheduled_window_generator_row body not found")
    win_body = win_entry.group("body")
    if "sched.mode" not in win_body:
        return fail("scheduled_window_generator_row must dispatch only by sched.window_mode")
    shape_dispatch_patterns = [
        r"cfg\.in_c\s*==",
        r"conv_desc\.in_c\s*==",
        r"cfg\.kernel\s*==",
        r"conv_desc\.kernel\s*==",
        r"cfg\.dilation\s*==",
        r"conv_desc\.dilation\s*==",
    ]
    for pattern in shape_dispatch_patterns:
        if re.search(pattern, win_body):
            return fail("scheduled_window_generator_row still dispatches by runtime shape")

    if "ROW_CONSUMER_ADD_STORE" not in core or "ROW_CONSUMER_UPSAMPLE_OUT" not in core:
        return fail("scheduled row consumer path does not cover add/store and final upsample output")
    if "ROW_CONSUMER_ADD_AFFINE_STORE" in core and "return ERR_UNSUPPORTED_OPCODE" not in core:
        return fail("ADD_AFFINE row-consumer mode must be explicit if not in current PARAM v3")

    fixed_builder = function_body(core, "build_legacy_uop_for_fixed_op")
    if not fixed_builder:
        return fail("fixed-op legacy builder body not found")
    allowed_fixed_ids = {1, 3, 4, 5, 6, 20, 34, 35, 36, 37, 38, 39, 53, 67, 68, 69, 70, 71}
    found_fixed_ids = {
        int(match.group(1))
        for match in re.finditer(r"ESP_INT8_BUILD_CASE\((\d+)U\)", fixed_builder)
    }
    extra_fixed_ids = sorted(found_fixed_ids - allowed_fixed_ids)
    missing_fixed_ids = sorted(allowed_fixed_ids - found_fixed_ids)
    if extra_fixed_ids:
        return fail(f"fixed-op builder still exposes non-exec-plan legacy UOP cases: {extra_fixed_ids}")
    if missing_fixed_ids:
        return fail(f"fixed-op builder is missing required exec-plan fixed UOP cases: {missing_fixed_ids}")

    if not re.search(r"store_conv_output_row\s*\([^)]*store_layout", conv_store, flags=re.DOTALL):
        return fail("store_conv_output_row is not driven by scheduled store_layout")
    store_fn = re.search(
        r"bool\s+store_conv_output_row\s*\([^)]*\)\s*\{(?P<body>.*?)\n\}",
        conv_store,
        flags=re.DOTALL,
    )
    if not store_fn:
        return fail("store_conv_output_row body not found")
    store_body = store_fn.group("body")
    for token in ("on_chip_memory_write_packed_tile", "write_packed_cross_word"):
        if token in store_body:
            return fail(f"scheduled store hot path calls generic writer: {token}")
    if "dst.c.to_uint() == cfg.out_c.to_uint()" in store_body or re.search(r"\bcompact_row\s*=", store_body):
        return fail("store_conv_output_row still infers layout from runtime tensor shape")
    for token in ("on_chip_memory_write_packed_tile", "write_packed_cross_word", "write_packed_one_word"):
        if token in memory:
            return fail(f"unused PARAM-v1 packed write primitive still exists in memory.cpp: {token}")

    risky_enum_cast = re.compile(
        r"static_cast<\s*(?:u8_t|u16_t|u32_t)\s*>\(\s*"
        r"(?:BANK_|TID_|LS_|UOP_|ACT_|ERR_|POST_|ROW_CONSUMER_|STORE_LAYOUT_|EXEC_)[A-Za-z0-9_]*\s*\)"
    )
    for path in sorted(SRC_DIR.glob("*.cpp")) + sorted(INC_DIR.glob("*.hpp")):
        source = read(path)
        match = risky_enum_cast.search(source)
        if match:
            return fail(
                f"ambiguous enum-to-ap_uint static_cast in {path.relative_to(ROOT)}: {match.group(0)}"
            )

    if "avgpool_c3_group32_first_col_pack_write" in avgpool:
        return fail("avgpool first-column special caller can trigger generic-pack function cloning")
    if avgpool.count("avgpool_pixel_c3_generic_pack(") > 2:
        return fail("avgpool generic pixel pack has multiple call sites and may be cloned by HLS")
    if "compute_k_valid_lanes" not in sa_core:
        return fail("SA core must precompute K-valid lanes before the 32-lane MAC fanout")
    if "const u16_t k_idx = static_cast<u16_t>(kt * TK + tk)" in sa_core:
        return fail("SA core still computes K-tail compare inside every TM lane")
    if "load_conv_qparam_wordwise" not in param_dma or "load_affine_qparam_wordwise" not in param_dma:
        return fail("PARAM qparam loaders must use word-wise AXI reads to avoid INIT II pressure")
    if "qparam.bias[i] = read_i32_le" in param_dma or "qparam.mult[i] = read_i32_le" in param_dma:
        return fail("conv qparam loader still issues per-field AXI reads")
    if "qparam.mul[i] = read_i32_le" in param_dma or "qparam.bias[i] = read_i32_le" in param_dma:
        return fail("affine qparam loader still issues per-field AXI reads")
    if upsample.count("emit_fullres_rows(") > 2:
        return fail("upsample emits fullres rows through multiple specialized call sites")

    print("[S4-CHECK] PASS: schedule-executor path has no obsolete top-callable P6/PARAM-v1/old-WinGen logic")
    return 0


if __name__ == "__main__":
    sys.exit(main())
