#!/usr/bin/env python3
"""Gate HLS design-size reports for accidental heavy datapath cloning.

This script intentionally uses text scanning rather than a fragile table parser,
because Vitis report formatting changes across 2025.x minor versions. It is a
post-csynth audit helper: run it on csynth_design_size.rpt and optionally
csynth.rpt.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable


HEAVY_SINGLETONS = (
    "main_ctrl_run",
    "execute_issue",
    "conv_engine_exec",
    "vec_alu_engine_exec",
    "pool_engine_exec",
    "upsample_engine_exec",
    "run_conv_issue_once",
    "run_conv_rows_task",
    "shared_conv_row_engine",
    "scheduled_window_generator_row",
    "systolic_array_core_row",
    "post_process_row_to_buffer",
    "store_conv_output_row",
    "vec_alu_apply_stage",
    "shared_add_affine_vec_stage_canonical",
    "shared_affine_vec_engine",
    "shared_add_vec_engine",
    "avgpool_unit_checked",
    "upsample_fused_consume_logits_row",
)

WATCHED_FUNCTIONS = (
    "run_conv_rows_task",
    "run_conv_issue_once",
    "run_scheduled_graph",
    "run_scheduled_conv_op",
    "run_scheduled_block5_op",
    "run_block5_row_group",
    *HEAVY_SINGLETONS,
)


def read_reports(paths: Iterable[Path]) -> str:
    chunks: list[str] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(chunks)


def count_call_markers(text: str, name: str) -> list[int]:
    counts: list[int] = []
    for match in re.finditer(re.escape(name) + r".{0,120}?\((\d+)\s+calls?\)", text):
        counts.append(int(match.group(1)))
    return counts


def find_clone_names(text: str, name: str) -> list[str]:
    pattern = re.compile(rf"\b{re.escape(name)}_[0-9]+\b")
    return sorted(set(pattern.findall(text)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path, help="csynth_design_size.rpt and/or csynth.rpt")
    args = parser.parse_args()

    text = read_reports(args.reports)
    failures: list[str] = []

    for name in WATCHED_FUNCTIONS:
        calls = count_call_markers(text, name)
        clones = find_clone_names(text, name)
        print(f"{name}: call_markers={calls or '-'} clones={clones or '-'}")
        if name in HEAVY_SINGLETONS and any(count > 1 for count in calls):
            failures.append(f"{name} has multi-call marker {calls}")
        if name == "store_conv_output_row" and any(count >= 5 for count in calls):
            failures.append(f"{name} has branch-exploded call marker {calls}")
        if clones:
            failures.append(f"{name} cloned as {clones}")

    banned_clone_roots = (
        "run_conv_issue_once_",
        "run_conv_rows_task_",
        "execute_issue_",
        "conv_engine_exec_",
        "vec_alu_engine_exec_",
        "pool_engine_exec_",
        "upsample_engine_exec_",
        "vec_alu_apply_stage_",
        "shared_add_affine_vec_stage_",
        "shared_add_affine_vec_stage_canonical_",
        "shared_add_vec_engine_",
        "shared_affine_vec_engine_",
    )
    for root in banned_clone_roots:
        if root in text:
            failures.append(f"report contains banned clone root {root}")

    banned_patterns = (
        r"\(5\s+calls?\)",
        r"\(25\s+calls?\)",
    )
    for pattern in banned_patterns:
        if re.search(pattern, text):
            failures.append(f"report still contains suspicious marker {pattern}")

    if failures:
        print("FAIL:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("PASS: no watched heavy clone/call-explosion markers found")
    return 0


def self_test() -> int:
    bad = "execute_issue_1\nconv_engine_exec (2 calls)\nshared_affine_vec_engine_3\n"
    good = "execute_issue\nconv_engine_exec\nshared_affine_vec_engine\n"

    assert find_clone_names(bad, "execute_issue") == ["execute_issue_1"]
    assert find_clone_names(bad, "shared_affine_vec_engine") == ["shared_affine_vec_engine_3"]
    assert count_call_markers(bad, "conv_engine_exec") == [2]
    assert find_clone_names(good, "execute_issue") == []
    assert count_call_markers(good, "conv_engine_exec") == []
    print("PASS: parse_hls_design_size self-test")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--self-test", action="store_true")
    known, _ = parser.parse_known_args()
    if known.self_test:
        raise SystemExit(self_test())
    raise SystemExit(main())
