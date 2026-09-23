#!/usr/bin/env python3
"""Convert exec-prefix RTL logs into per-exec Conv attribution rows."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping


WIN_COLUMNS = ("cw0_idle_done", "cw1_active")
SA_COLUMNS = (
    "cs0_idle_done",
    "cs1_compute_wait_act",
    "cs2_emit_wait",
)
POST_COLUMNS = (
    "cp0_idle_done",
    "cp1_requant_wait",
    "cp2_write",
)


def _as_int(value: str | int | None, default: int = 0) -> int:
    if value is None or value == "" or value == "NA":
        return default
    if isinstance(value, int):
        return value
    return int(value, 0)


def parse_prefix_lines(lines: Iterable[str]) -> list[dict[str, str]]:
    """Read PREFIX_CUM/PREFIX_DELTA records from an app serial log."""
    header: list[str] | None = None
    records: list[dict[str, str]] = []
    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("record,prefix,"):
            header = next(csv.reader([line]))
            continue
        if not line.startswith(("PREFIX_CUM,", "PREFIX_DELTA,")):
            continue
        if header is None:
            raise ValueError("prefix CSV record appeared before its header")
        values = next(csv.reader([line]))
        if len(values) != len(header):
            raise ValueError(
                f"prefix CSV width mismatch: header={len(header)} row={len(values)}"
            )
        records.append(dict(zip(header, values)))
    return records


def _state_values(record: Mapping[str, str], columns: tuple[str, ...]) -> list[int]:
    return [_as_int(record.get(column)) for column in columns]


def _amplification(total: int, useful: int | None) -> float | None:
    if useful is None or useful <= 0:
        return None
    return total / useful


def build_useful_cycle_model(artifact_dir: Path) -> dict[int, dict[str, int]]:
    """Build per-exec lower bounds from the compiled PARAM/uop contract."""
    try:
        from tools.analyze_sa_utilization import (
            UOP_CONV,
            conv_stats,
            load_precision_contract,
            load_window_schedule_contract,
            parse_uops,
        )
    except ModuleNotFoundError:
        from analyze_sa_utilization import (  # type: ignore[no-redef]
            UOP_CONV,
            conv_stats,
            load_precision_contract,
            load_window_schedule_contract,
            parse_uops,
        )

    uops = {uop.index: uop for uop in parse_uops(artifact_dir / "uop_table.bin")}
    precision = load_precision_contract(artifact_dir)
    schedules = load_window_schedule_contract(artifact_dir)
    audit = json.loads((artifact_dir / "fusion_audit.json").read_text(encoding="utf-8"))
    exec_plan = list((artifact_dir / "exec_plan.bin").read_bytes())
    if len(exec_plan) % 4 != 0:
        raise ValueError("exec_plan.bin size is not a multiple of four bytes")

    active_plan = []
    for offset in range(0, len(exec_plan), 4):
        kind, desc_id, logical_uop, flags = exec_plan[offset : offset + 4]
        if kind == 255:
            break
        active_plan.append((kind, desc_id, logical_uop, flags))
    if len(active_plan) != len(audit):
        raise ValueError(
            f"fusion audit/exec plan count mismatch: {len(audit)} != {len(active_plan)}"
        )

    result: dict[int, dict[str, int]] = {}
    for exec_index, (plan_entry, audit_entry) in enumerate(zip(active_plan, audit)):
        logical_uop = int(audit_entry["logical_uop"])
        accepted_logical_uops = {logical_uop}
        if "affine_uop" in audit_entry:
            accepted_logical_uops.add(int(audit_entry["affine_uop"]))
        if plan_entry[2] not in accepted_logical_uops:
            raise ValueError(
                f"exec {exec_index} logical uop mismatch: "
                f"audit={sorted(accepted_logical_uops)} plan={plan_entry[2]}"
            )

        kind_name = str(audit_entry["kind"])
        if "branch_uops" in audit_entry:
            conv_uop_ids = [int(value) for value in audit_entry["branch_uops"]]
        elif kind_name.startswith("CONV_"):
            conv_uop_ids = [logical_uop]
        else:
            continue

        owner_cycles = {"win": 0, "sa": 0, "post": 0}
        for uop_id in conv_uop_ids:
            uop = uops.get(uop_id)
            if uop is None or uop.opcode != UOP_CONV:
                raise ValueError(f"exec {exec_index} references non-CONV U{uop_id}")
            activation_bits, weight_bits = precision.get(uop.param_id, (8, 8))
            pixel_parallel, compiled_k_tiles = schedules.get(uop.param_id, (False, 0))
            stats = conv_stats(
                uop,
                tm=32,
                tk=32,
                activation_bits=activation_bits,
                weight_bits=weight_bits,
                pixel_parallel=pixel_parallel,
            )
            if compiled_k_tiles and compiled_k_tiles != stats.k_tiles:
                raise ValueError(
                    f"U{uop_id}: schedule k_tiles={compiled_k_tiles}, "
                    f"model={stats.k_tiles}"
                )
            issue_rows = stats.out_h * stats.issue_count_per_row * stats.oc_tiles
            owner_cycles["win"] += issue_rows * stats.k_tiles
            owner_cycles["sa"] += issue_rows * stats.sa_cycles_per_issue
            owner_cycles["post"] += issue_rows * stats.postprocess_cycles_per_issue
        result[exec_index] = owner_cycles
    return result


def build_attribution_rows(
    records: Iterable[Mapping[str, str]],
    useful_by_exec: Mapping[int, Mapping[str, int]] | None = None,
) -> list[dict[str, object]]:
    """Build one attribution row per PREFIX_DELTA record."""
    useful_by_exec = useful_by_exec or {}
    result: list[dict[str, object]] = []
    for record in records:
        if record.get("record") != "PREFIX_DELTA":
            continue

        exec_index = _as_int(record.get("new_exec_index"), -1)
        conv_cycles = _as_int(record.get("s5_conv"))
        win_values = _state_values(record, WIN_COLUMNS)
        sa_values = _state_values(record, SA_COLUMNS)
        post_values = _state_values(record, POST_COLUMNS)
        win_sum = sum(win_values)
        sa_sum = sum(sa_values)
        post_sum = sum(post_values)
        win_active = sum(win_values[1:])
        sa_active = sum(sa_values[1:])
        post_active = sum(post_values[1:])
        sums_match = (
            win_sum == conv_cycles
            and sa_sum == conv_cycles
            and post_sum == conv_cycles
        )

        useful = useful_by_exec.get(exec_index, {})
        win_useful = useful.get("win")
        sa_useful = useful.get("sa")
        post_useful = useful.get("post")
        overheads = {
            "WIN": win_active - win_useful if win_useful is not None else None,
            "SA": sa_active - sa_useful if sa_useful is not None else None,
            "POST": post_active - post_useful if post_useful is not None else None,
        }
        known_overheads = {
            owner: value for owner, value in overheads.items() if value is not None
        }
        # Active bins include blocking and pipeline residency. Their maximum
        # (or excess over an ideal model) cannot identify a causal bottleneck.
        limiter_owner = "NONE" if conv_cycles == 0 else "UNRESOLVED"
        largest_excess_owner = (max(known_overheads, key=known_overheads.get)
                                if conv_cycles and known_overheads else "NONE")

        row: dict[str, object] = {
            "prefix": record.get("prefix", ""),
            "new_exec_index": exec_index,
            "logical_uop": _as_int(record.get("logical_uop"), -1),
            "kind": _as_int(record.get("kind"), -1),
            "conv_cycles": conv_cycles,
            "win_sum": win_sum,
            "sa_sum": sa_sum,
            "post_sum": post_sum,
            "win_active": win_active,
            "sa_active": sa_active,
            "post_active": post_active,
            "owner_sums_match": sums_match,
            "win_useful": win_useful,
            "sa_useful": sa_useful,
            "post_useful": post_useful,
            "win_nonuseful": overheads["WIN"],
            "sa_nonuseful": overheads["SA"],
            "post_nonuseful": overheads["POST"],
            "win_amplification": _amplification(win_active, win_useful),
            "sa_amplification": _amplification(sa_active, sa_useful),
            "post_amplification": _amplification(post_active, post_useful),
            "limiter_owner": limiter_owner,
            "largest_excess_owner": largest_excess_owner,
            "attribution_status": "state_residency_not_causal_stall",
        }
        for column, value in zip(WIN_COLUMNS, win_values):
            row[column] = value
            row[f"{column}_pct"] = value / conv_cycles if conv_cycles else 0.0
        for column, value in zip(SA_COLUMNS, sa_values):
            row[column] = value
            row[f"{column}_pct"] = value / conv_cycles if conv_cycles else 0.0
        for column, value in zip(POST_COLUMNS, post_values):
            row[column] = value
            row[f"{column}_pct"] = value / conv_cycles if conv_cycles else 0.0
        result.append(row)
    return result


def _load_useful(path: Path | None) -> dict[int, dict[str, int]]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "exec" in raw:
        raw = raw["exec"]
    return {
        int(index): {owner: int(value) for owner, value in values.items()}
        for index, values in raw.items()
    }


def _write_rows(rows: list[dict[str, object]], output) -> None:
    if not rows:
        raise ValueError("no PREFIX_DELTA rows found")
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="serial log containing prefix CSV")
    parser.add_argument(
        "--useful-json",
        type=Path,
        help="optional per-exec {'win','sa','post'} theoretical cycle map",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="derive useful-cycle lower bounds from this compiled artifact",
    )
    parser.add_argument("--output", type=Path, help="write CSV to this path")
    args = parser.parse_args()

    if args.useful_json is not None and args.artifact_dir is not None:
        parser.error("use only one of --useful-json and --artifact-dir")
    useful = (
        build_useful_cycle_model(args.artifact_dir)
        if args.artifact_dir is not None
        else _load_useful(args.useful_json)
    )
    records = parse_prefix_lines(args.log.read_text(encoding="utf-8").splitlines())
    rows = build_attribution_rows(records, useful)
    if args.output is None:
        _write_rows(rows, sys.stdout)
    else:
        with args.output.open("w", encoding="utf-8", newline="") as output:
            _write_rows(rows, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
