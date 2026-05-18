#!/usr/bin/env python3
"""Layer/stage performance attribution model for the ESP INT8 NPU.

This is a first-order offline model.  It separates what can be derived from
the exported uop schedule from what must be measured on board:

- per-CONV static estimates: activation reads, weight movement, window
  generation, SA MAC work, psum/output drain, writeback/RMW, PE fill;
- stage residuals: measured board cycles minus the model lower bound.

The residual is intentionally named "unattributed_or_stall" rather than
"FIFO stall" because exact FIFO empty/full cycles require hardware counters.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from analyze_sa_utilization import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_PREFIX_CYCLES,
    DEFAULT_STAGE_RANGES,
    UOP_CONV,
    Uop,
    act_read_segments_per_pixel,
    ceil_div,
    conv_out_dim,
    parse_profile_json,
    parse_uops,
)


TM = 32
TK = 32


@dataclass
class ConvAttribution:
    uop: int
    param_id: int
    src0: int
    dst: int
    in_shape: str
    out_shape: str
    kernel: int
    stride: int
    out_pixels: int
    k_total: int
    k_tiles: int
    out_c: int
    useful_macs: int
    pe_slot_macs: int
    pe_fill: float
    input_feature_read_ops_est: int
    input_feature_read_cycles_est: int
    weight_cache_cycles_est: int
    weight_stream_cycles_est: int
    window_pack_cycles_est: int
    window_gen_cycles_est: int
    sa_mac_cycles_est: int
    psum_output_drain_cycles_est: int
    output_write_ops_est: int
    output_write_cycles_est: int
    read_modify_write: str
    row_parallel_bound_cycles_est: int
    total_lower_bound_cycles_est: int
    dominant_estimated_component: str
    notes: str


@dataclass
class StageAttribution:
    name: str
    start_uop: int
    stop_uop: int
    measured_cycles: Optional[int]
    model_lower_bound_cycles: int
    unattributed_or_stall_cycles: Optional[int]
    measured_over_model: Optional[float]
    useful_macs: int
    pe_fill: float
    end_to_end_pe_util: Optional[float]
    dominant_estimated_component: str


def write_csv(path: Path, rows: Iterable[object]) -> None:
    row_dicts = [asdict(row) for row in rows]
    if not row_dicts:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dicts[0].keys()))
        writer.writeheader()
        writer.writerows(row_dicts)


def conv_dims(uop: Uop) -> Tuple[int, int, int, int, int, int]:
    kernel = 1 if uop.kernel == 1 else 3
    stride = int(uop.stride or 1)
    out_h = conv_out_dim(int(uop.in_h), stride)
    out_w = conv_out_dim(int(uop.in_w), stride)
    out_pixels = out_h * out_w
    k_total = int(uop.in_c) * kernel * kernel
    k_tiles = ceil_div(k_total, TK)
    return kernel, stride, out_h, out_w, out_pixels, k_total, k_tiles


def writeback_ops_per_pixel(dst: int, out_c: int, c_offset: int) -> Tuple[int, str]:
    """Estimate on-chip memory word operations for one output tile write.

    on_chip_memory_write_tile has three cases:
    - full aligned 32B write: one write op;
    - partial one-word write: read + write;
    - cross-word partial write: two reads + two writes.
    """

    byte0 = int(c_offset) & 31
    lane_count = out_c if out_c > 0 else TM
    padded_direct_dst = {4, 5, 6, 7, 11, 12, 13, 14, 15, 16}
    if byte0 == 0 and dst in padded_direct_dst and lane_count <= 32:
        return 1, "none_padded_word_overwrite"
    if byte0 == 0 and lane_count == 32:
        return 1, "none_full_word"
    if byte0 + lane_count <= 32:
        return 2, "one_word_rmw"
    return 4, "cross_word_rmw"


def compact_row_write_ops(uop: Uop, out_h: int, out_w: int) -> Optional[Tuple[int, str]]:
    """Return full-word row-write ops for HLS compact scratch fast paths."""

    dst = int(uop.dst)
    out_c = int(uop.out_c)
    if int(uop.c_offset) != 0:
        return None
    local_dst = dst in {0x80, 0x81, 0x82, 0x83}
    if out_c == 2 and dst == 17 and out_w % 16 == 0:
        return out_h * (out_w // 16), "compact_c2_row_fullword"
    if out_c == 16 and dst == 2 and out_w % 32 == 0:
        return out_h * (out_w // 32) * 19, "compact_c16_into_c19_row_fullword"
    if not local_dst:
        return None
    if out_c == 12 and out_w % 8 == 0:
        return out_h * (out_w // 8) * 3, "compact_c12_row_fullword"
    if out_c == 16 and out_w % 2 == 0:
        return out_h * (out_w // 2), "compact_c16_row_fullword"
    if out_c == 25 and out_w % 32 == 0:
        return out_h * (out_w // 32) * 25, "compact_c25_row_fullword"
    if out_c == 28 and out_w % 8 == 0:
        return out_h * (out_w // 8) * 7, "compact_c28_row_fullword"
    return None


def window_read_ops_est(uop: Uop,
                        out_h: int,
                        out_w: int,
                        out_pixels: int,
                        kernel: int,
                        tk: int) -> Tuple[int, str]:
    in_c = int(uop.in_c)
    stride = int(uop.stride or 1)
    dilation = int(uop.dilation or 1)
    if kernel == 1:
        return out_pixels * ceil_div(in_c, tk), "1x1_ktile_read"
    if in_c == 3 and stride == 2 and dilation == 1:
        inner = max(out_w - 1, 0)
        row_reads = 9 + (inner // 2) * 3 + (inner & 1) * 3 if out_w else 0
        return out_h * row_reads, "first_layer_3x3_stride2_pair_segment"
    if in_c == 19 and stride == 2 and dilation == 1:
        if out_h <= 0 or out_w <= 0:
            return 0, "c19_3x3_stride2_pair_segment"
        top_row_reads = out_w * 9
        inner = out_w - 1
        inner_row_reads = 9 + (inner // 2) * 9 + (inner & 1) * 6
        return top_row_reads + (out_h - 1) * inner_row_reads, "c19_3x3_stride2_pair_segment"
    if in_c <= 32 and stride == 1 and dilation == 1:
        row_reads = 9 + (out_w - 1) * 3 if out_w else 0
        return out_h * row_reads, "smallc_3x3_stride1_reuse"
    return out_pixels * act_read_segments_per_pixel(uop, tk), "generic_segment_read"


def window_pack_cycles_est(uop: Uop,
                           out_h: int,
                           out_w: int,
                           out_pixels: int,
                           k_tiles: int,
                           kernel: int) -> Tuple[int, str]:
    """Estimate cycles spent packing activation words in the current win_gen.

    This is intentionally conservative and code-path aware:
    - first layer 3x3 stride-2 path reuses one column between neighbors;
    - small-C 3x3 path still has an inner lane packing loop with II=1;
    - 1x1 path emits one packed word per ktile with II=1;
    - general 3x3 path is modeled by packed segment reads.
    """

    in_c = int(uop.in_c)
    if kernel == 1:
        return out_pixels * k_tiles, "1x1_fast_ktile_ii1"
    if in_c == 3 and int(uop.stride or 1) == 2:
        inner = max(out_w - 1, 0)
        row_reads = 9 + (inner // 2) * 3 + (inner & 1) * 3 if out_w else 0
        return out_h * row_reads, "first_layer_3x3_stride2_pair_segment"
    if in_c == 19 and int(uop.stride or 1) == 2 and int(uop.dilation or 1) == 1:
        if out_h <= 0 or out_w <= 0:
            return 0, "c19_3x3_stride2_pair_segment"
        top_row_reads = out_w * 9
        inner = out_w - 1
        inner_row_reads = 9 + (inner // 2) * 9 + (inner & 1) * 6
        return top_row_reads + (out_h - 1) * inner_row_reads, "c19_3x3_stride2_pair_segment"
    if in_c <= 32:
        segments_per_kt = min(9, ceil_div(TK, max(in_c, 1)))
        return out_pixels * k_tiles * segments_per_kt, "smallc_3x3_segment_pack"
    return out_pixels * act_read_segments_per_pixel(uop, TK), "largec_3x3_segment_pack"


def conv_attribution(uop: Uop, tm: int, tk: int) -> ConvAttribution:
    kernel, stride, out_h, out_w, out_pixels, k_total, k_tiles = conv_dims(uop)
    out_c = int(uop.out_c)
    oc_tiles = ceil_div(out_c, tm)
    useful_macs = out_pixels * out_c * k_total
    pe_slot_macs = out_pixels * oc_tiles * tm * k_tiles * tk
    pe_fill = useful_macs / pe_slot_macs if pe_slot_macs else 0.0

    input_read_ops, input_read_note = window_read_ops_est(uop, out_h, out_w, out_pixels, kernel, tk)
    input_read_cycles = input_read_ops

    # Current stream path caches param weights once per uop, but feeds the row
    # dataflow region once per output row.
    weight_cache_cycles = oc_tiles * k_tiles * tm
    weight_stream_cycles = out_h * oc_tiles * k_tiles * tm

    win_pack, win_note = window_pack_cycles_est(uop, out_h, out_w, out_pixels, k_tiles, kernel)
    window_gen_cycles = max(input_read_cycles, win_pack)

    sa_mac_cycles = out_pixels * oc_tiles * k_tiles
    post_drain_per_pixel = ceil_div(out_c, 4)
    psum_drain_cycles = out_pixels * post_drain_per_pixel
    write_ops_per_pixel, rmw_mode = writeback_ops_per_pixel(int(uop.dst), out_c, int(uop.c_offset))
    compact_write = compact_row_write_ops(uop, out_h, out_w)
    if compact_write is not None:
        output_write_ops, rmw_mode = compact_write
    else:
        output_write_ops = out_pixels * write_ops_per_pixel
    output_write_cycles = output_write_ops

    # Row-level dataflow bound: window generation overlaps with row SA/post;
    # store_conv_output_row happens after the row dataflow region returns.
    win_row = ceil_div(window_gen_cycles, out_h) if out_h else 0
    row_core = oc_tiles * k_tiles * tm + out_w * (oc_tiles * k_tiles + post_drain_per_pixel)
    write_row = ceil_div(output_write_ops, out_h) if out_h else 0
    row_parallel_bound = max(win_row, row_core)
    total_lower_bound = weight_cache_cycles + out_h * (row_parallel_bound + write_row)

    components = {
        "window_gen": window_gen_cycles,
        "weight_stream": weight_stream_cycles,
        "sa_mac": sa_mac_cycles,
        "psum_output_drain": psum_drain_cycles,
        "output_write": output_write_cycles,
    }
    dominant = max(components, key=components.get)

    return ConvAttribution(
        uop=int(uop.index),
        param_id=int(uop.param_id),
        src0=int(uop.src0),
        dst=int(uop.dst),
        in_shape=f"{int(uop.in_h)}x{int(uop.in_w)}x{int(uop.in_c)}",
        out_shape=f"{out_h}x{out_w}x{out_c}",
        kernel=kernel,
        stride=stride,
        out_pixels=out_pixels,
        k_total=k_total,
        k_tiles=k_tiles,
        out_c=out_c,
        useful_macs=useful_macs,
        pe_slot_macs=pe_slot_macs,
        pe_fill=pe_fill,
        input_feature_read_ops_est=input_read_ops,
        input_feature_read_cycles_est=input_read_cycles,
        weight_cache_cycles_est=weight_cache_cycles,
        weight_stream_cycles_est=weight_stream_cycles,
        window_pack_cycles_est=win_pack,
        window_gen_cycles_est=window_gen_cycles,
        sa_mac_cycles_est=sa_mac_cycles,
        psum_output_drain_cycles_est=psum_drain_cycles,
        output_write_ops_est=output_write_ops,
        output_write_cycles_est=output_write_cycles,
        read_modify_write=rmw_mode,
        row_parallel_bound_cycles_est=row_parallel_bound * out_h,
        total_lower_bound_cycles_est=total_lower_bound,
        dominant_estimated_component=dominant,
        notes=f"{input_read_note}; {win_note}",
    )


def cycles_for_range(prefix_cycles: Dict[int, int], start: int, stop: int) -> Optional[int]:
    if stop not in prefix_cycles:
        return None
    before = 0
    for key in sorted(prefix_cycles):
        if key < start:
            before = prefix_cycles[key]
        else:
            break
    return prefix_cycles[stop] - before


def stage_attribution(
    conv_rows: List[ConvAttribution],
    profile: Dict[int, int],
    tm: int,
    tk: int,
) -> List[StageAttribution]:
    rows: List[StageAttribution] = []
    for name, start, stop in DEFAULT_STAGE_RANGES:
        convs = [row for row in conv_rows if start <= row.uop <= stop]
        model = sum(row.total_lower_bound_cycles_est for row in convs)
        useful = sum(row.useful_macs for row in convs)
        slots = sum(row.pe_slot_macs for row in convs)
        measured = cycles_for_range(profile, start, stop)
        residual = (measured - model) if measured is not None else None
        dominant = "none"
        if convs:
            component_sums = {
                "window_gen": sum(row.window_gen_cycles_est for row in convs),
                "weight_stream": sum(row.weight_stream_cycles_est for row in convs),
                "sa_mac": sum(row.sa_mac_cycles_est for row in convs),
                "psum_output_drain": sum(row.psum_output_drain_cycles_est for row in convs),
                "output_write": sum(row.output_write_cycles_est for row in convs),
            }
            dominant = max(component_sums, key=component_sums.get)
        rows.append(
            StageAttribution(
                name=name,
                start_uop=start,
                stop_uop=stop,
                measured_cycles=measured,
                model_lower_bound_cycles=model,
                unattributed_or_stall_cycles=residual,
                measured_over_model=(measured / model) if measured and model else None,
                useful_macs=useful,
                pe_fill=(useful / slots) if slots else 0.0,
                end_to_end_pe_util=(useful / (measured * tm * tk)) if measured else None,
                dominant_estimated_component=dominant,
            )
        )
    return rows


def parse_profile_text(path: Optional[Path]) -> Optional[Dict[int, int]]:
    if path is None:
        return None
    profile: Dict[int, int] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        # Expected app table line:
        # APP: DBG_U20_L20            20     63442548        634
        parts = line.strip().split()
        if len(parts) >= 5 and parts[0] == "APP:" and parts[1].startswith("DBG_U"):
            try:
                stop = int(parts[2])
                cycles = int(parts[3])
            except ValueError:
                continue
            profile[stop] = cycles
        if len(parts) >= 5 and parts[0] == "APP:" and parts[1] == "FULL_MODE_RUN":
            try:
                stop = int(parts[2])
                cycles = int(parts[3])
            except ValueError:
                continue
            profile[stop] = cycles
    return profile or None


def parse_hw_counters_text(path: Optional[Path]) -> Dict[str, int]:
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    counters: Dict[str, int] = {}
    for key in (
        "uop",
        "conv",
        "win_read",
        "win_words",
        "wgt",
        "sa_steps",
        "psum",
        "out_tiles",
        "rmw_ops",
        "model_cycles",
        "wgt_cyc",
        "win_cyc",
        "sa_cyc",
        "post_cyc",
        "write_cyc",
        "row_region_cyc",
    ):
        match = re.search(rf"\b{key}=(\d+)", text)
        if match:
            counters[key] = int(match.group(1))
    return counters


def print_summary(conv_rows: List[ConvAttribution], stages: List[StageAttribution]) -> None:
    total_useful = sum(row.useful_macs for row in conv_rows)
    total_slots = sum(row.pe_slot_macs for row in conv_rows)
    total_model = sum(row.total_lower_bound_cycles_est for row in conv_rows)
    print("=== Performance Attribution Model ===")
    print(f"conv uops: {len(conv_rows)}")
    print(f"useful MACs: {total_useful:,}")
    print(f"weighted PE fill: {total_useful / total_slots:.2%}")
    print(f"sum model lower bound: {total_model:,} cycles ({total_model / 100_000:.2f} ms @100MHz)")

    print("\n=== Highest model-cost CONV uops ===")
    for row in sorted(conv_rows, key=lambda r: r.total_lower_bound_cycles_est, reverse=True)[:10]:
        print(
            f"U{row.uop:02d} {row.in_shape}->{row.out_shape} k{row.kernel} "
            f"model={row.total_lower_bound_cycles_est/100_000:8.2f}ms "
            f"dom={row.dominant_estimated_component:18s} "
            f"PEfill={row.pe_fill:6.2%} RMW={row.read_modify_write}"
        )

    print("\n=== Stage measured residuals ===")
    for row in stages:
        measured = "n/a" if row.measured_cycles is None else f"{row.measured_cycles/100_000:.2f}ms"
        model = f"{row.model_lower_bound_cycles/100_000:.2f}ms"
        residual = "n/a" if row.unattributed_or_stall_cycles is None else f"{row.unattributed_or_stall_cycles/100_000:.2f}ms"
        ratio = "n/a" if row.measured_over_model is None else f"{row.measured_over_model:.2f}x"
        e2e = "n/a" if row.end_to_end_pe_util is None else f"{row.end_to_end_pe_util:.3%}"
        print(
            f"{row.name:24s} U{row.start_uop:02d}-U{row.stop_uop:02d} "
            f"meas={measured:>9s} model={model:>9s} residual={residual:>9s} "
            f"ratio={ratio:>6s} e2e={e2e:>8s} dom={row.dominant_estimated_component}"
        )


def print_full_run_summary(profile: Dict[int, int], hw_counters: Dict[str, int]) -> None:
    full_cycles = profile.get(75)
    model_cycles = hw_counters.get("model_cycles")
    if full_cycles is None or model_cycles is None:
        return
    residual = full_cycles - model_cycles
    ratio = full_cycles / model_cycles if model_cycles else 0.0
    print("\n=== Full-run actual vs hardware model ===")
    print(f"actual_cycles : {full_cycles:,} ({full_cycles / 100_000:.2f} ms @100MHz)")
    print(f"model_cycles  : {model_cycles:,} ({model_cycles / 100_000:.2f} ms @100MHz)")
    print(f"residual      : {residual:,} ({residual / 100_000:.2f} ms @100MHz)")
    print(f"actual/model  : {ratio:.3f}x")


def print_prof3_phase_summary(profile: Dict[int, int], hw_counters: Dict[str, int]) -> None:
    full_cycles = profile.get(75)
    row_region = hw_counters.get("row_region_cyc")
    write = hw_counters.get("write_cyc")
    if full_cycles is None or row_region is None or write is None:
        return
    phase_model = row_region + write
    residual = full_cycles - phase_model
    ratio = full_cycles / phase_model if phase_model else 0.0
    print("\n=== Profiling v3 phase model ===")
    for key, label in (
        ("wgt_cyc", "weight"),
        ("win_cyc", "window"),
        ("sa_cyc", "sa"),
        ("post_cyc", "post"),
        ("write_cyc", "write"),
        ("row_region_cyc", "row_region"),
    ):
        value = hw_counters.get(key)
        if value is not None:
            print(f"{label:12s}: {value:,} ({value / 100_000:.2f} ms @100MHz)")
    print(f"row+write   : {phase_model:,} ({phase_model / 100_000:.2f} ms @100MHz)")
    print(f"actual-row  : {residual:,} ({residual / 100_000:.2f} ms @100MHz)")
    print(f"actual/phase: {ratio:.3f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--uop-table", type=Path, default=None)
    parser.add_argument("--profile-json", type=Path, default=None)
    parser.add_argument("--profile-log", type=Path, default=None)
    parser.add_argument("--tm", type=int, default=TM)
    parser.add_argument("--tk", type=int, default=TK)
    parser.add_argument("--out-prefix", type=Path, default=None)
    args = parser.parse_args()

    uop_path = args.uop_table or (args.artifact_dir / "uop_table.bin")
    out_prefix = args.out_prefix or (args.artifact_dir / "perf_attribution")

    profile = parse_profile_text(args.profile_log)
    hw_counters = parse_hw_counters_text(args.profile_log)
    if profile is None:
        profile = parse_profile_json(args.profile_json) if args.profile_json else dict(DEFAULT_PREFIX_CYCLES)

    uops = parse_uops(uop_path)
    conv_rows = [conv_attribution(uop, args.tm, args.tk) for uop in uops if uop.opcode == UOP_CONV]
    stages = stage_attribution(conv_rows, profile, args.tm, args.tk)

    report = {
        "uop_table": str(uop_path),
        "profile_cycles": profile,
        "hw_counters": hw_counters,
        "full_run": {
            "actual_cycles": profile.get(75),
            "model_cycles": hw_counters.get("model_cycles"),
            "prof3_phase_model_cycles": (
                hw_counters["row_region_cyc"] + hw_counters["write_cyc"]
                if "row_region_cyc" in hw_counters and "write_cyc" in hw_counters
                else None
            ),
            "residual_cycles": (profile.get(75) - hw_counters["model_cycles"])
            if profile.get(75) is not None and "model_cycles" in hw_counters
            else None,
        },
        "model_notes": {
            "input_feature_read_cycles_est": "logical packed tile read calls; exact bank stalls require counters",
            "window_gen_cycles_est": "code-path-aware estimate for current win_gen packing loops",
            "weight_stream_cycles_est": "cached weights are still streamed once per output row",
            "psum_output_drain_cycles_est": "SA/post loops drain only valid output channels",
            "unattributed_or_stall_cycles": "measured stage cycles minus model lower bound; not purely FIFO stall",
        },
        "conv": [asdict(row) for row in conv_rows],
        "stage": [asdict(row) for row in stages],
    }

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(out_prefix.with_name(out_prefix.name + "_conv.csv"), conv_rows)
    write_csv(out_prefix.with_name(out_prefix.name + "_stage.csv"), stages)
    print_summary(conv_rows, stages)
    if hw_counters:
        print("\n=== Hardware profile counters from log ===")
        for key in sorted(hw_counters):
            print(f"{key:14s}: {hw_counters[key]:,}")
        print_full_run_summary(profile, hw_counters)
        print_prof3_phase_summary(profile, hw_counters)
    print(f"\nWrote {out_prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
