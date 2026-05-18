#!/usr/bin/env python3
"""Offline utilization model for the ESP INT8 HLS NPU.

The script parses the exported uop_table.bin and estimates:
- arithmetic lane fill of the TMxTK MAC array for every CONV uop;
- a simple lower-bound SA schedule model for the current row-stream datapath;
- end-to-end effective MAC utilization using measured board prefix cycles.

It intentionally does not require HLS or board access.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


UOP_LOAD_FM = 1
UOP_CONV = 2
UOP_POOL = 3
UOP_ADD = 4
UOP_AFFINE = 5
UOP_STORE = 6
UOP_END = 15

OPCODE_NAME = {
    UOP_LOAD_FM: "LOAD_FM",
    UOP_CONV: "CONV",
    UOP_POOL: "POOL",
    UOP_ADD: "ADD",
    UOP_AFFINE: "AFFINE",
    UOP_STORE: "STORE",
    UOP_END: "END",
}

DEFAULT_ARTIFACT_DIR = Path("D:/ESP_INT8/hw_artifacts/hw_constrained_qat_3ep_single")

# Latest board profile captured on 2026-05-13 after packed-read hardware update.
# These are cumulative prefix runs from the app, not one full MODE_RUN.
DEFAULT_PREFIX_CYCLES: Dict[int, int] = {
    0: 183_541,
    1: 4_867_898,
    2: 11_282_428,
    3: 12_418_562,
    4: 13_685_769,
    20: 41_949_124,
    39: 64_419_817,
    53: 73_345_908,
    69: 80_899_282,
    72: 81_841_561,
}


@dataclass
class Uop:
    index: int
    opcode: int
    flags: int
    src0: int
    src1: int
    dst: int
    param_id: int
    act_type: int
    in_h: int
    in_w: int
    in_c: int
    out_c: int
    kernel: int
    stride: int
    dilation: int
    padding: int
    c_offset: int
    valid_c: int
    qparam_id: int

    @property
    def opcode_name(self) -> str:
        return OPCODE_NAME.get(self.opcode, f"OP{self.opcode}")


@dataclass
class ConvStats:
    uop: int
    param_id: int
    src0: int
    dst: int
    in_shape: str
    out_shape: str
    kernel: int
    stride: int
    dilation: int
    out_pixels: int
    k_total: int
    k_tiles: int
    oc_tiles: int
    useful_macs: int
    pe_slot_macs: int
    arithmetic_fill: float
    sa_mac_occupancy_bound: float
    combined_sa_slot_bound: float
    row_weight_load_cycles: int
    sa_compute_cycles_lower_bound: int
    act_read_segments_per_pixel: int
    act_words_per_pixel: int
    read_segments_per_act_word: float


@dataclass
class PrefixStats:
    stop_after: int
    cycles: int
    useful_macs: int
    pe_capacity_slots: int
    end_to_end_util: float
    ms_at_100mhz: float


@dataclass
class IntervalStats:
    start_after: int
    stop_after: int
    cycles_delta: int
    useful_macs_delta: int
    pe_capacity_slots: int
    end_to_end_util: float
    ms_at_100mhz: float


@dataclass
class StageStats:
    name: str
    start_uop: int
    stop_uop: int
    conv_count: int
    measured_cycles: Optional[int]
    measured_ms_at_100mhz: Optional[float]
    useful_macs: int
    pe_slot_macs: int
    arithmetic_fill: float
    sa_compute_cycles_lower_bound: int
    end_to_end_util: Optional[float]
    measured_vs_sa_lower_bound: Optional[float]


DEFAULT_STAGE_RANGES: List[Tuple[str, int, int]] = [
    ("input/pool/front", 0, 4),
    ("level2_0", 5, 20),
    ("level2_b0_plus_b2_merge", 21, 39),
    ("level3_0", 40, 53),
    ("level3_b0", 54, 69),
    ("classifier", 70, 72),
]


def conv_out_dim(in_size: int, stride: int) -> int:
    stride = stride or 1
    return (in_size + stride - 1) // stride


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def parse_uops(path: Path) -> List[Uop]:
    data = path.read_bytes()
    if len(data) % 32 != 0:
        raise ValueError(f"{path} size {len(data)} is not a multiple of 32")

    uops: List[Uop] = []
    for index in range(len(data) // 32):
        fields = struct.unpack("<8B4H4B4HI", data[index * 32 : (index + 1) * 32])
        uops.append(
            Uop(
                index=index,
                opcode=fields[0],
                flags=fields[1],
                src0=fields[2],
                src1=fields[3],
                dst=fields[4],
                param_id=fields[5],
                act_type=fields[6],
                in_h=fields[8],
                in_w=fields[9],
                in_c=fields[10],
                out_c=fields[11],
                kernel=fields[12],
                stride=fields[13],
                dilation=fields[14],
                padding=fields[15],
                c_offset=fields[16],
                valid_c=fields[17],
                qparam_id=fields[18],
            )
        )
    return uops


def act_read_segments_per_pixel(uop: Uop, tk: int) -> int:
    kernel = 1 if uop.kernel == 1 else 3
    if kernel == 1:
        return ceil_div(uop.in_c, tk)
    # Current fast 3x3 path reads contiguous channel segments per spatial point.
    return kernel * kernel * ceil_div(uop.in_c, tk)


def conv_stats(uop: Uop, tm: int, tk: int) -> ConvStats:
    kernel = 1 if uop.kernel == 1 else 3
    stride = uop.stride or 1
    out_h = conv_out_dim(uop.in_h, stride)
    out_w = conv_out_dim(uop.in_w, stride)
    out_pixels = out_h * out_w
    k_total = uop.in_c * kernel * kernel
    k_tiles = ceil_div(k_total, tk)
    oc_tiles = ceil_div(uop.out_c, tm)
    useful_macs = out_pixels * uop.out_c * k_total
    pe_slot_macs = out_pixels * oc_tiles * tm * k_tiles * tk
    arithmetic_fill = useful_macs / pe_slot_macs if pe_slot_macs else 0.0

    # systolic_array_core_row has a kt loop plus a serial TM psum emit loop per pixel,
    # and reloads k_tiles*TM weight vectors for each row. This is a lower bound because
    # window generation, stream stalls, post-process, and memory writeback are omitted.
    row_weight_load_cycles = out_h * oc_tiles * k_tiles * tm
    pixel_mac_cycles = out_pixels * oc_tiles * k_tiles
    pixel_psum_cycles = out_pixels * oc_tiles * tm
    sa_compute_cycles_lower_bound = row_weight_load_cycles + pixel_mac_cycles + pixel_psum_cycles
    sa_mac_occupancy_bound = (
        pixel_mac_cycles / sa_compute_cycles_lower_bound if sa_compute_cycles_lower_bound else 0.0
    )
    combined_sa_slot_bound = arithmetic_fill * sa_mac_occupancy_bound
    read_segments = act_read_segments_per_pixel(uop, tk)

    return ConvStats(
        uop=uop.index,
        param_id=uop.param_id,
        src0=uop.src0,
        dst=uop.dst,
        in_shape=f"{uop.in_h}x{uop.in_w}x{uop.in_c}",
        out_shape=f"{out_h}x{out_w}x{uop.out_c}",
        kernel=kernel,
        stride=stride,
        dilation=uop.dilation or 1,
        out_pixels=out_pixels,
        k_total=k_total,
        k_tiles=k_tiles,
        oc_tiles=oc_tiles,
        useful_macs=useful_macs,
        pe_slot_macs=pe_slot_macs,
        arithmetic_fill=arithmetic_fill,
        sa_mac_occupancy_bound=sa_mac_occupancy_bound,
        combined_sa_slot_bound=combined_sa_slot_bound,
        row_weight_load_cycles=row_weight_load_cycles,
        sa_compute_cycles_lower_bound=sa_compute_cycles_lower_bound,
        act_read_segments_per_pixel=read_segments,
        act_words_per_pixel=k_tiles,
        read_segments_per_act_word=read_segments / k_tiles if k_tiles else 0.0,
    )


def prefix_stats(conv_rows: List[ConvStats], prefix_cycles: Dict[int, int], tm: int, tk: int) -> List[PrefixStats]:
    rows: List[PrefixStats] = []
    for stop, cycles in sorted(prefix_cycles.items()):
        useful = sum(row.useful_macs for row in conv_rows if row.uop <= stop)
        capacity = cycles * tm * tk
        rows.append(
            PrefixStats(
                stop_after=stop,
                cycles=cycles,
                useful_macs=useful,
                pe_capacity_slots=capacity,
                end_to_end_util=useful / capacity if capacity else 0.0,
                ms_at_100mhz=cycles / 100_000.0,
            )
        )
    return rows


def interval_stats(conv_rows: List[ConvStats], prefix_cycles: Dict[int, int], tm: int, tk: int) -> List[IntervalStats]:
    stops = sorted(prefix_cycles)
    rows: List[IntervalStats] = []
    prev_stop = -1
    prev_cycles = 0
    prev_useful = 0
    for stop in stops:
        cycles = prefix_cycles[stop]
        useful = sum(row.useful_macs for row in conv_rows if row.uop <= stop)
        delta_cycles = cycles - prev_cycles
        delta_useful = useful - prev_useful
        capacity = delta_cycles * tm * tk
        rows.append(
            IntervalStats(
                start_after=prev_stop,
                stop_after=stop,
                cycles_delta=delta_cycles,
                useful_macs_delta=delta_useful,
                pe_capacity_slots=capacity,
                end_to_end_util=delta_useful / capacity if capacity else 0.0,
                ms_at_100mhz=delta_cycles / 100_000.0,
            )
        )
        prev_stop = stop
        prev_cycles = cycles
        prev_useful = useful
    return rows


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


def stage_stats(conv_rows: List[ConvStats], prefix_cycles: Dict[int, int], tm: int, tk: int) -> List[StageStats]:
    rows: List[StageStats] = []
    for name, start, stop in DEFAULT_STAGE_RANGES:
        convs = [row for row in conv_rows if start <= row.uop <= stop]
        useful = sum(row.useful_macs for row in convs)
        slots = sum(row.pe_slot_macs for row in convs)
        lower = sum(row.sa_compute_cycles_lower_bound for row in convs)
        measured = cycles_for_range(prefix_cycles, start, stop)
        rows.append(
            StageStats(
                name=name,
                start_uop=start,
                stop_uop=stop,
                conv_count=len(convs),
                measured_cycles=measured,
                measured_ms_at_100mhz=(measured / 100_000.0) if measured is not None else None,
                useful_macs=useful,
                pe_slot_macs=slots,
                arithmetic_fill=(useful / slots) if slots else 0.0,
                sa_compute_cycles_lower_bound=lower,
                end_to_end_util=(useful / (measured * tm * tk)) if measured else None,
                measured_vs_sa_lower_bound=(measured / lower) if measured and lower else None,
            )
        )
    return rows


def parse_profile_json(path: Optional[Path]) -> Dict[int, int]:
    if path is None:
        return dict(DEFAULT_PREFIX_CYCLES)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return {int(k): int(v) for k, v in raw.items()}
    if isinstance(raw, list):
        return {int(item["stop_after"]): int(item["cycles"]) for item in raw}
    raise ValueError("profile must be a dict {stop_after: cycles} or a list of objects")


def write_csv(path: Path, rows: Iterable[object]) -> None:
    row_dicts = [asdict(row) for row in rows]
    if not row_dicts:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dicts[0].keys()))
        writer.writeheader()
        writer.writerows(row_dicts)


def print_summary(
    conv_rows: List[ConvStats],
    prefixes: List[PrefixStats],
    intervals: List[IntervalStats],
    stages: List[StageStats],
) -> None:
    useful_total = sum(row.useful_macs for row in conv_rows)
    slot_total = sum(row.pe_slot_macs for row in conv_rows)
    lower_cycles_total = sum(row.sa_compute_cycles_lower_bound for row in conv_rows)

    print("=== SA Utilization Summary ===")
    print(f"conv uops: {len(conv_rows)}")
    print(f"useful MACs: {useful_total:,}")
    print(f"PE slot MACs from tiling: {slot_total:,}")
    print(f"weighted arithmetic fill: {useful_total / slot_total:.4%}")
    print(f"SA row-core lower-bound cycles: {lower_cycles_total:,} ({lower_cycles_total / 100_000:.2f} ms @100MHz)")
    if prefixes:
        last = prefixes[-1]
        print(
            f"latest prefix U{last.stop_after:02d}: {last.cycles:,} cycles "
            f"({last.ms_at_100mhz:.2f} ms), e2e useful-MAC/PE-slot util={last.end_to_end_util:.4%}"
        )

    print("\n=== Lowest arithmetic-fill CONV uops ===")
    for row in sorted(conv_rows, key=lambda r: r.arithmetic_fill)[:8]:
        print(
            f"U{row.uop:02d} p{row.param_id:02d} {row.in_shape}->{row.out_shape} "
            f"k{row.kernel} K={row.k_total} kt={row.k_tiles} fill={row.arithmetic_fill:.2%} "
            f"SA_bound={row.combined_sa_slot_bound:.2%} reads/act={row.read_segments_per_act_word:.2f}"
        )

    print("\n=== Largest useful-MAC CONV uops ===")
    for row in sorted(conv_rows, key=lambda r: r.useful_macs, reverse=True)[:8]:
        print(
            f"U{row.uop:02d} p{row.param_id:02d} useful={row.useful_macs/1e6:.1f}M "
            f"{row.in_shape}->{row.out_shape} k{row.kernel} fill={row.arithmetic_fill:.2%} "
            f"reads/pixel={row.act_read_segments_per_pixel}"
        )

    print("\n=== Prefix / interval effective utilization ===")
    for prefix in prefixes:
        print(
            f"prefix U{prefix.stop_after:02d}: {prefix.ms_at_100mhz:8.2f} ms "
            f"util={prefix.end_to_end_util:7.3%} useful={prefix.useful_macs/1e6:8.1f}M"
        )
    print("")
    for interval in intervals:
        print(
            f"interval U{interval.start_after + 1:02d}-U{interval.stop_after:02d}: "
            f"{interval.ms_at_100mhz:8.2f} ms util={interval.end_to_end_util:7.3%} "
            f"useful={interval.useful_macs_delta/1e6:8.1f}M"
        )

    print("\n=== Stage bottleneck view ===")
    for stage in stages:
        measured_ms = "n/a" if stage.measured_ms_at_100mhz is None else f"{stage.measured_ms_at_100mhz:.2f}"
        e2e = "n/a" if stage.end_to_end_util is None else f"{stage.end_to_end_util:.3%}"
        ratio = "n/a" if stage.measured_vs_sa_lower_bound is None else f"{stage.measured_vs_sa_lower_bound:.2f}x"
        print(
            f"{stage.name:24s} U{stage.start_uop:02d}-U{stage.stop_uop:02d} "
            f"conv={stage.conv_count:2d} measured_ms={measured_ms:>8s} "
            f"useful={stage.useful_macs/1e6:8.1f}M fill={stage.arithmetic_fill:7.2%} "
            f"e2e={e2e:>8s} measured/SA_lb={ratio:>8s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--uop-table", type=Path, default=None)
    parser.add_argument("--profile-json", type=Path, default=None)
    parser.add_argument("--tm", type=int, default=32)
    parser.add_argument("--tk", type=int, default=32)
    parser.add_argument("--out-prefix", type=Path, default=None)
    args = parser.parse_args()

    uop_path = args.uop_table or (args.artifact_dir / "uop_table.bin")
    out_prefix = args.out_prefix or (args.artifact_dir / "sa_utilization")
    uops = parse_uops(uop_path)
    conv_rows = [conv_stats(uop, args.tm, args.tk) for uop in uops if uop.opcode == UOP_CONV]
    profile = parse_profile_json(args.profile_json)
    prefixes = prefix_stats(conv_rows, profile, args.tm, args.tk)
    intervals = interval_stats(conv_rows, profile, args.tm, args.tk)
    stages = stage_stats(conv_rows, profile, args.tm, args.tk)

    report = {
        "uop_table": str(uop_path),
        "tm": args.tm,
        "tk": args.tk,
        "prefix_cycles": profile,
        "conv": [asdict(row) for row in conv_rows],
        "prefix": [asdict(row) for row in prefixes],
        "interval": [asdict(row) for row in intervals],
        "stage": [asdict(row) for row in stages],
        "summary": {
            "conv_count": len(conv_rows),
            "useful_macs": sum(row.useful_macs for row in conv_rows),
            "pe_slot_macs": sum(row.pe_slot_macs for row in conv_rows),
            "weighted_arithmetic_fill": (
                sum(row.useful_macs for row in conv_rows) / sum(row.pe_slot_macs for row in conv_rows)
            ),
            "sa_compute_cycles_lower_bound": sum(row.sa_compute_cycles_lower_bound for row in conv_rows),
        },
    }

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    (out_prefix.with_suffix(".json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(out_prefix.with_name(out_prefix.name + "_conv.csv"), conv_rows)
    write_csv(out_prefix.with_name(out_prefix.name + "_prefix.csv"), prefixes)
    write_csv(out_prefix.with_name(out_prefix.name + "_interval.csv"), intervals)
    write_csv(out_prefix.with_name(out_prefix.name + "_stage.csv"), stages)
    print_summary(conv_rows, prefixes, intervals, stages)
    print(f"\nWrote {out_prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
