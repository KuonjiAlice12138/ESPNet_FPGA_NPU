#!/usr/bin/env python3
"""Gate HLS package/export on report-level red flags."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PatternRule:
    name: str
    pattern: re.Pattern[str]


BLOCKERS = [
    PatternRule("HLS error", re.compile(r"\bERROR:\s*\[HLS", re.IGNORECASE)),
    PatternRule("failed synthesis", re.compile(r"\b(Failed to run synthesis|Pre-synthesis failed|Encountered problem during source synthesis)\b", re.IGNORECASE)),
    PatternRule("dataflow merge feedback", re.compile(r"(HLS 214-475|Merging processes|feedback on)", re.IGNORECASE)),
    PatternRule("stream read/write in same function", re.compile(r"(HLS 200-975|cannot be read and written in the same function)", re.IGNORECASE)),
    PatternRule("dataflow return-value violation", re.compile(r"(HLS 200-964|HLS 200-965|HLS 200-966|cannot have a return value|failed dataflow checking)", re.IGNORECASE)),
    PatternRule("scalar m_axi address computation", re.compile(r"(HLS 214-323|Address computation on scalar port)", re.IGNORECASE)),
    PatternRule("resource over-utilized", re.compile(r"(DRC UTLZ-1|over-utilized|requires more .* than are available)", re.IGNORECASE)),
]


HIGH_RISKS = [
    PatternRule("pipeline directive not satisfied", re.compile(r"Unable to satisfy pipeline directive", re.IGNORECASE)),
    PatternRule("II violation", re.compile(r"(HLS 200-880|II Violation|Final II = ([2-9]|[1-9][0-9]+))", re.IGNORECASE)),
    PatternRule("pipeline disabled", re.compile(r"#pragma HLS PIPELINE off|PIPELINE off", re.IGNORECASE)),
    PatternRule("dataflow process throughput warning", re.compile(r"(HLS 200-1449|HLS 200-1450)", re.IGNORECASE)),
    PatternRule("dataflow form issue", re.compile(r"(Dataflow form checks found|HLS 200-471)", re.IGNORECASE)),
    PatternRule("large compile design", re.compile(r"There were [1-9][0-9]{5,} instructions in the design", re.IGNORECASE)),
    PatternRule("ambiguous conversion/operator", re.compile(r"(ambiguous conversion|operator '==' is ambiguous)", re.IGNORECASE)),
    PatternRule("fifo-w256-d32 instance (capacity=31 < typical 32-burst)", re.compile(r"fifo_w256_d32_A", re.IGNORECASE)),
    PatternRule("stream depth=32 (at-risk for DATAFLOW burst overflow)", re.compile(r"depth\s*=\s*32", re.IGNORECASE)),
]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def existing(paths: list[Path]) -> list[Path]:
    return [p for p in paths if p.exists() and p.is_file()]


def latest_files(directory: Path, pattern: str, count: int) -> list[Path]:
    if not directory.exists():
        return []
    files = [p for p in directory.glob(pattern) if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:count]


def default_inputs(root: Path, backup_count: int) -> list[Path]:
    log_dir = root / "ESP_INT8_hls" / "ESP_INT8_hls" / "logs"
    report_dirs = [
        root / "ESP_INT8_hls" / "ESP_INT8_hls" / "hls" / "syn" / "report",
        root / "ESP_INT8_hls" / "hls_work" / "hls" / "syn" / "report",
    ]
    src_dirs = [
        root / "ESP_INT8_hls" / "src",
    ]
    verilog_dir = root / "ESP_INT8_hls" / "ESP_INT8_hls" / "hls" / "syn" / "verilog"

    files = existing(
        [
            log_dir / "hls_compile.log",
            log_dir / "ESP_INT8_hls.steps.log",
            log_dir / "hls_run_package.log",
        ]
    )
    files.extend(latest_files(log_dir, "hls_compile_*.backup.log", backup_count))

    for report_dir in report_dirs:
        files.extend(existing([report_dir / "hls_syn.rpt", report_dir / "csynth.rpt", report_dir / "csynth_design_size.rpt"]))
        files.extend(latest_files(report_dir, "*csynth*.rpt", 3))

    for sd in src_dirs:
        if sd.exists():
            files.extend(sorted([p for p in sd.glob("*.cpp") if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)[:3])

    if verilog_dir.exists():
        files.extend(sorted([p for p in verilog_dir.glob("*row_region*.v") if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)[:1])

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(path)
    return deduped


def all_inputs(root: Path) -> list[Path]:
    bases = [
        root / "ESP_INT8_hls" / "ESP_INT8_hls" / "logs",
        root / "ESP_INT8_hls" / "ESP_INT8_hls" / "hls" / "syn" / "report",
        root / "ESP_INT8_hls" / "hls_work" / "hls" / "syn" / "report",
        root / "ESP_INT8_hls" / "ESP_INT8_hls" / "hls" / "syn" / "verilog",
        root / "ESP_INT8_hls" / "src",
    ]
    files: list[Path] = []
    for base in bases:
        if base.exists():
            files.extend([p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in {".log", ".rpt", ".txt", ".v", ".cpp", ".hpp"}])
    return files


def scan_file(path: Path, rules: list[PatternRule], max_matches: int) -> list[tuple[str, int, str]]:
    matches: list[tuple[str, int, str]] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for lineno, line in enumerate(f, start=1):
                for rule in rules:
                    if rule.pattern.search(line):
                        matches.append((rule.name, lineno, line.rstrip()))
                        break
                if len(matches) >= max_matches:
                    break
    except OSError as exc:
        matches.append(("read failure", 0, str(exc)))
    return matches


def print_matches(title: str, paths: list[Path], rules: list[PatternRule], max_matches: int) -> int:
    total = 0
    print(f"\n{title}")
    for path in paths:
        matches = scan_file(path, rules, max_matches)
        if not matches:
            continue
        total += len(matches)
        print(f"- {rel(path)}")
        for name, lineno, text in matches:
            loc = f":{lineno}" if lineno else ""
            print(f"  [{name}] {rel(path)}{loc}: {text[:240]}")
    if total == 0:
        print("- none")
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root. Default: parent of tools/")
    parser.add_argument("--all", action="store_true", help="Scan all historical logs/reports, not only active/latest files.")
    parser.add_argument("--backup-count", type=int, default=0, help="Latest hls_compile_*.backup.log files to include.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero on high-risk warnings as well as blockers.")
    parser.add_argument("--max-matches", type=int, default=40, help="Maximum matches printed per file per severity.")
    args = parser.parse_args()

    root = args.root.resolve()
    paths = all_inputs(root) if args.all else default_inputs(root, args.backup_count)
    paths = [p for p in paths if p.exists() and p.is_file()]
    paths.sort(key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)

    print(f"HLS report audit root: {root}")
    print("Scanned files:")
    for path in paths:
        print(f"- {rel(path)}")

    blockers = print_matches("BLOCKERS", paths, BLOCKERS, args.max_matches)
    high_risks = print_matches("HIGH-RISK WARNINGS", paths, HIGH_RISKS, args.max_matches)

    print(f"\nSummary: blockers={blockers}, high_risks={high_risks}")
    if blockers:
        print("Result: FAIL. Do not package, export hardware, rebuild platform, or run board tests.")
        return 2
    if args.strict and high_risks:
        print("Result: FAIL under --strict. Review high-risk warnings before downstream steps.")
        return 1
    print("Result: PASS for blocker gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
