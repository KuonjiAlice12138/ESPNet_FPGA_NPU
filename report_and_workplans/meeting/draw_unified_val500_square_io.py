#!/usr/bin/env python3
"""Draw the tested mode table and square-I/O figures for VAL500."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from matplotlib.ticker import StrMethodFormatter

MEETING = Path(__file__).resolve().parent
DEFAULT_CSV = MEETING / "unified_val500_square_io.csv"
DEFAULT_AUDIT = (
    MEETING.parents[1] / "hw_artifacts" / "unified_val500_1122"
    / "results_TEST0000" / "audit.json"
)
DEFAULT_OUTPUT = MEETING / "unified_val500_figures"
CASE_ORDER = ("G240B2", "G512B2", "G2S3M0", "G2S3M1")
PL_CYCLES_PER_MS = 100_000

PATH_FULL_CONV = "全卷积路径"
PATH_CROP_DS = "剪裁下采样"

# Refined academic color palette
INK = "#162433"
TEXT_SUB = "#4C5E6E"
MUTED = "#6D7E8D"
SPINE = "#8EA0AE"
GRID = "#E6ECF1"
RULE = "#DCE4EC"
CARD_BG = "#F7F9FB"
CARD_EDGE = "#D5DFE8"

NATIVE = "#1D5B8C"
NATIVE_SOFT = "#E9F2FA"
NATIVE_BORDER = "#C3DAEE"

LITE = "#C05636"
LITE_SOFT = "#FCEEE8"
LITE_BORDER = "#EFC9B9"

STAGES = (
    ("卷积计算", (5,), "#2B628A"),
    ("行级后处理与输出", (6,), "#CF6A44"),
    ("块级后处理", (7,), "#5D8B7A"),
    ("向量与池化", (8, 9), "#8BA3B7"),
    ("加载与控制", (0, 1, 2, 3, 4, 10, 11), "#D7E0E7"),
)


@dataclass(frozen=True)
class Case:
    key: str
    path: str
    input_side: int
    encoder_side: int
    mask_side: int
    pa: float
    miou: float
    latency_ms: float
    vehicle_iou: float
    macs: int
    stages_ms: tuple[float, ...]

    @property
    def is_native(self) -> bool:
        return self.path == "native"

    @property
    def series(self) -> str:
        return PATH_FULL_CONV if self.is_native else PATH_CROP_DS

    @property
    def color(self) -> str:
        return NATIVE if self.is_native else LITE

    @property
    def soft_color(self) -> str:
        return NATIVE_SOFT if self.is_native else LITE_SOFT

    @property
    def marker(self) -> str:
        return "o" if self.is_native else "D"

    @property
    def route(self) -> str:
        return f"{self.input_side} → {self.encoder_side} → {self.mask_side}"

    @property
    def gops(self) -> float:
        return 2 * self.macs / (self.latency_ms * 1e6)


def read_cases(csv_path: Path, audit_path: Path) -> list[Case]:
    with csv_path.open(encoding="utf-8-sig", newline="") as fp:
        rows = {r["case_id"]: r for r in csv.DictReader(fp)}
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audited = {c["key"]: c for c in audit["cases"]}
    if set(rows) != set(CASE_ORDER) or audit["errors"]:
        raise ValueError("Expected the four audited square-I/O cases.")

    cases = []
    for key in CASE_ORDER:
        row, checked = rows[key], audited[key]
        h_in, h_enc, h_mask = (int(row[f]) for f in (
            "input_side_px", "encoder_side_px", "mask_side_px"
        ))
        stage = tuple(checked["stage_cycles"][str(i)]["cycles"] for i in range(13))
        if (
            int(row["val_images"]) != checked["samples"] or checked["samples"] != 500
            or checked["mask_shape"] != [h_mask, h_mask] or checked["classes"] != 2
            or sum(stage) != checked["stage_total"] or stage[12] != 0
        ):
            raise ValueError(f"{key}: inconsistent sample, output or stage totals.")
        pa, miou, latency = (float(row[f]) for f in (
            "pa_pct", "miou_pct", "mean_mode_run_rtl_ms"
        ))
        if not all(math.isclose(a, b, rel_tol=0, abs_tol=5.1e-7) for a, b in (
            (pa, 100 * checked["PA"]), (miou, 100 * checked["mIoU"]),
            (latency, checked["mean_rtl_ms"]),
        )):
            raise ValueError(f"{key}: CSV metrics disagree with the board audit.")
        case = Case(
            key, row["path"], h_in, h_enc, h_mask, pa, miou, latency,
            100 * checked["per_class_IoU"][0], int(row["conv_macs_per_image"]),
            tuple(c / PL_CYCLES_PER_MS for c in stage),
        )
        if not math.isclose(
            case.gops / 1000, float(row["e2e_effective_conv_tops"]),
            rel_tol=0, abs_tol=5.1e-10,
        ):
            raise ValueError(f"{key}: effective throughput disagrees with MACs and time.")
        cases.append(case)
    return cases


@dataclass(frozen=True)
class TestedMode:
    key: str
    source_shape: tuple[int, int, int]
    encoder_shape: tuple[int, int]
    mask_shape: tuple[int, int]
    classes: int
    is_crop_ds: bool


def read_tested_modes(audit_path: Path) -> list[TestedMode]:
    tools_path = str(MEETING.parents[1] / "tools")
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    from geometry_contract import runtime_selection

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    results = {r["key"]: r for r in audit["cases"]}
    catalog = audit_path.parent.parent / "board" / "VALRUNS.CSV"
    with catalog.open(encoding="utf-8-sig", newline="") as fp:
        records = list(csv.DictReader(fp))
    if (audit["errors"] or len(records) != len(results)
            or {r["key"] for r in records} != results.keys()
            or not set(CASE_ORDER).issubset(results)):
        raise ValueError("Board validation catalog and audited cases disagree.")

    modes = {}
    for row in records:
        key = row["key"]
        checked = results[key]
        selection = runtime_selection(
            int(row["profile"]), int(row["classes"]),
            src_mode=int(row["source"]), mask_mode=int(row["mask"]),
        )
        if (selection.classes != checked["classes"] or checked["samples"] != 500
                or list(selection.mask_shape) != checked["mask_shape"]):
            raise ValueError(f"{key}: invalid audited mode or sample count.")
        encoder = selection.profile.geometry
        modes[key] = TestedMode(
            key, selection.source_shape, (encoder.input_height, encoder.input_width),
            selection.mask_shape, selection.classes, selection.src_mode != 0,
        )
    return [modes[k] for k in CASE_ORDER] + [m for k, m in modes.items() if k not in CASE_ORDER]


def configure_style() -> None:
    system_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    for name in ("msyh.ttc", "msyhbd.ttc"):
        path = system_fonts / name
        if path.is_file():
            font_manager.fontManager.addfont(path)
    families = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in ("Microsoft YaHei", "Noto Sans CJK SC", "Noto Sans SC")
                   if f in families), None)
    if not family:
        raise RuntimeError("Install Microsoft YaHei or Noto Sans SC to render the labels.")
    mpl.rcParams.update({
        "font.family": [family, "DejaVu Sans"],
        "font.size": 12.5,
        "axes.unicode_minus": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": TEXT_SUB,
        "ytick.color": TEXT_SUB,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "path",
        "svg.hashsalt": "unified-val500-square-io",
    })


def hrule(fig, x0: float, x1: float, y: float, *, color: str = RULE, lw: float = 0.9) -> None:
    fig.add_artist(Line2D([x0, x1], [y, y], transform=fig.transFigure, color=color, lw=lw))


def canvas(title: str, subtitle: str, footnote: str):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=135)
    fig.text(.060, .930, title, fontsize=23, weight="bold", color=INK, va="center")
    fig.text(.060, .870, subtitle, fontsize=12.5, color=TEXT_SUB, va="center")
    hrule(fig, .060, .945, .826, color=RULE, lw=1.0)
    hrule(fig, .060, .122, .826, color=NATIVE, lw=2.2)
    hrule(fig, .060, .945, .102, color=RULE, lw=0.9)
    fig.text(.060, .060, footnote, fontsize=10.5, color=MUTED, va="center")
    return fig


def axes_style(ax, *, grid_y: bool = False, left: bool = False) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_visible(left)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(axis="x", length=3.5, width=0.9, color=SPINE, pad=7, labelsize=11.5)
    ax.tick_params(axis="y", length=3.5 if left else 0, width=0.9, color=SPINE, pad=7, labelsize=11.5)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, lw=0.85, linestyle="-")
    if grid_y:
        ax.yaxis.grid(True, color=GRID, lw=0.85, linestyle="-")


def add_path_legend(fig, cases: list[Case], *, y: float = .870) -> None:
    fig.legend(
        handles=[
            Line2D([], [], linestyle="none", marker=c.marker, color=c.color,
                   markeredgecolor="white", markeredgewidth=0.9,
                   markersize=8.8, label=c.series)
            for c in (cases[0], cases[2])
        ],
        loc="center right", bbox_to_anchor=(.948, y), ncols=2,
        frameon=False, handletextpad=.4, columnspacing=1.35, fontsize=12.2,
    )


def case_mark(ax, x: float, y: float, case: Case, *, size: float = 125, halo: bool = False) -> None:
    if halo:
        ax.scatter(x, y, s=size * 2.45, marker=case.marker, c=case.color,
                   alpha=.16, edgecolors="none", zorder=4)
    ax.scatter(x, y, s=size, marker=case.marker, c=case.color,
               edgecolors="white", linewidths=1.5, zorder=6)


def draw_case_rows(ax, cases: list[Case], *, labels: bool = True,
                   label_x: float = -0.025, bottom_pad: float = .46) -> None:
    ax.set_ylim(len(cases) - 1.0 + bottom_pad, -.54)
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels(["" for _ in cases])
    ax.axhline(1.5, color="#CCD6DF", linewidth=0.95, linestyle=(0, (4, 3)), zorder=1)
    if not labels:
        return
    trans = ax.get_yaxis_transform()
    for i, c in enumerate(cases):
        ax.text(label_x, i - 0.12, c.route, transform=trans,
                ha="right", va="center", fontsize=12.8, weight="bold", color=INK)
        ax.text(label_x, i + 0.16, c.series, transform=trans,
                ha="right", va="center", fontsize=10.5, weight="medium", color=c.color)


def axes_aspect(fig, ax) -> float:
    bbox = ax.get_position()
    fw, fh = fig.get_size_inches()
    return (bbox.width * fw) / (bbox.height * fh)


def draw_pill(fig, ax, x: float, y: float, width: float, height: float, text: str,
              *, bg: str, edge: str, fg: str, fontsize: float = 11.2) -> None:
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.0,rounding_size=0.0055",
        mutation_aspect=axes_aspect(fig, ax),
        facecolor=bg, edgecolor=edge, linewidth=0.85, zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, fontsize=fontsize, weight="bold", color=fg,
            ha="center", va="center", zorder=4)


def coverage_table(modes: list[TestedMode]):
    fig = canvas(
        "统一 INT8 NPU 板测模式总览",
        f"同一套硬件运行时切换{PATH_FULL_CONV}与{PATH_CROP_DS}  ·  共 {len(modes)} 组配置，每组 500 张验证集实测",
        f"注：表内为本轮已上板验证的 {len(modes)} 组配置；对宽图采用{PATH_CROP_DS}时，先截取中心正方形，输出掩码仅对应中心区域。",
    )
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()

    left, right = .060, .945
    top = .790
    header_h = .056
    group_h = .038
    row_h = .0525
    col_x = (.148, .288, .445, .608, .765, .892)
    headers = ("输入分辨率 (H×W)", "输入类型", "处理路径", "编码器分辨率", "输出掩码分辨率", "类别数")

    # Booktabs header band & top/mid rules
    ax.add_patch(Rectangle((left, top - header_h), right - left, header_h,
                           facecolor="#F1F5F8", edgecolor="none", zorder=1))
    ax.hlines(top, left, right, color=INK, lw=1.8, zorder=4)
    ax.hlines(top - header_h, left, right, color=SPINE, lw=1.0, zorder=4)
    for x, label in zip(col_x, headers):
        ax.text(x, top - header_h / 2, label, fontsize=12.5, weight="bold",
                color=INK, ha="center", va="center", zorder=5)

    def shape_label(shape: tuple[int, int]) -> str:
        return f"{shape[0]}×{shape[1]}"

    groups = (
        ("主对照组：正方形输入（用于后续精度与时延对比）", modes[:len(CASE_ORDER)], True),
        ("其余已测配置", modes[len(CASE_ORDER):], False),
    )

    cursor_y = top - header_h
    for g_idx, (group_title, items, is_focus) in enumerate(groups):
        if g_idx > 0:
            ax.hlines(cursor_y, left, right, color="#B8C5D0", lw=0.95, zorder=4)
        cursor_y -= group_h
        ax.add_patch(Rectangle((left, cursor_y), right - left, group_h,
                               facecolor="#F8FAFC" if is_focus else "#F9FAFB",
                               edgecolor="none", zorder=1))
        ax.text(left + .014, cursor_y + group_h / 2, group_title,
                fontsize=11.0, weight="bold",
                color="#264863" if is_focus else TEXT_SUB,
                ha="left", va="center", zorder=4)
        ax.hlines(cursor_y, left, right, color=GRID, lw=0.75, zorder=3)

        for r_idx, mode in enumerate(items):
            cursor_y -= row_h
            bg = "#FCFDFE" if (r_idx % 2 == 0) else "#F7F9FB"
            if is_focus:
                bg = "#F5FAFD" if not mode.is_crop_ds else "#FEF8F5"
            ax.add_patch(Rectangle((left, cursor_y), right - left, row_h,
                                   facecolor=bg, edgecolor="none", zorder=1))
            if is_focus:
                ax.add_patch(Rectangle((left, cursor_y + .005), .0042, row_h - .010,
                                       facecolor=LITE if mode.is_crop_ds else NATIVE,
                                       edgecolor="none", zorder=3))
            if r_idx + 1 < len(items):
                ax.hlines(cursor_y, left, right, color=GRID, lw=0.65, zorder=2)

            yc = cursor_y + row_h / 2
            values = (
                shape_label(mode.source_shape[:2]),
                "灰度" if mode.source_shape[2] == 1 else "RGB",
                PATH_CROP_DS if mode.is_crop_ds else PATH_FULL_CONV,
                shape_label(mode.encoder_shape),
                shape_label(mode.mask_shape),
                f"{mode.classes} 类",
            )
            for j, (x, val) in enumerate(zip(col_x, values)):
                if j == 2:
                    draw_pill(
                        fig, ax, x, yc, width=.116, height=.033, text=val,
                        bg=LITE_SOFT if mode.is_crop_ds else NATIVE_SOFT,
                        edge=LITE_BORDER if mode.is_crop_ds else NATIVE_BORDER,
                        fg=LITE if mode.is_crop_ds else NATIVE,
                        fontsize=11.0,
                    )
                else:
                    is_multiclass = (j == 5 and mode.classes > 2)
                    ax.text(
                        x, yc, val, fontsize=12.8, ha="center", va="center",
                        color=INK if (is_focus or is_multiclass) else "#2B3C4C",
                        weight="bold" if is_multiclass else "normal",
                        zorder=4,
                    )

    ax.hlines(cursor_y, left, right, color=INK, lw=1.8, zorder=4)
    return fig


def accuracy_latency(cases: list[Case]):
    fig = canvas(
        "分割精度与推理延时的权衡",
        "Cityscapes 500 张验证集  ·  灰度二分类 INT8  ·  标注：输入 → 编码器 → 输出（方形边长）",
        "注：延时为板端单图推理均值（不含 SD 卡读写）；两组 512×512 配置使用相同输入与真值标注，精度按各自输出网格评估。",
    )
    add_path_legend(fig, cases)

    ax = fig.add_axes((.095, .210, .850, .572))
    axes_style(ax, grid_y=True, left=True)
    ax.set(xlim=(-5, 255), ylim=(76.6, 90.2),
           xticks=[0, 50, 100, 150, 200, 250],
           yticks=[78, 80, 82, 84, 86, 88, 90])
    ax.set_xlabel("平均推理延时 (ms / 图)", fontsize=12.5, labelpad=9)
    ax.set_ylabel("mIoU (%)", fontsize=12.5, labelpad=8)

    native, lite = cases[1], cases[3]
    guide = "#5D7282"

    # L-shaped trade-off comparison guide between G512B2 and G2S3M1
    ax.annotate(
        "", (lite.latency_ms, native.miou), (native.latency_ms, native.miou),
        arrowprops={"arrowstyle": "-|>", "color": guide, "lw": 1.35,
                    "linestyle": (0, (4, 3)), "shrinkA": 10, "shrinkB": 5,
                    "mutation_scale": 12},
        zorder=3,
    )
    ax.scatter(lite.latency_ms, native.miou, s=26, color="white",
               edgecolors=guide, linewidths=1.35, zorder=4)
    ax.annotate(
        "", (lite.latency_ms, lite.miou), (lite.latency_ms, native.miou),
        arrowprops={"arrowstyle": "-|>", "color": guide, "lw": 1.35,
                    "linestyle": (0, (4, 3)), "shrinkA": 5, "shrinkB": 11,
                    "mutation_scale": 12},
        zorder=3,
    )

    speedup = native.latency_ms / lite.latency_ms
    lat_drop_pct = (1.0 - lite.latency_ms / native.latency_ms) * 100.0
    miou_drop = native.miou - lite.miou
    mid_x = (native.latency_ms + lite.latency_ms) / 2
    ax.text(mid_x, native.miou + 1.48, "512×512 同分辨率对比",
            ha="center", va="center", fontsize=13.5, weight="bold", color=INK)
    ax.text(mid_x, native.miou + 0.62,
            f"推理加速 {speedup:.2f}×（延时降低 {lat_drop_pct:.1f}%）",
            ha="center", va="center", fontsize=12.2, color=TEXT_SUB)
    ax.text(lite.latency_ms + 4.8, (native.miou + lite.miou) / 2 + 0.15,
            f"mIoU 降低 {miou_drop:.2f} 个百分点",
            ha="left", va="center", fontsize=12.0, weight="medium", color=TEXT_SUB,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                  "edgecolor": "none", "alpha": 0.85},
            zorder=4)

    # Data points and clean non-overlapping callouts
    label_specs = {
        "G240B2": (10.0, 82.65, "left", "arc3,rad=0.14"),
        "G512B2": (225.41, 84.78, "center", "arc3,rad=0.0"),
        "G2S3M0": (7.0, 77.55, "left", "arc3,rad=-0.12"),
        "G2S3M1": (84.0, 78.95, "left", "arc3,rad=0.0"),
    }
    for c in cases:
        matched = c in (native, lite)
        case_mark(ax, c.latency_ms, c.miou, c,
                  size=145 if matched else 95, halo=matched)
        lx, ly, align, rad = label_specs[c.key]
        ax.annotate(
            f"{c.route}\n{c.latency_ms:.2f} ms · {c.miou:.2f}%",
            (c.latency_ms, c.miou), xytext=(lx, ly), ha=align, va="center",
            fontsize=11.4, weight="bold" if matched else "normal",
            color=c.color, linespacing=1.32,
            arrowprops={"arrowstyle": "-", "color": c.color, "lw": 0.95,
                        "alpha": 0.72, "shrinkA": 4, "shrinkB": 8,
                        "connectionstyle": rad},
            zorder=5,
        )

    # Inset comparison card in axes-fraction coordinates with geometrically circular corners
    cx0, cy0, cw, ch = 0.542, 0.052, 0.436, 0.375
    card = FancyBboxPatch(
        (cx0, cy0), cw, ch,
        boxstyle="round,pad=0.0,rounding_size=0.009",
        mutation_aspect=axes_aspect(fig, ax),
        transform=ax.transAxes, facecolor=CARD_BG,
        edgecolor=CARD_EDGE, linewidth=1.0, zorder=4,
    )
    ax.add_patch(card)

    def ctext(rx: float, ry: float, text: str, **kwargs) -> None:
        ax.text(cx0 + rx * cw, cy0 + ry * ch, text,
                transform=ax.transAxes, zorder=5, **kwargs)

    ctext(0.05, 0.86, "512×512 细项对比", fontsize=12.0, weight="bold", color=INK,
          ha="left", va="center")
    ctext(0.95, 0.86, "前景车辆降幅更明显", fontsize=10.5, color=LITE, weight="medium",
          ha="right", va="center")

    cols = (
        (0.05, "指标", "left", MUTED),
        (0.41, PATH_FULL_CONV, "center", NATIVE),
        (0.65, PATH_CROP_DS, "center", LITE),
        (0.95, "差值", "right", MUTED),
    )
    for rx, label, align, color in cols:
        ctext(rx, 0.65, label, fontsize=10.3, weight="bold", color=color,
              ha=align, va="center")

    ax.add_artist(Line2D(
        [cx0 + 0.05 * cw, cx0 + 0.95 * cw],
        [cy0 + 0.56 * ch, cy0 + 0.56 * ch],
        transform=ax.transAxes, color="#CBD6E0", lw=0.9, zorder=5,
    ))

    rows_data = (
        (0.41, "整体 PA", f"{native.pa:.2f}%", f"{lite.pa:.2f}%",
         f"{lite.pa - native.pa:+.2f}", False),
        (0.18, "车辆 IoU", f"{native.vehicle_iou:.2f}%", f"{lite.vehicle_iou:.2f}%",
         f"{lite.vehicle_iou - native.vehicle_iou:+.2f}", True),
    )
    for ry, metric, v_nat, v_lit, v_diff, emphasize in rows_data:
        ctext(0.05, ry, metric, fontsize=11.4, weight="bold" if emphasize else "normal",
              color=INK, ha="left", va="center")
        ctext(0.41, ry, v_nat, fontsize=11.4, color=INK, ha="center", va="center")
        ctext(0.65, ry, v_lit, fontsize=11.4, color=INK, ha="center", va="center")
        ctext(0.95, ry, f"{v_diff} pp", fontsize=11.4,
              weight="bold" if emphasize else "normal",
              color=LITE if emphasize else TEXT_SUB, ha="right", va="center")

    return fig


def latency_breakdown(cases: list[Case]):
    fig = canvas(
        "推理时延阶段拆解",
        "100 MHz 硬件阶段计数器实测  ·  标注：输入 → 编码器 → 输出（方形边长）",
        "注：取自各组首图 100 MHz RTL 阶段计数；行级后处理含双线性上采样与掩码写回。",
    )
    fig.legend(
        handles=[Patch(facecolor=color, edgecolor="none", label=label)
                 for label, _, color in STAGES],
        loc="center left", bbox_to_anchor=(.060, .774), frameon=False, ncols=5,
        fontsize=11.8, handlelength=1.25, handletextpad=.45, columnspacing=1.6,
        borderaxespad=0,
    )

    ax = fig.add_axes((.215, .210, .730, .498))
    axes_style(ax)
    draw_case_rows(ax, cases, labels=True, label_x=-0.022)
    ax.set(xlim=(0, 255), xticks=[0, 50, 100, 150, 200, 250])
    ax.set_xlabel("单张图片推理延时 (ms)", fontsize=12.5, labelpad=9)

    for idx, c in enumerate(cases):
        start = 0.0
        total = sum(c.stages_ms)
        for s_idx, (_, ids, color) in enumerate(STAGES):
            duration = sum(c.stages_ms[i] for i in ids)
            ax.barh(idx, duration, left=start, height=.46, color=color,
                    edgecolor="white", linewidth=0.75, zorder=3)
            # Label wide segments directly without clutter
            if duration >= 14.5:
                text_color = "#1A2E40" if s_idx >= 3 else "white"
                if duration >= 100.0:
                    label = f"{duration:.1f} ms ({duration / total * 100:.1f}%)"
                else:
                    label = f"{duration:.1f}"
                ax.text(start + duration / 2, idx, label,
                        ha="center", va="center", fontsize=10.3, weight="bold",
                        color=text_color, zorder=4)
            start += duration
        ax.text(start + 3.5, idx, f"{start:.2f} ms", va="center", ha="left",
                fontsize=12.2, weight="bold", color=INK)

    # Anchored comparison bracket & callout card linking G2S3M0 (row 2) and G2S3M1 (row 3)
    row0, row1 = cases[2].stages_ms[6], cases[3].stages_ms[6]
    conv_ms = cases[2].stages_ms[5]
    increase = sum(cases[3].stages_ms) - sum(cases[2].stages_ms)

    bx = 102.0
    card_left_data = 113.0
    ax.plot([82.0, bx, bx, 96.5], [2.0, 2.0, 3.0, 3.0],
            color="#C9876E", lw=1.2, zorder=4)
    ax.annotate(
        "", (card_left_data - 1.0, 2.50), (bx, 2.50),
        arrowprops={"arrowstyle": "-|>", "color": "#C9876E", "lw": 1.25,
                    "shrinkA": 0, "shrinkB": 2, "mutation_scale": 11},
        zorder=4,
    )

    # Callout card in axes-fraction coordinates with circular corners and ample right margin
    cx0, cy0, cw, ch = 0.445, 0.085, 0.535, 0.345
    card = FancyBboxPatch(
        (cx0, cy0), cw, ch,
        boxstyle="round,pad=0.0,rounding_size=0.010",
        mutation_aspect=axes_aspect(fig, ax),
        transform=ax.transAxes, facecolor="#FDF7F4",
        edgecolor="#EAC9BA", linewidth=1.0, zorder=4,
    )
    ax.add_patch(card)
    ax.text(cx0 + 0.045 * cw, cy0 + 0.73 * ch,
            f"输出升至 512×512：延时增加 +{increase:.2f} ms",
            transform=ax.transAxes, fontsize=12.6, weight="bold",
            color=LITE, va="center", zorder=5)
    ax.text(cx0 + 0.045 * cw, cy0 + 0.44 * ch,
            f"编码器卷积耗时不变（同为 {conv_ms:.2f} ms）",
            transform=ax.transAxes, fontsize=11.0, color=INK, va="center", zorder=5)
    ax.text(cx0 + 0.045 * cw, cy0 + 0.19 * ch,
            f"增量全部来自行级后处理与输出（{row0:.2f} → {row1:.2f} ms）",
            transform=ax.transAxes, fontsize=11.0, color=TEXT_SUB, va="center", zorder=5)

    return fig


def macs_throughput(cases: list[Case]):
    fig = canvas(
        "卷积计算量与端到端有效吞吐",
        "同一 INT8 NPU（100 MHz）实测  ·  标注：输入 → 编码器 → 输出（方形边长）",
        "注：端到端有效吞吐 = 单图卷积 MAC × 2 ÷ 平均推理耗时；非卷积阶段（后处理、池化与输出等）仅计入总耗时，不计入算量。",
    )
    add_path_legend(fig, cases)

    fig.text(.215, .765, "(a) 单图卷积计算量", fontsize=14.0, weight="bold", color=INK)
    fig.text(.618, .765, "(b) 端到端有效吞吐", fontsize=14.0, weight="bold", color=INK)

    a = fig.add_axes((.215, .210, .335, .505))
    b = fig.add_axes((.618, .210, .327, .505))
    for ax in (a, b):
        axes_style(ax)
        draw_case_rows(ax, cases, labels=(ax is a), label_x=-0.045, bottom_pad=.65)

    a.set(xlim=(0, .92), xticks=[0, .2, .4, .6, .8])
    a.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    a.set_xlabel("GMAC / 图", fontsize=12.5, labelpad=9)

    b.set(xlim=(0, 7.0), xticks=[0, 1, 2, 3, 4, 5, 6])
    b.set_xlabel("GOPS", fontsize=12.5, labelpad=9)

    base_macs = cases[0].macs
    for i, c in enumerate(cases):
        gmacs = c.macs / 1e9
        a.barh(i, gmacs, color=c.color, height=.42, alpha=.92,
               edgecolor="none", zorder=3)
        if i == 1:
            ratio = c.macs / base_macs
            a.text(gmacs + .024, i, f"{gmacs:.3f}", va="center", ha="left",
                   fontsize=12.5, weight="bold", color=c.color)
            a.text(gmacs + .152, i, f"({ratio:.2f}×)", va="center", ha="left",
                   fontsize=10.8, color=MUTED)
        else:
            a.text(gmacs + .024, i, f"{gmacs:.3f}", va="center", ha="left",
                   fontsize=12.5, weight="bold", color=c.color)

        b.hlines(i, 0, c.gops, color=c.color, alpha=.30, lw=3.2,
                 capstyle="round", zorder=3)
        case_mark(b, c.gops, i, c, size=135, halo=(i in (1, 3)))
        b.text(c.gops + .22, i, f"{c.gops:.2f}", va="center", ha="left",
               fontsize=12.5, weight="bold", color=c.color)

    # Clean, unclipped throughput-drop indicator on Panel B between G2S3M0 (row 2) and G2S3M1 (row 3)
    ref_gops = cases[2].gops
    drop_gops = cases[3].gops
    drop_pct = (1.0 - drop_gops / ref_gops) * 100.0
    guide_y = 3.38
    b.plot([ref_gops, ref_gops], [2.16, guide_y], color=LITE, lw=1.05,
           linestyle=(0, (3, 2.5)), alpha=.75, zorder=2)
    b.plot([drop_gops, drop_gops], [3.16, guide_y], color=LITE, lw=1.05,
           linestyle=(0, (3, 2.5)), alpha=.75, zorder=2)
    b.annotate(
        "", (drop_gops, guide_y), (ref_gops, guide_y),
        arrowprops={"arrowstyle": "-|>", "color": LITE, "lw": 1.25,
                    "shrinkA": 0, "shrinkB": 1, "mutation_scale": 10},
        zorder=4,
    )
    b.text(drop_gops - 0.16, guide_y, f"行级后处理成为瓶颈 -{drop_pct:.1f}%",
           ha="right", va="center", fontsize=10.6, weight="medium", color=LITE,
           bbox={"boxstyle": "round,pad=0.18", "facecolor": "white",
                 "edgecolor": "none", "alpha": 0.85},
           zorder=5)

    return fig


def save_figure(fig, output: Path, title: str) -> None:
    fig.savefig(output / f"{title}.png", dpi=240)
    fig.savefig(output / f"{title}.svg", metadata={"Date": None})
    plt.close(fig)


def render_all(cases: list[Case], output: Path, *, audit_path: Path = DEFAULT_AUDIT) -> None:
    configure_style()
    output.mkdir(parents=True, exist_ok=True)
    save_figure(coverage_table(read_tested_modes(audit_path)), output, "00_npu_mode_matrix")
    for title, draw in (
        ("01_accuracy_latency", accuracy_latency),
        ("02_latency_breakdown", latency_breakdown),
        ("03_macs_throughput", macs_throughput),
    ):
        save_figure(draw(cases), output, title)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render_all(read_cases(args.csv, args.audit), args.output_dir, audit_path=args.audit)
    print(f"Wrote four PNG/SVG pairs to {args.output_dir}")


if __name__ == "__main__":
    main()
