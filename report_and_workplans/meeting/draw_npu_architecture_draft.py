"""Draw draft architecture diagrams for the ESP INT8 FPGA NPU.

Outputs:
  - npu_system_architecture_draft.png
  - conv_engine_pipeline_draft.png

The diagrams intentionally use a monochrome palette so they can be redrawn in
PowerPoint without depending on the exact colors used by this script.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.font_manager import FontProperties


OUT_DIR = Path(__file__).resolve().parent
FONT_PATH = Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf")


def font(size: float, bold: bool = False) -> FontProperties:
    if FONT_PATH.exists():
        return FontProperties(fname=str(FONT_PATH), size=size, weight="bold" if bold else "normal")
    return FontProperties(family="Microsoft YaHei", size=size, weight="bold" if bold else "normal")


def setup_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    *,
    size: float = 12,
    bold: bool = False,
    fill: str = "white",
    radius: float = 0.02,
    lw: float = 1.5,
    z: int = 3,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=fill,
        edgecolor="black",
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        multialignment="center",
        fontproperties=font(size, bold),
        color="black",
        zorder=z + 1,
    )


def section_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    *,
    fill: str = "#f2f2f2",
    size: float = 13,
) -> None:
    patch = Rectangle(
        (x, y),
        w,
        h,
        facecolor=fill,
        edgecolor="black",
        linewidth=2.0,
        zorder=0,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.015,
        y + h - 0.025,
        label,
        ha="left",
        va="top",
        fontproperties=font(size, True),
        color="black",
        zorder=1,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    label: str | None = None,
    label_offset: tuple[float, float] = (0.0, 0.0),
    lw: float = 1.4,
    dashed: bool = False,
    z: int = 1,
) -> None:
    style = "--" if dashed else "-"
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=lw,
            linestyle=style,
            color="black",
            shrinkA=5,
            shrinkB=5,
            zorder=z,
        )
    )
    if label:
        mx = (start[0] + end[0]) / 2 + label_offset[0]
        my = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontproperties=font(9), zorder=z + 1)


def draw_system_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 9), dpi=180)
    setup_axis(ax)
    ax.text(0.5, 0.965, "ESP INT8 FPGA NPU 推理系统架构（草稿）", ha="center", va="top", fontproperties=font(22, True))
    ax.text(0.5, 0.925, "PS 负责控制与外部数据访问，PL 负责片上数据流与计算", ha="center", va="top", fontproperties=font(11))

    section_box(ax, 0.035, 0.11, 0.225, 0.76, "PS Side\nZynq UltraScale+ MPSoC", fill="#eeeeee", size=14)
    section_box(ax, 0.285, 0.11, 0.68, 0.76, "PL Side\nNPU datapath", fill="#f7f7f7", size=14)

    # Draw connectors before entity boxes so they stay behind labels and nodes.
    arrow(ax, (0.16, 0.62), (0.36, 0.75), label="控制", label_offset=(0.0, 0.025))
    arrow(ax, (0.16, 0.43), (0.36, 0.68), label="输入/参数", label_offset=(-0.005, 0.02))
    arrow(ax, (0.16, 0.25), (0.31, 0.25), label="AXI", label_offset=(0.0, 0.028))
    arrow(ax, (0.49, 0.75), (0.77, 0.75), label="执行计划", label_offset=(0.0, 0.025))
    arrow(ax, (0.50, 0.68), (0.80, 0.49), label="片上读写", label_offset=(0.0, 0.02))
    arrow(ax, (0.68, 0.54), (0.68, 0.42), label="卷积行", label_offset=(0.035, 0.0))
    arrow(ax, (0.68, 0.31), (0.68, 0.22), label="后处理结果", label_offset=(0.05, 0.0))
    arrow(ax, (0.79, 0.18), (0.90, 0.18), label="输出", label_offset=(0.0, 0.03))
    arrow(ax, (0.90, 0.32), (0.90, 0.25), label="中间张量", label_offset=(0.06, 0.0), dashed=True)

    # PS blocks.
    box(ax, 0.075, 0.57, 0.14, 0.10, "ARM Core\n应用与驱动", size=11, fill="white")
    box(ax, 0.075, 0.38, 0.14, 0.10, "DDR4\n输入 / 参数", size=11, fill="white")
    box(ax, 0.075, 0.20, 0.14, 0.09, "AXI\n控制接口", size=11, fill="white")

    # PL control and data movement.
    box(ax, 0.33, 0.70, 0.17, 0.10, "Frame DMA\n输入搬运", size=11, fill="white")
    box(ax, 0.54, 0.70, 0.17, 0.10, "Main Ctrl\n执行计划", size=11, fill="white")
    box(ax, 0.77, 0.38, 0.15, 0.22, "On-chip Memory\n\n特征图缓存\n权重缓存\nRow buffer", size=11, fill="white")

    # Conv Engine with hardware-level sub-blocks.  Keep the frame title out of
    # the inner pipeline so it remains readable when the figure is resized.
    conv_frame = FancyBboxPatch(
        (0.33, 0.43),
        0.38,
        0.22,
        boxstyle="round,pad=0.006,rounding_size=0.015",
        facecolor="#e6e6e6",
        edgecolor="black",
        linewidth=1.8,
        zorder=2,
    )
    ax.add_patch(conv_frame)
    ax.text(0.35, 0.632, "Conv Engine", ha="left", va="top", fontproperties=font(14, True), zorder=4)
    box(ax, 0.35, 0.485, 0.095, 0.06, "Window\nGenerator", size=9.2, fill="white", radius=0.01)
    box(ax, 0.46, 0.485, 0.115, 0.06, "Input FIFO\n输入流", size=9.2, fill="white", radius=0.01)
    box(ax, 0.59, 0.485, 0.095, 0.06, "32×32\nMAC Array", size=9.2, fill="white", radius=0.01)
    arrow(ax, (0.445, 0.515), (0.46, 0.515), lw=1.0, z=3)
    arrow(ax, (0.575, 0.515), (0.59, 0.515), lw=1.0, z=3)
    ax.text(0.64, 0.475, "psum → 量化 / 激活", ha="center", va="center", fontproperties=font(9.2), zorder=4)

    box(ax, 0.33, 0.30, 0.38, 0.085, "Row PPU\nAdd · Affine · Concat · Block5 · 写回", size=11, bold=True, fill="white")
    box(ax, 0.33, 0.16, 0.14, 0.085, "Vec / Fixed\n固定向量算子", size=10.5, fill="white")
    box(ax, 0.49, 0.16, 0.12, 0.085, "AvgPool\n平均池化", size=10.5, fill="white")
    box(ax, 0.63, 0.16, 0.16, 0.085, "Upsample / Mask\n上采样与类别选择", size=10.5, fill="white")
    box(ax, 0.81, 0.16, 0.11, 0.085, "Output\n结果", size=10.5, fill="white")

    # Short labels clarify that the memory block is shared rather than a bus.
    ax.text(0.845, 0.625, "共享片上存储", ha="center", va="bottom", fontproperties=font(9.5, True))
    ax.text(0.50, 0.405, "行级数据流边界", ha="center", va="center", fontproperties=font(9), style="italic")

    fig.savefig(path, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def draw_conv_pipeline(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 9), dpi=180)
    setup_axis(ax)
    ax.text(0.5, 0.965, "Conv Engine 内部流水线（草稿）", ha="center", va="top", fontproperties=font(22, True))
    ax.text(0.5, 0.925, "一个输出位置：K tile 顺序累加；不同位置：各流水级可以重叠", ha="center", va="top", fontproperties=font(11))

    # Main pipeline row.
    stages = [
        (0.035, 0.62, 0.115, 0.14, "片上特征图\n缓存"),
        (0.175, 0.62, 0.13, 0.14, "Window\nGenerator"),
        (0.33, 0.62, 0.10, 0.14, "Input\nFIFO"),
        (0.455, 0.62, 0.145, 0.14, "32×32 MAC Array\n32 K lane × 32 输出 lane"),
        (0.63, 0.62, 0.10, 0.14, "psum\nFIFO"),
        (0.755, 0.62, 0.115, 0.14, "量化 /\n激活"),
        (0.895, 0.62, 0.09, 0.14, "Row PPU\n/ 写回"),
    ]
    for x, y, w, h, label in stages:
        arrow(ax, (x - 0.012, y + h / 2), (x, y + h / 2), lw=1.4, z=1)
    for i in range(len(stages) - 1):
        x, y, w, h, _ = stages[i]
        nx, ny, nw, nh, _ = stages[i + 1]
        arrow(ax, (x + w, y + h / 2), (nx, ny + nh / 2), lw=1.4, z=1)
    for x, y, w, h, label in stages:
        box(ax, x, y, w, h, label, size=10.5 if w < 0.12 else 11, bold=("MAC" in label), fill="#eeeeee" if "MAC" in label else "white", radius=0.012)

    # Control and parameter side path.
    box(ax, 0.035, 0.40, 0.16, 0.09, "PARAM\n窗口计划 / K tile / 量化参数", size=10.5, fill="white")
    arrow(ax, (0.195, 0.445), (0.52, 0.62), label="控制与参数", label_offset=(0.0, 0.025), dashed=True, lw=1.1)

    # Explain the packet and the two different kinds of parallelism.
    box(ax, 0.235, 0.38, 0.29, 0.10, "一个 K tile：32 个输入 lane 同时广播到 32 个输出 lane\n最多 32×32 = 1024 个 INT8 MAC", size=10.5, fill="#f7f7f7")
    box(ax, 0.55, 0.38, 0.405, 0.10, "同一输出位置：K tile 之间必须累加完成\n不同输出位置：WindowGen、SA、量化/激活通过 FIFO 重叠", size=10.5, fill="#f7f7f7")

    # Timeline area.
    ax.text(0.035, 0.315, "输出行内的时间关系", ha="left", va="center", fontproperties=font(14, True))
    x0, x1 = 0.20, 0.94
    y_rows = [0.255, 0.205, 0.155, 0.105]
    labels = ["WindowGen", "SA", "量化 / 激活", "PPU / 写回"]
    for y, label in zip(y_rows, labels):
        ax.text(0.175, y + 0.018, label, ha="right", va="center", fontproperties=font(10.5, True))
        ax.plot([x0, x1], [y, y], color="black", linewidth=0.7, zorder=0)

    # Pipeline bars: each block is an abstract issue, not a cycle-accurate scale.
    bar_h = 0.027
    pattern = [
        (0.20, 0.31, "x0: K0-Kn"),
        (0.32, 0.43, "x1: K0-Kn"),
        (0.44, 0.55, "x2: K0-Kn"),
        (0.56, 0.67, "x3: K0-Kn"),
        (0.68, 0.79, "x4: K0-Kn"),
    ]
    for i, (bx, ex, label) in enumerate(pattern):
        bar(ax, bx, y_rows[0] + 0.004, ex - bx, bar_h, label, fill="#d9d9d9")
        bar(ax, bx + 0.055, y_rows[1] + 0.004, ex - bx, bar_h, label.split(":")[0], fill="#bfbfbf")
        bar(ax, bx + 0.11, y_rows[2] + 0.004, ex - bx, bar_h, label.split(":")[0], fill="#a6a6a6")
        bar(ax, bx + 0.165, y_rows[3] + 0.004, ex - bx, bar_h, label.split(":")[0], fill="#8c8c8c", text_color="white")

    ax.text(0.20, 0.065, "行 r 的所有横向位置完成后，才进入行级 PPU；当前使用单一 row buffer。", ha="left", va="center", fontproperties=font(10.5))
    ax.text(0.20, 0.035, "下一步可用两个受控 row buffer：Conv(row r+1) 与 PPU(row r) 重叠，但不复制 SA。", ha="left", va="center", fontproperties=font(10.5))

    # Boundary marker and callout.
    ax.plot([0.86, 0.86], [0.08, 0.30], color="black", linestyle="--", linewidth=1.2)
    ax.text(0.865, 0.29, "行边界", ha="left", va="bottom", fontproperties=font(9.5, True))

    fig.savefig(path, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def bar(ax: plt.Axes, x: float, y: float, w: float, h: float, label: str, *, fill: str, text_color: str = "black") -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.002,rounding_size=0.006",
        facecolor=fill,
        edgecolor="black",
        linewidth=0.8,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontproperties=font(8), color=text_color, zorder=3)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_system_architecture(OUT_DIR / "npu_system_architecture_draft.png")
    draw_conv_pipeline(OUT_DIR / "conv_engine_pipeline_draft.png")
    print(f"wrote {OUT_DIR / 'npu_system_architecture_draft.png'}")
    print(f"wrote {OUT_DIR / 'conv_engine_pipeline_draft.png'}")


if __name__ == "__main__":
    main()
