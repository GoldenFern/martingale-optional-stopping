"""第六章绘图：美式看跌期权与 LSM。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from core.visualization import (
    COLOR_BLUE,
    COLOR_GRAY,
    COLOR_ORANGE,
    COLOR_RED,
    add_axes_note,
    add_compact_colorbar,
    add_endpoint_label,
    add_hline_label,
    add_probability_contour,
    new_figure,
    plot_box_series,
    save_figure,
    set_style,
    write_figure_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "output" / "data"

MANIFEST_ROWS = [
    {
        "figure": "ch06_paths",
        "chapter": "ch06_option",
        "role": "supporting explanatory",
        "figure_type": "paths",
        "core_conclusion": "提前行权主要发生在深度价内路径，而到期行权集中在边界附近。",
        "data_files": ["output/data/exp6_paths.npz"],
        "statistics": "15 条展示路径；红点为提前行权，橙色三角为到期行权。",
    },
    {
        "figure": "ch06_boundary",
        "chapter": "ch06_option",
        "role": "supporting explanatory",
        "figure_type": "heatmap",
        "core_conclusion": "行权概率在低股价区域明显升高，并形成随到期临近上弯的边界。",
        "data_files": ["output/data/exp6_boundary.npz"],
        "statistics": "二维行权概率热力图；白色等值线对应 50% 行权概率。",
    },
    {
        "figure": "ch06_price",
        "chapter": "ch06_option",
        "role": "hero quantitative",
        "figure_type": "boxplot+theory",
        "core_conclusion": "美式价格始终高于欧式价格，差额即提前行权溢价。",
        "data_files": ["output/data/exp6_price.csv", "output/data/exp6_price_batches.npz"],
        "statistics": "每个 S0 的 LSM 批次箱线图，对比欧式价格与立即行权 payoff。",
    },
]


def _widths_from_x(x: np.ndarray, ratio: float = 0.35) -> list[float]:
    x = np.asarray(x, dtype=float)
    widths = []
    for i, value in enumerate(x):
        if len(x) == 1:
            widths.append(0.15)
        elif i == 0:
            widths.append((x[i + 1] - value) * ratio)
        elif i == len(x) - 1:
            widths.append((value - x[i - 1]) * ratio)
        else:
            widths.append(min(value - x[i - 1], x[i + 1] - value) * ratio)
    return widths


def fig6_1_paths() -> None:
    """图 6.1：股价路径与行权时刻。"""
    set_style()
    data = np.load(DATA_DIR / "exp6_paths.npz", allow_pickle=True)
    s_paths = data["S"]
    tau = data["tau"]
    dt = 1.0 / 50

    fig, ax = new_figure(kind="paths")
    n_early, n_maturity = 0, 0
    for i, path in enumerate(s_paths):
        ax.plot(np.arange(len(path)) * dt, path, lw=0.6, color=COLOR_GRAY, alpha=0.28, zorder=1)
        t_i = tau[i] * dt
        if tau[i] < len(path) - 1:
            ax.scatter(t_i, path[tau[i]], color=COLOR_RED, s=16, zorder=5, alpha=0.9)
            n_early += 1
        elif path[tau[i]] < 40:
            ax.scatter(t_i, path[tau[i]], color=COLOR_ORANGE, marker="^", s=18, zorder=5, alpha=0.9)
            n_maturity += 1

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_RED, markeredgecolor=COLOR_RED, markersize=5, label=f"提前行权 {n_early} 条"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=COLOR_ORANGE, markeredgecolor=COLOR_ORANGE, markersize=5, label=f"到期行权 {n_maturity} 条"),
    ]
    ax.set_xlabel("时间 $t$")
    ax.set_ylabel("股价 $S_t$")
    add_hline_label(ax, 40.0, "执行价 $K$", color=COLOR_BLUE, x_frac=0.98)
    ax.legend(handles=handles, loc="upper right")
    save_figure(fig, "ch06_paths.pdf")


def fig6_2_boundary() -> None:
    """图 6.2：行权概率热力图。"""
    set_style()
    data = np.load(DATA_DIR / "exp6_boundary.npz")
    t_grid, s_grid, prob = data["t_grid"], data["s_grid"], data["prob"]

    fig, ax = new_figure(kind="heatmap")
    image = ax.contourf(t_grid, s_grid, prob.T, levels=np.linspace(0, 1, 21), cmap="cividis", vmin=0, vmax=1)
    add_probability_contour(ax, t_grid, s_grid, prob, level=0.5, color="white", linewidth=0.9)
    add_compact_colorbar(fig, image, ax, label="行权概率", shrink=0.8, fraction=0.055)
    ax.set_xlabel("时间 $t$")
    ax.set_ylabel("股价 $S$")
    add_axes_note(ax, "白色线：50% 行权概率")
    save_figure(fig, "ch06_boundary.pdf")


def fig6_3_price() -> None:
    """图 6.3：美式与欧式价格对比。"""
    set_style()
    df = pd.read_csv(DATA_DIR / "exp6_price.csv")
    batches = np.load(DATA_DIR / "exp6_price_batches.npz")

    fig, ax = new_figure(kind="trend")
    x = df["S0"].to_numpy()
    samples = [batches[f"S0_{s0:.1f}"] for s0 in x]
    plot_box_series(
        ax,
        x,
        samples,
        width=_widths_from_x(x, ratio=0.3),
        facecolor=COLOR_BLUE,
        edgecolor=COLOR_BLUE,
        median_color=COLOR_RED,
        label="美式期权（LSM 批次）",
    )
    ax.plot(df["S0"], df["european"], "-", color=COLOR_RED, lw=1.0)
    ax.plot(df["S0"], df["payoff"], "--", color=COLOR_GRAY, lw=0.9, alpha=0.75)

    ax.set_xlabel("初始股价 $S_0$")
    ax.set_ylabel("期权价格")
    add_endpoint_label(ax, df["S0"].iloc[-1], df["european"].iloc[-1], "欧式 BS 价格", color=COLOR_RED, y_pad_frac=0.03)
    add_endpoint_label(ax, df["S0"].iloc[-1], df["payoff"].iloc[-1], "立即行权 payoff", color=COLOR_GRAY, y_pad_frac=-0.03, va="top")
    ax.legend(loc="upper right")
    save_figure(fig, "ch06_price.pdf")


if __name__ == "__main__":
    set_style()
    print("绘制图 6.1: 路径与行权位置...")
    fig6_1_paths()
    print("绘制图 6.2: 行权边界热力图...")
    fig6_2_boundary()
    print("绘制图 6.3: 价格对比...")
    fig6_3_price()
    write_figure_manifest(MANIFEST_ROWS)
    print("第六章三张图绘制完成。")
