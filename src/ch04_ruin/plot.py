"""第四章绘图：赌徒破产与 OST 条件失效。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import numpy as np
import pandas as pd

from core.visualization import (
    COLOR_BLUE,
    COLOR_GRAY,
    COLOR_RED,
    add_axes_note,
    add_hline_label,
    emphasize_log_grid,
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
        "figure": "ch04_convergence",
        "chapter": "ch04_ruin",
        "role": "hero quantitative",
        "figure_type": "boxplot+reference",
        "core_conclusion": "双边停时满足 OST 并收敛到 0，单边自然停时不收敛到 0。",
        "data_files": ["output/data/exp4_convergence.csv", "output/data/exp4_convergence_batches.npz"],
        "statistics": "不同路径数下的批次箱线图，对照理论水平线 0 和 10。",
    },
    {
        "figure": "ch04_truncation",
        "chapter": "ch04_ruin",
        "role": "supporting explanatory",
        "figure_type": "boxplot+reference",
        "core_conclusion": "每个固定 N 的期望仍为 0，但极限随机变量几乎处处趋于 10。",
        "data_files": ["output/data/exp4_truncation.csv", "output/data/exp4_truncation_batches.npz"],
        "statistics": "12 个独立批次样本均值箱线图，展示极限与期望不可交换。",
    },
    {
        "figure": "ch04_tail",
        "chapter": "ch04_ruin",
        "role": "supporting explanatory",
        "figure_type": "loglog-trend",
        "core_conclusion": "单边停时呈重尾，近似遵循 t^{-1/2}，双边停时则显著更轻尾。",
        "data_files": ["output/data/exp4_tail.npz"],
        "statistics": "双对数生存函数；灰色参考线为对齐后的 t^{-1/2}。",
    },
]


def _box_widths(x: np.ndarray, ratio: float = 0.35) -> list[float]:
    x = np.asarray(x, dtype=float)
    widths = []
    for i, value in enumerate(x):
        if len(x) == 1:
            widths.append(value * ratio * 0.1 if value > 0 else ratio)
        elif i == 0:
            widths.append((x[i + 1] - value) * ratio)
        elif i == len(x) - 1:
            widths.append((value - x[i - 1]) * ratio)
        else:
            widths.append(min(value - x[i - 1], x[i + 1] - value) * ratio)
    return widths


def fig4_1_convergence() -> None:
    """图 4.1：双边与单边停时的收敛对比。"""
    set_style()
    df = pd.read_csv(DATA_DIR / "exp4_convergence.csv")
    batches = np.load(DATA_DIR / "exp4_convergence_batches.npz")

    fig, ax = new_figure(kind="trend")
    for stop_type, color, label, y_theory in [
        ("two_sided", COLOR_BLUE, "双边停时（OST 成立）", 0.0),
        ("one_sided", COLOR_RED, "单边停时（OST 失效）", 10.0),
    ]:
        sub = df[df["stop_type"] == stop_type]
        x = sub["n_paths"].to_numpy()
        samples = [batches[f'{"two" if stop_type == "two_sided" else "one"}_{int(m)}'] for m in x]
        plot_box_series(
            ax,
            x,
            samples,
            width=_box_widths(x, ratio=0.28),
            facecolor=color,
            edgecolor=color,
            median_color=COLOR_GRAY,
            label=label,
        )
        ax.axhline(y_theory, color=color, lw=0.7, ls="--", alpha=0.5)

    ax.set_xscale("log")
    emphasize_log_grid(ax)
    ax.set_xlabel("Monte Carlo 路径数 $M$")
    ax.set_ylabel("$\\mathbb{E}[S_\\tau]$ 估计值")
    add_hline_label(ax, 0.0, "OST 预测 0", color=COLOR_BLUE, x_frac=0.98)
    add_hline_label(ax, 10.0, "单边极限 10", color=COLOR_RED, x_frac=0.98)
    ax.legend(loc="lower right")
    save_figure(fig, "ch04_convergence.pdf")


def fig4_2_truncation() -> None:
    """图 4.2：截断序列。"""
    set_style()
    df = pd.read_csv(DATA_DIR / "exp4_truncation.csv")
    batches = np.load(DATA_DIR / "exp4_truncation_batches.npz")

    fig, ax = new_figure(kind="trend")
    x = df["N"].to_numpy()
    samples = [batches[f"N_{int(n_value)}"] for n_value in x]
    plot_box_series(
        ax,
        x,
        samples,
        width=_box_widths(x, ratio=0.28),
        facecolor=COLOR_BLUE,
        edgecolor=COLOR_BLUE,
        median_color=COLOR_RED,
        label="12 个独立批次",
    )

    ax.set_xscale("log")
    emphasize_log_grid(ax)
    ax.set_xlabel("截断值 $N$")
    ax.set_ylabel("$S_{\\tau \\wedge N}$ 的样本均值")
    add_hline_label(ax, 0.0, "$\\mathbb{E}[S_{\\tau\\wedge N}]=0$", color=COLOR_GRAY, linestyle=":", x_frac=0.98)
    add_hline_label(ax, 10.0, "极限 $S_\\tau=10$", color=COLOR_RED, x_frac=0.98)
    ax.legend(loc="lower left")
    save_figure(fig, "ch04_truncation.pdf")


def fig4_3_tail() -> None:
    """图 4.3：停时尾部对比。"""
    set_style()
    data = np.load(DATA_DIR / "exp4_tail.npz")

    fig, ax = new_figure(kind="trend")
    for t_values, survival, color, label, line_style in [
        (data["t_two"], data["surv_two"], COLOR_BLUE, "双边障碍", "-"),
        (data["t_one"], data["surv_one"], COLOR_RED, "单边障碍", "--"),
    ]:
        mask = survival > 1e-4
        ax.loglog(t_values[mask], survival[mask], color=color, lw=0.95, ls=line_style, label=label)

    anchor_t = min(500, len(data["surv_one"]) - 1)
    anchor_surv = max(data["surv_one"][anchor_t], 1e-12)
    t_ref = np.logspace(np.log10(50), np.log10(data["t_one"].max()), 100)
    ref = anchor_surv * np.sqrt(anchor_t / t_ref)
    ax.loglog(t_ref, ref, "--", color=COLOR_GRAY, lw=0.8, alpha=0.75, label="$C t^{-1/2}$")

    ax.set_xlabel("$t$")
    ax.set_ylabel("$P(\\tau > t)$")
    emphasize_log_grid(ax)
    ax.legend(loc="lower left")
    add_axes_note(ax, "实线：双边\n虚线：单边")
    save_figure(fig, "ch04_tail.pdf")


if __name__ == "__main__":
    set_style()
    print("绘制图 4.1: 收敛对比...")
    fig4_1_convergence()
    print("绘制图 4.2: 截断序列...")
    fig4_2_truncation()
    print("绘制图 4.3: 停时尾部...")
    fig4_3_tail()
    write_figure_manifest(MANIFEST_ROWS)
    print("第四章三张图绘制完成。")
