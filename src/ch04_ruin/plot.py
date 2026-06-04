"""第四章绘图：赌徒破产与 OST 条件失效。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.visualization import (
    COLOR_BLUE,
    COLOR_GRAY,
    COLOR_RED,
    emphasize_log_grid,
    new_figure,
    plot_box_series,
    save_figure,
    set_style,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "output" / "data"


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
    """图 4.1：双边 vs 单边收敛对比。"""
    set_style()
    df = pd.read_csv(DATA_DIR / "exp4_convergence.csv")
    batches = np.load(DATA_DIR / "exp4_convergence_batches.npz")

    fig, ax = new_figure()
    for stop_type, color, label, y_theory in [
        ("two_sided", COLOR_BLUE, "双边停时（OST 成立）", 0),
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
        ax.axhline(y_theory, color=color, lw=0.6, ls="--", alpha=0.5)

    ax.set_xscale("log")
    emphasize_log_grid(ax)
    ax.set_xlabel("Monte Carlo 路径数 $M$")
    ax.set_ylabel("$\\mathbb{E}[S_\\tau]$ 估计值")
    ax.set_title("OST 收敛对比：双边成立 vs 单边失效")
    ax.legend(loc="best", fontsize=8)
    save_figure(fig, "ch04_convergence.pdf")


def fig4_2_truncation() -> None:
    """图 4.2：截断偏误。"""
    set_style()
    df = pd.read_csv(DATA_DIR / "exp4_truncation.csv")
    batches = np.load(DATA_DIR / "exp4_truncation_batches.npz")

    fig, ax = new_figure()
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
        label="批次样本均值分布",
    )
    ax.axhline(0.0, color=COLOR_GRAY, lw=0.8, ls=":", label="每个固定 $N$: $\\mathbb{E}[S_{\\tau\\wedge N}]=0$")
    ax.axhline(10.0, color=COLOR_RED, lw=0.8, ls="--", label="几乎处处极限 $S_\\tau=b=10$")

    ax.set_xscale("log")
    emphasize_log_grid(ax)
    ax.set_xlabel("截断值 $N$")
    ax.set_ylabel("$S_{\\tau \\wedge N}$ 的样本均值")
    ax.set_title("截断序列与不可交换极限")
    ax.legend(loc="lower left", fontsize=7)
    save_figure(fig, "ch04_truncation.pdf")


def fig4_3_tail() -> None:
    """图 4.3：停时尾部对比（双对数）。"""
    set_style()
    data = np.load(DATA_DIR / "exp4_tail.npz")

    fig, ax = new_figure()
    for t_values, survival, color, label in [
        (data["t_two"], data["surv_two"], COLOR_BLUE, "双边障碍"),
        (data["t_one"], data["surv_one"], COLOR_RED, "单边障碍"),
    ]:
        mask = survival > 1e-4
        ax.loglog(t_values[mask], survival[mask], color=color, lw=0.8, label=label)

    anchor_t = min(500, len(data["surv_one"]) - 1)
    anchor_surv = max(data["surv_one"][anchor_t], 1e-12)
    t_ref = np.logspace(np.log10(50), np.log10(data["t_one"].max()), 100)
    ref = anchor_surv * np.sqrt(anchor_t / t_ref)
    ax.loglog(t_ref, ref, "--", color=COLOR_GRAY, lw=0.8, alpha=0.75, label="$C t^{-1/2}$")

    ax.set_xlabel("$t$")
    ax.set_ylabel("$P(\\tau > t)$")
    ax.set_title("停时尾部对比 (双对数)")
    emphasize_log_grid(ax)
    ax.legend(loc="lower left", fontsize=8)
    save_figure(fig, "ch04_tail.pdf")


if __name__ == "__main__":
    set_style()
    print("绘制图 4.1: 收敛对比 ...")
    fig4_1_convergence()
    print("绘制图 4.2: 截断偏误 ...")
    fig4_2_truncation()
    print("绘制图 4.3: 停时尾部 ...")
    fig4_3_tail()
    print("第四章三张图绘制完成。")
