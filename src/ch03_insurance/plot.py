"""第三章绘图：保险破产模型。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import expon

from ch03_insurance.surplus_model import SurplusProcess, exp_claim_mgf_factory, find_adjustment_R
from core.visualization import (
    COLOR_BLUE,
    COLOR_BLUE_LIGHT,
    COLOR_GRAY,
    COLOR_ORANGE,
    COLOR_RED,
    add_axes_note,
    add_endpoint_label,
    add_hline_label,
    emphasize_log_grid,
    new_figure,
    new_figure_dual,
    plot_box_series,
    plot_path_group,
    save_figure,
    set_style,
    write_figure_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "output" / "data"

MANIFEST_ROWS = [
    {
        "figure": "ch03_paths",
        "chapter": "ch03_insurance",
        "role": "supporting explanatory",
        "figure_type": "paths",
        "core_conclusion": "盈余路径分为缓慢上升的存活样本与触及破产线的少数样本。",
        "data_files": ["output/data/exp3_ruin_prob.csv"],
        "statistics": "固定种子 42 的 15 条示意路径；破产时刻以 x 标记。",
    },
    {
        "figure": "ch03_ruin_prob",
        "chapter": "ch03_insurance",
        "role": "hero quantitative",
        "figure_type": "boxplot+theory",
        "core_conclusion": "精确解、Lundberg 上界与 Monte Carlo 共同呈现指数衰减结构。",
        "data_files": ["output/data/exp3_ruin_prob.csv", "output/data/exp3_ruin_prob_batches.npz"],
        "statistics": "有限时窗 Monte Carlo 批次箱线图；与精确解和上界在半对数坐标下比较。",
    },
    {
        "figure": "ch03_martingale",
        "chapter": "ch03_insurance",
        "role": "supporting explanatory",
        "figure_type": "dual",
        "core_conclusion": "指数变换把带正漂移的盈余过程转化为围绕 1 波动的指数鞅。",
        "data_files": ["output/data/exp3_ruin_prob.csv"],
        "statistics": "固定种子 123 的单路径示意；上下子图共用时间轴。",
    },
]


def fig3_1_surplus_paths(seed: int = 42) -> None:
    """图 3.1：盈余过程路径。"""
    set_style()
    np.random.seed(seed)
    lam, mu, theta = 1.0, 1.0, 0.2
    c = lam * mu * (1 + theta)
    claim_dist = expon(scale=mu)
    T, u = 50.0, 5.0

    fig, ax = new_figure(kind="paths")
    alive_x, alive_y, ruined_x, ruined_y = [], [], [], []
    ruin_points = []
    for _ in range(15):
        proc = SurplusProcess(u, c, lam, claim_dist)
        times, values = proc.simulate_path(T)
        if values[-1] < 0:
            ruined_x.append(times)
            ruined_y.append(values)
            ruin_points.append((times[-1], values[-1]))
        else:
            alive_x.append(times)
            alive_y.append(values)

    plot_path_group(ax, alive_x, alive_y, color=COLOR_GRAY, alpha=0.28, lw=0.7, zorder=1, step_where="post")
    plot_path_group(ax, ruined_x, ruined_y, color=COLOR_RED, alpha=0.82, lw=0.85, zorder=4, step_where="post")
    for t_ruin, u_ruin in ruin_points:
        ax.scatter(t_ruin, u_ruin, color=COLOR_RED, s=18, marker="x", zorder=6)

    ax.set_xlim(0, None)
    ax.set_xlabel("时间 $t$")
    ax.set_ylabel("盈余 $U_t$")
    handles = [
        plt.Line2D([0], [0], color=COLOR_GRAY, lw=0.8, alpha=0.7, label="未破产"),
        plt.Line2D([0], [0], color=COLOR_RED, lw=0.8, label="破产"),
    ]
    ax.legend(handles=handles, loc="upper right")
    add_hline_label(ax, 0.0, "破产线 $U=0$", color=COLOR_RED, x_frac=0.98)
    add_axes_note(ax, f"$\\theta=0.2,\\ u=5,\\ T=50$\n破产 {len(ruined_y)}/15")
    save_figure(fig, "ch03_paths.pdf")


def fig3_2_ruin_prob() -> None:
    """图 3.2：破产概率与初始资本。"""
    set_style()
    df = pd.read_csv(DATA_DIR / "exp3_ruin_prob.csv")
    batches = np.load(DATA_DIR / "exp3_ruin_prob_batches.npz")

    fig, ax = new_figure(kind="trend")
    positions = []
    samples = []
    for u in df["u"].to_numpy():
        vals = batches[f"u_{int(u)}"]
        positive = vals[vals > 0]
        if positive.size > 0:
            positions.append(u)
            samples.append(positive)
    if positions:
        plot_box_series(
            ax,
            np.asarray(positions),
            samples,
            width=0.45,
            facecolor=COLOR_BLUE,
            edgecolor=COLOR_BLUE,
            median_color=COLOR_RED,
            label="Monte Carlo 批次",
        )

    ax.plot(df["u"], df["psi_lundberg"], "--", color=COLOR_ORANGE, lw=1.2, dashes=(5, 2))
    ax.plot(df["u"], df["psi_exact"], "-", color=COLOR_RED, lw=1.05)
    ax.set_yscale("log")
    emphasize_log_grid(ax)
    ax.set_xlabel("初始资本 $u$")
    ax.set_ylabel("破产概率 $\\psi(u)$")
    add_endpoint_label(ax, df["u"].iloc[-1], df["psi_exact"].iloc[-1], "精确解 $\\psi(u)$", color=COLOR_RED, y_pad_frac=0.02)
    add_endpoint_label(ax, df["u"].iloc[-1], df["psi_lundberg"].iloc[-1], "Lundberg 上界", color=COLOR_ORANGE, y_pad_frac=0.045)
    ax.legend(loc="upper right")
    save_figure(fig, "ch03_ruin_prob.pdf")


def fig3_3_martingale_dual(seed: int = 123) -> None:
    """图 3.3：盈余过程与指数鞅。"""
    set_style()
    np.random.seed(seed)
    lam, mu, theta = 1.0, 1.0, 0.2
    c = lam * mu * (1 + theta)
    claim_dist = expon(scale=mu)
    mgf = exp_claim_mgf_factory("exponential", rate=1 / mu)
    R = find_adjustment_R(lam, c, mgf)
    u, T = 5.0, 100.0

    proc = SurplusProcess(u, c, lam, claim_dist)
    times, u_vals = proc.simulate_path(T)
    m_vals = np.exp(-R * u_vals)

    fig, (ax1, ax2) = new_figure_dual(kind="dual")
    ax1.step(times, u_vals, where="post", color=COLOR_BLUE, lw=0.8)
    ax1.set_xlim(0, None)
    ax1.set_ylabel("盈余 $U_t$")
    add_axes_note(ax1, "正漂移向上")
    add_hline_label(ax1, 0.0, "破产线 $U=0$", color=COLOR_RED, x_frac=0.98)

    ax2.step(times, m_vals, where="post", color=COLOR_BLUE_LIGHT, lw=0.8)
    ax2.set_xlim(0, None)
    ax2.set_xlabel("时间 $t$")
    ax2.set_ylabel("$M_t = e^{-R U_t}$")
    add_hline_label(ax2, 1.0, "$M_t=1$", color=COLOR_GRAY, linestyle=":", x_frac=0.98)
    add_axes_note(ax2, f"$R={R:.4f}$\n围绕 1 波动")
    save_figure(fig, "ch03_martingale.pdf")


if __name__ == "__main__":
    set_style()
    print("绘制图 3.1: 盈余路径...")
    fig3_1_surplus_paths()
    print("绘制图 3.2: 破产概率与初始资本...")
    fig3_2_ruin_prob()
    print("绘制图 3.3: 盈余过程与指数鞅...")
    fig3_3_martingale_dual()
    write_figure_manifest(MANIFEST_ROWS)
    print("第三章三张图绘制完成。")
