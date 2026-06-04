"""第二章绘图：Moran 模型。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import invgauss

from ch02_moran.moran_model import MoranProcess
from core.visualization import (
    COLOR_BLUE,
    COLOR_GRAY_LIGHT,
    COLOR_RED,
    add_axes_note,
    add_endpoint_label,
    new_figure,
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
        "figure": "ch02_paths",
        "chapter": "ch02_moran",
        "role": "supporting explanatory",
        "figure_type": "paths",
        "core_conclusion": "Moran 路径在吸收边界 0 和 1 处终止，固定与灭绝由随机漂变驱动。",
        "data_files": ["output/data/exp2_fixation.csv", "output/data/exp2_tau_dist.csv"],
        "statistics": "固定种子 42 的示意路径，停时用终点圆点标记。",
    },
    {
        "figure": "ch02_fixation",
        "chapter": "ch02_moran",
        "role": "hero quantitative",
        "figure_type": "boxplot+theory",
        "core_conclusion": "固定概率与初始频率线性一致，直接验证 OST 等式。",
        "data_files": ["output/data/exp2_fixation.csv", "output/data/exp2_fixation_batches.npz"],
        "statistics": "每个频率使用 10 个独立批次箱线图，并与理论直线 y=x 对比。",
    },
    {
        "figure": "ch02_tau_dist",
        "chapter": "ch02_moran",
        "role": "supporting explanatory",
        "figure_type": "distribution",
        "core_conclusion": "停时分布呈显著右偏，均值高于中位数。",
        "data_files": ["output/data/exp2_tau_dist.csv"],
        "statistics": "10000 条路径的经验分布，叠加逆高斯拟合曲线。",
    },
]


def fig2_1_paths(N: int = 100, x0: int = 50, n_display: int = 10, seed: int = 42) -> None:
    """图 2.1：基因频率路径。"""
    set_style()
    np.random.seed(seed)
    model = MoranProcess(N)
    paths_all, outcomes = [], []
    for _ in range(150):
        model.reset(x0)
        path = model.simulate_path()
        paths_all.append(path / N)
        outcomes.append(path[-1] == N)

    fix_idx = [i for i, outcome in enumerate(outcomes) if outcome]
    ext_idx = [i for i, outcome in enumerate(outcomes) if not outcome]
    n_each = max(n_display // 2, 1)
    chosen_fix = np.random.choice(fix_idx, min(n_each, len(fix_idx)), replace=False)
    chosen_ext = np.random.choice(ext_idx, min(n_each, len(ext_idx)), replace=False)

    fig, ax = new_figure(kind="paths")
    fix_x = [np.arange(len(paths_all[i])) for i in chosen_fix]
    fix_y = [paths_all[i] for i in chosen_fix]
    ext_x = [np.arange(len(paths_all[i])) for i in chosen_ext]
    ext_y = [paths_all[i] for i in chosen_ext]
    plot_path_group(ax, fix_x[1:], fix_y[1:], color=COLOR_BLUE, alpha=0.45, lw=0.7, zorder=2)
    plot_path_group(ax, ext_x[1:], ext_y[1:], color=COLOR_RED, alpha=0.42, lw=0.7, zorder=2)
    plot_path_group(ax, fix_x[:1], fix_y[:1], color=COLOR_BLUE, alpha=0.92, lw=1.15, zorder=4)
    plot_path_group(ax, ext_x[:1], ext_y[:1], color=COLOR_RED, alpha=0.9, lw=1.15, zorder=4)

    for idx in chosen_fix:
        tau = len(paths_all[idx]) - 1
        ax.scatter(tau, paths_all[idx][-1], color=COLOR_BLUE, s=10, alpha=0.85, zorder=5)
    for idx in chosen_ext:
        tau = len(paths_all[idx]) - 1
        ax.scatter(tau, paths_all[idx][-1], color=COLOR_RED, s=10, alpha=0.85, zorder=5)

    ax.axhline(0, color=COLOR_GRAY_LIGHT, lw=0.8, ls="--", alpha=0.9)
    ax.axhline(1, color=COLOR_GRAY_LIGHT, lw=0.8, ls="--", alpha=0.9)
    ax.set_xlim(0, None)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("代数")
    ax.set_ylabel("A 等位基因频率")
    handles = [
        plt.Line2D([0], [0], color=COLOR_BLUE, lw=1, label="固定"),
        plt.Line2D([0], [0], color=COLOR_RED, lw=1, label="灭绝"),
    ]
    ax.legend(handles=handles, loc="upper right")
    add_axes_note(ax, f"$N={N},\\ x_0/N=0.5$\n停时用圆点标记")
    save_figure(fig, "ch02_paths.pdf")


def fig2_2_fixation(n_paths: int = 5000) -> None:
    """图 2.2：固定概率与初始频率。"""
    _ = n_paths
    set_style()
    df = pd.read_csv(DATA_DIR / "exp2_fixation.csv")
    batches = np.load(DATA_DIR / "exp2_fixation_batches.npz")
    sub = df[df["N"] == 100]

    fig, ax = new_figure(kind="trend")
    x = sub["initial_freq"].to_numpy()
    samples = [batches[f"{freq:.1f}"] for freq in x]
    plot_box_series(
        ax,
        x,
        samples,
        width=0.035,
        facecolor=COLOR_BLUE,
        edgecolor=COLOR_BLUE,
        median_color=COLOR_RED,
        label="Monte Carlo 批次",
    )

    xs = np.linspace(0, 1, 50)
    ax.plot(xs, xs, "--", color=COLOR_RED, lw=1.1)
    ax.set_xlabel("初始频率 $x_0/N$")
    ax.set_ylabel("固定概率")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    add_endpoint_label(ax, xs[-1], xs[-1], "理论 $y=x$", color=COLOR_RED, va="bottom")
    ax.legend(loc="upper left")
    save_figure(fig, "ch02_fixation.pdf")


def fig2_3_tau_dist() -> None:
    """图 2.3：停时分布。"""
    set_style()
    tau = np.loadtxt(DATA_DIR / "exp2_tau_dist.csv", delimiter=",")

    fig, ax = new_figure(kind="distribution")
    ax.hist(
        tau,
        bins=80,
        density=True,
        alpha=0.5,
        color=COLOR_BLUE,
        edgecolor="white",
        linewidth=0.3,
        label="模拟",
    )

    min_tau = 50
    mu_hat, loc_hat, scale_hat = invgauss.fit(tau, floc=min_tau)
    xs = np.linspace(0, tau.max(), 300)
    ax.plot(
        xs,
        invgauss.pdf(xs, mu_hat, loc=loc_hat, scale=scale_hat),
        color=COLOR_RED,
        lw=1.2,
        label="逆高斯拟合",
    )

    ax.set_xlim(0, None)
    ax.set_xlabel("停时 $\\tau$（代）")
    ax.set_ylabel("概率密度")
    ax.legend(loc="upper right")
    add_axes_note(ax, f"均值 {tau.mean():.0f}\n中位数 {np.median(tau):.0f}")
    save_figure(fig, "ch02_tau_dist.pdf")


if __name__ == "__main__":
    set_style()
    print("绘制图 2.1: 基因频率路径...")
    fig2_1_paths()
    print("绘制图 2.2: 固定概率与初始频率...")
    fig2_2_fixation()
    print("绘制图 2.3: 停时分布...")
    fig2_3_tau_dist()
    write_figure_manifest(MANIFEST_ROWS)
    print("第二章三张图绘制完成。")
