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
from core.visualization import COLOR_BLUE, COLOR_RED, new_figure, plot_box_series, save_figure, set_style

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "output" / "data"


def fig2_1_paths(N: int = 50, x0: int = 25, n_display: int = 10, seed: int = 42) -> None:
    """图 2.1：基因频率轨迹叠加。"""
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

    fig, ax = new_figure()
    for idx_list, color, label in [
        (fix_idx, COLOR_BLUE, "A固定"),
        (ext_idx, COLOR_RED, "a固定"),
    ]:
        chosen = np.random.choice(idx_list, min(n_each, len(idx_list)), replace=False)
        for idx in chosen:
            freq = paths_all[idx]
            tau = len(freq) - 1
            ax.plot(freq, alpha=0.5, lw=0.4, color=color)
            ax.scatter(tau, freq[-1], color=color, s=6, alpha=0.7, zorder=5)

    ax.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.6)
    ax.axhline(1, color="gray", lw=0.5, ls="--", alpha=0.6)
    ax.set_xlim(0, None)
    ax.set_xlabel("代数")
    ax.set_ylabel("A 等位基因频率")
    ax.set_title(f"Moran 模型样本路径（$N={N}$）")
    handles = [
        plt.Line2D([0], [0], color=COLOR_BLUE, lw=1, label="A固定"),
        plt.Line2D([0], [0], color=COLOR_RED, lw=1, label="a固定"),
    ]
    ax.legend(handles=handles, loc="best")
    save_figure(fig, "ch02_paths.pdf")


def fig2_2_fixation(n_paths: int = 5000) -> None:
    """图 2.2：固定概率 vs A等位基因初始频率，仅 N=50。"""
    _ = n_paths
    set_style()
    df = pd.read_csv(DATA_DIR / "exp2_fixation.csv")
    sub = df[df["N"] == 50]

    fig, ax = new_figure()
    x = sub["initial_freq"].to_numpy()
    y = sub["p_fixation_mc"].to_numpy()

    ax.plot(x, y, "o", color=COLOR_BLUE, markersize=5, label="Monte Carlo 估计")
    ax.plot(x, x, "--", color=COLOR_RED, lw=1.15, label="理论值 $y=x_0/N$")

    ax.set_xlabel("A 等位基因初始频率 $x_0/N$")
    ax.set_ylabel("固定概率")
    ax.set_title("固定概率与 A 等位基因初始频率（$N=50$）")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    save_figure(fig, "ch02_fixation.pdf")


def fig2_3_tau_comparison() -> None:
    """图 2.3：不同 A等位基因初始频率下 Monte Carlo 停时均值与理论解对比。"""
    set_style()
    df = pd.read_csv(DATA_DIR / "exp2_fixation.csv")
    sub = df[df["N"] == 50]

    fig, ax = new_figure()
    x = sub["initial_freq"].to_numpy()
    y_mc = sub["tau_mc_mean"].to_numpy()
    y_theory = sub["tau_theory"].to_numpy()

    ax.plot(x, y_mc, "o", color=COLOR_BLUE, markersize=5, label="Monte Carlo 估计")
    ax.plot(x, y_theory, "--", color=COLOR_RED, lw=1.15, label="理论值（三对角系统求解）")

    ax.set_xlabel("A 等位基因初始频率 $x_0/N$")
    ax.set_ylabel("期望停时 $\\mathbb{E}[\\tau]$")
    ax.set_title("停时期望：Monte Carlo 与理论解对比（$N=50$）")
    ax.legend(loc="upper left", fontsize=8)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    save_figure(fig, "ch02_tau_sim.pdf")


def fig2_4_tau_dist() -> None:
    """图 2.4：停时分布直方图，N=50。"""
    set_style()
    tau = np.loadtxt(DATA_DIR / "exp2_tau_dist.csv", delimiter=",")

    fig, ax = new_figure()
    ax.hist(
        tau,
        bins=80,
        density=True,
        alpha=0.55,
        color=COLOR_BLUE,
        edgecolor="white",
        linewidth=0.3,
        label="模拟",
    )

    min_tau = 25
    mu_hat, loc_hat, scale_hat = invgauss.fit(tau, floc=min_tau)
    xs = np.linspace(0, tau.max(), 300)
    ax.plot(
        xs,
        invgauss.pdf(xs, mu_hat, loc=loc_hat, scale=scale_hat),
        color=COLOR_RED,
        lw=1.2,
        label="逆高斯分布拟合",
    )

    ax.set_xlim(0, 10000)
    ax.set_xlabel("停时 $\\tau$（代）")
    ax.set_ylabel("概率密度")
    ax.set_title("停时分布（$N=50, x_0=25$）")
    ax.legend(loc="upper right", fontsize=8)
    save_figure(fig, "ch02_tau_dist.pdf")


if __name__ == "__main__":
    set_style()
    print("绘制图 2.1: 基因频率轨迹 ...")
    fig2_1_paths()
    print("绘制图 2.2: 固定概率 vs A等位基因初始频率 ...")
    fig2_2_fixation()
    print("绘制图 2.3: 停时期望对比 ...")
    fig2_3_tau_comparison()
    print("绘制图 2.4: 停时分布 ...")
    fig2_4_tau_dist()
    print("第二章四张图绘制完成。")
