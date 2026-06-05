"""第三章绘图：保险破产模型。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import expon

from ch03_insurance.surplus_model import SurplusProcess
from core.visualization import (
    COLOR_BLUE,
    COLOR_GRAY,
    COLOR_GREEN,
    COLOR_ORANGE,
    COLOR_PURPLE,
    COLOR_RED,
    FIG_H,
    FIG_W,
    emphasize_log_grid,
    new_figure,
    new_figure_dual,
    plot_box_series,
    save_figure,
    set_style,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "output" / "data"


def _unpack_saved_paths(path_data, prefix: str) -> list[dict]:
    """Load padded sample paths back into variable-length arrays."""
    times = path_data[f"{prefix}_times"]
    m_vals = path_data[f"{prefix}_m_vals"]
    lengths = path_data[f"{prefix}_lengths"]
    seeds = path_data[f"{prefix}_seeds"]
    taus = path_data[f"{prefix}_taus"]

    records = []
    for i, length in enumerate(lengths):
        records.append(
            {
                "times": times[i, :length],
                "m_vals": m_vals[i, :length],
                "seed": int(seeds[i]),
                "tau": float(taus[i]),
            }
        )
    return records


def fig3_1_surplus_paths(seed: int = 42) -> None:
    """图 3.1：盈余过程轨迹。"""
    set_style()
    np.random.seed(seed)
    lam, mu, theta = 1.0, 1.0, 0.5
    c = lam * mu * (1 + theta)
    claim_dist = expon(scale=mu)
    T, u = 200.0, 5.0

    fig, ax = new_figure()
    n_ruined = 0
    for _ in range(15):
        proc = SurplusProcess(u, c, lam, claim_dist)
        times, values = proc.simulate_path(T)
        ruined = values[-1] < 0
        if ruined:
            color, alpha, zorder = COLOR_RED, 0.8, 5
            n_ruined += 1
        else:
            color, alpha, zorder = COLOR_GRAY, 0.4, 1
        # Use plot (not step) to show continuous premium growth between claims
        ax.plot(times, values, color=color, alpha=alpha, lw=0.6, zorder=zorder)
        if ruined:
            ax.scatter(times[-1], values[-1], color=COLOR_RED, s=15, marker="x", zorder=6)

    # Expected value line: E[U_t] = u + (c - lambda*mu)*t = u + theta*lambda*mu*t
    t_line = np.linspace(0, T, 500)
    expected = u + (c - lam * mu) * t_line
    ax.plot(t_line, expected, "k--", lw=1.0, alpha=0.7, label="$\\mathbb{E}[U_t]$")

    ax.axhline(0, color=COLOR_RED, lw=0.7, ls="--", alpha=0.7)
    ax.set_xlim(0, None)
    ax.set_xlabel("时间 $t$")
    ax.set_ylabel("盈余 $U_t$")
    ax.set_title(f"Cramér-Lundberg 盈余路径（破产 {n_ruined}/15）")
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=COLOR_GRAY, lw=0.8, alpha=0.6, label="未破产"),
        Line2D([0], [0], color=COLOR_RED, lw=0.8, label="破产"),
        Line2D([0], [0], color="black", lw=1.0, ls="--", alpha=0.7, label="$\\mathbb{E}[U_t]$"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    save_figure(fig, "ch03_paths.pdf")


def fig3_2_ruin_prob() -> None:
    """图 3.2：破产概率 vs 初始资本（半对数）。"""
    set_style()
    df = pd.read_csv(DATA_DIR / "exp3_ruin_prob.csv")

    fig, ax = new_figure()
    x = df["u"].to_numpy()
    y = df["psi_mc"].to_numpy()

    ax.plot(x, y, "o", color=COLOR_BLUE, markersize=5, label="有限时窗 Monte Carlo 估计（$P(\\tau \\leqslant 200)$）")
    ax.plot(df["u"], df["psi_lundberg"], "--", color=COLOR_ORANGE, lw=1.2, label="Lundberg 上界 $e^{-Ru}$")

    ax.set_yscale("log")
    emphasize_log_grid(ax)
    ax.set_xlabel("初始资本 $u$")
    ax.set_ylabel("破产概率 $\\psi(u)$")
    ax.set_title("破产概率的指数衰减")
    ax.legend(loc="upper right", fontsize=8)
    save_figure(fig, "ch03_ruin_prob.pdf")


def fig3_3_martingale_dual() -> None:
    """图 3.3：未破产路径、破产路径与均值曲线。"""
    set_style()
    mean_df = pd.read_csv(DATA_DIR / "exp3_martingale_mean.csv")
    path_data = np.load(DATA_DIR / "exp3_martingale_paths.npz")
    survived_paths = _unpack_saved_paths(path_data, "survived")
    ruined_paths = _unpack_saved_paths(path_data, "ruined")
    m0 = float(path_data["m0"])
    R = float(path_data["R"])
    T = float(path_data["T"])
    x = mean_df["t"].to_numpy()
    mean_vals = mean_df["m_mean"].to_numpy()

    fig = plt.figure(figsize=(FIG_W * 1.16, FIG_H * 1.9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.08])
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, :])

    survive_colors = [COLOR_BLUE, COLOR_GREEN, COLOR_GRAY, COLOR_PURPLE]
    ruin_colors = [COLOR_RED, COLOR_ORANGE]
    all_path_values = np.concatenate(
        [
            np.concatenate([rec["m_vals"] for rec in survived_paths]),
            np.concatenate([rec["m_vals"] for rec in ruined_paths]),
        ]
    )
    y_min = max(np.min(all_path_values) * 0.75, 1e-6)
    y_max = np.max(all_path_values) * 1.2

    for color, rec in zip(survive_colors, survived_paths):
        ax1.plot(rec["times"], rec["m_vals"], color=color, lw=0.85, label=f"seed={rec['seed']}")
    ax1.set_yscale("log")
    ax1.set_xlim(0, T)
    ax1.set_ylim(y_min, y_max)
    ax1.set_ylabel("$M_t = e^{-R U_t}$")
    ax1.set_title("未破产样本路径")
    emphasize_log_grid(ax1)

    for color, rec in zip(ruin_colors, ruined_paths):
        ax2.plot(rec["times"], rec["m_vals"], color=color, lw=0.85, label=f"seed={rec['seed']}")
        ax2.scatter(rec["times"][-1], rec["m_vals"][-1], color=color, marker="x", s=22, zorder=5)
    ax2.set_yscale("log")
    ax2.set_xlim(0, T)
    ax2.set_ylim(y_min, y_max)
    ax2.set_title("破产样本路径")
    emphasize_log_grid(ax2)

    ax3.plot(x, mean_vals, color="black", lw=1.35)
    ax3.set_xlim(0, T)
    ax3.set_xlabel("时间 $t$")
    ax3.set_ylabel("$\\widehat{\\mathbb{E}}[M_{t \\wedge \\tau}]$")
    ax3.set_title(f"停止指数鞅的均值曲线（理论初值 $M_0={m0:.3f}$，$R={R:.4f}$）")
    save_figure(fig, "ch03_martingale.pdf")


if __name__ == "__main__":
    set_style()
    print("绘制图 3.1: 盈余轨迹 ...")
    fig3_1_surplus_paths()
    print("绘制图 3.2: 破产概率 vs 初始资本 ...")
    fig3_2_ruin_prob()
    print("绘制图 3.3: 三组指数鞅诊断路径 ...")
    fig3_3_martingale_dual()
    print("第三章三张图绘制完成。")
