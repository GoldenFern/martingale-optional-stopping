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
    COLOR_GRAY,
    COLOR_GREEN,
    COLOR_ORANGE,
    COLOR_RED,
    emphasize_log_grid,
    new_figure,
    new_figure_dual,
    plot_box_series,
    save_figure,
    set_style,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "output" / "data"


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
        ax.step(times, values, where="post", color=color, alpha=alpha, lw=0.6, zorder=zorder)
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


def fig3_3_martingale_dual(seed: int = 777) -> None:
    """图 3.3：双子图，盈余轨迹与指数鞅。"""
    set_style()
    lam, mu, theta = 1.0, 1.0, 0.5
    c = lam * mu * (1 + theta)
    claim_dist = expon(scale=mu)
    mgf = exp_claim_mgf_factory("exponential", rate=1 / mu)
    R = find_adjustment_R(lam, c, mgf)
    u, T = 5.0, 200.0

    # Try multiple seeds to find a path where M_t stays visible
    best_path = None
    for seed_try in [777, 42, 123, 999, 333, 555, 888, 111, 222, 444]:
        np.random.seed(seed_try)
        proc = SurplusProcess(u, c, lam, claim_dist)
        times, u_vals = proc.simulate_path(T)
        m_vals = np.exp(-R * u_vals)
        # Pick a path that doesn't go bankrupt and has M_t in a visible range
        if u_vals[-1] >= 0 and np.max(m_vals) < 5.0:
            best_path = (times, u_vals, m_vals)
            break

    if best_path is None:
        np.random.seed(seed)
        proc = SurplusProcess(u, c, lam, claim_dist)
        times, u_vals = proc.simulate_path(T)
        m_vals = np.exp(-R * u_vals)
    else:
        times, u_vals, m_vals = best_path

    m0 = np.exp(-R * u)

    fig, (ax1, ax2) = new_figure_dual()
    ax1.step(times, u_vals, where="post", color=COLOR_BLUE, lw=0.6)
    t_line = np.linspace(0, T, 500)
    expected = u + (c - lam * mu) * t_line
    ax1.plot(t_line, expected, "k--", lw=1.0, alpha=0.7, label="$\\mathbb{E}[U_t]$")
    ax1.axhline(0, color=COLOR_RED, lw=0.6, ls="--", alpha=0.5)
    ax1.set_xlim(0, None)
    ax1.set_ylabel("盈余 $U_t$")
    ax1.set_title("盈余过程")
    ax1.legend(loc="upper left", fontsize=7)

    ax2.step(times, m_vals, where="post", color=COLOR_GREEN, lw=0.6)
    ax2.axhline(m0, color=COLOR_GRAY, lw=0.6, ls=":", alpha=0.6,
                label=f"初始值 $M_0 = e^{{-R u}} = {m0:.3f}$")
    ax2.set_xlim(0, None)
    ax2.set_xlabel("时间 $t$")
    ax2.set_ylabel("$M_t = e^{-R U_t}$")
    ax2.set_title(f"指数鞅（$R={R:.4f}$）")
    ax2.legend(loc="upper right", fontsize=7)
    save_figure(fig, "ch03_martingale.pdf")


if __name__ == "__main__":
    set_style()
    print("绘制图 3.1: 盈余轨迹 ...")
    fig3_1_surplus_paths()
    print("绘制图 3.2: 破产概率 vs 初始资本 ...")
    fig3_2_ruin_prob()
    print("绘制图 3.3: 盈余 + 指数鞅双子图 ...")
    fig3_3_martingale_dual()
    print("第三章三张图绘制完成。")
