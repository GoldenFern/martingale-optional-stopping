"""第二章绘图：Moran 模型。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import invgauss

from ch02_moran.moran_model import MoranProcess, expected_tau_moran
from core.visualization import COLOR_BLUE, COLOR_RED, FIG_H, FIG_W, new_figure, plot_box_series, save_figure, set_style
from scipy.interpolate import CubicSpline

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "output" / "data"


def fig2_1_paths(pop_size: int = 50, initial_allele_count: int = 25, n_display: int = 10, seed: int = 42) -> None:
    """图 2.1：基因频率轨迹叠加。"""
    set_style()
    np.random.seed(seed)
    model = MoranProcess(pop_size)
    paths_all, outcomes = [], []
    for _ in range(150):
        model.reset(initial_allele_count)
        path = model.simulate_path()
        paths_all.append(path / pop_size)
        outcomes.append(path[-1] == pop_size)
    fixation_indices = [i for i, outcome in enumerate(outcomes) if outcome]
    extinction_indices = [i for i, outcome in enumerate(outcomes) if not outcome]
    count_per_group = max(n_display // 2, 1)

    fig, ax = new_figure()
    for idx_list, color, label in [
        (fixation_indices, COLOR_BLUE, "A固定"),
        (extinction_indices, COLOR_RED, "a固定"),
    ]:
        chosen = np.random.choice(idx_list, min(count_per_group, len(idx_list)), replace=False)
        for path_idx in chosen:
            allele_freq_path = paths_all[path_idx]
            stopping_time = len(allele_freq_path) - 1
            ax.plot(allele_freq_path, alpha=0.5, lw=0.4, color=color)
            ax.scatter(stopping_time, allele_freq_path[-1], color=color, s=6, alpha=0.7, zorder=5)

    ax.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.6)
    ax.axhline(1, color="gray", lw=0.5, ls="--", alpha=0.6)
    ax.set_xlim(0, None)
    ax.set_xlabel("代数")
    ax.set_ylabel("A 等位基因频率")
    ax.set_title(f"Moran 模型样本路径（$N={pop_size}$）")
    handles = [
        plt.Line2D([0], [0], color=COLOR_BLUE, lw=1, label="A固定"),
        plt.Line2D([0], [0], color=COLOR_RED, lw=1, label="a固定"),
    ]
    ax.legend(handles=handles, loc="best")
    save_figure(fig, "ch02_paths.pdf")


def fig2_2_fixation(num_paths: int = 50000) -> None:
    """图 2.2：固定概率 vs A等位基因初始频率，仅 N=50。"""
    set_style()
    results_df = pd.read_csv(DATA_DIR / "exp2_fixation.csv")
    subset_df = results_df[results_df["N"] == 50]
    batch_data = np.load(DATA_DIR / "exp2_fixation_batches.npz")

    initial_freq = subset_df["initial_freq"].to_numpy()
    fixation_prob_mc = subset_df["p_fixation_mc"].to_numpy()
    # Binomial SE — exact for independent Bernoulli trials
    fixation_se = np.sqrt(fixation_prob_mc * (1 - fixation_prob_mc) / num_paths)
    residuals = fixation_prob_mc - initial_freq

    fig = plt.figure(figsize=(FIG_W, FIG_H * 1.28), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1], hspace=0.12)
    ax_main = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax_main)

    # ---- top: main comparison with confidence band ----
    x_dense = np.linspace(0, 1, 200)
    theory_se_band = 1.96 * np.sqrt(x_dense * (1 - x_dense) / num_paths)
    ax_main.fill_between(x_dense, x_dense - theory_se_band, x_dense + theory_se_band,
                         color=COLOR_RED, alpha=0.10, lw=0, label="95% 置信带（二项 SE）")
    ax_main.plot(x_dense, x_dense, "-", color=COLOR_RED, lw=1.15,
                 label="理论值 $y=x_0/N$")
    ax_main.plot(initial_freq, fixation_prob_mc, "o", color=COLOR_BLUE,
                 markersize=5, markerfacecolor="white", markeredgewidth=1.0,
                 label="Monte Carlo 估计")
    ax_main.set_ylabel("固定概率")
    ax_main.set_title("固定概率与 A 等位基因初始频率（$N=50$）")
    ax_main.legend(loc="upper left", fontsize=8)
    ax_main.set_xlim(-0.02, 1.02)
    ax_main.set_ylim(-0.02, 1.02)
    ax_main.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # ---- bottom: residuals ----
    ax_res.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.7)
    ax_res.errorbar(initial_freq, residuals, yerr=1.96 * fixation_se,
                    fmt="o", color=COLOR_BLUE, markersize=4.5, capsize=4,
                    elinewidth=1.0, markerfacecolor="white", markeredgewidth=1.0)
    ax_res.set_xlabel("A 等位基因初始频率 $x_0/N$")
    ax_res.set_ylabel("残差")
    ax_res.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    save_figure(fig, "ch02_fixation.pdf")


def fig2_3_tau_comparison(num_paths: int = 50000) -> None:
    """图 2.3：不同 A等位基因初始频率下 Monte Carlo 停时均值与理论解对比，含残差子图。"""
    set_style()
    results_df = pd.read_csv(DATA_DIR / "exp2_fixation.csv")
    subset_df = results_df[results_df["N"] == 50]

    initial_freq = subset_df["initial_freq"].to_numpy()
    mean_tau_mc = subset_df["tau_mc_mean"].to_numpy()
    mean_tau_theory = subset_df["tau_theory"].to_numpy()
    tau_se = subset_df["tau_mc_se"].to_numpy()
    residuals = mean_tau_mc - mean_tau_theory

    # Full theory curve at all interior states
    N = 50
    tau_full = expected_tau_moran(N)
    freq_full = np.arange(0, N + 1) / N  # x-coords for all states
    # Interpolate SE to all interior states (use only sampled nonzero SE)
    se_interp = CubicSpline(np.concatenate([[0], initial_freq, [1]]),
                            np.concatenate([[0.0], tau_se, [0.0]]))
    se_full = se_interp(freq_full)
    se_full = np.maximum(se_full, 0)  # clip negative interpolation artifacts

    fig = plt.figure(figsize=(FIG_W, FIG_H * 1.28), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1], hspace=0.12)
    ax_main = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax_main)

    # ---- top: main comparison with confidence band ----
    ax_main.fill_between(freq_full, tau_full - 1.96 * se_full, tau_full + 1.96 * se_full,
                         color=COLOR_RED, alpha=0.10, lw=0, label="95% 置信带（插值 SE）")
    ax_main.plot(freq_full, tau_full, "-", color=COLOR_RED, lw=1.15,
                 label="理论值（三对角系统求解）")
    ax_main.plot(initial_freq, mean_tau_mc, "o", color=COLOR_BLUE,
                 markersize=5, markerfacecolor="white", markeredgewidth=1.0,
                 label="Monte Carlo 估计")
    ax_main.set_ylabel("期望停时 $\\mathbb{E}[\\tau]$")
    ax_main.set_title("停时期望：Monte Carlo 与理论解对比（$N=50$）")
    ax_main.legend(loc="upper left", fontsize=8)
    ax_main.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # ---- bottom: residuals ----
    ax_res.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.7)
    ax_res.errorbar(initial_freq, residuals, yerr=1.96 * tau_se,
                    fmt="o", color=COLOR_BLUE, markersize=4.5, capsize=4,
                    elinewidth=1.0, markerfacecolor="white", markeredgewidth=1.0)
    ax_res.set_xlabel("A 等位基因初始频率 $x_0/N$")
    ax_res.set_ylabel("残差")
    ax_res.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    save_figure(fig, "ch02_tau_sim.pdf")


def fig2_4_tau_dist() -> None:
    """图 2.4：停时分布直方图，N=50。"""
    set_style()
    tau_samples = np.loadtxt(DATA_DIR / "exp2_tau_dist.csv", delimiter=",")

    fig, ax = new_figure()
    ax.hist(
        tau_samples,
        bins=80,
        density=True,
        alpha=0.55,
        color=COLOR_BLUE,
        edgecolor="white",
        linewidth=0.3,
        label="模拟",
    )

    min_tau = 25
    shape_fit, loc_fit, scale_fit = invgauss.fit(tau_samples, floc=min_tau)
    fitted_x = np.linspace(0, tau_samples.max(), 300)
    ax.plot(
        fitted_x,
        invgauss.pdf(fitted_x, shape_fit, loc=loc_fit, scale=scale_fit),
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
