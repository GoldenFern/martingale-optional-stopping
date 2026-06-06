"""第四章绘图：赌徒破产与 OST 条件失效。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.processes import SymmetricRW
from core.visualization import (
    COLOR_BLUE,
    COLOR_GRAY,
    COLOR_RED,
    FIG_H,
    FIG_W,
    emphasize_log_grid,
    new_figure,
    new_figure_dual,
    save_figure,
    set_style,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "output" / "data"

LOWER_BARRIER, UPPER_BARRIER = 20, 10  # asymmetric: a=20, b=10


def fig4_1_paths(seed: int = 42) -> None:
    """图 4.1：双边障碍下的样本轨道（a=20, b=10）。"""
    set_style()
    np.random.seed(seed)
    random_walk = SymmetricRW(0.5)

    fig, ax = new_figure()
    up_target = 10   # P(up) = a/(a+b) = 20/30 = 2/3
    down_target = 5  # P(down) = b/(a+b) = 10/30 = 1/3
    up_count, down_count = 0, 0
    for _ in range(200):
        random_walk.reset(0.0)
        path = [0.0]
        for _step in range(10000):
            path.append(random_walk.step())
            if path[-1] >= UPPER_BARRIER or path[-1] <= -LOWER_BARRIER:
                break
        path_array = np.array(path)
        if path_array[-1] >= UPPER_BARRIER and up_count < up_target:
            color = COLOR_BLUE
            tau_value = len(path_array) - 1
            ax.plot(path_array, color=color, alpha=0.55, lw=0.45)
            ax.scatter(tau_value, path_array[-1], color=color, s=8, zorder=5)
            up_count += 1
        elif path_array[-1] <= -LOWER_BARRIER and down_count < down_target:
            color = COLOR_RED
            tau_value = len(path_array) - 1
            ax.plot(path_array, color=color, alpha=0.55, lw=0.45)
            ax.scatter(tau_value, path_array[-1], color=color, s=8, zorder=5)
            down_count += 1
        if up_count >= up_target and down_count >= down_target:
            break

    ax.axhline(UPPER_BARRIER, color=COLOR_BLUE, lw=0.6, ls="--", alpha=0.5)
    ax.axhline(-LOWER_BARRIER, color=COLOR_RED, lw=0.6, ls="--", alpha=0.5)
    ax.axhline(0, color=COLOR_GRAY, lw=0.4, ls=":", alpha=0.4)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=COLOR_BLUE, lw=1, label="吸收于 $+10$"),
        Line2D([0], [0], color=COLOR_RED, lw=1, label="吸收于 $-20$"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=8)
    ax.set_xlim(0, None)
    ax.set_xlabel("步数 $n$")
    ax.set_ylabel("$S_n$")
    ax.set_title(f"双边障碍样本轨道（$a={LOWER_BARRIER}, b={UPPER_BARRIER}$）")
    save_figure(fig, "ch04_paths.pdf")


def fig4_2_unilateral_paths(max_steps: int = 10000, seed: int = 43) -> None:
    """图 4.2：单边障碍下的样本轨道，上下子图。"""
    set_style()
    np.random.seed(seed)
    random_walk = SymmetricRW(0.5)

    fig, (ax1, ax2) = new_figure_dual()

    # Collect paths
    reached_paths = []
    truncated_paths = []
    for _ in range(500):
        random_walk.reset(0.0)
        path = [0.0]
        for _step in range(max_steps):
            path.append(random_walk.step())
            if path[-1] >= UPPER_BARRIER:
                reached_paths.append(np.array(path))
                break
        else:
            truncated_paths.append(np.array(path))
        if len(reached_paths) >= 12 and len(truncated_paths) >= 4:
            break

    # Top: paths that reach +b
    for path_array in reached_paths[:12]:
        tau_value = len(path_array) - 1
        ax1.plot(path_array, color=COLOR_BLUE, alpha=0.5, lw=0.45)
        ax1.scatter(tau_value, path_array[-1], color=COLOR_BLUE, s=8, zorder=5)
    ax1.axhline(UPPER_BARRIER, color=COLOR_BLUE, lw=0.6, ls="--", alpha=0.5)
    ax1.axhline(0, color=COLOR_GRAY, lw=0.4, ls=":", alpha=0.4)
    ax1.set_xlim(0, None)
    ax1.set_xlabel("步数 $n$")
    ax1.set_ylabel("$S_n$")
    ax1.set_title("成功达到 $+10$ 的路径")

    # Bottom: truncated paths
    for path_array in truncated_paths[:4]:
        ax2.plot(path_array, color=COLOR_RED, alpha=0.55, lw=0.45)
    ax2.axhline(UPPER_BARRIER, color=COLOR_BLUE, lw=0.6, ls="--", alpha=0.5)
    ax2.axhline(0, color=COLOR_GRAY, lw=0.4, ls=":", alpha=0.4)
    ax2.set_xlim(0, None)
    ax2.set_xlabel("步数 $n$")
    ax2.set_ylabel("$S_n$")
    ax2.set_title(f"未达到 $+10$ 的路径（截断 $N={max_steps}$）")

    save_figure(fig, "ch04_unilateral.pdf")


def fig4_3_tail() -> None:
    """图 4.3：停时尾部对比（双对数）。"""
    set_style()
    data = np.load(DATA_DIR / "exp4_tail.npz")

    fig, ax = new_figure()
    for time_values, survival_probs, color, label in [
        (data["t_two"], data["surv_two"], COLOR_BLUE, "双边障碍"),
        (data["t_one"], data["surv_one"], COLOR_RED, "单边障碍"),
    ]:
        positive_mask = survival_probs > 1e-4
        ax.loglog(time_values[positive_mask], survival_probs[positive_mask], color=color, lw=0.8, label=label)

    anchor_time = min(500, len(data["surv_one"]) - 1)
    anchor_survival = max(data["surv_one"][anchor_time], 1e-12)
    reference_times = np.logspace(np.log10(50), np.log10(data["t_one"].max()), 100)
    reference_survival = anchor_survival * np.sqrt(anchor_time / reference_times)
    ax.loglog(reference_times, reference_survival, "--", color=COLOR_GRAY, lw=0.8, alpha=0.75, label="$C t^{-1/2}$")

    ax.set_xlabel("$t$")
    ax.set_ylabel("$P(\\tau > t)$")
    ax.set_title(f"停时尾部对比（双对数，$a={LOWER_BARRIER}, b={UPPER_BARRIER}$）")
    emphasize_log_grid(ax)
    ax.legend(loc="lower left", fontsize=8)
    save_figure(fig, "ch04_tail.pdf")


def fig4_4_truncation_convergence() -> None:
    """图 4.4：不同截断 N 下 E[S_{τ_N}] 的收敛行为，含残差子图。"""
    set_style()
    df = pd.read_csv(DATA_DIR / "exp4_truncation.csv")
    truncation_limits = df["N"].to_numpy()
    means = df["mean"].to_numpy()
    ses = df["se"].to_numpy()

    fig = plt.figure(figsize=(FIG_W, FIG_H * 1.28), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1], hspace=0.12)
    ax_main = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax_main)

    # Top: E[S_{τ_N}] vs N (log scale for N)
    ax_main.errorbar(truncation_limits, means, yerr=1.96 * ses,
                     fmt="o", color=COLOR_BLUE, markersize=4.5, capsize=4,
                     elinewidth=1.0, markerfacecolor="white", markeredgewidth=1.0)
    ax_main.axhline(0, color=COLOR_GRAY, lw=0.6, ls="--", alpha=0.7)
    ax_main.set_xscale("log")
    ax_main.set_ylabel("$\\mathbb{E}[S_{\\tau_N}]$")
    ax_main.set_title(f"截断停时期望收敛（单边障碍 $b={UPPER_BARRIER}$）")
    emphasize_log_grid(ax_main)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # Bottom: residuals (E[S_{τ_N}] - 0 = E[S_{τ_N}])
    ax_res.axhline(0, color=COLOR_GRAY, lw=0.6, ls="--", alpha=0.7)
    ax_res.errorbar(truncation_limits, means, yerr=1.96 * ses,
                    fmt="o", color=COLOR_BLUE, markersize=4.5, capsize=4,
                    elinewidth=1.0, markerfacecolor="white", markeredgewidth=1.0)
    ax_res.set_xscale("log")
    ax_res.set_xlabel("截断 $N$")
    ax_res.set_ylabel("残差")
    emphasize_log_grid(ax_res)

    save_figure(fig, "ch04_truncation.pdf")


if __name__ == "__main__":
    set_style()
    print("绘制图 4.1: 双边样本轨道 ...")
    fig4_1_paths()
    print("绘制图 4.2: 单边样本轨道 ...")
    fig4_2_unilateral_paths()
    print("绘制图 4.3: 停时尾部 ...")
    fig4_3_tail()
    print("绘制图 4.4: 截断收敛 ...")
    fig4_4_truncation_convergence()
    print("第四章四张图绘制完成。")
