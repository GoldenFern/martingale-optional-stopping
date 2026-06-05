"""第四章绘图：赌徒破产与 OST 条件失效。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np

from core.processes import SymmetricRW
from core.visualization import (
    COLOR_BLUE,
    COLOR_GRAY,
    COLOR_RED,
    emphasize_log_grid,
    new_figure,
    new_figure_dual,
    save_figure,
    set_style,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "output" / "data"

A, B = 20, 10  # asymmetric: a=20, b=10


def fig4_1_paths(seed: int = 42) -> None:
    """图 4.1：双边障碍下的样本轨道（a=20, b=10）。"""
    set_style()
    np.random.seed(seed)
    rw = SymmetricRW(0.5)

    fig, ax = new_figure()
    n_display = 12
    n_up, n_down = 0, 0
    for _ in range(200):
        rw.reset(0.0)
        path = [0.0]
        for _step in range(10000):
            path.append(rw.step())
            if path[-1] >= B or path[-1] <= -A:
                break
        path_arr = np.array(path)
        if path_arr[-1] >= B and n_up < n_display // 2:
            color = COLOR_BLUE
            tau_val = len(path_arr) - 1
            ax.plot(path_arr, color=color, alpha=0.55, lw=0.45)
            ax.scatter(tau_val, path_arr[-1], color=color, s=8, zorder=5)
            n_up += 1
        elif path_arr[-1] <= -A and n_down < n_display // 2:
            color = COLOR_RED
            tau_val = len(path_arr) - 1
            ax.plot(path_arr, color=color, alpha=0.55, lw=0.45)
            ax.scatter(tau_val, path_arr[-1], color=color, s=8, zorder=5)
            n_down += 1
        if n_up >= n_display // 2 and n_down >= n_display // 2:
            break

    ax.axhline(B, color=COLOR_BLUE, lw=0.6, ls="--", alpha=0.5)
    ax.axhline(-A, color=COLOR_RED, lw=0.6, ls="--", alpha=0.5)
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
    ax.set_title(f"双边障碍样本轨道（$a={A}, b={B}$）")
    save_figure(fig, "ch04_paths.pdf")


def fig4_2_unilateral_paths(max_steps: int = 5000, seed: int = 43) -> None:
    """图 4.2：单边障碍下的样本轨道，左右子图。"""
    set_style()
    np.random.seed(seed)
    rw = SymmetricRW(0.5)

    fig, (ax1, ax2) = new_figure_dual()

    # Collect paths
    reached_paths = []
    truncated_paths = []
    for _ in range(500):
        rw.reset(0.0)
        path = [0.0]
        for _step in range(max_steps):
            path.append(rw.step())
            if path[-1] >= B:
                reached_paths.append(np.array(path))
                break
        else:
            truncated_paths.append(np.array(path))
        if len(reached_paths) >= 8 and len(truncated_paths) >= 8:
            break

    # Left: paths that reach +B
    for path_arr in reached_paths[:8]:
        tau_val = len(path_arr) - 1
        ax1.plot(path_arr, color=COLOR_BLUE, alpha=0.5, lw=0.45)
        ax1.scatter(tau_val, path_arr[-1], color=COLOR_BLUE, s=8, zorder=5)
    ax1.axhline(B, color=COLOR_BLUE, lw=0.6, ls="--", alpha=0.5)
    ax1.axhline(0, color=COLOR_GRAY, lw=0.4, ls=":", alpha=0.4)
    ax1.set_xlim(0, None)
    ax1.set_xlabel("步数 $n$")
    ax1.set_ylabel("$S_n$")
    ax1.set_title("成功达到 $+10$ 的路径")

    # Right: truncated paths
    for path_arr in truncated_paths[:8]:
        ax2.plot(path_arr, color=COLOR_RED, alpha=0.55, lw=0.45)
    ax2.axhline(B, color=COLOR_BLUE, lw=0.6, ls="--", alpha=0.5)
    ax2.axhline(0, color=COLOR_GRAY, lw=0.4, ls=":", alpha=0.4)
    ax2.set_xlim(0, None)
    ax2.set_xlabel("步数 $n$")
    ax2.set_ylabel("$S_n$")
    ax2.set_title(f"未达到 $+10$ 的路径（截断，$N={max_steps}$）")

    fig.suptitle(f"单边障碍样本轨道（$b={B}$）", y=1.01, fontsize=11)
    save_figure(fig, "ch04_unilateral.pdf")


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
    ax.set_title(f"停时尾部对比（双对数，$a={A}, b={B}$）")
    emphasize_log_grid(ax)
    ax.legend(loc="lower left", fontsize=8)
    save_figure(fig, "ch04_tail.pdf")


if __name__ == "__main__":
    set_style()
    print("绘制图 4.1: 双边样本轨道 ...")
    fig4_1_paths()
    print("绘制图 4.2: 单边样本轨道 ...")
    fig4_2_unilateral_paths()
    print("绘制图 4.3: 停时尾部 ...")
    fig4_3_tail()
    print("第四章三张图绘制完成。")
