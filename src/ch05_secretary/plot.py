"""第五章绘图：秘书问题与 Snell 包络。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

from core.visualization import (
    COLOR_BLUE,
    COLOR_GRAY,
    COLOR_GREEN,
    COLOR_ORANGE,
    COLOR_RED,
    new_figure,
    new_figure_dual,
    plot_box_series,
    save_figure,
    set_style,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "output" / "data"


def _widths_from_x(x: np.ndarray, ratio: float = 0.35) -> list[float]:
    x = np.asarray(x, dtype=float)
    widths = []
    for i, value in enumerate(x):
        if len(x) == 1:
            widths.append(0.03)
        elif i == 0:
            widths.append((x[i + 1] - value) * ratio)
        elif i == len(x) - 1:
            widths.append((value - x[i - 1]) * ratio)
        else:
            widths.append(min(value - x[i - 1], x[i + 1] - value) * ratio)
    return widths


def _sample_indices(n: int, target: int = 12) -> np.ndarray:
    if n <= target:
        return np.arange(n)
    return np.unique(np.round(np.linspace(0, n - 1, target)).astype(int))


def fig5_1_success() -> None:
    """图 5.1：成功率 vs 观察比例 r/N，使用误差棒。"""
    set_style()
    df = pd.read_csv(DATA_DIR / "exp5_success.csv")

    fig, ax = new_figure()
    colors = [COLOR_BLUE, COLOR_RED, COLOR_ORANGE, COLOR_GREEN]
    for color, (n_value, group) in zip(colors, df.groupby("N")):
        group = group.sort_values("r_frac").drop_duplicates(subset="r", keep="first")
        x_all = group["r_frac"].to_numpy()
        y_all = group["success_mc"].to_numpy()
        se_all = group["se"].to_numpy()
        keep = _sample_indices(len(x_all), target=15)
        x = x_all[keep]
        y = y_all[keep]
        se = se_all[keep]
        theory_all = group["theory"].to_numpy()
        from core.visualization import plot_with_ci
        plot_with_ci(
            ax, x, y, se,
            label=f"$N={n_value}$",
            color=color,
            marker="o",
            theory_y=theory_all[keep],
            theory_label=f"$N={n_value}$ 理论" if n_value == 20 else "",
            theory_color=color,
            theory_style="-",
        )

    ax.axvline(1 / np.e, color=COLOR_GRAY, lw=0.6, ls="--", alpha=0.6, label="$r^*/N = 1/e$")
    ax.axhline(1 / np.e, color=COLOR_GRAY, lw=0.6, ls=":", alpha=0.6)

    ax.set_xlabel("观察比例 $r/N$")
    ax.set_ylabel("选中最优者的概率")
    ax.set_title("秘书问题成功率")
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.legend(loc="upper right", fontsize=7)
    save_figure(fig, "ch05_success.pdf")


def fig5_2_snell() -> None:
    """图 5.2：Snell 包络 V_n 与即时 payoff g(n)。"""
    set_style()
    data = np.load(DATA_DIR / "exp5_snell.npz")
    v_vals, g_vals = data["V"], data["g"]
    N = len(v_vals)
    n = np.arange(N)

    fig, ax = new_figure()
    ax.plot(n, v_vals, "-", color=COLOR_BLUE, lw=1.25, label="Snell 包络 $V_n$")
    ax.plot(n, g_vals, "--", color=COLOR_RED, lw=1.0, label="即时收益 $g(n)$")

    cross = np.where(np.isclose(v_vals, g_vals))[0]
    if len(cross) > 0:
        r_star = cross[0]
        ax.axvspan(0, r_star, color=COLOR_BLUE, alpha=0.06, lw=0)
        ax.axvspan(r_star, n[-1], color=COLOR_RED, alpha=0.04, lw=0)
        ax.axvline(r_star, color=COLOR_GRAY, lw=0.8, ls=":", alpha=0.8)
        ax.annotate(
            f"$r^*={r_star}$",
            xy=(r_star, v_vals[r_star]),
            xytext=(r_star + 1.0, v_vals[r_star] + 0.08),
            arrowprops={"arrowstyle": "->", "lw": 0.6, "color": COLOR_GRAY},
            fontsize=8,
            color=COLOR_GRAY,
        )

    ax.set_xlabel("候选人序号 $n$")
    ax.set_ylabel("条件成功概率")
    ax.set_title(f"Snell 包络与停止阈值（$N={N}$）")
    ax.legend(loc="lower left", fontsize=8)
    save_figure(fig, "ch05_snell.pdf")


def fig5_3_paths(seed: int = 123) -> None:
    """图 5.3：秘书问题排名路径可视化（成功 vs 失败）。"""
    set_style()
    np.random.seed(seed)
    N = 100
    r_star = int(N / np.e)  # 37

    # Generate trials until we get one success and one failure
    success_path = None
    failure_path = None
    for _ in range(10000):
        ranks = np.random.permutation(N) + 1  # 1=best
        benchmark = np.min(ranks[:r_star])
        stopped = False
        for i in range(r_star, N):
            if ranks[i] < benchmark:
                if ranks[i] == 1:
                    success_path = ranks
                else:
                    failure_path = ranks
                stopped = True
                break
        if not stopped:
            failure_path = ranks
        if success_path is not None and failure_path is not None:
            break

    if success_path is None or failure_path is None:
        return

    fig, (ax1, ax2) = new_figure_dual()

    for ax, ranks_arr, title in [
        (ax1, success_path, "成功选择"),
        (ax2, failure_path, "失败选择"),
    ]:
        n_vals = np.arange(1, N + 1)
        ax.plot(n_vals, ranks_arr, color=COLOR_GRAY, lw=0.5, alpha=0.7)
        # Mark the benchmark period
        ax.axvspan(1, r_star, color=COLOR_BLUE, alpha=0.06, lw=0)
        # Mark the selection period
        ax.axvspan(r_star, N, color=COLOR_RED, alpha=0.04, lw=0)
        ax.axvline(r_star, color=COLOR_GRAY, lw=0.8, ls=":", alpha=0.8)
        # Mark where the strategy stopped (the first historical best in selection phase)
        benchmark = np.min(ranks_arr[:r_star])
        for i in range(r_star, N):
            if ranks_arr[i] < benchmark:
                ax.scatter(i + 1, ranks_arr[i], color=COLOR_RED if title == "失败选择" else COLOR_BLUE,
                          s=30, zorder=10, marker="*")
                break
        # Mark global best
        best_pos = np.argmin(ranks_arr) + 1
        ax.scatter(best_pos, 1, color=COLOR_GREEN, s=40, zorder=10, marker="D")

        ax.set_ylabel("排名（1=最优）")
        ax.set_xlabel("候选人序号 $n$")
        ax.set_title(title)
        ax.set_xlim(0, N + 1)
        ax.set_ylim(0, N + 1)
        ax.invert_yaxis()

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor=COLOR_GREEN, markersize=6, label="全局最优"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=COLOR_BLUE, markersize=8, label="策略停止点"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle(f"秘书问题排名路径（$N={N}, r^*={r_star}$）", y=1.04, fontsize=11)
    save_figure(fig, "ch05_paths.pdf")


if __name__ == "__main__":
    set_style()
    print("绘制图 5.1: 成功率 vs r/N...")
    fig5_1_success()
    print("绘制图 5.2: Snell 包络...")
    fig5_2_snell()
    print("绘制图 5.3: 排名路径可视化...")
    fig5_3_paths()
    print("第五章三张图绘制完成。")
