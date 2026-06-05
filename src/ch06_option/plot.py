"""第六章绘图：美式看跌期权 LSM。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.visualization import COLOR_BLUE, COLOR_GRAY, COLOR_ORANGE, COLOR_RED, new_figure, plot_box_series, save_figure, set_style

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "output" / "data"


def _widths_from_x(x_values: np.ndarray, ratio: float = 0.35) -> list[float]:
    x_values = np.asarray(x_values, dtype=float)
    widths = []
    for i, value in enumerate(x_values):
        if len(x_values) == 1:
            widths.append(0.15)
        elif i == 0:
            widths.append((x_values[i + 1] - value) * ratio)
        elif i == len(x_values) - 1:
            widths.append((value - x_values[i - 1]) * ratio)
        else:
            widths.append(min(value - x_values[i - 1], x_values[i + 1] - value) * ratio)
    return widths


def fig6_1_paths() -> None:
    """图 6.1：15 条股价路径与 LSM 行权位置。"""
    set_style()
    data = np.load(DATA_DIR / "exp6_paths.npz", allow_pickle=True)
    stock_paths = data["S"]
    exercise_times = data["tau"]
    time_step = 1.0 / 50

    fig, ax = new_figure()
    early_exercise_count, maturity_exercise_count = 0, 0
    for path_idx in range(len(stock_paths)):
        ax.plot(np.arange(len(stock_paths[path_idx])) * time_step, stock_paths[path_idx],
                lw=0.4, color=COLOR_GRAY, alpha=0.5)
        exercise_time_i = exercise_times[path_idx] * time_step
        if exercise_times[path_idx] < len(stock_paths[path_idx]) - 1:
            ax.scatter(exercise_time_i, stock_paths[path_idx, exercise_times[path_idx]],
                      color=COLOR_RED, s=16, zorder=5, alpha=0.9)
            early_exercise_count += 1
        elif stock_paths[path_idx, exercise_times[path_idx]] < 40:
            ax.scatter(exercise_time_i, stock_paths[path_idx, exercise_times[path_idx]],
                      color=COLOR_ORANGE, marker="^", s=18, zorder=5, alpha=0.9)
            maturity_exercise_count += 1

    ax.axhline(40, color=COLOR_BLUE, lw=0.7, ls="--", alpha=0.65, label="执行价 $K$")
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=COLOR_BLUE, lw=0.8, ls="--", label="执行价 $K$"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_RED,
               markeredgecolor=COLOR_RED, markersize=5, label=f"提前行权 {early_exercise_count} 条"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=COLOR_ORANGE,
               markeredgecolor=COLOR_ORANGE, markersize=5, label=f"到期行权 {maturity_exercise_count} 条"),
    ]
    ax.set_xlabel("时间 $t$")
    ax.set_ylabel("股价 $S_t$")
    ax.set_title("美式看跌期权路径与行权时刻")
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    save_figure(fig, "ch06_paths.pdf")


def fig6_2_boundary() -> None:
    """图 6.2：行权边界热力图。"""
    set_style()
    data = np.load(DATA_DIR / "exp6_boundary.npz")
    time_grid, price_grid, exercise_prob = data["t_grid"], data["s_grid"], data["prob"]

    fig, ax = new_figure()
    contour_plot = ax.contourf(time_grid, price_grid, exercise_prob.T,
                               levels=np.linspace(0, 1, 21), cmap="viridis", vmin=0, vmax=1)
    ax.contour(time_grid, price_grid, exercise_prob.T, levels=[0.5], colors="white", linewidths=0.8)
    fig.colorbar(contour_plot, ax=ax, label="行权概率", shrink=0.82)
    ax.set_xlabel("时间 $t$")
    ax.set_ylabel("股价 $S$")
    ax.set_title("行权概率热力图")
    save_figure(fig, "ch06_boundary.pdf")


def fig6_3_price() -> None:
    """图 6.3：三线价格对比。"""
    set_style()
    results_df = pd.read_csv(DATA_DIR / "exp6_price.csv")
    batch_data = np.load(DATA_DIR / "exp6_price_batches.npz")

    fig, ax = new_figure()
    initial_prices = results_df["S0"].to_numpy()
    batch_samples = [batch_data[f"S0_{s0:.1f}"] for s0 in initial_prices]
    plot_box_series(
        ax,
        initial_prices,
        batch_samples,
        width=_widths_from_x(initial_prices, ratio=0.3),
        facecolor=COLOR_BLUE,
        edgecolor=COLOR_BLUE,
        median_color=COLOR_RED,
        label="美式期权（LSM 批次分布）",
    )
    ax.plot(results_df["S0"], results_df["european"], "-", color=COLOR_RED, lw=1.0,
            label="欧式期权（Black-Scholes）")
    ax.plot(results_df["S0"], results_df["payoff"], "--", color=COLOR_GRAY, lw=0.8, alpha=0.7,
            label="立即行权收益 $\\max(K-S_0,0)$")

    ax.set_xlabel("初始股价 $S_0$")
    ax.set_ylabel("期权价格")
    ax.set_title("美式与欧式看跌期权价格")
    ax.legend(loc="upper right", fontsize=8)
    save_figure(fig, "ch06_price.pdf")


if __name__ == "__main__":
    set_style()
    print("绘制图 6.1: 路径 + 行权位置...")
    fig6_1_paths()
    print("绘制图 6.2: 行权边界热力图...")
    fig6_2_boundary()
    print("绘制图 6.3: 三线价格对比...")
    fig6_3_price()
    print("第六章三张图绘制完成。")
