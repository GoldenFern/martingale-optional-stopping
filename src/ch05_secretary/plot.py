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
    """图 5.1：成功率 vs 观察比例 r/N。"""
    set_style()
    df = pd.read_csv(DATA_DIR / "exp5_success.csv")
    batches = np.load(DATA_DIR / "exp5_success_batches.npz")

    fig, ax = new_figure()
    colors = [COLOR_BLUE, COLOR_RED, COLOR_ORANGE, COLOR_GREEN]
    for color, (n_value, group) in zip(colors, df.groupby("N")):
        group = group.sort_values("r_frac").drop_duplicates(subset="r", keep="first")
        x_all = group["r_frac"].to_numpy()
        keys_all = group["batch_key"].to_numpy()
        keep = _sample_indices(len(x_all), target=13)
        x = x_all[keep]
        samples = [batches[key] for key in keys_all[keep]]
        plot_box_series(
            ax,
            x,
            samples,
            width=_widths_from_x(x, ratio=0.22),
            facecolor=color,
            edgecolor=color,
            median_color=COLOR_GRAY,
            label=f"$N={n_value}$",
        )
        ax.plot(group["r_frac"], group["theory"], "-", color=color, lw=1.0)

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


def fig5_3_control() -> None:
    """图 5.3：随机对照组 vs 秘书问题。"""
    set_style()
    df_sec = pd.read_csv(DATA_DIR / "exp5_success.csv")
    df_ctrl = pd.read_csv(DATA_DIR / "exp5_control.csv")
    batches_sec = np.load(DATA_DIR / "exp5_success_batches.npz")
    batches_ctrl = np.load(DATA_DIR / "exp5_control_batches.npz")

    fig, ax = new_figure()

    sub_sec = df_sec[df_sec["N"] == 100].sort_values("r_frac").drop_duplicates(subset="r", keep="first")
    keep = _sample_indices(len(sub_sec), target=13)
    x_sec = sub_sec["r_frac"].to_numpy()[keep]
    sec_keys = sub_sec["batch_key"].to_numpy()[keep]
    sec_samples = [batches_sec[key] for key in sec_keys]
    plot_box_series(
        ax,
        x_sec,
        sec_samples,
        width=_widths_from_x(x_sec, ratio=0.22),
        facecolor=COLOR_BLUE,
        edgecolor=COLOR_BLUE,
        median_color=COLOR_GRAY,
        label="秘书问题（$N=100$）",
    )
    ax.plot(sub_sec["r_frac"], sub_sec["theory"], "-", color=COLOR_BLUE, lw=1.0)

    sub_ctrl = df_ctrl[df_ctrl["N"] == 100].sort_values("r_frac").drop_duplicates(subset="r", keep="first")
    keep_ctrl = _sample_indices(len(sub_ctrl), target=13)
    x_ctrl = sub_ctrl["r_frac"].to_numpy()[keep_ctrl]
    ctrl_keys = sub_ctrl["batch_key"].to_numpy()[keep_ctrl]
    ctrl_samples = [batches_ctrl[key] for key in ctrl_keys]
    plot_box_series(
        ax,
        x_ctrl,
        ctrl_samples,
        width=_widths_from_x(x_ctrl, ratio=0.22),
        facecolor=COLOR_RED,
        edgecolor=COLOR_RED,
        median_color=COLOR_GRAY,
        label="随机选择基线",
    )
    ax.axhline(1.0 / 100, color=COLOR_RED, lw=0.6, ls=":", alpha=0.5)

    ax.set_xlabel("观察比例 $r/N$")
    ax.set_ylabel("成功概率")
    ax.set_title("秘书问题与随机对照")
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.legend(loc="upper right", fontsize=8)
    save_figure(fig, "ch05_control.pdf")


if __name__ == "__main__":
    set_style()
    print("绘制图 5.1: 成功率 vs r/N...")
    fig5_1_success()
    print("绘制图 5.2: Snell 包络...")
    fig5_2_snell()
    print("绘制图 5.3: 随机对照组...")
    fig5_3_control()
    print("第五章三张图绘制完成。")
