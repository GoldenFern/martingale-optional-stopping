"""Shared plotting helpers."""

import warnings
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", message=".*does not have a glyph.*")
warnings.filterwarnings("ignore", message=".*substituting with a dummy symbol.*")

FIG_W, FIG_H = 6.3, 3.55

COLOR_BLUE = "#3b8ad4"
COLOR_RED = "#B2182B"
COLOR_GREEN = "#1B7837"
COLOR_GRAY = "#737373"
COLOR_ORANGE = "#D95F02"
COLOR_PURPLE = "#756BB1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURE_DIRS = (
    PROJECT_ROOT / "output" / "figures",
    PROJECT_ROOT / "paper" / "figures",
)


def set_style() -> None:
    """Apply the pre-beautification paper plotting style."""
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "Noto Sans CJK SC",
                "Source Han Sans SC",
                "SimHei",
                "Arial",
                "DejaVu Sans",
            ],
            "mathtext.fontset": "dejavusans",
            "font.size": 9.6,
            "axes.titlesize": 10.2,
            "axes.labelsize": 9.8,
            "legend.fontsize": 8.8,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.format": "pdf",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.which": "major",
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.75,
            "legend.frameon": False,
            "lines.linewidth": 1.15,
            "axes.titlepad": 6,
        }
    )


def ensure_output_dir() -> None:
    for path in FIGURE_DIRS:
        path.mkdir(parents=True, exist_ok=True)


def new_figure() -> tuple[plt.Figure, plt.Axes]:
    """Create a single-panel figure."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)
    return fig, ax


def new_figure_dual() -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Create a two-row figure."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(FIG_W, FIG_H * 1.28), constrained_layout=True
    )
    return fig, (ax1, ax2)


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Save PDF outputs to both output and paper figure directories."""
    ensure_output_dir()
    for directory in FIGURE_DIRS:
        fig.savefig(directory / filename, format="pdf", bbox_inches="tight")
    plt.close(fig)


def emphasize_log_grid(ax: plt.Axes) -> None:
    """Strengthen grids for log-scale plots."""
    ax.grid(True, which="major", linewidth=0.5, alpha=0.75)
    ax.grid(True, which="minor", linewidth=0.3, alpha=0.35)


def plot_box_series(
    ax: plt.Axes,
    x_positions: np.ndarray,
    samples: list[np.ndarray],
    width: float | list[float] = 0.06,
    facecolor: str = COLOR_BLUE,
    edgecolor: str = COLOR_BLUE,
    median_color: str = COLOR_RED,
    label: str = "",
):
    """Draw a boxplot series."""
    if len(samples) == 0:
        return None
    box = ax.boxplot(
        samples,
        positions=x_positions,
        widths=width,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": median_color, "linewidth": 1.2},
        whiskerprops={"color": edgecolor, "linewidth": 0.9},
        capprops={"color": edgecolor, "linewidth": 0.9},
        boxprops={"edgecolor": edgecolor, "linewidth": 0.9},
    )
    for patch in box["boxes"]:
        patch.set_facecolor(facecolor)
        patch.set_alpha(0.28)
    if label:
        ax.plot([], [], color=edgecolor, linewidth=6, alpha=0.28, label=label)
    return box


def plot_paths(
    ax: plt.Axes,
    paths: np.ndarray,
    stopping_times: np.ndarray,
    reached_stop: np.ndarray,
    n_display: int = 20,
    colors: tuple = (COLOR_BLUE, COLOR_RED),
    labels: tuple = ("固定", "灭绝"),
    alpha: float = 0.5,
    lw: float = 0.4,
) -> None:
    """Draw selected sample paths split by outcome."""
    reached_indices = np.where(reached_stop)[0]
    unreached_indices = np.where(~reached_stop)[0]
    count_per_group = max(n_display // 2, 1)
    selected_reached = (
        np.random.choice(reached_indices, min(count_per_group, len(reached_indices)), replace=False)
        if len(reached_indices) > 0 else []
    )
    selected_unreached = (
        np.random.choice(unreached_indices, min(count_per_group, len(unreached_indices)), replace=False)
        if len(unreached_indices) > 0 else []
    )

    for path_indices, color, label in [
        (selected_reached, colors[0], labels[0]),
        (selected_unreached, colors[1], labels[1]),
    ]:
        for path_idx in path_indices:
            stop_time = stopping_times[path_idx]
            ax.plot(paths[path_idx, : stop_time + 1], color=color, alpha=alpha, lw=lw)
            ax.scatter(stop_time, paths[path_idx, stop_time], color=color, s=6, zorder=5, alpha=0.8)

    handles = [
        plt.Line2D([0], [0], color=colors[0], lw=1, label=labels[0]),
        plt.Line2D([0], [0], color=colors[1], lw=1, label=labels[1]),
    ]
    ax.legend(handles=handles, loc="best", frameon=False)


def plot_with_ci(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    se: np.ndarray,
    label: str = "",
    color: str = COLOR_BLUE,
    marker: str = "o",
    theory_y: Optional[np.ndarray] = None,
    theory_label: str = "理论值",
    theory_color: str = COLOR_RED,
    theory_style: str = "--",
    theory_lw: float = 1.0,
) -> None:
    """Draw point estimates with 95% CI."""
    ax.errorbar(
        x,
        y,
        yerr=1.96 * se,
        fmt=marker,
        color=color,
        capsize=2,
        markersize=3,
        lw=0.8,
        label=label,
    )
    if theory_y is not None:
        ax.plot(x, theory_y, theory_style, color=theory_color, lw=theory_lw, label=theory_label)


def plot_distribution(
    ax: plt.Axes,
    data: np.ndarray,
    bins: int = 50,
    label: str = "",
    color: str = COLOR_BLUE,
    fit_dist: Optional[object] = None,
) -> None:
    """Draw a histogram with an optional fitted density."""
    ax.hist(
        data,
        bins=bins,
        density=True,
        alpha=0.5,
        color=color,
        edgecolor="white",
        linewidth=0.3,
        label=label,
    )
    if fit_dist is not None:
        x_values = np.linspace(data.min(), data.max(), 200)
        ax.plot(x_values, fit_dist.pdf(x_values), color=COLOR_RED, lw=1.2, label="理论密度")


def plot_loglog_tail(
    ax: plt.Axes, time_values: np.ndarray, survival_probs: np.ndarray, label: str = "", color: str = COLOR_BLUE
) -> None:
    """Draw a survival tail on log-log axes."""
    positive_mask = survival_probs > 0
    ax.loglog(time_values[positive_mask], survival_probs[positive_mask], lw=0.8, color=color, label=label)
