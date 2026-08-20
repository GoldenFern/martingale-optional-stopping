"""Shared plotting helpers."""

import warnings
from pathlib import Path

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
    output_name = Path(filename)
    if output_name.name != str(filename) or output_name.suffix.lower() != ".pdf":
        raise ValueError("filename must be a PDF basename")
    ensure_output_dir()
    for directory in FIGURE_DIRS:
        fig.savefig(directory / output_name.name, format="pdf", bbox_inches="tight")
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


