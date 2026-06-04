"""Shared figure styling, export, and manifest helpers."""

from __future__ import annotations

import csv
import warnings
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

warnings.filterwarnings("ignore", message=".*does not have a glyph.*")
warnings.filterwarnings("ignore", message=".*substituting with a dummy symbol.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_FIGURE_DIR = PROJECT_ROOT / "output" / "figures"
PAPER_FIGURE_DIR = PROJECT_ROOT / "paper" / "figures"
FIGURE_DIRS = (OUTPUT_FIGURE_DIR, PAPER_FIGURE_DIR)
FIGURE_MANIFEST_PATH = OUTPUT_FIGURE_DIR / "figure_manifest.csv"

FIGURE_SIZES = {
    "default": (5.8, 3.8),
    "paths": (6.2, 4.0),
    "trend": (5.8, 3.8),
    "distribution": (5.8, 3.8),
    "dual": (6.2, 4.8),
    "heatmap": (5.6, 4.4),
}

COLOR_BLUE = "#0F4D92"
COLOR_BLUE_LIGHT = "#3775BA"
COLOR_RED = "#B64342"
COLOR_GREEN = "#2E9E44"
COLOR_GRAY = "#4D4D4D"
COLOR_GRAY_LIGHT = "#CFCECE"
COLOR_ORANGE = "#D95F02"
COLOR_PURPLE = "#9A4D8E"

PREFERRED_SANS_FONTS = [
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Microsoft YaHei",
    "Arial",
    "DejaVu Sans",
]
AVAILABLE_FONTS = {font.name for font in font_manager.fontManager.ttflist}
SELECTED_SANS_FONTS = [font for font in PREFERRED_SANS_FONTS if font in AVAILABLE_FONTS] or ["DejaVu Sans"]

MANIFEST_FIELDS = [
    "figure",
    "chapter",
    "role",
    "figure_type",
    "core_conclusion",
    "data_files",
    "statistics",
]


def _figure_size(kind: str) -> tuple[float, float]:
    return FIGURE_SIZES.get(kind, FIGURE_SIZES["default"])


def set_style() -> None:
    """Apply a restrained publication-style theme for paper figures."""
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": SELECTED_SANS_FONTS,
            "mathtext.fontset": "dejavusans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.4,
            "axes.titlesize": 9.2,
            "axes.labelsize": 9.0,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "figure.dpi": 220,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.85,
            "axes.grid": True,
            "axes.grid.which": "major",
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.35,
            "grid.alpha": 0.55,
            "legend.frameon": False,
            "legend.handlelength": 1.7,
            "legend.handletextpad": 0.45,
            "legend.labelspacing": 0.35,
            "legend.borderaxespad": 0.25,
            "lines.linewidth": 1.15,
            "axes.titlepad": 4,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
        }
    )


def ensure_output_dir() -> None:
    for path in FIGURE_DIRS:
        path.mkdir(parents=True, exist_ok=True)


def new_figure(kind: str = "default") -> tuple[plt.Figure, plt.Axes]:
    """Create a single-panel figure with a size chosen by figure type."""
    fig, ax = plt.subplots(figsize=_figure_size(kind), constrained_layout=True)
    return fig, ax


def new_figure_dual(kind: str = "dual") -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Create a vertically stacked dual-panel figure."""
    width, height = _figure_size(kind)
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(width, height),
        constrained_layout=True,
    )
    return fig, (ax1, ax2)


def save_figure(fig: plt.Figure, filename: str, *, png_dpi: int = 260) -> None:
    """Save a paper-facing PDF plus editable/preview exports."""
    ensure_output_dir()
    requested = Path(filename)
    stem = requested.stem if requested.suffix else requested.name

    pdf_name = f"{stem}.pdf"
    svg_name = f"{stem}.svg"
    png_name = f"{stem}.png"

    for directory in FIGURE_DIRS:
        fig.savefig(directory / pdf_name, format="pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_FIGURE_DIR / svg_name, format="svg", bbox_inches="tight")
    fig.savefig(OUTPUT_FIGURE_DIR / png_name, format="png", dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)


def emphasize_log_grid(ax: plt.Axes) -> None:
    """Add visible but quiet major/minor grids to log-scale axes."""
    ax.grid(True, which="major", linewidth=0.45, alpha=0.65)
    ax.grid(True, which="minor", linewidth=0.25, alpha=0.3)


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
    """Draw a semi-transparent batch boxplot series."""
    if not samples:
        return None
    box = ax.boxplot(
        samples,
        positions=x_positions,
        widths=width,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": median_color, "linewidth": 1.1},
        whiskerprops={"color": edgecolor, "linewidth": 0.85},
        capprops={"color": edgecolor, "linewidth": 0.85},
        boxprops={"edgecolor": edgecolor, "linewidth": 0.85},
    )
    for patch in box["boxes"]:
        patch.set_facecolor(facecolor)
        patch.set_alpha(0.24)
    if label:
        ax.plot([], [], color=edgecolor, linewidth=5, alpha=0.28, label=label)
    return box


def plot_path_group(
    ax: plt.Axes,
    x_series: list[np.ndarray],
    y_series: list[np.ndarray],
    *,
    color: str = COLOR_GRAY,
    alpha: float = 0.35,
    lw: float = 0.65,
    zorder: int = 1,
    step_where: str | None = None,
) -> None:
    """Draw a path collection with consistent background/foreground styling."""
    for x_values, y_values in zip(x_series, y_series):
        if step_where:
            ax.step(
                x_values,
                y_values,
                where=step_where,
                color=color,
                alpha=alpha,
                lw=lw,
                zorder=zorder,
            )
        else:
            ax.plot(
                x_values,
                y_values,
                color=color,
                alpha=alpha,
                lw=lw,
                zorder=zorder,
            )


def _offset_on_scale(value: float, lower: float, upper: float, pad_frac: float, scale: str) -> float:
    if scale == "log" and value > 0 and lower > 0 and upper > 0:
        return value * (upper / lower) ** pad_frac
    return value + (upper - lower) * pad_frac


def add_endpoint_label(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    color: str = COLOR_RED,
    x_pad_frac: float = 0.015,
    y_pad_frac: float = 0.0,
    ha: str = "left",
    va: str = "center",
    fontsize: Optional[float] = None,
) -> None:
    """Place a compact direct label next to a line endpoint."""
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    ax.text(
        _offset_on_scale(x, x_lo, x_hi, x_pad_frac, ax.get_xscale()),
        _offset_on_scale(y, y_lo, y_hi, y_pad_frac, ax.get_yscale()),
        text,
        color=color,
        fontsize=fontsize or plt.rcParams["legend.fontsize"],
        ha=ha,
        va=va,
        clip_on=False,
    )


def add_hline_label(
    ax: plt.Axes,
    y: float,
    text: str,
    *,
    color: str = COLOR_RED,
    linestyle: str = "--",
    linewidth: float = 0.85,
    alpha: float = 0.8,
    x_frac: float = 0.985,
    y_pad_frac: float = 0.012,
) -> None:
    """Draw a horizontal reference line with a compact right-edge label."""
    ax.axhline(y, color=color, lw=linewidth, ls=linestyle, alpha=alpha)
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    ax.text(
        x_lo + (x_hi - x_lo) * x_frac,
        _offset_on_scale(y, y_lo, y_hi, y_pad_frac, ax.get_yscale()),
        text,
        color=color,
        fontsize=plt.rcParams["legend.fontsize"],
        ha="right",
        va="bottom",
        clip_on=False,
    )


def add_vline_label(
    ax: plt.Axes,
    x: float,
    text: str,
    *,
    color: str = COLOR_GRAY,
    linestyle: str = "--",
    linewidth: float = 0.85,
    alpha: float = 0.8,
    y_frac: float = 0.965,
    rotation: float = 90,
) -> None:
    """Draw a vertical reference line with a top-edge label."""
    ax.axvline(x, color=color, lw=linewidth, ls=linestyle, alpha=alpha)
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    ax.text(
        x + (x_hi - x_lo) * 0.008,
        y_lo + (y_hi - y_lo) * y_frac,
        text,
        color=color,
        fontsize=plt.rcParams["legend.fontsize"],
        ha="left",
        va="top",
        rotation=rotation,
        clip_on=False,
    )


def add_axes_note(
    ax: plt.Axes,
    text: str,
    *,
    x: float = 0.98,
    y: float = 0.96,
    ha: str = "right",
    va: str = "top",
) -> None:
    """Place a compact note inside an axes."""
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=plt.rcParams["legend.fontsize"],
        color=COLOR_GRAY,
        ha=ha,
        va=va,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.2},
    )


def add_compact_colorbar(
    fig: plt.Figure,
    mappable,
    ax: plt.Axes,
    *,
    label: str,
    shrink: float = 0.82,
    pad: float = 0.02,
    fraction: float = 0.05,
):
    """Attach a compact colorbar tuned for paper figures."""
    cbar = fig.colorbar(mappable, ax=ax, shrink=shrink, pad=pad, fraction=fraction)
    cbar.set_label(label)
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(length=2.5, width=0.6)
    return cbar


def add_probability_contour(
    ax: plt.Axes,
    t_grid: np.ndarray,
    s_grid: np.ndarray,
    values: np.ndarray,
    *,
    level: float = 0.5,
    color: str = "white",
    linewidth: float = 0.9,
) -> None:
    """Draw a compact contour for a probability threshold."""
    ax.contour(t_grid, s_grid, values.T, levels=[level], colors=color, linewidths=linewidth)


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
):
    """Draw a point estimate with 95% confidence intervals."""
    ax.errorbar(x, y, yerr=1.96 * se, fmt=marker, color=color, capsize=2, markersize=3, lw=0.8, label=label)
    if theory_y is not None:
        ax.plot(x, theory_y, theory_style, color=theory_color, lw=theory_lw, label=theory_label)


def plot_distribution(
    ax: plt.Axes,
    data: np.ndarray,
    bins: int = 50,
    label: str = "",
    color: str = COLOR_BLUE,
    fit_dist: Optional[object] = None,
):
    """Draw a histogram with an optional fitted density curve."""
    ax.hist(
        data,
        bins=bins,
        density=True,
        alpha=0.42,
        color=color,
        edgecolor="white",
        linewidth=0.35,
        label=label,
    )
    if fit_dist is not None:
        xs = np.linspace(data.min(), data.max(), 200)
        ax.plot(xs, fit_dist.pdf(xs), color=COLOR_RED, lw=1.2, label="理论密度")


def plot_loglog_tail(
    ax: plt.Axes,
    t: np.ndarray,
    survival: np.ndarray,
    label: str = "",
    color: str = COLOR_BLUE,
    *,
    linestyle: str = "-",
    lw: float = 0.9,
):
    """Draw a survival tail on log-log axes, skipping zero values."""
    mask = survival > 0
    ax.loglog(t[mask], survival[mask], linestyle=linestyle, lw=lw, color=color, label=label)


def write_figure_manifest(rows: list[dict[str, object]]) -> None:
    """Upsert figure metadata rows into the shared manifest."""
    ensure_output_dir()
    existing: dict[str, dict[str, str]] = {}
    if FIGURE_MANIFEST_PATH.exists():
        with FIGURE_MANIFEST_PATH.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                existing[row["figure"]] = {field: row.get(field, "") for field in MANIFEST_FIELDS}

    for row in rows:
        normalized = {field: "" for field in MANIFEST_FIELDS}
        for field in MANIFEST_FIELDS:
            value = row.get(field, "")
            if field == "data_files" and isinstance(value, (list, tuple)):
                normalized[field] = "; ".join(str(item) for item in value)
            else:
                normalized[field] = str(value)
        existing[normalized["figure"]] = normalized

    ordered_rows = [existing[key] for key in sorted(existing)]
    with FIGURE_MANIFEST_PATH.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(ordered_rows)
