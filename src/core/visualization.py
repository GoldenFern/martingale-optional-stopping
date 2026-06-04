"""统一绘图工具。"""
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')  # 非交互后端，适合批量出图
import matplotlib.pyplot as plt
import numpy as np

# 抑制字体相关警告（CJK 字体在 mathtext 中的已知限制）
warnings.filterwarnings('ignore', message='.*does not have a glyph.*')
warnings.filterwarnings('ignore', message='.*substituting with a dummy symbol.*')

# 统一的 16:9 图片尺寸。论文为双栏排版，避免过宽过扁。
FIG_W, FIG_H = 6.3, 3.55  # inches, 16:9

# 论文用统一配色
COLOR_BLUE = '#2166AC'
COLOR_RED = '#B2182B'
COLOR_GREEN = '#1B7837'
COLOR_GRAY = '#737373'
COLOR_ORANGE = '#D95F02'
COLOR_PURPLE = '#756BB1'

# 项目根目录（用于 output/paper 路径）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_FIGURE_DIRS = (
    _PROJECT_ROOT / 'output' / 'figures',
    _PROJECT_ROOT / 'paper' / 'figures',
)


def set_style():
    """设置全局 matplotlib 样式。"""
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': [
            'Microsoft YaHei', 'Noto Sans CJK SC', 'Source Han Sans SC',
            'SimHei', 'Arial', 'DejaVu Sans',
        ],
        'mathtext.fontset': 'dejavusans',
        'font.size': 8.8,
        'axes.titlesize': 9.4,
        'axes.labelsize': 9,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.dpi': 180,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.format': 'pdf',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.grid.which': 'major',
        'grid.color': '#D9D9D9',
        'grid.linewidth': 0.45,
        'grid.alpha': 0.75,
        'legend.frameon': False,
        'lines.linewidth': 1.15,
        'axes.titlepad': 6,
    })


def ensure_output_dir():
    for path in _FIGURE_DIRS:
        path.mkdir(parents=True, exist_ok=True)


def new_figure() -> tuple[plt.Figure, plt.Axes]:
    """创建单个 16:9 子图的 figure。"""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)
    return fig, ax


def new_figure_dual() -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """创建双轴（上下）16:9 figure。"""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(FIG_W, FIG_H * 1.28), constrained_layout=True
    )
    return fig, (ax1, ax2)


def save_figure(fig: plt.Figure, filename: str):
    """保存图片到 output/figures/ 与 paper/figures/。"""
    ensure_output_dir()
    for directory in _FIGURE_DIRS:
        fig.savefig(directory / filename, format='pdf', bbox_inches='tight')
    plt.close(fig)


def emphasize_log_grid(ax: plt.Axes):
    """为半对数或双对数图添加轻量主/次网格。"""
    ax.grid(True, which='major', linewidth=0.5, alpha=0.75)
    ax.grid(True, which='minor', linewidth=0.3, alpha=0.35)


def plot_paths(ax: plt.Axes, paths: np.ndarray, stopping_times: np.ndarray,
               reached_stop: np.ndarray, n_display: int = 20,
               colors: tuple = (COLOR_BLUE, COLOR_RED),
               labels: tuple = ('固定', '灭绝'),
               alpha: float = 0.5, lw: float = 0.4):
    """绘制多条过程路径。

    将 paths 按 reached_stop 的 True/False 分成两组，分别绘制。
    stopping_times 处用圆点标记。
    """
    trues = np.where(reached_stop)[0]
    falses = np.where(~reached_stop)[0]

    # 各取 n_display//2 条
    n_each = max(n_display // 2, 1)
    idx_true = np.random.choice(trues, min(n_each, len(trues)), replace=False) \
        if len(trues) > 0 else []
    idx_false = np.random.choice(falses, min(n_each, len(falses)), replace=False) \
        if len(falses) > 0 else []

    for idx, color, label in [(idx_true, colors[0], labels[0]),
                                (idx_false, colors[1], labels[1])]:
        for i in idx:
            tau = stopping_times[i]
            ax.plot(paths[i, :tau + 1], color=color, alpha=alpha, lw=lw)
            ax.scatter(tau, paths[i, tau], color=color, s=6, zorder=5, alpha=0.8)

    # 去重 legend
    handles = [
        plt.Line2D([0], [0], color=colors[0], lw=1, label=labels[0]),
        plt.Line2D([0], [0], color=colors[1], lw=1, label=labels[1]),
    ]
    ax.legend(handles=handles, loc='best', frameon=False)


def plot_with_ci(ax: plt.Axes, x: np.ndarray, y: np.ndarray, se: np.ndarray,
                 label: str = '', color: str = COLOR_BLUE, marker: str = 'o',
                 theory_y: Optional[np.ndarray] = None,
                 theory_label: str = '理论值', theory_color: str = COLOR_RED,
                 theory_style: str = '--', theory_lw: float = 1.0):
    """带 95% 置信区间的散点连线 + 可选理论曲线。"""
    ax.errorbar(x, y, yerr=1.96 * se, fmt=marker, color=color,
                capsize=2, markersize=3, lw=0.8, label=label)
    if theory_y is not None:
        ax.plot(x, theory_y, theory_style, color=theory_color,
                lw=theory_lw, label=theory_label)


def plot_distribution(ax: plt.Axes, data: np.ndarray, bins: int = 50,
                      label: str = '', color: str = COLOR_BLUE,
                      fit_dist: Optional[object] = None):
    """直方图 + 可选拟合密度曲线。"""
    ax.hist(data, bins=bins, density=True, alpha=0.5, color=color,
            edgecolor='white', linewidth=0.3, label=label)
    if fit_dist is not None:
        xs = np.linspace(data.min(), data.max(), 200)
        ax.plot(xs, fit_dist.pdf(xs), color=COLOR_RED, lw=1.2, label='理论密度')


def plot_loglog_tail(ax: plt.Axes, t: np.ndarray, survival: np.ndarray,
                     label: str = '', color: str = COLOR_BLUE):
    """双对数停时尾部图。"""
    # 去掉 survival=0 的点（log(0) 无效）
    mask = survival > 0
    ax.loglog(t[mask], survival[mask], lw=0.8, color=color, label=label)
