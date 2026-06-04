"""第二章绘图：Moran 模型"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))

import numpy as np
from scipy.stats import expon
import matplotlib.pyplot as plt
from ch02_moran.moran_model import MoranProcess

from core.visualization import (
    set_style, new_figure, save_figure,
    COLOR_BLUE, COLOR_RED,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'output' / 'data'


def fig2_1_paths(N=100, x0=50, n_display=10, seed=42):
    """图 2.1：多条基因频率轨迹叠加。"""
    set_style()
    np.random.seed(seed)
    model = MoranProcess(N)
    paths_all, outcomes = [], []
    for _ in range(150):
        model.reset(x0)
        path = model.simulate_path()
        paths_all.append(path / N)
        outcomes.append(path[-1] == N)
    fix_idx = [i for i, o in enumerate(outcomes) if o]
    ext_idx = [i for i, o in enumerate(outcomes) if not o]
    n_each = max(n_display // 2, 1)

    fig, ax = new_figure()
    for idx_list, color, label in [
        (fix_idx, COLOR_BLUE, '固定'),
        (ext_idx, COLOR_RED, '灭绝'),
    ]:
        chosen = np.random.choice(idx_list, min(n_each, len(idx_list)), replace=False)
        for i in chosen:
            freq = paths_all[i]
            tau = len(freq) - 1
            ax.plot(freq, alpha=0.5, lw=0.4, color=color)
            ax.scatter(tau, freq[-1], color=color, s=6, alpha=0.7, zorder=5)

    ax.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.6)
    ax.axhline(1, color='gray', lw=0.5, ls='--', alpha=0.6)
    ax.set_xlim(0, None)
    ax.set_xlabel('代数')
    ax.set_ylabel('A 等位基因频率')
    ax.set_title(f'Moran 模型样本路径（$N={N}$）')
    handles = [
        plt.Line2D([0], [0], color=COLOR_BLUE, lw=1, label='固定'),
        plt.Line2D([0], [0], color=COLOR_RED, lw=1, label='灭绝'),
    ]
    ax.legend(handles=handles, loc='best')
    save_figure(fig, 'ch02_paths.pdf')


def fig2_2_fixation(n_paths=5000):
    """图 2.2：固定概率 vs 初始频率，仅 N=100。"""
    set_style()
    import pandas as pd
    df = pd.read_csv(DATA_DIR / 'exp2_fixation.csv')
    # 只保留 N=100
    sub = df[df['N'] == 100]

    fig, ax = new_figure()
    p_hat = sub['p_fixation_mc'].to_numpy()
    p_se = np.sqrt(p_hat * (1 - p_hat) / n_paths)
    ax.errorbar(sub['initial_freq'], sub['p_fixation_mc'],
                yerr=1.96 * p_se,
                fmt='o', color=COLOR_BLUE, capsize=2, markersize=4,
                lw=0, label='Monte Carlo（95% CI）')

    xs = np.linspace(0, 1, 50)
    ax.plot(xs, xs, '--', color=COLOR_RED, lw=1.15,
            label='理论值 $y=x_0/N$')

    ax.set_xlabel('初始频率 $x_0/N$')
    ax.set_ylabel('固定概率')
    ax.set_title('固定概率与初始频率（$N=100$）')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    save_figure(fig, 'ch02_fixation.pdf')


def fig2_3_tau_dist():
    """图 2.3：停时分布直方图，N=100。"""
    set_style()
    tau = np.loadtxt(DATA_DIR / 'exp2_tau_dist.csv', delimiter=',')

    fig, ax = new_figure()
    n_bins = 80
    ax.hist(tau, bins=n_bins, density=True, alpha=0.55, color=COLOR_BLUE,
            edgecolor='white', linewidth=0.3, label='模拟')

    # 指数分布拟合
    loc, scale = expon.fit(tau)
    xs = np.linspace(0, tau.max(), 300)
    ax.plot(xs, expon.pdf(xs, loc, scale), color=COLOR_RED, lw=1.2,
            label='指数分布拟合')

    ax.set_xlim(0, None)
    ax.set_xlabel('停时 $\\tau$（代）')
    ax.set_ylabel('概率密度')
    ax.set_title('停时分布（$N=100, x_0=50$）')
    ax.legend(loc='upper right', fontsize=8)
    save_figure(fig, 'ch02_tau_dist.pdf')


if __name__ == '__main__':
    set_style()
    print("绘制图 2.1: 基因频率轨迹 ...")
    fig2_1_paths()
    print("绘制图 2.2: 固定概率 vs 初始频率 ...")
    fig2_2_fixation()
    print("绘制图 2.3: 停时分布 ...")
    fig2_3_tau_dist()
    print("第二章三张图绘制完成。")
