"""第四章绘图：赌徒破产 —— OST 条件失效"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.visualization import (
    set_style, new_figure, save_figure, emphasize_log_grid,
    COLOR_BLUE, COLOR_RED, COLOR_GRAY,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'output' / 'data'


def fig4_1_convergence():
    """图 4.1：双边 vs 单边收敛对比。"""
    set_style()
    df = pd.read_csv(DATA_DIR / 'exp4_convergence.csv')

    fig, ax = new_figure()
    for stype, color, label, y_theory in [
        ('two_sided', COLOR_BLUE, '双边停时（OST 成立）', 0),
        ('one_sided', COLOR_RED, '单边停时（OST 失效）', 10.0),
    ]:
        sub = df[df['stop_type'] == stype]
        ax.errorbar(sub['n_paths'], sub['mean'], yerr=1.96 * sub['se'],
                    fmt='o-', color=color, capsize=2, markersize=3.5, lw=0.8,
                    label=label)
        ax.axhline(y_theory, color=color, lw=0.6, ls='--', alpha=0.5)

    ax.set_xscale('log')
    emphasize_log_grid(ax)
    ax.set_xlabel('Monte Carlo 路径数 $M$')
    ax.set_ylabel('$\\mathbb{E}[S_\\tau]$ 估计值')
    ax.set_title('OST 收敛对比：双边成立 vs 单边失效')
    ax.legend(loc='best', fontsize=8)
    save_figure(fig, 'ch04_convergence.pdf')


def fig4_2_truncation():
    """图 4.2：截断偏误。"""
    set_style()
    df = pd.read_csv(DATA_DIR / 'exp4_truncation.csv')

    fig, ax = new_figure()
    ax.errorbar(df['N'], df['mean'], yerr=1.96 * df['se'],
                fmt='o', color=COLOR_BLUE, capsize=1.5, markersize=2.5,
                lw=0, alpha=0.75, label='模拟估计（95% CI）')
    ax.axhline(0.0, color=COLOR_GRAY, lw=0.8, ls=':',
               label='每个固定 $N$: $\\mathbb{E}[S_{\\tau\\wedge N}]=0$')
    ax.axhline(10.0, color=COLOR_RED, lw=0.8, ls='--',
               label='几乎处处极限 $S_\\tau=b=10$')

    ax.set_xscale('log')
    emphasize_log_grid(ax)
    ax.set_xlabel('截断值 $N$')
    ax.set_ylabel('$S_{\\tau \\wedge N}$ 的样本均值')
    ax.set_title('截断序列与不可交换极限')
    ax.legend(loc='lower left', fontsize=7)
    save_figure(fig, 'ch04_truncation.pdf')


def fig4_3_tail():
    """图 4.3：停时尾部对比（双对数）。"""
    set_style()
    data = np.load(DATA_DIR / 'exp4_tail.npz')

    fig, ax = new_figure()
    for t, surv, color, label in [
        (data['t_two'], data['surv_two'], COLOR_BLUE, '双边障碍'),
        (data['t_one'], data['surv_one'], COLOR_RED, '单边障碍'),
    ]:
        mask = surv > 1e-4
        ax.loglog(t[mask], surv[mask], color=color, lw=0.8, label=label)

    # 参考线：与单边样本在 anchor_t 处对齐的 t^{-1/2} 重尾。
    anchor_t = min(500, len(data['surv_one']) - 1)
    anchor_surv = max(data['surv_one'][anchor_t], 1e-12)
    t_ref = np.logspace(np.log10(50), np.log10(data['t_one'].max()), 100)
    ref = anchor_surv * np.sqrt(anchor_t / t_ref)
    ax.loglog(t_ref, ref, '--', color=COLOR_GRAY,
              lw=0.8, alpha=0.75, label='$C t^{-1/2}$')

    ax.set_xlabel('$t$')
    ax.set_ylabel('$P(\\tau > t)$')
    ax.set_title('停时尾部对比 (双对数)')
    emphasize_log_grid(ax)
    ax.legend(loc='lower left', fontsize=8)
    save_figure(fig, 'ch04_tail.pdf')


if __name__ == '__main__':
    set_style()
    print("绘制图 4.1: 收敛对比 ...")
    fig4_1_convergence()
    print("绘制图 4.2: 截断偏误 ...")
    fig4_2_truncation()
    print("绘制图 4.3: 停时尾部 ...")
    fig4_3_tail()
    print("第四章三张图绘制完成。")
