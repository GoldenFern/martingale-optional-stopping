"""第五章绘图：秘书问题与 Snell 包络"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.visualization import (
    set_style, new_figure, save_figure,
    COLOR_BLUE, COLOR_RED, COLOR_GREEN, COLOR_ORANGE, COLOR_GRAY,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'output' / 'data'


def fig5_1_success():
    """图 5.1: 成功率 vs 观察比例 r/N。"""
    set_style()
    df = pd.read_csv(DATA_DIR / 'exp5_success.csv')

    fig, ax = new_figure()
    colors = [COLOR_BLUE, COLOR_RED, COLOR_ORANGE, COLOR_GREEN]
    for color, (N, grp) in zip(colors, df.groupby('N')):
        ax.errorbar(grp['r_frac'], grp['success_mc'],
                     yerr=1.96 * grp['se'],
                     fmt='o', color=color, capsize=1, markersize=2,
                     lw=0, alpha=0.45, label=f'$N={N}$')
        # 理论光滑曲线
        idx = np.argsort(grp['r_frac'])
        ax.plot(grp['r_frac'].iloc[idx], grp['theory'].iloc[idx],
                '-', color=color, lw=1.0)

    ax.axvline(1 / np.e, color=COLOR_GRAY, lw=0.6, ls='--', alpha=0.6,
               label='$r^*/N = 1/e$')
    ax.axhline(1 / np.e, color=COLOR_GRAY, lw=0.6, ls=':', alpha=0.6)

    ax.set_xlabel('观察比例 $r/N$')
    ax.set_ylabel('选中最优者的概率')
    ax.set_title('秘书问题成功率')
    ax.legend(loc='upper right', fontsize=7)
    save_figure(fig, 'ch05_success.pdf')


def fig5_2_snell():
    """图 5.2: Snell 包络 V_n 与即时 payoff g(n)。"""
    set_style()
    data = np.load(DATA_DIR / 'exp5_snell.npz')
    V, g = data['V'], data['g']
    N = len(V)
    n = np.arange(N)

    fig, ax = new_figure()
    ax.plot(n, V, '-', color=COLOR_BLUE, lw=1.25, label='Snell 包络 $V_n$')
    ax.plot(n, g, '--', color=COLOR_RED, lw=1.0, label='即时收益 $g(n)$')

    # 标注相交区域：V_n == g(n) 时应该停止
    cross = np.where(np.isclose(V, g))[0]
    if len(cross) > 0:
        r_star = cross[0]
        ax.axvspan(0, r_star, color=COLOR_BLUE, alpha=0.06, lw=0)
        ax.axvspan(r_star, n[-1], color=COLOR_RED, alpha=0.04, lw=0)
        ax.axvline(r_star, color=COLOR_GRAY, lw=0.8, ls=':', alpha=0.8)
        ax.annotate(f'$r^*={r_star}$',
                    xy=(r_star, V[r_star]), xytext=(r_star + 1.0, V[r_star] + 0.08),
                    arrowprops={'arrowstyle': '->', 'lw': 0.6, 'color': COLOR_GRAY},
                    fontsize=8, color=COLOR_GRAY)

    ax.set_xlabel('候选人序号 $n$')
    ax.set_ylabel('条件成功概率')
    ax.set_title(f'Snell 包络与停止阈值（$N={N}$）')
    ax.legend(loc='lower left', fontsize=8)
    save_figure(fig, 'ch05_snell.pdf')


def fig5_3_control():
    """图 5.3: 鞅对照组 vs 秘书问题。"""
    set_style()
    df_sec = pd.read_csv(DATA_DIR / 'exp5_success.csv')
    df_ctrl = pd.read_csv(DATA_DIR / 'exp5_control.csv')

    fig, ax = new_figure()

    # 秘书问题 (N=100)
    sub_sec = df_sec[df_sec['N'] == 100]
    ax.errorbar(sub_sec['r_frac'], sub_sec['success_mc'],
                 yerr=1.96 * sub_sec['se'],
                 fmt='o', color=COLOR_BLUE, capsize=1, markersize=2,
                 lw=0, alpha=0.45, label='秘书问题（$N=100$）')
    idx = np.argsort(sub_sec['r_frac'])
    ax.plot(sub_sec['r_frac'].iloc[idx], sub_sec['theory'].iloc[idx],
            '-', color=COLOR_BLUE, lw=1.0)

    # 对照组 (N=100)
    sub_ctrl = df_ctrl[df_ctrl['N'] == 100]
    ax.errorbar(sub_ctrl['r_frac'], sub_ctrl['success_mc'],
                 yerr=1.96 * sub_ctrl['se'],
                 fmt='s', color=COLOR_RED, capsize=1, markersize=2,
                 lw=0, alpha=0.55, label='鞅噪声对照')
    ax.axhline(1.0 / 100, color=COLOR_RED, lw=0.6, ls=':', alpha=0.5)

    ax.set_xlabel('观察比例 $r/N$')
    ax.set_ylabel('成功概率')
    ax.set_title('秘书问题与鞅对照')
    ax.legend(loc='upper right', fontsize=8)
    save_figure(fig, 'ch05_control.pdf')


if __name__ == '__main__':
    set_style()
    print("绘制图 5.1: 成功率 vs r/N...")
    fig5_1_success()
    print("绘制图 5.2: Snell 包络...")
    fig5_2_snell()
    print("绘制图 5.3: 鞅对照组...")
    fig5_3_control()
    print("第五章三张图绘制完成。")
