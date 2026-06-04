"""第五章实验：秘书问题与 Snell 包络"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))

import numpy as np
import pandas as pd
from secretary_model import (
    simulate_secretary_batch, snell_envelope,
    theoretical_success_prob, simulate_noise_control,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'output' / 'data'


def run_success_experiment(N_values=(20, 50, 100, 200),
                           n_r_points=50, n_trials=5000, seed=42):
    """实验1: 成功率 vs r/N。"""
    np.random.seed(seed)
    rows = []
    for N in N_values:
        for r_frac in np.linspace(0.02, 0.98, n_r_points):
            r = max(1, int(r_frac * N))
            success = simulate_secretary_batch(N, r, n_trials)
            se = np.sqrt(success * (1 - success) / n_trials)
            theory = theoretical_success_prob(r, N)
            rows.append({
                'N': N, 'r': r, 'r_frac': r / N,
                'success_mc': success, 'se': se,
                'theory': theory,
            })
    df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / 'exp5_success.csv', index=False)
    return df


def run_snell_experiment(N=20):
    """实验2: Snell 包络精确计算。"""
    V, g = snell_envelope(N)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(DATA_DIR / 'exp5_snell.npz', V=V, g=g)
    return V, g


def run_control_experiment(N_values=(20, 50, 100, 200),
                           n_r_points=30, n_trials=2000, seed=42):
    """实验3: 鞅对照组。"""
    np.random.seed(seed)
    rows = []
    for N in N_values:
        for r_frac in np.linspace(0.02, 0.98, n_r_points):
            r = max(1, int(r_frac * N))
            success = simulate_noise_control(N, r, n_trials)
            se = np.sqrt(success * (1 - success) / n_trials)
            rows.append({
                'N': N, 'r': r, 'r_frac': r / N,
                'success_mc': success, 'se': se,
            })
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / 'exp5_control.csv', index=False)
    return df


if __name__ == '__main__':
    print("=== 第五章实验：秘书问题 ===")
    print("实验1: 成功率 vs r/N ...")
    df = run_success_experiment()
    for N in [20, 50, 100, 200]:
        sub = df[df['N'] == N]
        best = sub.loc[sub['success_mc'].idxmax()]
        print(f"  N={N:3d}: best r*={best['r']}, max P={best['success_mc']:.4f}, "
              f"理论最优 ≈ {1/np.e:.4f}")

    print("实验2: Snell 包络 (N=20) ...")
    V, g = run_snell_experiment()
    print(f"  V_0={V[0]:.4f}, g_0={g[0]:.4f}")

    print("实验3: 鞅对照组 ...")
    df_c = run_control_experiment()
    print(f"  对照组成功率 ≈ {df_c['success_mc'].mean():.4f} (接近 1/N)")
    print("第五章实验完成。")
