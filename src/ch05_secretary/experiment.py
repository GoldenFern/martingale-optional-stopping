"""第五章实验：秘书问题与 Snell 包络"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))

import numpy as np
import pandas as pd
from core.batching import split_batch_sizes
from ch05_secretary.secretary_model import (
    simulate_secretary_batch, snell_envelope,
    theoretical_success_prob,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'output' / 'data'


def _batch_key(N: int, point_idx: int) -> str:
    return f'N{N}_idx{point_idx:03d}'


def run_success_experiment(N_values=(20, 50, 100, 200),
                           n_r_points=50, n_trials=5000, seed=42, n_batches=10):
    """实验1: 成功率 vs r/N，保存批次成功率。"""
    np.random.seed(seed)
    rows = []
    batches = {}
    batch_sizes = split_batch_sizes(n_trials, n_batches)
    total_trials = sum(batch_sizes)
    for N in N_values:
        for point_idx, r_frac in enumerate(np.linspace(0.02, 0.98, n_r_points)):
            r = max(1, int(r_frac * N))
            batch_vals = np.empty(n_batches)
            total_successes = 0
            for b, batch_size in enumerate(batch_sizes):
                success_b = simulate_secretary_batch(N, r, batch_size)
                batch_vals[b] = success_b
                total_successes += success_b * batch_size
            success = total_successes / total_trials
            se = np.sqrt(success * (1 - success) / total_trials)
            theory = theoretical_success_prob(r, N)
            batch_key = _batch_key(N, point_idx)
            batches[batch_key] = batch_vals
            rows.append({
                'N': N, 'r': r, 'r_frac': r / N,
                'success_mc': success, 'se': se,
                'theory': theory,
                'batch_key': batch_key,
            })
    df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / 'exp5_success.csv', index=False)
    np.savez(DATA_DIR / 'exp5_success_batches.npz', **batches)
    return df


def run_snell_experiment(N=20):
    """实验2: Snell 包络精确计算。"""
    V, g = snell_envelope(N)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(DATA_DIR / 'exp5_snell.npz', V=V, g=g)
    return V, g


def run_control_experiment(N_values=(20, 50, 100, 200),
                           n_r_points=30, n_trials=2000, seed=42, n_batches=10):
    """实验3: 随机选择基线（模拟真实随机排列），保存批次成功率。"""
    np.random.seed(seed)
    rows = []
    batches = {}
    batch_sizes = split_batch_sizes(n_trials, n_batches)
    total_trials = sum(batch_sizes)
    for N in N_values:
        for point_idx, r_frac in enumerate(np.linspace(0.02, 0.98, n_r_points)):
            r = max(1, int(r_frac * N))
            batch_vals = np.empty(n_batches)
            for b, batch_size in enumerate(batch_sizes):
                # 每批次模拟 batch_size 次随机排列，每次随机选一个位置
                successes = 0
                for _ in range(batch_size):
                    best_pos = np.random.randint(0, N)  # 全局最优的随机位置
                    pick_pos = np.random.randint(r, N)   # 在可选范围内随机选
                    if pick_pos == best_pos:
                        successes += 1
                batch_vals[b] = successes / batch_size
            batches[_batch_key(N, point_idx)] = batch_vals
            success = batch_vals.mean()
            se = batch_vals.std(ddof=1) / np.sqrt(n_batches)
            rows.append({
                'N': N, 'r': r, 'r_frac': r / N,
                'success_mc': success, 'se': se,
                'batch_key': _batch_key(N, point_idx),
            })
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / 'exp5_control.csv', index=False)
    np.savez(DATA_DIR / 'exp5_control_batches.npz', **batches)
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

    print("实验3: 随机选择基线 ...")
    df_c = run_control_experiment()
    print(f"  对照组成功率 ≈ {df_c['success_mc'].mean():.4f} (接近 1/N)")
    print("第五章实验完成。")
