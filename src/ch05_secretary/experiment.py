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


def _batch_key(num_candidates: int, point_idx: int) -> str:
    return f'N{num_candidates}_idx{point_idx:03d}'


def run_success_experiment(candidate_counts=(20, 50, 100, 200),
                           num_r_points=50, num_trials=5000, seed=42, num_batches=10):
    """实验1: 成功率 vs r/N，保存批次成功率。"""
    np.random.seed(seed)
    rows = []
    batches = {}
    batch_sizes = split_batch_sizes(num_trials, num_batches)
    total_trials = sum(batch_sizes)
    for num_candidates in candidate_counts:
        for r_frac_idx, threshold_fraction in enumerate(np.linspace(0.02, 0.98, num_r_points)):
            observe_until = max(1, int(threshold_fraction * num_candidates))
            batch_vals = np.empty(num_batches)
            total_successes = 0
            for batch_idx, batch_size in enumerate(batch_sizes):
                batch_success_rate = simulate_secretary_batch(num_candidates, observe_until, batch_size)
                batch_vals[batch_idx] = batch_success_rate
                total_successes += batch_success_rate * batch_size
            success_rate = total_successes / total_trials
            std_error = np.sqrt(success_rate * (1 - success_rate) / total_trials)
            theoretical_prob = theoretical_success_prob(observe_until, num_candidates)
            batch_key = _batch_key(num_candidates, r_frac_idx)
            batches[batch_key] = batch_vals
            rows.append({
                'N': num_candidates, 'r': observe_until, 'r_frac': observe_until / num_candidates,
                'success_mc': success_rate, 'se': std_error,
                'theory': theoretical_prob,
                'batch_key': batch_key,
            })
    results_df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(DATA_DIR / 'exp5_success.csv', index=False)
    np.savez(DATA_DIR / 'exp5_success_batches.npz', **batches)
    return results_df


def run_snell_experiment(num_candidates=20):
    """实验2: Snell 包络精确计算。"""
    value_function, immediate_payoff = snell_envelope(num_candidates)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(DATA_DIR / 'exp5_snell.npz', V=value_function, g=immediate_payoff)
    return value_function, immediate_payoff


def run_control_experiment(candidate_counts=(20, 50, 100, 200),
                           num_r_points=30, num_trials=2000, seed=42, num_batches=10):
    """实验3: 随机选择基线（模拟真实随机排列），保存批次成功率。"""
    np.random.seed(seed)
    rows = []
    batches = {}
    batch_sizes = split_batch_sizes(num_trials, num_batches)
    total_trials = sum(batch_sizes)
    for num_candidates in candidate_counts:
        for r_frac_idx, threshold_fraction in enumerate(np.linspace(0.02, 0.98, num_r_points)):
            observe_until = max(1, int(threshold_fraction * num_candidates))
            batch_vals = np.empty(num_batches)
            for batch_idx, batch_size in enumerate(batch_sizes):
                # 每批次模拟 batch_size 次随机排列，每次随机选一个位置
                successes = 0
                for _ in range(batch_size):
                    best_candidate_position = np.random.randint(0, num_candidates)  # 全局最优的随机位置
                    picked_position = np.random.randint(observe_until, num_candidates)   # 在可选范围内随机选
                    if picked_position == best_candidate_position:
                        successes += 1
                batch_vals[batch_idx] = successes / batch_size
            batches[_batch_key(num_candidates, r_frac_idx)] = batch_vals
            success_rate = batch_vals.mean()
            std_error = batch_vals.std(ddof=1) / np.sqrt(num_batches)
            rows.append({
                'N': num_candidates, 'r': observe_until, 'r_frac': observe_until / num_candidates,
                'success_mc': success_rate, 'se': std_error,
                'batch_key': _batch_key(num_candidates, r_frac_idx),
            })
    control_df = pd.DataFrame(rows)
    control_df.to_csv(DATA_DIR / 'exp5_control.csv', index=False)
    np.savez(DATA_DIR / 'exp5_control_batches.npz', **batches)
    return control_df


if __name__ == '__main__':
    print("=== 第五章实验：秘书问题 ===")
    print("实验1: 成功率 vs r/N ...")
    results_df = run_success_experiment()
    for num_candidates in [20, 50, 100, 200]:
        subset_df = results_df[results_df['N'] == num_candidates]
        best = subset_df.loc[subset_df['success_mc'].idxmax()]
        print(f"  N={num_candidates:3d}: best r*={best['r']}, max P={best['success_mc']:.4f}, "
              f"理论最优 ≈ {1/np.e:.4f}")

    print("实验2: Snell 包络 (N=20) ...")
    V_n, g_n = run_snell_experiment()
    print(f"  V_0={V_n[0]:.4f}, g_0={g_n[0]:.4f}")

    print("实验3: 随机选择基线 ...")
    control_results = run_control_experiment()
    print(f"  对照组成功率 ≈ {control_results['success_mc'].mean():.4f} (接近 1/N)")
    print("第五章实验完成。")
