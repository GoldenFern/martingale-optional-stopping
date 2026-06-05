"""第二章实验：Moran 模型中固定概率与停时"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))

import numpy as np
import pandas as pd
from ch02_moran.moran_model import MoranProcess, expected_tau_moran
from core.batching import split_batch_sizes

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'output' / 'data'


def run_fixation_experiment(N=50, n_paths=5000, seed=42):
    """实验：固定概率 vs A等位基因初始频率，仅 N=50。"""
    np.random.seed(seed)
    freq_values = np.arange(0.1, 1.0, 0.1)
    rows = []
    fixation_batches = {}
    n_batches = 10
    batch_sizes = split_batch_sizes(n_paths, n_batches)
    total_paths = sum(batch_sizes)
    tau_exact = expected_tau_moran(N)
    for freq in freq_values:
        x0 = int(freq * N)
        model = MoranProcess(N)
        fixations = 0
        tau_samples = []
        batch_vals = np.empty(n_batches)
        for b, batch_size in enumerate(batch_sizes):
            batch_fix = 0
            for _ in range(batch_size):
                model.reset(x0)
                path = model.simulate_path()
                fixed = int(path[-1] == N)
                if fixed:
                    fixations += 1
                    batch_fix += 1
                tau_samples.append(len(path) - 1)
            batch_vals[b] = batch_fix / batch_size
        p_fix = fixations / total_paths
        tau_mean = np.mean(tau_samples)
        tau_se = np.std(tau_samples, ddof=1) / np.sqrt(total_paths)
        fixation_batches[f'{freq:.1f}'] = batch_vals
        rows.append({
            'N': N, 'initial_freq': freq, 'x0': x0,
            'p_fixation_mc': p_fix, 'p_fixation_theory': freq,
            'tau_mc_mean': tau_mean, 'tau_mc_se': tau_se,
            'tau_theory': tau_exact[x0],
        })
    df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / 'exp2_fixation.csv', index=False)
    np.savez(DATA_DIR / 'exp2_fixation_batches.npz', **fixation_batches)
    return df


def run_tau_distribution(N=50, x0=None, n_paths=10000, seed=42):
    """实验：停时分布。"""
    np.random.seed(seed)
    if x0 is None:
        x0 = N // 2
    model = MoranProcess(N)
    tau_samples = np.empty(n_paths, dtype=int)
    for i in range(n_paths):
        model.reset(x0)
        path = model.simulate_path()
        tau_samples[i] = len(path) - 1
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savetxt(DATA_DIR / 'exp2_tau_dist.csv', tau_samples, fmt='%d', delimiter=',')
    return tau_samples


if __name__ == '__main__':
    print("=== 第二章实验：Moran 模型 ===")
    print("实验1: 固定概率 vs A等位基因初始频率 (N=50)...")
    df = run_fixation_experiment()
    print(f"  保存 {len(df)} 行到 output/data/exp2_fixation.csv")
    print("实验2: 停时分布 (N=50, x0=25)...")
    tau = run_tau_distribution()
    print(f"  停时均值={tau.mean():.0f}, 中位数={np.median(tau):.0f}, "
          f"最大={tau.max()}")
    print(f"  保存到 output/data/exp2_tau_dist.csv")
    print("第二章实验完成。")
