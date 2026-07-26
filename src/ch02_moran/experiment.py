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


def run_fixation_experiment(pop_size: int = 50, num_paths: int = 50000, seed: int = 42) -> pd.DataFrame:
    """实验：固定概率 vs A等位基因初始频率，仅 N=50。"""
    np.random.seed(seed)
    allele_frequencies = np.arange(0.1, 1.0, 0.1)
    rows = []
    fixation_batches = {}
    num_batches = 10
    batch_sizes = split_batch_sizes(num_paths, num_batches)
    total_paths = sum(batch_sizes)
    tau_exact = expected_tau_moran(pop_size)
    for freq in allele_frequencies:
        initial_allele_count = int(freq * pop_size)
        model = MoranProcess(pop_size)
        fixations = 0
        tau_samples = []
        batch_vals = np.empty(num_batches)
        for batch_idx, batch_size in enumerate(batch_sizes):
            tau_batch, fixed_batch = model.simulate_batch_until_absorption(
                initial_allele_count, batch_size)
            fixations += int(fixed_batch.sum())
            batch_vals[batch_idx] = fixed_batch.mean()
            tau_samples.append(tau_batch)
        tau_all = np.concatenate(tau_samples)
        p_fix = fixations / total_paths
        tau_mean = tau_all.mean()
        tau_se = tau_all.std(ddof=1) / np.sqrt(total_paths)
        fixation_batches[f'{freq:.1f}'] = batch_vals
        rows.append({
            'N': pop_size, 'initial_freq': freq, 'x0': initial_allele_count,
            'p_fixation_mc': p_fix, 'p_fixation_theory': freq,
            'tau_mc_mean': tau_mean, 'tau_mc_se': tau_se,
            'tau_theory': tau_exact[initial_allele_count],
        })
    results_df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(DATA_DIR / 'exp2_fixation.csv', index=False, encoding='utf-8')
    np.savez(DATA_DIR / 'exp2_fixation_batches.npz', **fixation_batches)
    return results_df


def run_tau_distribution(pop_size: int = 50, initial_allele_count: int | None = None,
                         num_paths: int = 10000, seed: int = 42) -> np.ndarray:
    """实验：停时分布。"""
    np.random.seed(seed)
    if initial_allele_count is None:
        initial_allele_count = pop_size // 2
    model = MoranProcess(pop_size)
    tau_samples, _ = model.simulate_batch_until_absorption(initial_allele_count, num_paths)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savetxt(DATA_DIR / 'exp2_tau_dist.csv', tau_samples, fmt='%d', delimiter=',')
    return tau_samples


if __name__ == '__main__':
    print("=== 第二章实验：Moran 模型 ===")
    print("实验1: 固定概率 vs A等位基因初始频率 (N=50)...")
    results_df = run_fixation_experiment()
    print(f"  保存 {len(results_df)} 行到 output/data/exp2_fixation.csv")
    print("实验2: 停时分布 (N=50, x0=25)...")
    tau = run_tau_distribution()
    print(f"  停时均值={tau.mean():.0f}, 中位数={np.median(tau):.0f}, "
          f"最大={tau.max()}")
    print(f"  保存到 output/data/exp2_tau_dist.csv")
    print("第二章实验完成。")
