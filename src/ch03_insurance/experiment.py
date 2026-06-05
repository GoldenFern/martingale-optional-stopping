"""第三章实验：保险破产概率"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))

import numpy as np
import pandas as pd
from scipy.stats import expon
from core.batching import split_batch_sizes
from ch03_insurance.surplus_model import (
    SurplusProcess, find_adjustment_R, exp_claim_mgf_factory,
    lundberg_psi_exact, psi_exact_exponential,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'output' / 'data'


def run_R_experiment(lam=1.0, mu=1.0, theta_values=(0.1, 0.2, 0.5)):
    claim_dist = expon(scale=mu)
    mgf = exp_claim_mgf_factory('exponential', rate=1/mu)
    rows = []
    for theta in theta_values:
        expected_claim = lam * mu
        c = expected_claim * (1 + theta)
        R = find_adjustment_R(lam, c, mgf)
        rows.append({'theta': theta, 'c': c, 'R': R, 'lam': lam, 'mu': mu})
    df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / 'exp3_R.csv', index=False)
    return df


def run_ruin_prob_experiment(lam=1.0, mu=1.0, theta=0.5,
                              u_values=np.arange(0, 21),
                              n_paths=16000, T=200.0, seed=42,
                              n_batches=8):
    """有限时窗破产概率实验，并保存批次样本以便绘制箱线图。"""
    np.random.seed(seed)
    expected_claim = lam * mu
    c = expected_claim * (1 + theta)
    claim_dist = expon(scale=mu)
    mgf = exp_claim_mgf_factory('exponential', rate=1/mu)
    R = find_adjustment_R(lam, c, mgf)

    rows = []
    batch_results = {}
    batch_sizes = split_batch_sizes(n_paths, n_batches)
    total_paths = sum(batch_sizes)
    for u in u_values:
        proc = SurplusProcess(u, c, lam, claim_dist)
        batch_vals = np.empty(n_batches)
        ruins_total = 0
        for b, batch_size in enumerate(batch_sizes):
            ruins = 0
            for _ in range(batch_size):
                ruined, _, _, _ = proc.simulate_until_ruin_or_T(T)
                if ruined:
                    ruins += 1
            batch_vals[b] = ruins / batch_size
            ruins_total += ruins
        psi_mc = ruins_total / total_paths
        psi_se = np.sqrt(psi_mc * (1 - psi_mc) / total_paths)
        psi_lundberg = lundberg_psi_exact(u, R)
        psi_exact = psi_exact_exponential(u, lam, c, mu)
        batch_results[f'u_{u}'] = batch_vals
        rows.append({
            'u': u, 'psi_mc': psi_mc, 'psi_se': psi_se,
            'psi_lundberg': psi_lundberg, 'psi_exact': psi_exact,
            'R': R, 'theta': theta,
        })

    df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / 'exp3_ruin_prob.csv', index=False)
    np.savez(DATA_DIR / 'exp3_ruin_prob_batches.npz', **batch_results)
    return df


if __name__ == '__main__':
    print("=== 第三章实验：保险破产模型 ===")
    print("实验1: 调整系数 R ...")
    df_R = run_R_experiment()
    for _, row in df_R.iterrows():
        print(f"  theta={row['theta']:.1f}, c={row['c']:.2f}, R={row['R']:.6f}")

    print("实验2: 破产概率 vs 初始资本 (theta=0.5, 16000 paths / 8 batches)...")
    df = run_ruin_prob_experiment()
    for u_check in [0, 5, 10, 15, 20]:
        row = df[df['u'] == u_check].iloc[0]
        print(f"  u={u_check:2d}, psi_MC={row['psi_mc']:.4f} +/- {1.96*row['psi_se']:.4f}, "
              f"psi_exact={row['psi_exact']:.4f}")
    print("  保存到 output/data/exp3_ruin_prob.csv")
    print("第三章实验完成。")
