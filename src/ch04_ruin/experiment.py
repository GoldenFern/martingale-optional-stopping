"""第四章实验：赌徒破产 —— OST 条件失效"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))

import numpy as np
import pandas as pd
from core.processes import SymmetricRW
from core.stopping_times import ExitInterval, HittingLevel, Truncated
from core.simulation import MonteCarloSimulation

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'output' / 'data'


def run_convergence_experiment(a=20, b=10,
                               path_counts=np.logspace(2, 4, 8).astype(int),
                               max_steps=10000, n_repeats=6, seed=42):
    """双边 vs 单边收敛对比。"""
    rw = SymmetricRW(0.5)
    rows = []
    batches = {}
    for M in path_counts:
        # 双边
        tau_two = ExitInterval(-a, b)
        sim_two = MonteCarloSimulation(rw, tau_two)
        means_two = np.empty(n_repeats)
        for r in range(n_repeats):
            mean, _ = sim_two.estimate_expectation(
                0.0, M, max_steps, seed + 1000 * r + int(M)
            )
            means_two[r] = mean
        rows.append({
            'stop_type': 'two_sided', 'n_paths': M,
            'mean': means_two.mean(), 'se': means_two.std(ddof=1),
            'reached_frac': 1.0,
        })
        batches[f'two_{M}'] = means_two
        # 单边
        tau_one = HittingLevel(b, 'up')
        sim_one = MonteCarloSimulation(rw, tau_one)
        means_one = np.empty(n_repeats)
        reached_fracs = np.empty(n_repeats)
        for r in range(n_repeats):
            result = sim_one.run(
                0.0, M, max_steps, seed + 50000 + 1000 * r + int(M)
            )
            reached = result.reached_stop
            reached_fracs[r] = reached.mean()
            means_one[r] = result.stopped_values[reached].mean() if reached.any() else np.nan
        rows.append({
            'stop_type': 'one_sided', 'n_paths': M,
            'mean': np.nanmean(means_one), 'se': np.nanstd(means_one, ddof=1),
            'reached_frac': reached_fracs.mean(),
        })
        batches[f'one_{M}'] = means_one
    df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / 'exp4_convergence.csv', index=False)
    np.savez(DATA_DIR / 'exp4_convergence_batches.npz', **batches)
    return df


def run_truncation_experiment(a=20, b=10,
                              N_values=np.logspace(1, 5, 20).astype(int),
                              n_paths=1000, seed=42, n_repeats=12):
    """截断停时样本均值的批次分布。"""
    rw = SymmetricRW(0.5)
    hits = HittingLevel(b, 'up')
    rows = []
    batches = {}
    for N in N_values:
        tau = Truncated(hits, N)
        sim = MonteCarloSimulation(rw, tau)
        batch_means = np.empty(n_repeats)
        for r in range(n_repeats):
            mean, _ = sim.estimate_expectation(0.0, n_paths, N, seed + 1000 * r + int(N))
            batch_means[r] = mean
        mean = batch_means.mean()
        se = batch_means.std(ddof=1)
        rows.append({'N': N, 'mean': mean, 'se': se})
        batches[f'N_{N}'] = batch_means
    df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / 'exp4_truncation.csv', index=False)
    np.savez(DATA_DIR / 'exp4_truncation_batches.npz', **batches)
    return df


def run_tail_experiment(a=20, b=10, max_steps=5000, n_paths=3000, seed=42):
    """停时尾部分布：双边 vs 单边。"""
    rw = SymmetricRW(0.5)
    # 双边
    tau_two = ExitInterval(-a, b)
    sim_two = MonteCarloSimulation(rw, tau_two)
    t2, surv2 = sim_two.estimate_tail(0.0, n_paths, max_steps, seed)
    # 单边
    tau_one = HittingLevel(b, 'up')
    sim_one = MonteCarloSimulation(rw, tau_one)
    t1, surv1 = sim_one.estimate_tail(0.0, n_paths, max_steps, seed)
    np.savez(DATA_DIR / 'exp4_tail.npz',
             t_two=t2, surv_two=surv2,
             t_one=t1, surv_one=surv1)
    return (t2, surv2), (t1, surv1)


if __name__ == '__main__':
    print("=== 第四章实验：赌徒破产 ===")
    print("实验1: 双边 vs 单边收敛 ...")
    df = run_convergence_experiment()
    for stype in ['two_sided', 'one_sided']:
        sub = df[df['stop_type'] == stype]
        print(f"  {stype}: M=100 -> {sub.iloc[0]['mean']:.4f}, "
              f"M=10000 -> {sub.iloc[-1]['mean']:.4f}")

    print("实验2: 截断偏误 ...")
    df_t = run_truncation_experiment()
    print(f"  N=10: E[S_{{t∧N}}]={df_t.iloc[0]['mean']:.4f}")
    print(f"  N=1000: E[S_{{t∧N}}]={df_t.iloc[10]['mean']:.4f}")
    print(f"  N=100000: E[S_{{t∧N}}]={df_t.iloc[-1]['mean']:.4f}")

    print("实验3: 停时尾部对比 ...")
    (t2, s2), (t1, s1) = run_tail_experiment()
    mask2 = t2 <= 1000
    mask1 = t1 <= 1000
    print(f"  双边 P(τ>500)={s2[500]:.4f}, P(τ>1000)={s2[1000]:.4f}")
    print(f"  单边 P(τ>500)={s1[500]:.4f}, P(τ>1000)={s1[1000]:.4f}")
    print("第四章实验完成。")
