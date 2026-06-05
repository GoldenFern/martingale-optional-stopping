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


def run_convergence_experiment(lower_barrier=20, upper_barrier=10,
                               path_counts=np.logspace(2, 4, 8).astype(int),
                               max_steps=10000, num_repeats=6, seed=42):
    """双边 vs 单边收敛对比。"""
    random_walk = SymmetricRW(0.5)
    rows = []
    batches = {}
    for path_count in path_counts:
        # 双边
        exit_stop = ExitInterval(-lower_barrier, upper_barrier)
        two_sided_sim = MonteCarloSimulation(random_walk, exit_stop)
        two_sided_means = np.empty(num_repeats)
        for repeat_idx in range(num_repeats):
            estimated_mean, _ = two_sided_sim.estimate_expectation(
                0.0, path_count, max_steps, seed + 1000 * repeat_idx + int(path_count)
            )
            two_sided_means[repeat_idx] = estimated_mean
        rows.append({
            'stop_type': 'two_sided', 'n_paths': path_count,
            'mean': two_sided_means.mean(), 'se': two_sided_means.std(ddof=1),
            'reached_frac': 1.0,
        })
        batches[f'two_{path_count}'] = two_sided_means
        # 单边
        hitting_stop = HittingLevel(upper_barrier, 'up')
        one_sided_sim = MonteCarloSimulation(random_walk, hitting_stop)
        one_sided_means = np.empty(num_repeats)
        reached_fractions = np.empty(num_repeats)
        for repeat_idx in range(num_repeats):
            result = one_sided_sim.run(
                0.0, path_count, max_steps, seed + 50000 + 1000 * repeat_idx + int(path_count)
            )
            reached_mask = result.reached_stop
            reached_fractions[repeat_idx] = reached_mask.mean()
            one_sided_means[repeat_idx] = result.stopped_values[reached_mask].mean() if reached_mask.any() else np.nan
        rows.append({
            'stop_type': 'one_sided', 'n_paths': path_count,
            'mean': np.nanmean(one_sided_means), 'se': np.nanstd(one_sided_means, ddof=1),
            'reached_frac': reached_fractions.mean(),
        })
        batches[f'one_{path_count}'] = one_sided_means
    results_df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(DATA_DIR / 'exp4_convergence.csv', index=False)
    np.savez(DATA_DIR / 'exp4_convergence_batches.npz', **batches)
    return results_df


def run_truncation_experiment(lower_barrier=20, upper_barrier=10,
                              max_step_values=np.logspace(1, 5, 20).astype(int),
                              num_paths=1000, seed=42, num_repeats=12):
    """截断停时样本均值的批次分布。"""
    random_walk = SymmetricRW(0.5)
    level_stop = HittingLevel(upper_barrier, 'up')
    rows = []
    batches = {}
    for truncation_limit in max_step_values:
        truncated_stop = Truncated(level_stop, truncation_limit)
        mc_sim = MonteCarloSimulation(random_walk, truncated_stop)
        batch_mean_values = np.empty(num_repeats)
        for repeat_idx in range(num_repeats):
            estimated_mean, _ = mc_sim.estimate_expectation(0.0, num_paths, truncation_limit,
                                                            seed + 1000 * repeat_idx + int(truncation_limit))
            batch_mean_values[repeat_idx] = estimated_mean
        mean_estimate = batch_mean_values.mean()
        std_error = batch_mean_values.std(ddof=1)
        rows.append({'N': truncation_limit, 'mean': mean_estimate, 'se': std_error})
        batches[f'N_{truncation_limit}'] = batch_mean_values
    results_df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(DATA_DIR / 'exp4_truncation.csv', index=False)
    np.savez(DATA_DIR / 'exp4_truncation_batches.npz', **batches)
    return results_df


def run_tail_experiment(lower_barrier=20, upper_barrier=10, max_steps=5000, num_paths=3000, seed=42):
    """停时尾部分布：双边 vs 单边。"""
    random_walk = SymmetricRW(0.5)
    # 双边
    exit_stop = ExitInterval(-lower_barrier, upper_barrier)
    two_sided_sim = MonteCarloSimulation(random_walk, exit_stop)
    tail_time_two, survival_two = two_sided_sim.estimate_tail(0.0, num_paths, max_steps, seed)
    # 单边
    hitting_stop = HittingLevel(upper_barrier, 'up')
    one_sided_sim = MonteCarloSimulation(random_walk, hitting_stop)
    tail_time_one, survival_one = one_sided_sim.estimate_tail(0.0, num_paths, max_steps, seed)
    np.savez(DATA_DIR / 'exp4_tail.npz',
             t_two=tail_time_two, surv_two=survival_two,
             t_one=tail_time_one, surv_one=survival_one)
    return (tail_time_two, survival_two), (tail_time_one, survival_one)


if __name__ == '__main__':
    print("=== 第四章实验：赌徒破产 ===")
    print("实验1: 双边 vs 单边收敛 ...")
    results_df = run_convergence_experiment()
    for stop_type in ['two_sided', 'one_sided']:
        subset_df = results_df[results_df['stop_type'] == stop_type]
        print(f"  {stop_type}: M=100 -> {subset_df.iloc[0]['mean']:.4f}, "
              f"M=10000 -> {subset_df.iloc[-1]['mean']:.4f}")

    print("实验2: 截断偏误 ...")
    df_t = run_truncation_experiment()
    print(f"  N=10: E[S_{{t∧N}}]={df_t.iloc[0]['mean']:.4f}")
    print(f"  N=1000: E[S_{{t∧N}}]={df_t.iloc[10]['mean']:.4f}")
    print(f"  N=100000: E[S_{{t∧N}}]={df_t.iloc[-1]['mean']:.4f}")

    print("实验3: 停时尾部对比 ...")
    (t2, s2), (t1, s1) = run_tail_experiment()
    two_mask = t2 <= 1000
    one_mask = t1 <= 1000
    print(f"  双边 P(τ>500)={s2[500]:.4f}, P(τ>1000)={s2[1000]:.4f}")
    print(f"  单边 P(τ>500)={s1[500]:.4f}, P(τ>1000)={s1[1000]:.4f}")
    print("第四章实验完成。")
