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


def run_R_experiment(claim_intensity=1.0, mean_claim=1.0, safety_loadings=(0.1, 0.2, 0.5)):
    claim_distr = expon(scale=mean_claim)
    mgf = exp_claim_mgf_factory('exponential', rate=1/mean_claim)
    rows = []
    for safety_loading in safety_loadings:
        expected_claim = claim_intensity * mean_claim
        premium_rate = expected_claim * (1 + safety_loading)
        adj_coefficient = find_adjustment_R(claim_intensity, premium_rate, mgf)
        rows.append({'theta': safety_loading, 'c': premium_rate, 'R': adj_coefficient,
                     'lam': claim_intensity, 'mu': mean_claim})
    results_df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(DATA_DIR / 'exp3_R.csv', index=False)
    return results_df


def run_ruin_prob_experiment(claim_intensity=1.0, mean_claim=1.0, safety_loading=0.5,
                              capital_values=np.arange(0, 21),
                              num_paths=16000, time_horizon=200.0, seed=42,
                              num_batches=8):
    """有限时窗破产概率实验，并保存批次样本以便绘制箱线图。"""
    np.random.seed(seed)
    expected_claim = claim_intensity * mean_claim
    premium_rate = expected_claim * (1 + safety_loading)
    claim_distr = expon(scale=mean_claim)
    mgf = exp_claim_mgf_factory('exponential', rate=1/mean_claim)
    adj_coefficient = find_adjustment_R(claim_intensity, premium_rate, mgf)

    rows = []
    batch_results = {}
    batch_sizes = split_batch_sizes(num_paths, num_batches)
    total_paths = sum(batch_sizes)
    for initial_capital in capital_values:
        surplus_process = SurplusProcess(initial_capital, premium_rate, claim_intensity, claim_distr)
        batch_vals = np.empty(num_batches)
        ruins_total = 0
        for batch_idx, batch_size in enumerate(batch_sizes):
            ruins = 0
            for _ in range(batch_size):
                ruined, _, _, _ = surplus_process.simulate_until_ruin_or_T(time_horizon)
                if ruined:
                    ruins += 1
            batch_vals[batch_idx] = ruins / batch_size
            ruins_total += ruins
        psi_mc = ruins_total / total_paths
        psi_se = np.sqrt(psi_mc * (1 - psi_mc) / total_paths)
        psi_lundberg = lundberg_psi_exact(initial_capital, adj_coefficient)
        psi_exact = psi_exact_exponential(initial_capital, claim_intensity, premium_rate, mean_claim)
        batch_results[f'u_{initial_capital}'] = batch_vals
        rows.append({
            'u': initial_capital, 'psi_mc': psi_mc, 'psi_se': psi_se,
            'psi_lundberg': psi_lundberg, 'psi_exact': psi_exact,
            'R': adj_coefficient, 'theta': safety_loading,
        })

    results_df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(DATA_DIR / 'exp3_ruin_prob.csv', index=False)
    np.savez(DATA_DIR / 'exp3_ruin_prob_batches.npz', **batch_results)
    return results_df


def _simulate_stopped_exp_martingale_grid(
    initial_capital: float,
    premium_rate: float,
    claim_intensity: float,
    claim_distr,
    adj_coefficient: float,
    time_grid: np.ndarray,
) -> np.ndarray:
    r"""Simulate one path of M_{t \wedge \tau} = exp(-R U_{t \wedge \tau}) on a fixed grid."""
    values = np.empty_like(time_grid, dtype=float)
    current_time = 0.0
    current_surplus = initial_capital
    next_claim_time = np.random.exponential(1 / claim_intensity)
    ruined = False
    frozen_value = np.nan

    for i, target_time in enumerate(time_grid):
        if ruined:
            values[i] = frozen_value
            continue

        while next_claim_time <= target_time:
            current_surplus += premium_rate * (next_claim_time - current_time)
            current_time = next_claim_time
            current_surplus -= claim_distr.rvs()
            if current_surplus < 0:
                ruined = True
                frozen_value = np.exp(-adj_coefficient * current_surplus)
                break
            next_claim_time += np.random.exponential(1 / claim_intensity)

        if ruined:
            values[i] = frozen_value
        else:
            values[i] = np.exp(-adj_coefficient * (current_surplus + premium_rate * (target_time - current_time)))

    return values


def _select_spread_records(records: list[dict], sort_key: str, num_to_select: int) -> list[dict]:
    """Pick a few representative records spread across a sorted summary statistic."""
    ordered = sorted(records, key=lambda rec: rec[sort_key])
    quantiles = np.linspace(0.15, 0.85, num_to_select)
    positions = quantiles * (len(ordered) - 1)
    indices = []
    for pos in positions:
        idx = int(round(pos))
        if idx in indices:
            for step in range(1, len(ordered)):
                right = idx + step
                left = idx - step
                if right < len(ordered) and right not in indices:
                    idx = right
                    break
                if left >= 0 and left not in indices:
                    idx = left
                    break
        indices.append(idx)
    indices.sort()
    return [ordered[idx] for idx in indices]


def _pack_path_records(records: list[dict]) -> dict[str, np.ndarray]:
    """Store variable-length paths in padded arrays for reproducible plotting."""
    max_len = max(len(rec['times']) for rec in records)
    n_records = len(records)
    times = np.full((n_records, max_len), np.nan)
    surplus_vals = np.full((n_records, max_len), np.nan)
    martingale_vals = np.full((n_records, max_len), np.nan)
    lengths = np.empty(n_records, dtype=int)
    seeds = np.empty(n_records, dtype=int)
    taus = np.empty(n_records, dtype=float)

    for i, rec in enumerate(records):
        length = len(rec['times'])
        lengths[i] = length
        seeds[i] = rec['seed']
        taus[i] = rec['tau']
        times[i, :length] = rec['times']
        surplus_vals[i, :length] = rec['surplus_vals']
        martingale_vals[i, :length] = rec['martingale_vals']

    return {
        'times': times,
        'surplus_vals': surplus_vals,
        'martingale_vals': martingale_vals,
        'lengths': lengths,
        'seeds': seeds,
        'taus': taus,
    }


def _collect_martingale_path_examples(
    initial_capital: float,
    premium_rate: float,
    claim_intensity: float,
    claim_distr,
    adj_coefficient: float,
    time_horizon: float,
    num_survived: int = 4,
    num_ruined: int = 2,
    min_candidates: int = 15,
    max_seed_value: int = 5000,
) -> tuple[list[dict], list[dict]]:
    """Collect representative survived and ruined sample paths."""
    survived_candidates = []
    ruined_candidates = []

    for seed in range(1, max_seed_value + 1):
        np.random.seed(seed)
        surplus_process = SurplusProcess(initial_capital, premium_rate, claim_intensity, claim_distr)
        times, surplus_vals = surplus_process.simulate_path(time_horizon)
        martingale_vals = np.exp(-adj_coefficient * surplus_vals)
        record = {
            'seed': seed,
            'times': times,
            'surplus_vals': surplus_vals,
            'martingale_vals': martingale_vals,
            'tau': float(times[-1]),
            'm_end': float(martingale_vals[-1]),
            'ruined': bool(surplus_vals[-1] < 0),
        }
        if record['ruined']:
            ruined_candidates.append(record)
        else:
            survived_candidates.append(record)

        if (
            len(survived_candidates) >= min_candidates
            and len(ruined_candidates) >= min_candidates
        ):
            break

    if len(survived_candidates) < num_survived or len(ruined_candidates) < num_ruined:
        raise RuntimeError("Unable to find enough representative martingale paths.")

    survived = _select_spread_records(survived_candidates, sort_key='m_end', num_to_select=num_survived)
    ruined = _select_spread_records(ruined_candidates, sort_key='tau', num_to_select=num_ruined)
    return survived, ruined


def run_martingale_mean_experiment(
    claim_intensity=1.0,
    mean_claim=1.0,
    safety_loading=0.5,
    initial_capital=5.0,
    time_horizon=50.0,
    num_paths=16000,
    num_grid_points=101,
    seed=24680,
):
    r"""Estimate E[M_{t \wedge \tau}] and save representative survived/ruined paths."""
    np.random.seed(seed)
    premium_rate = claim_intensity * mean_claim * (1 + safety_loading)
    claim_distr = expon(scale=mean_claim)
    mgf = exp_claim_mgf_factory('exponential', rate=1 / mean_claim)
    adj_coefficient = find_adjustment_R(claim_intensity, premium_rate, mgf)
    time_grid = np.linspace(0.0, time_horizon, num_grid_points)
    martingale_initial = np.exp(-adj_coefficient * initial_capital)

    sum_martingale = np.zeros_like(time_grid)
    sum_sq_martingale = np.zeros_like(time_grid)
    for _ in range(num_paths):
        vals = _simulate_stopped_exp_martingale_grid(initial_capital, premium_rate, claim_intensity,
                                                      claim_distr, adj_coefficient, time_grid)
        sum_martingale += vals
        sum_sq_martingale += vals * vals

    mean_martingale = sum_martingale / num_paths
    if num_paths > 1:
        sample_var = (sum_sq_martingale - num_paths * mean_martingale * mean_martingale) / (num_paths - 1)
        sample_var = np.maximum(sample_var, 0.0)
        std_err_martingale = np.sqrt(sample_var / num_paths)
    else:
        std_err_martingale = np.zeros_like(mean_martingale)

    rows = []
    for t, mean_val, se_val in zip(time_grid, mean_martingale, std_err_martingale):
        rows.append({
            't': t,
            'm_mean': mean_val,
            'm_se': se_val,
            'm0': martingale_initial,
            'R': adj_coefficient,
            'u': initial_capital,
            'T': time_horizon,
            'n_paths': num_paths,
        })

    survived_paths, ruined_paths = _collect_martingale_path_examples(
        initial_capital=initial_capital,
        premium_rate=premium_rate,
        claim_intensity=claim_intensity,
        claim_distr=claim_distr,
        adj_coefficient=adj_coefficient,
        time_horizon=time_horizon,
    )
    survived_pack = _pack_path_records(survived_paths)
    ruined_pack = _pack_path_records(ruined_paths)

    results_df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(DATA_DIR / 'exp3_martingale_mean.csv', index=False)
    np.savez(
        DATA_DIR / 'exp3_martingale_paths.npz',
        survived_times=survived_pack['times'],
        survived_surplus_vals=survived_pack['surplus_vals'],
        survived_martingale_vals=survived_pack['martingale_vals'],
        survived_lengths=survived_pack['lengths'],
        survived_seeds=survived_pack['seeds'],
        survived_taus=survived_pack['taus'],
        ruined_times=ruined_pack['times'],
        ruined_surplus_vals=ruined_pack['surplus_vals'],
        ruined_martingale_vals=ruined_pack['martingale_vals'],
        ruined_lengths=ruined_pack['lengths'],
        ruined_seeds=ruined_pack['seeds'],
        ruined_taus=ruined_pack['taus'],
        m0=martingale_initial,
        R=adj_coefficient,
        u=initial_capital,
        T=time_horizon,
    )
    return results_df


if __name__ == '__main__':
    print("=== 第三章实验：保险破产模型 ===")
    print("实验1: 调整系数 R ...")
    df_R = run_R_experiment()
    for _, row in df_R.iterrows():
        print(f"  theta={row['theta']:.1f}, c={row['c']:.2f}, R={row['R']:.6f}")

    print("实验2: 破产概率 vs 初始资本 (theta=0.5, 16000 paths / 8 batches)...")
    results_df = run_ruin_prob_experiment()
    for u_check in [0, 5, 10, 15, 20]:
        row = results_df[results_df['u'] == u_check].iloc[0]
        print(f"  u={u_check:2d}, psi_MC={row['psi_mc']:.4f} +/- {1.96*row['psi_se']:.4f}, "
              f"psi_exact={row['psi_exact']:.4f}")
    print("  保存到 output/data/exp3_ruin_prob.csv")
    print("实验3: 指数鞅的样本路径与均值曲线 (theta=0.5, 16000 paths)...")
    df_m = run_martingale_mean_experiment()
    m0 = df_m['m0'].iloc[0]
    for t_check in [0, 10, 20, 30, 40, 50]:
        row = df_m[np.isclose(df_m['t'], t_check)].iloc[0]
        print(f"  t={t_check:2d}, mean={row['m_mean']:.4f}, theory={m0:.4f}")
    print("  保存到 output/data/exp3_martingale_mean.csv")
    print("第三章实验完成。")
