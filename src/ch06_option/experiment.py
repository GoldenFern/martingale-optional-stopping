"""第六章实验：美式期权 LSM 定价"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))

import numpy as np
import pandas as pd
from core.batching import split_batch_sizes
from ch06_option.lsm_pricer import LSMPricer, black_scholes_put

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'output' / 'data'


def run_price_comparison(strike_price=40, risk_free_rate=0.06, volatility=0.2, maturity=1.0,
                         num_time_steps=50, num_paths=30000,
                         seed=42, num_batches=8):
    """美式 vs 欧式期权价格对比。"""
    initial_prices = np.linspace(36, 44, 17)
    rows = []
    batches = {}
    batch_sizes = split_batch_sizes(num_paths, num_batches)
    for initial_price in initial_prices:
        european_price = black_scholes_put(initial_price, strike_price, risk_free_rate, volatility, maturity)
        immediate_payoff = max(strike_price - initial_price, 0)
        batch_vals = np.empty(num_batches)
        for batch_idx, batch_size in enumerate(batch_sizes):
            batch_pricer = LSMPricer(
                initial_price, strike_price, risk_free_rate, volatility, maturity,
                num_time_steps=num_time_steps, num_paths=batch_size, seed=seed + 1000 * batch_idx
            )
            batch_price, _, _, _, _ = batch_pricer.price()
            batch_vals[batch_idx] = batch_price
        american_price = batch_vals.mean()
        american_std_error = batch_vals.std(ddof=1) / np.sqrt(num_batches)
        batches[f'S0_{initial_price:.1f}'] = batch_vals
        rows.append({
            'S0': initial_price, 'american': american_price, 'am_se': american_std_error,
            'european': european_price, 'payoff': immediate_payoff,
        })
    results_df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(DATA_DIR / 'exp6_price.csv', index=False)
    np.savez(DATA_DIR / 'exp6_price_batches.npz', **batches)
    return results_df


def run_exercise_boundary(initial_stock_price=40, strike_price=40, risk_free_rate=0.06,
                          volatility=0.2, maturity=1.0, num_time_steps=50,
                          num_paths=30000, seed=42):
    """计算行权边界。"""
    pricer = LSMPricer(initial_stock_price, strike_price, risk_free_rate, volatility, maturity,
                       num_time_steps=num_time_steps, num_paths=num_paths, seed=seed)
    time_grid, price_grid, exercise_prob = pricer.exercise_boundary()
    np.savez(DATA_DIR / 'exp6_boundary.npz',
             t_grid=time_grid, s_grid=price_grid, prob=exercise_prob)
    return time_grid, price_grid, exercise_prob


def run_example_paths(initial_stock_price=40, strike_price=40, risk_free_rate=0.06,
                      volatility=0.2, maturity=1.0, num_time_steps=50,
                      num_paths=30000, seed=42):
    """获取 LSM 定价的完整输出（用于绘制路径图）。"""
    pricer = LSMPricer(initial_stock_price, strike_price, risk_free_rate, volatility, maturity,
                       num_time_steps=num_time_steps, num_paths=num_paths, seed=seed)
    option_price, std_error, stock_paths, exercise_times, exercise_mask = pricer.price()
    # 仅保存前 15 条路径用于可视化
    np.savez(DATA_DIR / 'exp6_paths.npz',
             S=stock_paths[:15], tau=exercise_times[:15], exercised=exercise_mask[:15])
    return stock_paths[:15], exercise_times[:15], exercise_mask[:15]


if __name__ == '__main__':
    print("=== 第六章实验：美式期权 LSM 定价 ===")
    print("实验1: 价格对比 (30000 paths)...")
    results_df = run_price_comparison()
    for _, row in results_df.iterrows():
        if row['S0'] % 2 == 0:
            print(f"  S0={row['S0']:.0f}, Amer={row['american']:.4f} +/- {1.96*row['am_se']:.4f}, "
                  f"Euro={row['european']:.4f}")

    print("实验2: 行权边界...")
    run_exercise_boundary()
    print("  boundary saved.")

    print("实验3: 路径样本...")
    stock_paths, exercise_times, _ = run_example_paths()
    print(f"  saved {len(stock_paths)} paths.")
    print("第六章实验完成。")
