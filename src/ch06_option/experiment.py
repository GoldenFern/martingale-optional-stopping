"""第六章实验：美式期权 LSM 定价"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))

import numpy as np
import pandas as pd
from lsm_pricer import LSMPricer, black_scholes_put

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'output' / 'data'


def run_price_comparison(K=40, r=0.06, sigma=0.2, T=1.0, N=50, M=30000, seed=42):
    """美式 vs 欧式期权价格对比。"""
    S0_values = np.linspace(36, 44, 17)
    rows = []
    for S0 in S0_values:
        pricer = LSMPricer(S0, K, r, sigma, T, N=N, M=M, seed=seed)
        am_price, am_se, _, _, _ = pricer.price()
        eu_price = black_scholes_put(S0, K, r, sigma, T)
        payoff_val = max(K - S0, 0)
        rows.append({
            'S0': S0, 'american': am_price, 'am_se': am_se,
            'european': eu_price, 'payoff': payoff_val,
        })
    df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / 'exp6_price.csv', index=False)
    return df


def run_exercise_boundary(S0=40, K=40, r=0.06, sigma=0.2, T=1.0, N=50,
                          M=30000, seed=42):
    """计算行权边界。"""
    pricer = LSMPricer(S0, K, r, sigma, T, N=N, M=M, seed=seed)
    t_grid, s_grid, prob = pricer.exercise_boundary()
    np.savez(DATA_DIR / 'exp6_boundary.npz',
             t_grid=t_grid, s_grid=s_grid, prob=prob)
    return t_grid, s_grid, prob


def run_example_paths(S0=40, K=40, r=0.06, sigma=0.2, T=1.0, N=50,
                      M=30000, seed=42):
    """获取 LSM 定价的完整输出（用于绘制路径图）。"""
    pricer = LSMPricer(S0, K, r, sigma, T, N=N, M=M, seed=seed)
    price, se, S, tau, exercised = pricer.price()
    # 仅保存前 15 条路径用于可视化
    np.savez(DATA_DIR / 'exp6_paths.npz',
             S=S[:15], tau=tau[:15], exercised=exercised[:15])
    return S[:15], tau[:15], exercised[:15]


if __name__ == '__main__':
    print("=== 第六章实验：美式期权 LSM 定价 ===")
    print("实验1: 价格对比 (30000 paths)...")
    df = run_price_comparison()
    for _, row in df.iterrows():
        if row['S0'] % 2 == 0:
            print(f"  S0={row['S0']:.0f}, Amer={row['american']:.4f} +/- {1.96*row['am_se']:.4f}, "
                  f"Euro={row['european']:.4f}")

    print("实验2: 行权边界...")
    run_exercise_boundary()
    print("  boundary saved.")

    print("实验3: 路径样本...")
    S, tau, _ = run_example_paths()
    print(f"  saved {len(S)} paths.")
    print("第六章实验完成。")
