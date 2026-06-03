"""Longstaff-Schwartz 最小二乘 Monte Carlo 美式期权定价"""
import numpy as np
from numpy.linalg import lstsq


def black_scholes_put(S, K, r, sigma, T):
    """欧式看跌期权 Black-Scholes 公式。"""
    from scipy.stats import norm
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


class LSMPricer:
    """Longstaff-Schwartz 美式期权定价器。"""

    def __init__(self, S0, K, r, sigma, T, N=50, M=20000, seed=42):
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.N = N       # 时间步数
        self.M = M       # 模拟路径数
        self.seed = seed
        self.dt = T / N
        self.df = np.exp(-r * self.dt)
        self._priced = False

    def simulate_paths(self):
        """前向模拟 GBM 路径。返回 (M, N+1)。"""
        np.random.seed(self.seed)
        S = np.zeros((self.M, self.N + 1))
        S[:, 0] = self.S0
        for t in range(1, self.N + 1):
            Z = np.random.randn(self.M)
            S[:, t] = S[:, t - 1] * np.exp(
                (self.r - 0.5 * self.sigma**2) * self.dt
                + self.sigma * np.sqrt(self.dt) * Z
            )
        return S

    def price(self):
        """执行 LSM 算法，返回 (价格, SE, 路径, 停时指数, 行权掩码)。"""
        S = self.simulate_paths()
        payoff = lambda s: np.maximum(self.K - s, 0)  # 看跌期权

        cashflow = payoff(S[:, -1]).copy()
        tau = np.full(self.M, self.N, dtype=int)
        exercise_decision = np.zeros((self.M, self.N + 1), dtype=bool)

        for t in range(self.N - 1, 0, -1):
            immediate_all = payoff(S[:, t])
            itm = immediate_all > 0
            if itm.sum() < 3:
                continue

            # 将当前已知未来现金流折现回 t 时刻，作为 continuation value 的样本。
            Y = cashflow[itm] * np.exp(-self.r * self.dt * (tau[itm] - t))
            X = S[itm, t]

            # 基函数：1, S, S^2
            A = np.column_stack([np.ones_like(X), X, X**2])
            beta, *_ = lstsq(A, Y, rcond=None)
            C = A @ beta

            # 决策
            immediate = payoff(X)
            exercise = immediate > C
            itm_idx = np.where(itm)[0]
            exercise_decision[itm_idx, t] = exercise
            exercise_idx = itm_idx[exercise]
            tau[exercise_idx] = t
            cashflow[exercise_idx] = immediate[exercise]

        # 定价
        final_payoff = payoff(S[np.arange(self.M), tau])
        discounted_payoffs = np.exp(-self.r * self.dt * tau) * final_payoff
        price = discounted_payoffs.mean()
        se = discounted_payoffs.std(ddof=1) / np.sqrt(self.M)
        exercise_decision[:, self.N] = payoff(S[:, self.N]) > 0
        self._priced = True
        return price, se, S, tau, exercise_decision

    def exercise_boundary(self, n_grid=50):
        """计算 (t, S) 网格上的行权概率。"""
        price_val, se, S, tau, exercised = self.price()
        t_grid = np.linspace(self.dt, self.T, self.N)
        s_grid = np.linspace(0.5 * self.K, 1.5 * self.K, n_grid)
        boundary_prob = np.zeros((self.N, n_grid))

        for i in range(self.N):
            s_vals = S[:, i + 1]  # 时刻 t_{i+1} 的股价
            for j, s0 in enumerate(s_grid):
                in_bin = np.abs(s_vals - s0) < (s_grid[1] - s_grid[0])
                if in_bin.sum() > 10:
                    boundary_prob[i, j] = exercised[in_bin, i + 1].mean()

        return t_grid, s_grid, boundary_prob
