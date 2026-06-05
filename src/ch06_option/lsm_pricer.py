"""LSM (Longstaff-Schwartz) 美式期权定价 + Black-Scholes 基准"""
import numpy as np
from numpy.linalg import lstsq


def black_scholes_put(stock_price, strike_price, risk_free_rate, volatility, maturity):
    """Black-Scholes 公式：欧式看跌期权价格。

    d1 = (ln(S/K) + (r + σ²/2)T) / (σ√T),  d2 = d1 − σ√T
    """
    from scipy.stats import norm
    d1 = (np.log(stock_price / strike_price) + (risk_free_rate + 0.5 * volatility**2) * maturity) / (
        volatility * np.sqrt(maturity))
    d2 = d1 - volatility * np.sqrt(maturity)
    return strike_price * np.exp(-risk_free_rate * maturity) * norm.cdf(-d2) - stock_price * norm.cdf(-d1)


class LSMPricer:
    """Longstaff-Schwartz 最小二乘 Monte Carlo 美式看跌期权定价器。

    参数
    ----
    initial_stock_price : float  — 初始股价 S₀
    strike_price : float        — 行权价 K
    risk_free_rate : float      — 无风险利率 r
    volatility : float          — 波动率 σ
    maturity : float            — 到期时间 T
    num_time_steps : int        — 离散化时间步数 N
    num_paths : int             — Monte Carlo 模拟路径数 M
    seed : int                  — 随机种子
    """

    def __init__(self, initial_stock_price, strike_price, risk_free_rate, volatility,
                 maturity, num_time_steps=50, num_paths=20000, seed=42):
        self.initial_stock_price = initial_stock_price
        self.strike_price = strike_price
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.maturity = maturity
        self.num_time_steps = num_time_steps
        self.num_paths = num_paths
        self.seed = seed
        self.time_step = maturity / num_time_steps
        self.discount_factor = np.exp(-risk_free_rate * self.time_step)
        self._priced = False

    def simulate_paths(self):
        """前向模拟 GBM 路径。返回 (num_paths, num_time_steps+1)。"""
        np.random.seed(self.seed)
        price_paths = np.zeros((self.num_paths, self.num_time_steps + 1))
        price_paths[:, 0] = self.initial_stock_price
        for step_idx in range(1, self.num_time_steps + 1):
            normal_random = np.random.randn(self.num_paths)
            price_paths[:, step_idx] = price_paths[:, step_idx - 1] * np.exp(
                (self.risk_free_rate - 0.5 * self.volatility**2) * self.time_step
                + self.volatility * np.sqrt(self.time_step) * normal_random
            )
        return price_paths

    def price(self):
        """执行 LSM 算法，返回 (价格, SE, 路径, 停时指数, 行权掩码)。"""
        price_paths = self.simulate_paths()
        payoff_fn = lambda s: np.maximum(self.strike_price - s, 0)  # 看跌期权

        cash_flow = payoff_fn(price_paths[:, -1]).copy()
        exercise_time = np.full(self.num_paths, self.num_time_steps, dtype=int)
        exercise_decision = np.zeros((self.num_paths, self.num_time_steps + 1), dtype=bool)

        for step_idx in range(self.num_time_steps - 1, 0, -1):
            immediate_payoff = payoff_fn(price_paths[:, step_idx])
            in_the_money = immediate_payoff > 0
            if in_the_money.sum() < 3:
                continue

            # 将当前已知未来现金流折现回 t 时刻，作为 continuation value 的样本。
            discounted_cashflows = cash_flow[in_the_money] * np.exp(
                -self.risk_free_rate * self.time_step * (exercise_time[in_the_money] - step_idx))
            prices_at_t = price_paths[in_the_money, step_idx]

            # 基函数：1, S, S^2
            basis_matrix = np.column_stack([np.ones_like(prices_at_t), prices_at_t, prices_at_t**2])
            regression_coefficients, *_ = lstsq(basis_matrix, discounted_cashflows, rcond=None)
            continuation_estimate = basis_matrix @ regression_coefficients

            # 决策
            immediate_itm = payoff_fn(prices_at_t)
            should_exercise = immediate_itm > continuation_estimate
            itm_indices = np.where(in_the_money)[0]
            exercise_decision[itm_indices, step_idx] = should_exercise
            exercise_indices = itm_indices[should_exercise]
            exercise_time[exercise_indices] = step_idx
            cash_flow[exercise_indices] = immediate_itm[should_exercise]

        # 定价
        payoff_at_exercise = payoff_fn(price_paths[np.arange(self.num_paths), exercise_time])
        discounted_payoffs = np.exp(-self.risk_free_rate * self.time_step * exercise_time) * payoff_at_exercise
        option_price = discounted_payoffs.mean()
        std_error = discounted_payoffs.std(ddof=1) / np.sqrt(self.num_paths)
        exercise_decision[:, self.num_time_steps] = payoff_fn(price_paths[:, self.num_time_steps]) > 0
        self._priced = True
        return option_price, std_error, price_paths, exercise_time, exercise_decision

    def exercise_boundary(self, grid_size=50):
        """计算 (t, S) 网格上的行权概率。"""
        option_price, std_error, price_paths, exercise_time, exercise_mask = self.price()
        time_grid = np.linspace(self.time_step, self.maturity, self.num_time_steps)
        price_grid = np.linspace(0.5 * self.strike_price, 1.5 * self.strike_price, grid_size)
        boundary_prob = np.zeros((self.num_time_steps, grid_size))

        for step_idx in range(self.num_time_steps):
            stock_values = price_paths[:, step_idx + 1]  # 时刻 t_{i+1} 的股价
            for price_idx, stock_grid_point in enumerate(price_grid):
                in_bin_mask = np.abs(stock_values - stock_grid_point) < (price_grid[1] - price_grid[0])
                if in_bin_mask.sum() > 10:
                    boundary_prob[step_idx, price_idx] = exercise_mask[in_bin_mask, step_idx + 1].mean()

        return time_grid, price_grid, boundary_prob
