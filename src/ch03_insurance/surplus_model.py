"""Cramér-Lundberg 保险盈余过程 + 指数鞅"""
import numpy as np
from scipy.optimize import brentq


class SurplusProcess:
    """Cramér-Lundberg 盈余过程。

    U_t = u + c·t − Σ_{i=1}^{N_t} Y_i

    u : 初始资本
    c : 保费速率（单位时间）
    λ : 泊松索赔强度
    Y ~ claim_distribution: 索赔额分布（需实现 rvs 方法）
    """

    def __init__(self, initial_capital: float, premium_rate: float, claim_intensity: float, claim_distribution):
        self.initial_capital = initial_capital
        self.premium_rate = premium_rate
        self.claim_intensity = claim_intensity
        self.claim_distribution = claim_distribution
        self._current_time = 0.0
        self._surplus = initial_capital

    def reset(self, initial_capital: float = None):
        if initial_capital is not None:
            self.initial_capital = initial_capital
        self._current_time = 0.0
        self._surplus = self.initial_capital

    @property
    def current_time(self) -> float:
        return self._current_time

    @property
    def surplus(self) -> float:
        return self._surplus

    def step_until_next_claim(self):
        """跳到下一次索赔时刻，更新盈余。返回 (t_new, surplus_new)。"""
        interarrival_time = np.random.exponential(1 / self.claim_intensity)
        self._current_time += interarrival_time
        self._surplus += self.premium_rate * interarrival_time
        claim_amount = self.claim_distribution.rvs()
        self._surplus -= claim_amount
        return self._current_time, self._surplus

    def simulate_path(self, time_horizon: float) -> tuple[np.ndarray, np.ndarray]:
        """生成 [0, T] 上的盈余轨迹（事件驱动）。

        返回 (time_points, surplus_values)，time_points[0]=0, surplus_values[0]=u。
        """
        self.reset()
        time_points = [0.0]
        surplus_values = [self._surplus]
        while self._current_time < time_horizon:
            # 纯保费增长直到下次索赔
            interarrival_time = np.random.exponential(1 / self.claim_intensity)
            claim_amount = self.claim_distribution.rvs()
            # 在索赔时刻增加一个点
            self._current_time += interarrival_time
            self._surplus += self.premium_rate * interarrival_time
            time_points.append(self._current_time)
            surplus_values.append(self._surplus)
            # 索赔后
            self._surplus -= claim_amount
            time_points.append(self._current_time)
            surplus_values.append(self._surplus)
            if self._surplus < 0:
                break
        return np.array(time_points), np.array(surplus_values)

    def simulate_until_ruin_or_T(self, time_horizon: float):
        """模拟直到破产或 T，返回 (ruined: bool, ruin_time: float, traj_times, traj_surplus)。"""
        self.reset()
        traj_times = [0.0]
        traj_surplus = [self._surplus]
        while self._current_time < time_horizon and self._surplus >= 0:
            interarrival_time = np.random.exponential(1 / self.claim_intensity)
            claim_amount = self.claim_distribution.rvs()
            self._current_time += interarrival_time
            self._surplus += self.premium_rate * interarrival_time
            traj_times.append(self._current_time)
            traj_surplus.append(self._surplus)
            self._surplus -= claim_amount
            traj_times.append(self._current_time)
            traj_surplus.append(self._surplus)
        ruined = self._surplus < 0
        return ruined, self._current_time, np.array(traj_times), np.array(traj_surplus)


def find_adjustment_R(claim_intensity: float, premium_rate: float, claim_mgf,
                       search_lower: float = 1e-12, search_upper: float = 10.0) -> float:
    """求解调整系数 R > 0，满足 λ(M_Y(R) − 1) = Rc。

    claim_mgf(r) = E[e^{rY}] 索赔额的矩母函数。
    使用 brentq 求根。
    """
    def lundberg_equation(rate):
        return claim_intensity * (claim_mgf(rate) - 1) - rate * premium_rate

    # 安全负荷条件：c > λ E[Y]
    # 此时 f'(0) = λ E[Y] - c < 0，且 f(0) = 0
    # f(r) 先负后正 → 存在唯一正根（对大多数索赔分布）
    try:
        return brentq(lundberg_equation, search_lower, search_upper, xtol=1e-12)
    except ValueError:
        # 搜索更宽的区间
        for upper_bound in [20, 50, 100]:
            try:
                return brentq(lundberg_equation, search_lower, upper_bound, xtol=1e-12)
            except ValueError:
                continue
        raise ValueError(f"无法在 (0, {search_upper}) 内找到调整系数 R")


def exp_claim_mgf_factory(dist_name: str, **params):
    """索赔额矩母函数工厂。

    'exponential': M_Y(r) = 1 / (1 - r/rate)  (r < rate)
    'gamma': M_Y(r) = (1 - r/rate)^{-shape}  (r < rate)
    """
    if dist_name == 'exponential':
        claim_rate = params.get('rate', 1.0)
        def mgf(rate):
            if rate >= claim_rate:
                return np.inf
            return 1.0 / (1.0 - rate / claim_rate)
        return mgf
    else:
        raise ValueError(f"未知分布: {dist_name}")


def lundberg_psi_exact(initial_capital: float, adjustment_coefficient: float) -> float:
    """Lundberg 指数上界：ψ(u) ≤ e^{−Ru}。这里返回的上界。"""
    return np.exp(-adjustment_coefficient * initial_capital)


def psi_exact_exponential(initial_capital: float, claim_intensity: float, premium_rate: float, mean_claim: float) -> float:
    """指数索赔下破产概率的精确解。

    ψ(u) = (1 / (1 + θ)) * exp(−θ u / (μ (1 + θ)))
    其中 θ = c/(λμ) − 1 = 安全负荷。
    """
    safety_loading = premium_rate / (claim_intensity * mean_claim) - 1
    if safety_loading <= 0:
        return 1.0  # 安全负荷为负，必然破产
    ruin_prob_coeff = 1.0 / (1.0 + safety_loading)
    ruin_exponent = -safety_loading * initial_capital / (mean_claim * (1.0 + safety_loading))
    return ruin_prob_coeff * np.exp(ruin_exponent)
