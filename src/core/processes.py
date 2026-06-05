"""鞅过程抽象基类与具体实现"""
import numpy as np
from abc import ABC, abstractmethod


class Martingale(ABC):
    """鞅过程抽象基类。所有具体过程需实现 reset 和 step。"""

    @abstractmethod
    def reset(self, initial_state: float):
        """重置初始状态。"""
        ...

    @abstractmethod
    def step(self) -> float:
        """前进一个时间步，返回新状态。"""
        ...

    def simulate_path(self, n_steps: int) -> np.ndarray:
        """生成 n_steps 步的路径（不含初始值），返回形状 (n_steps,)。"""
        path = np.empty(n_steps)
        for step_idx in range(n_steps):
            path[step_idx] = self.step()
        return path


class SymmetricRW(Martingale):
    """对称随机游走 S_n = ∑ η_i, η_i ∈ {+1, −1} 各概率 1/2。"""

    def __init__(self, prob_up: float = 0.5):
        if not np.isclose(prob_up, 0.5):
            raise ValueError("SymmetricRW requires p=0.5")
        self.prob_up = prob_up
        self._state = 0.0

    def reset(self, initial_state: float):
        self._state = initial_state

    def step(self) -> float:
        step_value = 1.0 if np.random.rand() < self.prob_up else -1.0
        self._state += step_value
        return self._state


class AsymmetricRW(Martingale):
    """不对称随机游走 S_n = ∑ η_i, η_i ∈ {+1, −1}, P(η=+1)=p。

    同时提供 Wald 鞅 W_n = (q/p)^{S_n}，以及补偿游走 M_n = S_n − (2p−1)n
    （后者在 p≠0.5 时才是鞅）。
    """

    def __init__(self, prob_up: float):
        if not (0 < prob_up < 1):
            raise ValueError("p must be in (0,1)")
        self.prob_up = prob_up
        self.prob_down = 1 - prob_up
        self._state = 0.0
        self._step_count = 0

    def reset(self, initial_state: float):
        self._state = initial_state
        self._step_count = 0

    def step(self) -> float:
        step_value = 1.0 if np.random.rand() < self.prob_up else -1.0
        self._state += step_value
        self._step_count += 1
        return self._state

    @property
    def step_count(self) -> int:
        return self._step_count

    def wald_value(self) -> float:
        """返回当前 Wald 鞅值 (q/p)^{S_n}。"""
        if self.prob_up == 0.5:
            return 1.0  # 对称情况恒为1
        return (self.prob_down / self.prob_up) ** self._state

    def compensated_value(self) -> float:
        """返回补偿游走值 S_n − (2p−1)n（p≠0.5 时是鞅）。"""
        return self._state - (2 * self.prob_up - 1) * self._step_count


class GeometricBrownianMotion:
    """几何布朗运动 dS = r·S·dt + σ·S·dW（非鞅），精确对数正态离散化。

    在风险中性测度 Q 下，GBM 有显式解：
    S_{t+Δt} = S_t · exp{(r − σ²/2)Δt + σ√Δt · Z}，  Z ∼ N(0,1)
    """

    def __init__(self, risk_free_rate: float, volatility: float):
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self._state = 0.0

    def reset(self, initial_state: float):
        self._state = initial_state

    def step(self, time_step: float) -> float:
        """精确对数正态离散化，保持 S > 0 且分布上精确。"""
        self._state *= np.exp(
            (self.risk_free_rate - 0.5 * self.volatility**2) * time_step
            + self.volatility * np.sqrt(time_step) * np.random.randn()
        )
        return self._state

    def simulate_path(self, time_horizon: float, num_steps: int) -> np.ndarray:
        """生成时间 [0,T] 上 N 步的路径，返回形状 (N+1,) 含 S_0。"""
        time_step = time_horizon / num_steps
        path = np.empty(num_steps + 1)
        path[0] = self._state
        for step_idx in range(1, num_steps + 1):
            path[step_idx] = self.step(time_step)
        return path
