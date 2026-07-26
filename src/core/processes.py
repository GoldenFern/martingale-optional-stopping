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


