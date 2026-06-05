"""鞅过程抽象基类与具体实现"""
import numpy as np
from abc import ABC, abstractmethod


class Martingale(ABC):
    """鞅过程抽象基类。所有具体过程需实现 reset 和 step。"""

    @abstractmethod
    def reset(self, x0: float):
        """重置初始状态。"""
        ...

    @abstractmethod
    def step(self) -> float:
        """前进一个时间步，返回新状态。"""
        ...

    def simulate_path(self, n_steps: int) -> np.ndarray:
        """生成 n_steps 步的路径（不含初始值），返回形状 (n_steps,)。"""
        path = np.empty(n_steps)
        for i in range(n_steps):
            path[i] = self.step()
        return path


class SymmetricRW(Martingale):
    """对称随机游走 S_n = ∑ η_i, η_i ∈ {+1, −1} 各概率 1/2。"""

    def __init__(self, p: float = 0.5):
        if not np.isclose(p, 0.5):
            raise ValueError("SymmetricRW requires p=0.5")
        self.p = p
        self._state = 0.0

    def reset(self, x0: float):
        self._state = x0

    def step(self) -> float:
        eta = 1.0 if np.random.rand() < self.p else -1.0
        self._state += eta
        return self._state


class AsymmetricRW(Martingale):
    """不对称随机游走 S_n = ∑ η_i, η_i ∈ {+1, −1}, P(η=+1)=p。

    同时提供 Wald 鞅 W_n = (q/p)^{S_n}，以及补偿游走 M_n = S_n − (2p−1)n
    （后者在 p≠0.5 时才是鞅）。
    """

    def __init__(self, p: float):
        if not (0 < p < 1):
            raise ValueError("p must be in (0,1)")
        self.p = p
        self.q = 1 - p
        self._state = 0.0
        self._n = 0

    def reset(self, x0: float):
        self._state = x0
        self._n = 0

    def step(self) -> float:
        eta = 1.0 if np.random.rand() < self.p else -1.0
        self._state += eta
        self._n += 1
        return self._state

    @property
    def n(self) -> int:
        return self._n

    def wald_value(self) -> float:
        """返回当前 Wald 鞅值 (q/p)^{S_n}。"""
        if self.p == 0.5:
            return 1.0  # 对称情况恒为1
        return (self.q / self.p) ** self._state

    def compensated_value(self) -> float:
        """返回补偿游走值 S_n − (2p−1)n（p≠0.5 时是鞅）。"""
        return self._state - (2 * self.p - 1) * self._n


class GeometricBrownianMotion:
    """几何布朗运动 dS = r·S·dt + σ·S·dW（非鞅），精确对数正态离散化。

    在风险中性测度 Q 下，GBM 有显式解：
    S_{t+Δt} = S_t · exp{(r − σ²/2)Δt + σ√Δt · Z}，  Z ∼ N(0,1)
    """

    def __init__(self, r: float, sigma: float):
        self.r = r
        self.sigma = sigma
        self._state = 0.0

    def reset(self, x0: float):
        self._state = x0

    def step(self, dt: float) -> float:
        """精确对数正态离散化，保持 S > 0 且分布上精确。"""
        self._state *= np.exp(
            (self.r - 0.5 * self.sigma**2) * dt
            + self.sigma * np.sqrt(dt) * np.random.randn()
        )
        return self._state

    def simulate_path(self, T: float, N: int) -> np.ndarray:
        """生成时间 [0,T] 上 N 步的路径，返回形状 (N+1,) 含 S_0。"""
        dt = T / N
        path = np.empty(N + 1)
        path[0] = self._state
        for i in range(1, N + 1):
            path[i] = self.step(dt)
        return path
