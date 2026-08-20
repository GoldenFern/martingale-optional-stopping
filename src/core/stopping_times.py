"""停时类族"""
from abc import ABC, abstractmethod
from numbers import Integral
import numpy as np


class StoppingTime(ABC):
    """停时抽象基类。"""

    @abstractmethod
    def should_stop(self, state: float, step_idx: int, history: np.ndarray = None) -> bool:
        """判断在当前状态和步数下是否应停止。"""
        ...

    def supports_batch(self) -> bool:
        """是否提供向量化判停 stop_mask（供模拟引擎的快速路径使用）。"""
        return False

    def stop_mask(self, states: np.ndarray, step_indices: np.ndarray) -> np.ndarray:
        """逐元素判停：states 形状 (paths, steps)，step_indices 形状 (1, steps)。"""
        raise NotImplementedError


class FixedTime(StoppingTime):
    """固定时刻停时 τ = N。"""

    def __init__(self, stop_step: int):
        if isinstance(stop_step, bool) or not isinstance(stop_step, Integral) or stop_step < 0:
            raise ValueError("stop_step must be a non-negative integer")
        self.stop_step = stop_step

    def should_stop(self, state: float, step_idx: int, history=None) -> bool:
        return step_idx >= self.stop_step

    def supports_batch(self) -> bool:
        return True

    def stop_mask(self, states: np.ndarray, step_indices: np.ndarray) -> np.ndarray:
        return np.broadcast_to(step_indices >= self.stop_step, states.shape)


class HittingLevel(StoppingTime):
    """单边首次命中停时 τ = inf{n: S_n ≥ level} 或 τ = inf{n: S_n ≤ level}。"""

    def __init__(self, level: float, direction: str = 'up'):
        """
        direction: 'up'   → 向上首次 ≥ level
                   'down' → 向下首次 ≤ level
        """
        if direction not in ('up', 'down'):
            raise ValueError("direction must be 'up' or 'down'")
        self.level = level
        self.direction = direction

    def should_stop(self, state: float, step_idx: int, history=None) -> bool:
        if self.direction == 'up':
            return state >= self.level
        else:
            return state <= self.level

    def supports_batch(self) -> bool:
        return True

    def stop_mask(self, states: np.ndarray, step_indices: np.ndarray) -> np.ndarray:
        if self.direction == 'up':
            return states >= self.level
        return states <= self.level


class ExitInterval(StoppingTime):
    """双边首次逃出停时 τ = inf{n: S_n ≤ lower 或 S_n ≥ upper}。"""

    def __init__(self, lower: float, upper: float):
        if lower >= upper:
            raise ValueError("lower must be less than upper")
        self.lower = lower
        self.upper = upper

    def should_stop(self, state: float, step_idx: int, history=None) -> bool:
        return state <= self.lower or state >= self.upper

    def supports_batch(self) -> bool:
        return True

    def stop_mask(self, states: np.ndarray, step_indices: np.ndarray) -> np.ndarray:
        return (states <= self.lower) | (states >= self.upper)


class Truncated(StoppingTime):
    """截断停时 τ_N = min(τ, N)，装饰另一个停时。"""

    def __init__(self, inner: StoppingTime, max_steps: int):
        if isinstance(max_steps, bool) or not isinstance(max_steps, Integral) or max_steps < 0:
            raise ValueError("max_steps must be a non-negative integer")
        self.inner = inner
        self.max_steps = max_steps

    def should_stop(self, state: float, step_idx: int, history=None) -> bool:
        if step_idx >= self.max_steps:
            return True
        return self.inner.should_stop(state, step_idx, history)

    def supports_batch(self) -> bool:
        return self.inner.supports_batch()

    def stop_mask(self, states: np.ndarray, step_indices: np.ndarray) -> np.ndarray:
        return self.inner.stop_mask(states, step_indices) | (step_indices >= self.max_steps)


