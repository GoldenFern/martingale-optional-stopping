"""停时类族"""
from abc import ABC, abstractmethod
import numpy as np


class StoppingTime(ABC):
    """停时抽象基类。"""

    @abstractmethod
    def should_stop(self, state: float, step_idx: int, history: np.ndarray = None) -> bool:
        """判断在当前状态和步数下是否应停止。"""
        ...


class FixedTime(StoppingTime):
    """固定时刻停时 τ = N。"""

    def __init__(self, stop_step: int):
        self.stop_step = stop_step

    def should_stop(self, state: float, step_idx: int, history=None) -> bool:
        return step_idx >= self.stop_step


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


class ExitInterval(StoppingTime):
    """双边首次逃出停时 τ = inf{n: S_n ≤ lower 或 S_n ≥ upper}。"""

    def __init__(self, lower: float, upper: float):
        self.lower = lower
        self.upper = upper

    def should_stop(self, state: float, step_idx: int, history=None) -> bool:
        return state <= self.lower or state >= self.upper


class Truncated(StoppingTime):
    """截断停时 τ_N = min(τ, N)，装饰另一个停时。"""

    def __init__(self, inner: StoppingTime, max_steps: int):
        self.inner = inner
        self.max_steps = max_steps

    def should_stop(self, state: float, step_idx: int, history=None) -> bool:
        if step_idx >= self.max_steps:
            return True
        return self.inner.should_stop(state, step_idx, history)


class FirstRecord(StoppingTime):
    """"历史最优首次出现"停时——用于秘书问题。

    τ = inf{n ≥ r: X_n > max(X_0, ..., X_{n-1})}
    其中 max_history 由调用方维护，停时判断依赖外部传入的 current_max。
    """

    def __init__(self, threshold: float):
        self.threshold = threshold

    def should_stop(self, state: float, step_idx: int, history=None) -> bool:
        return state > self.threshold
