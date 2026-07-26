"""Monte Carlo 模拟引擎"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from .processes import Martingale
from .stopping_times import StoppingTime


@dataclass
class SimulationResult:
    """模拟结果容器。"""
    paths: np.ndarray              # (num_paths, max_steps+1)，含初始值
    stopping_times: np.ndarray     # (num_paths,) 各路径停时索引
    stopped_values: np.ndarray     # (num_paths,) 各路径停时处的过程值
    reached_stop: np.ndarray       # (num_paths,) bool，是否在 max_steps 内触发停时
    config: dict = field(default_factory=dict)


class MonteCarloSimulation:
    """Monte Carlo 模拟引擎。"""

    def __init__(self, process: Martingale, stopping_time: StoppingTime):
        self.process = process
        self.stopping_time = stopping_time

    def run(self, initial_state: float, num_paths: int, max_steps: int,
            seed: Optional[int] = None, store_paths: bool = True) -> SimulationResult:
        """运行 num_paths 条独立路径，每条最多 max_steps 步。

        store_paths=False 时不保存完整路径矩阵（返回的 paths 为空数组），
        避免在 max_steps 很大时分配 O(num_paths × max_steps) 的内存；
        仅估计停时值/停时分布时无需整条路径。随机数消耗与 True 时完全一致。
        """
        if seed is not None:
            np.random.seed(seed)

        paths = np.full((num_paths, max_steps + 1), np.nan) if store_paths else np.empty((0, 0))
        stopping_times = np.full(num_paths, max_steps, dtype=int)
        stopped_values = np.empty(num_paths)
        reached_stop = np.zeros(num_paths, dtype=bool)

        for path_idx in range(num_paths):
            self.process.reset(initial_state)
            if store_paths:
                paths[path_idx, 0] = initial_state
            state = initial_state
            for step_idx in range(1, max_steps + 1):
                state = self.process.step()
                if store_paths:
                    paths[path_idx, step_idx] = state
                history = paths[path_idx, :step_idx] if store_paths else None
                if self.stopping_time.should_stop(state, step_idx, history):
                    stopping_times[path_idx] = step_idx
                    stopped_values[path_idx] = state
                    reached_stop[path_idx] = True
                    break
            else:
                # 未在 max_steps 内触发
                stopped_values[path_idx] = state

        return SimulationResult(
            paths=paths,
            stopping_times=stopping_times,
            stopped_values=stopped_values,
            reached_stop=reached_stop,
            config={'initial_state': initial_state, 'num_paths': num_paths, 'max_steps': max_steps},
        )

    def estimate_expectation(self, initial_state: float, num_paths: int, max_steps: int,
                             seed: Optional[int] = None,
                             include_all: bool = False) -> tuple[float, float]:
        """估计 E[X_τ] 及其标准误差。

        默认仅使用在 max_steps 内触发停时的路径；若 include_all=True，
        则将所有路径（含截断）纳入计算，截断路径的停时值取其当前状态。
        """
        result = self.run(initial_state, num_paths, max_steps, seed, store_paths=False)
        if include_all:
            values = result.stopped_values
        else:
            reached_mask = result.reached_stop
            if reached_mask.sum() == 0:
                return float('nan'), float('nan')
            values = result.stopped_values[reached_mask]
        estimated_mean = np.mean(values)
        std_error = np.std(values, ddof=1) / np.sqrt(len(values))
        return estimated_mean, std_error

    def estimate_tail(self, initial_state: float, num_paths: int, max_steps: int,
                      seed: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """估计停时分布的生存函数 P(τ > t)。"""
        result = self.run(initial_state, num_paths, max_steps, seed, store_paths=False)
        time_values = np.arange(0, max_steps + 1)
        # P(τ > t) = 1 − F(t)：对停时计数做累积和，替代逐 t 计数的 O(T·M) 循环
        tau_counts = np.bincount(result.stopping_times, minlength=max_steps + 1)
        survival_probs = (num_paths - np.cumsum(tau_counts)) / num_paths
        return time_values, survival_probs
