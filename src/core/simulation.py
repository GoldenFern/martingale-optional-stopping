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
            seed: Optional[int] = None) -> SimulationResult:
        """运行 num_paths 条独立路径，每条最多 max_steps 步。

        返回 SimulationResult，其中 paths[:,0] 为初始值。
        """
        if seed is not None:
            np.random.seed(seed)

        paths = np.full((num_paths, max_steps + 1), np.nan)
        stopping_times = np.full(num_paths, max_steps, dtype=int)
        stopped_values = np.empty(num_paths)
        reached_stop = np.zeros(num_paths, dtype=bool)

        for path_idx in range(num_paths):
            self.process.reset(initial_state)
            paths[path_idx, 0] = initial_state
            for step_idx in range(1, max_steps + 1):
                state = self.process.step()
                paths[path_idx, step_idx] = state
                if self.stopping_time.should_stop(state, step_idx, paths[path_idx, :step_idx]):
                    stopping_times[path_idx] = step_idx
                    stopped_values[path_idx] = state
                    reached_stop[path_idx] = True
                    break
            else:
                # 未在 max_steps 内触发
                stopped_values[path_idx] = paths[path_idx, -1]

        return SimulationResult(
            paths=paths,
            stopping_times=stopping_times,
            stopped_values=stopped_values,
            reached_stop=reached_stop,
            config={'initial_state': initial_state, 'num_paths': num_paths, 'max_steps': max_steps},
        )

    def estimate_expectation(self, initial_state: float, num_paths: int, max_steps: int,
                             seed: Optional[int] = None) -> tuple[float, float]:
        """估计 E[X_τ] 及其标准误差（仅使用在 max_steps 内触发停时的路径）。"""
        result = self.run(initial_state, num_paths, max_steps, seed)
        reached_mask = result.reached_stop
        if reached_mask.sum() == 0:
            return float('nan'), float('nan')
        reached_values = result.stopped_values[reached_mask]
        estimated_mean = np.mean(reached_values)
        std_error = np.std(reached_values, ddof=1) / np.sqrt(len(reached_values))
        return estimated_mean, std_error

    def estimate_convergence(self, initial_state: float, path_counts: np.ndarray,
                             max_steps: int, num_repeats: int = 5,
                             seed: Optional[int] = None
                             ) -> tuple[np.ndarray, np.ndarray]:
        """对不同样本量估计 E[X_τ]，返回 (means, ses)。"""
        if seed is not None:
            np.random.seed(seed)

        num_levels = len(path_counts)
        repeated_means = np.empty((num_repeats, num_levels))
        for repeat_idx in range(num_repeats):
            for level_idx, path_count in enumerate(path_counts):
                estimated_mean, _ = self.estimate_expectation(initial_state, int(path_count), max_steps)
                repeated_means[repeat_idx, level_idx] = estimated_mean

        mean_estimates = repeated_means.mean(axis=0)
        std_errors = repeated_means.std(axis=0, ddof=1)
        return mean_estimates, std_errors

    def estimate_tail(self, initial_state: float, num_paths: int, max_steps: int,
                      seed: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """估计停时分布的生存函数 P(τ > t)。"""
        result = self.run(initial_state, num_paths, max_steps, seed)
        time_values = np.arange(0, max_steps + 1)
        # 对于每个 t，count τ > t
        survival_probs = np.array([np.sum(result.stopping_times > time_point) / num_paths
                             for time_point in time_values])
        return time_values, survival_probs
