"""Monte Carlo 模拟引擎"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from .processes import Martingale
from .stopping_times import StoppingTime


@dataclass
class SimulationResult:
    """模拟结果容器。"""
    paths: np.ndarray              # (n_paths, max_steps+1)，含初始值
    stopping_times: np.ndarray     # (n_paths,) 各路径停时索引
    stopped_values: np.ndarray     # (n_paths,) 各路径停时处的过程值
    reached_stop: np.ndarray       # (n_paths,) bool，是否在 max_steps 内触发停时
    config: dict = field(default_factory=dict)


class MonteCarloSimulation:
    """Monte Carlo 模拟引擎。"""

    def __init__(self, process: Martingale, stopping_time: StoppingTime):
        self.process = process
        self.stopping_time = stopping_time

    def run(self, x0: float, n_paths: int, max_steps: int,
            seed: Optional[int] = None) -> SimulationResult:
        """运行 n_paths 条独立路径，每条最多 max_steps 步。

        返回 SimulationResult，其中 paths[:,0] 为初始值 x0。
        """
        if seed is not None:
            np.random.seed(seed)

        paths = np.full((n_paths, max_steps + 1), np.nan)
        stopping_times = np.full(n_paths, max_steps, dtype=int)
        stopped_values = np.empty(n_paths)
        reached_stop = np.zeros(n_paths, dtype=bool)

        for i in range(n_paths):
            self.process.reset(x0)
            paths[i, 0] = x0
            for t in range(1, max_steps + 1):
                state = self.process.step()
                paths[i, t] = state
                if self.stopping_time.should_stop(state, t, paths[i, :t]):
                    stopping_times[i] = t
                    stopped_values[i] = state
                    reached_stop[i] = True
                    break
            else:
                # 未在 max_steps 内触发
                stopped_values[i] = paths[i, -1]

        return SimulationResult(
            paths=paths,
            stopping_times=stopping_times,
            stopped_values=stopped_values,
            reached_stop=reached_stop,
            config={'x0': x0, 'n_paths': n_paths, 'max_steps': max_steps},
        )

    def estimate_expectation(self, x0: float, n_paths: int, max_steps: int,
                             seed: Optional[int] = None) -> tuple[float, float]:
        """估计 E[X_τ] 及其标准误差（仅使用在 max_steps 内触发停时的路径）。"""
        result = self.run(x0, n_paths, max_steps, seed)
        reached = result.reached_stop
        if reached.sum() == 0:
            return float('nan'), float('nan')
        vals = result.stopped_values[reached]
        mean = np.mean(vals)
        se = np.std(vals, ddof=1) / np.sqrt(len(vals))
        return mean, se

    def estimate_convergence(self, x0: float, path_counts: np.ndarray,
                             max_steps: int, n_repeats: int = 5,
                             seed: Optional[int] = None
                             ) -> tuple[np.ndarray, np.ndarray]:
        """对不同样本量估计 E[X_τ]，返回 (means, ses)。"""
        if seed is not None:
            np.random.seed(seed)

        n_levels = len(path_counts)
        all_means = np.empty((n_repeats, n_levels))
        for r in range(n_repeats):
            for j, M in enumerate(path_counts):
                mean, _ = self.estimate_expectation(x0, int(M), max_steps)
                all_means[r, j] = mean

        means = all_means.mean(axis=0)
        ses = all_means.std(axis=0, ddof=1)
        return means, ses

    def estimate_tail(self, x0: float, n_paths: int, max_steps: int,
                      seed: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """估计停时分布的生存函数 P(τ > t)。"""
        result = self.run(x0, n_paths, max_steps, seed)
        t_values = np.arange(0, max_steps + 1)
        # 对于每个 t，count τ > t
        survival = np.array([np.sum(result.stopping_times > t) / n_paths
                             for t in t_values])
        return t_values, survival
