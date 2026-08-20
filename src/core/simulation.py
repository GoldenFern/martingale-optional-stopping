"""Monte Carlo 模拟引擎"""
from numbers import Integral
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
        仅估计停时值/停时分布时无需整条路径。批量路径与逐路径路径
        分布等价，但会以不同顺序消耗随机数；相同 seed 只保证同一种
        执行模式内的可复现性。
        """
        self._validate_run_arguments(num_paths, max_steps)
        if seed is not None:
            np.random.seed(seed)

        if (not store_paths and hasattr(self.process, 'batch_increments')
                and self.stopping_time.supports_batch()):
            return self._run_batched(initial_state, num_paths, max_steps)

        paths = np.full((num_paths, max_steps + 1), np.nan) if store_paths else np.empty((0, 0))
        stopping_times = np.full(num_paths, max_steps, dtype=int)
        stopped_values = np.empty(num_paths)
        reached_stop = np.zeros(num_paths, dtype=bool)

        for path_idx in range(num_paths):
            self.process.reset(initial_state)
            if store_paths:
                paths[path_idx, 0] = initial_state
            state = initial_state
            history = paths[path_idx, :1] if store_paths else None
            if self.stopping_time.should_stop(state, 0, history):
                stopping_times[path_idx] = 0
                stopped_values[path_idx] = state
                reached_stop[path_idx] = True
                continue
            for step_idx in range(1, max_steps + 1):
                state = self.process.step()
                if store_paths:
                    paths[path_idx, step_idx] = state
                history = paths[path_idx, :step_idx + 1] if store_paths else None
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

    def _run_batched(self, initial_state: float, num_paths: int, max_steps: int) -> SimulationResult:
        """向量化快速路径：跨路径批量推进，分块生成增量以限制内存。

        语义与逐路径循环一致（包括第 0 步判停），但随机数按路径×块
        的顺序消耗，与逐路径版本的抽样顺序不同。
        """
        stopping_times = np.full(num_paths, max_steps, dtype=int)
        stopped_values = np.empty(num_paths)
        reached_stop = np.zeros(num_paths, dtype=bool)
        states = np.full(num_paths, float(initial_state))
        initial_mask = self.stopping_time.stop_mask(
            states[:, None], np.zeros((1, 1), dtype=int)
        )[:, 0]
        stopping_times[initial_mask] = 0
        stopped_values[initial_mask] = states[initial_mask]
        reached_stop[initial_mask] = True
        active = ~initial_mask

        # 每块约 4e6 个元素，控制临时矩阵在几十 MB 量级
        chunk_steps = max(1, min(max_steps, 4_000_000 // max(num_paths, 1)))
        step_base = 0
        while step_base < max_steps and active.any():
            n_steps = min(chunk_steps, max_steps - step_base)
            idx = np.flatnonzero(active)
            increments = self.process.batch_increments(len(idx), n_steps)
            cum_states = states[idx, None] + np.cumsum(increments, axis=1)
            step_indices = np.arange(step_base + 1, step_base + n_steps + 1)[None, :]
            mask = self.stopping_time.stop_mask(cum_states, step_indices)

            hit_any = mask.any(axis=1)
            first_hit = np.argmax(mask, axis=1)
            rows_hit = idx[hit_any]
            stopping_times[rows_hit] = step_base + 1 + first_hit[hit_any]
            stopped_values[rows_hit] = cum_states[hit_any, first_hit[hit_any]]
            reached_stop[rows_hit] = True
            active[rows_hit] = False

            states[idx[~hit_any]] = cum_states[~hit_any, -1]
            step_base += n_steps

        stopped_values[~reached_stop] = states[~reached_stop]
        return SimulationResult(
            paths=np.empty((0, 0)),
            stopping_times=stopping_times,
            stopped_values=stopped_values,
            reached_stop=reached_stop,
            config={'initial_state': initial_state, 'num_paths': num_paths, 'max_steps': max_steps},
        )

    @staticmethod
    def _validate_run_arguments(num_paths: int, max_steps: int) -> None:
        if isinstance(num_paths, bool) or not isinstance(num_paths, Integral) or num_paths <= 0:
            raise ValueError("num_paths must be a positive integer")
        if isinstance(max_steps, bool) or not isinstance(max_steps, Integral) or max_steps < 0:
            raise ValueError("max_steps must be a non-negative integer")

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
        """估计停时分布的生存函数 P(τ > t)。

        未在观察窗口内触发的路径是右删失样本，故在 t=max_steps 时仍
        计入 P(τ > t)，而不会被误认为恰在 max_steps 停止。
        """
        result = self.run(initial_state, num_paths, max_steps, seed, store_paths=False)
        time_values = np.arange(0, max_steps + 1)
        # P(τ > t) = 1 − F(t)：对停时计数做累积和，替代逐 t 计数的 O(T·M) 循环
        tau_counts = np.bincount(
            result.stopping_times[result.reached_stop], minlength=max_steps + 1
        )
        survival_probs = (num_paths - np.cumsum(tau_counts)) / num_paths
        return time_values, survival_probs
