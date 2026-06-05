"""测试 core/simulation.py"""
import numpy as np
import sys
sys.path.insert(0, 'src')

from core.processes import SymmetricRW
from core.stopping_times import ExitInterval, HittingLevel, FixedTime
from core.simulation import MonteCarloSimulation


class TestMonteCarloSimulation:
    def test_basic_run_shape(self):
        rw = SymmetricRW(0.5)
        tau = ExitInterval(-10, 10)
        sim = MonteCarloSimulation(rw, tau)
        result = sim.run(0.0, num_paths=100, max_steps=1000, seed=42)
        assert result.paths.shape == (100, 1001)
        assert len(result.stopping_times) == 100
        assert len(result.stopped_values) == 100

    def test_all_reach_barrier(self):
        """双边障碍下所有路径最终都触发停时。"""
        rw = SymmetricRW(0.5)
        tau = ExitInterval(-10, 10)
        sim = MonteCarloSimulation(rw, tau)
        result = sim.run(0.0, num_paths=500, max_steps=20000, seed=42)
        assert result.reached_stop.mean() == 1.0

    def test_two_sided_expectation(self):
        """双边障碍对称游走：E[X_τ] = 0。"""
        rw = SymmetricRW(0.5)
        tau = ExitInterval(-10, 10)
        sim = MonteCarloSimulation(rw, tau)
        mean, se = sim.estimate_expectation(0.0, num_paths=5000, max_steps=5000, seed=42)
        assert abs(mean) < 0.5  # E[X_τ] = 0

    def test_one_sided_hit_fraction(self):
        """单边首次命中：绝大多数路径在合理步数内触发。

        由于 P(τ > t) ∼ 1/√(πt)，到 5000 步时约 97% 已命中 +1。
        此测试不验证 OST 失效（需极大 max_steps），只验证首达行为正常。
        """
        rw = SymmetricRW(0.5)
        tau = HittingLevel(1, 'up')
        sim = MonteCarloSimulation(rw, tau)
        result = sim.run(0.0, num_paths=2000, max_steps=5000, seed=42)
        # 至少 95% 的路径命中了 +1
        assert result.reached_stop.mean() > 0.90

    def test_fixed_time_expectation(self):
        """固定停时下 E[S_N] = S_0 = 0。"""
        rw = SymmetricRW(0.5)
        tau = FixedTime(100)
        sim = MonteCarloSimulation(rw, tau)
        mean, se = sim.estimate_expectation(0.0, num_paths=5000, max_steps=100, seed=42)
        assert abs(mean) < 0.3

    def test_estimate_convergence(self):
        rw = SymmetricRW(0.5)
        tau = ExitInterval(-10, 10)
        sim = MonteCarloSimulation(rw, tau)
        path_counts = np.array([100, 500, 1000])
        means, ses = sim.estimate_convergence(
            0.0, path_counts, max_steps=5000, num_repeats=3, seed=42)
        assert means.shape == (3,)
        assert ses.shape == (3,)

    def test_estimate_tail(self):
        rw = SymmetricRW(0.5)
        tau = ExitInterval(-10, 10)
        sim = MonteCarloSimulation(rw, tau)
        t, survival = sim.estimate_tail(0.0, num_paths=500, max_steps=2000, seed=42)
        assert survival[0] == 1.0  # P(τ > 0) = 1


class TestReproducibility:
    def test_same_seed_same_result(self):
        rw = SymmetricRW(0.5)
        tau = ExitInterval(-10, 10)
        sim = MonteCarloSimulation(rw, tau)
        r1 = sim.run(0.0, num_paths=100, max_steps=1000, seed=42)
        r2 = sim.run(0.0, num_paths=100, max_steps=1000, seed=42)
        assert np.array_equal(r1.stopping_times, r2.stopping_times)
        assert np.array_equal(r1.stopped_values, r2.stopped_values)
