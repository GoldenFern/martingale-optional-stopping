"""测试 core/simulation.py"""
import numpy as np

from core.processes import SymmetricRW
from core.stopping_times import ExitInterval, HittingLevel, FixedTime, Truncated
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

    def test_estimate_tail(self):
        rw = SymmetricRW(0.5)
        tau = ExitInterval(-10, 10)
        sim = MonteCarloSimulation(rw, tau)
        t, survival = sim.estimate_tail(0.0, num_paths=500, max_steps=2000, seed=42)
        assert survival[0] == 1.0  # P(τ > 0) = 1


class TestBatchedFastPath:
    """向量化快速路径与逐路径循环的统计等价性。"""

    def test_matches_scalar_statistics(self):
        """双边障碍：吸收概率与平均停时应在 5σ 内一致。"""
        rw = SymmetricRW(0.5)
        tau = ExitInterval(-10, 10)
        sim = MonteCarloSimulation(rw, tau)
        # store_paths=True 强制走逐路径循环，False 走向量化路径
        res_scalar = sim.run(0.0, num_paths=3000, max_steps=5000, seed=42, store_paths=True)
        res_batched = sim.run(0.0, num_paths=3000, max_steps=5000, seed=43, store_paths=False)
        assert res_batched.reached_stop.mean() == 1.0
        # 停时值只能落在两个障碍上
        assert set(np.unique(res_batched.stopped_values)) <= {-10.0, 10.0}
        p_scalar = (res_scalar.stopped_values == 10.0).mean()
        p_batched = (res_batched.stopped_values == 10.0).mean()
        se = np.sqrt(0.5 * 0.5 / 3000) * np.sqrt(2)
        assert abs(p_scalar - p_batched) < 5 * se
        tau_se = res_scalar.stopping_times.std() / np.sqrt(3000) * np.sqrt(2)
        assert abs(res_scalar.stopping_times.mean() - res_batched.stopping_times.mean()) < 5 * tau_se

    def test_truncated_all_stop(self):
        """截断停时：向量化路径同样保证全部在 N 步内停止。"""
        rw = SymmetricRW(0.5)
        trunc = Truncated(HittingLevel(10, 'up'), 500)
        sim = MonteCarloSimulation(rw, trunc)
        result = sim.run(0.0, num_paths=1000, max_steps=500, seed=42, store_paths=False)
        assert result.reached_stop.mean() == 1.0
        assert np.all(result.stopping_times <= 500)
        # 命中 +10 的路径停在 10，其余停在截断步
        hit10 = result.stopped_values == 10.0
        assert np.all(result.stopping_times[~hit10] == 500)


class TestReproducibility:
    def test_same_seed_same_result(self):
        rw = SymmetricRW(0.5)
        tau = ExitInterval(-10, 10)
        sim = MonteCarloSimulation(rw, tau)
        r1 = sim.run(0.0, num_paths=100, max_steps=1000, seed=42)
        r2 = sim.run(0.0, num_paths=100, max_steps=1000, seed=42)
        assert np.array_equal(r1.stopping_times, r2.stopping_times)
        assert np.array_equal(r1.stopped_values, r2.stopped_values)
