"""测试第四章：赌徒破产 —— OST 条件失效"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.processes import SymmetricRW
from core.stopping_times import ExitInterval, HittingLevel, Truncated
from core.simulation import MonteCarloSimulation

LOWER_BARRIER = 20
UPPER_BARRIER = 10


class TestTwoSidedExit:
    """双边障碍下 OST 成立。"""

    def test_all_paths_reach_barrier(self):
        """双边障碍下所有路径最终触及吸收边界。"""
        random_walk = SymmetricRW(0.5)
        exit_stop = ExitInterval(-LOWER_BARRIER, UPPER_BARRIER)
        sim = MonteCarloSimulation(random_walk, exit_stop)
        result = sim.run(0.0, num_paths=200, max_steps=20000, seed=42)
        assert result.reached_stop.mean() == 1.0

    def test_expectation_zero(self):
        """双边障碍下 E[S_τ] = 0（OST 成立）。"""
        random_walk = SymmetricRW(0.5)
        exit_stop = ExitInterval(-LOWER_BARRIER, UPPER_BARRIER)
        sim = MonteCarloSimulation(random_walk, exit_stop)
        estimated_mean, _ = sim.estimate_expectation(
            0.0, num_paths=3000, max_steps=10000, seed=42
        )
        # E[S_τ] = 0 by OST; allowed tolerance for MC noise
        assert abs(estimated_mean) < 0.5

    def test_expected_absorption_time_near_ab(self):
        """验证 E[τ] ≈ a·b = 200（双边对称游走）。"""
        random_walk = SymmetricRW(0.5)
        exit_stop = ExitInterval(-LOWER_BARRIER, UPPER_BARRIER)
        sim = MonteCarloSimulation(random_walk, exit_stop)
        result = sim.run(0.0, num_paths=3000, max_steps=20000, seed=42)
        mean_tau = result.stopping_times.mean()
        # 理论值 a*b = 20*10 = 200，允许 MC 误差
        assert mean_tau == pytest.approx(200.0, rel=0.15)

    def test_absorption_probabilities(self):
        """验证吸收概率 P(上界) = a/(a+b) = 20/30 ≈ 0.667。"""
        random_walk = SymmetricRW(0.5)
        exit_stop = ExitInterval(-LOWER_BARRIER, UPPER_BARRIER)
        sim = MonteCarloSimulation(random_walk, exit_stop)
        result = sim.run(0.0, num_paths=3000, max_steps=20000, seed=42)
        prob_up = (result.stopped_values == UPPER_BARRIER).mean()
        # 理论值 a/(a+b) = 20/30 = 2/3
        assert prob_up == pytest.approx(2.0 / 3.0, abs=0.04)


class TestOneSidedHitting:
    """单边障碍下 OST 条件失效。"""

    def test_all_reached_paths_hit_barrier(self):
        """所有触发停时的路径都在 +b 处停止。"""
        random_walk = SymmetricRW(0.5)
        hitting_stop = HittingLevel(UPPER_BARRIER, "up")
        sim = MonteCarloSimulation(random_walk, hitting_stop)
        result = sim.run(0.0, num_paths=500, max_steps=20000, seed=42)
        reached = result.reached_stop
        # 所有已触发停时的路径，停时值应恰好为 +b
        assert np.all(result.stopped_values[reached] == UPPER_BARRIER)

    def test_expectation_equals_barrier_for_reached(self):
        """停时触发路径的 E[S_τ] = b ≠ 0 = E[S_0]（OST 失效）。"""
        random_walk = SymmetricRW(0.5)
        hitting_stop = HittingLevel(UPPER_BARRIER, "up")
        sim = MonteCarloSimulation(random_walk, hitting_stop)
        estimated_mean, _ = sim.estimate_expectation(
            0.0, num_paths=5000, max_steps=5000, seed=42
        )
        # 只对到达的路径取平均，S_τ ≡ b = 10，与 0 差距明显
        assert estimated_mean == pytest.approx(float(UPPER_BARRIER), abs=0.1)

    def test_most_paths_hit_within_moderate_steps(self):
        """绝大多数路径在有限步内命中上界。"""
        random_walk = SymmetricRW(0.5)
        hitting_stop = HittingLevel(UPPER_BARRIER, "up")
        sim = MonteCarloSimulation(random_walk, hitting_stop)
        result = sim.run(0.0, num_paths=1000, max_steps=5000, seed=42)
        # 绝大多数命中（理论值 P(τ_b < 5000) ≈ 0.92，留足 MC 误差余量）
        assert result.reached_stop.mean() > 0.85


class TestTruncation:
    """截断停时保持 OST 成立。"""

    def test_truncation_preserves_ost(self):
        """对任意截断 N，E[S_{τ∧N}] = 0（有界停时 → OST 条件1 满足）。"""
        random_walk = SymmetricRW(0.5)
        hitting_stop = HittingLevel(UPPER_BARRIER, "up")
        for truncation_limit in [50, 200, 1000]:
            truncated_stop = Truncated(hitting_stop, truncation_limit)
            sim = MonteCarloSimulation(random_walk, truncated_stop)
            estimated_mean, _ = sim.estimate_expectation(
                0.0, num_paths=2000, max_steps=truncation_limit, seed=42
            )
            assert abs(estimated_mean) < 1.0, (
                f"E[S_{{τ∧{truncation_limit}}}] = {estimated_mean:.3f}, expected ≈ 0"
            )

    def test_truncation_all_reach_by_construction(self):
        """截断停时保证所有路径在 N 步内停止。"""
        random_walk = SymmetricRW(0.5)
        hitting_stop = HittingLevel(UPPER_BARRIER, "up")
        truncated_stop = Truncated(hitting_stop, 100)
        sim = MonteCarloSimulation(random_walk, truncated_stop)
        result = sim.run(0.0, num_paths=200, max_steps=100, seed=42)
        assert result.reached_stop.mean() == 1.0
        assert np.all(result.stopping_times <= 100)


class TestTailEstimation:
    """停时尾部分布的 Monte Carlo 估计。"""

    def test_survival_starts_at_one(self):
        """P(τ > 0) = 1。"""
        random_walk = SymmetricRW(0.5)
        exit_stop = ExitInterval(-LOWER_BARRIER, UPPER_BARRIER)
        sim = MonteCarloSimulation(random_walk, exit_stop)
        time_values, survival_probs = sim.estimate_tail(
            0.0, num_paths=300, max_steps=1000, seed=42
        )
        assert survival_probs[0] == 1.0

    def test_survival_non_increasing(self):
        """生存函数 P(τ > t) 关于 t 非增。"""
        random_walk = SymmetricRW(0.5)
        exit_stop = ExitInterval(-LOWER_BARRIER, UPPER_BARRIER)
        sim = MonteCarloSimulation(random_walk, exit_stop)
        _, survival_probs = sim.estimate_tail(
            0.0, num_paths=300, max_steps=1000, seed=42
        )
        assert np.all(np.diff(survival_probs) <= 0)

    def test_two_sided_lighter_tail_than_one_sided(self):
        """双边停时尾部衰减远快于单边（轻尾 vs 重尾）。"""
        random_walk = SymmetricRW(0.5)
        exit_stop = ExitInterval(-LOWER_BARRIER, UPPER_BARRIER)
        hitting_stop = HittingLevel(UPPER_BARRIER, "up")

        sim_two = MonteCarloSimulation(random_walk, exit_stop)
        sim_one = MonteCarloSimulation(random_walk, hitting_stop)

        _, surv_two = sim_two.estimate_tail(0.0, num_paths=1000, max_steps=3000, seed=42)
        _, surv_one = sim_one.estimate_tail(0.0, num_paths=1000, max_steps=3000, seed=42)

        # 在 t=2000 处，双边停时未触发的比例应远低于单边
        t_check = min(2000, len(surv_two) - 1, len(surv_one) - 1)
        assert surv_two[t_check] < surv_one[t_check], (
            f"Two-sided survival {surv_two[t_check]:.4f} should be < "
            f"one-sided {surv_one[t_check]:.4f} at t={t_check}"
        )


class TestReproducibility:
    """随机种子确定性。"""

    def test_same_seed_same_result(self):
        random_walk = SymmetricRW(0.5)
        exit_stop = ExitInterval(-LOWER_BARRIER, UPPER_BARRIER)
        sim = MonteCarloSimulation(random_walk, exit_stop)
        result1 = sim.run(0.0, num_paths=100, max_steps=1000, seed=123)
        result2 = sim.run(0.0, num_paths=100, max_steps=1000, seed=123)
        assert np.array_equal(result1.stopping_times, result2.stopping_times)
        assert np.array_equal(result1.stopped_values, result2.stopped_values)
