"""测试 ch03_insurance/surplus_model.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
import pytest
from scipy.stats import expon
from ch03_insurance.surplus_model import (
    SurplusProcess, find_adjustment_R, exp_claim_mgf_factory,
    lundberg_psi_exact, psi_exact_exponential,
)


class TestSurplusProcess:
    def test_initial_state(self):
        claim_distr = expon(scale=1.0)
        proc = SurplusProcess(initial_capital=10.0, premium_rate=1.5, claim_intensity=1.0, claim_distribution=claim_distr)
        assert proc.surplus == 10.0
        assert proc.current_time == 0.0

    def test_reset(self):
        claim_distr = expon(scale=1.0)
        proc = SurplusProcess(initial_capital=10.0, premium_rate=1.5, claim_intensity=1.0, claim_distribution=claim_distr)
        proc.step_until_next_claim()
        proc.reset(20.0)
        assert proc.surplus == 20.0
        assert proc.current_time == 0.0

    def test_step_advances_time(self):
        claim_distr = expon(scale=1.0)
        proc = SurplusProcess(initial_capital=10.0, premium_rate=1.5, claim_intensity=1.0, claim_distribution=claim_distr)
        time_before = proc.current_time
        proc.step_until_next_claim()
        assert proc.current_time > time_before

    def test_simulate_path_shape(self):
        claim_distr = expon(scale=1.0)
        proc = SurplusProcess(initial_capital=10.0, premium_rate=1.5, claim_intensity=1.0, claim_distribution=claim_distr)
        times, values = proc.simulate_path(time_horizon=20.0)
        assert len(times) >= 2
        assert len(times) == len(values)
        assert times[0] == 0.0
        assert values[0] == 10.0

    def test_simulate_until_ruin_or_T(self):
        claim_distr = expon(scale=0.1)  # 小索赔 → 不容易破产
        proc = SurplusProcess(initial_capital=100.0, premium_rate=2.0, claim_intensity=1.0, claim_distribution=claim_distr)
        ruined, ruin_time, traj_times, traj_surplus = proc.simulate_until_ruin_or_T(time_horizon=50.0)
        # 高初始资本 + 小索赔 → 大概率不破产
        assert ruin_time > 0


class TestAdjustmentR:
    def test_known_exponential(self):
        """指数索赔 M_Y(r)=1/(1-μr)，调整系数 R = θ/(μ(1+θ))。"""
        claim_intensity, mean_claim, safety_loading = 1.0, 0.5, 0.3
        premium_rate = claim_intensity * mean_claim * (1 + safety_loading)
        mgf = exp_claim_mgf_factory('exponential', rate=1/mean_claim)
        adj_coefficient = find_adjustment_R(claim_intensity, premium_rate, mgf)
        expected_R = safety_loading / (mean_claim * (1 + safety_loading))
        assert adj_coefficient == pytest.approx(expected_R, rel=1e-3)

    def test_R_positive_with_safety_loading(self):
        """安全负荷正时 R > 0。"""
        claim_intensity, mean_claim, safety_loading = 1.0, 1.0, 0.5
        premium_rate = claim_intensity * mean_claim * (1 + safety_loading)
        mgf = exp_claim_mgf_factory('exponential', rate=1/mean_claim)
        adj_coefficient = find_adjustment_R(claim_intensity, premium_rate, mgf)
        assert adj_coefficient > 0

    def test_R_larger_for_larger_theta(self):
        """安全负荷越大，调整系数越大。"""
        mean_claim = 1.0
        mgf = exp_claim_mgf_factory('exponential', rate=1/mean_claim)
        adj_low = find_adjustment_R(1.0, 1.0 * mean_claim * 1.1, mgf)
        adj_high = find_adjustment_R(1.0, 1.0 * mean_claim * 1.5, mgf)
        assert adj_high > adj_low


class TestLundberg:
    def test_psi_at_zero(self):
        assert lundberg_psi_exact(0.0, 0.5) == 1.0

    def test_psi_decreasing(self):
        assert lundberg_psi_exact(10.0, 0.5) < lundberg_psi_exact(5.0, 0.5)


class TestPsiExponential:
    def test_psi_at_zero(self):
        psi = psi_exact_exponential(0.0, claim_intensity=1.0, premium_rate=1.5, mean_claim=1.0)
        assert pytest.approx(psi, abs=1e-4) == 1.0 / (1.0 + 0.5)  # 1/(1+θ)

    def test_psi_decreasing_in_u(self):
        psi_low = psi_exact_exponential(5.0, claim_intensity=1.0, premium_rate=1.5, mean_claim=1.0)
        psi_high = psi_exact_exponential(10.0, claim_intensity=1.0, premium_rate=1.5, mean_claim=1.0)
        assert psi_high < psi_low

    def test_ruin_certain_without_safety_loading(self):
        psi = psi_exact_exponential(10.0, claim_intensity=1.0, premium_rate=0.9, mean_claim=1.0)
        assert psi == 1.0  # c <= λμ → 必然破产


class TestMGF:
    def test_exponential_mgf_at_zero(self):
        mgf = exp_claim_mgf_factory('exponential', rate=0.5)
        assert mgf(0.0) == 1.0

    def test_exponential_mgf_blows_up(self):
        mgf = exp_claim_mgf_factory('exponential', rate=2.0)
        assert mgf(3.0) == np.inf
