"""测试 ch02_moran/moran_model.py"""
import numpy as np
import pytest

from ch02_moran.moran_model import MoranProcess, expected_tau_moran


class TestMoranProcess:
    def test_initialization(self):
        mp = MoranProcess(50)
        mp.reset(25)
        assert mp.state == 25

    def test_step_bounds(self):
        mp = MoranProcess(50)
        mp.reset(25)
        for _ in range(1000):
            s = mp.step()
            assert 0 <= s <= 50
            # 每步最多变化 1
            # (不测这个因为 step 不返回旧状态)

    def test_martingale_property(self):
        """验证 E[X_{n+1} | X_n=i] = i。"""
        np.random.seed(42)
        N = 50
        mp = MoranProcess(N)
        n_trials = 5000
        for i in [10, 25, 40]:
            mp.reset(i)
            next_vals = np.empty(n_trials)
            for k in range(n_trials):
                mp._allele_count = i  # 直接设回 i
                next_vals[k] = mp.step()
            assert np.mean(next_vals) == pytest.approx(i, abs=0.2)

    def test_absorbing(self):
        mp = MoranProcess(50)
        mp.reset(0)
        path = mp.simulate_path()
        assert path[-1] == 0
        assert len(path) == 1  # 立即停下

        mp.reset(50)
        path = mp.simulate_path()
        assert path[-1] == 50
        assert len(path) == 1


class TestBatchAbsorption:
    def test_fixation_prob_matches_theory(self):
        """批量模拟：固定概率 ≈ x0/N（5σ 容差）。"""
        np.random.seed(42)
        mp = MoranProcess(20)
        tau, fixed = mp.simulate_batch_until_absorption(5, 4000)
        se = np.sqrt(0.25 * 0.75 / 4000)
        assert abs(fixed.mean() - 0.25) < 5 * se
        assert np.all(tau > 0)

    def test_initially_absorbed(self):
        mp = MoranProcess(20)
        tau0, fixed0 = mp.simulate_batch_until_absorption(0, 10)
        assert np.all(tau0 == 0) and not fixed0.any()
        tau_n, fixed_n = mp.simulate_batch_until_absorption(20, 10)
        assert np.all(tau_n == 0) and fixed_n.all()


class TestExpectedTau:
    def test_shape(self):
        tau = expected_tau_moran(50)
        assert len(tau) == 51
        assert tau[0] == 0.0
        assert tau[50] == 0.0

    def test_symmetry(self):
        """从 i 和从 N-i 出发的期望停时应相等。"""
        N = 50
        tau = expected_tau_moran(N)
        assert tau[10] == pytest.approx(tau[40], rel=0.01)

    def test_all_positive_interior(self):
        tau = expected_tau_moran(50)
        assert np.all(tau[1:50] > 0)
