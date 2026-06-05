"""测试 core/processes.py"""
import numpy as np
import pytest
import sys
sys.path.insert(0, 'src')

from core.processes import SymmetricRW, AsymmetricRW, GeometricBrownianMotion


class TestSymmetricRW:
    def test_initial_state(self):
        rw = SymmetricRW(0.5)
        rw.reset(10.0)
        assert rw._state == 10.0

    def test_martingale_property(self):
        """大量模拟验证 E[S_n] = S_0（鞅性质）。"""
        rw = SymmetricRW(0.5)
        n_paths = 20000
        n_steps = 100
        final_vals = np.empty(n_paths)
        for i in range(n_paths):
            rw.reset(0.0)
            for _ in range(n_steps):
                rw.step()
            final_vals[i] = rw._state
        # E[S_n] 应在 0 附近，标准误差 = σ√n / √M
        se = np.sqrt(n_steps) / np.sqrt(n_paths)  # σ=1 for ±1 steps
        assert abs(np.mean(final_vals)) < 3 * se

    def test_variance_growth(self):
        """验证 Var(S_n) = n（独立增量方差可加）。"""
        rw = SymmetricRW(0.5)
        n_paths = 20000
        n_steps = 50
        final_vals = np.empty(n_paths)
        for i in range(n_paths):
            rw.reset(0.0)
            for _ in range(n_steps):
                rw.step()
            final_vals[i] = rw._state
        var_est = np.var(final_vals, ddof=0)
        # 标准差约 sqrt(2*50^2/20000) ~ 0.5
        assert abs(var_est - n_steps) < 1.5

    def test_p_0_5_only(self):
        with pytest.raises(ValueError):
            SymmetricRW(0.6)

    def test_simulate_path_shape(self):
        rw = SymmetricRW(0.5)
        rw.reset(0.0)
        path = rw.simulate_path(100)
        assert path.shape == (100,)
        # 重置后初始值正确
        assert rw._state == path[-1]  # 最后一步状态


class TestAsymmetricRW:
    def test_step_updates_counter(self):
        rw = AsymmetricRW(0.3)
        rw.reset(0.0)
        rw.step()
        assert rw.step_count == 1

    def test_wald_martingale_symmetric(self):
        """p=0.5 时 Wald 鞅恒为 1。"""
        rw = AsymmetricRW(0.5)
        rw.reset(5.0)
        for _ in range(100):
            rw.step()
        assert rw.wald_value() == pytest.approx(1.0)

    def test_wald_martingale_expectation(self):
        """Wald 鞅的期望应为 1。减少步数以控制方差。"""
        rw = AsymmetricRW(0.4)
        n_paths = 100000
        n_steps = 10
        wald_vals = np.empty(n_paths)
        for i in range(n_paths):
            rw.reset(0.0)
            for _ in range(n_steps):
                rw.step()
            wald_vals[i] = rw.wald_value()
        assert np.mean(wald_vals) == pytest.approx(1.0, abs=0.05)

    def test_compensated_martingale(self):
        """补偿游走 S_n − (2p−1)n 应为鞅，期望 0。"""
        p = 0.4
        rw = AsymmetricRW(p)
        n_paths = 20000
        n_steps = 100
        vals = np.empty(n_paths)
        for i in range(n_paths):
            rw.reset(0.0)
            for _ in range(n_steps):
                rw.step()
            vals[i] = rw.compensated_value()
        assert np.mean(vals) == pytest.approx(0.0, abs=0.2)

    def test_p_validation(self):
        with pytest.raises(ValueError):
            AsymmetricRW(0.0)
        with pytest.raises(ValueError):
            AsymmetricRW(1.0)


class TestGBM:
    def test_positive_prices(self):
        gbm = GeometricBrownianMotion(risk_free_rate=0.05, volatility=0.2)
        gbm.reset(100.0)
        path = gbm.simulate_path(1.0, 252)
        assert np.all(path > 0)

    def test_path_shape(self):
        gbm = GeometricBrownianMotion(risk_free_rate=0.05, volatility=0.2)
        gbm.reset(100.0)
        path = gbm.simulate_path(1.0, 100)
        assert len(path) == 101  # N+1 含初始值
