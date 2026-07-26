"""测试 core/processes.py"""
import numpy as np
import pytest

from core.processes import SymmetricRW


class TestSymmetricRW:
    def test_initial_state(self):
        rw = SymmetricRW(0.5)
        rw.reset(10.0)
        assert rw._state == 10.0

    def test_martingale_property(self):
        """大量模拟验证 E[S_n] = S_0（鞅性质）。"""
        np.random.seed(42)
        rw = SymmetricRW(0.5)
        n_paths = 20000
        n_steps = 100
        final_vals = np.empty(n_paths)
        for i in range(n_paths):
            rw.reset(0.0)
            for _ in range(n_steps):
                rw.step()
            final_vals[i] = rw._state
        # E[S_n] 应在 0 附近，标准误差 = σ√n / √M；5σ 容差避免偶发失败
        se = np.sqrt(n_steps) / np.sqrt(n_paths)  # σ=1 for ±1 steps
        assert abs(np.mean(final_vals)) < 5 * se

    def test_variance_growth(self):
        """验证 Var(S_n) = n（独立增量方差可加）。"""
        np.random.seed(42)
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
        # 方差估计的标准差约 sqrt(2*50^2/20000) ~ 0.5，取 5σ 容差
        assert abs(var_est - n_steps) < 2.5

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


