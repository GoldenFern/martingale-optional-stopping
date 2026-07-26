"""测试 core/stopping_times.py"""
from core.stopping_times import (
    FixedTime, HittingLevel, ExitInterval, Truncated,
)


class TestFixedTime:
    def test_stops_at_N(self):
        ft = FixedTime(10)
        assert not ft.should_stop(0.0, 5)
        assert ft.should_stop(0.0, 10)
        assert ft.should_stop(0.0, 15)


class TestHittingLevel:
    def test_hitting_up(self):
        hl = HittingLevel(5.0, 'up')
        assert not hl.should_stop(4.9, 10)
        assert hl.should_stop(5.0, 10)
        assert hl.should_stop(10.0, 10)

    def test_hitting_down(self):
        hl = HittingLevel(-3.0, 'down')
        assert not hl.should_stop(-2.0, 10)
        assert hl.should_stop(-3.0, 10)
        assert hl.should_stop(-5.0, 10)

    def test_invalid_direction(self):
        import pytest
        with pytest.raises(ValueError):
            HittingLevel(0.0, 'left')


class TestExitInterval:
    def test_inside_interval(self):
        ei = ExitInterval(-5.0, 5.0)
        assert not ei.should_stop(0.0, 10)
        assert not ei.should_stop(-4.9, 10)
        assert not ei.should_stop(4.9, 10)

    def test_exit_on_boundary(self):
        ei = ExitInterval(-5.0, 5.0)
        assert ei.should_stop(-5.0, 10)
        assert ei.should_stop(5.0, 10)
        assert ei.should_stop(10.0, 10)
        assert ei.should_stop(-10.0, 10)


class TestTruncated:
    def test_truncation_overrides(self):
        inner = HittingLevel(100.0, 'up')  # 几乎永远不会触发
        trunc = Truncated(inner, 50)
        assert not trunc.should_stop(0.0, 30)
        assert trunc.should_stop(0.0, 50)

    def test_inner_stops_first(self):
        inner = HittingLevel(5.0, 'up')
        trunc = Truncated(inner, 50)
        assert trunc.should_stop(5.0, 10)


