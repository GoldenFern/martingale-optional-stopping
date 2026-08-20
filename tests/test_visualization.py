"""Tests for shared plotting helpers."""
import matplotlib.pyplot as plt
import pytest

from core.visualization import save_figure


def test_save_figure_rejects_non_basename():
    fig = plt.figure()
    try:
        with pytest.raises(ValueError):
            save_figure(fig, "../outside.pdf")
    finally:
        plt.close(fig)
