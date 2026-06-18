"""Tests for batch splitting helpers and batch-keyed outputs."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from core.batching import split_batch_sizes


def test_split_batch_sizes_balanced():
    sizes = split_batch_sizes(10, 3)
    assert sizes == [4, 3, 3]
    assert sum(sizes) == 10


def test_split_batch_sizes_rejects_too_many_batches():
    with pytest.raises(ValueError):
        split_batch_sizes(3, 4)


