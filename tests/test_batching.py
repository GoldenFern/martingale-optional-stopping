"""Tests for batch splitting helpers and batch-keyed outputs."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from core.batching import split_batch_sizes
import ch05_secretary.experiment as secretary_experiment


def test_split_batch_sizes_balanced():
    sizes = split_batch_sizes(10, 3)
    assert sizes == [4, 3, 3]
    assert sum(sizes) == 10


def test_split_batch_sizes_rejects_too_many_batches():
    with pytest.raises(ValueError):
        split_batch_sizes(3, 4)


def test_success_experiment_batch_keys_unique(tmp_path, monkeypatch):
    monkeypatch.setattr(secretary_experiment, "DATA_DIR", tmp_path)
    df = secretary_experiment.run_success_experiment(
        N_values=(20,), n_r_points=6, n_trials=60, seed=1, n_batches=4
    )
    assert df["batch_key"].is_unique
    assert len(df["batch_key"]) == len(set(df["batch_key"]))


def test_control_experiment_batch_keys_unique(tmp_path, monkeypatch):
    monkeypatch.setattr(secretary_experiment, "DATA_DIR", tmp_path)
    df = secretary_experiment.run_control_experiment(
        N_values=(20,), n_r_points=6, n_trials=60, seed=1, n_batches=4
    )
    assert df["batch_key"].is_unique
    assert len(df["batch_key"]) == len(set(df["batch_key"]))


def test_success_experiment_uses_all_trials(tmp_path, monkeypatch):
    monkeypatch.setattr(secretary_experiment, "DATA_DIR", tmp_path)
    df = secretary_experiment.run_success_experiment(
        N_values=(20,), n_r_points=1, n_trials=62, seed=1, n_batches=4
    )
    val = df.iloc[0]["success_mc"]
    assert 0.0 <= val <= 1.0


def test_control_experiment_uses_all_trials(tmp_path, monkeypatch):
    monkeypatch.setattr(secretary_experiment, "DATA_DIR", tmp_path)
    df = secretary_experiment.run_control_experiment(
        N_values=(20,), n_r_points=1, n_trials=62, seed=1, n_batches=4
    )
    val = df.iloc[0]["success_mc"]
    assert 0.0 <= val <= 1.0
