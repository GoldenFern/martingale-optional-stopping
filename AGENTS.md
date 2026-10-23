# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## Project overview

This is a course project (上海交通大学 MATH4704 随机过程) studying the Optional Stopping Theorem (OST) through Monte Carlo simulation. Three models form the core paper: Moran genetic drift (ch02), Cramér-Lundberg insurance ruin (ch03), and gambler's ruin (ch04).

## Commands

```bash
# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_ruin.py -v

# Run a single test
pytest tests/test_moran.py::TestMoranProcess::test_martingale_property -v

# Run all experiments for a chapter (generates output/data/*.csv and *.npz)
python src/ch02_moran/experiment.py
python src/ch03_insurance/experiment.py
python src/ch04_ruin/experiment.py

# Regenerate all figures (reads from output/data/, writes to output/figures/ and paper/figures/)
python src/ch02_moran/plot.py
python src/ch03_insurance/plot.py
python src/ch04_ruin/plot.py

# Compile the paper
cd paper
latexmk -xelatex main.tex
```

## Architecture

### Two-layer structure

- **`src/core/`** — reusable base layer shared by all chapters:
  - `processes.py` — `Martingale` ABC, `SymmetricRW`
  - `stopping_times.py` — `StoppingTime` ABC, `FixedTime`, `HittingLevel`, `ExitInterval`, `Truncated`
  - `simulation.py` — `MonteCarloSimulation` engine and `SimulationResult` dataclass
  - `batching.py` — `split_batch_sizes()` for variance estimation across independent batches
  - `visualization.py` — unified matplotlib style (`set_style()`), color constants, `save_figure()` (writes to both `output/figures/` and `paper/figures/`)

- **`src/ch0X_*/`** — per-chapter layer, each following the same triad: model definition → experiment script → plot script. Chapter 4 (gambler's ruin) has no model file — it directly uses `SymmetricRW` from core.

### Data flow

Experiments (`experiment.py`) → CSV/NPZ in `output/data/` → Plot scripts (`plot.py`) → PDF figures in `output/figures/` + `paper/figures/` → Paper LaTeX includes from `paper/figures/`.

### Paper structure

`paper/main.tex` includes ch01–ch05 (conclusion).

## Important conventions

- **OST applies in two-sided, fails in one-sided**: The gambler's ruin chapter demonstrates this asymmetry. For one-sided hitting, `E[S_τ] = b ≠ 0`, but for any finite truncation N, `E[S_{τ∧N}] = 0` by OST condition 1 (bounded stopping time). The core phenomenon is `lim E ≠ E lim`.
- **`estimate_expectation()` filters by `reached_stop`**: It only averages over paths where the stopping time triggered within `max_steps`. For truncated stopping times this is always all paths; for one-sided hitting it excludes paths that wandered indefinitely.
- **Batch estimation pattern**: Experiments use `split_batch_sizes()` to divide trials into independent batches, storing per-batch success rates/means for boxplot visualization and proper SE estimation across batches (not just across individual paths).
- **Prefer vectorized operations**: For simulations and data processing, use NumPy array operations (batched paths, boolean masks, cumulative sums) instead of per-path or per-step Python loops. Keep explicit loops only for inherently sequential logic or when they are clearly clearer.
- **`output/data/` and `output/figures/` are tracked** (so the paper can be compiled from committed data without re-running experiments). `paper/figures/` is also tracked (paper compilation needs them).
- **`docs/` is gitignored** — it contains reference PDFs and third-party GitHub repos, not project code.
- **Python 3.13, NumPy, SciPy, Matplotlib, Pandas** are the only runtime dependencies (pytest for tests); see `requirements.txt`.
