# Numerical Simulation of the Optional Stopping Theorem for Martingales

**English** | [简体中文](README.zh-CN.md)

Spring Semester, 2025–2026 Academic Year · Shanghai Jiao Tong University, Stochastic Processes (MATH4704) Course Project

## Why This Topic

While studying martingales, I posted the following on QQ:

> Consider a fair coin-flipping process: heads adds +1 and tails adds −1.
> Starting from 0, what is the expected number of coin flips needed for the cumulative score to reach 1?
> Suppose we adopt this strategy: stop as soon as the cumulative winnings reach 1. After playing many rounds, it seems one can "guarantee a profit." There is also a more classic strategy: double the stake on every bet until a win, so that a single win recovers all previous stakes.
> These seemingly profitable strategies all violate the boundedness requirement of the optional stopping theorem. The consequence is that the answer to my earlier question is positive infinity.
> For the first time, I felt that infinity was right beside me. Martingales are truly fascinating.

This is why I decided to systematically study the Optional Stopping Theorem (OST) through numerical simulation. The paper selects three representative stochastic process models: the Moran population genetics process, the Cramér–Lundberg insurance ruin process, and the gambler's ruin process. It uses Monte Carlo experiments to illustrate the full theoretical spectrum of the OST.

## Project Structure

```
Martingale/
├── paper/                     # LaTeX paper project
│   ├── main.tex               # main file (entry point)
│   ├── setup.tex              # document class setup and package loading
│   ├── nomenclature.tex       # nomenclature
│   ├── references.bib         # BibLaTeX bibliography
│   ├── sections/              # chapter TeX sources
│   │   ├── ch01_intro.tex     # Chapter 1: Introduction
│   │   ├── ch02_moran.tex     # Chapter 2: Moran model
│   │   ├── ch03_insurance.tex # Chapter 3: Insurance ruin model
│   │   ├── ch04_ruin.tex      # Chapter 4: Gambler's ruin model
│   │   └── ch05_conclusion.tex # Chapter 5: Conclusion and outlook
│   └── figures/               # paper figures (PDF)
├── src/                       # Python numerical experiment code
│   ├── core/                  # shared core layer
│   │   ├── processes.py       # martingale process abstract base class and implementations
│   │   ├── stopping_times.py  # stopping-time abstract base class and common subclasses
│   │   ├── simulation.py      # general Monte Carlo simulation engine
│   │   ├── batching.py        # batch estimation
│   │   └── visualization.py   # unified plotting style
│   ├── ch02_moran/            # Chapter 2: Moran model experiments
│   │   ├── moran_model.py     # model definition
│   │   ├── experiment.py      # experiment script
│   │   └── plot.py            # plotting script
│   ├── ch03_insurance/        # Chapter 3: Insurance ruin model experiments
│   │   ├── surplus_model.py   # model definition
│   │   ├── experiment.py      # experiment script
│   │   └── plot.py            # plotting script
│   ├── ch04_ruin/             # Chapter 4: Gambler's ruin model experiments
│   │   ├── experiment.py      # experiment script
│   │   └── plot.py            # plotting script
├── tests/                     # unit tests
│   ├── conftest.py            # pytest configuration (src/ import path)
│   ├── test_processes.py
│   ├── test_stopping_times.py
│   ├── test_simulation.py
│   ├── test_batching.py
│   ├── test_visualization.py
│   ├── test_moran.py
│   ├── test_ruin.py
│   └── test_insurance.py
└── output/                    # experiment output
    ├── data/                  # simulation data (CSV/NPZ)
    └── figures/               # generated figures
```

### Design Principles

- **Reusable core layer**: `src/core/` provides reusable building blocks (process base classes, stopping-time base classes, and the simulation engine). Each chapter's experiment layer only needs to define its model and parameters, without reimplementing simulation logic.
- **Consistent module responsibilities**: Each chapter follows the same division of work. The model definition file constructs the chapter-specific stochastic process and stopping time, the experiment script calls the core engine to run batch simulations and save the data, and the plotting script generates the paper figures from that data.
- **Separation of data and code**: Experiment outputs (data and figures) are stored under `output/` for reproducibility and version control.

## Requirements

- Python 3.13
- NumPy 2.1+
- SciPy 1.15+
- Matplotlib 3.10+
- Pandas 2.2+
- pytest 8+ (only needed to run the tests)

Install the dependencies with `pip install -r requirements.txt`.

## Building the Paper

### Dependencies

- A TeX distribution (TeX Live 2024+ recommended) that includes XeLaTeX and BibLaTeX.
- The Shanghai Jiao Tong University thesis template `sjtuthesis`. This project borrows its document class and style, but the paper is a course project rather than a degree thesis.

### Build Steps

```bash
cd paper
xelatex main.tex
biber main
xelatex main.tex
xelatex main.tex
```

Or use `latexmk`:

```bash
cd paper
latexmk -xelatex main.tex
```

## Running the Numerical Experiments

### Environment

- Python 3.13+
- NumPy, SciPy, Matplotlib, and Pandas (see `requirements.txt`)

### Usage

```bash
# Run the experiments for one chapter
python src/ch02_moran/experiment.py
python src/ch03_insurance/experiment.py
python src/ch04_ruin/experiment.py

# Run the tests
pytest tests/
```

## Acknowledgements

I thank the course instructor, Deng Zhang, and the teaching assistants for their teaching and guidance.

## AI Usage

This project was completed with assistance from DeepSeek V4 (in the Claude Code environment), GPT 5.4 (in the Codex environment), and Gemini 3.5 Flash (for general Q&A). I manually reviewed the paper's results.
