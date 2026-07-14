# Alpha-Lab

Alpha-Lab is a notebook-friendly personal research framework for equity factors. It provides a transparent path from local or cached prices to factors, portfolio weights, backtests, diagnostics, reproducible experiments, and leakage-safe walk-forward models.

It is intentionally not a production trading simulator. Corporate-action reconstruction, point-in-time index membership, delisting returns, borrow constraints, market impact, and order-book simulation remain outside its scope.

## Architecture

```text
CSV / Parquet / optional yfinance
              │
              ▼
   data contract + fixed universe sample
              │
              ▼
 factors + transforms + metadata ──► factor diagnostics
              │
              ▼
 portfolio weights ──► schedule ──► NumPy accounting
                                         │
                                         ▼
                         result + metrics + experiment artifacts

 features + forward labels ──► whole-date walk-forward ML ──► weights
```

Public functions such as `load_prices`, `momentum`, `top_k_long_only`, and `run_backtest` remain compatible with the original project. Internally, data validation, scheduling, accounting, results, experiments, and ML are isolated and independently testable.

## Installation

Python 3.10 or newer is required.

```bash
uv sync
uv sync --group dev
```

With pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional capabilities:

```bash
pip install -e '.[data]'    # PyArrow / Parquet
pip install -e '.[market]'  # yfinance
pip install -e '.[ml]'      # scikit-learn
pip install -e '.[all]'
```

## Reproducible sampled backtest

The bundled file contains 196 adjusted-close columns. Sampling happens once before the backtest and is fixed by its seed:

```python
import pandas as pd

from alpha_lab import (
    default_backtest_config,
    load_sampled_prices,
    momentum,
    run_backtest,
    top_k_long_only,
    zscore,
)

file_name = "random_sampled_stocks_adj_close.csv"
candidates = pd.read_csv(f"data/raw/{file_name}", nrows=0).columns[1:].tolist()

prices, selection = load_sampled_prices(
    candidates,
    "2018-01-01",
    "2024-12-31",
    sample_size=30,
    seed=42,
    csv_file=file_name,
)
factor = zscore(momentum(prices, lookback=60, lag=1), axis=1)
weights = top_k_long_only(factor, k_pct=0.2, buffer_pct=0.05)
config = default_backtest_config().with_updates(max_turnover=1.0)
result = run_backtest(weights, prices=prices, config=config)

print(selection.selection_id, selection.selected)
print(result.summary_stats())
```

`selection` records the candidate and selected tickers, seed, filtering rules, exclusions, and data fingerprint. Passing no sampling request preserves the full-universe workflow.

## Data sources and contracts

`load_prices` supports:

- wide CSV: `date,AAPL,MSFT,...`;
- long CSV: `date,ticker,open,high,low,close,volume`;
- wide or long Parquet through `source="parquet"`;
- any object implementing the `DataSource` protocol;
- optional cached yfinance through `YFinanceDataSource`.

Wide CSV reads project only requested ticker columns. Long CSV reads in chunks. Parquet uses column projection where possible. Duplicate dates or `(date, ticker)` rows are rejected by default instead of silently aggregated.

The canonical panel has a sorted, unique, timezone-naive `DatetimeIndex`, deterministic ticker columns, numeric values, and an explicit missing-value policy. See [Data contract](docs/DATA_CONTRACT.md).

## Factor research

Available research tools include:

- single- and multi-period momentum;
- winsorization, z-scores, ranks, and industry/continuous-exposure neutralization;
- `FactorDefinition` and `FactorResult` metadata;
- Pearson/Spearman IC and ICIR;
- coverage, dispersion, quantile returns, cumulative spread, factor turnover, and rank stability.

```python
from alpha_lab.metrics.factor_diagnostics import (
    compute_forward_returns,
    generate_factor_tear_sheet,
)

future = compute_forward_returns(prices, horizon=20, delay=1)
tear_sheet = generate_factor_tear_sheet(factor, future, quantiles=5, min_obs=10)
print(tear_sheet.ic_stats)
```

## Backtest semantics

The engine distinguishes signal dates from execution dates. `execution_delay_days=1` maps a signal to the next available trading day. Missing or non-positive execution prices are not traded; existing holdings use the last known valid valuation price. Transaction costs, turnover caps, and rebalance thresholds are applied once inside the accounting layer.

The hot loop uses preallocated NumPy arrays and returns pandas equity, returns, positions, and reconciled trade records. Trade rows include signal date, execution date, side, shares, price, notional, and allocated cost.

## Experiments

`alpha_lab.experiments` saves normalized configuration, environment versions, fingerprints, warnings, metrics, equity, positions, returns, trades, predictions, and diagnostic tables. Run IDs depend on configuration and fingerprints; creation timestamps are recorded separately.

```python
from alpha_lab.experiments import run_and_save_experiment

saved = run_and_save_experiment(
    weights,
    prices,
    config,
    {"run_name": "momentum-60", "seed": 42},
    "artifacts",
    fingerprints={"universe": selection.selection_id},
)
print(saved.path)
```

## Walk-forward ML

Scikit-learn support is optional. The default model pipeline uses median imputation, standardization, and Ridge regression. Every preprocessing step is refit within its training window.

```python
from alpha_lab.ml import WalkForwardSplit, build_supervised_dataset, run_walk_forward

dataset = build_supervised_dataset(
    {"momentum_60": factor}, prices, horizon=5, delay=1
)
splitter = WalkForwardSplit(
    min_train_dates=252,
    test_dates=63,
    gap_dates=6,
)
ml_result = run_walk_forward(dataset, splitter)
```

Splits operate on complete dates, so stocks from one date cannot be divided between train and test sets.

## Runnable examples

```bash
uv run python examples/momentum_backtest.py
uv run python examples/sampled_momentum_backtest.py
uv run python examples/factor_diagnostics.py
uv run python examples/walk_forward_ml.py
uv run python examples/experiment_artifacts.py
uv run python benchmarks/backtest_benchmark.py
```

The scripts are the automated source of truth. Notebooks provide additional explanation and plots.

## Project structure

```text
alpha_lab/
  backtest/      validation, scheduling, NumPy accounting, result, orchestration
  config/        data, factor, portfolio, backtest, experiment, and ML configs
  data/          contracts, sampling, cache, CSV, Parquet, optional yfinance
  experiments/   artifact save/load, runner, comparisons
  factors/       factor definitions, metadata, transforms
  metrics/       performance, drawdown, trading, factor diagnostics
  ml/            supervised datasets, walk-forward splits and models
  portfolio/     rebalance schedules, weighting, constraints
benchmarks/      repeatable performance measurements
examples/        runnable local workflows
tests/           synthetic unit and integration tests
```

## Verification

```bash
uv run python -m compileall alpha_lab
uv run pytest
uv run ruff check alpha_lab tests examples benchmarks
```

See the [refactoring roadmap](docs/REFACTORING_ROADMAP.md), [approved design](docs/superpowers/specs/2026-07-14-alpha-lab-research-framework-design.md), [implementation plan](docs/superpowers/plans/2026-07-14-alpha-lab-research-framework-plan.md), and [glossary](docs/GLOSSARY.md).

## Data limitations

The bundled sample and a current ticker list are convenient for learning, but they do not provide point-in-time constituents and therefore do not eliminate survivorship bias. yfinance availability and adjusted-price conventions can change. Record source assumptions and fingerprints before interpreting a result as investment evidence.
