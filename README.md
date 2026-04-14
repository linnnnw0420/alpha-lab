# Alpha-Lab

Alpha-Lab is a small, notebook-friendly framework for factor research in equities. It focuses on the usual research loop: load local price data, build factors, turn scores into portfolio weights, run a simple backtest, and inspect the results.

The codebase is lightweight on purpose. It is closer to a research sandbox than a production trading system.

## What is already here

- Local price loading from CSV files in either wide or long format
- Momentum factor calculation, including multi-horizon momentum
- Common factor transforms such as winsorization, z-score normalization, and rank normalization
- Portfolio construction helpers for long-only, long-short, and proportional weighting
- A rebalancing backtest engine with basic transaction cost and turnover controls
- Performance, drawdown, trading, and factor-diagnostics utilities

## Current scope

- This repository is library-first and notebook-first. `main.py` is still a placeholder.
- CSV is the only implemented data source right now. Parquet is declared but not implemented.
- The momentum and backtest pieces are the most complete parts of the project.
- Some config and ML-related modules are scaffolding for later work.
- `neutralize_industry()` is a stub.
- Notebook coverage is uneven:
  - `04_factor_diagnostics.ipynb` and `99_sanity_check.ipynb` contain real work
  - `01_factor_demo.ipynb` and `03_ml_cross_section.ipynb` are empty
  - `02_backtest_demo.ipynb` exists but is effectively a placeholder

## Installation

Python 3.10+ is required.

If you use `uv`:

```bash
uv sync
```

If you prefer plain `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For notebook work:

```bash
uv sync --group dev
```

## Quick Start

```python
from alpha_lab import (
    UNIVERSE_DEMO,
    default_backtest_config,
    load_prices,
    momentum,
    zscore,
    top_k_long_only,
    run_backtest,
    generate_backtest_summary,
)

# The sample file in data/raw uses a wide format:
# Date, TICKER_1, TICKER_2, ...
prices = load_prices(
    universe=UNIVERSE_DEMO,
    start_date="2018-01-01",
    end_date="2024-12-31",
    csv_file="random_sampled_stocks_adj_close.csv",
)

factor = momentum(prices, lookback=60, lag=1)
factor = zscore(factor, axis=1)
weights = top_k_long_only(factor, k_pct=0.2, buffer_pct=0.05)

cfg = default_backtest_config().with_updates(
    start_date="2018-01-01",
    end_date="2024-12-31",
    rebalance_freq="M",
)

result = run_backtest(weights=weights, prices=prices, config=cfg)
summary = generate_backtest_summary(result)

print(summary["total_return"])
print(summary["sharpe_ratio"])
```

## Data Layout

By default the project looks for data under `data/raw/`.

The CSV loader supports two shapes:

1. Wide format

```text
date,AAPL,MSFT,GOOGL
2024-01-02,185.64,370.87,139.56
2024-01-03,184.25,368.11,138.45
```

2. Long format

```text
date,ticker,open,high,low,close,volume
2024-01-02,AAPL,184.0,186.1,183.4,185.64,53412000
2024-01-02,MSFT,369.8,372.4,368.7,370.87,22111000
```

If you do not pass `csv_file=...`, the loader searches for:

- `close.csv`
- `prices_close.csv`
- `prices.csv`
- `ohlcv.csv`

This repository also includes a sample file at `data/raw/random_sampled_stocks_adj_close.csv`. It is a wide-format adjusted-close panel with 196 tickers and dates from `2010-01-01` to `2025-12-31`.

## Project Structure

```text
alpha_lab/
  backtest/      Backtest engine and result container
  config/        Paths, universe, factor, backtest, and ML config objects
  data/          Data loaders and source adapters
  factors/       Factor definitions and factor transforms
  metrics/       Performance, drawdown, trading, and diagnostics utilities
  portfolio/     Rebalance scheduling and weighting logic
  utils/         Logging, dates, math, typing, and random helpers
notebooks/       Research notebooks and sanity checks
data/raw/        Local sample data
```

## Useful Modules

- `alpha_lab.data.loader`: load universes, prices, and returns
- `alpha_lab.factors.momentum`: momentum factor helpers
- `alpha_lab.factors.transform`: winsorize, z-score, rank normalization
- `alpha_lab.portfolio.weighting`: factor-to-weight conversion
- `alpha_lab.backtest.engine`: core backtest loop
- `alpha_lab.metrics.summary`: one-shot backtest summaries
- `alpha_lab.metrics.factor_diagnostics`: IC and forward-return diagnostics

## Path Configuration

The project resolves paths from the repository root by default, but you can override them with environment variables:

- `ALPHA_LAB_ROOT`
- `ALPHA_LAB_DATA_DIR`
- `ALPHA_LAB_ARTIFACT_DIR`
- `ALPHA_LAB_CACHE_DIR`
- `ALPHA_LAB_LOGS_DIR`

## Notes

- Rebalance frequencies are `D`, `W`, and `M`.
- The backtest engine applies an execution delay and basic turnover controls through `BacktestConfig`.
- Trading metrics depend on positions and recorded trades being available.
- The package currently depends on `numpy`, `pandas`, `matplotlib`, and `scipy`.

## Next sensible improvements

- Add a real backtest demo notebook
- Implement Parquet support
- Fill in the ML pipeline or remove the placeholder surface area
- Add tests around loader edge cases and notebook examples
