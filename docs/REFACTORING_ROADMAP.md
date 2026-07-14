# Alpha-Lab refactoring and evolution roadmap

## Implementation status — 2026-07-14

Path B, the personal research framework, has been implemented. The audit below is retained as the historical rationale.

| Roadmap item | Status | Implementation |
| --- | --- | --- |
| Project hygiene and test harness | Complete | Generated files ignored/removed; synthetic pytest suite, Ruff, examples, and benchmarks added |
| Explicit data contract | Complete | Canonical panel validation, missing/duplicate policy, fingerprints, and documentation |
| CSV efficiency | Complete | Wide column projection and chunked long reads |
| Parquet | Complete | Local adapter, projected reads, and cache helpers |
| Data-source extensibility | Complete | `DataSource` protocol and optional cached yfinance adapter |
| Fixed random stock sampling | Complete | Seeded once-per-experiment selection with audit record and pre-read column sampling |
| Backtest decomposition | Complete | Validation, schedule, accounting, result, and orchestration modules |
| Backtest correctness/performance | Complete | Delay/cost/turnover/missing-price tests and preallocated NumPy accounting |
| Industry neutralization | Complete | Categorical and continuous cross-sectional residualization |
| Factor metadata and diagnostics | Complete | Metadata, IC/ICIR, coverage, quantiles, spread, turnover, and stability |
| Configuration normalization | Complete | Data, portfolio, experiment configs and serializable existing configs |
| Experiment artifacts/comparison | Complete | Deterministic run IDs, JSON/Parquet artifacts, reload and comparison |
| Walk-forward ML | Complete | Optional sklearn linear models, whole-date expanding/rolling splits, leakage checks |
| Runnable education path | Complete | Four scripts, tutorial notebooks, architecture, glossary, and data contract docs |
| Production-like simulator (Path C) | Outside scope | Requires point-in-time constituents, delistings, corporate actions, liquidity and execution data |

The project remains backward-compatible at its documented top-level API. Correctness fixes are covered by regression tests.

This note is a project-wide audit of the current Alpha-Lab code and docs. It is written for the next iteration of the project: keep the notebook-friendly learning workflow, but make the core safer, easier to test, and easier to extend into more realistic factor research.

## Executive summary

Alpha-Lab already has a useful spine for learning factor research:

1. load local prices;
2. compute cross-sectional factors;
3. transform scores;
4. build target weights;
5. run a simple rebalance backtest;
6. inspect returns, drawdowns, trading, and factor diagnostics.

The biggest opportunities are not to add more features immediately. The next step should be to make the existing research loop reproducible and testable, then add extension points one layer at a time.

Recommended order:

1. **Project hygiene and test harness**: remove generated files, add a real test suite, and make the demo path executable in CI.
2. **Data contract cleanup**: define explicit wide/long price schemas, implement Parquet or remove it from the public API until ready, and make missing-data behavior deterministic.
3. **Backtest correctness pass**: split the engine into validation, scheduling, execution, accounting, and reporting units; add tests for execution delay, turnover, costs, and alignment.
4. **Factor research workflow**: finish industry/sector neutralization, add forward-return diagnostics examples, and standardize factor metadata.
5. **Future research extensions**: only after the above, add ML, richer data sources, and realistic trading simulation.

## Current strengths

- The package API is approachable: `alpha_lab.__init__` exports the main loader, factor, portfolio, backtest, and metrics functions.
- The README clearly positions the project as a lightweight research sandbox rather than a production trading system.
- The code is organized into sensible domains: `data`, `factors`, `portfolio`, `backtest`, `metrics`, `config`, and `utils`.
- The included sample adjusted-close panel gives the project a self-contained onboarding path.
- The backtest result object and summary helpers make notebook usage convenient.

## High-priority refactors

### 1. Add tests before expanding the feature surface

There is no dedicated test suite in the repository. Before touching strategy logic, add `tests/` with small synthetic fixtures that do not depend on the large sample CSV.

Start with these cases:

- CSV loader: wide CSV, long CSV, missing tickers, date filtering, invalid fields.
- Factor transforms: z-score behavior with NaNs and constant rows, winsorization bounds, rank normalization ordering.
- Portfolio weighting: long-only sums to 1, long-short sums to zero when balanced, turnover limits behave as expected.
- Backtest engine: one-asset buy-and-hold, execution delay, transaction cost drag, empty weights/prices errors.
- Metrics: known equity curve for CAGR, Sharpe, volatility, max drawdown, and drawdown duration.

A good first CI target is:

```bash
python -m compileall alpha_lab
python -m pytest
```

### 2. Split the backtest engine into smaller accounting components

`alpha_lab/backtest/engine.py` currently owns configuration setup, optional data loading, rebalance schedule alignment, execution delay, portfolio simulation, trade recording, return calculation, and result construction. That is fine for a first notebook prototype, but it makes correctness hard to reason about.

Suggested split:

- `backtest/validation.py`: validate prices, weights, date ranges, and column alignment.
- `backtest/schedule.py`: rebalance date selection and execution-delay mapping.
- `backtest/accounting.py`: cash, holdings, transaction costs, turnover, and daily valuation.
- `backtest/result.py`: `BacktestResult` and serialization helpers.
- `backtest/engine.py`: orchestration only.

This will make it much easier to test accounting rules without constructing the entire project stack.

### 3. Make the data layer explicit and honest

`load_prices()` exposes `source="parquet"`, but Parquet raises `NotImplementedError`. Either implement a Parquet adapter under `alpha_lab/data/sources/` or remove the option until it is supported.

Also define a data contract in docs and tests:

- accepted column names for long format;
- required date column behavior;
- whether prices are adjusted or raw;
- how duplicate `(date, ticker)` rows are handled;
- how missing prices are forward-filled;
- how calendars are inferred.

The project should eventually expose a minimal `DataSource` protocol such as:

```python
class DataSource(Protocol):
    def load_prices(self, tickers, start_date, end_date, field) -> pd.DataFrame: ...
```

Then CSV, Parquet, SQL, and API adapters can be plugged in without expanding `load_prices()` conditionals.

### 4. Finish or remove stubs

The README and code both expose future features that are not complete yet. Keep scaffolding only when it guides the next step; otherwise it increases confusion for new learners.

Current examples:

- `neutralize_industry()` is a public factor transform but raises `NotImplementedError`.
- `apply_weight_constraints()` is marked as a v1 stub but already includes meaningful constraint logic; it should be documented and tested as real behavior or renamed to clarify its maturity.
- ML config objects exist, but there is no ML pipeline.
- `main.py` is a placeholder while the project is library/notebook-first.

Recommendation: mark incomplete APIs as experimental in docs and avoid exporting them from top-level modules until they are tested.

### 5. Normalize configuration and metadata

The config layer is useful, but it mixes active configuration with future placeholders. For a research sandbox, prioritize a few concrete config objects:

- `DataConfig`: data source, file name/path, price field, fill policy.
- `FactorConfig`: factor name, parameters, transform stack, lag policy.
- `PortfolioConfig`: weighting rule, constraints, rebalance frequency.
- `BacktestConfig`: execution delay, costs, cash, date range.
- `ExperimentConfig`: seed, universe, artifacts directory, run name.

This creates a clean route toward reproducible experiments and notebook-to-script migration.

## Medium-priority improvements

### Documentation and examples

- Turn the placeholder notebooks into a linear tutorial path:
  - `01_data_and_momentum.ipynb`
  - `02_portfolio_and_backtest.ipynb`
  - `03_factor_diagnostics.ipynb`
  - `04_experiment_template.ipynb`
- Add a concise architecture diagram to the README.
- Add a glossary for factor research terms: signal date, execution date, rebalance date, forward return, IC, turnover, gross/net exposure, transaction cost.
- Add one scriptable example under `examples/` so users can run a demo without opening Jupyter.

### Type hints and optional dependencies

Several modules use optional imports for packages that are already required project dependencies. This was useful in an early scaffold, but now it makes error paths and typing less clear.

Recommendation:

- Treat `pandas` and `numpy` as required in runtime modules.
- Keep optional handling only for truly optional packages such as `torch` or future ML libraries.
- Add `pandas-stubs` and a type checker later, after tests are in place.

### Code style cleanup

The code is readable but has early-prototype artifacts:

- mixed English/Chinese comments are useful for learning, but long bilingual comments sometimes obscure the actual logic;
- a few generated/macOS files were committed and should stay out of version control;
- some modules expose broad public APIs before tests exist;
- docstrings sometimes describe planned behavior more confidently than the implementation supports.

Keep bilingual explanations where they teach domain concepts, but move lengthy conceptual notes into docs and leave code comments for invariants and tricky implementation details.

## Future product directions

### Path A: Learning sandbox

If the goal is education, keep the project small:

- one clean data source;
- one or two robust factors;
- transparent backtest accounting;
- excellent notebooks and diagrams;
- many small tests.

This is the best near-term direction.

### Path B: Research framework

If the goal is personal research tooling, add:

- experiment configs and saved artifacts;
- factor library with metadata;
- IC/quantile/turnover tear sheets;
- walk-forward train/test splits;
- Parquet cache;
- result comparison utilities.

### Path C: Production-like simulator

Only pursue this after the research framework is stable. It requires:

- corporate action assumptions;
- survivorship-bias controls;
- delisting handling;
- liquidity and slippage models;
- benchmark and risk model integration;
- order-level simulation;
- audit logs and reproducibility guarantees.

## Suggested first milestone

A focused first milestone could be completed without changing the external user workflow:

1. add `.gitignore` entries for generated files and caches;
2. remove committed `.DS_Store` files;
3. add `tests/` with loader, factor, portfolio, metrics, and one simple backtest smoke test;
4. add an `examples/momentum_backtest.py` that mirrors the README quick start;
5. update README to point to this roadmap and the runnable example.

After that, each new feature can be added behind a test and a documented example.
