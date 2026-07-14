# Alpha-Lab Research Framework Implementation Plan

Design: `docs/superpowers/specs/2026-07-14-alpha-lab-research-framework-design.md`

Status: Complete (2026-07-14). See `docs/IMPLEMENTATION_SUMMARY.md` for verification and benchmark results.

## Delivery rules

- Preserve documented top-level APIs.
- Add or update focused tests before each behavioral implementation.
- Keep live-network behavior optional and mock it in tests.
- Use pandas at public boundaries and NumPy in the accounting hot path.
- Do not mix unrelated existing worktree edits into implementation commits.
- Keep every phase importable and testable.

## Phase 1 — Test and project baseline

1. Add pytest, coverage, Ruff, pandas stubs, PyArrow, optional scikit-learn, and optional yfinance dependency groups in `pyproject.toml`.
2. Add `alpha_lab/exceptions.py` with `AlphaLabError`, `DataContractError`, `AlignmentError`, `LookaheadError`, and `ConfigurationError`.
3. Add synthetic fixtures in `tests/conftest.py`.
4. Add baseline tests for current factor transforms, portfolio constraints, metrics, scheduling, and one-asset accounting.
5. Add benchmark entry points under `benchmarks/` without hard timing assertions.

Acceptance: package compiles, pytest discovers the suite, and reference behavior is recorded.

## Phase 2 — Data contracts and sources

1. Add `alpha_lab/data/contracts.py` with `PriceDataContract`, `normalize_price_panel`, and deterministic `fingerprint_frame`.
2. Add `alpha_lab/data/sources/base.py` with a runtime-checkable `DataSource` protocol.
3. Refactor CSV loading to inspect the header, project wide columns, support chunked long reads, and use the common normalizer.
4. Add `ParquetDataSource` and compatibility handling in `load_prices(source="parquet")`.
5. Add `alpha_lab/data/cache.py` with fingerprinted Parquet cache keys and atomic cache writes.
6. Add `UniverseSelection` and `sample_universe`; add `load_sampled_prices` as an additive convenience API.
7. Add an optional `YFinanceDataSource` that batches downloads and uses the local cache; imports fail with an actionable extras message only when used.
8. Test schema errors, duplicates, dates, fields, fills, projection, Parquet round trips, deterministic sampling, and mocked yfinance results.

Acceptance: local CSV and Parquet produce the same canonical panel, and fixed-seed samples reproduce exactly.

## Phase 3 — Backtest decomposition and optimization

1. Move `BacktestResult` to `alpha_lab/backtest/result.py`; preserve re-exports.
2. Add `alpha_lab/backtest/validation.py` for one-pass index/column/date validation.
3. Add `alpha_lab/backtest/schedule.py` for signal-to-execution integer mappings.
4. Add `alpha_lab/backtest/accounting.py` with a preallocated NumPy simulation and explicit missing valuation policy.
5. Reduce `engine.py` to configuration, optional loading, orchestration, and result construction.
6. Include signal date, execution date, side, shares, price, notional, and cost in trade records.
7. Test cash/holdings reconciliation, delay, missing prices, costs, thresholds, turnover caps, empty weights, and backward imports.
8. Add a benchmark that compares the retained reference implementation and optimized implementation for result equivalence and runtime.

Acceptance: regression fixtures agree within tolerance except documented bug fixes; the sample-data benchmark reports a material speedup or records why no optimization was retained.

## Phase 4 — Configuration and factor research

1. Add serializable `DataConfig`, `PortfolioConfig`, and `ExperimentConfig`; add `to_dict` support consistently while keeping existing configs.
2. Add `FactorDefinition`, `FactorResult`, and data/definition fingerprints.
3. Implement `neutralize_industry` and general `neutralize` for categorical and continuous exposures.
4. Extend diagnostics with coverage, dispersion, quantile returns, cumulative spread, turnover, stability, and a compact tear-sheet result.
5. Vectorize transforms where current `DataFrame.apply` is unnecessary and test NaN/constant behavior.

Acceptance: factor metadata round-trips, neutralized residuals have near-zero group/exposure means, and diagnostics align dates/tickers explicitly.

## Phase 5 — Experiments and artifacts

1. Add `alpha_lab/experiments/runner.py`, `artifacts.py`, and `comparison.py`.
2. Generate deterministic run IDs from normalized configuration and input fingerprints while recording a separate creation timestamp.
3. Save configuration/summary as JSON and tables as Parquet with CSV fallback.
4. Reload an experiment result and compare saved or in-memory runs.
5. Test paths with temporary directories, JSON normalization, save/load, and result comparisons.

Acceptance: an end-to-end momentum experiment can be saved, reloaded, and compared without copying raw source data.

## Phase 6 — Leakage-safe ML

1. Add `alpha_lab/ml/dataset.py` for `(date, ticker)` features and forward labels.
2. Add `alpha_lab/ml/split.py` for expanding and rolling whole-date walk-forward splits.
3. Add `alpha_lab/ml/models.py` for optional sklearn pipelines and Ridge/Lasso/ElasticNet factories.
4. Add `alpha_lab/ml/walk_forward.py` for per-window preprocessing, fitting, prediction, and metrics.
5. Return a prediction panel compatible with existing portfolio functions.
6. Test split boundaries, label timing, preprocessing isolation, estimator injection, and intentional lookahead failures.

Acceptance: a synthetic predictive feature completes walk-forward training and flows into a backtest with no date shared across train/test windows.

## Phase 7 — Examples and documentation

1. Add runnable scripts for sampled momentum, factor diagnostics, experiments, and walk-forward ML.
2. Replace empty/placeholder notebooks with the five-step tutorial path, using scripts as the testable source of truth.
3. Update README with architecture, installation extras, data assumptions, sampling, experiment, and ML examples.
4. Add a glossary and data contract document.
5. Reconcile every roadmap item as complete, deferred, or Path C.

Acceptance: bundled-data examples run without network; optional examples fail clearly when their extra is absent.

## Phase 8 — Final verification

Run:

```bash
uv run python -m compileall alpha_lab
uv run pytest
uv run ruff check alpha_lab tests examples benchmarks
uv run python examples/momentum_backtest.py
uv run python examples/sampled_momentum_backtest.py
uv run python benchmarks/backtest_benchmark.py
```

Record:

- test and lint results;
- before/after benchmark timing and memory;
- compatibility and correctness changes;
- optional dependency status;
- known survivorship/data limitations;
- work explicitly left for Path C.
