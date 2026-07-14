# Alpha-Lab Path B implementation summary

Date: 2026-07-14

## Outcome

Alpha-Lab now implements the approved personal research framework while preserving the original documented top-level workflow.

## Delivered

- Canonical price-data contract with explicit duplicate, missing-value, date, ticker, dtype, and adjusted-price assumptions.
- Projected wide CSV reads, chunked long CSV reads, Parquet source/predicates, normalized Parquet cache, and a `DataSource` protocol.
- Fixed-per-experiment seeded universe sampling with selection audit records and pre-read column sampling.
- Optional batched yfinance adapter with local Parquet cache and mocked network tests.
- Backtest decomposition into validation, execution schedule, NumPy accounting, result, and orchestration modules.
- Reconciled trade records with signal/execution dates, direction, shares, notional, and costs.
- Factor definitions/results, categorical industry neutralization, continuous exposure neutralization, and stable fingerprints.
- IC/ICIR, coverage, dispersion, quantile returns, cumulative spread, factor turnover, and rank stability.
- Serializable data, factor, portfolio, backtest, experiment, universe, and ML configurations.
- Deterministic experiment IDs, environment/config/fingerprint records, JSON summaries, Parquet tables, reload, and comparison.
- Optional scikit-learn Ridge/Lasso/ElasticNet pipelines and whole-date expanding/rolling walk-forward evaluation.
- Purged-window validation based on label horizon plus execution delay.
- Runnable momentum, fixed-sample, diagnostics, experiment-artifact, and ML examples.
- Updated notebooks, architecture documentation, data contract, glossary, roadmap, and CLI demo.

## Correctness changes

- Factor z-scores preserve original missing values instead of silently converting every NaN to zero.
- Whole-frame winsorization now uses actual pooled quantiles instead of the minimum/maximum of column quantiles.
- Constant factor/return cross-sections are skipped before correlation, preventing undefined-correlation warnings.
- Missing or non-positive prices cannot execute a trade; existing holdings use the last valid mark for valuation.
- Signal dates and execution dates are explicitly mapped and recorded.
- Walk-forward training is rejected when the purge gap is smaller than `delay + horizon`.
- Duplicate dates and duplicate long-form `(date, ticker)` keys raise explicit contract errors.

## Performance

The repeatable benchmark uses 2,000 business dates and 196 assets with a one-time equal-weight allocation.

| Implementation | Wall time |
| --- | ---: |
| Legacy-style pandas loop | 0.5254 s |
| Preallocated NumPy accounting | 0.0308 s |
| Measured speedup | 17.05× |

Times depend on hardware and load. The benchmark asserts numerical equivalence but CI does not impose a brittle timing threshold.

## Verification

- `uv run pytest -q`: 32 passed.
- `uv run python -m compileall -q alpha_lab`: passed.
- `uv run ruff check alpha_lab tests examples benchmarks main.py`: passed.
- `uv run ruff format --check alpha_lab tests examples benchmarks main.py`: passed.
- Six notebook files parsed as valid JSON.
- All five example workflows and the accounting benchmark completed on local data without live network access.

## Compatibility

Existing documented imports such as `load_prices`, `momentum`, `zscore`, `top_k_long_only`, `run_backtest`, and `BacktestResult` remain available. Private engine helpers commonly used by old notebooks are retained as compatibility wrappers.

## Remaining limitations

- Current or randomly sampled ticker lists do not remove survivorship bias.
- The bundled data does not provide point-in-time constituents, delisting returns, or independently reconstructed corporate actions.
- yfinance availability and adjusted-price conventions are external assumptions; the live network path was not used in final verification.
- The simulator remains daily and weight-driven. Liquidity, borrow, market impact, intraday orders, and institutional risk models belong to Path C.
