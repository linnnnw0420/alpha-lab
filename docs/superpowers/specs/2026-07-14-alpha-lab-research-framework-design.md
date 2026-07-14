# Alpha-Lab Personal Research Framework Design

Date: 2026-07-14
Status: Approved for implementation planning
Target: Path B — personal factor-research framework

## 1. Purpose and scope

Alpha-Lab will evolve from an early notebook-oriented prototype into a reproducible personal equity factor-research framework. The implementation will preserve the existing beginner-friendly workflow while making data handling, portfolio accounting, factor diagnostics, experiments, and machine-learning research explicit and testable.

This project includes:

- the high- and medium-priority work in `docs/REFACTORING_ROADMAP.md`;
- performance measurement and targeted removal of redundant internal work;
- CSV and Parquet data sources behind a common contract;
- a reproducible, fixed-per-experiment random universe sampler;
- a cached, optional yfinance data adapter;
- a decomposed and optimized backtest engine;
- factor metadata, neutralization, and research diagnostics;
- experiment configuration, artifacts, comparison, and reproducibility;
- leakage-safe cross-sectional walk-forward machine learning using optional scikit-learn;
- runnable scripts, a linear notebook tutorial, and a comprehensive test suite.

The following production-simulator features are outside this target:

- order-book or intraday execution simulation;
- corporate-action reconstruction beyond the assumptions carried by adjusted input data;
- point-in-time constituent history and automatic survivorship-bias removal;
- delisting-return reconstruction;
- institutional risk models, borrow availability, and market-impact calibration;
- a hard-coded Nasdaq website scraper.

The data-source protocol will leave room for those capabilities without claiming that they exist.

## 2. Compatibility policy

Existing public calls documented in the README and notebooks remain supported, including `load_prices`, `momentum`, factor transforms, portfolio weighting helpers, `run_backtest`, `BacktestResult`, and summary functions.

New components are additive. Existing top-level functions become compatibility facades over smaller internal units where appropriate. A public behavior that must eventually change will first emit `DeprecationWarning` and receive a migration note. This implementation will not remove an existing public function merely because an internal interface is cleaner.

When a correctness bug is found, correctness takes precedence over reproducing the bug. The change must be covered by a focused test and documented as a behavioral correction.

## 3. Architecture and module boundaries

The target package boundaries are:

```text
alpha_lab/
├── config/        Serializable data, factor, portfolio, backtest, and experiment configs
├── data/
│   ├── contracts  Schema validation, missing-value policy, and panel normalization
│   └── sources/   CSV, Parquet, and optional cached yfinance adapters
├── factors/
│   ├── library    Factor calculations such as momentum
│   ├── transform  Winsorization, normalization, and neutralization
│   └── metadata   Factor definitions, parameters, lag, direction, and transform history
├── portfolio/     Weight construction, constraints, and rebalance rules
├── backtest/
│   ├── validation Input and alignment validation
│   ├── schedule   Signal-date, rebalance-date, and execution-date mapping
│   ├── accounting Cash, holdings, costs, trades, and daily valuation
│   ├── result     Results and serialization helpers
│   └── engine     Orchestration only
├── metrics/       Performance, drawdown, trading, and factor diagnostics
├── experiments/   Run configuration, artifact storage, loading, and comparisons
└── ml/            Features, labels, walk-forward splits, estimators, and evaluation
```

Each unit has one primary responsibility and communicates through documented pandas tables, configuration dataclasses, or small result objects. The core research exchange shape remains a pandas `DataFrame` with a sorted, unique `DatetimeIndex` and ticker columns.

## 4. Data contracts and sources

### 4.1 Canonical price panel

A normalized price panel must satisfy these rules:

- index: timezone-naive, normalized, sorted, unique `DatetimeIndex`;
- columns: unique, non-empty ticker strings in deterministic order;
- values: numeric prices; finite positive values are tradable;
- requested field: explicit for long OHLCV data and documented for wide panels;
- date filtering: inclusive start and end dates;
- duplicates: rejected by default with duplicate keys in the error message; an explicit aggregation policy may be selected;
- missing values: preserved, forward-filled with an optional limit, or dropped according to an explicit policy;
- adjusted versus raw prices: declared as metadata/assumption rather than inferred silently;
- trading calendar: inferred from the normalized panel unless supplied explicitly.

The source protocol returns this canonical panel or source data that is passed through the same normalizer. `load_prices` remains the compatibility entry point.

### 4.2 Efficient local reads

- Wide CSV reads inspect the header first, select the universe, and load only the date and requested ticker columns.
- Long CSV reads can operate in chunks and discard unselected tickers and fields before concatenation.
- Parquet reads use column projection and date predicates where the installed engine permits them.
- Normalized CSV data can be cached as Parquet. The cache key includes the source-file fingerprint, requested field, date range, universe, and normalization/fill policy.
- Data is `float64` by default. `float32` is an explicit memory-saving option and is never selected silently.

### 4.3 Optional yfinance adapter

The yfinance adapter is an optional dependency and is not needed to import or test the core package. It downloads ticker batches, normalizes the result through the same contract, and writes a local Parquet cache. Network failures identify failed tickers and preserve successful cached batches. Tests use fakes and local fixtures; CI does not require live network access.

The framework explicitly reports that sampling current tickers or downloading current yfinance history does not remove survivorship bias.

## 5. Reproducible universe sampling

Random selection occurs before factor calculation and backtesting, not inside the accounting engine. The default mode samples once per experiment and holds the selected universe fixed for the entire run.

The sampler accepts:

- candidate tickers;
- sample size;
- random seed;
- requested date range;
- optional minimum observations, coverage ratio, and required start/end coverage.

It returns both the selected panel and a selection record containing the candidate universe, selected tickers, excluded tickers with reasons, seed, filtering rules, and input-data fingerprint. The same ordered input data, filters, sample size, and seed must return the same ordered selection.

No sample size means current full-universe behavior. Per-rebalance resampling is not part of the default workflow and will not be hidden behind `run_backtest`.

## 6. Backtest behavior

### 6.1 Processing stages

The engine performs these stages in order:

1. validate and normalize prices and weights;
2. align columns and the configured date range once;
3. select rebalance signal dates;
4. map signal dates to execution dates using the trading calendar and configured delay;
5. simulate cash, holdings, transaction costs, and mark-to-market valuation;
6. construct positions, trades, equity, returns, and the result object;
7. compute reports without rerunning accounting.

Signal-date and execution-date semantics must be visible in outputs and tests. An execution date may never precede its signal date.

### 6.2 Accounting invariants

- cash plus marked holdings equals equity within numerical tolerance;
- transaction costs cannot increase equity;
- an asset with a missing or non-positive execution price is not traded;
- holdings with a missing valuation price use the documented valuation policy rather than becoming silently worthless;
- turnover caps and rebalance thresholds have explicit units and are applied once;
- target weights are aligned to the available universe deterministically;
- empty prices and empty target weights follow documented, distinct error/hold-cash behavior;
- recorded trades reconcile with changes in holdings and cash.

### 6.3 Performance model

The public boundary remains pandas-based. After one validation and alignment pass, the accounting hot path uses preallocated NumPy arrays for prices, holdings, cash, equity, and position output. It avoids constructing or aligning pandas `Series` inside the daily loop. Pandas results are assembled once after simulation.

Schedule construction produces integer index mappings once. Metrics reuse the result's returns, drawdown series, and aggregated trades instead of recomputing identical intermediates.

## 7. Factor research

Existing factor functions continue to return pandas objects. The additive `FactorDefinition` and `FactorResult` interfaces capture:

- stable factor name and description;
- parameters and direction;
- source field and lookback;
- lag and signal-time assumption;
- ordered transform history;
- optional values and data fingerprint.

Neutralization supports categorical industry labels and continuous exposure matrices. For each date, it fits a cross-sectional least-squares model with an intercept and returns residual scores. Missing scores/exposures are excluded for that date and restored as missing in output. Rank-deficient designs use a stable least-squares solution and expose coverage diagnostics.

Diagnostics include:

- Pearson and Spearman information coefficients;
- mean IC, IC standard deviation, ICIR, and positive-IC rate;
- factor coverage and cross-sectional dispersion;
- quantile returns, top-minus-bottom spread, and cumulative spread;
- factor/portfolio turnover and stability;
- forward-return generation with explicit horizon and lag.

Every diagnostic aligns factor values and future returns by date and ticker explicitly.

## 8. Experiments and artifacts

The concrete configuration surface consists of `DataConfig`, `FactorConfig`, `PortfolioConfig`, `BacktestConfig`, and `ExperimentConfig`. Configurations serialize to stable JSON-compatible dictionaries.

An experiment run records:

- normalized configuration;
- run name, deterministic run identifier, timestamp, and seed;
- package and key dependency versions;
- source and selected-universe fingerprints;
- factor metadata;
- metrics and important warnings/assumptions;
- predictions, weights, trades, equity, and compact diagnostic tables as configured.

JSON is used for configuration and scalar summaries; CSV or Parquet is used for tabular outputs. Large raw source data is referenced by fingerprint and path rather than copied into every run. A loader reconstructs saved results, and comparison utilities build metric tables across runs.

## 9. Leakage-safe machine learning

Scikit-learn is an optional dependency group. The first supported estimators are Ridge, Lasso, ElasticNet, and any compatible user-provided estimator.

The canonical supervised data has `(date, ticker)` observations, feature columns, and an explicitly named forward-return label. Label generation starts strictly after the signal date. Walk-forward splitting operates on whole dates so the same date cannot appear in both training and test observations.

Supported split modes:

- expanding window: all eligible past dates train each future test window;
- rolling window: only the configured trailing training dates are used.

The default preprocessing pipeline imputes missing feature values from training data, standardizes features from training data, and fits the estimator. Preprocessors are refit inside each window. Predictions are reassembled as a date-by-ticker panel and can flow through existing portfolio and backtest APIs.

The implementation raises `LookaheadError` when feature dates, label horizons, training boundaries, prediction dates, or execution-delay assumptions violate temporal ordering.

## 10. Errors and validation

Stable exception categories are:

- `DataContractError` for malformed schemas, invalid dates, duplicates, or invalid prices;
- `AlignmentError` for incompatible prices, factors, weights, exposures, or calendars;
- `LookaheadError` for temporal leakage;
- `ConfigurationError` for invalid configuration combinations.

Missing data is handled by an explicit configured policy. The system does not silently guess a field, resolve duplicate records, or swallow alignment failures. Error messages identify the failing field, date, ticker, or configuration where feasible.

## 11. Testing and performance acceptance

### 11.1 Correctness tests

The test suite uses small synthetic fixtures and covers:

- wide and long CSV; Parquet; date/field filtering; duplicates; missing tickers; fill policies;
- fixed-seed sampling, coverage filtering, and source fingerprinting;
- factor transforms with missing and constant rows;
- neutralization and factor metadata;
- long-only, long-short, proportional weights, constraints, and turnover limits;
- schedule and execution-delay mapping;
- one-asset buy-and-hold, cash, costs, thresholds, missing prices, and trade reconciliation;
- known performance and drawdown curves;
- IC, quantile return, and turnover diagnostics;
- experiment save/load and result comparison;
- expanding and rolling walk-forward splits, preprocessing isolation, and leakage failures;
- end-to-end data-to-factor-to-backtest and ML-to-backtest workflows.

Core invariants include deterministic sampling, weight/exposure constraints, cash-and-holdings reconciliation, cost monotonicity, and nonnegative temporal ordering between signal and execution.

### 11.2 Benchmarks

Repeatable benchmarks compare:

- current and refactored accounting on synthetic and bundled sample panels;
- full versus projected CSV reads and CSV versus cached Parquet reads;
- factor diagnostics on increasing date/ticker dimensions.

Benchmark output records wall time and peak memory where practical. Ordinary CI verifies that benchmarks run and that optimized and reference outputs agree; it does not enforce fragile absolute timing thresholds. The bundled approximately 196-ticker daily sample is the main realistic local scenario. Optimizations are retained only when the benchmark or complexity analysis demonstrates value without obscuring accounting rules.

### 11.3 Project-level acceptance

Completion requires:

- `python -m compileall alpha_lab` succeeds;
- the full pytest suite succeeds;
- format/type checks configured by the implementation plan succeed;
- README and script examples execute successfully on bundled data;
- core behavior and compatibility are documented;
- the roadmap labels each item completed, deliberately deferred, or outside Path B;
- final reporting includes API changes, correctness fixes, benchmark results, tests, known data limitations, and remaining Path C work.

## 12. Documentation and teaching path

The README will include a compact architecture/data-flow diagram, installation extras, a fixed-universe quick start, and links to runnable examples.

The tutorial path will be:

1. data loading, contracts, and reproducible universe sampling;
2. momentum, transformations, and factor diagnostics;
3. portfolio construction and transparent backtest accounting;
4. reproducible experiments and result comparison;
5. leakage-safe cross-sectional walk-forward ML.

Script examples are the automated source of truth. Notebooks may provide richer commentary and plots, but core validation does not depend on persisted notebook execution state.

## 13. Delivery sequence

Implementation planning will decompose the work into independently verifiable phases:

1. test harness, project tooling, errors, and baseline benchmarks;
2. data contracts, projected CSV, Parquet, caching, sampling, and optional yfinance;
3. backtest validation, schedule, accounting, result extraction, and performance optimization;
4. configuration normalization, factor metadata, neutralization, and diagnostics;
5. experiments, artifacts, reload, and comparisons;
6. optional scikit-learn support and walk-forward research;
7. examples, notebooks, README, roadmap reconciliation, and full verification.

Each phase must leave the public package importable and the completed test surface passing.
