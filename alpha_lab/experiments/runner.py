"""Small orchestration helper for factor-to-backtest experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from alpha_lab.backtest.engine import run_backtest
from alpha_lab.config.backtest import BacktestConfig
from alpha_lab.experiments.artifacts import SavedExperiment, save_experiment
from alpha_lab.metrics.summary import generate_backtest_summary


def run_and_save_experiment(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    backtest_config: BacktestConfig,
    experiment_config: object,
    output_dir: Path | str,
    *,
    fingerprints: dict[str, str] | None = None,
) -> SavedExperiment:
    result = run_backtest(weights, prices=prices, config=backtest_config)
    metrics = generate_backtest_summary(result)
    return save_experiment(
        output_dir,
        {"experiment": experiment_config, "backtest": backtest_config},
        metrics=metrics,
        result=result,
        fingerprints=fingerprints,
    )


__all__ = ["run_and_save_experiment"]
