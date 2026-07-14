import numpy as np
import pandas as pd
import pytest

from alpha_lab.config import default_ml_model_config, default_ml_split_config
from alpha_lab.exceptions import LookaheadError
from alpha_lab.experiments import compare_experiments, load_experiment, make_run_id, save_experiment
from alpha_lab.ml import WalkForwardSplit, build_supervised_dataset, run_walk_forward


def test_experiment_save_load_and_compare(tmp_path) -> None:
    config = {"name": "demo", "seed": 42}
    first = save_experiment(
        tmp_path, config, metrics={"sharpe_ratio": 1.2}, tables={"x": pd.Series([1, 2])}
    )
    loaded = load_experiment(first.path)
    assert loaded.run_id == make_run_id(config)
    comparison = compare_experiments({"first": loaded})
    assert comparison.loc["sharpe_ratio", "first"] == 1.2


def test_walk_forward_uses_whole_ordered_dates(price_panel: pd.DataFrame) -> None:
    feature = price_panel.pct_change().fillna(0.0)
    dataset = build_supervised_dataset({"momentum": feature}, price_panel, horizon=1, delay=0)
    splitter = WalkForwardSplit(min_train_dates=4, test_dates=2, gap_dates=1)
    for train_index, test_index in splitter.split(dataset):
        train_dates = dataset.iloc[train_index].index.get_level_values("date")
        test_dates = dataset.iloc[test_index].index.get_level_values("date")
        assert train_dates.max() < test_dates.min()
        assert set(train_dates).isdisjoint(test_dates)
    result = run_walk_forward(dataset, splitter)
    assert not result.prediction_panel.empty
    assert not result.window_metrics.empty
    assert np.isfinite(result.window_metrics["mse"]).any()


def test_rolling_window_limits_training_dates(price_panel: pd.DataFrame) -> None:
    dataset = build_supervised_dataset({"value": 1 / price_panel}, price_panel, horizon=1)
    splitter = WalkForwardSplit(
        min_train_dates=3, test_dates=1, mode="rolling", train_window_dates=4
    )
    for train_index, _ in splitter.split(dataset):
        dates = dataset.iloc[train_index].index.get_level_values("date").unique()
        assert len(dates) <= 4


def test_walk_forward_rejects_unpurged_labels(price_panel: pd.DataFrame) -> None:
    dataset = build_supervised_dataset({"value": 1 / price_panel}, price_panel, horizon=2, delay=1)
    splitter = WalkForwardSplit(min_train_dates=3, test_dates=1, gap_dates=2)
    with pytest.raises(LookaheadError, match="too small"):
        run_walk_forward(dataset, splitter)


def test_ml_configs_are_concrete_and_serializable() -> None:
    assert default_ml_model_config().model_type == "ridge"
    assert default_ml_split_config().to_dict()["walk_forward"] is False
    with pytest.raises(ValueError, match="chronologically"):
        default_ml_split_config().with_updates(train_end="2025-01-01")
