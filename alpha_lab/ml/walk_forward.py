"""Leakage-safe window-by-window model fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.exceptions import AlignmentError, LookaheadError
from alpha_lab.ml.models import make_linear_model
from alpha_lab.ml.split import WalkForwardSplit


@dataclass
class WalkForwardResult:
    predictions: pd.Series
    prediction_panel: pd.DataFrame
    window_metrics: pd.DataFrame


def run_walk_forward(
    dataset: pd.DataFrame,
    splitter: WalkForwardSplit,
    *,
    label: str | None = None,
    estimator=None,
) -> WalkForwardResult:
    try:
        from sklearn.base import clone
    except ImportError as exc:
        raise ImportError("ML support requires `pip install alpha-lab[ml]`") from exc
    label_name = label or str(dataset.attrs.get("label_name", "forward_return"))
    if label_name not in dataset:
        raise AlignmentError(f"Label column {label_name!r} is missing")
    feature_names = [column for column in dataset.columns if column != label_name]
    if not feature_names:
        raise AlignmentError("No feature columns remain after selecting the label")
    horizon = int(dataset.attrs.get("horizon", 1))
    delay = int(dataset.attrs.get("delay", 0))
    if horizon < 1 or delay < 0:
        raise LookaheadError("Invalid forward-label timing metadata")
    required_gap = horizon + delay
    if splitter.gap_dates < required_gap:
        raise LookaheadError(
            f"gap_dates={splitter.gap_dates} is too small for delay={delay} and "
            f"horizon={horizon}; use at least {required_gap}"
        )
    base_estimator = estimator or make_linear_model("ridge")
    predictions = pd.Series(np.nan, index=dataset.index, name="prediction", dtype=float)
    rows: list[dict[str, object]] = []
    for window, (train_index, test_index) in enumerate(splitter.split(dataset), start=1):
        train = dataset.iloc[train_index].dropna(subset=[label_name])
        test = dataset.iloc[test_index]
        if train.empty or test.empty:
            continue
        train_dates = pd.DatetimeIndex(train.index.get_level_values("date"))
        test_dates = pd.DatetimeIndex(test.index.get_level_values("date"))
        if train_dates.max() >= test_dates.min():
            raise LookaheadError("Training observations overlap or follow the test window")
        model = clone(base_estimator)
        model.fit(train[feature_names], train[label_name])
        predicted = np.asarray(model.predict(test[feature_names]), dtype=float)
        predictions.iloc[test_index] = predicted
        actual = test[label_name].to_numpy(dtype=float)
        valid = np.isfinite(actual) & np.isfinite(predicted)
        mse = float(np.mean((actual[valid] - predicted[valid]) ** 2)) if valid.any() else np.nan
        correlation = (
            float(np.corrcoef(actual[valid], predicted[valid])[0, 1])
            if valid.sum() > 1 and np.std(actual[valid]) > 0 and np.std(predicted[valid]) > 0
            else np.nan
        )
        rows.append(
            {
                "window": window,
                "train_start": train_dates.min(),
                "train_end": train_dates.max(),
                "test_start": test_dates.min(),
                "test_end": test_dates.max(),
                "n_train": len(train),
                "n_test": len(test),
                "mse": mse,
                "correlation": correlation,
            }
        )
    available = predictions.dropna()
    panel = available.unstack("ticker") if not available.empty else pd.DataFrame()
    return WalkForwardResult(predictions, panel, pd.DataFrame(rows))


__all__ = ["WalkForwardResult", "run_walk_forward"]
