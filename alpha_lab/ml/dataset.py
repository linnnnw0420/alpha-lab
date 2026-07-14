"""Cross-sectional feature and forward-label construction."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from alpha_lab.exceptions import AlignmentError, LookaheadError
from alpha_lab.metrics.factor_diagnostics import compute_forward_returns


def build_supervised_dataset(
    features: Mapping[str, pd.DataFrame],
    prices: pd.DataFrame,
    *,
    horizon: int = 1,
    delay: int = 1,
    label_name: str = "forward_return",
    drop_missing_label: bool = True,
) -> pd.DataFrame:
    """Align wide feature panels into `(date, ticker)` observations."""
    if not features:
        raise AlignmentError("At least one feature panel is required")
    if horizon < 1:
        raise LookaheadError("horizon must be >= 1")
    if delay < 0:
        raise LookaheadError("delay must be non-negative")
    common_dates = prices.index
    common_assets = prices.columns
    for name, panel in features.items():
        if not isinstance(panel.index, pd.DatetimeIndex):
            raise AlignmentError(f"Feature {name!r} must have a DatetimeIndex")
        common_dates = common_dates.intersection(panel.index)
        common_assets = common_assets.intersection(panel.columns)
    if common_dates.empty or common_assets.empty:
        raise AlignmentError("Features and prices have no common date/ticker observations")

    columns: list[pd.Series] = []
    for name, panel in features.items():
        series = panel.loc[common_dates, common_assets].stack(future_stack=True)
        series.name = name
        columns.append(series)
    labels = compute_forward_returns(prices, horizon=horizon, delay=delay)
    label = labels.loc[common_dates, common_assets].stack(future_stack=True)
    label.name = label_name
    dataset = pd.concat([*columns, label], axis=1)
    dataset.index.names = ["date", "ticker"]
    dataset = dataset.replace([np.inf, -np.inf], np.nan).sort_index()
    if drop_missing_label:
        dataset = dataset.dropna(subset=[label_name])
    dataset.attrs.update(
        label_name=label_name,
        horizon=int(horizon),
        delay=int(delay),
        feature_names=list(features),
    )
    return dataset


__all__ = ["build_supervised_dataset"]
