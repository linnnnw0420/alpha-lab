"""Whole-date expanding and rolling walk-forward splits."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from alpha_lab.exceptions import ConfigurationError, LookaheadError


@dataclass(frozen=True)
class WalkForwardSplit:
    min_train_dates: int
    test_dates: int = 1
    step_dates: int | None = None
    mode: Literal["expanding", "rolling"] = "expanding"
    train_window_dates: int | None = None
    gap_dates: int = 0

    def __post_init__(self) -> None:
        if self.min_train_dates < 1 or self.test_dates < 1:
            raise ConfigurationError("min_train_dates and test_dates must be >= 1")
        if self.gap_dates < 0:
            raise ConfigurationError("gap_dates must be non-negative")
        if self.mode == "rolling" and not self.train_window_dates:
            raise ConfigurationError("rolling mode requires train_window_dates")
        if self.train_window_dates is not None and self.train_window_dates < self.min_train_dates:
            raise ConfigurationError("train_window_dates must be >= min_train_dates")

    def split(
        self, observations: pd.DataFrame | pd.Series | pd.Index
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        dates = _observation_dates(observations)
        unique_dates = pd.DatetimeIndex(dates.unique()).sort_values()
        step = self.step_dates or self.test_dates
        test_start = self.min_train_dates + self.gap_dates
        while test_start < len(unique_dates):
            test_end = min(test_start + self.test_dates, len(unique_dates))
            train_end = test_start - self.gap_dates
            train_start = 0
            if self.mode == "rolling":
                train_start = max(0, train_end - int(self.train_window_dates or 0))
            train_dates = unique_dates[train_start:train_end]
            test_window = unique_dates[test_start:test_end]
            if len(train_dates) < self.min_train_dates:
                test_start += step
                continue
            if train_dates.max() >= test_window.min():
                raise LookaheadError("Training dates must precede test dates")
            train_index = np.flatnonzero(dates.isin(train_dates))
            test_index = np.flatnonzero(dates.isin(test_window))
            if len(train_index) and len(test_index):
                yield train_index, test_index
            test_start += step


def _observation_dates(observations: pd.DataFrame | pd.Series | pd.Index) -> pd.DatetimeIndex:
    index = observations if isinstance(observations, pd.Index) else observations.index
    if isinstance(index, pd.MultiIndex):
        level = "date" if "date" in index.names else 0
        values = index.get_level_values(level)
    else:
        values = index
    try:
        return pd.DatetimeIndex(pd.to_datetime(values, errors="raise")).normalize()
    except Exception as exc:
        raise LookaheadError("Walk-forward observations require parseable dates") from exc


__all__ = ["WalkForwardSplit"]
