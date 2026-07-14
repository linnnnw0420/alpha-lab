"""Signal, rebalance, and execution-date mapping."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from alpha_lab.exceptions import AlignmentError, ConfigurationError


@dataclass(frozen=True)
class ExecutionSchedule:
    target_weights: pd.DataFrame
    signal_dates: pd.Series


def build_execution_schedule(
    weights: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
    rebalance_dates: pd.DatetimeIndex,
    delay_days: int,
) -> ExecutionSchedule:
    if delay_days < 0:
        raise ConfigurationError("delay_days must be non-negative")
    calendar = pd.DatetimeIndex(trading_calendar).normalize()
    signals = weights.index.intersection(pd.DatetimeIndex(rebalance_dates).normalize())
    if signals.empty:
        empty = weights.iloc[0:0].copy()
        return ExecutionSchedule(empty, pd.Series(dtype="datetime64[ns]", name="signal_date"))
    locations = calendar.get_indexer(signals)
    if (locations < 0).any():
        raise AlignmentError("Some signal dates are absent from the trading calendar")
    execution_locations = locations + delay_days
    valid = execution_locations < len(calendar)
    selected_signals = signals[valid]
    execution_dates = calendar[execution_locations[valid]]
    scheduled = weights.loc[selected_signals].copy()
    scheduled.index = execution_dates
    signal_series = pd.Series(selected_signals, index=execution_dates, name="signal_date")
    if scheduled.index.has_duplicates:
        keep = ~scheduled.index.duplicated(keep="last")
        scheduled = scheduled.loc[keep]
        signal_series = signal_series.loc[keep]
    return ExecutionSchedule(scheduled.sort_index(), signal_series.sort_index())


__all__ = ["ExecutionSchedule", "build_execution_schedule"]
