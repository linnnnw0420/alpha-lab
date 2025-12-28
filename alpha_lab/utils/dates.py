from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from alpha_lab.utils.typing import DateLike, PandasDatetimeIndex, PandasTimestamp, RebalanceFreq

# Late import pandas (optional dependency check)
try:
    import pandas as pd
except ImportError:
    pd = None # type: ignore[assignment]

# -----------------------------------------------------------------------------
# Frequency <-> annualization factor mapping
# -----------------------------------------------------------------------------

_ANNUALIZATION_FACTORS: dict[str, float] = {
    "D": 252.0, # trading days per year
    "W": 52.0, # weeks
    "M": 12.0, # months
}

def annualization_factor(freq: RebalanceFreq) -> float:
    """
    Return the annualization multiplier for a given frequency.

    Args: freq: 'D' (daily), 'W' (weekly) 'M' (monthly)

    Returns:
        Number of periods per year (252 for daily, 52 for weekly, 12 for monthly).
    
    Raises:
        ValueError: if freq is not recognized.
    """
    if freq not in _ANNUALIZATION_FACTORS:
        raise ValueError(f"Unknown freq: {freq!r}, must be one of the {list(_ANNUALIZATION_FACTORS.keys())}")
    return _ANNUALIZATION_FACTORS[freq]

# -----------------------------------------------------------------------------
# Date parsing & alignment
# -----------------------------------------------------------------------------

def parse_date(value: DateLike) -> PandasTimestamp:
    """
    Convert DateLike (str/date/datetime) into pd.Timestamp.

    Args:
        value: ISO date string, datetime.date, or datetime.datetime

    Returns:
        pd.Timestamp (timezone-naive)

    Raises:
        ImportError: if pandas is not installed
        ValueError: if string cannot be parsed
    """
    if pd is None:
        raise ImportError("pandas is required for parse_date()")

    if isinstance(value, pd.Timestamp):
        return value.normalize()
    if isinstance(value, datetime):
        return pd.Timestamp(value).normalize()
    if isinstance(value, date):
        return pd.Timestamp(value).normalize()
    if isinstance(value, str):
        return pd.Timestamp(value.strip()).normalize()
    raise TypeError(f"Cannot parse {type(value).__name__} as date") 

def align_to_trading_day(
    dt: DateLike,
    trading_calendar: PandasDatetimeIndex,
    method: Literal["previous", "next"] = "previous",
) -> PandasTimestamp:
    """
    Align a date to the nearest trading day in the calendar.

    Args:
        dt: target date
        trading_calendar: sorted DatetimeIndex of all trading days
        method: 'previous' (nearest past day) or 'next' (nearest future day)

    Returns:
        The aligned trading day as pd.Timestamp

    Raises:
        ValueError: if method is invalid or date is out of calendar bounds
    """
    if pd is None:
        raise ImportError("pandas is required for align_to_trading_day()")
    
    ts = parse_date(dt)
    

    if ts in trading_calendar:
        return ts
    
    if method == "previous":
        candidates = trading_calendar[trading_calendar <= ts]
        if candidates.empty:
            raise ValueError(f"No trading day on or before {ts}")
        return candidates[-1]
    elif method == "next":
        candidates = trading_calendar[trading_calendar >= ts]
        if candidates.empty:
            raise ValueError(f"No trading day on or after {ts}")
        return candidates[0]
    else:
        raise ValueError(f"method must be 'previous' or 'next', got: {method!r}")
    
# -----------------------------------------------------------------------------
# Rebalance date generation
# -----------------------------------------------------------------------------

def generate_rebalance_dates(
    trading_calendar: PandasDatetimeIndex,
    freq: RebalanceFreq,
) -> PandasDatetimeIndex:
    """
    Generate rebalance dates from a full trading calendar.

    Args:
        trading_calendar: complete sorted DatetimeIndex of all trading days
        freq: 'D' (daily), 'W' (weekly), or 'M' (monthly)

    Returns:
        Subset of trading_calendar at the specified frequency

    Notes:
        - Daily: all days
        - Weekly: last trading day of each week
        - Monthly: last trading day of each month
    """
    if pd is None:
        raise ImportError("pandas is required for generate_rebalance_dates")
    
    if freq == "D":
        return trading_calendar
    
    # Convert to Series with dummy values to use resample
    s = pd.Series(1, index=trading_calendar)

    if freq == "W":
        # resample to week-end, take last available day
        resampled = s.resample("W").last()
    elif freq == "M":
        # resample to month-end, take last available day
        resampled = s.resample("M").last()
    else:
        raise ValueError(f"Unknown freq: {freq!r}")
    
    # Drop NaN (in case there's a gap), return index
    return resampled.dropna().index

__all__ = [
    "annualization_factor",
    "parse_date",
    "align_to_trading_day",
    "generate_rebalance_dates",
]