from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, TypeAlias

# -----------------------------------------------------------------------------
# Optional third-party typing: do NOT make pandas/numpy a hard dependency here.
# If installed, provide concrete types; otherwise fall back to Any.
# -----------------------------------------------------------------------------

try:  # pragma: no cover
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore[assignment]

try:  # pragma: no cover
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

# -----------------------------------------------------------------------------
# Core domain primitives (keep minimal and stable)
# -----------------------------------------------------------------------------

Ticker: TypeAlias = str
Asset: TypeAlias = str
FactorName: TypeAlias = str
ModelName: TypeAlias = str
RunName: TypeAlias = str

Bps: TypeAlias = float  # basis points, e.g. 5.0 = 5 bps

# -----------------------------------------------------------------------------
# Date/time related
# -----------------------------------------------------------------------------

# Keep DateLike broad; normalize/validate should happen in utils.dates or config objects.

DateLike: TypeAlias = str | date | datetime

# -----------------------------------------------------------------------------
# Common literals used across modules (safe, no dependencies)
# -----------------------------------------------------------------------------

RebalanceFreq: TypeAlias = Literal["D", "W", "M"]  # day/week/month
PriceField: TypeAlias = Literal["open", "high", "low", "close", "vwap"]

# -----------------------------------------------------------------------------
# Pandas/Numpy-ish aliases (precise if installed, Any if not)
# -----------------------------------------------------------------------------

if pd is not None:
    PandasDataFrame: TypeAlias = pd.DataFrame
    PandasSeries: TypeAlias = pd.Series
    PandasIndex: TypeAlias = pd.Index
    PandasTimestamp: TypeAlias = pd.Timestamp
    PandasDatetimeIndex: TypeAlias = pd.DatetimeIndex
else:
    PandasDataFrame: TypeAlias = Any
    PandasSeries: TypeAlias = Any
    PandasIndex: TypeAlias = Any
    PandasTimestamp: TypeAlias = Any
    PandasDatetimeIndex: TypeAlias = Any

if np is not None:
    NpArray: TypeAlias = np.ndarray
else:
    NpArray: TypeAlias = Any


# -----------------------------------------------------------------------------
# Higher-level “shapes” used in this project (document intent)
# -----------------------------------------------------------------------------

# Long-form panel: MultiIndex(date, asset) -> columns: OHLCV etc
PricePanel: TypeAlias = PandasDataFrame

# Factor values: MultiIndex(date, asset) -> columns: factor names
FactorPanel: TypeAlias = PandasDataFrame

# Weights: index: asset, value: weight
WeightVector: TypeAlias = PandasSeries

# Equity curve: index: date, value: equity
EquityCurve: TypeAlias = PandasSeries

# Trades table: each row is a fill/trade record
TradeFrame: TypeAlias = PandasDataFrame

# -----------------------------------------------------------------------------
# Misc path aliases (standard library only)
# -----------------------------------------------------------------------------

PathLike: TypeAlias = str | Path


__all__ = [
    # primitives
    "Ticker",
    "Asset",
    "FactorName",
    "ModelName",
    "RunName",
    "Bps",
    # date/time
    "DateLike",
    # literals
    "RebalanceFreq",
    "PriceField",
    # pandas/numpy
    "PandasDataFrame",
    "PandasSeries",
    "PandasIndex",
    "PandasTimestamp",
    "PandasDatetimeIndex",
    "NpArray",
    # project shapes
    "PricePanel",
    "FactorPanel",
    "WeightVector",
    "EquityCurve",
    "TradeFrame",
    # paths
    "PathLike",
]
