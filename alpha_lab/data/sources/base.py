"""Data-source protocol used by local and remote adapters."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class DataSource(Protocol):
    def load_prices(
        self,
        tickers: list[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        field: str = "close",
    ) -> pd.DataFrame: ...


__all__ = ["DataSource"]
