"""Local Parquet price source."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from alpha_lab.data.contracts import PriceDataContract, normalize_price_panel
from alpha_lab.exceptions import DataContractError


@dataclass(frozen=True)
class ParquetDataSource:
    path: Path | str
    contract: PriceDataContract = PriceDataContract()

    def load_prices(
        self,
        tickers: list[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        field: str = "close",
    ) -> pd.DataFrame:
        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")
        filters = [("date", ">=", start_date), ("date", "<=", end_date)]
        try:
            frame = pd.read_parquet(path, columns=list(tickers), filters=filters)
        except (KeyError, ValueError, TypeError):
            try:
                frame = pd.read_parquet(path, columns=list(tickers))
            except (KeyError, ValueError):
                frame = pd.read_parquet(path)
        except ImportError as exc:
            raise ImportError("Parquet support requires `pip install alpha-lab[data]`") from exc
        except Exception:
            frame = pd.read_parquet(path)
        if "date" in frame.columns and not isinstance(frame.index, pd.DatetimeIndex):
            frame = frame.set_index("date")
        if "ticker" in frame.columns:
            if field not in frame.columns:
                raise DataContractError(f"Long Parquet data is missing field {field!r}")
            if frame.index.name == "date":
                frame = frame.reset_index()
            frame = frame.pivot(index="date", columns="ticker", values=field)
        return normalize_price_panel(
            frame,
            contract=self.contract,
            start_date=start_date,
            end_date=end_date,
            tickers=tickers,
        )


def load_parquet_prices(
    path: Path | str,
    tickers: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    field: str = "close",
) -> pd.DataFrame:
    return ParquetDataSource(path).load_prices(tickers, start_date, end_date, field)


__all__ = ["ParquetDataSource", "load_parquet_prices"]
