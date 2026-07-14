"""Canonical schemas and fingerprints for research data."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

import numpy as np
import pandas as pd

from alpha_lab.exceptions import DataContractError

MissingPolicy = Literal["preserve", "ffill", "drop"]
DuplicatePolicy = Literal["raise", "last", "mean"]


@dataclass(frozen=True)
class PriceDataContract:
    """Rules used to normalize a wide date-by-ticker price panel."""

    missing: MissingPolicy = "preserve"
    forward_fill_limit: int | None = 5
    duplicates: DuplicatePolicy = "raise"
    dtype: Literal["float64", "float32"] = "float64"
    adjusted: bool | None = None

    def __post_init__(self) -> None:
        if self.missing not in {"preserve", "ffill", "drop"}:
            raise DataContractError(f"Unknown missing policy: {self.missing!r}")
        if self.duplicates not in {"raise", "last", "mean"}:
            raise DataContractError(f"Unknown duplicate policy: {self.duplicates!r}")
        if self.forward_fill_limit is not None and self.forward_fill_limit < 0:
            raise DataContractError("forward_fill_limit must be non-negative or None")


def normalize_price_panel(
    panel: pd.DataFrame,
    *,
    contract: PriceDataContract | None = None,
    start_date: object | None = None,
    end_date: object | None = None,
    tickers: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Return a deterministic canonical price panel."""
    contract = contract or PriceDataContract()
    if not isinstance(panel, pd.DataFrame):
        raise DataContractError(f"prices must be a DataFrame, got {type(panel).__name__}")

    result = panel.copy()
    try:
        index = pd.to_datetime(result.index, errors="raise")
    except Exception as exc:
        raise DataContractError("Price index must contain parseable dates") from exc
    index = pd.DatetimeIndex(index)
    if index.tz is not None:
        index = index.tz_convert(None)
    result.index = index.normalize()
    result.index.name = "date"

    columns = [str(column).strip() for column in result.columns]
    if any(not column for column in columns):
        raise DataContractError("Ticker columns must be non-empty strings")
    if len(set(columns)) != len(columns):
        duplicates = sorted({column for column in columns if columns.count(column) > 1})
        raise DataContractError(f"Duplicate ticker columns: {duplicates[:10]}")
    result.columns = columns

    if result.index.has_duplicates:
        duplicate_dates = result.index[result.index.duplicated(keep=False)].unique()
        if contract.duplicates == "raise":
            raise DataContractError(
                f"Duplicate price dates: {duplicate_dates[:5].strftime('%Y-%m-%d').tolist()}"
            )
        if contract.duplicates == "last":
            result = result[~result.index.duplicated(keep="last")]
        else:
            result = result.groupby(level=0, sort=True).mean(numeric_only=True)

    try:
        result = result.apply(pd.to_numeric, errors="raise").astype(contract.dtype)
    except Exception as exc:
        raise DataContractError("Price values must be numeric") from exc
    result = result.replace([np.inf, -np.inf], np.nan).sort_index()

    if start_date is not None:
        result = result.loc[result.index >= pd.Timestamp(start_date).normalize()]
    if end_date is not None:
        result = result.loc[result.index <= pd.Timestamp(end_date).normalize()]

    if tickers is not None:
        requested = list(dict.fromkeys(str(ticker).strip() for ticker in tickers))
        result = result.reindex(columns=requested)

    if contract.missing == "ffill":
        result = result.ffill(limit=contract.forward_fill_limit)
    elif contract.missing == "drop":
        result = result.dropna(how="any")
    return result


def fingerprint_frame(frame: pd.DataFrame | pd.Series) -> str:
    """Create a stable content fingerprint including labels and values."""
    digest = sha256()
    digest.update(type(frame).__name__.encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    if isinstance(frame, pd.DataFrame):
        digest.update(pd.util.hash_pandas_object(pd.Index(frame.columns)).values.tobytes())
        digest.update("|".join(map(str, frame.dtypes)).encode())
    else:
        digest.update(str(frame.dtype).encode())
        digest.update(str(frame.name).encode())
    return digest.hexdigest()


__all__ = [
    "DuplicatePolicy",
    "MissingPolicy",
    "PriceDataContract",
    "fingerprint_frame",
    "normalize_price_panel",
]
