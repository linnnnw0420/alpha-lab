"""One-pass validation and alignment for backtest inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.data.contracts import PriceDataContract, normalize_price_panel
from alpha_lab.exceptions import AlignmentError, DataContractError


def validate_backtest_inputs(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    start_date: object,
    end_date: object,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if prices.empty:
        raise DataContractError("Price data is empty")
    normalized_prices = normalize_price_panel(
        prices,
        contract=PriceDataContract(missing="preserve"),
        start_date=start_date,
        end_date=end_date,
    )
    if normalized_prices.empty:
        raise DataContractError("No prices remain in the configured date range")
    if not isinstance(weights, pd.DataFrame):
        raise DataContractError("Weights must be a DataFrame")
    normalized_weights = weights.copy()
    try:
        index = pd.DatetimeIndex(
            pd.to_datetime(normalized_weights.index, errors="raise")
        ).normalize()
    except Exception as exc:
        raise DataContractError("Weight index must contain parseable dates") from exc
    if index.has_duplicates:
        raise AlignmentError("Weight dates must be unique")
    normalized_weights.index = index
    normalized_weights.index.name = "date"
    normalized_weights = normalized_weights.sort_index()
    unknown = normalized_weights.columns.difference(normalized_prices.columns)
    if len(unknown):
        raise AlignmentError(f"Weight tickers missing from prices: {unknown[:10].tolist()}")
    normalized_weights = normalized_weights.reindex(
        columns=normalized_prices.columns, fill_value=0.0
    )
    normalized_weights = normalized_weights.apply(pd.to_numeric, errors="coerce")
    normalized_weights = normalized_weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return normalized_prices, normalized_weights


__all__ = ["validate_backtest_inputs"]
