"""Backtest result container and serialization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.config.backtest import BacktestConfig
from alpha_lab.utils.dates import annualization_factor


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    positions: pd.DataFrame
    returns: pd.Series
    config: BacktestConfig
    trades: pd.DataFrame | None = None

    def total_return(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        return float(self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1.0)

    def save_trades(self, path: str | None = None) -> str | None:
        if self.trades is None or self.trades.empty:
            return None
        if path is None:
            from alpha_lab.config import get_paths

            destination = get_paths().artifacts_dir / "trades.csv"
        else:
            destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.trades.to_csv(destination, index=False)
        return str(destination)

    def summary_stats(self) -> dict[str, float]:
        if self.returns.empty:
            return {}
        factor = annualization_factor("D")
        mean = float(self.returns.mean())
        std = float(self.returns.std())
        annualized_return = mean * factor
        annualized_vol = std * np.sqrt(factor)
        return {
            "total_return": self.total_return(),
            "annualized_return": annualized_return,
            "annualized_vol": annualized_vol,
            "sharpe_ratio": annualized_return / annualized_vol if annualized_vol > 1e-12 else 0.0,
            "n_days": float(len(self.returns)),
        }


__all__ = ["BacktestResult"]
