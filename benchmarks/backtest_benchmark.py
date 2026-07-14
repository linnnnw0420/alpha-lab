"""Repeatable benchmark for the NumPy accounting hot path."""

from time import perf_counter

import numpy as np
import pandas as pd

from alpha_lab.backtest.accounting import simulate_portfolio
from alpha_lab.config.backtest import BacktestConfig


def slow_reference(prices: pd.DataFrame, weights: pd.DataFrame, initial_cash: float) -> pd.Series:
    holdings = pd.Series(0.0, index=prices.columns)
    cash = initial_cash
    equity = pd.Series(index=prices.index, dtype=float)
    for date in prices.index:
        row = prices.loc[date]
        value = cash + (holdings * row).sum()
        if date in weights.index:
            target = weights.loc[date]
            desired = target * value / row
            trades = desired - holdings
            cash -= (trades * row).sum()
            holdings = desired
        equity.loc[date] = cash + (holdings * row).sum()
    return equity


def main() -> None:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2015-01-01", periods=2_000, freq="B")
    columns = [f"S{number:03d}" for number in range(196)]
    returns = rng.normal(0.0002, 0.015, size=(len(dates), len(columns)))
    prices = pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=columns)
    weights = pd.DataFrame(
        [np.full(len(columns), 1 / len(columns))], index=dates[:1], columns=columns
    )
    config = BacktestConfig(
        start_date=dates[0],
        end_date=dates[-1],
        rebalance_freq="D",
        initial_cash=1_000_000,
        commission_bps=0,
        slippage_bps=0,
        price_field="close",
        max_turnover=1,
        execution_delay_days=0,
    )
    started = perf_counter()
    reference = slow_reference(prices, weights, config.initial_cash)
    reference_seconds = perf_counter() - started
    started = perf_counter()
    optimized = simulate_portfolio(prices, weights, config, record_trades=False).equity_curve
    optimized_seconds = perf_counter() - started
    np.testing.assert_allclose(optimized, reference, rtol=1e-11, atol=1e-6)
    print(
        {
            "dates": len(dates),
            "assets": len(columns),
            "reference_seconds": round(reference_seconds, 4),
            "optimized_seconds": round(optimized_seconds, 4),
            "speedup": round(reference_seconds / optimized_seconds, 2),
        }
    )


if __name__ == "__main__":
    main()
