"""Backtest orchestration with backward-compatible public entry points."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from alpha_lab.backtest.accounting import simulate_portfolio
from alpha_lab.backtest.result import BacktestResult
from alpha_lab.backtest.schedule import build_execution_schedule
from alpha_lab.backtest.validation import validate_backtest_inputs
from alpha_lab.config.backtest import BacktestConfig, default_backtest_config
from alpha_lab.config.universe import UniverseConfig
from alpha_lab.data.loader import load_prices
from alpha_lab.portfolio.rebalance import generate_rebalance_schedule
from alpha_lab.utils.typing import Ticker


def run_backtest(
    weights: pd.DataFrame,
    prices: pd.DataFrame | None = None,
    config: BacktestConfig | None = None,
    universe: UniverseConfig | list[Ticker] | tuple[Ticker, ...] | None = None,
    execution_field: Literal["open", "close"] = "close",
    record_trades: bool = True,
    verbose: bool = False,
) -> BacktestResult:
    """Convert target weights and prices into an equity curve."""
    del verbose  # retained for API compatibility; logging belongs outside the hot path
    config = config or default_backtest_config()
    if prices is None:
        if universe is None:
            raise ValueError("Must provide universe if prices is None")
        prices = load_prices(
            universe,
            config.start_date,
            config.end_date,
            field=execution_field,
            align_dates=True,
        )
    prices, weights = validate_backtest_inputs(
        prices,
        weights,
        start_date=config.start_date,
        end_date=config.end_date,
    )
    rebalance_dates = generate_rebalance_schedule(
        prices.index,
        config.start_date,
        config.end_date,
        config.rebalance_freq,
    )
    schedule = build_execution_schedule(
        weights,
        prices.index,
        rebalance_dates,
        int(config.execution_delay_days),
    )
    output = simulate_portfolio(
        prices,
        schedule.target_weights,
        config,
        signal_dates=schedule.signal_dates,
        record_trades=record_trades,
    )
    returns = output.equity_curve.pct_change().fillna(0.0).rename("return")
    return BacktestResult(
        equity_curve=output.equity_curve,
        positions=output.positions,
        returns=returns,
        config=config,
        trades=output.trades,
    )


def _align_weights_to_schedule(
    weights: pd.DataFrame, rebalance_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Compatibility helper retained for downstream notebooks."""
    return weights.loc[weights.index.intersection(rebalance_dates)]


def _shift_weights_by_trading_day(
    weights: pd.DataFrame, trading_calendar: pd.DatetimeIndex, delay_days: int
) -> pd.DataFrame:
    schedule = build_execution_schedule(weights, trading_calendar, weights.index, delay_days)
    return schedule.target_weights


def _simulate_portfolio(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    config: BacktestConfig,
    record_trades: bool,
    verbose: bool = False,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame | None]:
    del verbose
    output = simulate_portfolio(prices, weights, config, record_trades=record_trades)
    return output.equity_curve, output.positions, output.trades


__all__ = ["BacktestResult", "run_backtest"]
