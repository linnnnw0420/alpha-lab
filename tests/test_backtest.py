import numpy as np
import pandas as pd
import pytest

from alpha_lab.backtest.engine import BacktestResult, run_backtest
from alpha_lab.config.backtest import BacktestConfig
from alpha_lab.exceptions import AlignmentError, DataContractError


def _config(dates: pd.DatetimeIndex, **updates) -> BacktestConfig:
    values = dict(
        start_date=dates[0],
        end_date=dates[-1],
        rebalance_freq="D",
        initial_cash=1_000.0,
        commission_bps=0.0,
        slippage_bps=0.0,
        price_field="close",
        max_turnover=1.0,
        rebalance_threshold=0.0,
        execution_delay_days=0,
    )
    values.update(updates)
    return BacktestConfig(**values)


def test_one_asset_buy_and_hold() -> None:
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]}, index=dates)
    weights = pd.DataFrame({"A": [1.0]}, index=dates[:1])
    result = run_backtest(weights, prices, _config(dates))
    np.testing.assert_allclose(result.equity_curve, [1000.0, 1100.0, 1210.0])
    assert isinstance(result, BacktestResult)
    assert result.trades is not None
    assert {"signal_date", "execution_date", "side", "cost"} <= set(result.trades)


def test_execution_delay_uses_later_price() -> None:
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]}, index=dates)
    weights = pd.DataFrame({"A": [1.0]}, index=dates[:1])
    result = run_backtest(weights, prices, _config(dates, execution_delay_days=1))
    np.testing.assert_allclose(result.equity_curve, [1000.0, 1000.0, 1100.0])
    assert result.trades.iloc[0]["signal_date"] == dates[0]
    assert result.trades.iloc[0]["execution_date"] == dates[1]


def test_costs_reduce_equity() -> None:
    dates = pd.date_range("2024-01-02", periods=2, freq="B")
    prices = pd.DataFrame({"A": [100.0, 100.0]}, index=dates)
    weights = pd.DataFrame({"A": [1.0]}, index=dates[:1])
    free = run_backtest(weights, prices, _config(dates))
    costly = run_backtest(weights, prices, _config(dates, commission_bps=100.0))
    assert costly.equity_curve.iloc[-1] < free.equity_curve.iloc[-1]
    assert costly.trades["cost"].sum() == pytest.approx(10.0)


def test_turnover_cap_limits_initial_investment() -> None:
    dates = pd.date_range("2024-01-02", periods=2, freq="B")
    prices = pd.DataFrame({"A": [100.0, 100.0]}, index=dates)
    weights = pd.DataFrame({"A": [1.0]}, index=dates[:1])
    result = run_backtest(weights, prices, _config(dates, max_turnover=0.3))
    assert result.trades["notional"].sum() == pytest.approx(300.0)
    assert result.positions.iloc[0, 0] == pytest.approx(0.3)


def test_empty_weights_hold_cash(price_panel: pd.DataFrame, daily_config: BacktestConfig) -> None:
    result = run_backtest(
        pd.DataFrame(index=[], columns=price_panel.columns), price_panel, daily_config
    )
    assert (result.equity_curve == daily_config.initial_cash).all()


def test_invalid_inputs_are_explicit(
    price_panel: pd.DataFrame, daily_config: BacktestConfig
) -> None:
    with pytest.raises(DataContractError, match="empty"):
        run_backtest(pd.DataFrame(), pd.DataFrame(), daily_config)
    weights = pd.DataFrame({"UNKNOWN": [1.0]}, index=price_panel.index[:1])
    with pytest.raises(AlignmentError, match="missing"):
        run_backtest(weights, price_panel, daily_config)


def test_missing_execution_price_does_not_trade(daily_config: BacktestConfig) -> None:
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    prices = pd.DataFrame({"A": [np.nan, 100.0, 110.0]}, index=dates)
    weights = pd.DataFrame({"A": [1.0]}, index=dates[:1])
    result = run_backtest(
        weights, prices, daily_config.with_updates(start_date=dates[0], end_date=dates[-1])
    )
    assert result.trades is None
    assert (result.equity_curve == 1000.0).all()
