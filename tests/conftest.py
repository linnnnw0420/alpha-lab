import numpy as np
import pandas as pd
import pytest

from alpha_lab.config.backtest import BacktestConfig


@pytest.fixture
def dates() -> pd.DatetimeIndex:
    return pd.date_range("2024-01-02", periods=12, freq="B", name="date")


@pytest.fixture
def price_panel(dates: pd.DatetimeIndex) -> pd.DataFrame:
    values = np.arange(len(dates), dtype=float)[:, None]
    return pd.DataFrame(
        100.0 + values * np.array([[1.0, 1.5, 0.5, 2.0]]),
        index=dates,
        columns=["A", "B", "C", "D"],
    )


@pytest.fixture
def daily_config(dates: pd.DatetimeIndex) -> BacktestConfig:
    return BacktestConfig(
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
