"""NumPy portfolio accounting core."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.config.backtest import BacktestConfig


@dataclass
class AccountingOutput:
    equity_curve: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame | None


def simulate_portfolio(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    config: BacktestConfig,
    *,
    signal_dates: pd.Series | None = None,
    record_trades: bool = True,
) -> AccountingOutput:
    dates = prices.index
    assets = prices.columns
    n_dates, n_assets = prices.shape
    raw_prices = prices.to_numpy(dtype=float, copy=False)
    mark_prices = pd.DataFrame(raw_prices, index=dates).ffill().to_numpy()
    mark_prices = np.where(np.isfinite(mark_prices) & (mark_prices > 0), mark_prices, 0.0)
    targets = target_weights.to_numpy(dtype=float, copy=False)
    execution_lookup = {date: position for position, date in enumerate(target_weights.index)}

    equity = np.empty(n_dates, dtype=float)
    positions = np.zeros((n_dates, n_assets), dtype=float)
    holdings = np.zeros(n_assets, dtype=float)
    cash = float(config.initial_cash)
    cost_rate = (float(config.commission_bps) + float(config.slippage_bps)) / 10_000.0
    trade_records: list[dict[str, object]] = []

    for day in range(n_dates):
        execution_prices = raw_prices[day]
        valuation_prices = mark_prices[day]
        portfolio_value = cash + float(np.dot(holdings, valuation_prices))
        execution_position = execution_lookup.get(dates[day])
        if execution_position is not None:
            target = np.nan_to_num(targets[execution_position], nan=0.0, posinf=0.0, neginf=0.0)
            tradable = np.isfinite(execution_prices) & (execution_prices > 0)
            current_weights = np.divide(
                holdings * valuation_prices,
                portfolio_value,
                out=np.zeros(n_assets),
                where=portfolio_value > 1e-12,
            )
            desired_shares = holdings.copy()
            desired_shares[tradable] = (
                target[tradable] * portfolio_value / execution_prices[tradable]
            )
            share_delta = desired_shares - holdings
            notional = np.zeros(n_assets, dtype=float)
            notional[tradable] = share_delta[tradable] * execution_prices[tradable]
            gross_notional = float(np.abs(notional).sum())
            turnover = gross_notional / portfolio_value if portfolio_value > 1e-12 else 0.0

            if turnover >= float(config.rebalance_threshold):
                if turnover > float(config.max_turnover) and turnover > 0:
                    scale = float(config.max_turnover) / turnover
                    adjusted = current_weights + scale * (target - current_weights)
                    desired_shares[tradable] = (
                        adjusted[tradable] * portfolio_value / execution_prices[tradable]
                    )
                    share_delta = desired_shares - holdings
                    notional[:] = 0.0
                    notional[tradable] = share_delta[tradable] * execution_prices[tradable]
                    gross_notional = float(np.abs(notional).sum())

                total_cost = gross_notional * cost_rate
                cash -= float(notional.sum()) + total_cost
                holdings = desired_shares
                if record_trades and gross_notional > 1e-12:
                    signal = (
                        signal_dates.get(dates[day], dates[day])
                        if signal_dates is not None
                        else dates[day]
                    )
                    traded = np.flatnonzero(np.abs(share_delta) > 1e-12)
                    for asset_position in traded:
                        absolute = abs(notional[asset_position])
                        trade_records.append(
                            {
                                "signal_date": pd.Timestamp(signal),
                                "date": dates[day],
                                "execution_date": dates[day],
                                "asset": assets[asset_position],
                                "side": "buy" if share_delta[asset_position] > 0 else "sell",
                                "shares": share_delta[asset_position],
                                "price": execution_prices[asset_position],
                                "value": notional[asset_position],
                                "notional": absolute,
                                "cost": total_cost * absolute / gross_notional,
                            }
                        )

        portfolio_value = cash + float(np.dot(holdings, valuation_prices))
        equity[day] = portfolio_value
        if abs(portfolio_value) > 1e-12:
            positions[day] = holdings * valuation_prices / portfolio_value

    trades = pd.DataFrame(trade_records) if trade_records else None
    return AccountingOutput(
        equity_curve=pd.Series(equity, index=dates, name="equity"),
        positions=pd.DataFrame(positions, index=dates, columns=assets),
        trades=trades,
    )


__all__ = ["AccountingOutput", "simulate_portfolio"]
