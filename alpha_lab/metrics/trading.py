"""
Trading-related metrics calculations.
交易相关指标计算模块.

Key functions / 核心函数:
- compute_trading_metrics: turnover / holdings / concentration / cost metrics
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.utils.logging import get_logger
from alpha_lab.utils.math import safe_divide
from alpha_lab.utils.typing import PandasDataFrame, PandasSeries

logger = get_logger(__name__)

_WEIGHT_EPS = 1e-12


def compute_trading_metrics(
    equity_curve: PandasSeries,
    positions: PandasDataFrame | None = None,
    trades: PandasDataFrame | None = None,
    commission_bps: float | None = None,
    slippage_bps: float | None = None,
) -> dict[str, float]:
    """
    Compute trading/portfolio health metrics.
    计算交易与组合健康度指标.

    Metrics (if data available) / 指标(有数据则计算):
        - turnover_avg / turnover_median / turnover_max
        - n_trade_days / turnover_total
        - holdings_avg / holdings_max
        - hhi_avg / hhi_max (concentration)
        - cost_est_total / cost_est_pct (estimated transaction cost)

    Args / 参数:
        equity_curve: 净值序列 (date -> equity value)
        positions: 每日持仓权重矩阵 (date x asset)
        trades: 交易记录表 (date, asset, shares, price, value)
        commission_bps: 手续费(基点); None 表示 0
        slippage_bps: 滑点(基点); None 表示 0

    Returns / 返回:
        dict of metrics
    """
    if equity_curve is None or equity_curve.empty:
        logger.warning("Empty equity curve for trading metrics")
        return {}

    metrics: dict[str, float] = {}

    # ---------------------------------------------------------------------
    # Turnover metrics (from trades)
    # ---------------------------------------------------------------------
    if trades is not None and not trades.empty:
        trade_values = _aggregate_trade_value(trades)
        if not trade_values.empty:
            equity_on_trade_days = equity_curve.reindex(trade_values.index).dropna()
            aligned = trade_values.loc[equity_on_trade_days.index]

            turnover = aligned / equity_on_trade_days
            turnover = turnover.replace([np.inf, -np.inf], 0.0).fillna(0.0)

            metrics["turnover_avg"] = float(turnover.mean())
            metrics["turnover_median"] = float(turnover.median())
            metrics["turnover_max"] = float(turnover.max())
            metrics["n_trade_days"] = float(turnover.shape[0])

            turnover_total = safe_divide(aligned.sum(), equity_curve.iloc[0], default=0.0)
            metrics["turnover_total"] = float(turnover_total)

            # Estimated transaction cost (bps-based)
            commission = 0.0 if commission_bps is None else float(commission_bps)
            slippage = 0.0 if slippage_bps is None else float(slippage_bps)
            cost_rate = (commission + slippage) / 10000.0
            if cost_rate > 0:
                cost_total = aligned.sum() * cost_rate
                metrics["cost_est_total"] = float(cost_total)
                metrics["cost_est_pct"] = float(
                    safe_divide(cost_total, equity_curve.iloc[0], default=0.0)
                )
        else:
            logger.debug("Trades provided but no valid trade values")

    # ---------------------------------------------------------------------
    # Holdings / concentration metrics (from positions)
    # ---------------------------------------------------------------------
    if positions is not None and not positions.empty:
        weights = positions.fillna(0.0).astype(float)
        weights_abs = weights.abs()

        holdings = (weights_abs > _WEIGHT_EPS).sum(axis=1)
        if not holdings.empty:
            metrics["holdings_avg"] = float(holdings.mean())
            metrics["holdings_max"] = float(holdings.max())

        # Normalize by invested weight (ignore cash)
        weight_sum = weights_abs.sum(axis=1).replace(0.0, np.nan)
        weights_norm = weights_abs.div(weight_sum, axis=0)
        hhi = (weights_norm**2).sum(axis=1).fillna(0.0)
        if not hhi.empty:
            metrics["hhi_avg"] = float(hhi.mean())
            metrics["hhi_max"] = float(hhi.max())

    if not metrics:
        logger.debug("No trading metrics computed (missing trades/positions)")

    return metrics


def _aggregate_trade_value(trades: PandasDataFrame) -> PandasSeries:
    """
    Aggregate absolute trade value by date.
    将交易记录按日期汇总成绝对成交金额.
    """
    if trades is None or trades.empty:
        return pd.Series(dtype=float)

    if "date" not in trades.columns:
        raise ValueError("trades must have 'date' column")

    if "value" in trades.columns:
        values = trades["value"].astype(float).abs()
    elif {"shares", "price"} <= set(trades.columns):
        values = (trades["shares"].astype(float) * trades["price"].astype(float)).abs()
    else:
        raise ValueError("trades must have 'value' or ('shares' and 'price') columns")

    dates = pd.to_datetime(trades["date"], errors="coerce").dt.normalize()
    if dates.isna().any():
        logger.warning("Some trade dates failed to parse; dropping invalid rows")

    df = pd.DataFrame({"date": dates, "value": values})
    df = df.dropna(subset=["date"])
    if df.empty:
        return pd.Series(dtype=float)

    return df.groupby("date")["value"].sum()


__all__ = ["compute_trading_metrics"]
