"""
Performance metrics calculations.
业绩指标计算模块.

Key functions / 核心函数:
- compute_cagr: 年化复合增长率 / compound annual growth rate
- compute_annualized_vol: 年化波动率 / annualized volatility
- compute_sharpe_ratio: 风险调整收益(夏普比率)/ risk-adjusted return
- compute_performance_metrics: 综合业绩统计 / comprehensive performance stats

业绩指标概述 / Performance Metrics Overview:
    - CAGR: 衡量长期平均年化收益
    - Volatility: 衡量收益的不确定性/风险
    - Sharpe Ratio: 每单位风险获得的超额收益,越高越好(>1 较好,>2 优秀)
    - Sortino Ratio: 只考虑下行风险的夏普比率
"""

from __future__ import annotations

import numpy as np

from alpha_lab.utils.dates import annualization_factor
from alpha_lab.utils.logging import get_logger
from alpha_lab.utils.math import safe_divide
from alpha_lab.utils.typing import PandasSeries, RebalanceFreq

logger = get_logger(__name__)


def compute_cagr(
    equity_curve: PandasSeries,
    freq: RebalanceFreq = "D",
) -> float:
    """
    Compute compound annual growth rate (CAGR).
    计算年化复合增长率.

    公式 / Formula:
        CAGR = (终值 / 初值) ^ (1 / 年数) - 1
        CAGR = (end_value / start_value) ^ (1 / years) - 1

    为什么用 CAGR / Why CAGR:
        - 消除不同时长回测的可比性问题
        - 考虑复利效应

    Args / 参数:
        equity_curve: 净值序列 (date -> equity value)
        freq: 数据频率 ('D' 日, 'W' 周, 'M' 月)

    Returns / 返回:
        CAGR 小数形式 (如 0.15 = 年化 15%)
    """

    if equity_curve.empty or len(equity_curve) < 2:
        logger.warning("Equity curve too short for CAGR calculation")
        return 0.0

    start_value = equity_curve.iloc[0]
    end_value = equity_curve.iloc[-1]

    if start_value <= 0:
        logger.warning(f"Invalid start value: {start_value}")
        return 0.0

    # Number of periods
    n_periods = len(equity_curve) - 1
    if n_periods <= 0:
        return 0.0

    # 年化因子 / Annualization factor
    ann_factor = annualization_factor(freq)

    # CAGR = (终值 / 初值) ^ (年化因子 / 期数) - 1
    # CAGR = (end_value / start_value) ^ (ann_factor / n_periods) - 1
    total_return = end_value / start_value
    if total_return <= 0:
        logger.warning(f"Negative total return: {total_return - 1:.2%}")
        return -1.0

    years = n_periods / ann_factor  # 转换为年数
    if years <= 0:
        logger.warning("Zero or negative years for CAGR calculation")
        return 0.0

    cagr = total_return ** (1.0 / years) - 1.0

    return cagr


def compute_annualized_vol(
    returns: PandasSeries,
    freq: RebalanceFreq = "D",
    ddof: int = 1,
) -> float:
    """
    Compute annualized volatility from returns.
    从收益率计算年化波动率.

    公式 / Formula:
        年化波动率 = 日波动率 × √252 (假设 freq='D')
        Ann_Vol = Daily_Vol × √ann_factor

    Args / 参数:
        returns: 收益率序列
        freq: 数据频率 ('D' 日, 'W' 周, 'M' 月)
        ddof: 标准差自由度 (1 = 样本标准差)

    Returns / 返回:
        年化波动率小数形式
    """
    if returns.empty:
        logger.warning("Empty returns series")
        return 0.0

    # Remove NaN
    clean_returns = returns.dropna()
    if len(clean_returns) < 2:
        logger.warning("Insufficient returns for volatility calculation")
        return 0.0

    # 计算标准差 / Compute std
    vol = clean_returns.std(ddof=ddof)

    # 年化:乘以 √年化因子 / Annualize: multiply by √ann_factor
    ann_factor = annualization_factor(freq)
    ann_vol = vol * np.sqrt(ann_factor)

    return ann_vol


def compute_sharpe_ratio(
    returns: PandasSeries,
    freq: RebalanceFreq = "D",
    risk_free_rate: float = 0.0,
    ddof: int = 1,
) -> float:
    """
    Compute Sharpe ratio (annualized).
    计算年化夏普比率.

    公式 / Formula:
        Sharpe = (年化收益率 - 无风险利率) / 年化波动率
        Sharpe = (Ann_Return - Rf) / Ann_Vol

    夏普比率解读 / Interpreting Sharpe Ratio:
        < 0: 跑输无风险资产
        0-1: 一般
        1-2: 较好
        > 2: 优秀

    Args / 参数:
        returns: 收益率序列
        freq: 数据频率 ('D' 日, 'W' 周, 'M' 月)
        risk_free_rate: 年化无风险利率 (默认 0.0)
        ddof: 标准差自由度

    Returns / 返回:
        夏普比率
    """
    if returns.empty:
        logger.warning("Empty returns series")
        return 0.0

    clean_returns = returns.dropna()
    if len(clean_returns) < 2:
        logger.warning("Insufficient returns for Sharpe calculation")
        return 0.0

    # 年化因子 / Annualization factor
    ann_factor = annualization_factor(freq)

    # 平均收益率(每期)/ Mean return (per period)
    mean_return = clean_returns.mean()

    # 年化收益率 / Annualized return
    ann_return = mean_return * ann_factor

    # 超额收益 = 年化收益 - 无风险利率 / Excess return
    excess_return = ann_return - risk_free_rate

    # 年化波动率 / Annualized vol
    vol = clean_returns.std(ddof=ddof)
    ann_vol = vol * np.sqrt(ann_factor)

    # 夏普比率 = 超额收益 / 波动率 / Sharpe ratio
    sharpe = safe_divide(excess_return, ann_vol, default=0.0)

    return sharpe


def compute_performance_metrics(
    equity_curve: PandasSeries,
    returns: PandasSeries | None = None,
    freq: RebalanceFreq = "D",
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """
    Compute comprehensive performance metrics.
    计算综合业绩指标.

    返回指标 / Returned Metrics:
        - total_return: 总收益率
        - cagr: 年化复合增长率
        - annualized_vol: 年化波动率
        - sharpe_ratio: 夏普比率
        - win_rate: 胜率(正收益天数占比)
        - best_day: 单日最佳收益
        - worst_day: 单日最差收益
        - n_periods: 总期数

    Args / 参数:
        equity_curve: 净值序列 (date -> equity value)
        returns: 收益率序列 (如果为 None,从 equity_curve 计算)
        freq: 数据频率 ('D' 日, 'W' 周, 'M' 月)
        risk_free_rate: 年化无风险利率

    Returns / 返回:
        业绩指标字典
    """
    if equity_curve.empty:
        logger.warning("Empty equity curve")
        return {}

    # Compute returns if not provided
    if returns is None:
        returns = equity_curve.pct_change().fillna(0.0)

    # 总收益率 / Total return
    total_return = (
        (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0 if len(equity_curve) > 0 else 0.0
    )

    # 年化复合增长率 / CAGR
    cagr = compute_cagr(equity_curve, freq)

    # 年化波动率 / Volatility
    ann_vol = compute_annualized_vol(returns, freq)

    # 夏普比率 / Sharpe ratio
    sharpe = compute_sharpe_ratio(returns, freq, risk_free_rate)

    # 胜率 = 正收益天数 / 总天数 / Win rate
    clean_returns = returns.dropna()
    win_rate = (clean_returns > 0).sum() / len(clean_returns) if len(clean_returns) > 0 else 0.0

    # 单日最佳/最差收益 / Best/worst day
    best_day = clean_returns.max() if not clean_returns.empty else 0.0
    worst_day = clean_returns.min() if not clean_returns.empty else 0.0

    # 总期数 / Number of periods
    n_periods = len(equity_curve)

    metrics = {
        "total_return": total_return,
        "cagr": cagr,
        "annualized_vol": ann_vol,
        "sharpe_ratio": sharpe,
        "win_rate": win_rate,
        "best_day": best_day,
        "worst_day": worst_day,
        "n_periods": n_periods,
    }

    logger.debug(f"Computed performance metrics: Sharpe={sharpe:.2f}, CAGR={cagr:.2%}")

    return metrics


def compute_sortino_ratio(
    returns: PandasSeries,
    freq: RebalanceFreq = "D",
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
) -> float:
    """
    Compute Sortino ratio (downside risk-adjusted return).
    计算索提诺比率(下行风险调整收益).

    与夏普比率的区别 / Difference from Sharpe:
        - 夏普比率使用总波动率(包括上涨)
        - 索提诺比率只使用下行波动率(只惩罚损失)

    公式 / Formula:
        Sortino = (年化收益 - 无风险利率) / 下行波动率
        Sortino = (Ann_Return - Rf) / Downside_Vol

    Args / 参数:
        returns: 收益率序列
        freq: 数据频率
        risk_free_rate: 年化无风险利率
        target_return: 目标收益率(用于计算下行偏差,默认 0.0)

    Returns / 返回:
        索提诺比率

    Note / 注意:
        Uses annualized mean excess return over annualized downside deviation.
    """
    if returns.empty:
        logger.warning("Empty returns series")
        return 0.0

    clean_returns = returns.dropna()
    if len(clean_returns) < 2:
        return 0.0

    ann_factor = annualization_factor(freq)

    # 年化收益率 / Annualized return
    mean_return = clean_returns.mean()
    ann_return = mean_return * ann_factor
    excess_return = ann_return - risk_free_rate

    # 下行偏差(只考虑负收益)/ Downside deviation (only negative returns)
    downside_returns = clean_returns[clean_returns < target_return]
    if downside_returns.empty:
        # 没有负收益时,Sortino 为无穷大(如果有正超额收益)
        return np.inf if excess_return > 0 else 0.0

    # 下行波动率 / Downside volatility
    downside_std = downside_returns.std()
    downside_vol = downside_std * np.sqrt(ann_factor)

    # Sortino = 超额收益 / 下行波动率
    sortino = safe_divide(excess_return, downside_vol, default=0.0)

    return sortino


__all__ = [
    "compute_cagr",
    "compute_annualized_vol",
    "compute_sharpe_ratio",
    "compute_performance_metrics",
    "compute_sortino_ratio",
]
