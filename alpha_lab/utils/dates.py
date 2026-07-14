"""
Date utilities for alpha_lab.
日期工具模块.

Key functions / 核心函数:
- annualization_factor: 获取年化因子 / get annualization multiplier
- parse_date: 解析日期字符串 / parse date string to Timestamp
- align_to_trading_day: 对齐到交易日 / align date to trading calendar
- generate_rebalance_dates: 生成调仓日期 / generate rebalance dates from calendar
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from alpha_lab.utils.typing import DateLike, PandasDatetimeIndex, PandasTimestamp, RebalanceFreq

# Late import pandas (optional dependency check)
try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]

# -----------------------------------------------------------------------------
# Frequency <-> annualization factor mapping
# 频率 <-> 年化因子映射
# -----------------------------------------------------------------------------

_ANNUALIZATION_FACTORS: dict[str, float] = {
    "D": 252.0,  # 每年交易日数 / trading days per year
    "W": 52.0,  # 每年周数 / weeks per year
    "M": 12.0,  # 每年月数 / months per year
}


def annualization_factor(freq: RebalanceFreq) -> float:
    """
    Return the annualization multiplier for a given frequency.
    返回指定频率对应的年化乘数.

    用途 / Usage:
        - 日波动率  ->  年化波动率: 日波动率 × √252
        - 月收益率  ->  年化收益率: 月收益率 × 12

    Args / 参数:
        freq: 'D' (每日), 'W' (每周), 'M' (每月)

    Returns / 返回:
        每年的周期数 (日=252, 周=52, 月=12)

    Raises:
        ValueError: 如果 freq 不被识别
    """
    if freq not in _ANNUALIZATION_FACTORS:
        raise ValueError(
            f"Unknown freq: {freq!r}, must be one of the {list(_ANNUALIZATION_FACTORS.keys())}"
        )
    return _ANNUALIZATION_FACTORS[freq]


# -----------------------------------------------------------------------------
# Date parsing & alignment / 日期解析与对齐
# -----------------------------------------------------------------------------


def parse_date(value: DateLike) -> PandasTimestamp:
    """
    Convert DateLike (str/date/datetime) into pd.Timestamp.
    将 DateLike(字符串/日期/日期时间)转换为 pd.Timestamp.

    支持的输入格式 / Supported Input Formats:
        - "2024-01-15" (ISO 字符串)
        - datetime.date(2024, 1, 15)
        - datetime.datetime(2024, 1, 15)
        - pd.Timestamp("2024-01-15")

    Args / 参数:
        value: ISO 日期字符串,datetime.date 或 datetime.datetime

    Returns / 返回:
        pd.Timestamp (无时区)

    Raises:
        ImportError: 如果 pandas 未安装
        ValueError: 如果字符串无法解析
    """
    if pd is None:
        raise ImportError("pandas is required for parse_date()")

    if isinstance(value, pd.Timestamp):
        return value.normalize()
    if isinstance(value, datetime):
        return pd.Timestamp(value).normalize()
    if isinstance(value, date):
        return pd.Timestamp(value).normalize()
    if isinstance(value, str):
        return pd.Timestamp(value.strip()).normalize()
    raise TypeError(f"Cannot parse {type(value).__name__} as date")


def align_to_trading_day(
    dt: DateLike,
    trading_calendar: PandasDatetimeIndex,
    method: Literal["previous", "next"] = "previous",
) -> PandasTimestamp:
    """
    Align a date to the nearest trading day in the calendar.
    将日期对齐到日历中最近的交易日.

    使用场景 / Use Cases:
        - 输入日期是周末或假日时,找到前一个/后一个交易日
        - 确保回测日期都落在交易日上

    Args / 参数:
        dt: 目标日期
        trading_calendar: 已排序的交易日历 (DatetimeIndex)
        method: 'previous' (前一个交易日) 或 'next' (后一个交易日)

    Returns / 返回:
        对齐后的交易日 (pd.Timestamp)

    Raises:
        ValueError: 如果 method 无效或日期超出日历范围
    """
    if pd is None:
        raise ImportError("pandas is required for align_to_trading_day()")

    ts = parse_date(dt)

    if ts in trading_calendar:
        return ts

    if method == "previous":
        # 找到 <= ts 的最后一个交易日 / Find last trading day <= ts
        candidates = trading_calendar[trading_calendar <= ts]
        if candidates.empty:
            raise ValueError(f"No trading day on or before {ts}")
        return candidates[-1]
    elif method == "next":
        # 找到 >= ts 的第一个交易日 / Find first trading day >= ts
        candidates = trading_calendar[trading_calendar >= ts]
        if candidates.empty:
            raise ValueError(f"No trading day on or after {ts}")
        return candidates[0]
    else:
        raise ValueError(f"method must be 'previous' or 'next', got: {method!r}")


# -----------------------------------------------------------------------------
# Rebalance date generation / 调仓日期生成
# -----------------------------------------------------------------------------


def generate_rebalance_dates(
    trading_calendar: PandasDatetimeIndex,
    freq: RebalanceFreq,
) -> PandasDatetimeIndex:
    """
    Generate rebalance dates from a full trading calendar.
    从完整的交易日历生成调仓日期.

    生成规则 / Generation Rules:
        - 'D' (Daily): 返回所有交易日
        - 'W' (Weekly): 每周的最后一个交易日
        - 'M' (Monthly): 每月的最后一个交易日

    为什么用"最后一个"交易日 / Why "last" trading day:
        - 月末通常数据更完整
        - 符合行业惯例

    Args / 参数:
        trading_calendar: 完整的已排序交易日历 (DatetimeIndex)
        freq: 'D' (每日), 'W' (每周), 或 'M' (每月)

    Returns / 返回:
        交易日历的子集(指定频率的调仓日期)
    """
    if pd is None:
        raise ImportError("pandas is required for generate_rebalance_dates")

    if freq == "D":
        # 每日:返回所有交易日 / Daily: return all trading days
        return trading_calendar

    # 转换为 Series 以便使用 groupby / Convert to Series to use groupby
    cal_series = pd.Series(trading_calendar, index=trading_calendar)

    if freq == "W":
        # 按年-周分组,取每周最后一个交易日
        # Group by year-week, take last trading day of each week
        grouped = cal_series.groupby(
            [cal_series.index.isocalendar().year, cal_series.index.isocalendar().week]
        ).last()
    elif freq == "M":
        # 按年-月分组,取每月最后一个交易日
        # Group by year-month, take last trading day of each month
        grouped = cal_series.groupby([cal_series.index.year, cal_series.index.month]).last()
    else:
        raise ValueError(f"Unknown freq: {freq!r}")

    # 返回实际的交易日期(而非周期结束日期)
    # Return the actual trading dates (not the period end dates)
    return pd.DatetimeIndex(grouped.values, name="date")


__all__ = [
    "annualization_factor",
    "parse_date",
    "align_to_trading_day",
    "generate_rebalance_dates",
]
