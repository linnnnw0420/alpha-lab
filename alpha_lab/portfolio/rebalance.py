"""
Rebalance schedule generation.
调仓日期生成模块.

Key function / 核心函数:
- generate_rebalance_schedule: 从交易日历生成调仓日期 / produce rebalance dates from trading calendar

调仓频率说明 / Rebalance Frequency:
    'D' = 每日 (Daily) - 每个交易日调仓
    'W' = 每周 (Weekly) - 每周最后一个交易日调仓
    'M' = 每月 (Monthly) - 每月最后一个交易日调仓

为什么使用月末而非月初 / Why End of Month:
    - 月末数据更完整
    - 避免假期带来的调仓延迟
    - 行业惯例
"""

from __future__ import annotations

import pandas as pd

from alpha_lab.utils.dates import generate_rebalance_dates, parse_date
from alpha_lab.utils.logging import get_logger
from alpha_lab.utils.typing import DateLike, PandasDatetimeIndex, RebalanceFreq

logger = get_logger(__name__)

# Cache for rebalance schedules / 调仓日期缓存
_REBALANCE_CACHE: dict[str, PandasDatetimeIndex] = {}


def generate_rebalance_schedule(
    trading_calendar: PandasDatetimeIndex,
    start_date: DateLike,
    end_date: DateLike,
    freq: RebalanceFreq,
    use_cache: bool = True,
) -> PandasDatetimeIndex:
    """
    Generate rebalance dates within date range.
    在指定日期范围内生成调仓日期.

    调仓日期的确定规则 / How Rebalance Dates are Determined:
    - 'D': 每个交易日
    - 'W': 每周的最后一个交易日(通常是周五,除非周五是假日)
    - 'M': 每月的最后一个交易日(可能是 28-31 号中的某一天)

    Args / 参数:
        trading_calendar: 完整的交易日历(已排序的 DatetimeIndex)
        start_date: 回测开始日期
        end_date: 回测结束日期
        freq: 调仓频率 'D' (每日), 'W' (每周), 'M' (每月)
        use_cache: 是否使用缓存的调仓日期表

    Returns / 返回:
        DatetimeIndex: 调仓日期列表

    Example / 示例:
        >>> # 生成月度调仓日期
        >>> schedule = generate_rebalance_schedule(
        ...     trading_calendar=cal,
        ...     start_date="2020-01-01",
        ...     end_date="2024-12-31",
        ...     freq="M"
        ... )
    """
    start_ts = parse_date(start_date)
    end_ts = parse_date(end_date)

    if start_ts > end_ts:
        raise ValueError("start_date must be <= end_date")

    # Generate cache key
    cal_key = f"{trading_calendar.min()}_{trading_calendar.max()}_{len(trading_calendar)}"
    cache_key = f"{cal_key}_{start_ts}_{end_ts}_{freq}"

    if use_cache and cache_key in _REBALANCE_CACHE:
        logger.debug(f"Using cached rebalance schedule: freq={freq}")
        return _REBALANCE_CACHE[cache_key]

    logger.debug(
        f"Generating rebalance schedule: freq={freq}, {start_ts.date()} to {end_ts.date()}"
    )

    # 过滤日历到指定日期范围 / Filter calendar to date range
    cal_subset = trading_calendar[(trading_calendar >= start_ts) & (trading_calendar <= end_ts)]

    if cal_subset.empty:
        logger.warning(f"No trading days in range {start_ts.date()} to {end_ts.date()}")
        return pd.DatetimeIndex([], name="date")

    # 按指定频率生成调仓日期 / Generate rebalance dates at specified frequency
    rebalance_dates = generate_rebalance_dates(cal_subset, freq)

    logger.info(f"Generated {len(rebalance_dates)} rebalance dates (freq={freq})")

    # 缓存结果 / Cache result
    if use_cache:
        _REBALANCE_CACHE[cache_key] = rebalance_dates
        # 限制缓存大小 / Limit cache size
        if len(_REBALANCE_CACHE) > 50:
            oldest = next(iter(_REBALANCE_CACHE))
            del _REBALANCE_CACHE[oldest]
            logger.debug("Cleared oldest rebalance cache entry")

    return rebalance_dates


def clear_rebalance_cache() -> None:
    """
    Clear rebalance schedule cache.
    清除调仓日期缓存.
    """
    global _REBALANCE_CACHE
    _REBALANCE_CACHE.clear()
    logger.debug("Cleared rebalance cache")


__all__ = [
    "generate_rebalance_schedule",
    "clear_rebalance_cache",
]
