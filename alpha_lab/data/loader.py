
"""
Data loader: central interface for loading price/returns data.
数据加载器:加载价格/收益数据的核心接口.

Key functions / 核心函数:
- load_universe: 获取股票池列表 / get list of tickers
- load_prices: 加载价格面板 (date x asset) / load price panel
- load_returns: 从价格计算收益率 / compute returns from prices

工作流 / Workflow:
    1. load_universe() 获取股票列表
    2. load_prices() 从 CSV 加载价格矩阵
    3. load_returns() 计算日收益率(可选)
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

from alpha_lab.config import get_paths, UniverseConfig, get_universe as get_universe_tickers
from alpha_lab.config.paths import Paths
from alpha_lab.data.sources.csv_source import load_csv_prices
from alpha_lab.utils.dates import parse_date, align_to_trading_day
from alpha_lab.utils.logging import get_logger
from alpha_lab.utils.typing import (
    DateLike,
    PandasDataFrame,
    PandasDatetimeIndex,
    PriceField,
    Ticker,
)

logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# Internal cache for trading calendar / 交易日历内部缓存
# -----------------------------------------------------------------------------

_TRADING_CALENDAR_CACHE: dict[str, PandasDatetimeIndex] = {}


# -----------------------------------------------------------------------------
# Public API / 公共 API
# -----------------------------------------------------------------------------

def load_universe(
    universe: UniverseConfig | list[Ticker] | tuple[Ticker, ...],
    as_of: DateLike | None = None,
) -> list[Ticker]:
    """
    Load universe as list of tickers.
    加载股票池,返回股票代码列表.

    Args / 参数:
        universe: UniverseConfig 或股票代码列表/元组
        as_of: 参考日期(当 UniverseConfig.dynamic_by_price=True 时生效)
    
    Returns / 返回:
        股票代码字符串列表 / List of ticker strings
    """
    if isinstance(universe, UniverseConfig):
        tickers = get_universe_tickers(universe, as_of=as_of)
        if universe.dynamic_by_price and as_of is not None:
            tickers = _filter_universe_by_price(
                tickers=tickers,
                as_of=as_of,
                field=universe.price_field,
                min_valid_price=universe.min_valid_price,
            )
    elif isinstance(universe, (list, tuple)):
        tickers = [str(t).strip() for t in universe if str(t).strip()]
    else:
        raise TypeError(
            f"universe must be UniverseConfig or list/tuple. got {type(universe).__name__}"
        )

    if not tickers:
        raise ValueError("Universe cannot be empty")
    
    # Validate ticker format
    for ticker in tickers:
        if not ticker or not isinstance(ticker, str):
            raise ValueError(f"Invalid ticker: {ticker!r}")
        if len(ticker) > 20:
            raise ValueError(f"Ticker too long: {ticker!r}")
        
    logger.debug(f"Loaded universe: {len(tickers)} tickers")
    return tickers

def load_prices(
    universe: UniverseConfig | list[Ticker] | tuple[Ticker, ...],
    start_date: DateLike,
    end_date: DateLike,
    field: PriceField = "close",
    source: Literal["csv", "parquet"] = "csv",
    align_dates: bool = True,
    forward_fill_limit: int | None = 5,
    csv_file: str | None = None,
) -> PandasDataFrame:
    """
    Load price panel (date x asset).
    加载价格面板,行=日期,列=资产.

    数据处理流程 / Data Processing Flow:
    1. 验证输入参数 / Validate inputs
    2. 从数据源加载原始价格 / Load raw prices from source
    3. 对齐到交易日历并前向填充缺失值 / Align to calendar & forward-fill

    Args / 参数: 
        universe: 股票池配置或股票列表 / UniverseConfig or list of tickers
        start_date: 开始日期(包含)/ start date (inclusive)
        end_date: 结束日期(包含)/ end date (inclusive)
        field: 价格字段 'open'/'high'/'low'/'close'/'vwap'
        source: 数据源 'csv' 或 'parquet'
        align_dates: 是否对齐到交易日历并前向填充 / align to trading calendar & forward-fill
        forward_fill_limit: 最大前向填充天数 (None=无限制) / max days to ffill
        csv_file: 指定CSV文件名 (如 'nasdaq_50_stocks_2023.csv')，None则按默认规则搜索
    
    Returns / 返回:
        DataFrame: index=date, columns=tickers, values=prices
        价格矩阵,行索引=日期,列=股票代码,值=价格
    """
    # Validate inputs
    tickers = load_universe(universe)
    start_ts = parse_date(start_date)
    end_ts = parse_date(end_date)

    if start_ts > end_ts:
        raise ValueError(f"start_date must be <= end_date")
    
    valid_fields = {"open", "high", "low", "close", "vwap"}
    if field not in valid_fields:
        raise ValueError(f"field must be one of {valid_fields}, got {field!r}")
    
    logger.info(
        f"Loading {field} prices: {len(tickers)} tickers, "
        f"{start_ts.date()} to {end_ts.date()}"
    )

    # Load from source
    paths = get_paths()

    if source == "csv":
        prices_raw = load_csv_prices(
            tickers=tickers,
            start_date=start_ts,
            end_date=end_ts,
            field=field,
            paths=paths,
            csv_file=csv_file,
        )
    elif source == "parquet":
        raise NotImplementedError("Parquet source not yet implemented")
    else:
        raise ValueError(f"Unknown source: {source!r}")

    # Handle empty result
    if prices_raw.empty:
        logger.warning("No price data found")
        return pd.DataFrame(
            index=pd.DatetimeIndex([], name="date"),
            columns=tickers,
            dtype=float,
        )
    
    if not isinstance(prices_raw.index, pd.DatetimeIndex):
        raise TypeError("Price data must have DatetimeIndex")
    
    # Align to calendar if requested
    if align_dates:
        prices_aligned = _align_to_calendar(
            prices_raw, start_ts, end_ts, forward_fill_limit
        )
    else:
        prices_aligned = prices_raw
    
    # Ensure all tickers present (fill with NaN if missing) 
    missing_tickers = set(tickers) - set(prices_aligned.columns)
    if missing_tickers:
        logger.warning(f"Missing {len(missing_tickers)} tickers")
        for ticker in missing_tickers:
            prices_aligned[ticker] = float("nan")

    # Reorder to match requested universe
    prices_final = prices_aligned[tickers].copy()
    
    logger.info(f"Loaded: {len(prices_final)} days x {len(prices_final.columns)} assets")
    return prices_final

def load_returns(
    prices: PandasDataFrame | None = None,
    universe: UniverseConfig | list[Ticker] | tuple[Ticker, ...] | None = None,
    start_date: DateLike | None = None,
    end_date: DateLike | None = None,
    periods: int = 1,
    method: Literal["simple", "log"] = "simple",
    **load_kwargs,
) -> PandasDataFrame:
    """
    Compute returns from prices.
    从价格数据计算收益率.

    公式 / Formula:
    - simple (简单收益率): (P[t] - P[t-1]) / P[t-1] = P[t]/P[t-1] - 1
    - log (对数收益率): ln(P[t] / P[t-1])

    Args / 参数:
        prices: 预加载的价格数据 (如果为 None,则自动加载)
        universe: 股票池 (当 prices 为 None 时必须提供)
        start_date: 开始日期 (当 prices 为 None 时必须提供)
        end_date: 结束日期 (当 prices 为 None 时必须提供)
        periods: 计算周期 (1 = 日收益率, 5 = 周收益率, 20 ≈ 月收益率)
        method: 'simple' (简单收益率) 或 'log' (对数收益率)
        **load_kwargs: 传递给 load_prices() 的其他参数

    Returns / 返回:
        DataFrame: 与 prices 形状相同,前 'periods' 行为 NaN
    """
    # Load prices if not provided
    if prices is None:
        if universe is None or start_date is None or end_date is None:
            raise ValueError("must provide universe, start_date, end_date if prices is None")
        prices = load_prices(
            universe=universe,
            start_date=start_date,
            end_date=end_date,
            **load_kwargs,
        )

    if periods < 1:
        raise ValueError(f"periods must be >= 1, got {periods}")
    
    if method not in {"simple", "log"}:
        raise ValueError(f"method must be 'simple or 'log', got {method!r}")

    logger.debug(f"Computing {periods}-period {method} returns")

    # Compute returns
    if method == "simple":
        returns = prices.pct_change(periods=periods)
    else: # log
        import numpy as np
        returns = pd.DataFrame(
            np.log(prices / prices.shift(periods)),
            index=prices.index,
            columns=prices.columns,
        )

    return returns

# -----------------------------------------------------------------------------
# Internal helpers / 内部辅助函数
# -----------------------------------------------------------------------------

def _align_to_calendar(
    prices: PandasDataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    forward_fill_limit: int | None = 5,
) -> PandasDataFrame:
    """
    Align prices to full trading calendar and forward-fill missing dates.
    将价格对齐到完整的交易日历,并前向填充缺失日期.

    为什么需要对齐 / Why alignment is needed:
    - 原始数据可能有缺失的交易日
    - 不同资产可能有不同的停牌日
    - 前向填充确保每天都有价格(用于持仓估值)

    Args / 参数:
        prices: 原始价格数据
        start_date: 开始日期
        end_date: 结束日期
        forward_fill_limit: 最大前向填充天数(防止长期停牌股票被错误填充)

    Returns / 返回:
        对齐后的价格矩阵
    """
    # 从原始数据提取交易日历 / Extract trading calendar from raw data
    calendar = _get_trading_calendar(prices.index)
    calendar_subset = calendar[(calendar >= start_date) & (calendar <= end_date)]

    if calendar_subset.empty:
        logger.warning(f"No trading days in range")
        return pd.DataFrame(
            index=pd.DatetimeIndex([], name="date"),
            columns=prices.columns,
            dtype=float,
        )

    # 重建索引并前向填充 / Reindex and forward-fill
    prices_aligned = prices.reindex(calendar_subset)

    if forward_fill_limit is not None:
        prices_aligned = prices_aligned.ffill(limit=forward_fill_limit)
    else:
        prices_aligned = prices_aligned.ffill()
    
    return prices_aligned

def _get_trading_calendar(date_index: PandasDatetimeIndex) -> PandasDatetimeIndex:
    """
    Extract and cache trading calendar from date index.
    从日期索引提取交易日历并缓存.

    缓存策略 / Caching Strategy:
    - 使用日期范围和长度作为缓存键
    - 避免重复排序和去重操作
    """
    cache_key = f"{date_index.min()}_{date_index.max()}_{len(date_index)}"

    if cache_key not in _TRADING_CALENDAR_CACHE:
        calendar = pd.DatetimeIndex(sorted(date_index.unique()))
        _TRADING_CALENDAR_CACHE[cache_key] = calendar
        logger.debug(f"Cached calendar: {len(calendar)} days")
    
    return _TRADING_CALENDAR_CACHE[cache_key]

def _filter_universe_by_price(
    tickers: list[Ticker],
    as_of: DateLike,
    field: PriceField,
    min_valid_price: float = 0.0,
) -> list[Ticker]:
    """
    Filter tickers by price availability at a specific date.
    按指定日期的有效价格过滤股票池.
    """
    if not tickers:
        return []

    as_of_ts = parse_date(as_of)
    paths = get_paths()

    prices = load_csv_prices(
        tickers=tickers,
        start_date=as_of_ts,
        end_date=as_of_ts,
        field=field,
        paths=paths,
    )
    if prices.empty:
        logger.warning(f"No prices found for as_of={as_of_ts.date()}, returning original universe")
        return tickers

    row = prices.iloc[-1]
    valid = row.notna() & (row > min_valid_price)
    filtered = [t for t in tickers if bool(valid.get(t, False))]
    if not filtered:
        logger.warning(f"No tradable tickers at {as_of_ts.date()}, returning original universe")
        return tickers

    logger.debug(f"Filtered universe by price: {len(tickers)} -> {len(filtered)}")
    return filtered

def infer_tradable_mask(
    prices: PandasDataFrame,
    min_valid_price: float = 0.0,
) -> PandasDataFrame:
    """
    Infer tradable mask from price availability.
    用价格可用性推断可交易掩码.
    """
    if prices.empty:
        return pd.DataFrame(index=prices.index, columns=prices.columns, dtype=bool)
    return (prices.notna()) & (prices > min_valid_price)

__all__ = [
    "load_universe",
    "load_prices",
    "load_returns",
    "infer_tradable_mask",
]
