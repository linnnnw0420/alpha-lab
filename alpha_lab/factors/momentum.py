"""
Momentum factor calculations.
动量因子计算模块.

Key functions / 核心函数:
- momentum: 简单回溯动量 / simple lookback momentum
- momentum_multi_period: 多周期复合动量 / composite multi-period momentum

动量因子原理 / Momentum Factor Theory:
    动量因子基于"过去表现好的股票未来可能继续表现好"的假设.
    The momentum factor is based on the hypothesis that stocks that have 
    performed well in the past may continue to perform well in the future.
    
    常用的动量因子:
    - 20日动量 (≈1个月)
    - 60日动量 (≈3个月)  
    - 120日动量 (≈6个月)
    - 252日动量 (≈1年)
"""

from __future__ import annotations

import logging
from typing import Literal
import pandas as pd
import numpy as np

from alpha_lab.utils.logging import get_logger
from alpha_lab.utils.typing import PandasDataFrame

logger = get_logger(__name__)

# Cache for computed momentum to avoid recomputation / 动量计算缓存,避免重复计算
_MOMENTUM_CACHE: dict[str, PandasDataFrame] = {}

def momentum(
    prices: PandasDataFrame,
    lookback: int,
    lag: int = 1,
    method: Literal["simple", "log"] = "simple",
    min_periods: int | None = None,
    use_cache: bool = True,
) -> PandasDataFrame:
    """
    Compute momentum factor: return from t-lookback-lag to t-lag.
    计算动量因子:从 t-lookback-lag 到 t-lag 的收益率.

    公式 / Formula:
        simple: price[t-lag] / price[t-lookback-lag] - 1
        log: ln(price[t-lag] / price[t-lookback-lag])

    时间线图示 / Timeline Illustration:
        |------lookback------|--lag--|
        t-lookback-lag       t-lag    t (今天)
        ^起点价格             ^终点价格 ^当前日期(不使用)

    为什么需要 lag? / Why lag is needed:
        lag=1 表示使用昨天的价格计算,避免使用今天的价格(可能还未确定).
        这是避免前瞻偏差 (lookahead bias) 的关键设置.

    Args / 参数:
        prices: 价格矩阵 (date x asset),通常是收盘价
        lookback: 回溯期数(如 20 表示 20 天)
        lag: 延迟期数 (default 1 = 使用昨天的价格)
        method: 'simple' (简单收益率) 或 'log' (对数收益率)
        min_periods: 最小有效数据点数
        use_cache: 是否使用缓存结果

    Returns / 返回:
        DataFrame: 与 prices 形状相同,动量因子值

    Example / 示例:
        # 20日动量,使用昨天收盘价
        # 20-day momentum using yesterday's close
        mom_20d = momentum(prices, lookback=20, lag=1)
    """
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")
    if lag < 0:
        raise ValueError(f"lag must be >= 0, got {lag}")
    if method not in {"simple", "log"}:
        raise ValueError(f"method must be 'simple' or 'log', got {method!r}")
    if min_periods is None:
        min_periods = lookback
    
    # Generate cache key using content hash instead of object id
    # This ensures cache validity even when DataFrame content changes
    try:
        data_hash = hash((
            tuple(prices.index),
            tuple(prices.columns),
            prices.values.tobytes() if hasattr(prices.values, 'tobytes') else str(prices.values)
        ))
    except (TypeError, ValueError):
        # Fallback: disable caching for unhashable data
        data_hash = None
        use_cache = False
    
    cache_key = f"{data_hash}_{lookback}_{lag}_{method}_{min_periods}"
    if use_cache and cache_key in _MOMENTUM_CACHE:
        logger.debug(f"Using cached momentum: lookback={lookback}, lag={lag}")
        return _MOMENTUM_CACHE[cache_key].copy()
    
    logger.debug(
        f"Computing momentum: lookback={lookback}, lag={lag}, "
        f"method={method}, min_periods={min_periods}"
    )

    # 将价格向后移动 lag 期,避免前瞻偏差
    # Shift prices by lag to avoid lookahead bias
    prices_lagged = prices.shift(lag)

    # 计算回溯期内的收益率 / Compute returns over lookback period
    total_lag = lookback + lag
    prices_start = prices.shift(total_lag)  # 回溯起点的价格

    if method == "simple":
        # 简单收益率: (终点价格 / 起点价格) - 1
        factor = (prices_lagged / prices_start) - 1.0
    else:  # log
        # 对数收益率: ln(终点价格 / 起点价格)
        factor = np.log(prices_lagged / prices_start)
    
    # 应用最小有效期数掩码 / Apply min_periods mask
    if min_periods > 0:
        # 统计窗口内有效数据点数 / Count valid periods in window
        valid_count = prices.notna().rolling(window=total_lag, min_periods=1).sum()
        factor = factor.where(valid_count >= min_periods)
    
    # 缓存结果 / Cache result
    if use_cache:
        _MOMENTUM_CACHE[cache_key] = factor.copy()
        # 限制缓存大小,避免内存溢出 / Limit cache size to avoid memory overflow
        if len(_MOMENTUM_CACHE) > 100:
            oldest_key = next(iter(_MOMENTUM_CACHE))
            del _MOMENTUM_CACHE[oldest_key]
            logger.debug("Cleared oldest momentum cache entry")

    return factor

def momentum_multi_period(
    prices: PandasDataFrame,
    lookbacks: list[int] | tuple[int, ...],
    weights: list[float] | tuple[float, ...] | None = None,
    lag: int = 1,
    method: Literal["simple", "log"] = "simple",
    normalize_each: bool = True,
) -> PandasDataFrame:
    """
    Compute composite momentum from multiple lookback periods.
    计算多周期复合动量因子.
    
    适用场景 / Use Cases:
    - 结合短/中/长期动量信号 / Combine short/medium/long-term momentum signals
    - 平滑单一周期动量的噪声 / Smooth out noise from single-period momentum

    计算流程 / Calculation Flow:
    1. 分别计算每个周期的动量 / Compute momentum for each lookback period
    2. (可选) 对每个动量做截面 z-score 标准化 / (Optional) Z-score normalize each
    3. 按权重加权求和 / Weighted sum by given weights

    Args / 参数:
        prices: 价格矩阵 (date x asset)
        lookbacks: 回溯期列表 (如 [20, 60, 120] 表示 1月/3月/6月)
        weights: 每个回溯期的权重 (默认等权重)
        lag: 延迟期数,避免前瞻偏差
        method: 'simple' 或 'log'
        normalize_each: 是否在合成前对每个周期做 z-score 标准化

    Returns / 返回:
        DataFrame: 加权复合动量因子
    
    Example / 示例:
        # 结合 1个月,3个月,6个月动量,等权重
        # Combine 1M, 3M, 6M momentum with equal weight
        mom_combo = momentum_multi_period(prices, lookbacks=[20, 60, 120])
    """
    if not lookbacks:
        raise ValueError("lookbacks must not be empty")
    
    n_periods = len(lookbacks)

    if weights is None:
        weights = [1.0 / n_periods] * n_periods
    else:
        if len(weights) != n_periods:
            raise ValueError(f"weights length {len(weights)} != lookbacks length {n_periods}")
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        if abs(total_weight) < 1e-12:
            raise ValueError("weights sum to zero")
        weights = [w / total_weight for w in weights]
    
    logger.info(
        f"Computing multi-period momentum: lookbacks={lookbacks}, "
        f"weights={[f'{w:.3f}' for w in weights]}"
    )

    # 计算每个周期的动量分量 / Compute each momentum component
    components = []
    for lb, wt in zip(lookbacks, weights):
        mom = momentum(prices, lookback=lb, lag=lag, method=method)

        if normalize_each:
            # 截面 z-score 标准化(每天内部标准化)
            # Cross-sectional z-score at each date
            from alpha_lab.factors.transform import zscore
            mom = zscore(mom, axis=1)
        
        components.append(mom * wt)  # 乘以权重
    
    # 加权求和 / Sum weighted components
    composite = pd.concat(components, axis=1).groupby(level=0, axis=1).sum()

    return composite

def clear_momentum_cache() -> None:
    """
    Clear momentum cache. Useful for memory management or testing.
    清除动量因子缓存.用于内存管理或测试场景.
    """
    global _MOMENTUM_CACHE
    _MOMENTUM_CACHE.clear()
    logger.debug("Cleared momentum cache")

__all__ = [
    "momentum",
    "momentum_multi_period",
    "clear_momentum_cache",
]