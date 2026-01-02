"""
Factor transformation utilities.
因子变换工具模块.

Key functions / 核心函数:
- winsorize: 截断极端值 / clip extreme values
- zscore: 标准化到均值=0, 标准差=1 / standardize to mean=0, std=1
- rank_normalize: 转换为排名 (0-1) / convert to ranks (0-1)

因子变换的目的 / Purpose of Factor Transformation:
    1. 消除异常值影响 (winsorize)
    2. 使不同因子可比较 (zscore)
    3. 消除线性关系假设 (rank_normalize)

典型使用顺序 / Typical Usage Order:
    raw_factor  ->  winsorize  ->  zscore (or rank_normalize)  ->  portfolio weights
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd
import numpy as np

from alpha_lab.utils.logging import get_logger
from alpha_lab.utils.typing import PandasDataFrame, PandasSeries

logger = get_logger(__name__)

def winsorize(
    data: PandasSeries | PandasDataFrame,
    lower: float = 0.01,
    upper: float = 0.99,
    axis: int | None = None,
    inclusive: Literal["both", "neither", "left", "right"] = "both",
) -> PandasDataFrame | PandasSeries:
    """
    Clip extreme values to percentile thresholds.
    将极端值截断到指定百分位数阈值.

    为什么需要 Winsorize / Why Winsorize:
        - 因子中的极端值会导致权重过度集中
        - 异常值可能是数据错误
        - 提高因子的稳健性

    Args / 参数:
        data: DataFrame 或 Series
        lower: 下百分位数阈值 (0-1),如 0.01 表示 1%
        upper: 上百分位数阈值 (0-1),如 0.99 表示 99%
        axis: 计算方向
              0 = 列方向 (时间序列)
              1 = 行方向 (截面,每个日期内部)
              None = 整个数据集
        inclusive: 阈值是否包含边界
    
    Example / 示例:
        # 截面 winsorize(每个日期内部)
        # Cross-sectional winsorization (each date)
        factor_win = winsorize(factor, lower=0.01, upper=0.99, axis=1)
    """
    if not (0 <= lower <= 1) or not (0 <= upper <= 1):
        raise ValueError(f"percentiles must be in [0, 1], got lower={lower}, upper={upper}")
    if lower >= upper:
        raise ValueError(f"lower must be < upper, got {lower} >= {upper}")
    
    logger.debug(f"Winsorizing: lower={lower:.2%}, upper={upper:.2%}, axis={axis}")

    if isinstance(data, pd.Series):
        q_lower = data.quantile(lower)
        q_upper = data.quantile(upper)
        return data.clip(lower=q_lower, upper=q_upper)
    
    # DataFrame
    if axis is None:
        # Winsorize entire DataFrame as one distribution
        q_lower = data.quantile(lower).min()
        q_upper = data.quantile(upper).max()
        return data.clip(lower=q_lower, upper=q_upper)
    
    elif axis == 0:
        # Column-wise (time-series)
        return data.apply(lambda col: col.clip(
            lower=col.quantile(lower),
            upper=col.quantile(upper)
        ), axis=0)
    
    elif axis == 1:
        # Row-wise (cross-section at each date)
        return data.apply(lambda row: row.clip(
            lower=row.quantile(lower),
            upper=row.quantile(upper)
        ), axis=1)
    
    else:
        raise ValueError(f"axis must be 0, 1, or None, got {axis}")

def zscore(
    data: PandasDataFrame | PandasSeries,
    axis: int | None = None,
    ddof: int = 1,
    min_periods: int = 2,
) -> PandasDataFrame | PandasSeries:
    """
    Standardize to z-scores: (x - mean) / std.
    Z-score 标准化:(x - 均值) / 标准差.

    公式 / Formula:
        z = (x - μ) / σ

    Z-score 的意义 / Meaning of Z-score:
        z = 0: 等于均值
        z = 1: 高于均值一个标准差
        z = -2: 低于均值两个标准差

    Args / 参数:
        data: DataFrame 或 Series
        axis: 计算方向
              0 = 列方向 (时间序列标准化)
              1 = 行方向 (截面标准化,常用于因子)
              None = 整个数据集
        ddof: 标准差自由度 (1 = 样本标准差, 0 = 总体标准差)
        min_periods: 最小有效数据点数
    
    Returns / 返回:
        Z-scored data (与输入形状相同)
    
    Example / 示例:
        # 截面 z-score(每个日期内部标准化)
        # Cross-sectional z-score at each date
        factor_z = zscore(factor, axis=1)
    """
    logger.debug(f"Computing z-score: axis={axis}, ddof={ddof}")
    if isinstance(data, pd.Series):
        mean = data.mean()
        std = data.std(ddof=ddof)
        if std < 1e-12 or pd.isna(std):
            logger.warning("Standard deviation is zero or NaN, returning zeros")
            return pd.Series(0.0, index=data.index)
        return (data - mean) / std

    # DataFrame
    if axis is None:
        # 整体标准化 / Standardize entire DataFrame
        mean = data.mean().mean()
        std = data.std().mean()
        if std < 1e-12:
            logger.warning("Standard deviation is near zero")
            return pd.DataFrame(0.0, index=data.index, columns=data.columns)
        return (data - mean) / std
    
    elif axis == 0:
        # 列方向(时间序列)/ Column-wise (time series)
        std = data.std(axis=0, ddof=ddof)
        std_safe = std.replace(0, np.nan)  # 避免除以零 / Replace zero std with NaN
        result = (data - data.mean(axis=0)) / std_safe
        return result.fillna(0.0)  # 零标准差填充为0
    
    elif axis == 1:
        # 行方向(截面,每个日期)/ Row-wise (cross-section at each date)
        result = data.sub(data.mean(axis=1), axis=0).div(
            data.std(axis=1, ddof=ddof).replace(0, np.nan), axis=0
        )
        return result.fillna(0.0)  # 零标准差填充为0
    
    else:
        raise ValueError(f"axis must be 0, 1, or None, got {axis}")

def rank_normalize(
        data: PandasDataFrame | PandasSeries,
        axis: int | None = None,
        method: Literal["average", "min", "max", "first", "dense"] = "average",
        pct: bool = True,
) -> PandasDataFrame | PandasSeries:
    """
    Convert values to ranks, optionally scaled to [0, 1].
    将数值转换为排名,可选缩放到 [0, 1] 区间.

    排名标准化的优势 / Advantages of Rank Normalization:
        - 消除异常值影响
        - 不假设数据分布
        - 使因子值均匀分布

    Args / 参数:
        data: DataFrame 或 Series
        axis: 计算方向
              0 = 列方向 (时间序列排名)
              1 = 行方向 (截面排名,常用于因子)
              None = 整个数据集
        method: 处理平局的方法 (参见 pandas.rank)
                'average' = 平均排名
                'min' = 最小排名
                'max' = 最大排名
                'first' = 按出现顺序
                'dense' = 密集排名
        pct: 如果 True,缩放排名到 [0, 1];如果 False,保持整数排名

    Returns / 返回:
        Ranked data (与输入形状相同)

    Example / 示例:
        # 截面排名,缩放到 [0, 1]
        # Cross-sectional rank at each date, scaled to [0, 1]
        factor_rank = rank_normalize(factor, axis=1, pct=True)
    """
    logger.debug(f"Rank normalizing: axis={axis}, method={method}, pct={pct}")

    if isinstance(data, pd.Series):
        return data.rank(method=method, pct=pct)
    
    # DataFrame
    if axis is None:
        # Rank entire DataFrame
        stacked = data.stack()
        ranked = stacked.rank(method=method, pct=pct)
        return ranked.unstack()
    elif axis == 0:
        # Column-wise
        return data.rank(axis=0, method=method, pct=pct)
    elif axis == 1:
        # Row-wise (cross-section)
        return data.rank(axis=1, method=method, pct=pct)
    else:
        raise ValueError(f"axis must be 0, 1, or None, got {axis}")

def neutralize_industry(
    factor: PandasDataFrame,
    industry_map: dict[str, str] | PandasSeries,
) -> PandasDataFrame:
    """
    Industry-neutralize factor: subtract industry mean at each date.
    行业中性化:在每个日期减去行业均值.

    行业中性化的目的 / Purpose of Industry Neutralization:
        - 消除因子中的行业偏差
        - 使因子在行业内部有效,而非仅选择好行业

    Args / 参数:
        factor: 因子矩阵 (date x asset)
        industry_map: 资产 -> 行业映射 (dict 或 Series)

    Returns / 返回:
        行业中性化后的因子 (形状相同)

    Note / 注意:
        v0 版本 - 预留给未来实现.
        v0 implementation - marked for future enhancement.
    """
    logger.warning("neutralize_industry is v0 stub - will be implemented in v1")
    raise NotImplementedError("Industry neutralization not yet implemented")

__all__ = [
    "winsorize",
    "zscore",
    "rank_normalize",
    "neutralize_industry",
]