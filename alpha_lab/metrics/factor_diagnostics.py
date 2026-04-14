"""
Factor diagnostics: IC and related statistics.
因子诊断: 信息系数(IC)与统计汇总.

Key functions / 核心函数:
- compute_ic_series: cross-sectional IC series
- compute_ic_stats: summary stats for IC series
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from alpha_lab.utils.logging import get_logger
from alpha_lab.utils.math import safe_divide
from alpha_lab.utils.typing import PandasDataFrame, PandasSeries

logger = get_logger(__name__)

def compute_ic_series(
    factor: PandasDataFrame,
    forward_returns: PandasDataFrame,
    method: Literal["spearman", "pearson"] = "spearman",
    min_obs: int = 5,
) -> PandasSeries:
    """
    Compute IC series between factor and forward returns (cross-sectional).
    计算因子与未来收益的截面 IC 序列.

    Args / 参数:
        factor: 因子矩阵 (date x asset)
        forward_returns: 未来收益矩阵 (date x asset)
        method: "spearman" or "pearson"
        min_obs: 每个日期最少有效样本数

    Returns / 返回:
        Series: index=date, value=IC
    """
    if factor.empty or forward_returns.empty:
        logger.warning("Empty factor/forward_returns for IC calculation")
        return pd.Series(dtype=float)

    common_dates = factor.index.intersection(forward_returns.index)
    common_assets = factor.columns.intersection(forward_returns.columns)
    if common_dates.empty or common_assets.empty:
        logger.warning("No overlapping dates/assets for IC calculation")
        return pd.Series(dtype=float)

    ic_values = []
    ic_dates = []

    for date in common_dates:
        f = factor.loc[date, common_assets]
        r = forward_returns.loc[date, common_assets]
        mask = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
        if int(mask.sum()) < min_obs:
            continue
        ic = f[mask].corr(r[mask], method=method)
        if pd.isna(ic):
            continue
        ic_values.append(float(ic))
        ic_dates.append(date)

    if not ic_values:
        return pd.Series(dtype=float)

    return pd.Series(ic_values, index=pd.DatetimeIndex(ic_dates), name="ic")

def compute_forward_returns(
    prices: PandasDataFrame,
    horizon: int,
    delay: int = 0,
) -> PandasDataFrame:
    """
    Compute forward returns with optional execution delay.
    计算带延迟的未来收益(与执行日对齐).

    Returns / 返回:
        fwd_ret[t] = prices[t+delay+horizon] / prices[t+delay] - 1
    """
    if prices.empty:
        return pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if delay < 0:
        raise ValueError(f"delay must be >= 0, got {delay}")

    start = prices.shift(-delay)
    end = prices.shift(-(delay + horizon))
    fwd = (end / start) - 1.0
    return fwd

def compute_ic_stats(ic_series: PandasSeries) -> dict[str, float]:
    """
    Compute summary stats for IC series.
    计算 IC 序列统计汇总.
    """
    if ic_series is None or ic_series.empty:
        return {}

    ic_clean = ic_series.dropna()
    if ic_clean.empty:
        return {}

    mean_ic = float(ic_clean.mean())
    std_ic = float(ic_clean.std(ddof=1))
    ic_ir = safe_divide(mean_ic, std_ic, default=0.0)
    hit_rate = float((ic_clean > 0).sum() / len(ic_clean))

    return {
        "ic_mean": mean_ic,
        "ic_std": std_ic,
        "ic_ir": ic_ir,
        "ic_hit_rate": hit_rate,
        "ic_n": float(len(ic_clean)),
    }

__all__ = ["compute_ic_series", "compute_forward_returns", "compute_ic_stats"]
