"""
Factor diagnostics: IC and related statistics.
因子诊断: 信息系数(IC)与统计汇总.

Key functions / 核心函数:
- compute_ic_series: cross-sectional IC series
- compute_ic_stats: summary stats for IC series
"""

from __future__ import annotations

from dataclasses import dataclass
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
        if f[mask].nunique() < 2 or r[mask].nunique() < 2:
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


def compute_factor_coverage(factor: PandasDataFrame) -> PandasDataFrame:
    """Per-date valid coverage and cross-sectional dispersion."""
    finite = factor.replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(
        {
            "coverage": finite.notna().mean(axis=1),
            "n_valid": finite.notna().sum(axis=1).astype(float),
            "dispersion": finite.std(axis=1, ddof=1),
        },
        index=factor.index,
    )


def compute_quantile_returns(
    factor: PandasDataFrame,
    forward_returns: PandasDataFrame,
    *,
    quantiles: int = 5,
    min_obs: int | None = None,
) -> PandasDataFrame:
    """Mean forward return for equal-count cross-sectional factor buckets."""
    if quantiles < 2:
        raise ValueError("quantiles must be >= 2")
    minimum = min_obs or quantiles
    dates = factor.index.intersection(forward_returns.index)
    assets = factor.columns.intersection(forward_returns.columns)
    records: list[pd.Series] = []
    for date in dates:
        pair = (
            pd.DataFrame(
                {"factor": factor.loc[date, assets], "return": forward_returns.loc[date, assets]}
            )
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if len(pair) < minimum:
            continue
        ranks = pair["factor"].rank(method="first")
        bucket = pd.qcut(ranks, q=min(quantiles, len(pair)), labels=False) + 1
        row = pair.groupby(bucket)["return"].mean()
        row.name = date
        records.append(row)
    if not records:
        return pd.DataFrame(columns=range(1, quantiles + 1), dtype=float)
    result = pd.DataFrame(records).reindex(columns=range(1, quantiles + 1))
    result.columns.name = "quantile"
    return result


def compute_factor_turnover(factor: PandasDataFrame, top_quantile: float = 0.2) -> PandasSeries:
    """Fraction of the previous top bucket replaced on each date."""
    if not 0 < top_quantile <= 1:
        raise ValueError("top_quantile must be in (0, 1]")
    memberships: list[set[str]] = []
    dates: list[pd.Timestamp] = []
    for date, row in factor.iterrows():
        valid = row.replace([np.inf, -np.inf], np.nan).dropna()
        count = max(1, int(np.ceil(len(valid) * top_quantile))) if len(valid) else 0
        memberships.append(set(valid.nlargest(count).index) if count else set())
        dates.append(pd.Timestamp(date))
    values = [0.0]
    for previous, current in zip(memberships, memberships[1:], strict=False):
        values.append(1.0 if not previous else 1.0 - len(previous & current) / len(previous))
    return pd.Series(values, index=pd.DatetimeIndex(dates), name="factor_turnover")


def compute_rank_stability(factor: PandasDataFrame) -> PandasSeries:
    """Spearman correlation between consecutive cross-sectional ranks."""
    ranks = factor.rank(axis=1, pct=True)
    values = [np.nan]
    for position in range(1, len(ranks)):
        values.append(ranks.iloc[position - 1].corr(ranks.iloc[position], method="spearman"))
    return pd.Series(values, index=factor.index, name="rank_stability")


@dataclass
class FactorTearSheet:
    ic_series: PandasSeries
    ic_stats: dict[str, float]
    quantile_returns: PandasDataFrame
    cumulative_spread: PandasSeries
    coverage: PandasDataFrame
    turnover: PandasSeries
    rank_stability: PandasSeries


def generate_factor_tear_sheet(
    factor: PandasDataFrame,
    forward_returns: PandasDataFrame,
    *,
    quantiles: int = 5,
    method: Literal["spearman", "pearson"] = "spearman",
    min_obs: int = 5,
) -> FactorTearSheet:
    ic = compute_ic_series(factor, forward_returns, method=method, min_obs=min_obs)
    quantile = compute_quantile_returns(
        factor, forward_returns, quantiles=quantiles, min_obs=min_obs
    )
    spread = (
        quantile[quantiles] - quantile[1]
        if {1, quantiles} <= set(quantile.columns)
        else pd.Series(dtype=float)
    )
    cumulative = (1.0 + spread.fillna(0.0)).cumprod() - 1.0
    cumulative.name = "cumulative_spread"
    return FactorTearSheet(
        ic_series=ic,
        ic_stats=compute_ic_stats(ic),
        quantile_returns=quantile,
        cumulative_spread=cumulative,
        coverage=compute_factor_coverage(factor),
        turnover=compute_factor_turnover(factor, top_quantile=1 / quantiles),
        rank_stability=compute_rank_stability(factor),
    )


__all__ = [
    "FactorTearSheet",
    "compute_factor_coverage",
    "compute_factor_turnover",
    "compute_forward_returns",
    "compute_ic_series",
    "compute_ic_stats",
    "compute_quantile_returns",
    "compute_rank_stability",
    "generate_factor_tear_sheet",
]
