"""
Factor to weight conversion.
因子到权重转换模块.

Key functions / 核心函数:
- top_k_long_only: 做多前 K% 资产(等权重)/ equal-weight top K% assets
- top_k_long_short: 做多前 K%,做空后 K% / long top K%, short bottom K%
- proportional_weights: 按因子分数比例分配权重 / weights proportional to factor scores

典型工作流 / Typical Workflow:
    factor (因子值)  ->  top_k_long_only/top_k_long_short  ->  weights (权重)
                                                         ↓
                                                   run_backtest()
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from alpha_lab.utils.logging import get_logger
from alpha_lab.utils.typing import PandasDataFrame, PandasSeries

logger = get_logger(__name__)

def top_k_long_only(
    factor: PandasDataFrame,
    k_pct: float = 0.2,
    equal_weight: bool = True,
) -> PandasDataFrame:
    """
    Long-only strategy: equal-weight top K% assets by factor score.
    纯多头策略:等权重做多因子得分最高的前 K% 资产.

    策略逻辑 / Strategy Logic:
    1. 每个日期,按因子值排序
    2. 选择因子值最高的前 K% 资产
    3. 等权重分配(或按排名加权)

    Args / 参数:
        factor: 因子矩阵 (date x asset),因子分数
        k_pct: 持仓资产比例 (0-1),如 0.2 = 前 20%
        equal_weight: 如果 True,等权重;如果 False,按排名加权

    Returns / 返回:
        DataFrame (date x asset): 目标权重,每天权重和为 1.0

    Example / 示例:
        >>> # 做多动量因子前20%的股票
        >>> weights = top_k_long_only(momentum_factor, k_pct=0.2)
    """

    if not (0 < k_pct <= 1):
        raise ValueError(f"k_pct must be in (0, 1], got {k_pct}")
    
    if factor.empty:
        logger.warning("Empty factor DataFrame, returning empty weights")
        return pd.DataFrame(index=factor.index, columns=factor.columns, dtype=float)
    
    logger.debug(f"Computing long-only weights: top {k_pct:.1%}, equal_weight={equal_weight}")

    weights = pd.DataFrame(0.0, index=factor.index, columns=factor.columns)

    for date in factor.index:
        scores = factor.loc[date]

        # 过滤有效分数(非 NaN 且有限)
        # Filter valid scores (non-NaN and finite)
        valid_scores = scores.dropna()
        valid_scores = valid_scores[np.isfinite(valid_scores)]
        if valid_scores.empty:
            logger.debug(f"No valid scores on {date}, skipping")
            continue
        
        n_assets = len(valid_scores)
        n_select = max(1, int(n_assets * k_pct))  # 至少选择 1 个资产

        # 选择因子值最高的 K 个资产 / Select top K assets
        top_assets = valid_scores.nlargest(n_select).index

        if equal_weight:
            # 等权重 / Equal weight
            weights.loc[date, top_assets] = 1.0 / len(top_assets)
        else:
            # 按归一化排名加权 / Weight by normalized rank
            ranks = valid_scores[top_assets].rank(pct=True)
            rank_sum = ranks.sum()
            if rank_sum > 0:
                weights.loc[date, top_assets] = ranks / rank_sum
            else:
                # 回退到等权重 / Fallback to equal weight if rank sum is zero
                weights.loc[date, top_assets] = 1.0 / len(top_assets)
        
    return weights
    
def top_k_long_short(
    factor: PandasDataFrame,
    k_pct: float = 0.2,
    long_weight: float = 1.0,
    short_weight: float = 1.0,
) -> PandasDataFrame:
    """
    Long-short strategy: long top K%, short bottom K%.
    多空策略:做多前 K%,做空后 K%.

    策略逻辑 / Strategy Logic:
    1. 做多因子值最高的前 K% 资产(权重为正)
    2. 做空因子值最低的后 K% 资产(权重为负)
    3. 可调整多空两边的权重比例

    美元中性 / Dollar-Neutral:
        当 long_weight = short_weight = 1.0 时,组合是美元中性的
        (多头金额 = 空头金额)

    Args / 参数:
        factor: 因子矩阵 (date x asset),因子分数
        k_pct: 每边的资产比例 (0-0.5),如 0.2 = 前/后各 20%
        long_weight: 多头总权重 (默认 1.0)
        short_weight: 空头总权重 (默认 1.0,实现美元中性)

    Returns / 返回:
        DataFrame (date x asset): 目标权重,正=多头,负=空头

    Example / 示例:
        >>> # 美元中性:多头 $1,空头 $1
        >>> weights = top_k_long_short(momentum_factor, k_pct=0.2)
        
        >>> # 多头偏向:多头 $1.5,空头 $0.5
        >>> weights = top_k_long_short(factor, k_pct=0.2, long_weight=1.5, short_weight=0.5)
    """
    if not (0 < k_pct <= 0.5):
        raise ValueError(f"k_pct must be in (0, 0.5] for long-short, got {k_pct}")
    if long_weight <= 0 or short_weight < 0:
        raise ValueError(f"weights must be >= 0, got long={long_weight}, short={short_weight}")
    
    if factor.empty:
        logger.warning("Empty factor DataFrame, returning empty weights")
        return pd.DataFrame(index=factor.index, columns=factor.columns, dtype=float)
    
    logger.debug(
        f"Computing long-short weights: top/bottom {k_pct:.1%}, "
        f"long={long_weight:.2f}, short={short_weight:.2f}"
    )

    weights = pd.DataFrame(0.0, index=factor.index, columns=factor.columns)

    for date in factor.index:
        scores = factor.loc[date]

        # 过滤有效分数(非 NaN 且有限)
        # Filter valid scores (non-NaN and finite)
        valid_scores = scores.dropna()
        valid_scores = valid_scores[np.isfinite(valid_scores)]
        if valid_scores.empty:
            logger.debug(f"No valid scores on {date}, skipping")
            continue

        n_assets = len(valid_scores)
        # 多空策略至少需要 2 个资产 / Need at least 2 assets for long-short
        if n_assets < 2:
            logger.debug(f"Only {n_assets} valid assets on {date}, need at least 2 for long-short")
            continue
            
        n_select = max(1, int(n_assets * k_pct))

        # 做多前 K 个资产 / Long top K
        top_assets = valid_scores.nlargest(n_select).index
        weights.loc[date, top_assets] = long_weight / len(top_assets)

        # 做空后 K 个资产 / Short bottom K
        bottom_assets = valid_scores.nsmallest(n_select).index
        weights.loc[date, bottom_assets] = -short_weight / len(bottom_assets)

    return weights

def proportional_weights(
    factor: PandasDataFrame,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    target_leverage: float = 1.0,
) -> PandasDataFrame:
    """
    Weights proportional to factor scores (after clipping and normalizing).
    按因子分数比例分配权重(经过截断和归一化处理).

    与等权重的区别 / Difference from Equal Weight:
        - 等权重:所有选中资产权重相同
        - 比例权重:因子分数越高,权重越大

    Args / 参数:
        factor: 因子矩阵 (date x asset),因子分数
        min_weight: 每个资产的最小权重(归一化后应用)
        max_weight: 每个资产的最大权重(归一化后应用)
        target_leverage: 总敞口 (1.0 = 满仓)

    Returns / 返回:
        DataFrame (date x asset): 目标权重

    Example / 示例:
        >>> # 按动量比例分配权重,单股最大 10%
        >>> weights = proportional_weights(momentum, max_weight=0.1)
    """
    if min_weight < 0 or max_weight <= 0:
        raise ValueError(f"Invalid weight bounds: min={min_weight}, max={max_weight}")
    if min_weight >= max_weight:
        raise ValueError(f"min_weight must be < max_weight")
    if target_leverage <= 0:
        raise ValueError(f"target_leverage must be > 0, got {target_leverage}")

    logger.debug(
        f"Computing proportional weights: min={min_weight:.2%}, max={max_weight:.2%}, "
        f"leverage={target_leverage:.2f}"
    )

    weights = pd.DataFrame(0.0, index=factor.index, columns=factor.columns)

    for date in factor.index:
        scores = factor.loc[date]

        # 过滤有效分数 / Filter valid scores
        valid_scores = scores.dropna()
        if valid_scores.empty:
            continue
    
        # 如果有负分数,平移到非负 / Shift scores to be non-negative if needed
        min_score = valid_scores.min()
        if min_score < 0:
            adjusted = valid_scores - min_score
        else:
            adjusted = valid_scores

        # 归一化到目标杠杆 / Normalize to sum to target leverage
        total = adjusted.sum()
        if total < 1e-12:
            # 所有分数相同,使用等权重 / All scores are identical, use equal weight
            weights.loc[date, valid_scores.index] = target_leverage / len(valid_scores)
            continue

        raw_weights = adjusted / total * target_leverage

        # 截断到边界 / Clip to bounds
        clipped = raw_weights.clip(lower=min_weight, upper=max_weight)

        # 截断后重新归一化 / Renormalize after clipping
        clipped_sum = clipped.sum()
        if clipped_sum > 1e-12:
            weights.loc[date, clipped.index] = clipped / clipped_sum * target_leverage
        
    return weights

def apply_weight_constraints(
    weights: PandasDataFrame,
    max_position_size: float | None = None,
    max_turnover: float | None = None,
    prev_weights: PandasSeries | None = None, 
) -> PandasDataFrame:
    """
    Apply position size and turnover constraints (v1 feature stub).
    应用持仓上限和换手率约束(v1 功能预留).

    Args / 参数:
        weights: 目标权重 (date x asset)
        max_position_size: 单个资产的最大权重
        max_turnover: 每次调仓的最大换手率
        prev_weights: 前一期权重(用于计算换手率)

    Returns / 返回:
        约束后的权重

    Note / 注意:
        v0 版本预留接口 - v1 完整实现.
        v0 stub - full implementation in v1.
    """
    logger.warning("apply_weight_constraints is v0 stub - will be implemented in v1")
    raise NotImplementedError("Weight constraints not yet implemented")

__all__ = [
    "top_k_long_only",
    "top_k_long_short",
    "proportional_weights",
    "apply_weight_constraints",
]
