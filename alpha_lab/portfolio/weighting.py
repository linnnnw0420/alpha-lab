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

import numpy as np
import pandas as pd

from alpha_lab.utils.logging import get_logger
from alpha_lab.utils.typing import PandasDataFrame, PandasSeries

logger = get_logger(__name__)


def top_k_long_only(
    factor: PandasDataFrame,
    k_pct: float = 0.2,
    equal_weight: bool = True,
    buffer_pct: float = 0.0,
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
        buffer_pct: 换仓缓冲比例(0-1),>0 启用 hysteresis/no-trade band
                   buffer_pct=0 表示每期完全按 top-K 重算

    Returns / 返回:
        DataFrame (date x asset): 目标权重,每天权重和为 1.0

    Example / 示例:
        >>> # 做多动量因子前20%的股票
        >>> weights = top_k_long_only(momentum_factor, k_pct=0.2, buffer_pct=0.05)
    """

    if not (0 < k_pct <= 1):
        raise ValueError(f"k_pct must be in (0, 1], got {k_pct}")
    if buffer_pct < 0:
        raise ValueError(f"buffer_pct must be >= 0, got {buffer_pct}")

    if factor.empty:
        logger.warning("Empty factor DataFrame, returning empty weights")
        return pd.DataFrame(index=factor.index, columns=factor.columns, dtype=float)

    use_hysteresis = buffer_pct > 0
    logger.debug(
        f"Computing long-only weights: top {k_pct:.1%}, equal_weight={equal_weight}, "
        f"buffer_pct={buffer_pct:.1%}"
    )

    weights = pd.DataFrame(0.0, index=factor.index, columns=factor.columns)
    prev_selected: list[str] = []

    for date in factor.index:
        scores = factor.loc[date]

        # 过滤有效分数(非 NaN 且有限)
        # Filter valid scores (non-NaN and finite)
        valid_scores = scores.dropna()
        valid_scores = valid_scores[np.isfinite(valid_scores)]
        if valid_scores.empty:
            logger.debug(f"No valid scores on {date}, skipping")
            continue

        ranked = valid_scores.sort_values(ascending=False)
        n_assets = len(ranked)
        n_select = max(1, int(n_assets * k_pct))  # 至少选择 1 个资产

        if not use_hysteresis or not prev_selected:
            # 选择因子值最高的 K 个资产 / Select top K assets
            selected = list(ranked.index[:n_select])
        else:
            buffer_n = max(0, int(n_assets * buffer_pct))
            keep_cutoff = min(n_assets, n_select + buffer_n)
            buffer_assets = list(ranked.index[:keep_cutoff])

            # 保留仍在 top-(k+buffer) 的旧持仓
            # Keep previous holdings if still in top-(k+buffer)
            keep = [t for t in prev_selected if t in buffer_assets]
            selected = keep[:n_select]

            # 用最新 top-k 填满
            # Fill with current top-k
            for t in ranked.index[:n_select]:
                if t not in selected:
                    selected.append(t)
                if len(selected) >= n_select:
                    break

            # 如果还不够,从 buffer 内补齐
            if len(selected) < n_select:
                for t in buffer_assets:
                    if t not in selected:
                        selected.append(t)
                    if len(selected) >= n_select:
                        break

        prev_selected = selected

        if equal_weight:
            # 等权重 / Equal weight
            weights.loc[date, selected] = 1.0 / len(selected)
        else:
            # 按归一化排名加权 / Weight by normalized rank
            ranks = valid_scores[selected].rank(pct=True)
            rank_sum = ranks.sum()
            if rank_sum > 0:
                weights.loc[date, selected] = ranks / rank_sum
            else:
                # 回退到等权重 / Fallback to equal weight if rank sum is zero
                weights.loc[date, selected] = 1.0 / len(selected)

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
        raise ValueError("min_weight must be < max_weight")
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
    Apply position size and turnover constraints.
    应用持仓上限和换手率约束.

    Args / 参数:
        weights: 目标权重 (date x asset)
        max_position_size: 单个资产的最大权重
        max_turnover: 每次调仓的最大换手率
        prev_weights: 前一期权重(用于计算换手率)

    Returns / 返回:
        约束后的权重

    Note / 注意:
        - 先做持仓上限截断,再做换手率限制
        - 如果传入 prev_weights,第一期用它对齐;否则假设上一期为 0
    """
    if weights.empty:
        logger.warning("Empty weights DataFrame")
        return weights.copy()
    if max_position_size is not None and max_position_size <= 0:
        raise ValueError(f"max_position_size must be > 0, got {max_position_size}")
    if max_turnover is not None and max_turnover < 0:
        raise ValueError(f"max_turnover must be >= 0, got {max_turnover}")

    adjusted = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
    prev = None
    if prev_weights is not None:
        prev = prev_weights.reindex(weights.columns).fillna(0.0).astype(float)

    for date in weights.index:
        target = weights.loc[date].fillna(0.0).astype(float)

        # (1) 单票上限 / Position cap
        if max_position_size is not None:
            orig_long = target.clip(lower=0.0).sum()
            orig_short = (-target.clip(upper=0.0)).sum()
            target = target.clip(lower=-max_position_size, upper=max_position_size)
            target = _normalize_long_short(target, orig_long, orig_short)

        # (2) 换手限制 / Turnover limit
        if max_turnover is not None:
            if prev is None:
                prev = pd.Series(0.0, index=weights.columns)
            target = _apply_turnover_limit(prev, target, max_turnover)

        adjusted.loc[date] = target
        prev = target

    return adjusted


def _normalize_long_short(
    weights: PandasSeries,
    target_long: float,
    target_short: float,
) -> PandasSeries:
    """
    Normalize long/short sides separately to preserve original exposure.
    分别归一化多空两侧,保持原始敞口规模.
    """
    long_side = weights.clip(lower=0.0)
    short_side = -weights.clip(upper=0.0)

    if target_long > 1e-12 and long_side.sum() > 1e-12:
        long_side = long_side * (target_long / long_side.sum())
    if target_short > 1e-12 and short_side.sum() > 1e-12:
        short_side = short_side * (target_short / short_side.sum())

    return long_side - short_side


def _apply_turnover_limit(
    prev_weights: PandasSeries,
    target_weights: PandasSeries,
    max_turnover: float,
) -> PandasSeries:
    """
    Cap turnover by scaling the move from prev to target.
    通过缩放 (target-prev) 的步长控制换手率.
    """
    delta = target_weights - prev_weights
    turnover = delta.abs().sum()
    if turnover <= max_turnover or turnover < 1e-12:
        return target_weights
    scaling = max_turnover / turnover
    return prev_weights + scaling * delta


__all__ = [
    "top_k_long_only",
    "top_k_long_short",
    "proportional_weights",
    "apply_weight_constraints",
]
