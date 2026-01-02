"""
Drawdown metrics calculations.

Key functions:
- compute_drawdown_series: compute drawdown at each point
- compute_max_drawdown: find maximum drawdown
- compute_drawdown_metrics: comprehensive drawdown stats
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.utils.logging import get_logger
from alpha_lab.utils.typing import PandasSeries

logger = get_logger(__name__)

def compute_drawdown_series(equity_curve: PandasSeries) -> PandasSeries:
    """
    Compute drawdown series from equity curve.

    Drawdown at time t = (equity[t] - running_max[t]) / running_max[t]

    Args:
        equity_curve: Series (date -> equity value)

    Returns:
        Series (date -> drawdown), negative values indicate drawdown
    """
    if equity_curve.empty:
        logger.warning("Empty equity curve")
        return pd.Series(dtype=float)
    
    # Running maximum
    running_max = equity_curve.expanding().max()

    # Drawdown
    drawdown = (equity_curve - running_max) / running_max

    # Replace inf/nan with 0
    drawdown = drawdown.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return drawdown

def compute_max_drawdown(equity_curve: PandasSeries) -> float:
    """
    Compute maximum drawdown.

    Args:
        equity_curve: Series (date -> equity value)

    Returns:
        Maximum drawdown as positive decimal (e.g., 0.25 = 25% drawdown)
    """
    if equity_curve.empty:
        logger.warning("Empty equity curve")
        return 0.0
    
    drawdown = compute_drawdown_series(equity_curve)

    # Max drawdown is the most negative value
    max_dd = abs(drawdown.min())

    return max_dd

def compute_drawdown_metrics(equity_curve: PandasSeries) -> dict[str, float]:
    """
    Compute comprehensive drawdown metrics.

    Args:
        equity_curve: Series (date -> equity value)

    Returns:
        Dict of drawdown metrics including:
        - max_drawdown: maximum drawdown
        - avg_drawdown: average drawdown (when in drawdown)
        - max_drawdown_duration: longest drawdown period (in days)
        - current_drawdown: current drawdown
    """
    if equity_curve.empty:
        logger.warning("Empty equity curve")
        return {}

    drawdown = compute_drawdown_series(equity_curve)

    # Max drawdown
    max_dd = abs(drawdown.min())

    # Average drawdown (only when in drawdown)
    in_drawdown = drawdown[drawdown < 0]
    avg_dd = abs(in_drawdown.mean()) if not in_drawdown.empty else 0.0

    # Current drawdown
    current_dd = abs(drawdown.iloc[-1]) if not drawdown.empty else 0.0

    # Drawdown duration
    max_dd_duration = _compute_max_drawdown_duration(drawdown)

    metrics = {
        "max_drawdown": max_dd,
        "avg_drawdown": avg_dd,
        "current_drawdown": current_dd,
        "max_drawdown_duration": max_dd_duration,
    }

    logger.debug(f"Computed drawdown metrics: max_dd={max_dd:.2%}, duration={max_dd_duration}")

    return metrics


def _compute_max_drawdown_duration(drawdown: PandasSeries) -> int:
    """
    Compute maximum drawdown duration (number of periods underwater).

    Args:
        drawdown: drawdown series (negative when underwater)

    Returns:
        Maximum number of consecutive periods in drawdown
    """
    if drawdown.empty:
        return 0
    
    # Identify drawdown periods (negative drawdown)
    in_dd = (drawdown < -1e-6).astype(int)
    
    if in_dd.sum() == 0:
        return 0  # No drawdown periods

    # Find consecutive runs
    dd_groups = (in_dd != in_dd.shift()).cumsum()
    dd_runs = in_dd.groupby(dd_groups).sum()
    
    # Filter to only drawdown runs (where sum > 0)
    dd_runs_positive = dd_runs[dd_runs > 0]
    
    max_duration = int(dd_runs_positive.max()) if not dd_runs_positive.empty else 0
    
    return max_duration

def compute_calmar_ratio(
    equity_curve: PandasSeries,
    freq: str = "D",
) -> float:
    """
    Compute Calmar ratio: CAGR / Max Drawdown.

    Args:
        equity_curve: Series (date -> equity value)
        freq: data frequency

    Returns:
        Calmar ratio

    Note:
        v0 implementation - basic version.
    """
    from alpha_lab.metrics.performance import compute_cagr
    from alpha_lab.utils.math import safe_divide

    if equity_curve.empty:
        return 0.0
    
    cagr = compute_cagr(equity_curve, freq)
    max_dd = compute_max_drawdown(equity_curve)

    calmar = safe_divide(cagr, max_dd, default=0.0)

    return calmar

__all__ = [
    "compute_drawdown_series",
    "compute_max_drawdown",
    "compute_drawdown_metrics",
    "compute_calmar_ratio",
]