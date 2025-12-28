from __future__ import annotations

import math
from typing import Any

try: # optional
    import numpy as np #type: ignore
except ImportError: # pragma: no cover
    np = None

def safe_divide(a: float, b: float, default: float = 0.0, eps: float = 1e-12) -> float:
    """
    Divide a by b safely; return default if |b| is too small.

    Args:
        a: numerator
        b: denominator
        default: value to return when denominator is (near) zero
        eps: tolerance to treat b as zero

    Returns:
        a / b if |b| > eps, else default
    """

    return a / b if abs(b) > eps else default

def annualize_return(period_return: float, periods_per_year: float) -> float:
    """
    Convert per-period arithmetic return to anualized compounded return.

    Formula: (1 + r) ** periods per year - 1

    Args:
        period_return: per-period arithmetic return (e.g., mean daily return)
        periods_per_year: number of periods in a year (e.g., 252 for daily)

    Returns:
        Annualized compounded return
    """
    return (1.0 + period_return) ** periods_per_year - 1.0

def annualize_vol(vol: float, periods_per_year: float) -> float:
    """
    Annualize volatility (std of per-period returns).

    Formula: vol * sqrt(periods_per_year)
    """
    root = math.sqrt(periods_per_year)
    return vol * root

def to_log_return(r: float) -> float:
    """
    Convert arithmetic return to log return:
    """
    return math.log1p(r)

def from_log_return(lr: float) -> float:
    """
    Convert log return back to arithmetic return exp(lr) - 1
    """
    return math.expm1(lr) 

def np_to_array(x: Any) -> Any:
    """
    Convert input to numpy array if numpy is available; otherwise return input.
    """
    if np is None:
        return x
    return np.asarray(x)

__all__ = [
    "safe_divide",
    "annualize_return",
    "annualize_vol",
    "to_log_return",
    "from_log_return",
    "np_to_array",
]