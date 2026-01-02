# alpha_lab/factors/__init__.py
"""
Factor calculation module.

Provides:
- momentum: momentum factor calculation
- transform: winsorize, zscore, rank utilities
"""

from __future__ import annotations

from alpha_lab.factors.momentum import momentum, momentum_multi_period
from alpha_lab.factors.transform import winsorize, zscore, rank_normalize

__all__ = [
    "momentum",
    "momentum_multi_period",
    "winsorize",
    "zscore",
    "rank_normalize",
]