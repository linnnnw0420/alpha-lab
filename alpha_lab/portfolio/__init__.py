# alpha_lab/portfolio/__init__.py
"""
Portfolio construction module.

Provides:
- rebalance: generate rebalance dates
- weighting: convert factor scores to target weights
"""

from __future__ import annotations

from alpha_lab.portfolio.rebalance import generate_rebalance_schedule
from alpha_lab.portfolio.weighting import (
    apply_weight_constraints,
    proportional_weights,
    top_k_long_only,
    top_k_long_short,
)

__all__ = [
    "generate_rebalance_schedule",
    "top_k_long_only",
    "top_k_long_short",
    "proportional_weights",
    "apply_weight_constraints",
]
