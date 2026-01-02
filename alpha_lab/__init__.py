"""
Alpha-Lab: Quantitative Factor Research Framework
==================================================

A lightweight framework for:
- Factor construction and transformation
- Portfolio optimization and weighting
- Backtesting and performance analysis
- Data loading and management

Quick Start
-----------
>>> from alpha_lab import get_paths, load_prices, run_backtest
>>> from alpha_lab.factors import momentum, zscore
>>> from alpha_lab.portfolio import top_k_long_only

Version
-------
"""

__version__ = "0.1.0"
__author__ = "Alpha-Lab Contributors"

# -----------------------------------------------------------------------------
# Core config utilities (most commonly used)
# -----------------------------------------------------------------------------

from alpha_lab.config import (
    # Path management
    Paths,
    get_paths,
    ensure_dir,
    # Backtest configuration
    BacktestConfig,
    default_backtest_config,
    # Universe configuration
    UniverseConfig,
    UNIVERSE_DEMO,
    get_universe,
)

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

from alpha_lab.data.loader import (
    load_universe,
    load_prices,
    load_returns,
)

# -----------------------------------------------------------------------------
# Factors (lazy import to avoid circular deps, but expose common ones)
# -----------------------------------------------------------------------------

from alpha_lab.factors.momentum import momentum
from alpha_lab.factors.transform import zscore, winsorize, rank_normalize

# -----------------------------------------------------------------------------
# Portfolio
# -----------------------------------------------------------------------------

from alpha_lab.portfolio.weighting import (
    top_k_long_only,
    top_k_long_short,
    proportional_weights,
)
from alpha_lab.portfolio.rebalance import generate_rebalance_schedule

# -----------------------------------------------------------------------------
# Backtest
# -----------------------------------------------------------------------------

from alpha_lab.backtest.engine import run_backtest, BacktestResult

# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

from alpha_lab.metrics.summary import (
    generate_metrics_summary,
    generate_backtest_summary,
)
from alpha_lab.metrics.performance import (
    compute_cagr,
    compute_sharpe_ratio,
    compute_annualized_vol,
)
from alpha_lab.metrics.drawdown import (
    compute_drawdown_series,
    compute_max_drawdown,
    compute_drawdown_metrics,
)

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

__all__ = [
    # Version
    "__version__",
    # Config
    "Paths",
    "get_paths",
    "ensure_dir",
    "BacktestConfig",
    "default_backtest_config",
    "UniverseConfig",
    "UNIVERSE_DEMO",
    "get_universe",
    # Data
    "load_universe",
    "load_prices",
    "load_returns",
    # Factors
    "momentum",
    "zscore",
    "winsorize",
    "rank_normalize",
    # Portfolio
    "top_k_long_only",
    "top_k_long_short",
    "proportional_weights",
    "generate_rebalance_schedule",
    # Backtest
    "run_backtest",
    "BacktestResult",
    # Metrics
    "generate_metrics_summary",
    "generate_backtest_summary",
    "compute_cagr",
    "compute_sharpe_ratio",
    "compute_annualized_vol",
    "compute_drawdown_series",
    "compute_max_drawdown",
    "compute_drawdown_metrics",
]
