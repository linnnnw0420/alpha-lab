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

__version__ = "0.2.0"
__author__ = "Alpha-Lab Contributors"

# -----------------------------------------------------------------------------
# Core config utilities (most commonly used)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Backtest
# -----------------------------------------------------------------------------
from alpha_lab.backtest.engine import BacktestResult, run_backtest
from alpha_lab.config import (
    UNIVERSE_DEMO,
    # Backtest configuration
    BacktestConfig,
    DataConfig,
    ExperimentConfig,
    FactorConfig,
    # Path management
    Paths,
    PortfolioConfig,
    # Universe configuration
    UniverseConfig,
    default_backtest_config,
    ensure_dir,
    get_paths,
    get_universe,
)

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
from alpha_lab.data.loader import (
    load_prices,
    load_returns,
    load_sampled_prices,
    load_universe,
)
from alpha_lab.data.sampling import UniverseSelection, sample_universe
from alpha_lab.exceptions import (
    AlignmentError,
    ConfigurationError,
    DataContractError,
    LookaheadError,
)
from alpha_lab.factors.metadata import FactorDefinition, FactorResult

# -----------------------------------------------------------------------------
# Factors (lazy import to avoid circular deps, but expose common ones)
# -----------------------------------------------------------------------------
from alpha_lab.factors.momentum import momentum
from alpha_lab.factors.transform import (
    neutralize,
    neutralize_industry,
    rank_normalize,
    winsorize,
    zscore,
)
from alpha_lab.metrics.drawdown import (
    compute_drawdown_metrics,
    compute_drawdown_series,
    compute_max_drawdown,
)
from alpha_lab.metrics.performance import (
    compute_annualized_vol,
    compute_cagr,
    compute_sharpe_ratio,
)

# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
from alpha_lab.metrics.summary import (
    generate_backtest_summary,
    generate_metrics_summary,
)
from alpha_lab.portfolio.rebalance import generate_rebalance_schedule

# -----------------------------------------------------------------------------
# Portfolio
# -----------------------------------------------------------------------------
from alpha_lab.portfolio.weighting import (
    proportional_weights,
    top_k_long_only,
    top_k_long_short,
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
    "DataConfig",
    "ExperimentConfig",
    "FactorConfig",
    "PortfolioConfig",
    "default_backtest_config",
    "UniverseConfig",
    "UNIVERSE_DEMO",
    "get_universe",
    # Data
    "load_universe",
    "load_prices",
    "load_returns",
    "load_sampled_prices",
    "sample_universe",
    "UniverseSelection",
    # Factors
    "momentum",
    "zscore",
    "winsorize",
    "rank_normalize",
    "neutralize",
    "neutralize_industry",
    "FactorDefinition",
    "FactorResult",
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
    "AlignmentError",
    "ConfigurationError",
    "DataContractError",
    "LookaheadError",
]
