"""
Alpha-Lab Utilities Module
===========================

Common utility functions and type definitions used across the project.

This module provides:
- Logging utilities with consistent formatting
- Date/time handling and calendar operations
- Random seed management for reproducibility
- Financial math utilities (annualization, safe division, etc.)
- Type aliases for better code readability

Usage Examples
--------------
Logging:
    >>> from alpha_lab.utils import setup_logging, get_logger
    >>> setup_logging(log_level="INFO")
    >>> logger = get_logger(__name__)
    >>> logger.info("Starting backtest")

Date operations:
    >>> from alpha_lab.utils import parse_date, generate_rebalance_dates
    >>> from alpha_lab.utils import annualization_factor
    >>> ts = parse_date("2024-01-15")
    >>> factor = annualization_factor("D")  # 252 for daily

Random seed:
    >>> from alpha_lab.utils import set_global_seed
    >>> set_global_seed(42)  # Reproducible results

Math utilities:
    >>> from alpha_lab.utils import annualize_return, safe_divide
    >>> annual_ret = annualize_return(0.01, 252)
    >>> ratio = safe_divide(a, b, default=0.0)

Type annotations:
    >>> from alpha_lab.utils import DateLike, PricePanel, FactorPanel
    >>> def load_data(start: DateLike, end: DateLike) -> PricePanel:
    ...     ...

Design Principles
-----------------
1. **Pure utilities**: No business logic, only reusable helpers
2. **Defensive coding**: All functions handle edge cases gracefully
3. **Type safety**: Full type hints for all public functions
4. **Dependencies**: pandas and NumPy are required; research extras are optional
5. **Performance**: Efficient implementations, avoid unnecessary copies

Dependency Rules
----------------
- utils/ depends ONLY on: stdlib, numpy, pandas (optional imports)
- utils/ does NOT depend on: config/ (to avoid circular deps), data/, factors/, etc.
- All other modules can depend on utils/

See Also
--------
- config/: Project configuration objects
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Date and Time Utilities
# -----------------------------------------------------------------------------
from alpha_lab.utils.dates import (
    align_to_trading_day,
    annualization_factor,
    generate_rebalance_dates,
    parse_date,
)

# -----------------------------------------------------------------------------
# Logging Utilities
# -----------------------------------------------------------------------------
from alpha_lab.utils.logging import (
    get_logger,
    setup_logging,
)

# -----------------------------------------------------------------------------
# Mathematical Utilities
# -----------------------------------------------------------------------------
from alpha_lab.utils.math import (
    annualize_return,
    annualize_vol,
    from_log_return,
    np_to_array,
    safe_divide,
    to_log_return,
)

# -----------------------------------------------------------------------------
# Random Seed Management
# -----------------------------------------------------------------------------
from alpha_lab.utils.random import (
    new_numpy_random_generator,
    set_global_seed,
)

# -----------------------------------------------------------------------------
# Type Aliases and Definitions
# -----------------------------------------------------------------------------
from alpha_lab.utils.typing import (
    Asset,
    Bps,
    # Date/time
    DateLike,
    EquityCurve,
    FactorName,
    FactorPanel,
    ModelName,
    NpArray,
    # Pandas/Numpy types
    PandasDataFrame,
    PandasDatetimeIndex,
    PandasIndex,
    PandasSeries,
    PandasTimestamp,
    # Path
    PathLike,
    PriceField,
    # Project-specific shapes
    PricePanel,
    # Literals
    RebalanceFreq,
    RunName,
    # Primitives
    Ticker,
    TradeFrame,
    WeightVector,
)

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    # Date/time utilities
    "annualization_factor",
    "parse_date",
    "align_to_trading_day",
    "generate_rebalance_dates",
    # Random seed
    "set_global_seed",
    "new_numpy_random_generator",
    # Math utilities
    "safe_divide",
    "annualize_return",
    "annualize_vol",
    "to_log_return",
    "from_log_return",
    "np_to_array",
    # Type aliases - Primitives
    "Ticker",
    "Asset",
    "FactorName",
    "ModelName",
    "RunName",
    "Bps",
    # Type aliases - Date/time
    "DateLike",
    # Type aliases - Literals
    "RebalanceFreq",
    "PriceField",
    # Type aliases - Path
    "PathLike",
    # Type aliases - Pandas/Numpy
    "PandasDataFrame",
    "PandasSeries",
    "PandasIndex",
    "PandasTimestamp",
    "PandasDatetimeIndex",
    "NpArray",
    # Type aliases - Project shapes
    "PricePanel",
    "FactorPanel",
    "WeightVector",
    "EquityCurve",
    "TradeFrame",
]

# -----------------------------------------------------------------------------
# Version and metadata
# -----------------------------------------------------------------------------

__version__ = "0.2.0"
__author__ = "Alpha-Lab Contributors"

# -----------------------------------------------------------------------------
# Module initialization
# -----------------------------------------------------------------------------


def _check_optional_dependencies() -> dict[str, bool]:
    """
    Check which optional dependencies are available.

    Returns:
        Dict mapping package name to availability status
    """
    deps = {}

    from importlib.util import find_spec

    deps["pandas"] = True
    deps["numpy"] = True
    deps["pyarrow"] = find_spec("pyarrow") is not None
    deps["sklearn"] = find_spec("sklearn") is not None
    deps["yfinance"] = find_spec("yfinance") is not None

    return deps


# Store dependency status (useful for debugging)
_OPTIONAL_DEPS = _check_optional_dependencies()


def get_optional_deps() -> dict[str, bool]:
    """
    Get status of optional dependencies.

    Returns:
        Dict mapping package name to availability status

    Example:
        >>> from alpha_lab.utils import get_optional_deps
        >>> deps = get_optional_deps()
        >>> if deps["pandas"]:
        ...     print("Pandas is available")
    """
    return _OPTIONAL_DEPS.copy()


# Add to public API
__all__.append("get_optional_deps")
