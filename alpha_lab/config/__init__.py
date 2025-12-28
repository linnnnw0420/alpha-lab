"""
Alpha-Lab Configuration Module
==============================

Central configuration management for the entire project.

This module provides a unified interface to all configuration objects,
including paths, backtest parameters, universe definitions, factor settings,
and ML configurations.

Usage Examples
--------------
Basic imports:
    >>> from alpha_lab.config import BacktestConfig, default_backtest_config
    >>> from alpha_lab.config import get_paths, ensure_dir
    >>> from alpha_lab.config import UNIVERSE_DEMO, get_universe

Path management:
    >>> paths = get_paths()
    >>> data_path = paths.data_dir
    >>> ensure_dir(paths.artifacts_dir)

Backtest configuration:
    >>> cfg = default_backtest_config()
    >>> cfg = cfg.with_updates(start_date="2020-01-01", commission_bps=10.0)

Universe management:
    >>> tickers = get_universe(UNIVERSE_DEMO)

Factor configuration:
    >>> from alpha_lab.config import FactorParam, FactorComboConfig

ML configuration:
    >>> from alpha_lab.config import default_ml_model_config

Design Principles
-----------------
1. **Immutable configs**: All config objects use frozen dataclasses
2. **Validation**: All inputs are validated in __post_init__
3. **Type safety**: Full type hints for IDE support
4. **No business logic**: Pure configuration, no data processing
5. **Environment aware**: Supports env var overrides for paths

Dependency Rules
----------------
- config/ depends ONLY on: stdlib, utils/typing, utils/dates (minimal)
- config/ does NOT depend on: data/, factors/, backtest/, ml/, etc.
- All upper layers depend on config/

See Also
--------
- utils/: Common utilities used by config
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Path Management
# -----------------------------------------------------------------------------

from alpha_lab.config.paths import (
    Paths,
    get_paths,
    ensure_dir,
)

# -----------------------------------------------------------------------------
# Backtest Configuration
# -----------------------------------------------------------------------------

from alpha_lab.config.backtest import (
    BacktestConfig,
    BacktestFreq,
    PriceField,
    PriceFieldType,
    default_backtest_config,
)

# -----------------------------------------------------------------------------
# Universe Configuration
# -----------------------------------------------------------------------------

from alpha_lab.config.universe import (
    UniverseConfig,
    UNIVERSE_DEMO,
    get_universe,
)

# -----------------------------------------------------------------------------
# Factor Configuration
# -----------------------------------------------------------------------------

from alpha_lab.config.factors import (
    FactorParam,
    FactorComboConfig,
    DEFAULT_FACTOR_PARAMS,
    DEFAULT_FACTOR_COMBOS,
)

# -----------------------------------------------------------------------------
# ML Configuration
# -----------------------------------------------------------------------------

from alpha_lab.config.ml import (
    MLModelConfig,
    MLFeatureConfig,
    MLSplitConfig,
    default_ml_model_config,
    default_ml_feature_config,
    default_ml_split_config,
)

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

__all__ = [
    # Path management
    "Paths",
    "get_paths",
    "ensure_dir",
    # Backtest configuration
    "BacktestConfig",
    "BacktestFreq",
    "PriceField",
    "PriceFieldType",
    "default_backtest_config",
    # Universe configuration
    "UniverseConfig",
    "UNIVERSE_DEMO",
    "get_universe",
    # Factor configuration
    "FactorParam",
    "FactorComboConfig",
    "DEFAULT_FACTOR_PARAMS",
    "DEFAULT_FACTOR_COMBOS",
    # ML configuration
    "MLModelConfig",
    "MLFeatureConfig",
    "MLSplitConfig",
    "default_ml_model_config",
    "default_ml_feature_config",
    "default_ml_split_config",
]

# -----------------------------------------------------------------------------
# Version and metadata
# -----------------------------------------------------------------------------

__version__ = "0.1.0"
__author__ = "Alpha-Lab Contributors"

# -----------------------------------------------------------------------------
# Module initialization and validation
# -----------------------------------------------------------------------------

def _validate_config_module() -> None:
    """
    Perform module-level validation checks.
    
    This runs on import to catch common misconfigurations early.
    """
    # Ensure critical paths can be resolved
    try:
        paths = get_paths()
        # Don't create dirs here; just verify resolution works
        assert paths.project_root.exists(), "Project root not found"
    except Exception as e:
        import warnings
        warnings.warn(
            f"Config module initialization warning: {e}\n"
            f"Some path resolution may fail. Set ALPHA_LAB_ROOT if needed.",
            RuntimeWarning,
            stacklevel=2,
        )

# Run validation on import (non-blocking)
_validate_config_module()