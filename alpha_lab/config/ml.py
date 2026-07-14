from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date, datetime

from alpha_lab.utils.typing import DateLike


@dataclass(frozen=True, slots=True)
class MLModelConfig:
    """
    Machine learning model configuration.

    Args:
        model_type: Model identifier ('random_forest', 'xgboost', 'lightgbm', etc.)
        params: Hyperparameters dict (model-specific)

    Examples:
        >>> MLModelConfig(model_type='random_forest', params={'n_estimators': 100, 'max_depth': 5})
    """

    model_type: str
    params: dict[str, object]

    def __post_init__(self) -> None:
        if not self.model_type.strip():
            raise ValueError("model_type cannot be empty")
        if not isinstance(self.params, dict):
            raise TypeError(f"params must be dict, got {type(self.params).__name__}")

    def with_updates(self, **kwargs) -> MLModelConfig:
        """Return a new config with updated fields."""
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MLFeatureConfig:
    """
    Feature engineering configuration for ML models.

    Args:
        feature_window: Lookback window for feature construction (in periods)
        target_horizon: Forward-looking horizonfor target variable
        use_factors: List of factor names to include as featrues
        use_tech_indicators: List of technical indicator names to include
    """

    feature_window: int
    target_horizon: int
    use_factors: list[str]
    use_tech_indicators: list[str]

    def __post_init__(self) -> None:
        if self.feature_window <= 0:
            raise ValueError(f"feature_window must be > 0, got {self.feature_window}")
        if self.target_horizon <= 0:
            raise ValueError(f"target_horizon must be > 0, got {self.target_horizon}")
        # Allow empty lists (user might only use factors or only tech indicators)
        if not isinstance(self.use_factors, list):
            raise TypeError("use_factors must be a list")
        if not isinstance(self.use_tech_indicators, list):
            raise TypeError("use_tech_indicators must be a list")

    def with_updates(self, **kwargs) -> MLFeatureConfig:
        """Return a new config with updated fields."""
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MLSplitConfig:
    """
    Train/validation/test split configuration.

    Args:
        train_start: Training period start date
        train_end: Training period end date
        valid_start: Validation period start date
        valid_end: Validation period end date
        test_start: Test period start date
        test_end: Test period end date
        walk_forward: If True, enable walk-forward validation

    Notes:
        - Dates are stored as-is; validation happens in higher-level modules
        - For walk-forward, the splits define the initial window
    """

    train_start: DateLike
    train_end: DateLike
    valid_start: DateLike
    valid_end: DateLike
    test_start: DateLike
    test_end: DateLike
    walk_forward: bool = False

    def __post_init__(self) -> None:
        values = [
            self.train_start,
            self.train_end,
            self.valid_start,
            self.valid_end,
            self.test_start,
            self.test_end,
        ]
        normalized = [value.date() if isinstance(value, datetime) else value for value in values]
        parsed = [
            date.fromisoformat(value) if isinstance(value, str) else value for value in normalized
        ]
        if any(not isinstance(value, date) for value in parsed):
            raise TypeError("ML split dates must be ISO strings, date, or datetime")
        if parsed != sorted(parsed):
            raise ValueError("ML split dates must be chronologically ordered")

    def with_updates(self, **kwargs) -> MLSplitConfig:
        """Return a new config with updated fields."""
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Default configurations (examples for demos/notebooks)
# -----------------------------------------------------------------------------


def default_ml_model_config() -> MLModelConfig:
    """
    Return a transparent default Ridge model config.

    Override in practice:
        cfg = default_ml_model_config()
        cfg = cfg.with_updates(params={'n_estimators': 200})
    """
    return MLModelConfig(
        model_type="ridge",
        params={"alpha": 1.0},
    )


def default_ml_feature_config() -> MLFeatureConfig:
    """
    Return a reasonable default feature config.
    """
    return MLFeatureConfig(
        feature_window=20,  # 20-period lookback
        target_horizon=5,  # predict 5-period ahead return
        use_factors=[],  # Empty by default, fill in after factors are defined
        use_tech_indicators=[],  # Empty by default
    )


def default_ml_split_config() -> MLSplitConfig:
    """
    Return a reasonable default train/valid/test split.
    """
    return MLSplitConfig(
        train_start="2015-01-01",
        train_end="2019-12-31",
        valid_start="2020-01-01",
        valid_end="2021-12-31",
        test_start="2022-01-01",
        test_end="2024-12-31",
        walk_forward=False,
    )


__all__ = [
    "MLModelConfig",
    "MLFeatureConfig",
    "MLSplitConfig",
    "default_ml_model_config",
    "default_ml_feature_config",
    "default_ml_split_config",
]
