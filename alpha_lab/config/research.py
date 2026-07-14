"""Serializable configuration for reproducible research runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from alpha_lab.exceptions import ConfigurationError


class SerializableConfig:
    def to_dict(self) -> dict[str, object]:
        return asdict(self)  # type: ignore[arg-type]


@dataclass(frozen=True)
class DataConfig(SerializableConfig):
    source: Literal["csv", "parquet", "yfinance"] = "csv"
    path: str | None = None
    field: str = "close"
    adjusted: bool | None = True
    missing_policy: Literal["preserve", "ffill", "drop"] = "ffill"
    forward_fill_limit: int | None = 5
    sample_size: int | None = None

    def __post_init__(self) -> None:
        if self.sample_size is not None and self.sample_size < 1:
            raise ConfigurationError("sample_size must be >= 1 or None")


@dataclass(frozen=True)
class FactorConfig(SerializableConfig):
    name: str = "momentum"
    parameters: dict[str, object] = field(default_factory=lambda: {"lookback": 60})
    transforms: tuple[str, ...] = ("zscore",)
    lag: int = 1

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ConfigurationError("factor name must not be empty")
        if self.lag < 0:
            raise ConfigurationError("factor lag must be non-negative")


@dataclass(frozen=True)
class PortfolioConfig(SerializableConfig):
    method: Literal["top_k_long_only", "top_k_long_short", "proportional"] = "top_k_long_only"
    k_pct: float = 0.2
    buffer_pct: float = 0.05
    max_position_size: float | None = None

    def __post_init__(self) -> None:
        if not 0 < self.k_pct <= 1:
            raise ConfigurationError("k_pct must be in (0, 1]")


@dataclass(frozen=True)
class ExperimentConfig(SerializableConfig):
    run_name: str = "experiment"
    seed: int = 42
    artifacts_dir: str = "artifacts"
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.run_name.strip():
            raise ConfigurationError("run_name must not be empty")
        object.__setattr__(self, "artifacts_dir", str(Path(self.artifacts_dir)))


__all__ = [
    "DataConfig",
    "ExperimentConfig",
    "FactorConfig",
    "PortfolioConfig",
    "SerializableConfig",
]
