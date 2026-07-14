from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class FactorParam:
    """
    Configuration for a single factor calculation.
    """

    name: str
    lookback: int  # 回看窗口长度
    neutralize_industry: bool = False  # 是否行业中性化
    winsorize: bool = True  # 是否去极值
    zscore: bool = True  # 是否标准化为 z-score

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Factor name cannot be empty.")
        if self.lookback <= 0:
            raise ValueError(f"lookback must be > 0, got {self.lookback}")

    def with_updates(self, **kwargs) -> FactorParam:
        """Return a new config with updated fields."""
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FactorComboConfig:
    """
    Configuration for combining multiple factors.
    """

    name: str
    factors: dict[str, float]  # 因子名 -> 权重

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Combo name cannot be empty")
        if not self.factors:
            raise ValueError("factors dict cannot be empty")

        # 简单的权重检查
        # 不强制 sum=1, 因为有时候可能是打分加总
        for fname, _weight in self.factors.items():
            if not fname.strip():
                raise ValueError("Factor name in combo cannot be empty")

    def with_updates(self, **kwargs) -> FactorComboConfig:
        """Return a new config with updated fields."""
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_FACTOR_PARAMS: dict[str, FactorParam] = {
    "momentum_3m": FactorParam(name="momentum_3m", lookback=60),
    "momentum_6m": FactorParam(name="momentum_6m", lookback=120),
}

DEFAULT_FACTOR_COMBOS: dict[str, FactorComboConfig] = {
    "balanced_momentum": FactorComboConfig(
        name="balanced_momentum", factors={"momentum_3m": 0.5, "momentum_6m": 0.5}
    ),
}

__all__ = [
    "FactorParam",
    "FactorComboConfig",
    "DEFAULT_FACTOR_PARAMS",
    "DEFAULT_FACTOR_COMBOS",
]
