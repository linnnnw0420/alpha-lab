from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime
from enum import Enum
from typing import TypeAlias

from alpha_lab.utils.typing import DateLike

# ----------- Enums (Standard Library) ------------ 


class BacktestFreq(str, Enum):
    """
    Rebalance frequency enumeration.
    
    Inherits from str to allow direct string comparisons and JSON serialization.
    """
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"
    
    @classmethod
    def all_values(cls) -> tuple[str, ...]:
        """Return all valid frequency values."""
        return tuple(member.value for member in cls)
    
    def __str__(self) -> str:
        """Return the string value for easy printing."""
        return self.value


class PriceField(str, Enum):
    """
    Price field enumeration for backtest execution.
    
    Inherits from str to allow direct string comparisons and JSON serialization.
    """
    OPEN = "open"
    CLOSE = "close"
    HIGH = "high"
    LOW = "low"
    VWAP = "vwap"
    
    @classmethod
    def all_values(cls) -> tuple[str, ...]:
        """Return all valid price field values."""
        return tuple(member.value for member in cls)
    
    def __str__(self) -> str:
        """Return the string value for easy printing."""
        return self.value


# Type aliases for backwards compatibility and type hints
RebalanceFreq: TypeAlias = str | BacktestFreq
PriceFieldType: TypeAlias = str | PriceField

# ------------- Helpers ----------------

def _normalize_date(value: DateLike, field_name: str) -> str:
    '''
    Normalize input into ISO date string: YYYY-MM-DD.
    Keep it strict to avoid timezone/time-of-day ambiguity.
    '''

    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        s = value.strip()
        # strict ISO date validation
        try:
            date.fromisoformat(s) # raises ValueError if invalid
        except ValueError as e:
            raise ValueError(
                f"{field_name} must be ISO date 'YYYY-MM-DD', got: {value!r}"
            ) from e
        return s

    raise TypeError(f"{field_name} must be str/date/datetime, got: {type(value).__name__}")

def _validate_non_negative(name: str, value: float) -> None:
    """Validate that a numeric value is non-negative."""
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got: {value}")


def _normalize_freq(value: RebalanceFreq) -> str:
    """Normalize frequency to string value."""
    if isinstance(value, BacktestFreq):
        return value.value
    if isinstance(value, str):
        if value not in BacktestFreq.all_values():
            raise ValueError(
                f"rebalance_freq must be one of {BacktestFreq.all_values()}, "
                f"got {value!r}"
            )
        return value
    raise TypeError(
        f"rebalance_freq must be str or BacktestFreq, "
        f"got {type(value).__name__}"
    )


def _normalize_price_field(value: PriceFieldType) -> str:
    """Normalize price field to string value."""
    if isinstance(value, PriceField):
        return value.value
    if isinstance(value, str):
        if value not in PriceField.all_values():
            raise ValueError(
                f"price_field must be one of {PriceField.all_values()}, "
                f"got {value!r}"
            )
        return value
    raise TypeError(
        f"price_field must be str or PriceField, "
        f"got {type(value).__name__}"
    )


# ---------------- Core config -------------

@dataclass(slots=True, frozen=True)
class BacktestConfig:
    """
    Backtest configuration (lightweight; no pandas dependency).

    Notes:
        - start_date/end_date are normalized to 'YYYY-MM-DD' in __post_init__.
        - commission_bps/slippage_bps are in basis points (1 bps = 0.01%).
        - Supports both string values and Enum members for freq/price_field.
    
    Examples:
        Using enums:
        >>> cfg = BacktestConfig(
        ...     start_date="2020-01-01",
        ...     end_date="2024-12-31",
        ...     rebalance_freq=BacktestFreq.MONTHLY,
        ...     initial_cash=1_000_000.0,
        ...     commission_bps=5.0,
        ...     slippage_bps=2.0,
        ...     price_field=PriceField.CLOSE,
        ... )
        
        Using strings:
        >>> cfg = BacktestConfig(
        ...     start_date="2020-01-01",
        ...     end_date="2024-12-31",
        ...     rebalance_freq="M",
        ...     initial_cash=1_000_000.0,
        ...     commission_bps=5.0,
        ...     slippage_bps=2.0,
        ...     price_field="close",
        ... )
    """

    start_date: DateLike
    end_date: DateLike

    rebalance_freq: RebalanceFreq  # 'D'/'W'/'M' or BacktestFreq enum
    initial_cash: float
    
    commission_bps: float  # fee bps (round-trip handled elsewhere)
    slippage_bps: float  # slippage, bps

    price_field: PriceFieldType  # 'open'/'close'/etc or PriceField enum
    benchmark: str | None = None
    
    def __post_init__(self) -> None:
        # Normalize dates into ISO strings
        object.__setattr__(self, "start_date", _normalize_date(self.start_date, "start_date"))
        object.__setattr__(self, "end_date", _normalize_date(self.end_date, "end_date"))

        if self.start_date > self.end_date:
            raise ValueError(
                f"start_date must be <= end_date, "
                f"got {self.start_date} > {self.end_date}"
            )

        # Normalize and validate frequency
        object.__setattr__(self, "rebalance_freq", _normalize_freq(self.rebalance_freq))

        # Normalize and validate price field
        object.__setattr__(self, "price_field", _normalize_price_field(self.price_field))

        if self.initial_cash <= 0:
            raise ValueError(f"initial_cash must be > 0, got: {self.initial_cash}")

        _validate_non_negative("commission_bps", float(self.commission_bps))
        _validate_non_negative("slippage_bps", float(self.slippage_bps))
    
    def with_updates(self, **kwargs) -> BacktestConfig:
        """
        Create a new config with updated fields (re-validates via __post_init__).
        """
        return replace(self, **kwargs)

def default_backtest_config() -> BacktestConfig:
    """
    Return a reasonable default backtest config for demos/notebooks.

    You are expected to override fields in notebook, e.g.:
        cfg = default_backtest_config()
        cfg = cfg.with_updates(start_date="2018-01-01", rebalance_freq=BacktestFreq.MONTHLY)
    """
    return BacktestConfig(
        start_date="2018-01-01",
        end_date="2024-12-31",
        rebalance_freq=BacktestFreq.MONTHLY,
        initial_cash=1_000_000.0,
        commission_bps=5.0,
        slippage_bps=2.0,
        price_field=PriceField.CLOSE,
        benchmark=None,  # e.g. "SPY" if you have it in your universe
    )

__all__ = [
    "BacktestConfig",
    "BacktestFreq",
    "PriceField",
    "RebalanceFreq",
    "PriceFieldType",
    "default_backtest_config",
]