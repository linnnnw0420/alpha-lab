from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from alpha_lab.utils.typing import DateLike, PriceField, Ticker

@dataclass(frozen=True, slots=True)
class UniverseConfig:
    """
    Static universe definition (simplified). Later you can add dynamic filters
    (by date, metadata, etc.). Optional dynamic_by_price uses price availability
    to filter tickers when as_of is provided.
    """
    name: str
    tickers: tuple[Ticker, ...]
    min_mkt_cap: float | None = None               # optional market-cap floor
    min_avg_turnover: float | None = None          # optional liquidity floor
    exclude_st: bool = True                        # placeholder for future ST filter
    dynamic_by_price: bool = False                 # filter by price availability when as_of is provided
    price_field: PriceField = "close"              # price field for dynamic filtering
    min_valid_price: float = 0.0                   # minimum price to treat as tradable

    def __post_init__(self) -> None:
        if not self.tickers:
            raise ValueError("tickers must not be empty")
        # ensure unique tickers
        if len(set(self.tickers)) != len(self.tickers):
            raise ValueError("tickers must be unique")
        # numeric sanity (if provided)
        if self.min_mkt_cap is not None and self.min_mkt_cap < 0:
            raise ValueError("min_mkt_cap must be >= 0")
        if self.min_avg_turnover is not None and self.min_avg_turnover < 0:
            raise ValueError("min_avg_turnover must be >= 0")
        if self.min_valid_price < 0:
            raise ValueError("min_valid_price must be >= 0")
        if self.price_field not in {"open", "high", "low", "close", "vwap"}:
            raise ValueError(f"price_field must be a valid field, got {self.price_field!r}")
        
    def with_updates(self, **kwargs) -> UniverseConfig:
        """Return a new config with updated fields (re-validates)."""
        return replace(self, **kwargs)
    
def _as_ticker_tuple(values: Sequence[str]) -> tuple[Ticker, ...]:
    return tuple(str(v).strip() for v in values if str(v).strip())

UNIVERSE_DEMO = UniverseConfig(
    name="DEMO_TOPS_US",
    tickers=_as_ticker_tuple(["AAPL", "MSFT", "GOOGL", "AMZN", "META"]),
    exclude_st=False,
)

def get_universe(config: UniverseConfig, as_of: DateLike | None = None) -> list[Ticker]:
    """
    Return the tradable tickers for the given date.
    Current implementation is static; future versions can filter by date/metadata.
    """
    # NOTE: as_of is reserved for future dynamic rules (index membership, listing status, etc.)
    return list(config.tickers)

__all__ = ["UniverseConfig", "UNIVERSE_DEMO", "get_universe"]
