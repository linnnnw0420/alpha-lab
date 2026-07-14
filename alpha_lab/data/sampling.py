"""Reproducible fixed-universe selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256

import numpy as np
import pandas as pd

from alpha_lab.data.contracts import fingerprint_frame, normalize_price_panel
from alpha_lab.exceptions import ConfigurationError


@dataclass(frozen=True)
class UniverseSelection:
    candidates: tuple[str, ...]
    selected: tuple[str, ...]
    excluded: dict[str, str]
    seed: int
    sample_size: int
    min_observations: int
    min_coverage: float
    require_start: bool
    require_end: bool
    data_fingerprint: str

    @property
    def selection_id(self) -> str:
        payload = repr(asdict(self)).encode()
        return sha256(payload).hexdigest()[:16]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["selection_id"] = self.selection_id
        return result


def sample_universe(
    prices: pd.DataFrame,
    sample_size: int,
    *,
    seed: int = 42,
    min_observations: int = 1,
    min_coverage: float = 0.0,
    require_start: bool = False,
    require_end: bool = False,
) -> tuple[pd.DataFrame, UniverseSelection]:
    """Filter and sample a universe once, returning its audit record."""
    if sample_size < 1:
        raise ConfigurationError("sample_size must be >= 1")
    if min_observations < 0:
        raise ConfigurationError("min_observations must be non-negative")
    if not 0 <= min_coverage <= 1:
        raise ConfigurationError("min_coverage must be in [0, 1]")

    panel = normalize_price_panel(prices)
    candidates = tuple(map(str, panel.columns))
    excluded: dict[str, str] = {}
    eligible: list[str] = []
    n_dates = max(len(panel), 1)
    for ticker in candidates:
        values = panel[ticker]
        valid = values.notna() & np.isfinite(values) & (values > 0)
        reasons: list[str] = []
        if int(valid.sum()) < min_observations:
            reasons.append("insufficient_observations")
        if float(valid.sum() / n_dates) < min_coverage:
            reasons.append("insufficient_coverage")
        if require_start and (values.empty or not bool(valid.iloc[0])):
            reasons.append("missing_start")
        if require_end and (values.empty or not bool(valid.iloc[-1])):
            reasons.append("missing_end")
        if reasons:
            excluded[ticker] = ",".join(reasons)
        else:
            eligible.append(ticker)

    if sample_size > len(eligible):
        raise ConfigurationError(
            f"sample_size={sample_size} exceeds {len(eligible)} eligible tickers"
        )
    rng = np.random.default_rng(seed)
    chosen_positions = np.sort(rng.choice(len(eligible), size=sample_size, replace=False))
    selected = tuple(eligible[position] for position in chosen_positions)
    record = UniverseSelection(
        candidates=candidates,
        selected=selected,
        excluded=excluded,
        seed=int(seed),
        sample_size=sample_size,
        min_observations=min_observations,
        min_coverage=min_coverage,
        require_start=require_start,
        require_end=require_end,
        data_fingerprint=fingerprint_frame(panel),
    )
    return panel.loc[:, list(selected)].copy(), record


__all__ = ["UniverseSelection", "sample_universe"]
