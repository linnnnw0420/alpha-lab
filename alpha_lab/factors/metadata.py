"""Metadata wrappers that keep the simple DataFrame factor API intact."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Literal

import pandas as pd

from alpha_lab.data.contracts import fingerprint_frame


@dataclass(frozen=True)
class FactorDefinition:
    name: str
    description: str = ""
    parameters: dict[str, object] = field(default_factory=dict)
    direction: Literal["higher", "lower"] = "higher"
    source_field: str = "close"
    lookback: int | None = None
    lag: int = 1
    transforms: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def definition_id(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str).encode()
        return sha256(payload).hexdigest()[:16]


@dataclass
class FactorResult:
    definition: FactorDefinition
    values: pd.DataFrame

    @property
    def data_fingerprint(self) -> str:
        return fingerprint_frame(self.values)

    def to_metadata(self) -> dict[str, object]:
        return {
            "definition": self.definition.to_dict(),
            "definition_id": self.definition.definition_id,
            "data_fingerprint": self.data_fingerprint,
            "shape": list(self.values.shape),
        }


__all__ = ["FactorDefinition", "FactorResult"]
