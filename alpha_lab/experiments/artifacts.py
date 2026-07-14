"""Experiment artifact persistence with compact, inspectable formats."""

from __future__ import annotations

import importlib.metadata
import json
import platform
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pandas as pd

from alpha_lab.backtest.result import BacktestResult


def _jsonable(value: object) -> object:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())  # type: ignore[union-attr]
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def make_run_id(config: object, fingerprints: Mapping[str, str] | None = None) -> str:
    payload = {"config": _jsonable(config), "fingerprints": dict(fingerprints or {})}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return sha256(encoded).hexdigest()[:16]


@dataclass
class SavedExperiment:
    path: Path
    metadata: dict[str, object]
    metrics: dict[str, float]
    tables: dict[str, pd.DataFrame | pd.Series]

    @property
    def run_id(self) -> str:
        return str(self.metadata["run_id"])


def save_experiment(
    root: Path | str,
    config: object,
    *,
    metrics: Mapping[str, float],
    result: BacktestResult | None = None,
    tables: Mapping[str, pd.DataFrame | pd.Series] | None = None,
    fingerprints: Mapping[str, str] | None = None,
    warnings: list[str] | None = None,
) -> SavedExperiment:
    run_id = make_run_id(config, fingerprints)
    path = Path(root) / run_id
    path.mkdir(parents=True, exist_ok=True)
    versions = {}
    for package in ("alpha-lab", "numpy", "pandas", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": _jsonable(config),
        "fingerprints": dict(fingerprints or {}),
        "warnings": list(warnings or []),
        "environment": {"python": platform.python_version(), "packages": versions},
    }
    (path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    (path / "metrics.json").write_text(
        json.dumps(dict(metrics), indent=2, sort_keys=True), encoding="utf-8"
    )
    collected: dict[str, pd.DataFrame | pd.Series] = dict(tables or {})
    if result is not None:
        collected.update(
            equity_curve=result.equity_curve,
            returns=result.returns,
            positions=result.positions,
        )
        if result.trades is not None:
            collected["trades"] = result.trades
    for name, table in collected.items():
        frame = table.to_frame() if isinstance(table, pd.Series) else table
        try:
            frame.to_parquet(path / f"{name}.parquet")
        except (ImportError, ValueError):
            frame.to_csv(path / f"{name}.csv")
    return SavedExperiment(path, metadata, dict(metrics), collected)


def load_experiment(path: Path | str) -> SavedExperiment:
    root = Path(path)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    tables: dict[str, pd.DataFrame] = {}
    for item in sorted(root.glob("*.parquet")):
        tables[item.stem] = pd.read_parquet(item)
    for item in sorted(root.glob("*.csv")):
        if item.stem not in tables:
            tables[item.stem] = pd.read_csv(item, index_col=0, parse_dates=True)
    return SavedExperiment(root, metadata, metrics, tables)


__all__ = ["SavedExperiment", "load_experiment", "make_run_id", "save_experiment"]
