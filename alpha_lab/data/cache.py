"""Small fingerprinted Parquet cache for normalized panels."""

from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd


def cache_key(source: Path | str, **options: Any) -> str:
    path = Path(source)
    stat = path.stat()
    payload = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "options": options,
    }
    return sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:24]


def write_panel_cache(panel: pd.DataFrame, cache_dir: Path | str, key: str) -> Path:
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{key}.parquet"
    temporary = directory / f".{key}.{os.getpid()}.tmp.parquet"
    try:
        panel.to_parquet(temporary)
    except ImportError as exc:
        raise ImportError("Parquet caching requires `pip install alpha-lab[data]`") from exc
    temporary.replace(destination)
    return destination


def read_panel_cache(cache_dir: Path | str, key: str) -> pd.DataFrame | None:
    path = Path(cache_dir) / f"{key}.parquet"
    return pd.read_parquet(path) if path.exists() else None


__all__ = ["cache_key", "read_panel_cache", "write_panel_cache"]
