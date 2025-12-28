from __future__ import annotations
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional, Iterable

_ROOT_MARKERS: tuple[str, ...] = ("pyproject.toml", ".git", "README.md")

def _find_project_root(start: Path, markers: Iterable[str] = _ROOT_MARKERS) -> Path:
    start = start.resolve()
    for p in (start, *start.parents):
        for m in markers:
            if (p / m).exists():
                return p
    #fallback: repo not found; use start (but make it explicit)
    return start

@dataclass(frozen=True)
class Paths:
    project_root: Path
    pkg_dir: Path

    data_dir: Path
    artifacts_dir: Path
    cache_dir: Path
    logs_dir: Path

    #common subdirs
    data_raw_dir: Path
    data_processed_dir: Path
    reports_dir: Path

def get_paths(root: Path | str | None = None) -> Paths:
    '''
    Central place to resolve project path.

    Resolution order:
    1) explicit 'root'; argument
    2) env var ALPHA_LAB_ROOT
    3) search upwards from this file for pyproject.toml/.git/README.md
    '''
    if root is None:
        env_root = os.environ.get("ALPHA_LAB_ROOT")
        root = Path(env_root).expanduser() if env_root else None
    
    if root is None:
        # this file: alpha_lab/config/paths.py -> pkg dir: alpha_lab/
        pkg_dir = Path(__file__).resolve().parents[1]
        project_root = _find_project_root(pkg_dir)
    else:
        project_root = root.expanduser().resolve()
        pkg_dir = project_root / "alpha_lab" # adjust if your layout changes
    
    #allow overrides for big folders
    data_dir = Path(os.environ.get("ALPHA_LAB_DATA_DIR", project_root / "data")).expanduser().resolve()
    artifacts_dir = Path(os.environ.get("ALPHA_LAB_ARTIFACT_DIR", project_root / "artifacts")).expanduser().resolve()
    cache_dir = Path(os.environ.get("ALPHA_LAB_CACHE_DIR", project_root / "cache")).expanduser().resolve()
    logs_dir = Path(os.environ.get("ALPHA_LAB_LOGS_DIR", project_root / "logs")).expanduser().resolve()
    
    data_raw_dir = data_dir / "raw"
    data_processed_dir = data_dir / "processed"
    reports_dir = project_root / "reports"

    return Paths(
        project_root=project_root,
        pkg_dir=pkg_dir,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        cache_dir=cache_dir,
        logs_dir=logs_dir,
        data_raw_dir=data_raw_dir,
        data_processed_dir=data_processed_dir,
        reports_dir=reports_dir,
    )

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path