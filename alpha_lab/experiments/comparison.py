"""Comparison tables for saved and in-memory experiments."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from alpha_lab.experiments.artifacts import SavedExperiment


def compare_experiments(
    experiments: Mapping[str, SavedExperiment], metrics: list[str] | None = None
) -> pd.DataFrame:
    table = pd.DataFrame({name: run.metrics for name, run in experiments.items()})
    if metrics is not None:
        table = table.reindex(metrics)
    table.index.name = "metric"
    return table


__all__ = ["compare_experiments"]
