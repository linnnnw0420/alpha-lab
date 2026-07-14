"""Optional scikit-learn estimator factories."""

from __future__ import annotations

from typing import Literal


def make_linear_model(
    kind: Literal["ridge", "lasso", "elastic_net"] = "ridge",
    *,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
):
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import ElasticNet, Lasso, Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError("ML support requires `pip install alpha-lab[ml]`") from exc
    estimators = {
        "ridge": Ridge(alpha=alpha),
        "lasso": Lasso(alpha=alpha),
        "elastic_net": ElasticNet(alpha=alpha, l1_ratio=l1_ratio),
    }
    if kind not in estimators:
        raise ValueError(f"Unknown linear model: {kind!r}")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimators[kind]),
        ]
    )


__all__ = ["make_linear_model"]
