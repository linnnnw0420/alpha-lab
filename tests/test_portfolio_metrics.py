import numpy as np
import pandas as pd
import pytest

from alpha_lab.metrics.drawdown import compute_drawdown_metrics, compute_max_drawdown
from alpha_lab.metrics.performance import compute_annualized_vol, compute_cagr, compute_sharpe_ratio
from alpha_lab.portfolio.weighting import (
    apply_weight_constraints,
    proportional_weights,
    top_k_long_only,
    top_k_long_short,
)


def test_portfolio_weight_invariants() -> None:
    factor = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], columns=list("ABCD"))
    long_only = top_k_long_only(factor, k_pct=0.5)
    long_short = top_k_long_short(factor, k_pct=0.25)
    proportional = proportional_weights(factor, target_leverage=1.0)
    assert long_only.sum(axis=1).iloc[0] == pytest.approx(1.0)
    assert long_short.sum(axis=1).iloc[0] == pytest.approx(0.0)
    assert long_short.abs().sum(axis=1).iloc[0] == pytest.approx(2.0)
    assert proportional.sum(axis=1).iloc[0] == pytest.approx(1.0)


def test_turnover_limit_is_respected() -> None:
    weights = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], columns=["A", "B"])
    constrained = apply_weight_constraints(weights, max_turnover=0.4)
    turnover = constrained.diff().abs().sum(axis=1).iloc[1]
    assert turnover <= 0.4 + 1e-12


def test_known_performance_and_drawdown() -> None:
    equity = pd.Series([100.0, 120.0, 90.0, 135.0])
    returns = equity.pct_change().fillna(0.0)
    assert compute_max_drawdown(equity) == pytest.approx(0.25)
    assert compute_drawdown_metrics(equity)["max_drawdown_duration"] == 1
    assert compute_cagr(equity) > 0
    assert compute_annualized_vol(returns) > 0
    assert np.isfinite(compute_sharpe_ratio(returns))
