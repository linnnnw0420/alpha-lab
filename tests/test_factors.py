import numpy as np
import pandas as pd
import pytest

from alpha_lab.factors.metadata import FactorDefinition, FactorResult
from alpha_lab.factors.transform import (
    neutralize,
    neutralize_industry,
    rank_normalize,
    winsorize,
    zscore,
)
from alpha_lab.metrics.factor_diagnostics import compute_forward_returns, generate_factor_tear_sheet


def test_transforms_handle_constant_and_nan_rows() -> None:
    frame = pd.DataFrame([[1.0, 1.0, np.nan], [1.0, 3.0, 9.0]], columns=list("ABC"))
    standardized = zscore(frame, axis=1)
    assert standardized.loc[0, ["A", "B"]].eq(0.0).all()
    assert np.isnan(standardized.loc[0, "C"])
    assert rank_normalize(frame, axis=1).loc[1, "C"] == 1.0
    clipped = winsorize(frame, lower=0.1, upper=0.9, axis=1)
    assert clipped.loc[1, "C"] < 9.0


def test_industry_neutralization_removes_group_means() -> None:
    factor = pd.DataFrame(
        [[1.0, 2.0, 3.0, 10.0, 11.0, 12.0]],
        index=pd.to_datetime(["2024-01-02"]),
        columns=list("ABCDEF"),
    )
    labels = {ticker: "tech" if ticker < "D" else "finance" for ticker in factor.columns}
    result = neutralize_industry(factor, labels)
    assert result.loc[:, list("ABC")].mean(axis=1).iloc[0] == pytest.approx(0.0, abs=1e-12)
    assert result.loc[:, list("DEF")].mean(axis=1).iloc[0] == pytest.approx(0.0, abs=1e-12)


def test_continuous_neutralization_removes_linear_exposure() -> None:
    exposure = pd.DataFrame({"size": np.arange(6.0)}, index=list("ABCDEF"))
    factor = pd.DataFrame([2.0 + 3.0 * exposure["size"]], index=pd.to_datetime(["2024-01-02"]))
    result = neutralize(factor, exposures=exposure)
    assert np.nanmax(np.abs(result.to_numpy())) < 1e-10


def test_factor_metadata_is_stable(price_panel: pd.DataFrame) -> None:
    definition = FactorDefinition("momentum", parameters={"lookback": 3}, lookback=3)
    result = FactorResult(definition, price_panel)
    assert len(definition.definition_id) == 16
    assert result.to_metadata()["shape"] == list(price_panel.shape)


def test_factor_tear_sheet_has_spread(price_panel: pd.DataFrame) -> None:
    factor = price_panel.pct_change()
    forward = compute_forward_returns(price_panel, horizon=1)
    tear = generate_factor_tear_sheet(factor, forward, quantiles=2, min_obs=2)
    assert {1, 2} <= set(tear.quantile_returns.columns)
    assert len(tear.coverage) == len(factor)
    assert tear.turnover.index.equals(factor.index)
