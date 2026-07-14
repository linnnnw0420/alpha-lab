import sys
import types

import pandas as pd
import pytest

from alpha_lab.config.paths import get_paths
from alpha_lab.data.cache import cache_key, read_panel_cache, write_panel_cache
from alpha_lab.data.contracts import PriceDataContract, fingerprint_frame, normalize_price_panel
from alpha_lab.data.loader import load_prices, load_sampled_prices
from alpha_lab.data.sampling import sample_universe
from alpha_lab.data.sources.csv_source import load_csv_prices
from alpha_lab.data.sources.parquet_source import ParquetDataSource
from alpha_lab.data.sources.yfinance_source import YFinanceDataSource
from alpha_lab.exceptions import ConfigurationError, DataContractError


def test_normalize_price_panel_sorts_filters_and_fills() -> None:
    panel = pd.DataFrame(
        {"B": [2.0, None, 4.0], "A": [1.0, 2.0, 3.0]},
        index=["2024-01-03", "2024-01-02", "2024-01-04"],
    )
    result = normalize_price_panel(
        panel,
        contract=PriceDataContract(missing="ffill", forward_fill_limit=1),
        start_date="2024-01-02",
        tickers=["A", "B", "MISSING"],
    )
    assert result.index.is_monotonic_increasing
    assert result.columns.tolist() == ["A", "B", "MISSING"]
    assert result.loc["2024-01-03", "B"] == 2.0


def test_normalize_rejects_duplicate_dates() -> None:
    panel = pd.DataFrame({"A": [1.0, 2.0]}, index=["2024-01-02", "2024-01-02"])
    with pytest.raises(DataContractError, match="Duplicate price dates"):
        normalize_price_panel(panel)


def test_fingerprint_changes_with_values(price_panel: pd.DataFrame) -> None:
    changed = price_panel.copy()
    changed.iloc[0, 0] += 1
    assert fingerprint_frame(price_panel) != fingerprint_frame(changed)


def test_fixed_universe_sample_is_reproducible(price_panel: pd.DataFrame) -> None:
    first, first_record = sample_universe(price_panel, 2, seed=7)
    second, second_record = sample_universe(price_panel, 2, seed=7)
    assert first.columns.tolist() == second.columns.tolist()
    assert first_record.selection_id == second_record.selection_id
    assert first_record.selected == tuple(first.columns)


def test_sample_rejects_oversized_request(price_panel: pd.DataFrame) -> None:
    with pytest.raises(ConfigurationError, match="exceeds"):
        sample_universe(price_panel, 99)


def test_wide_csv_projects_requested_columns(tmp_path) -> None:
    raw = pd.DataFrame(
        {"Date": ["2024-01-02", "2024-01-03"], "A": [1, 2], "B": [3, 4], "C": [5, 6]}
    )
    raw.to_csv(tmp_path / "wide.csv", index=False)
    paths = get_paths(tmp_path)
    paths.data_raw_dir.mkdir(parents=True)
    raw.to_csv(paths.data_raw_dir / "wide.csv", index=False)
    result = load_csv_prices(
        ["C", "A"],
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
        "close",
        paths,
        "wide.csv",
    )
    assert result.columns.tolist() == ["C", "A"]


def test_long_csv_and_duplicate_detection(tmp_path) -> None:
    paths = get_paths(tmp_path)
    paths.data_raw_dir.mkdir(parents=True)
    raw = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02"],
            "ticker": ["A", "B"],
            "close": [10.0, 20.0],
        }
    )
    raw.to_csv(paths.data_raw_dir / "long.csv", index=False)
    result = load_csv_prices(
        ["A"], pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02"), "close", paths, "long.csv"
    )
    assert result.loc[pd.Timestamp("2024-01-02"), "A"] == 10.0
    duplicate = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    duplicate.to_csv(paths.data_raw_dir / "duplicate.csv", index=False)
    with pytest.raises(DataContractError, match="Duplicate"):
        load_csv_prices(
            ["A"],
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-02"),
            "close",
            paths,
            "duplicate.csv",
        )


def test_parquet_source_round_trip(tmp_path, price_panel: pd.DataFrame) -> None:
    path = tmp_path / "prices.parquet"
    price_panel.to_parquet(path)
    result = ParquetDataSource(path).load_prices(
        ["B", "A"], price_panel.index[2], price_panel.index[5]
    )
    assert result.columns.tolist() == ["B", "A"]
    assert result.index[[0, -1]].tolist() == [price_panel.index[2], price_panel.index[5]]


def test_load_sampled_prices_reads_fixed_subset(tmp_path, monkeypatch) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03"],
            "A": [1.0, 2.0],
            "B": [2.0, 3.0],
            "C": [3.0, 4.0],
            "D": [4.0, 5.0],
        }
    )
    raw.to_csv(raw_dir / "prices.csv", index=False)
    monkeypatch.setenv("ALPHA_LAB_ROOT", str(tmp_path))
    prices, record = load_sampled_prices(
        ["A", "B", "C", "D"],
        "2024-01-02",
        "2024-01-03",
        sample_size=2,
        seed=9,
        csv_file="prices.csv",
    )
    assert tuple(prices.columns) == record.selected
    assert record.candidates == ("A", "B", "C", "D")


def test_panel_cache_round_trip(tmp_path, price_panel: pd.DataFrame) -> None:
    source = tmp_path / "source.csv"
    source.write_text("date,A\n2024-01-02,1\n", encoding="utf-8")
    key = cache_key(source, field="close", tickers=["A"])
    destination = write_panel_cache(price_panel, tmp_path / "cache", key)
    assert destination.exists()
    pd.testing.assert_frame_equal(
        read_panel_cache(tmp_path / "cache", key), price_panel, check_freq=False
    )


def test_csv_loader_normalized_cache(tmp_path, monkeypatch) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    pd.DataFrame({"date": ["2024-01-02", "2024-01-03"], "A": [10.0, 11.0]}).to_csv(
        raw_dir / "prices.csv", index=False
    )
    monkeypatch.setenv("ALPHA_LAB_ROOT", str(tmp_path))
    first = load_prices(["A"], "2024-01-02", "2024-01-03", csv_file="prices.csv", use_cache=True)
    second = load_prices(["A"], "2024-01-02", "2024-01-03", csv_file="prices.csv", use_cache=True)
    pd.testing.assert_frame_equal(first, second, check_freq=False)
    assert len(list((tmp_path / "cache" / "prices").glob("*.parquet"))) == 1


def test_yfinance_source_batches_and_caches(tmp_path, monkeypatch) -> None:
    calls = []

    def download(batch, **kwargs):
        calls.append((tuple(batch), kwargs))
        index = pd.date_range("2024-01-02", periods=2, freq="B")
        columns = pd.MultiIndex.from_product([["Close"], batch])
        return pd.DataFrame(
            [[10.0] * len(batch), [11.0] * len(batch)], index=index, columns=columns
        )

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=download))
    source = YFinanceDataSource(tmp_path, batch_size=10)
    first = source.load_prices(["A", "B"], pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"))
    second = source.load_prices(["A", "B"], pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"))
    assert len(calls) == 1
    pd.testing.assert_frame_equal(first, second)
