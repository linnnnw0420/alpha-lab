"""
CSV data source: load prices from local CSV files.
CSV 数据源:从本地 CSV 文件加载价格数据.

支持的格式 / Expected formats:
- 宽格式 Wide: date, AAPL, MSFT, ... (推荐 / preferred)
- 长格式 Long: date, ticker, open, high, low, close, volume

文件搜索顺序 / File search order:
1. {field}.csv (如 close.csv)
2. prices_{field}.csv
3. prices.csv 或 ohlcv.csv (必须包含对应字段列)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from alpha_lab.config.paths import Paths
from alpha_lab.data.contracts import PriceDataContract, normalize_price_panel
from alpha_lab.exceptions import DataContractError
from alpha_lab.utils.logging import get_logger
from alpha_lab.utils.typing import PandasDataFrame, PandasTimestamp, PriceField, Ticker

logger = get_logger(__name__)


def load_csv_prices(
    tickers: list[Ticker],
    start_date: PandasTimestamp,
    end_date: PandasTimestamp,
    field: PriceField,
    paths: Paths,
    csv_file: str | None = None,
    contract: PriceDataContract | None = None,
    chunksize: int = 100_000,
) -> PandasDataFrame:
    """
    Load prices from csv files.
    从 CSV 文件加载价格数据.

    数据处理流程 / Data Processing Flow:
    1. 定位 CSV 文件 / Locate CSV file
    2. 读取并解析日期列 / Read and parse date column
    3. 转换为宽格式 / Convert to wide format
    4. 过滤日期范围和股票 / Filter date range and tickers

    Args / 参数:
        tickers: 要加载的股票代码列表
        start_date: 开始日期(包含)
        end_date: 结束日期(包含)
        field: 价格字段 'open'/'high'/'low'/'close'/'vwap'
        paths: Paths 配置对象
        csv_file: 指定CSV文件名，None则按默认规则搜索

    Returns / 返回:
        DataFrame: index=日期, columns=股票代码
    """
    # 定位 CSV 文件 / Locate CSV file
    if csv_file is not None:
        csv_path = paths.data_raw_dir / csv_file
        if not csv_path.exists():
            raise FileNotFoundError(f"Specified CSV file not found: {csv_path}")
    else:
        csv_path = _find_csv_file(field, paths)
    logger.debug(f"Reading {csv_path}")

    contract = contract or PriceDataContract()
    try:
        header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    except Exception as exc:
        raise DataContractError(f"Failed to inspect CSV {csv_path}: {exc}") from exc
    date_columns = [column for column in header if str(column).lower() == "date"]
    if not date_columns:
        raise DataContractError(f"CSV must have a date column; found {header}")
    date_column = date_columns[0]
    ticker_columns = [column for column in header if str(column).lower() == "ticker"]

    try:
        if ticker_columns:
            ticker_column = ticker_columns[0]
            field_columns = [
                column for column in header if str(column).lower() == str(field).lower()
            ]
            if not field_columns:
                raise DataContractError(f"Long CSV is missing field {field!r}")
            field_column = field_columns[0]
            parts: list[pd.DataFrame] = []
            for chunk in pd.read_csv(
                csv_path,
                usecols=[date_column, ticker_column, field_column],
                chunksize=chunksize,
            ):
                chunk[ticker_column] = chunk[ticker_column].astype(str)
                chunk = chunk[chunk[ticker_column].isin(tickers)]
                if not chunk.empty:
                    parts.append(chunk)
            raw = (
                pd.concat(parts, ignore_index=True)
                if parts
                else pd.DataFrame(columns=[date_column, ticker_column, field_column])
            )
            raw = raw.rename(
                columns={date_column: "date", ticker_column: "ticker", field_column: str(field)}
            )
            raw["date"] = pd.to_datetime(raw["date"], errors="raise")
            if raw.duplicated(["date", "ticker"]).any():
                examples = raw.loc[
                    raw.duplicated(["date", "ticker"], keep=False), ["date", "ticker"]
                ]
                raise DataContractError(
                    f"Duplicate (date, ticker) rows: {examples.head().to_dict('records')}"
                )
            wide = raw.pivot(index="date", columns="ticker", values=str(field))
        else:
            available = {str(column): column for column in header}
            selected = [available[ticker] for ticker in tickers if ticker in available]
            raw = pd.read_csv(csv_path, usecols=[date_column, *selected])
            raw = raw.rename(columns={date_column: "date"})
            raw["date"] = pd.to_datetime(raw["date"], errors="raise")
            wide = raw.set_index("date")
    except DataContractError:
        raise
    except Exception as exc:
        raise DataContractError(f"Failed to read CSV {csv_path}: {exc}") from exc

    return normalize_price_panel(
        wide,
        contract=contract,
        start_date=start_date,
        end_date=end_date,
        tickers=tickers,
    )


# -----------------------------------------------------------------------------
# Helpers: file location / 辅助函数:文件定位
# -----------------------------------------------------------------------------


def _find_csv_file(field: PriceField, paths: Paths) -> Path:
    """
    Locate CSV file for given field.
    根据指定字段定位 CSV 文件.

    搜索顺序 / Search order:
    1. {field}.csv (如 close.csv)
    2. prices_{field}.csv
    3. prices.csv
    4. ohlcv.csv
    """
    search_paths = [
        paths.data_raw_dir / f"{field}.csv",
        paths.data_raw_dir / f"prices_{field}.csv",
        paths.data_raw_dir / "prices.csv",
        paths.data_raw_dir / "ohlcv.csv",
    ]

    for path in search_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"No CSV file found for '{field}'. Searched:\n"
        + "\n".join(f" - {p}" for p in search_paths)
        + f"\n\nPlace data in: {paths.data_raw_dir}"
    )


# -----------------------------------------------------------------------------
# Helpers: format conversion / 辅助函数:格式转换
# -----------------------------------------------------------------------------


def _convert_to_wide_format(
    df: PandasDataFrame,
    field: PriceField,
) -> PandasDataFrame:
    """
    Convert to wide format (date x ticker).
    转换为宽格式(日期 x 股票代码).

    支持两种输入格式 / Supports two input formats:
    - 宽格式 Wide: 已经是 date x ticker 格式,直接使用
    - 长格式 Long: date, ticker, field 格式,需要 pivot
    """
    if "date" not in df.columns:
        raise ValueError("CSV must have 'date' column")

    # 已经是宽格式(没有 'ticker' 列)/ Already wide (no 'ticker' column)
    if "ticker" not in df.columns:
        df_wide = df.set_index("date")
        if not isinstance(df_wide.index, pd.DatetimeIndex):
            raise ValueError("'date' column not parseable as datetime")
        df_wide.index.name = "date"
        return df_wide

    # 长格式:需要 pivot / Long format: pivot
    if field not in df.columns:
        raise ValueError(f"Long format CSV missing field '{field}'")

    try:
        df_wide = df.pivot(index="date", columns="ticker", values=field)
        df_wide.columns.name = None
        return df_wide
    except Exception as e:
        raise ValueError(f"Failed to pivot: {e}") from e


# -----------------------------------------------------------------------------
# Helpers: filtering / 辅助函数:过滤
# -----------------------------------------------------------------------------


def _filter_date_range(
    df: PandasDataFrame,
    start_date: PandasTimestamp,
    end_date: PandasTimestamp,
) -> PandasDataFrame:
    """
    Filter to date range (inclusive).
    过滤到指定日期范围(包含边界).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have DatetimeIndex")

    df_dates = pd.DatetimeIndex(df.index).normalize()
    mask = (df_dates >= start_date) & (df_dates <= end_date)
    df_filtered = df.loc[mask].copy()

    if df_filtered.empty:
        logger.warning(
            f"No data in range {start_date.date()} to {end_date.date()}. "
            f"Available: {df.index.min().date()} to {df.index.max().date()}"
        )

    return df_filtered


def _filter_tickers(
    df: PandasDataFrame,
    tickers: list[Ticker],
) -> PandasDataFrame:
    """
    Filter to requested tickers (preserve order).
    过滤到请求的股票代码(保持顺序).
    """
    available = set(df.columns)
    requested = set(tickers)

    found = requested & available
    missing = requested - available

    if missing:
        logger.debug(f"Tickers not in CSV: {sorted(missing)[:10]}")

    found_ordered = [t for t in tickers if t in found]

    if not found_ordered:
        logger.warning("No requested tickers found in CSV")
        return pd.DataFrame(index=df.index, columns=tickers, dtype=float)

    return df[found_ordered].copy()


__all__ = ["load_csv_prices"]
