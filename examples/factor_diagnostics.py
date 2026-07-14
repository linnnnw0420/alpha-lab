"""Generate a compact momentum factor tear sheet."""

import pandas as pd

from alpha_lab import load_prices, momentum, zscore
from alpha_lab.metrics.factor_diagnostics import compute_forward_returns, generate_factor_tear_sheet


def main() -> None:
    file_name = "random_sampled_stocks_adj_close.csv"
    tickers = pd.read_csv(f"data/raw/{file_name}", nrows=0).columns[1:41].tolist()
    prices = load_prices(tickers, "2018-01-01", "2024-12-31", csv_file=file_name)
    factor = zscore(momentum(prices, lookback=60), axis=1)
    future = compute_forward_returns(prices, horizon=20, delay=1)
    tear_sheet = generate_factor_tear_sheet(factor, future, quantiles=5, min_obs=10)
    print(tear_sheet.ic_stats)
    print(tear_sheet.quantile_returns.mean())


if __name__ == "__main__":
    main()
