"""Leakage-safe Ridge walk-forward predictions on local prices."""

import pandas as pd

from alpha_lab import load_prices, momentum, zscore
from alpha_lab.ml import WalkForwardSplit, build_supervised_dataset, run_walk_forward


def main() -> None:
    file_name = "random_sampled_stocks_adj_close.csv"
    tickers = pd.read_csv(f"data/raw/{file_name}", nrows=0).columns[1:21].tolist()
    prices = load_prices(tickers, "2018-01-01", "2024-12-31", csv_file=file_name)
    features = {
        "momentum_20": zscore(momentum(prices, 20), axis=1),
        "momentum_60": zscore(momentum(prices, 60), axis=1),
    }
    dataset = build_supervised_dataset(features, prices, horizon=5, delay=1)
    splitter = WalkForwardSplit(min_train_dates=252, test_dates=63, gap_dates=6)
    result = run_walk_forward(dataset, splitter)
    print(result.window_metrics.tail())
    print(f"predictions={result.predictions.notna().sum()}")


if __name__ == "__main__":
    main()
