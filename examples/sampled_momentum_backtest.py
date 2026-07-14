"""Select one reproducible random universe, then run momentum."""

import pandas as pd

from alpha_lab import (
    default_backtest_config,
    load_sampled_prices,
    momentum,
    run_backtest,
    top_k_long_only,
    zscore,
)


def main() -> None:
    file_name = "random_sampled_stocks_adj_close.csv"
    candidates = pd.read_csv(f"data/raw/{file_name}", nrows=0).columns[1:].tolist()
    prices, selection = load_sampled_prices(
        candidates,
        "2018-01-01",
        "2024-12-31",
        sample_size=30,
        seed=42,
        csv_file=file_name,
    )
    factor = zscore(momentum(prices, lookback=60), axis=1)
    weights = top_k_long_only(factor, k_pct=0.2)
    result = run_backtest(
        weights,
        prices=prices,
        config=default_backtest_config().with_updates(max_turnover=1.0),
    )
    print(f"selection_id={selection.selection_id} tickers={list(selection.selected)}")
    print(result.summary_stats())


if __name__ == "__main__":
    main()
