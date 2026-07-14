"""Run a local momentum backtest without opening Jupyter."""

import pandas as pd

from alpha_lab import (
    default_backtest_config,
    generate_backtest_summary,
    load_prices,
    momentum,
    run_backtest,
    top_k_long_only,
    zscore,
)


def main() -> None:
    file_name = "random_sampled_stocks_adj_close.csv"
    tickers = pd.read_csv(f"data/raw/{file_name}", nrows=0).columns[1:41].tolist()
    prices = load_prices(tickers, "2018-01-01", "2024-12-31", csv_file=file_name)
    scores = zscore(momentum(prices, lookback=60, lag=1), axis=1)
    weights = top_k_long_only(scores, k_pct=0.2, buffer_pct=0.05)
    config = default_backtest_config().with_updates(max_turnover=1.0)
    result = run_backtest(weights, prices=prices, config=config)
    summary = generate_backtest_summary(result)
    print({key: round(summary[key], 4) for key in ("total_return", "sharpe_ratio", "max_drawdown")})


if __name__ == "__main__":
    main()
