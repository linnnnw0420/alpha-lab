"""Save and reload a reproducible local momentum experiment."""

import tempfile

import pandas as pd

from alpha_lab import default_backtest_config, load_prices, momentum, top_k_long_only, zscore
from alpha_lab.data.contracts import fingerprint_frame
from alpha_lab.experiments import load_experiment, run_and_save_experiment


def main() -> None:
    file_name = "random_sampled_stocks_adj_close.csv"
    tickers = pd.read_csv(f"data/raw/{file_name}", nrows=0).columns[1:21].tolist()
    prices = load_prices(tickers, "2018-01-01", "2024-12-31", csv_file=file_name)
    factor = zscore(momentum(prices, 60), axis=1)
    weights = top_k_long_only(factor, k_pct=0.2)
    with tempfile.TemporaryDirectory() as directory:
        saved = run_and_save_experiment(
            weights,
            prices,
            default_backtest_config().with_updates(max_turnover=1.0),
            {"run_name": "momentum-60", "seed": 42},
            directory,
            fingerprints={"prices": fingerprint_frame(prices)},
        )
        loaded = load_experiment(saved.path)
        print(loaded.run_id, loaded.metrics["total_return"])


if __name__ == "__main__":
    main()
