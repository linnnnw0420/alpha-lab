# Price data contract

Alpha-Lab exchanges prices as a pandas `DataFrame` with dates on rows and tickers on columns.

## Canonical panel

- The index is a sorted, unique, timezone-naive, normalized `DatetimeIndex` named `date`.
- Ticker labels are unique, non-empty strings in deterministic requested order.
- Values are numeric. A finite positive value is tradable; missing, infinite, zero, and negative values are not.
- Start and end filters are inclusive.
- Missing values are preserved, forward-filled with an optional limit, or dropped according to `PriceDataContract`.
- `float64` is the default. `float32` must be requested explicitly.
- Adjusted/raw status is declared by the source or configuration; it is never guessed from values.

## Input shapes

Wide input contains one `date` column and one column per ticker. Long input requires `date`, `ticker`, and the requested field such as `close`. Duplicate dates in wide input and duplicate `(date, ticker)` keys in long input raise `DataContractError` by default.

## Calendar and missing prices

The observed normalized index is the default trading calendar. Forward filling fills values on that calendar; it does not invent weekends or exchange holidays. During a backtest, an invalid execution price prevents a trade. An existing holding is valued at its latest prior valid price.

## Sampling and bias

`sample_universe` applies data-quality filters and selects tickers once with NumPy's seeded generator. `load_sampled_prices` can preselect columns before a wide CSV read when no coverage filters require a full scan.

A fixed sample is reproducible, but sampling a current ticker list does not repair survivorship bias. Point-in-time membership and delisting data are outside the current framework.
