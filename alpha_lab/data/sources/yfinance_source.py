"""Optional yfinance source with deterministic local caching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from alpha_lab.data.contracts import PriceDataContract, normalize_price_panel


@dataclass
class YFinanceDataSource:
    cache_dir: Path | str
    batch_size: int = 50
    auto_adjust: bool = True

    def load_prices(
        self,
        tickers: list[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        field: str = "close",
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("yfinance support requires `pip install alpha-lab[market]`") from exc

        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / self._cache_name(tickers, start_date, end_date, field)
        if cache_path.exists():
            return normalize_price_panel(pd.read_parquet(cache_path), tickers=tickers)

        panels: list[pd.DataFrame] = []
        field_name = field.replace("_", " ").title()
        for offset in range(0, len(tickers), self.batch_size):
            batch = tickers[offset : offset + self.batch_size]
            raw = yf.download(
                batch,
                start=start_date.strftime("%Y-%m-%d"),
                end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                auto_adjust=self.auto_adjust,
                progress=False,
                group_by="column",
                threads=True,
            )
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                if field_name not in raw.columns.get_level_values(0):
                    continue
                panel = raw[field_name]
            else:
                if field_name not in raw.columns:
                    continue
                panel = raw[[field_name]].rename(columns={field_name: batch[0]})
            panels.append(panel)

        combined = pd.concat(panels, axis=1) if panels else pd.DataFrame()
        result = normalize_price_panel(
            combined,
            contract=PriceDataContract(adjusted=self.auto_adjust),
            start_date=start_date,
            end_date=end_date,
            tickers=tickers,
        )
        try:
            result.to_parquet(cache_path)
        except ImportError as exc:
            raise ImportError(
                "yfinance caching requires `pip install alpha-lab[data,market]`"
            ) from exc
        return result

    @staticmethod
    def _cache_name(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp, field: str) -> str:
        from hashlib import sha256

        payload = "|".join([*tickers, str(start.date()), str(end.date()), field])
        return f"yfinance-{sha256(payload.encode()).hexdigest()[:20]}.parquet"


__all__ = ["YFinanceDataSource"]
