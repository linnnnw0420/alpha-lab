# alpha_lab/backtest/__init__.py
"""
Backtesting engine module.

Provides:
- run_backtest: main entry point for running backtest
- BacktestResult: results container
"""

from __future__ import annotations

from alpha_lab.backtest.engine import run_backtest
from alpha_lab.backtest.result import BacktestResult

__all__ = [
    "run_backtest",
    "BacktestResult",
]
