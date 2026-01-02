"""
Metrics summary and reporting.

Key functions:
- generate_metrics_summary: compile all metrics into dict/DataFrame
- print_metrics_summary: pretty-print metrics report
"""

from __future__ import annotations

import pandas as pd

from alpha_lab.backtest.engine import BacktestResult
from alpha_lab.metrics.performance import compute_performance_metrics
from alpha_lab.metrics.drawdown import compute_drawdown_metrics
from alpha_lab.utils.logging import get_logger
from alpha_lab.utils.typing import PandasSeries, PandasDataFrame, RebalanceFreq

logger = get_logger(__name__)

def generate_metrics_summary(
    equity_curve: PandasSeries,
    returns: PandasSeries | None = None,
    freq: RebalanceFreq = "D",
    risk_free_rate: float = 0.0,
    as_dataframe: bool = False,
) -> dict[str, float] | PandasDataFrame:
    """
    Generate comprehensive metrics summary.

    Args:
        equity_curve: Series (date -> equity value)
        returns: Series of returns (if None, computed from equity_curve)
        freq: data frequency ('D', 'W', 'M')
        risk_free_rate: annual risk-free rate
        as_dataframe: if True, return as DataFrame; else dict

    Returns:
        Dict or DataFrame with all metrics
    """
    if equity_curve.empty:
        logger.warning("Empty equity curve")
        return pd.DataFrame() if as_dataframe else {}
    
    logger.info("Generating metrics summary")

    # Compute returns if not provided
    if returns is None:
        returns = equity_curve.pct_change().fillna(0.0)

    # Performance metrics
    perf_metrics = compute_performance_metrics(
        equity_curve=equity_curve,
        returns=returns,
        freq=freq,
        risk_free_rate=risk_free_rate,
    )

    # Drawdown metrics
    dd_metrics = compute_drawdown_metrics(equity_curve)

    # Combine all metrics
    all_metrics = {**perf_metrics, **dd_metrics}

    # Return as DataFrame if requested
    if as_dataframe:
        df = pd.DataFrame.from_dict(all_metrics, orient="index", columns=["Value"])
        df.index.name = "Metric"
        return df
    
    return all_metrics

def generate_backtest_summary(
    result: BacktestResult,
    as_dataframe: bool = False,
) -> dict[str, float] | PandasDataFrame:
    """
    Generate metrics summary from BacktestResult object.

    Args:
        result: BacktestResult from run_backtest()
        as_dataframe: if True, return as DataFrame

    Returns:
        Dict or DataFrame with all metrics
    """
    # NOTE: result.returns 是 equity_curve.pct_change(),是日频数据
    # 年化系数应该用 "D"(252),不是 rebalance_freq
    return generate_metrics_summary(
        equity_curve=result.equity_curve,
        returns=result.returns,
        freq="D",  # returns 是日频,不是 rebalance 频率
        risk_free_rate=0.0,  # v0: assume 0, can be added to config in v1
        as_dataframe=as_dataframe,
    )

def print_metrics_summary(
    metrics: dict[str, float] | PandasDataFrame,
    title: str = "Performance Summary",
) -> None:
    """
    Pretty-print metrics summary.

    Args:
        metrics: dict or DataFrame of metrics
        title: report title
    """
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

    if isinstance(metrics, pd.DataFrame):
        metrics_dict = metrics.to_dict()["Value"]
    else:
        metrics_dict = metrics
    
    if not metrics_dict:
        print("  No metrics available")
        print("=" * 60 + "\n")
        return

    # Group metrics by category
    performance_keys = [
        "total_return", "cagr", "annualized_vol", "sharpe_ratio",
        "win_rate", "best_day", "worst_day"
    ]
    drawdown_keys = [
        "max_drawdown", "avg_drawdown", "current_drawdown", "max_drawdown_duration"
    ]
    other_keys = ["n_periods"]
    # Print performance metrics
    print("\n  Performance Metrics:")
    print("  " + "-" * 56)
    for key in performance_keys:
        if key in metrics_dict:
            value = metrics_dict[key]
            label = _format_metric_label(key)
            formatted_value = _format_metric_value(key, value)
            print(f"    {label:<30} {formatted_value:>20}")

    # Print drawdown metrics
    print("\n  Drawdown Metrics:")
    print("  " + "-" * 56)
    for key in drawdown_keys:
        if key in metrics_dict:
            value = metrics_dict[key]
            label = _format_metric_label(key)
            formatted_value = _format_metric_value(key, value)
            print(f"    {label:<30} {formatted_value:>20}")
    
    # Print other metrics
    if any(key in metrics_dict for key in other_keys):
        print("\n  Other:")
        print("  " + "-" * 56)
        for key in other_keys:
            if key in metrics_dict:
                value = metrics_dict[key]
                label = _format_metric_label(key)
                formatted_value = _format_metric_value(key, value)
                print(f"    {label:<30} {formatted_value:>20}")

    print("\n" + "=" * 60 + "\n")

def _format_metric_label(key: str) -> str:
    """Format metric key into readable label."""
    label_map = {
        "total_return": "Total Return",
        "cagr": "CAGR",
        "annualized_vol": "Annualized Volatility",
        "sharpe_ratio": "Sharpe Ratio",
        "win_rate": "Win Rate",
        "best_day": "Best Day",
        "worst_day": "Worst Day",
        "max_drawdown": "Max Drawdown",
        "avg_drawdown": "Avg Drawdown",
        "current_drawdown": "Current Drawdown",
        "max_drawdown_duration": "Max DD Duration (days)",
        "n_periods": "Number of Periods",
    }
    return label_map.get(key, key.replace("_", " ").title())

def _format_metric_value(key: str, value: float) -> str:
    """Format metric value for display."""
    # Percentage metrics
    if key in ["total_return", "cagr", "annualized_vol", "win_rate",
               "best_day", "worst_day", "max_drawdown", "avg_drawdown", "current_drawdown"]:
        return f"{value:>8.2%}"
    # Ratio metrics
    elif key in ["sharpe_ratio", "sortino_ratio", "calmar_ratio"]:
        return f"{value:>8.2f}"
    # Integer metrics
    elif key in ["n_periods", "max_drawdown_duration"]:
        return f"{int(value):>8,d}"
    # Default: 2 decimal places
    else:
        return f"{value:>8.2f}"
    
def compare_strategies(
    results: dict[str, BacktestResult],
    metrics_to_compare: list[str] | None = None,
) -> PandasDataFrame:
    """
    Compare metrics across multiple strategies.

    Args:
        results: dict of strategy_name -> BacktestResult
        metrics_to_compare: list of metric names to include (None = all)

    Returns:
        DataFrame with strategies as columns, metrics as rows

    Note:
        v0 implementation - basic comparison table.
    """
    if not results:
        logger.warning("No results to compare")
        return pd.DataFrame()

    logger.info(f"Comparing {len(results)} strategies")

    # Generate metrics for each strategy
    all_metrics = {}
    for name, result in results.items():
        metrics = generate_backtest_summary(result, as_dataframe=False)
        all_metrics[name] = metrics

    # Convert to DataFrame
    comparison_df = pd.DataFrame(all_metrics)

    # Filter metrics if requested
    if metrics_to_compare is not None:
        available = [m for m in metrics_to_compare if m in comparison_df.index]
        comparison_df = comparison_df.loc[available]

    return comparison_df

__all__ = [
    "generate_metrics_summary",
    "generate_backtest_summary",
    "print_metrics_summary",
    "compare_strategies",
]