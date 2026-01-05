"""
Backtesting engine: convert weights + prices -> equity curve.
回测引擎:将目标权重 + 价格数据  ->  转换为净值曲线

Key components / 核心组件:
- run_backtest: main backtest loop / 主回测入口函数
- BacktestResult: results container with equity curve and positions / 回测结果容器
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
import numpy as np

from alpha_lab.config.backtest import BacktestConfig
from alpha_lab.config.universe import UniverseConfig
from alpha_lab.data.loader import load_prices, load_returns
from alpha_lab.portfolio.rebalance import generate_rebalance_schedule
from alpha_lab.utils.dates import parse_date
from alpha_lab.utils.logging import get_logger
from alpha_lab.utils.typing import (
    PandasDataFrame,
    PandasSeries,
    PandasDatetimeIndex,
    Ticker,
)

logger = get_logger(__name__)


# =============================================================================
# BacktestResult: 回测结果数据容器
# =============================================================================

@dataclass
class BacktestResult:
    """
    Container for backtest results.
    回测结果的数据容器类.
    
    Attributes / 属性:
        equity_curve: Series (date -> equity value) / 净值曲线,以日期为索引
        positions: DataFrame (date x asset), daily position weights / 每日持仓权重矩阵
        returns: Series (date -> daily return) / 日收益率序列
        config: BacktestConfig used for this run / 本次回测使用的配置对象
        trades: DataFrame of trades (optional) / 交易记录(可选)
    """
    equity_curve: PandasSeries
    positions: PandasDataFrame
    returns: PandasSeries
    config: BacktestConfig
    trades: PandasDataFrame | None = None

    def total_return(self) -> float:
        """
        Total return over backtest period.
        计算回测期间的总收益率.
        
        Formula / 公式: (期末净值 / 期初净值) - 1
        """
        if self.equity_curve.empty:
            return 0.0
        return (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1.0

    def save_trades(self, path: str | None = None) -> str | None:
        """
        Save trades DataFrame to CSV.
        将交易记录保存为 CSV 文件.

        Args / 参数:
            path: output path (default: artifacts/trades.csv)
                  输出路径(默认: artifacts/trades.csv)

        Returns / 返回:
            Path where trades were saved, or None if no trades
            保存的文件路径,如果没有交易记录则返回 None
        """
        if self.trades is None or self.trades.empty:
            logger.warning("No trades to save")
            return None
        
        from pathlib import Path
        from alpha_lab.config import get_paths
        
        if path is None:
            paths = get_paths()
            artifacts_dir = paths.root / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            path = str(artifacts_dir / "trades.csv")
        
        self.trades.to_csv(path, index=False)
        logger.info(f"Saved {len(self.trades)} trades to {path}")
        return path

    def summary_stats(self) -> dict[str, float]:
        """
        Quick summary statistics.
        快速计算汇总统计指标.
        
        Returns / 返回:
            dict containing: total_return(总收益), annualized_return(年化收益),
            annualized_vol(年化波动), sharpe_ratio(夏普比率), n_days(交易天数)
        """
        if self.returns.empty:
            return {}
        
        from alpha_lab.utils.dates import annualization_factor

        # NOTE: returns 是 equity_curve.pct_change(),是日频数据
        # 年化系数应该用 "D"(252),不是 rebalance_freq
        # returns is daily, so use "D"(252) for annualization
        ann_factor = annualization_factor("D")

        ret_mean = self.returns.mean()  # 平均日收益 / average daily return
        ret_std = self.returns.std()    # 日收益标准差 / daily return std

        total_ret = self.total_return()
        n_days = len(self.returns)

        # Simple annualization / 简单年化
        # 年化收益 = 日均收益 × 252 / annualized return = daily mean × 252
        ann_return = ret_mean * ann_factor
        # 年化波动 = 日波动 × sqrt(252) / annualized vol = daily std × sqrt(252)
        ann_vol = ret_std * np.sqrt(ann_factor)
        # 夏普比率 = 年化收益 / 年化波动 / sharpe = ann_return / ann_vol
        sharpe = ann_return / ann_vol if ann_vol > 1e-12 else 0.0

        return {
            "total_return": total_ret,
            "annualized_return": ann_return,
            "annualized_vol": ann_vol,
            "sharpe_ratio": sharpe,
            "n_days": n_days,
        }


# =============================================================================
# run_backtest: 主回测入口函数 / Main backtest entry point
# =============================================================================

def run_backtest(
    weights: PandasDataFrame,
    prices: PandasDataFrame | None = None,
    config: BacktestConfig | None = None,
    universe: UniverseConfig | list[Ticker] | tuple[Ticker, ...] | None = None,
    execution_field: Literal["open", "close"] = "close",
    record_trades: bool = True,
    verbose: bool = False,
) -> BacktestResult:
    """
    Run backtest: convert target weights -> equity curve.
    执行回测:将目标权重转换为净值曲线.
    
    整体流程 / Overall workflow:
    1. 加载配置 / Load config
    2. 加载价格数据 / Load price data  
    3. 生成调仓日期 / Generate rebalance schedule
    4. 模拟组合运行 / Simulate portfolio
    5. 返回结果 / Return results

    Args / 参数:
        weights: DataFrame (date x asset), target weights on rebalance dates
                 目标权重矩阵,行=调仓日期,列=资产,值=目标权重
        prices: pre-loaded prices (if None, will load from config)
                预加载的价格数据,如果为 None 则从配置加载
        config: BacktestConfig (if None, uses default)
                回测配置,如果为 None 则使用默认配置
        universe: required if prices is None
                  股票池,如果 prices 为 None 则必须提供
        execution_field: 'open' or 'close' for execution price
                        执行价格:'open'(开盘价) 或 'close'(收盘价)
        record_trades: whether to record individual trades (default True)
                       是否记录每笔交易(默认 True)
        verbose: whether to print detailed rebalance info (default False)
                 是否打印详细的调仓信息(默认 False)

    Returns / 返回:
        BacktestResult with equity curve and positions
        包含净值曲线和持仓的回测结果对象

    Example / 示例:
        >>> result = run_backtest(
        ...     weights=portfolio_weights,
        ...     config=cfg,
        ...     universe=["AAPL", "MSFT", "GOOGL"],
        ... )
        >>> print(result.total_return())
    """
    # Step 1: Setup config / 第一步:设置配置
    if config is None:
        from alpha_lab.config.backtest import default_backtest_config
        config = default_backtest_config()
        logger.info("Using default backtest config")
    
    logger.info(
        f"Running backtest: {config.start_date} to {config.end_date}, "
        f"freq={config.rebalance_freq}, initial_cash={config.initial_cash:,.0f}"
    )

    # Step 2: Load prices if not provided / 第二步:加载价格数据
    if prices is None:
        if universe is None:
            raise ValueError("Must provide universe if prices is None")
        
        logger.debug(f"Loading prices: field={execution_field}")
        prices = load_prices(
            universe=universe,
            start_date=config.start_date,
            end_date=config.end_date,
            field=execution_field,
            align_dates=True,
        )
    
    if prices.empty:
        raise ValueError("Price data is empty")

    # Validate weights / 验证权重数据
    if weights.empty:
        raise ValueError("Weights data is empty")
    
    # Step 3: Align weights to rebalance dates / 第三步:对齐权重到调仓日期
    trading_calendar = prices.index  # 交易日历 = 价格数据的索引
    
    # 生成调仓日期(如月末最后一个交易日)
    # Generate rebalance dates (e.g., last trading day of each month)
    rebalance_dates = generate_rebalance_schedule(
        trading_calendar=trading_calendar,
        start_date=config.start_date,
        end_date=config.end_date,
        freq=config.rebalance_freq,
    )

    # 过滤权重,只保留调仓日期 / Filter weights to rebalance dates only
    weights_aligned = _align_weights_to_schedule(weights, rebalance_dates)

    # Apply execution delay (signal day -> execution day)
    weights_exec = _shift_weights_by_trading_day(
        weights=weights_aligned,
        trading_calendar=trading_calendar,
        delay_days=int(config.execution_delay_days),
    )

    # Step 4: Run simulation / 第四步:运行模拟
    equity_curve, positions_daily, trades_df = _simulate_portfolio(
        weights=weights_exec    ,
        prices=prices,
        config=config,
        record_trades=record_trades,
        verbose=verbose,
    )

    # Step 5: Compute returns / 第五步:计算收益率
    # 日收益率 = 净值的日百分比变化 / daily return = pct_change of equity curve
    returns = equity_curve.pct_change().fillna(0.0)

    logger.info(
        f"Backtest complete: {len(equity_curve)} days, "
        f"final equity={equity_curve.iloc[-1]:,.0f}"
    )

    return BacktestResult(
        equity_curve=equity_curve,
        positions=positions_daily,
        returns=returns,
        config=config,
        trades=trades_df,
    )


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _shift_weights_by_trading_day(
    weights: PandasDataFrame, 
    trading_calendar: PandasDatetimeIndex,
    delay_days: int,
) -> PandasDataFrame:
    """
    Shift weights by N trading days (signal day -> execution day).
    将权重表按交易日历后平移 N 个交易日 (信号日 -> 成交日)

    Notes:
    - 如果平移后超出交易日历末尾，会丢弃这些信号
    - 如果多个信号映射到同一成交日（少见），保留最后一个
    """
    if delay_days <= 0 or weights.empty:
        return weights
    
    cal = pd.DatetimeIndex(trading_calendar).normalize()
    sig = pd.DatetimeIndex(weights.index).normalize()

    locs = cal.get_indexer(sig)
    if (locs < 0).any():
        missing = sig[locs < 0]
        raise ValueError(
            f"Some weight dates are not in trading_calendar (first few: {missing[:5].tolist()})"
        )
    
    exec_locs = locs + int(delay_days)
    ok = exec_locs < len(cal)
    dropped = int((~ok).sum())
    if dropped > 0:
        logger.info(f"execution_delay_days={delay_days}: dropped {dropped} signals beyond end_date")
    
    shifted = weights.iloc[ok].copy()
    shifted.index = cal[exec_locs[ok]]
    shifted = shifted[~pd.DatetimeIndex(shifted.index).duplicated(keep="last")].sort_index()
    return shifted

def _align_weights_to_schedule(
    weights: PandasDataFrame,
    rebalance_dates: PandasDatetimeIndex,
) -> PandasDataFrame:
    """
    Align weights to rebalance schedule, forward-fill between rebalances.
    将权重矩阵对齐到调仓日期表,在调仓日期之间前向填充.

    Args / 参数:
        weights: 权重矩阵 (date x asset),来自因子模型的原始输出
        rebalance_dates: 调仓日期列表,由 generate_rebalance_schedule 生成

    Returns / 返回:
        过滤后的权重矩阵,只保留调仓日期
    """
    # 找出权重索引和调仓日期的交集 / Find intersection of weights index and rebalance dates
    available_dates = weights.index.intersection(rebalance_dates)

    if available_dates.empty:
        logger.warning("No rebalance dates found in weights index")
        return pd.DataFrame()

    logger.debug(f"Aligned weights to {len(available_dates)} rebalance dates")
    return weights.loc[available_dates]

def _simulate_portfolio(
    weights: PandasDataFrame,
    prices: PandasDataFrame,
    config: BacktestConfig, 
    record_trades: bool,
    verbose: bool = False,
) -> tuple[PandasSeries, PandasDataFrame, PandasDataFrame | None]:
    """
    Simulate portfolio: apply weights on rebalance dates, track daily equity.
    模拟组合运行:在调仓日应用权重,每日跟踪净值.

    核心逻辑 / Core Logic:
    1. 遍历每个交易日 / Loop through each trading day
    2. 如果是调仓日,计算目标仓位并执行交易 / If rebalance day, compute target and trade
    3. 计算交易成本(手续费 + 滑点)/ Calculate transaction costs (commission + slippage)
    4. 每日按市值计价 / Mark to market daily
    
    注意 / Notes:
    - 当价格缺失(NaN)或为0时,该资产保持原持仓不交易
    - 手续费和滑点以基点(bps)为单位,1bps = 0.01%

    Args / 参数:
        weights: 目标权重矩阵,只包含调仓日 (已由 _align_weights_to_schedule 过滤)
        prices: 价格矩阵 (date x asset),包含所有交易日
        initial_cash: 初始资金
        commission_bps: 手续费(基点),如 5 表示 0.05%
        slippage_bps: 滑点(基点),如 5 表示 0.05%
        record_trades: 是否记录每笔交易明细
        verbose: 是否打印详细的调仓信息

    Returns / 返回:
        (equity_curve, positions_daily, trades_df)
        - equity_curve: 每日净值曲线 (Series)
        - positions_daily: 每日持仓权重矩阵 (DataFrame)
        - trades_df: 交易记录表 (DataFrame or None)
    """
    initial_cash = config.initial_cash
    commission_bps = config.commission_bps
    slippage_bps = config.slippage_bps
    max_turnover = config.max_turnover
    rebalance_threshold = config.rebalance_threshold
    # -------------------------------------------------------------------------
    # 边界情况处理 / Handle edge cases
    # -------------------------------------------------------------------------
    if prices.empty:
        logger.warning("Empty prices DataFrame")
        return (
            pd.Series(dtype=float),
            pd.DataFrame(dtype=float),
            None,
        )
    
    if weights.empty:
        logger.warning("Empty weights DataFrame, holding cash only")
        equity_series = pd.Series(initial_cash, index=prices.index)
        positions_matrix = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        return equity_series, positions_matrix, None
    
    # -------------------------------------------------------------------------
    # 初始化 / Initialization
    # -------------------------------------------------------------------------
    trading_days = prices.index  # 所有交易日
    all_assets = prices.columns  # 所有资产
    
    # 将调仓日期规范化以便比较 / Normalize rebalance dates for consistent comparison
    rebalance_dates_set = set(pd.DatetimeIndex(weights.index).normalize())

    # 初始化结果容器 / Initialize output containers
    equity_series = pd.Series(index=trading_days, dtype=float)  # 净值序列
    positions_matrix = pd.DataFrame(0.0, index=trading_days, columns=all_assets)  # 持仓矩阵
    trades_records = [] if record_trades else None  # 交易记录

    # 组合状态 / Portfolio state
    cash = initial_cash  # 当前现金
    holdings = pd.Series(0.0, index=all_assets)  # 当前持股数量(非金额)

    equity_series.iloc[0] = cash  # 第一天净值 = 初始资金

    # -------------------------------------------------------------------------
    # 主循环:遍历每个交易日 / Main loop: iterate through each trading day
    # -------------------------------------------------------------------------
    for i, date in enumerate(trading_days):
        # 获取当日价格 / Get current day's prices
        px = prices.loc[date]
        # 规范化日期以便比较 / Normalize date for comparison
        date_normalized = pd.Timestamp(date).normalize()

        # =====================================================================
        # 调仓逻辑:如果是调仓日则执行再平衡 / Rebalance if scheduled
        # =====================================================================
        if date_normalized in rebalance_dates_set:
            # -----------------------------------------------------------------
            # Step A: 获取目标权重 / Get target weights
            # -----------------------------------------------------------------
            try:
                target_weights = weights.loc[date].fillna(0.0)
            except KeyError:
                # 回退方案:按规范化日期查找 / Fallback: find by normalized date
                weight_mask = pd.DatetimeIndex(weights.index).normalize() == date_normalized
                if weight_mask.any():
                    target_weights = weights.loc[weight_mask].iloc[0].fillna(0.0)
                else:
                    logger.warning(f"No weights found for {date}, skipping rebalance")
                    continue

            # -----------------------------------------------------------------
            # Step B: 计算当前组合价值 / Calculate current portfolio value
            # -----------------------------------------------------------------
            portfolio_value = cash + (holdings * px).sum()

            # -----------------------------------------------------------------
            # Step C: P0-3 修复 - 处理 NaN/0 价格
            # 价格缺失的资产保持原持仓,不交易
            # P0-3 Fix: Handle NaN/0 prices - keep original position for missing prices
            # -----------------------------------------------------------------
            valid_price_mask = px.notna() & (px > 0)  # 有效价格掩码
            
            # 计算目标股数(仅对有效价格的资产)
            # Compute target shares (only for assets with valid prices)
            target_dollar = target_weights * portfolio_value  # 目标金额
            target_shares = holdings.copy()  # 默认保持原持仓
            
            # 只对有有效价格的资产计算目标仓位
            # Only compute target position for assets with valid prices
            for asset in target_weights.index:
                if valid_price_mask.get(asset, False):
                    target_shares[asset] = target_dollar[asset] / px[asset]
                # else: 保持 holdings[asset] 不变 / keep holdings unchanged

            # -----------------------------------------------------------------
            # Step D: 计算交易量 / Calculate trade amounts
            # -----------------------------------------------------------------
            trades = pd.Series(0.0, index=all_assets)
            for asset in all_assets:
                if valid_price_mask.get(asset, False):
                    trades[asset] = target_shares[asset] - holdings[asset]
                # else: 不交易 trades[asset] = 0 / no trade for invalid price

            # -----------------------------------------------------------------
            # Step E: P0-2 修复 - 计算换手率
            # turnover = 成交金额 / 组合净值
            # P0-2 Fix: turnover = trade_value / portfolio_value
            # -----------------------------------------------------------------
            trade_value = (trades.abs() * px.fillna(0)).sum()  # 成交金额
            turnover_pct = trade_value / portfolio_value if portfolio_value > 1e-6 else 0.0

            # =================================================================
            # Step E.1: 换手控制 - rebalance_threshold & max_turnover
            # Turnover control: skip if below threshold, cap if above max
            # =================================================================
            
            # (1) 低于阈值不调仓 / Skip rebalance if turnover below threshold
            if turnover_pct < rebalance_threshold:
                if verbose:
                    print(f"[{date.date()}] SKIP: turnover {turnover_pct:.2%} < threshold {rebalance_threshold:.2%}")
                logger.debug(f"{date.date()}: Skip rebalance, turnover {turnover_pct:.2%} < {rebalance_threshold:.2%}")
                continue
            
            # (2) 超过上限部分做调仓 / Partial rebalance if turnover exceeds max
            if turnover_pct > max_turnover:
                # 计算当前权重 / Calculate current weights
                current_weights = pd.Series(0.0, index=all_assets)
                if portfolio_value > 1e-6:
                    current_weights = (holdings * px.fillna(0)) / portfolio_value

                # 缩放因子: 只移动 max_turnover / turnover_pct 的距离
                # Scaling factor: move only max_turnover / turnover_pct of the way
                scaling = max_turnover / turnover_pct

                # 调整后权重 = 当前权重 + scaling x (目标权重 - 当前权重)
                # Adjusted weights = current + scaling * (target - current)
                adjusted_weights = current_weights + scaling * (target_weights - current_weights)

                # 重新计算目标股数和交易量 / Recalculate target shares and trades
                target_dollar = adjusted_weights * portfolio_value
                for asset in adjusted_weights.index:
                    if valid_price_mask.get(asset, False):
                        target_shares[asset] = target_dollar[asset] / px[asset]

                # 重新计算 trades
                for asset in all_assets:
                    if valid_price_mask.get(asset, False):
                        trades[asset] = target_shares[asset] - holdings[asset]
                
                # 更新换手率统计
                trade_value = (trades.abs() * px.fillna(0)).sum()
                actual_turnover = trade_value / portfolio_value if portfolio_value > 1e-6 else 0.0

                if verbose:
                    print(f"[{date.date()}] PARTIAL: turnover capped {turnover_pct:.2%} -> {actual_turnover:.2%}")
                logger.debug(f"{date.date()}: Partial rebalance, {turnover_pct:.2%} -> {actual_turnover:.2%}")
                
                turnover_pct = actual_turnover  # 更新用于后续记录
            
            # -----------------------------------------------------------------
            # Step F: 计算交易成本 / Apply transaction costs
            # commission = 手续费, slippage = 滑点
            # -----------------------------------------------------------------
            commission = 0.0
            slippage_cost = 0.0
            total_cost = 0.0
            if trade_value > 1e-6:
                commission = trade_value * (commission_bps / 10000.0)  # bps -> 小数
                slippage_cost = trade_value * (slippage_bps / 10000.0)
                total_cost = commission + slippage_cost
                
                cash -= total_cost  # 从现金中扣除交易成本

            # -----------------------------------------------------------------
            # Step G: 详细输出(可选)/ Verbose output (optional)
            # -----------------------------------------------------------------
            if verbose:
                # 调仓后的头寸 / Top holdings after rebalance
                target_positions = target_weights[target_weights > 0].sort_values(ascending=False)
                top_holdings = target_positions.head(5)
                top_str = ", ".join([f"{k}:{v:.1%}" for k, v in top_holdings.items()])
                
                print(f"\n[{date.date()}] REBALANCE")
                print(f"  Portfolio Value: ${portfolio_value:,.0f}")
                print(f"  Trade Value:     ${trade_value:,.0f}")
                print(f"  Turnover:        {turnover_pct:.1%}")
                print(f"  Cost:            ${total_cost:,.2f} (comm={commission:.2f}, slip={slippage_cost:.2f})")
                print(f"  Top Holdings:    {top_str}")

                logger.debug(
                    f"{date.date()}: Rebalance, turnover_$={trade_value:,.0f}, "
                    f"turnover_%={turnover_pct:.2%}, cost={total_cost:,.2f}"
                )

            # -----------------------------------------------------------------
            # Step H: 更新持仓 / Update holdings
            # -----------------------------------------------------------------
            trade_cash = -(trades * px.fillna(0)).sum()  # 买入为负,卖出为正
            cash += trade_cash
            holdings = target_shares.copy()

            # -----------------------------------------------------------------
            # Step I: 记录交易明细 / Record trades
            # -----------------------------------------------------------------
            if record_trades and trade_value > 1e-6:
                for asset in trades[trades.abs() > 1e-6].index:
                    trades_records.append({
                        "date": date,
                        "asset": asset,
                        "shares": trades[asset],
                        "price": px[asset],
                        "value": trades[asset] * px[asset],
                    })
        
        # =====================================================================
        # 每日市值计价 / Mark to market daily
        # =====================================================================
        portfolio_value = cash + (holdings * px).sum()
        equity_series.loc[date] = portfolio_value

        # 记录当日持仓权重 / Record daily position weights
        if portfolio_value > 1e-6:
            positions_matrix.loc[date] = (holdings * px) / portfolio_value

    # -------------------------------------------------------------------------
    # 整理交易记录 / Convert trades to DataFrame
    # -------------------------------------------------------------------------
    trades_df = pd.DataFrame(trades_records) if trades_records else None

    return equity_series, positions_matrix, trades_df