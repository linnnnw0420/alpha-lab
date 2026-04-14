"""
Backtest configuration module.
回测配置模块.

Key classes / 核心类:
- BacktestConfig: 回测配置数据类 / backtest configuration dataclass
- BacktestFreq: 调仓频率枚举 / rebalance frequency enum
- PriceField: 价格字段枚举 / price field enum

Key functions / 核心函数:
- default_backtest_config: 返回默认配置 / return default config

配置字段说明 / Configuration Fields:
    - start_date/end_date: 回测日期范围
    - rebalance_freq: 调仓频率 (D/W/M)
    - initial_cash: 初始资金
    - commission_bps: 手续费(基点)
    - slippage_bps: 滑点(基点)
    - price_field: 执行价格(open/close)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime
from enum import Enum
from typing import TypeAlias

from alpha_lab.utils.typing import DateLike

# ----------- Enums (Standard Library) ------------ 


class BacktestFreq(str, Enum):
    """
    Rebalance frequency enumeration.
    调仓频率枚举.
    
    继承 str 以支持直接字符串比较和 JSON 序列化.
    Inherits from str to allow direct string comparisons and JSON serialization.
    """
    DAILY = "D"     # 每日 / Daily
    WEEKLY = "W"    # 每周 / Weekly
    MONTHLY = "M"   # 每月 / Monthly
    
    @classmethod
    def all_values(cls) -> tuple[str, ...]:
        """Return all valid frequency values. / 返回所有有效的频率值."""
        return tuple(member.value for member in cls)
    
    def __str__(self) -> str:
        """Return the string value for easy printing."""
        return self.value


class PriceField(str, Enum):
    """
    Price field enumeration for backtest execution.
    回测执行价格字段枚举.
    
    继承 str 以支持直接字符串比较和 JSON 序列化.
    Inherits from str to allow direct string comparisons and JSON serialization.
    """
    OPEN = "open"   # 开盘价 / Open price
    CLOSE = "close" # 收盘价 / Close price
    HIGH = "high"   # 最高价 / High price
    LOW = "low"     # 最低价 / Low price
    VWAP = "vwap"   # 成交量加权平均价 / Volume-weighted average price
    
    @classmethod
    def all_values(cls) -> tuple[str, ...]:
        """Return all valid price field values. / 返回所有有效的价格字段值."""
        return tuple(member.value for member in cls)
    
    def __str__(self) -> str:
        """Return the string value for easy printing."""
        return self.value


# Type aliases for backwards compatibility and type hints
# 类型别名,用于向后兼容和类型提示
RebalanceFreq: TypeAlias = str | BacktestFreq
PriceFieldType: TypeAlias = str | PriceField

# ------------- Helpers / 辅助函数 ----------------

def _normalize_date(value: DateLike, field_name: str) -> str:
    '''
    Normalize input into ISO date string: YYYY-MM-DD.
    将输入规范化为 ISO 日期字符串格式: YYYY-MM-DD.

    保持严格格式以避免时区/时间歧义.
    Keep it strict to avoid timezone/time-of-day ambiguity.
    '''

    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        s = value.strip()
        # strict ISO date validation
        try:
            date.fromisoformat(s) # raises ValueError if invalid
        except ValueError as e:
            raise ValueError(
                f"{field_name} must be ISO date 'YYYY-MM-DD', got: {value!r}"
            ) from e
        return s

    raise TypeError(f"{field_name} must be str/date/datetime, got: {type(value).__name__}")

def _validate_non_negative(name: str, value: float) -> None:
    """
    Validate that a numeric value is non-negative.
    验证数值是否非负.
    """
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got: {value}")


def _normalize_freq(value: RebalanceFreq) -> str:
    """
    Normalize frequency to string value.
    将频率规范化为字符串值.
    """
    if isinstance(value, BacktestFreq):
        return value.value
    if isinstance(value, str):
        if value not in BacktestFreq.all_values():
            raise ValueError(
                f"rebalance_freq must be one of {BacktestFreq.all_values()}, "
                f"got {value!r}"
            )
        return value
    raise TypeError(
        f"rebalance_freq must be str or BacktestFreq, "
        f"got {type(value).__name__}"
    )


def _normalize_price_field(value: PriceFieldType) -> str:
    """
    Normalize price field to string value.
    将价格字段规范化为字符串值.
    """
    if isinstance(value, PriceField):
        return value.value
    if isinstance(value, str):
        if value not in PriceField.all_values():
            raise ValueError(
                f"price_field must be one of {PriceField.all_values()}, "
                f"got {value!r}"
            )
        return value
    raise TypeError(
        f"price_field must be str or PriceField, "
        f"got {type(value).__name__}"
    )


# ---------------- Core config / 核心配置 -------------

@dataclass(slots=True, frozen=True)
class BacktestConfig:
    """
    Backtest configuration (lightweight; no pandas dependency).
    回测配置(轻量级,无 pandas 依赖).

    Notes / 注意事项:
        - start_date/end_date 在 __post_init__ 中规范化为 'YYYY-MM-DD'
        - commission_bps/slippage_bps 以基点为单位 (1 bps = 0.01%)
        - 支持字符串值和枚举成员作为 freq/price_field 参数
    
    Examples / 示例:
        使用枚举 / Using enums:
        >>> cfg = BacktestConfig(
        ...     start_date="2020-01-01",
        ...     end_date="2024-12-31",
        ...     rebalance_freq=BacktestFreq.MONTHLY,
        ...     initial_cash=1_000_000.0,
        ...     commission_bps=5.0,
        ...     slippage_bps=2.0,
        ...     price_field=PriceField.CLOSE,
        ... )
        
        使用字符串 / Using strings:
        >>> cfg = BacktestConfig(
        ...     start_date="2020-01-01",
        ...     end_date="2024-12-31",
        ...     rebalance_freq="M",
        ...     initial_cash=1_000_000.0,
        ...     commission_bps=5.0,
        ...     slippage_bps=2.0,
        ...     price_field="close",
        ... )
    """

    start_date: DateLike      # 开始日期
    end_date: DateLike        # 结束日期

    rebalance_freq: RebalanceFreq  # 调仓频率 'D'/'W'/'M' 或 BacktestFreq 枚举
    initial_cash: float       # 初始资金
    
    commission_bps: float     # 手续费(基点)
    slippage_bps: float       # 滑点(基点)

    price_field: PriceFieldType  # 执行价格 'open'/'close'/等 或 PriceField 枚举
    benchmark: str | None = None # 基准指数(如 "SPY")

    max_turnover: float = 0.3   #最大可交易仓数
    rebalance_threshold: float = 0.0 
    execution_delay_days: int = 1
    
    def __post_init__(self) -> None:
        # Normalize dates into ISO strings
        object.__setattr__(self, "start_date", _normalize_date(self.start_date, "start_date"))
        object.__setattr__(self, "end_date", _normalize_date(self.end_date, "end_date"))

        if self.start_date > self.end_date:
            raise ValueError(
                f"start_date must be <= end_date, "
                f"got {self.start_date} > {self.end_date}"
            )

        # Normalize and validate frequency
        object.__setattr__(self, "rebalance_freq", _normalize_freq(self.rebalance_freq))

        # Normalize and validate price field
        object.__setattr__(self, "price_field", _normalize_price_field(self.price_field))

        if self.initial_cash <= 0:
            raise ValueError(f"initial_cash must be > 0, got: {self.initial_cash}")
        if not 0 <= self.max_turnover <= 1:
            raise ValueError(f"max_turnover must be in [0, 1], got {self.max_turnover}")

        _validate_non_negative("execution_delay_days", float(self.execution_delay_days))
        _validate_non_negative("rebalance_threshold", float(self.rebalance_threshold))
        _validate_non_negative("commission_bps", float(self.commission_bps))
        _validate_non_negative("slippage_bps", float(self.slippage_bps))
    
    def with_updates(self, **kwargs) -> BacktestConfig:
        """
        Create a new config with updated fields (re-validates via __post_init__).
        创建一个更新字段后的新配置(通过 __post_init__ 重新验证).
        """
        return replace(self, **kwargs)

def default_backtest_config() -> BacktestConfig:
    """
    Return a reasonable default backtest config for demos/notebooks.
    返回用于演示/笔记本的默认回测配置.

    返回值 / Returns:
        BacktestConfig 对象,包含以下默认值:
        - start_date: "2018-01-01"
        - end_date: "2024-12-31"
        - rebalance_freq: 月度 (MONTHLY)
        - initial_cash: 100 万美元
        - commission_bps: 5 基点
        - slippage_bps: 2 基点

    使用方式 / Usage:
        在笔记本中,你可以覆盖默认值:
        You are expected to override fields in notebook, e.g.:
        >>> cfg = default_backtest_config()
        >>> cfg = cfg.with_updates(start_date="2018-01-01", rebalance_freq=BacktestFreq.MONTHLY)
    """
    return BacktestConfig(
        start_date="2018-01-01",
        end_date="2024-12-31",
        rebalance_freq=BacktestFreq.MONTHLY,  # 月度调仓
        initial_cash=1_000_000.0,             # 初始资金 100 万
        commission_bps=5.0,                    # 手续费 5 基点 (0.05%)
        slippage_bps=2.0,                      # 滑点 2 基点 (0.02%)
        price_field=PriceField.CLOSE,          # 使用收盘价执行
        benchmark=None,                        # 基准,如 "SPY"(如果股票池包含的话)
    )

__all__ = [
    "BacktestConfig",
    "BacktestFreq",
    "PriceField",
    "RebalanceFreq",
    "PriceFieldType",
    "default_backtest_config",
]