"""
Strategies - Trading Strategy Implementations

This package contains trading strategy implementations:

- base: Base class for all strategies
- flash_crash: Flash crash volatility strategy
- multi_signal: Combined multi-signal strategy (flash crash + momentum +
  orderbook imbalance + time decay) for 5m and 15m markets

Usage:
    from strategies.base import BaseStrategy, StrategyConfig
    from strategies.flash_crash import FlashCrashStrategy, FlashCrashConfig
    from strategies.multi_signal import MultiSignalStrategy, MultiSignalConfig
"""

from strategies.base import BaseStrategy, StrategyConfig
from strategies.flash_crash import FlashCrashStrategy, FlashCrashConfig
from strategies.multi_signal import MultiSignalStrategy, MultiSignalConfig

__all__ = [
    "BaseStrategy",
    "StrategyConfig",
    "FlashCrashStrategy",
    "FlashCrashConfig",
    "MultiSignalStrategy",
    "MultiSignalConfig",
]
