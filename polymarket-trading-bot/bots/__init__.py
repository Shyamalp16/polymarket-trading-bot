"""
Dual-Bot Trading System for Polymarket

This package contains:
- Momentum Bot (Bot A): Impulse chaser for trend-following
- Mean Reversion Bot (Bot B): Pullback eater for contrarian trading
- Coordinator: Manages capital, conflicts, and risk

Usage:
    python -m bots.run_dual_bot --bankroll 100 --coin BTC
"""

from bots.momentum_bot import MomentumBot, MomentumConfig, MomentumSignal
from bots.mean_reversion_bot import MeanReversionBot, MeanReversionConfig, ReversionSignal
from bots.coordinator import Coordinator, CoordinatorConfig

__all__ = [
    "MomentumBot",
    "MomentumConfig", 
    "MomentumSignal",
    "MeanReversionBot",
    "MeanReversionConfig",
    "ReversionSignal",
    "Coordinator",
    "CoordinatorConfig",
]
