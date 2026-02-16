#!/usr/bin/env python3
"""
5-Minute Market Strategy Runner

Multi-signal strategy optimized for Polymarket 5-minute Up/Down markets.
Fast-paced with aggressive signal detection and tight TP/SL.

Configuration rationale (5-minute markets):
  - Shorter lookback (5s) since moves happen faster
  - Lower flash crash threshold (0.15) to catch smaller but frequent drops
  - Tighter momentum window (15s) and threshold (0.06) for quick trends
  - Lower orderbook imbalance threshold (2.0x) since liquidity is thinner
  - Time decay activates with 30% time remaining (last ~90 seconds)
  - Tight TP (+$0.05) and SL (-$0.03) to lock in quick scalps
  - 2 max positions (can hold both UP and DOWN simultaneously)
  - Faster market check (15s) since 5m windows rotate quickly
  - Won't enter with less than 60s remaining (late hold-to-expiry zone)

Usage:
    python apps/run_5m.py
    python apps/run_5m.py --coin BTC
    python apps/run_5m.py --coin ETH --size 10
    python apps/run_5m.py --coin SOL --size 5 --score 0.5
"""

import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path

# Suppress noisy logs
logging.getLogger("src.websocket_client").setLevel(logging.WARNING)
logging.getLogger("src.bot").setLevel(logging.WARNING)
logging.getLogger("src.client").setLevel(logging.WARNING)

# Auto-load .env file
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.console import Colors
from src.bot import TradingBot
from src.config import Config
from strategies.multi_signal import MultiSignalStrategy, MultiSignalConfig


# ---------------------------------------------------------------------------
# Optimal defaults for 5-minute markets
# ---------------------------------------------------------------------------

DEFAULTS_5M = dict(
    # Core
    market_duration=5,
    size=6.0,
    max_positions=1,
    take_profit=0.50,     # 50% profit target
    stop_loss=0.10,       # 10% stop loss (after first minute)
    early_stop_loss=0.30, # 30% stop loss during first minute of market
    early_sl_window=60,   # First 60 seconds use wider SL

    # Market timing
    market_check_interval=15.0,
    auto_switch_market=True,

    # Price tracking
    price_lookback_seconds=5,
    price_history_size=200,

    # Display / loop
    update_interval=0.05,
    order_refresh_interval=15.0,

    # Flash crash (mean reversion) -- smaller threshold for fast markets
    flash_crash_enabled=True,
    drop_threshold=0.08,

    # Momentum (trend following) -- short window, low threshold
    momentum_enabled=True,
    momentum_window=15,
    momentum_threshold=0.03,
    momentum_min_ticks=3,
    momentum_consistency=0.65,

    # Orderbook imbalance -- lower ratio since books are thinner
    imbalance_enabled=True,
    imbalance_signal_threshold=0.18,
    imbalance_depth=5,

    # Time decay (convergence) -- activate in last 30% (~90s)
    time_decay_enabled=True,
    time_decay_threshold_pct=0.30,

    # Signal combination
    min_signal_score=0.35,
    dynamic_threshold_base=0.35,
    dynamic_threshold_vol_adjustment=0.10,
    signal_cooldown=8.0,
    min_time_remaining=60,
)


def main():
    """Main entry point for 5-minute market strategy."""
    parser = argparse.ArgumentParser(
        description="Multi-Signal Strategy for Polymarket 5-minute markets"
    )
    parser.add_argument(
        "--coin",
        type=str,
        default="ETH",
        choices=["BTC", "ETH", "SOL", "XRP"],
        help="Coin to trade (default: ETH)",
    )
    parser.add_argument(
        "--size",
        type=float,
        default=DEFAULTS_5M["size"],
        help=f"Trade size in USDC (default: {DEFAULTS_5M['size']})",
    )
    parser.add_argument(
        "--score",
        type=float,
        default=DEFAULTS_5M["min_signal_score"],
        help=f"Min signal score to enter (default: {DEFAULTS_5M['min_signal_score']})",
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=DEFAULTS_5M["take_profit"],
        help=f"Take profit as fraction (default: {DEFAULTS_5M['take_profit']} = {DEFAULTS_5M['take_profit']:.0%})",
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=DEFAULTS_5M["stop_loss"],
        help=f"Stop loss as fraction (default: {DEFAULTS_5M['stop_loss']} = {DEFAULTS_5M['stop_loss']:.0%})",
    )
    parser.add_argument(
        "--no-flash-crash",
        action="store_true",
        help="Disable flash crash signal",
    )
    parser.add_argument(
        "--no-momentum",
        action="store_true",
        help="Disable momentum signal",
    )
    parser.add_argument(
        "--no-imbalance",
        action="store_true",
        help="Disable orderbook imbalance signal",
    )
    parser.add_argument(
        "--no-time-decay",
        action="store_true",
        help="Disable time decay signal",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--signal-log",
        action="store_true",
        help="Print live composite + per-signal score bars",
    )

    args = parser.parse_args()

    # Enable debug logging if requested
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("src.websocket_client").setLevel(logging.DEBUG)

    # Check environment
    private_key = os.environ.get("POLY_PRIVATE_KEY")
    safe_address = os.environ.get("POLY_SAFE_ADDRESS")

    if not private_key or not safe_address:
        print(
            f"{Colors.RED}Error: POLY_PRIVATE_KEY and POLY_SAFE_ADDRESS "
            f"must be set{Colors.RESET}"
        )
        print("Set them in .env file or export as environment variables")
        sys.exit(1)

    # Create bot
    config = Config.from_env()
    bot = TradingBot(config=config, private_key=private_key)

    if not bot.is_initialized():
        print(f"{Colors.RED}Error: Failed to initialize bot{Colors.RESET}")
        sys.exit(1)

    # Build strategy config from defaults + CLI overrides
    strategy_config = MultiSignalConfig(
        coin=args.coin.upper(),
        size=args.size,
        take_profit=args.take_profit,
        stop_loss=args.stop_loss,
        min_signal_score=args.score,
        flash_crash_enabled=DEFAULTS_5M["flash_crash_enabled"] and not args.no_flash_crash,
        momentum_enabled=DEFAULTS_5M["momentum_enabled"] and not args.no_momentum,
        imbalance_enabled=DEFAULTS_5M["imbalance_enabled"] and not args.no_imbalance,
        time_decay_enabled=DEFAULTS_5M["time_decay_enabled"] and not args.no_time_decay,
        # Non-overridable defaults
        market_duration=DEFAULTS_5M["market_duration"],
        max_positions=DEFAULTS_5M["max_positions"],
        market_check_interval=DEFAULTS_5M["market_check_interval"],
        auto_switch_market=DEFAULTS_5M["auto_switch_market"],
        price_lookback_seconds=DEFAULTS_5M["price_lookback_seconds"],
        price_history_size=DEFAULTS_5M["price_history_size"],
        update_interval=DEFAULTS_5M["update_interval"],
        order_refresh_interval=DEFAULTS_5M["order_refresh_interval"],
        drop_threshold=DEFAULTS_5M["drop_threshold"],
        momentum_window=DEFAULTS_5M["momentum_window"],
        momentum_threshold=DEFAULTS_5M["momentum_threshold"],
        momentum_min_ticks=DEFAULTS_5M["momentum_min_ticks"],
        momentum_consistency=DEFAULTS_5M["momentum_consistency"],
        imbalance_signal_threshold=DEFAULTS_5M["imbalance_signal_threshold"],
        imbalance_depth=DEFAULTS_5M["imbalance_depth"],
        time_decay_threshold_pct=DEFAULTS_5M["time_decay_threshold_pct"],
        dynamic_threshold_base=DEFAULTS_5M["dynamic_threshold_base"],
        dynamic_threshold_vol_adjustment=DEFAULTS_5M["dynamic_threshold_vol_adjustment"],
        signal_state_logging_enabled=args.signal_log,
        signal_cooldown=DEFAULTS_5M["signal_cooldown"],
        min_time_remaining=DEFAULTS_5M["min_time_remaining"],
    )

    # Print configuration
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(
        f"{Colors.BOLD}  Multi-Signal Strategy - {strategy_config.coin} "
        f"5-Minute Markets{Colors.RESET}"
    )
    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}\n")

    print("Configuration:")
    print(f"  Coin:           {strategy_config.coin}")
    print(f"  Size:           ${strategy_config.size:.2f}")
    print(f"  Take profit:    +{strategy_config.take_profit:.0%}")
    print(f"  Stop loss:      -{strategy_config.stop_loss:.0%}")
    print(f"  Max positions:  {strategy_config.max_positions}")
    print(f"  Min score:      {strategy_config.min_signal_score:.1f}")
    print()

    # Signal status
    signals = []
    if strategy_config.flash_crash_enabled:
        signals.append(f"  {Colors.GREEN}ON{Colors.RESET}  Flash Crash   (drop >= {strategy_config.drop_threshold:.2f} in {strategy_config.price_lookback_seconds}s)")
    else:
        signals.append(f"  {Colors.RED}OFF{Colors.RESET} Flash Crash")

    if strategy_config.momentum_enabled:
        signals.append(f"  {Colors.GREEN}ON{Colors.RESET}  Momentum      (move >= {strategy_config.momentum_threshold:.2f} in {strategy_config.momentum_window}s)")
    else:
        signals.append(f"  {Colors.RED}OFF{Colors.RESET} Momentum")

    if strategy_config.imbalance_enabled:
        signals.append(f"  {Colors.GREEN}ON{Colors.RESET}  OB Imbalance  (score >= {strategy_config.imbalance_signal_threshold:.2f})")
    else:
        signals.append(f"  {Colors.RED}OFF{Colors.RESET} OB Imbalance")

    if strategy_config.time_decay_enabled:
        signals.append(f"  {Colors.GREEN}ON{Colors.RESET}  Time Decay    (< {strategy_config.time_decay_threshold_pct:.0%} remaining)")
    else:
        signals.append(f"  {Colors.RED}OFF{Colors.RESET} Time Decay")

    print("Signal Detectors:")
    for s in signals:
        print(s)
    print()

    # Create and run strategy
    strategy = MultiSignalStrategy(bot=bot, config=strategy_config)

    try:
        asyncio.run(strategy.run())
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
