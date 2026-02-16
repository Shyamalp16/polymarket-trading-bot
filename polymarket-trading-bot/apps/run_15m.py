#!/usr/bin/env python3
"""
15-Minute Market Strategy Runner

Multi-signal strategy optimized for Polymarket 15-minute Up/Down markets.
More patient with wider thresholds, allowing for larger moves and higher-
confidence entries compared to the 5-minute variant.

Configuration rationale (15-minute markets):
  - Standard lookback (10s) giving more data to work with
  - Higher flash crash threshold (0.20) to filter noise and catch real crashes
  - Wider momentum window (30s) and threshold (0.08) for sustained trends
  - Higher orderbook imbalance threshold (2.5x) since books are deeper
  - Time decay activates with 20% time remaining (last ~3 minutes)
  - TP (+$0.10) and SL (-$0.05) matching the standard risk/reward profile
  - 2 max positions (can hold both UP and DOWN simultaneously)
  - Standard market check (30s) since windows are longer
  - Won't enter with less than 180s remaining (late hold-to-expiry zone)
  - Higher min signal score (0.7) for more selective entries

Usage:
    python apps/run_15m.py
    python apps/run_15m.py --coin BTC
    python apps/run_15m.py --coin ETH --size 10
    python apps/run_15m.py --coin SOL --size 5 --score 0.6
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
# Optimal defaults for 15-minute markets
# ---------------------------------------------------------------------------

DEFAULTS_15M = dict(
    # Core
    market_duration=15,
    size=5.0,
    max_positions=1,
    take_profit=0.75,     # 75% profit target
    stop_loss=0.10,       # 10% stop loss

    # Market timing
    market_check_interval=30.0,
    auto_switch_market=True,

    # Price tracking
    price_lookback_seconds=10,
    price_history_size=200,

    # Display / loop
    update_interval=0.1,
    order_refresh_interval=30.0,

    # Flash crash (mean reversion) -- higher threshold for deeper markets
    flash_crash_enabled=True,
    drop_threshold=0.08,

    # Momentum (trend following) -- wider window, higher threshold
    momentum_enabled=True,
    momentum_window=30,
    momentum_threshold=0.03,
    momentum_min_ticks=4,
    momentum_consistency=0.65,

    # Orderbook imbalance -- higher ratio since books are deeper
    imbalance_enabled=True,
    imbalance_signal_threshold=0.20,
    imbalance_depth=5,

    # Time decay (convergence) -- activate in last 20% (~3 minutes)
    time_decay_enabled=True,
    time_decay_threshold_pct=0.20,

    # Signal combination -- more selective
    min_signal_score=0.35,
    dynamic_threshold_base=0.35,
    dynamic_threshold_vol_adjustment=0.10,
    signal_cooldown=12.0,
    min_time_remaining=180,
)


def main():
    """Main entry point for 15-minute market strategy."""
    parser = argparse.ArgumentParser(
        description="Multi-Signal Strategy for Polymarket 15-minute markets"
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
        default=DEFAULTS_15M["size"],
        help=f"Trade size in USDC (default: {DEFAULTS_15M['size']})",
    )
    parser.add_argument(
        "--score",
        type=float,
        default=DEFAULTS_15M["min_signal_score"],
        help=f"Min signal score to enter (default: {DEFAULTS_15M['min_signal_score']})",
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=DEFAULTS_15M["take_profit"],
        help=f"Take profit in $ (default: {DEFAULTS_15M['take_profit']})",
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=DEFAULTS_15M["stop_loss"],
        help=f"Stop loss in $ (default: {DEFAULTS_15M['stop_loss']})",
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
        flash_crash_enabled=DEFAULTS_15M["flash_crash_enabled"] and not args.no_flash_crash,
        momentum_enabled=DEFAULTS_15M["momentum_enabled"] and not args.no_momentum,
        imbalance_enabled=DEFAULTS_15M["imbalance_enabled"] and not args.no_imbalance,
        time_decay_enabled=DEFAULTS_15M["time_decay_enabled"] and not args.no_time_decay,
        # Non-overridable defaults
        market_duration=DEFAULTS_15M["market_duration"],
        max_positions=DEFAULTS_15M["max_positions"],
        market_check_interval=DEFAULTS_15M["market_check_interval"],
        auto_switch_market=DEFAULTS_15M["auto_switch_market"],
        price_lookback_seconds=DEFAULTS_15M["price_lookback_seconds"],
        price_history_size=DEFAULTS_15M["price_history_size"],
        update_interval=DEFAULTS_15M["update_interval"],
        order_refresh_interval=DEFAULTS_15M["order_refresh_interval"],
        drop_threshold=DEFAULTS_15M["drop_threshold"],
        momentum_window=DEFAULTS_15M["momentum_window"],
        momentum_threshold=DEFAULTS_15M["momentum_threshold"],
        momentum_min_ticks=DEFAULTS_15M["momentum_min_ticks"],
        momentum_consistency=DEFAULTS_15M["momentum_consistency"],
        imbalance_signal_threshold=DEFAULTS_15M["imbalance_signal_threshold"],
        imbalance_depth=DEFAULTS_15M["imbalance_depth"],
        time_decay_threshold_pct=DEFAULTS_15M["time_decay_threshold_pct"],
        dynamic_threshold_base=DEFAULTS_15M["dynamic_threshold_base"],
        dynamic_threshold_vol_adjustment=DEFAULTS_15M["dynamic_threshold_vol_adjustment"],
        signal_state_logging_enabled=args.signal_log,
        signal_cooldown=DEFAULTS_15M["signal_cooldown"],
        min_time_remaining=DEFAULTS_15M["min_time_remaining"],
    )

    # Print configuration
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(
        f"{Colors.BOLD}  Multi-Signal Strategy - {strategy_config.coin} "
        f"15-Minute Markets{Colors.RESET}"
    )
    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}\n")

    print("Configuration:")
    print(f"  Coin:           {strategy_config.coin}")
    print(f"  Size:           ${strategy_config.size:.2f}")
    print(f"  Take profit:    +${strategy_config.take_profit:.2f}")
    print(f"  Stop loss:      -${strategy_config.stop_loss:.2f}")
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
