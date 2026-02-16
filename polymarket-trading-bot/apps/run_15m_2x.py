#!/usr/bin/env python3
"""
15-Minute 2x Hunter Strategy Runner

Aggressive multi-signal strategy optimized for high-return trades on
Polymarket 15-minute Up/Down markets. Targets 2x bankroll growth by
waiting patiently for extreme mispricings and riding convergence.

Strategy philosophy:
  - Only two signals enabled: Flash Crash + Time Decay
  - Flash Crash catches panic dumps where tokens trade far below fair value
  - Time Decay catches late-game convergence where outcome is nearly decided
  - Both produce entries at extreme prices (0.20-0.35) with huge upside
  - Tight SL (-$0.05) caps downside; wide TP (+$0.20) lets winners run
  - R:R ratio of ~4:1 to 10:1 means you only need 25-30% win rate
  - Trades less often but each win is worth 6-10 losses

Configuration rationale:
  - Flash crash threshold lowered to 0.15 (catch smaller but real crashes)
  - Time decay activates at 25% remaining (~3:45 left) with low divergence
  - Wide TP (+$0.20) to capture the full convergence move
  - Tight SL (-$0.05) to cut losers fast
  - Momentum and imbalance disabled (too small for 2x targets)
  - Lower min signal score (0.5) since each signal is already high-conviction
  - Longer cooldown (15s) for patience between entries
  - Max 2 positions (can hold UP + DOWN if both crash)
  - Safety cutoff at 180s (avoid new entries in late hold-to-expiry zone)

Expected performance (per 10 trades):
  - Win rate: ~30%
  - Avg win: +$5.00 (100% per trade)
  - Avg loss: -$0.83 (17% per trade)
  - Net: ~+$9.17 per 10 trades (18% return on capital)
  - 2x target: ~40 trades (4 cycles of 10)

Usage:
    python apps/run_15m_2x.py
    python apps/run_15m_2x.py --coin BTC
    python apps/run_15m_2x.py --coin ETH --size 10
    python apps/run_15m_2x.py --coin SOL --tp 0.25
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
# Optimal defaults for 2x hunting on 15-minute markets
# ---------------------------------------------------------------------------

DEFAULTS_2X = dict(
    # Core
    market_duration=15,
    size=5.0,
    max_positions=1,
    take_profit=1.50,       # 75% profit target
    stop_loss=0.10,         # 10% stop loss -- cut losers fast
    early_stop_loss=0.30,   # 30% stop loss during first minute of market
    early_sl_window=60,     # First 60 seconds use wider SL

    # Market timing
    market_check_interval=30.0,
    auto_switch_market=True,

    # Price tracking -- longer lookback to catch deeper crashes
    price_lookback_seconds=15,
    price_history_size=300,

    # Display / loop
    update_interval=0.1,
    order_refresh_interval=30.0,

    # Flash crash (mean reversion) -- lower threshold to catch more crashes
    flash_crash_enabled=True,
    drop_threshold=0.15,

    # Momentum -- DISABLED (too small for 2x targets)
    momentum_enabled=False,
    momentum_window=30,
    momentum_threshold=0.08,
    momentum_min_ticks=4,
    momentum_consistency=0.65,

    # Orderbook imbalance -- DISABLED (too small for 2x targets)
    imbalance_enabled=False,
    imbalance_signal_threshold=0.20,
    imbalance_depth=5,

    # Time decay (convergence) -- activate at 25% remaining (~3:45 left)
    time_decay_enabled=True,
    time_decay_threshold_pct=0.25,

    # Signal combination -- lower score since only 2 high-conviction signals
    min_signal_score=0.5,
    signal_cooldown=15.0,       # Patient between entries
    min_time_remaining=180,     # Avoid late-phase entries; hold existing positions instead
)


def main():
    """Main entry point for 2x hunter strategy."""
    parser = argparse.ArgumentParser(
        description="2x Hunter Strategy for Polymarket 15-minute markets"
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
        default=DEFAULTS_2X["size"],
        help=f"Trade size in USDC (default: {DEFAULTS_2X['size']})",
    )
    parser.add_argument(
        "--score",
        type=float,
        default=DEFAULTS_2X["min_signal_score"],
        help=f"Min signal score to enter (default: {DEFAULTS_2X['min_signal_score']})",
    )
    parser.add_argument(
        "--tp",
        type=float,
        default=DEFAULTS_2X["take_profit"],
        help=f"Take profit in $ (default: {DEFAULTS_2X['take_profit']})",
    )
    parser.add_argument(
        "--sl",
        type=float,
        default=DEFAULTS_2X["stop_loss"],
        help=f"Stop loss in $ (default: {DEFAULTS_2X['stop_loss']})",
    )
    parser.add_argument(
        "--with-momentum",
        action="store_true",
        help="Also enable momentum signal (off by default)",
    )
    parser.add_argument(
        "--with-imbalance",
        action="store_true",
        help="Also enable orderbook imbalance signal (off by default)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
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

    # Build strategy config
    strategy_config = MultiSignalConfig(
        coin=args.coin.upper(),
        size=args.size,
        take_profit=args.tp,
        stop_loss=args.sl,
        min_signal_score=args.score,
        # Core 2x signals: Flash Crash + Time Decay
        flash_crash_enabled=DEFAULTS_2X["flash_crash_enabled"],
        time_decay_enabled=DEFAULTS_2X["time_decay_enabled"],
        # Optional signals (off by default, opt-in via CLI)
        momentum_enabled=args.with_momentum,
        imbalance_enabled=args.with_imbalance,
        # Non-overridable defaults
        market_duration=DEFAULTS_2X["market_duration"],
        max_positions=DEFAULTS_2X["max_positions"],
        market_check_interval=DEFAULTS_2X["market_check_interval"],
        auto_switch_market=DEFAULTS_2X["auto_switch_market"],
        price_lookback_seconds=DEFAULTS_2X["price_lookback_seconds"],
        price_history_size=DEFAULTS_2X["price_history_size"],
        update_interval=DEFAULTS_2X["update_interval"],
        order_refresh_interval=DEFAULTS_2X["order_refresh_interval"],
        drop_threshold=DEFAULTS_2X["drop_threshold"],
        momentum_window=DEFAULTS_2X["momentum_window"],
        momentum_threshold=DEFAULTS_2X["momentum_threshold"],
        momentum_min_ticks=DEFAULTS_2X["momentum_min_ticks"],
        momentum_consistency=DEFAULTS_2X["momentum_consistency"],
        imbalance_signal_threshold=DEFAULTS_2X["imbalance_signal_threshold"],
        imbalance_depth=DEFAULTS_2X["imbalance_depth"],
        time_decay_threshold_pct=DEFAULTS_2X["time_decay_threshold_pct"],
        signal_cooldown=DEFAULTS_2X["signal_cooldown"],
        min_time_remaining=DEFAULTS_2X["min_time_remaining"],
    )

    # Print configuration
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(
        f"{Colors.BOLD}  2x Hunter Strategy - {strategy_config.coin} "
        f"15-Minute Markets{Colors.RESET}"
    )
    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}\n")

    # Risk/reward breakdown
    rr_ratio = strategy_config.take_profit / strategy_config.stop_loss
    print(f"{Colors.BOLD}Risk/Reward Profile:{Colors.RESET}")
    print(f"  Take profit:    +${strategy_config.take_profit:.2f} per share")
    print(f"  Stop loss:      -${strategy_config.stop_loss:.2f} per share")
    print(f"  R:R ratio:      {Colors.GREEN}{rr_ratio:.1f}:1{Colors.RESET}")
    print(f"  Breakeven rate: {1 / (1 + rr_ratio) * 100:.0f}% wins needed")
    print()

    print("Configuration:")
    print(f"  Coin:           {strategy_config.coin}")
    print(f"  Size:           ${strategy_config.size:.2f}")
    print(f"  Max positions:  {strategy_config.max_positions}")
    print(f"  Min score:      {strategy_config.min_signal_score:.1f}")
    print(f"  Cooldown:       {strategy_config.signal_cooldown:.0f}s")
    print()

    # Signal status
    signals = []
    if strategy_config.flash_crash_enabled:
        signals.append(
            f"  {Colors.GREEN}ON {Colors.RESET} Flash Crash   "
            f"(drop >= {strategy_config.drop_threshold:.2f} in "
            f"{strategy_config.price_lookback_seconds}s)"
        )
    else:
        signals.append(f"  {Colors.RED}OFF{Colors.RESET} Flash Crash")

    if strategy_config.time_decay_enabled:
        signals.append(
            f"  {Colors.GREEN}ON {Colors.RESET} Time Decay    "
            f"(< {strategy_config.time_decay_threshold_pct:.0%} remaining)"
        )
    else:
        signals.append(f"  {Colors.RED}OFF{Colors.RESET} Time Decay")

    if strategy_config.momentum_enabled:
        signals.append(
            f"  {Colors.GREEN}ON {Colors.RESET} Momentum      "
            f"(move >= {strategy_config.momentum_threshold:.2f} in "
            f"{strategy_config.momentum_window}s)"
        )
    else:
        signals.append(f"  {Colors.DIM}OFF{Colors.RESET} Momentum      (use --with-momentum to enable)")

    if strategy_config.imbalance_enabled:
        signals.append(
            f"  {Colors.GREEN}ON {Colors.RESET} OB Imbalance  "
            f"(score >= {strategy_config.imbalance_signal_threshold:.2f})"
        )
    else:
        signals.append(f"  {Colors.DIM}OFF{Colors.RESET} OB Imbalance  (use --with-imbalance to enable)")

    print("Signal Detectors:")
    for s in signals:
        print(s)

    # Expected performance
    print()
    print(f"{Colors.BOLD}Expected Performance (per 10 trades @ 30% win rate):{Colors.RESET}")
    example_entry = 0.30
    example_shares = strategy_config.size / example_entry
    example_win = strategy_config.take_profit * example_shares
    example_loss = strategy_config.stop_loss * example_shares
    print(f"  Example entry @ {example_entry:.2f}: {example_shares:.1f} shares")
    print(f"  Win PnL:   {Colors.GREEN}+${example_win:.2f}{Colors.RESET}  ({example_win / strategy_config.size * 100:.0f}% return)")
    print(f"  Loss PnL:  {Colors.RED}-${example_loss:.2f}{Colors.RESET}  ({example_loss / strategy_config.size * 100:.0f}% loss)")
    print(f"  3 wins:    +${3 * example_win:.2f}")
    print(f"  7 losses:  -${7 * example_loss:.2f}")
    print(f"  Net:       {Colors.GREEN}+${3 * example_win - 7 * example_loss:.2f}{Colors.RESET} per cycle")
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
