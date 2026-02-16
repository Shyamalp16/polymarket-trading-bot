"""
Flash Crash Strategy - Volatility Trading for 15-Minute Markets

This strategy monitors 15-minute Up/Down markets for sudden probability drops
and executes trades when probability crashes by a threshold within a lookback window.

Strategy Logic:
1. Auto-discover current 15-minute market for selected coin
2. Monitor orderbook prices in real-time via WebSocket
3. When either "Up" or "Down" probability drops by threshold:
   - Market buy the crashed side
4. Exit conditions:
   - Take profit: configurable (default +10 cents)
   - Stop loss: configurable (default -5 cents)

Usage:
    from strategies.flash_crash import FlashCrashStrategy, FlashCrashConfig

    strategy = FlashCrashStrategy(bot, config)
    await strategy.run()
"""

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from lib.console import Colors, format_countdown
from strategies.base import BaseStrategy, StrategyConfig
from src.bot import TradingBot
from src.websocket_client import OrderbookSnapshot


@dataclass
class AdaptiveCrashEvent:
    """Adaptive flash crash event payload."""

    side: str
    old_price: float
    new_price: float
    drop: float
    timestamp: float
    z_score: float


@dataclass
class FlashCrashConfig(StrategyConfig):
    """Flash crash strategy configuration."""

    drop_threshold: float = 0.30  # Absolute probability drop
    adaptive_detection_enabled: bool = True
    adaptive_window_seconds: float = 5.0
    adaptive_z_threshold: float = 2.5
    adaptive_min_drop: float = 0.15
    adaptive_min_returns: int = 12

    # Spot-aware cause classification (optional provider)
    spot_filter_enabled: bool = True
    spot_informed_threshold_pct: float = 0.30
    require_spot_data_for_entry: bool = False
    spot_change_provider: Optional[Callable[[float], Optional[float]]] = None

    # Time-gated entry and dynamic TP/SL assignment by time-to-expiry
    skip_entry_below_seconds: int = 120
    tp_far: float = 0.12      # > 10m
    sl_far: float = 0.06
    tp_mid: float = 0.08      # 5m - 10m
    sl_mid: float = 0.05
    tp_near: float = 0.05     # 2m - 5m
    sl_near: float = 0.04


class FlashCrashStrategy(BaseStrategy):
    """
    Flash Crash Trading Strategy.

    Monitors 15-minute markets for sudden price drops and trades
    the volatility with defined take-profit and stop-loss levels.
    """

    def __init__(self, bot: TradingBot, config: FlashCrashConfig):
        """Initialize flash crash strategy."""
        super().__init__(bot, config)
        self.flash_config = config

        # Update price tracker with our threshold
        self.prices.drop_threshold = config.drop_threshold

    async def on_book_update(self, snapshot: OrderbookSnapshot) -> None:
        """Handle orderbook update - check for flash crashes."""
        pass  # Price recording is done in base class

    async def on_tick(self, prices: Dict[str, float]) -> None:
        """Check for flash crash on each tick."""
        if not self.positions.can_open_position:
            return

        # Detect flash crash (adaptive by default, legacy fallback available)
        event = (
            self._detect_adaptive_flash_crash()
            if self.flash_config.adaptive_detection_enabled
            else self.prices.detect_flash_crash()
        )
        if event:
            if not self._apply_entry_time_gate():
                return

            informed, spot_change = self._is_informed_flow(event.side)
            if informed:
                self.log(
                    f"SUPPRESSED crash on {event.side.upper()}: "
                    f"spot move={spot_change:+.2f}% (informed flow)",
                    "warning",
                )
                return
            if (
                self.flash_config.spot_filter_enabled
                and self.flash_config.require_spot_data_for_entry
                and spot_change is None
            ):
                self.log(
                    "SUPPRESSED crash: spot filter enabled but no spot data available",
                    "warning",
                )
                return

            self.log(
                f"FLASH CRASH: {event.side.upper()} "
                f"drop {event.drop:.2f} ({event.old_price:.2f} -> {event.new_price:.2f})",
                "trade"
            )
            current_price = prices.get(event.side, 0)
            if current_price > 0:
                # TP/SL are set based on time remaining immediately before entry.
                self._assign_entry_risk_profile()
                await self.execute_buy(event.side, current_price)

    def _detect_adaptive_flash_crash(self) -> Optional[AdaptiveCrashEvent]:
        """
        Detect crash using z-score over a short window with an absolute floor.

        This avoids fixed-threshold rigidity across different price regimes.
        """
        now = time.time()
        window = self.flash_config.adaptive_window_seconds

        for side in ["up", "down"]:
            history = self.prices.get_history(side)
            if len(history) < (self.flash_config.adaptive_min_returns + 1):
                continue

            recent = [p for p in history if p.timestamp >= (now - window)]
            if len(recent) < 2:
                continue

            current = recent[-1].price
            local_peak = max(p.price for p in recent)
            drop = local_peak - current
            if drop < self.flash_config.adaptive_min_drop:
                continue

            # Build return distribution from the latest observations.
            lookback = history[-(self.flash_config.adaptive_min_returns + 1):]
            returns = [
                lookback[i].price - lookback[i - 1].price
                for i in range(1, len(lookback))
            ]
            if len(returns) < self.flash_config.adaptive_min_returns:
                continue

            mu = sum(returns) / len(returns)
            var = sum((r - mu) ** 2 for r in returns) / max(1, len(returns) - 1)
            sigma = math.sqrt(var) if var > 0 else 0.0
            if sigma <= 1e-9:
                continue

            window_return = recent[-1].price - recent[0].price
            steps = max(1, len(recent) - 1)
            expected = mu * steps
            expected_sigma = sigma * math.sqrt(steps)
            z_score = (window_return - expected) / max(expected_sigma, 1e-9)

            if z_score <= -self.flash_config.adaptive_z_threshold:
                return AdaptiveCrashEvent(
                    side=side,
                    old_price=local_peak,
                    new_price=current,
                    drop=drop,
                    timestamp=now,
                    z_score=z_score,
                )

        return None

    def _is_informed_flow(self, crashed_side: str) -> tuple[bool, Optional[float]]:
        """
        Classify event using optional BTC spot move input.

        If spot moves significantly in the direction that explains the
        crashed side move, treat it as informed and suppress mean reversion.
        """
        if not self.flash_config.spot_filter_enabled:
            return (False, None)

        spot_change = self._get_spot_change_pct(self.flash_config.adaptive_window_seconds)
        if spot_change is None:
            return (False, None)

        if abs(spot_change) < self.flash_config.spot_informed_threshold_pct:
            return (False, spot_change)

        aligned = (crashed_side == "up" and spot_change < 0) or (
            crashed_side == "down" and spot_change > 0
        )
        return (aligned, spot_change)

    def _get_spot_change_pct(self, window_seconds: float) -> Optional[float]:
        """Get BTC spot percentage move over a window from optional provider."""
        provider = self.flash_config.spot_change_provider
        if not provider:
            return None
        try:
            return provider(window_seconds)
        except Exception:
            return None

    def _seconds_remaining(self) -> Optional[int]:
        market = self.current_market
        if not market:
            return None
        mins, secs = market.get_countdown()
        if mins < 0:
            return None
        return mins * 60 + secs

    def _apply_entry_time_gate(self) -> bool:
        """Skip new entries when there is not enough time for reversion."""
        remaining = self._seconds_remaining()
        if remaining is None:
            return False
        if remaining < self.flash_config.skip_entry_below_seconds:
            self.log(
                f"SKIP entry: {remaining}s remaining (<{self.flash_config.skip_entry_below_seconds}s)",
                "warning",
            )
            return False
        return True

    def _assign_entry_risk_profile(self) -> None:
        """
        Set TP/SL by time-to-expiry before opening a new position.

        >10m: wider target and stop
        5-10m: moderate
        2-5m: tighter convergence profile
        """
        remaining = self._seconds_remaining()
        if remaining is None:
            return

        if remaining > 600:
            tp, sl = self.flash_config.tp_far, self.flash_config.sl_far
        elif remaining > 300:
            tp, sl = self.flash_config.tp_mid, self.flash_config.sl_mid
        else:
            tp, sl = self.flash_config.tp_near, self.flash_config.sl_near

        self.positions.take_profit = tp
        self.positions.stop_loss = sl

    def render_status(self, prices: Dict[str, float]) -> None:
        """Render TUI status display."""
        lines = []

        # Header
        ws_status = f"{Colors.GREEN}WS{Colors.RESET}" if self.is_connected else f"{Colors.RED}REST{Colors.RESET}"
        countdown = self._get_countdown_str()
        stats = self.positions.get_stats()

        lines.append(f"{Colors.BOLD}{'='*80}{Colors.RESET}")
        lines.append(
            f"{Colors.CYAN}[{self.config.coin}]{Colors.RESET} [{ws_status}] "
            f"Ends: {countdown} | Trades: {stats['trades_closed']} | PnL: ${stats['total_pnl']:+.2f}"
        )
        lines.append(f"{Colors.BOLD}{'='*80}{Colors.RESET}")

        # Orderbook display
        up_ob = self.market.get_orderbook("up")
        down_ob = self.market.get_orderbook("down")

        lines.append(f"{Colors.GREEN}{'UP':^39}{Colors.RESET}|{Colors.RED}{'DOWN':^39}{Colors.RESET}")
        lines.append(f"{'Bid':>9} {'Size':>9} | {'Ask':>9} {'Size':>9}|{'Bid':>9} {'Size':>9} | {'Ask':>9} {'Size':>9}")
        lines.append("-" * 80)

        # Get 5 levels
        up_bids = up_ob.bids[:5] if up_ob else []
        up_asks = up_ob.asks[:5] if up_ob else []
        down_bids = down_ob.bids[:5] if down_ob else []
        down_asks = down_ob.asks[:5] if down_ob else []

        for i in range(5):
            up_bid = f"{up_bids[i].price:>9.4f} {up_bids[i].size:>9.1f}" if i < len(up_bids) else f"{'--':>9} {'--':>9}"
            up_ask = f"{up_asks[i].price:>9.4f} {up_asks[i].size:>9.1f}" if i < len(up_asks) else f"{'--':>9} {'--':>9}"
            down_bid = f"{down_bids[i].price:>9.4f} {down_bids[i].size:>9.1f}" if i < len(down_bids) else f"{'--':>9} {'--':>9}"
            down_ask = f"{down_asks[i].price:>9.4f} {down_asks[i].size:>9.1f}" if i < len(down_asks) else f"{'--':>9} {'--':>9}"
            lines.append(f"{up_bid} | {up_ask}|{down_bid} | {down_ask}")

        lines.append("-" * 80)

        # Summary
        up_mid = up_ob.mid_price if up_ob else prices.get("up", 0)
        down_mid = down_ob.mid_price if down_ob else prices.get("down", 0)
        up_spread = self.market.get_spread("up")
        down_spread = self.market.get_spread("down")

        lines.append(
            f"Mid: {Colors.GREEN}{up_mid:.4f}{Colors.RESET}  Spread: {up_spread:.4f}           |"
            f"Mid: {Colors.RED}{down_mid:.4f}{Colors.RESET}  Spread: {down_spread:.4f}"
        )

        # History info
        up_history = self.prices.get_history_count("up")
        down_history = self.prices.get_history_count("down")
        lines.append(
            f"History: UP={up_history}/100 DOWN={down_history}/100 | "
            f"Detector: {'adaptive-z' if self.flash_config.adaptive_detection_enabled else 'fixed'} | "
            f"Window: {self.flash_config.adaptive_window_seconds:.1f}s | "
            f"MinDrop: {self.flash_config.adaptive_min_drop:.2f}"
        )
        lines.append(
            f"Risk profile: TP={self.positions.take_profit:.0%} SL={self.positions.stop_loss:.0%} | "
            f"Entry gate: >= {self.flash_config.skip_entry_below_seconds}s remaining"
        )

        lines.append(f"{Colors.BOLD}{'='*80}{Colors.RESET}")

        # Open Orders section
        lines.append(f"{Colors.BOLD}Open Orders:{Colors.RESET}")
        if self.open_orders:
            for order in self.open_orders[:5]:  # Show max 5 orders
                side = order.get("side", "?")
                price = float(order.get("price", 0))
                size = float(order.get("original_size", order.get("size", 0)))
                filled = float(order.get("size_matched", 0))
                order_id = order.get("id", "")[:8]
                token = order.get("asset_id", "")
                # Determine if UP or DOWN
                token_side = "UP" if token == self.token_ids.get("up") else "DOWN" if token == self.token_ids.get("down") else "?"
                color = Colors.GREEN if side == "BUY" else Colors.RED
                lines.append(f"  {color}{side:4}{Colors.RESET} {token_side:4} @ {price:.4f} Size: {size:.1f} Filled: {filled:.1f} ID: {order_id}...")
        else:
            lines.append(f"  {Colors.CYAN}(no open orders){Colors.RESET}")

        # Positions
        lines.append(f"{Colors.BOLD}Positions:{Colors.RESET}")
        all_positions = self.positions.get_all_positions()
        if all_positions:
            for pos in all_positions:
                current = prices.get(pos.side, 0)
                pnl = pos.get_pnl(current)
                pnl_pct = pos.get_pnl_percent(current)
                hold_time = pos.get_hold_time()
                color = Colors.GREEN if pnl >= 0 else Colors.RED

                lines.append(
                    f"  {Colors.BOLD}{pos.side.upper():4}{Colors.RESET} "
                    f"Entry: {pos.entry_price:.4f} | Current: {current:.4f} | "
                    f"Size: ${pos.size:.2f} | PnL: {color}${pnl:+.2f} ({pnl_pct:+.1f}%){Colors.RESET} | "
                    f"Hold: {hold_time:.0f}s"
                )
                lines.append(
                    f"       TP: {pos.take_profit_price:.4f} (+${pos.take_profit_delta:.2f}) | "
                    f"SL: {pos.stop_loss_price:.4f} (-${pos.stop_loss_delta:.2f})"
                )
        else:
            lines.append(f"  {Colors.CYAN}(no open positions){Colors.RESET}")

        # Recent logs
        if self._log_buffer.messages:
            lines.append("-" * 80)
            lines.append(f"{Colors.BOLD}Recent Events:{Colors.RESET}")
            for msg in self._log_buffer.get_messages():
                lines.append(f"  {msg}")

        # Render
        output = "\033[H\033[J" + "\n".join(lines)
        print(output, flush=True)

    def _get_countdown_str(self) -> str:
        """Get formatted countdown string."""
        market = self.current_market
        if not market:
            return "--:--"

        mins, secs = market.get_countdown()
        return format_countdown(mins, secs)

    def on_market_change(self, old_slug: str, new_slug: str) -> None:
        """Handle market change - clear price history."""
        self.prices.clear()
