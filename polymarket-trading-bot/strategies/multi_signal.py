"""
Multi-Signal Strategy - Combined Signal Trading for 5m and 15m Markets

This strategy combines four independent signal detectors and aggregates
their scores to make high-confidence entry decisions:

1. Flash Crash (Mean Reversion)
   - Detects sudden probability drops and buys the crash
   - Best in volatile/panicky markets

2. Momentum (Trend Following)
   - Detects consistent directional price movement
   - Rides trends when probability moves steadily in one direction

3. Orderbook Imbalance
   - Reads buy/sell pressure from the orderbook depth
   - Predicts short-term direction from volume asymmetry

4. Time Decay (Convergence)
   - Near expiry, binary outcomes converge to 0 or 1
   - Exploits slow-to-update prices when outcome is likely decided

Signal Combination:
    Each detector produces TradeSignal(side, score, reason).
    Scores are aggregated per side. If the best side's total score
    exceeds min_signal_score, a trade is entered.

Usage:
    from strategies.multi_signal import MultiSignalStrategy, MultiSignalConfig

    config = MultiSignalConfig(coin="ETH", market_duration=5, ...)
    strategy = MultiSignalStrategy(bot, config)
    await strategy.run()
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from lib.console import Colors, format_countdown
from strategies.base import BaseStrategy, StrategyConfig
from src.bot import TradingBot
from src.websocket_client import OrderbookSnapshot


# ---------------------------------------------------------------------------
# Signal data
# ---------------------------------------------------------------------------

@dataclass
class TradeSignal:
    """A directional trading signal from a detector."""

    side: str       # "up" or "down"
    score: float    # 0.0 to ~1.5 (can exceed 1.0 for strong signals)
    reason: str     # Human-readable reason


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MultiSignalConfig(StrategyConfig):
    """
    Configuration for the multi-signal strategy.

    All thresholds are tuned separately for 5m vs 15m markets.
    See apps/run_5m.py and apps/run_15m.py for recommended defaults.
    """

    # --- Flash crash signal ---
    flash_crash_enabled: bool = True
    drop_threshold: float = 0.20        # Absolute probability drop to trigger

    # --- Momentum signal ---
    momentum_enabled: bool = True
    momentum_window: int = 30           # Seconds to measure trend
    momentum_threshold: float = 0.08    # Min absolute price move
    momentum_min_ticks: int = 4         # Min data points in window
    momentum_consistency: float = 0.65  # Min % of ticks in same direction

    # --- Orderbook imbalance signal ---
    imbalance_enabled: bool = True
    imbalance_ratio_threshold: float = 2.5  # Bid/ask volume ratio to trigger
    imbalance_depth: int = 5                # Order book levels to consider

    # --- Time decay / convergence signal ---
    time_decay_enabled: bool = True
    time_decay_threshold_pct: float = 0.20  # Activate when this % of time remains
    time_decay_min_divergence: float = 0.15 # Min price distance from 0.50

    # --- Signal combination ---
    min_signal_score: float = 0.7       # Min combined score to enter
    signal_cooldown: float = 12.0       # Seconds between entries

    # --- Safety ---
    min_time_remaining: int = 60        # Don't open positions with less time (seconds)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class MultiSignalStrategy(BaseStrategy):
    """
    Multi-signal trading strategy for Polymarket Up/Down markets.

    Combines flash crash, momentum, orderbook imbalance, and time decay
    signals to make high-confidence trading decisions. Designed for both
    5-minute and 15-minute market durations with tunable parameters.
    """

    def __init__(self, bot: TradingBot, config: MultiSignalConfig):
        """Initialize multi-signal strategy."""
        super().__init__(bot, config)
        self.signal_config = config

        # Update price tracker with flash crash threshold
        self.prices.drop_threshold = config.drop_threshold

        # Signal state
        self._last_signal_time: float = 0.0
        self._last_signals: List[TradeSignal] = []
        self._total_signals_fired: int = 0

    # ------------------------------------------------------------------
    # BaseStrategy hooks
    # ------------------------------------------------------------------

    async def on_book_update(self, snapshot: OrderbookSnapshot) -> None:
        """Handle orderbook update - price recording is done in base class."""
        pass

    async def on_tick(self, prices: Dict[str, float]) -> None:
        """
        Main tick handler - collect signals and decide whether to trade.

        Called on every iteration of the strategy loop (~100ms).
        """
        if not self.positions.can_open_position:
            return

        # Cooldown check
        now = time.time()
        if now - self._last_signal_time < self.signal_config.signal_cooldown:
            return

        # Safety: don't enter too close to market expiry
        if self._is_too_close_to_expiry():
            return

        # Collect signals from all enabled detectors
        signals: List[TradeSignal] = []

        if self.signal_config.flash_crash_enabled:
            signals.extend(self._check_flash_crash(prices))

        if self.signal_config.momentum_enabled:
            signals.extend(self._check_momentum(prices))

        if self.signal_config.imbalance_enabled:
            signals.extend(self._check_orderbook_imbalance())

        if self.signal_config.time_decay_enabled:
            signals.extend(self._check_time_decay(prices))

        self._last_signals = signals

        if not signals:
            return

        # Aggregate scores per side
        side_scores: Dict[str, float] = {"up": 0.0, "down": 0.0}
        side_reasons: Dict[str, List[str]] = {"up": [], "down": []}

        for signal in signals:
            side_scores[signal.side] += signal.score
            side_reasons[signal.side].append(signal.reason)

        # Pick best side
        best_side = max(side_scores, key=lambda s: side_scores[s])
        best_score = side_scores[best_side]

        if best_score >= self.signal_config.min_signal_score:
            reasons = ", ".join(side_reasons[best_side])
            current_price = prices.get(best_side, 0)
            if current_price > 0:
                success = await self.execute_buy(best_side, current_price)
                if success:
                    # Only log a signal event when it actually resulted
                    # in an executed BUY attempt/position open.
                    self.log(
                        f"SIGNAL: {best_side.upper()} score={best_score:.2f} [{reasons}]",
                        "trade",
                    )
                    self._last_signal_time = now
                    self._total_signals_fired += 1

    # ------------------------------------------------------------------
    # Signal detectors
    # ------------------------------------------------------------------

    def _check_flash_crash(self, prices: Dict[str, float]) -> List[TradeSignal]:
        """
        Flash Crash Detector (Mean Reversion).

        Triggers when a side's probability drops rapidly within the
        lookback window. Score scales with drop magnitude.
        """
        signals: List[TradeSignal] = []
        event = self.prices.detect_flash_crash()

        if event:
            # Score: proportional to drop / threshold, capped at 1.5
            score = min(event.drop / self.signal_config.drop_threshold, 1.5)
            signals.append(TradeSignal(
                side=event.side,
                score=score,
                reason=f"crash:{event.drop:.2f}",
            ))

        return signals

    def _check_momentum(self, prices: Dict[str, float]) -> List[TradeSignal]:
        """
        Momentum Detector (Trend Following).

        Checks each side for consistent directional price movement
        within the momentum window. Only triggers when both the total
        move exceeds the threshold AND the consistency ratio is met.
        """
        signals: List[TradeSignal] = []
        now = time.time()
        cutoff = now - self.signal_config.momentum_window

        for side in ["up", "down"]:
            history = self.prices.get_history(side)
            recent = [p for p in history if p.timestamp >= cutoff]

            if len(recent) < self.signal_config.momentum_min_ticks:
                continue

            # Total price move over window
            total_move = recent[-1].price - recent[0].price
            if abs(total_move) < self.signal_config.momentum_threshold:
                continue

            # Count directional ticks for consistency check
            up_ticks = 0
            down_ticks = 0
            for i in range(1, len(recent)):
                diff = recent[i].price - recent[i - 1].price
                if diff > 0.001:
                    up_ticks += 1
                elif diff < -0.001:
                    down_ticks += 1

            total_ticks = up_ticks + down_ticks
            if total_ticks == 0:
                continue

            if total_move > 0:
                # Price rising on this side -> bullish for this side
                consistency = up_ticks / total_ticks
                if consistency >= self.signal_config.momentum_consistency:
                    score = min(
                        total_move / self.signal_config.momentum_threshold, 1.0
                    ) * 0.7
                    signals.append(TradeSignal(
                        side=side,
                        score=score,
                        reason=f"mom:+{total_move:.3f}",
                    ))
            else:
                # Price falling on this side -> bullish for the OTHER side
                consistency = down_ticks / total_ticks
                if consistency >= self.signal_config.momentum_consistency:
                    other_side = "down" if side == "up" else "up"
                    score = min(
                        abs(total_move) / self.signal_config.momentum_threshold, 1.0
                    ) * 0.7
                    signals.append(TradeSignal(
                        side=other_side,
                        score=score,
                        reason=f"mom:{side}-{abs(total_move):.3f}",
                    ))

        return signals

    def _check_orderbook_imbalance(self) -> List[TradeSignal]:
        """
        Orderbook Imbalance Detector.

        Compares total bid volume vs ask volume in the top N levels.
        A heavy bid side suggests buy pressure and likely price increase.
        """
        signals: List[TradeSignal] = []
        depth = self.signal_config.imbalance_depth

        for side in ["up", "down"]:
            ob = self.market.get_orderbook(side)
            if not ob or not ob.bids or not ob.asks:
                continue

            total_bid_size = sum(level.size for level in ob.bids[:depth])
            total_ask_size = sum(level.size for level in ob.asks[:depth])

            if total_ask_size <= 0:
                continue

            ratio = total_bid_size / total_ask_size

            if ratio >= self.signal_config.imbalance_ratio_threshold:
                # Strong buy pressure -> bullish for this side
                normalized = (ratio - 1.0) / (
                    self.signal_config.imbalance_ratio_threshold - 1.0
                )
                score = min(normalized, 1.0) * 0.5
                signals.append(TradeSignal(
                    side=side,
                    score=score,
                    reason=f"imbal:{ratio:.1f}x",
                ))

        return signals

    def _check_time_decay(self, prices: Dict[str, float]) -> List[TradeSignal]:
        """
        Time Decay / Convergence Detector.

        Near market expiry, binary outcomes should converge toward 0 or 1.
        If the price is already leaning one way (e.g., UP=0.70) with little
        time left, it should converge further. We buy the leading side.

        Score increases as expiry approaches and as price divergence grows.
        """
        signals: List[TradeSignal] = []
        market = self.current_market
        if not market:
            return signals

        mins, secs = market.get_countdown()
        if mins < 0:
            return signals

        total_secs_remaining = mins * 60 + secs
        market_duration_secs = self.signal_config.market_duration * 60

        if market_duration_secs <= 0:
            return signals

        time_pct_remaining = total_secs_remaining / market_duration_secs

        # Only activate when time remaining falls below threshold
        if time_pct_remaining > self.signal_config.time_decay_threshold_pct:
            return signals

        min_divergence = self.signal_config.time_decay_min_divergence

        for side in ["up", "down"]:
            price = prices.get(side, 0)
            if price <= 0:
                continue

            divergence = price - 0.5

            if divergence >= min_divergence:
                # Price already favoring this side -> should converge to 1.0
                # Score increases as time runs out and divergence grows
                time_urgency = 1.0 - time_pct_remaining  # 0.8 -> 1.0
                score = min(divergence * 2.0 * time_urgency, 1.0) * 0.8
                signals.append(TradeSignal(
                    side=side,
                    score=score,
                    reason=f"decay:{total_secs_remaining}s@{price:.2f}",
                ))

        return signals

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_too_close_to_expiry(self) -> bool:
        """Check if market is too close to expiry for new entries."""
        market = self.current_market
        if not market:
            return True

        mins, secs = market.get_countdown()
        if mins < 0:
            return True

        total_secs = mins * 60 + secs
        return total_secs < self.signal_config.min_time_remaining

    def _get_countdown_str(self) -> str:
        """Get formatted countdown string."""
        market = self.current_market
        if not market:
            return "--:--"
        mins, secs = market.get_countdown()
        return format_countdown(mins, secs)

    def on_market_change(self, old_slug: str, new_slug: str) -> None:
        """Handle market change - reset signal state."""
        self.prices.clear()
        self._last_signal_time = 0.0
        self._last_signals = []

    # ------------------------------------------------------------------
    # TUI Rendering
    # ------------------------------------------------------------------

    def render_status(self, prices: Dict[str, float]) -> None:
        """Render TUI status display with signal information."""
        lines: List[str] = []
        cfg = self.signal_config
        duration_label = f"{cfg.market_duration}m"

        # Header
        ws_status = (
            f"{Colors.GREEN}WS{Colors.RESET}"
            if self.is_connected
            else f"{Colors.RED}REST{Colors.RESET}"
        )
        countdown = self._get_countdown_str()
        stats = self.positions.get_stats()

        lines.append(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        lines.append(
            f"{Colors.CYAN}[{cfg.coin} {duration_label}]{Colors.RESET} "
            f"[{ws_status}] "
            f"Ends: {countdown} | "
            f"Trades: {stats['trades_closed']} | "
            f"PnL: ${stats['total_pnl']:+.2f} | "
            f"Signals: {self._total_signals_fired}"
        )
        lines.append(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")

        # Orderbook display
        up_ob = self.market.get_orderbook("up")
        down_ob = self.market.get_orderbook("down")

        lines.append(
            f"{Colors.GREEN}{'UP':^39}{Colors.RESET}|"
            f"{Colors.RED}{'DOWN':^39}{Colors.RESET}"
        )
        lines.append(
            f"{'Bid':>9} {'Size':>9} | {'Ask':>9} {'Size':>9}|"
            f"{'Bid':>9} {'Size':>9} | {'Ask':>9} {'Size':>9}"
        )
        lines.append("-" * 80)

        up_bids = up_ob.bids[:5] if up_ob else []
        up_asks = up_ob.asks[:5] if up_ob else []
        down_bids = down_ob.bids[:5] if down_ob else []
        down_asks = down_ob.asks[:5] if down_ob else []

        for i in range(5):
            ub = (
                f"{up_bids[i].price:>9.4f} {up_bids[i].size:>9.1f}"
                if i < len(up_bids) else f"{'--':>9} {'--':>9}"
            )
            ua = (
                f"{up_asks[i].price:>9.4f} {up_asks[i].size:>9.1f}"
                if i < len(up_asks) else f"{'--':>9} {'--':>9}"
            )
            db = (
                f"{down_bids[i].price:>9.4f} {down_bids[i].size:>9.1f}"
                if i < len(down_bids) else f"{'--':>9} {'--':>9}"
            )
            da = (
                f"{down_asks[i].price:>9.4f} {down_asks[i].size:>9.1f}"
                if i < len(down_asks) else f"{'--':>9} {'--':>9}"
            )
            lines.append(f"{ub} | {ua}|{db} | {da}")

        lines.append("-" * 80)

        # Price summary
        up_mid = up_ob.mid_price if up_ob else prices.get("up", 0)
        down_mid = down_ob.mid_price if down_ob else prices.get("down", 0)
        up_spread = self.market.get_spread("up")
        down_spread = self.market.get_spread("down")

        lines.append(
            f"Mid: {Colors.GREEN}{up_mid:.4f}{Colors.RESET}  "
            f"Spread: {up_spread:.4f}           |"
            f"Mid: {Colors.RED}{down_mid:.4f}{Colors.RESET}  "
            f"Spread: {down_spread:.4f}"
        )

        # Active signals
        lines.append(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        enabled = []
        if cfg.flash_crash_enabled:
            enabled.append(f"Crash(>{cfg.drop_threshold:.2f})")
        if cfg.momentum_enabled:
            enabled.append(f"Mom({cfg.momentum_window}s)")
        if cfg.imbalance_enabled:
            enabled.append(f"Imbal(>{cfg.imbalance_ratio_threshold:.1f}x)")
        if cfg.time_decay_enabled:
            enabled.append(f"Decay(<{cfg.time_decay_threshold_pct:.0%})")

        lines.append(
            f"{Colors.BOLD}Detectors:{Colors.RESET} {' | '.join(enabled)} | "
            f"Entry: >={cfg.min_signal_score:.1f} | "
            f"TP: +{cfg.take_profit:.0%} | SL: -{cfg.stop_loss:.0%}"
        )

        # Current signals
        if self._last_signals:
            sig_parts = []
            for sig in self._last_signals:
                color = Colors.GREEN if sig.side == "up" else Colors.RED
                sig_parts.append(
                    f"{color}{sig.side.upper()}{Colors.RESET} "
                    f"{sig.score:.2f} ({sig.reason})"
                )
            lines.append(f"{Colors.BOLD}Live Signals:{Colors.RESET} {' | '.join(sig_parts)}")
        else:
            lines.append(
                f"{Colors.BOLD}Live Signals:{Colors.RESET} "
                f"{Colors.DIM}(waiting for signals...){Colors.RESET}"
            )

        # Open Orders section
        lines.append(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        lines.append(f"{Colors.BOLD}Open Orders:{Colors.RESET}")
        if self.open_orders:
            for order in self.open_orders[:5]:
                side = order.get("side", "?")
                price = float(order.get("price", 0))
                size = float(order.get("original_size", order.get("size", 0)))
                filled = float(order.get("size_matched", 0))
                order_id = order.get("id", "")[:8]
                token = order.get("asset_id", "")
                token_side = (
                    "UP" if token == self.token_ids.get("up")
                    else "DOWN" if token == self.token_ids.get("down")
                    else "?"
                )
                color = Colors.GREEN if side == "BUY" else Colors.RED
                lines.append(
                    f"  {color}{side:4}{Colors.RESET} {token_side:4} "
                    f"@ {price:.4f} Size: {size:.1f} "
                    f"Filled: {filled:.1f} ID: {order_id}..."
                )
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
                    f"Entry: {pos.entry_price:.4f} | "
                    f"Current: {current:.4f} | "
                    f"Size: ${pos.size:.2f} | "
                    f"PnL: {color}${pnl:+.2f} ({pnl_pct:+.1f}%){Colors.RESET} | "
                    f"Hold: {hold_time:.0f}s"
                )
                lines.append(
                    f"       TP: {pos.take_profit_price:.4f} "
                    f"(+{pos.take_profit_delta:.0%}) | "
                    f"SL: {pos.stop_loss_price:.4f} "
                    f"(-{pos.stop_loss_delta:.0%})"
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
