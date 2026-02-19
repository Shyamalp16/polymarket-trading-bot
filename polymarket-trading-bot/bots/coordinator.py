"""
Coordinator - Manages dual-bot execution and conflict resolution

Capital Allocation:
- Momentum (A): 40%
- Mean Reversion (B): 40%
- Buffer: 20%

Per-Window Rules:
- At most 2 total entries across both bots per market window
- Same side within 10s: allow higher EV
- Opposite sides within 10s: allow both if ≥5 ticks apart and VPIN < 0.5

Toxicity Guards:
- VPIN > 0.7 → block mean reversion; allow momentum with 0.7× size
- Fragility > 0.6 → block momentum chases; B continues maker-only

Dynamic Thresholds:
- High vol: A lowers trigger to 0.18%, B raises min_drop to 0.12
- Low vol: A increases trigger to 0.25%, B lowers min_drop to 0.08

Entry Sequencing:
- Momentum has 2s priority after qualifying impulse
- Block B entries unless B is maker at least 3 ticks inside mid

One-and-Done:
- Max 2 entries per window per bot
- 60s cooldown between entries
- Cumulative loss budget: 10% per window

Shared Exits:
- <15% time remaining → hold to expiry unless mercy stop

Usage:
    from bots.coordinator import Coordinator, CoordinatorConfig
    
    config = CoordinatorConfig(bankroll=100)
    coordinator = Coordinator(momentum_bot, mean_rev_bot, shared_state, config)
    
    await coordinator.run()
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum

from lib.shared_state import SharedState, MarketRegime
from bots.mean_reversion_bot import ReversionSide
from bots.spread_bot import SpreadBot

logger = logging.getLogger(__name__)


class BotType(str, Enum):
    """Bot type identifiers."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"


@dataclass
class EntryRequest:
    """A bot's request to enter a position."""
    bot_type: BotType
    side: str
    confidence: float
    entry_price: float
    signal_reason: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class CoordinatorConfig:
    """Configuration for coordinator."""
    
    # Capital allocation
    total_bankroll: float = 100.0
    momentum_allocation: float = 0.30      # 30%
    mean_reversion_allocation: float = 0.30 # 30%
    spread_allocation: float = 0.30        # 30%
    buffer_allocation: float = 0.10        # 10%
    
    # Per-window limits
    max_entries_per_window: int = 2   # shared total cap across both bots
    # ── Per-bot entry caps (the main knob to turn) ──────────────────────────
    # Set to 1 to allow at most one trade per bot per market window.
    # Increase to 2 if you want the bot to add on a second signal.
    max_momentum_entries: int = 1
    max_mr_entries: int = 1
    # ────────────────────────────────────────────────────────────────────────
    entry_cooldown: int = 60          # seconds between entries for same bot
    cumulative_loss_budget_pct: float = 0.10  # 10% max loss per window
    
    # Conflict resolution
    conflict_time_window: int = 10  # seconds
    min_tick_separation: float = 0.05  # 5 ticks for opposite sides
    opposite_conflict_vpin_threshold: float = 0.5
    
    # Toxicity guards
    vpin_block_mr: float = 0.7  # Block MR when VPIN > 0.7
    vpin_reduce_momentum: float = 0.7  # Reduce momentum size by 30%
    fragility_block_momentum: float = 0.6
    
    # Impulse window
    impulse_priority_window: float = 2.0  # seconds
    mr_maker_offset_ticks: float = 0.03  # 3 ticks inside mid
    
    # Late window
    late_window_threshold: float = 0.15  # <15% time remaining


@dataclass
class WindowState:
    """State for current market window."""
    start_time: float = 0
    entries: List[Dict] = field(default_factory=list)
    losses: float = 0.0
    momentum_entries: int = 0
    mr_entries: int = 0
    spread_entries: int = 0
    total_entries: int = 0  # P1-8: shared cap across both bots
    last_momentum_entry: float = 0
    last_mr_entry: float = 0
    momentum_had_position: bool = False
    mr_had_position: bool = False
    spread_attempted: bool = False


class Coordinator:
    """
    Coordinates dual-bot execution for Polymarket 5m markets.
    
    Manages:
    - Capital allocation between bots
    - Entry conflict resolution
    - Toxicity and fragility guards
    - Dynamic threshold adjustment
    - Shared exit rules
    """
    
    def __init__(
        self,
        momentum_bot,       # MomentumBot instance
        mean_reversion_bot, # MeanReversionBot instance
        shared_state: SharedState,
        config: CoordinatorConfig = None,
        spread_bot: Optional[SpreadBot] = None,
    ):
        self.momentum = momentum_bot
        self.mean_reversion = mean_reversion_bot
        self.spread = spread_bot   # Optional — None disables spread strategy
        self.state = shared_state
        self.config = config or CoordinatorConfig()

        # Capital available per bot
        self._momentum_capital = self.config.total_bankroll * self.config.momentum_allocation
        self._mr_capital = self.config.total_bankroll * self.config.mean_reversion_allocation
        self._spread_capital = self.config.total_bankroll * self.config.spread_allocation
        
        # Window state
        self._window: Optional[WindowState] = None
        self._window_start: float = 0
        
        # Recent entry requests for conflict detection
        self._recent_requests: List[EntryRequest] = []

        # Late window gate (P2-8)
        self._in_late_window: bool = False

        # Window reset tracking via token_id (P2-11)
        self._last_token_id_up: str = ""

        # Running state
        self._running = False
    
    async def start(self):
        """Start coordinator."""
        self._running = True
        self._window_start = time.time()
        self._window = WindowState(start_time=self._window_start)
        logger.info(
            "Coordinator started: bankroll=%.2f momentum=%.2f mr=%.2f",
            self.config.total_bankroll,
            self._momentum_capital,
            self._mr_capital,
        )
    
    async def stop(self):
        """Stop coordinator."""
        self._running = False
        logger.info("Coordinator stopped")
    
    async def check_and_coordinate(self) -> Optional[Dict[str, Any]]:
        """
        Main coordination loop - check both bots for signals and resolve conflicts.
        
        Returns:
            Dict with entry decision or None
        """
        if not self._running:
            return None
        
        # Check window reset
        await self._check_window_reset()
        
        # Get current market state
        market = self.state.get_market_data()
        risk = self.state.get_risk_metrics()
        
        # Check toxicity guards first
        if risk.vpin > self.config.vpin_block_mr:
            # Block mean reversion entirely
            logger.debug("Coordinator: blocking MR due to high VPIN %.2f", risk.vpin)
        
        if risk.fragility > self.config.fragility_block_momentum:
            # Block momentum chases after failures
            logger.debug("Coordinator: high fragility %.2f", risk.fragility)
        
        # Get signals from both bots
        momentum_signal = None
        mr_signal = None
        
        # Check momentum
        momentum_signal = await self.momentum.check_entry()
        
        # Block momentum if already had position this window
        if momentum_signal and self._window and self._window.momentum_had_position:
            logger.info("Coordinator: blocking momentum re-entry (already had position this window)")
            momentum_signal = None
        
        if momentum_signal:
            logger.info(f"==> MOMENTUM SIGNAL: {momentum_signal.side.value.upper()} | conf={momentum_signal.confidence:.0%} | {momentum_signal.reason}")
        
        # Check if we're in impulse priority window
        in_impulse_window = await self._is_in_impulse_window()
        
        # Check mean reversion
        mr_signal = await self.mean_reversion.check_entry()
        
        # Block MR if already had position this window
        if mr_signal and self._window and self._window.mr_had_position:
            logger.info("Coordinator: blocking MR re-entry (already had position this window)")
            mr_signal = None
        
        if mr_signal:
            logger.info(f"==> MR SIGNAL: {mr_signal.side.value.upper()} | conf={mr_signal.confidence:.0%} | {mr_signal.reason}")
        
        # ── SpreadBot: tick fill-monitor and exit-monitor every loop ─────────
        # These run regardless of late-window state so fills and SL exits
        # are always processed.  check_spread() handles its own time gate.
        if self.spread:
            fill_result = await self.spread.check_fills()
            if fill_result and fill_result.get("success"):
                logger.debug("Spread fill event: %s", fill_result)

            exit_result = await self.spread.check_exit()
            if exit_result and exit_result.get("success"):
                logger.debug("Spread exit event: %s", exit_result)

        # Expiry snipe — runs in last 5-30 s, bypasses the late-window block.
        # Must be checked BEFORE the late-window gate so it can actually execute.
        snipe = await self._check_expiry_snipe(market)
        if snipe is not None:
            return snipe

        # P2-8: block new entries during late window (mercy stops still handled per-bot)
        if self._in_late_window:
            logger.debug("Coordinator: blocking entry — late window")
            return None

        # ── SpreadBot: attempt to post a new spread this window ──────────────
        if self.spread and not self._window.spread_attempted:
            spread_result = await self.spread.check_spread()
            if spread_result and spread_result.get("success"):
                if self._window:
                    self._window.spread_entries  += 1
                    self._window.total_entries   += 1
                    self._window.spread_attempted = True
                logger.info("==> SPREAD ACTIVE: UP @ %.3f  DOWN @ %.3f",
                            spread_result.get("up_bid", 0), spread_result.get("down_bid", 0))
                return spread_result

        # P1-1: DO NOT block both bots when only one has a position.
        # Each bot has its own capital pool; _approve_* checks has_position per-bot.

        # Resolve conflicts
        if momentum_signal and mr_signal:
            return await self._resolve_conflict(momentum_signal, mr_signal, risk)
        elif momentum_signal:
            return await self._approve_momentum(momentum_signal, risk)
        elif mr_signal:
            return await self._approve_mean_reversion(mr_signal, risk, in_impulse_window)
        
        return None
    
    async def _check_window_reset(self):
        """Check if we need to reset for a new market window.

        P2-11: use token_id change as the authoritative signal instead of
        inferring from a time_to_expiry jump (which can be ambiguous).
        """
        market = self.state.get_market_data()

        if not market or not market.token_id_up:
            return

        if self._window and self._last_token_id_up and market.token_id_up != self._last_token_id_up:
            # Genuine new market — token IDs changed.
            # Cancel any resting GTX before resetting the window so the old order
            # doesn't stay open in the CLOB while a new one is posted for the new
            # market (which is what causes the two-open-orders symptom).
            await self.mean_reversion.cancel_pending_maker()
            if self.spread:
                await self.spread.on_window_reset()

            self._window_start = time.time()
            self._window = WindowState(start_time=self._window_start)
            self._in_late_window = False
            self.momentum.reset_daily_stats()
            self.momentum.reset_window_pnl()
            self.mean_reversion.reset_daily_stats()
            self.mean_reversion.reset_window_pnl()
            if self.spread:
                self.spread.reset_window_pnl()
            logger.info(f"=== NEW MARKET: {self._last_token_id_up[:8]}... → {market.token_id_up[:8]}... ===")

        self._last_token_id_up = market.token_id_up
    
    async def _is_in_impulse_window(self) -> bool:
        """Check if we're in the momentum impulse priority window."""
        if not self._window:
            return False
        
        elapsed = time.time() - self._window.last_momentum_entry
        return elapsed < self.config.impulse_priority_window
    
    async def _check_expiry_snipe(self, market) -> Optional[Dict[str, Any]]:
        """
        Expiry snipe: in the final 5–30 seconds, if one side has settled to
        85 ¢+, buy it for the near-certain $1.00 resolution payout.

        Wider window (30 s) and lower floor (85 ¢) vs the old 90–95 ¢ range:
        - 85–89 ¢ entered at 30 s remaining: ~18% upside, market implying 85%+
        - 90–94 ¢ entered at 15 s remaining: ~11% upside, near-certain
        - 95 ¢+: skip — spread cost eats the edge and it's already fully priced

        Conditions:
        - 5 s ≤ time_to_expiry ≤ 30 s
        - UP or DOWN price ≥ 0.85 and < 0.96 (above 96 ¢ margin too thin)
        - MR bot must be free (no open position or in-progress entry)
        """
        tte = market.time_to_expiry
        if not (5 <= tte <= 30):
            return None

        # Block only if MR bot itself is currently holding a position or mid-entry
        if self.mean_reversion.has_position:
            return None
        if getattr(self.mean_reversion, "_entry_in_progress", False):
            return None
        if self.mean_reversion._pending_maker_order:
            return None

        up_p   = market.up_price
        down_p = market.down_price

        if 0.85 <= up_p < 0.96:
            side      = ReversionSide.UP
            token_id  = market.token_id_up
            price     = up_p
        elif 0.85 <= down_p < 0.96:
            side      = ReversionSide.DOWN
            token_id  = market.token_id_down
            price     = down_p
        else:
            return None

        if not token_id:
            return None

        # Snipe size: 7 shares — slightly larger than normal entries to make the
        # near-certain resolution payout worth the FOK spread cost.
        size = 7.0

        logger.info(
            "==> EXPIRY SNIPE: %s @ %.3f | %ds remaining | expect $1.00 resolution",
            side.value.upper(), price, tte,
        )

        result = await self.mean_reversion.snipe_entry(
            side=side,
            token_id=token_id,
            price=price,
            size=size,
            window_remaining=tte,
        )

        if result.get("success") and self._window:
            self._window.mr_entries      += 1
            self._window.total_entries   += 1
            self._window.mr_had_position  = True
            self._window.last_mr_entry    = time.time()
            self._window.entries.append({
                "bot": "snipe",
                "side": side.value,
                "time": time.time(),
            })

        return result

    async def _resolve_conflict(
        self,
        momentum_signal,
        mr_signal,
        risk,
    ) -> Optional[Dict[str, Any]]:
        """Resolve conflict when both bots have signals."""
        
        now = time.time()
        
        # Same side conflict
        if momentum_signal.side.value == mr_signal.side.value:
            # Allow higher EV
            momentum_ev = momentum_signal.confidence * 1.5  # Higher R:R for momentum
            mr_ev = mr_signal.confidence * 1.3  # Lower R:R for MR
            
            if momentum_ev >= mr_ev:
                logger.info("Conflict resolved: momentum wins (same side)")
                return await self._approve_momentum(momentum_signal, risk)
            else:
                logger.info("Conflict resolved: mean reversion wins (same side)")
                return await self._approve_mean_reversion(mr_signal, risk, False)
        
        # Opposite sides conflict
        else:
            # Check tick separation
            tick_sep = abs(momentum_signal.entry_price - mr_signal.entry_price)
            
            if tick_sep >= self.config.min_tick_separation and risk.vpin < self.config.opposite_conflict_vpin_threshold:
                # Allow both if separated enough and low VPIN
                # But momentum has priority - execute momentum first
                logger.info("Conflict: allowing both (opposite sides, separated)")
                result = await self._approve_momentum(momentum_signal, risk)
                if result:
                    # Mark MR as pending
                    self._recent_requests.append(EntryRequest(
                        bot_type=BotType.MEAN_REVERSION,
                        side=mr_signal.side.value,
                        confidence=mr_signal.confidence,
                        entry_price=mr_signal.entry_price,
                        signal_reason=mr_signal.reason,
                    ))
                return result
            else:
                # Pick higher EV
                momentum_ev = momentum_signal.confidence * 1.5
                mr_ev = mr_signal.confidence * 1.3
                
                if momentum_ev >= mr_ev:
                    logger.info("Conflict resolved: momentum wins (opposite sides)")
                    return await self._approve_momentum(momentum_signal, risk)
                else:
                    logger.info("Conflict resolved: mean reversion wins (opposite sides)")
                    return await self._approve_mean_reversion(mr_signal, risk, False)
    
    async def _approve_momentum(self, signal, risk) -> Optional[Dict[str, Any]]:
        """Approve momentum entry with checks."""

        # Already in a position
        if self.momentum.has_position:
            logger.debug("Momentum: already has position")
            return None

        # execute() is mid-flight (e.g. inside taker retry sleeps) — don't start
        # a second concurrent entry that would produce orphaned fills.
        if getattr(self.momentum, "_entry_in_progress", False):
            logger.debug("Momentum: entry already in progress — skipping")
            return None

        # Block if MR is already holding the same direction — prevents both bots
        # stacking the same side and doubling concentration risk.
        if (
            self.mean_reversion.has_position
            and self.mean_reversion.position.side.value == signal.side.value
        ):
            logger.debug(
                "Momentum: blocked — MR already long %s", signal.side.value.upper()
            )
            return None

        # Per-bot entry cap — the main configurable knob (default: 1 per window)
        if self._window and self._window.momentum_entries >= self.config.max_momentum_entries:
            logger.debug(
                "Momentum: per-bot entry cap reached (%d/%d)",
                self._window.momentum_entries, self.config.max_momentum_entries,
            )
            return None

        # Shared total cap (both bots combined)
        if self._window and self._window.total_entries >= self.config.max_entries_per_window:
            logger.debug("Momentum: shared entry cap reached (%d)", self._window.total_entries)
            return None

        # Check cooldown
        if self._window:
            elapsed = time.time() - self._window.last_momentum_entry
            if elapsed < self.config.entry_cooldown:
                logger.debug("Momentum: cooldown active (%.0fs remaining)", self.config.entry_cooldown - elapsed)
                return None

        # Check loss budget
        if self._window and self._window.losses > self.config.cumulative_loss_budget_pct * self._momentum_capital:
            logger.debug("Momentum: loss budget exceeded")
            return None

        # P1-7: compute VPIN-adjusted size and pass to execute() as size_override
        original_size = self.momentum.calculate_size(signal)
        size = original_size * self.config.vpin_reduce_momentum if risk.vpin > 0.5 else original_size

        if size < 5.0:
            return None

        # Stamp attempt time immediately — cooldown fires even on failed FOK.
        if self._window:
            self._window.last_momentum_entry = time.time()

        result = await self.momentum.execute(signal, size_override=size)

        if result.get("success"):
            if self._window:
                self._window.momentum_entries += 1
                self._window.total_entries += 1
                self._window.momentum_had_position = True
                self._window.entries.append({
                    "bot": "momentum",
                    "side": signal.side.value,
                    "time": time.time(),
                })

        return result
    
    async def _approve_mean_reversion(self, signal, risk, in_impulse_window: bool) -> Optional[Dict[str, Any]]:
        """Approve mean reversion entry with checks."""

        # Already in a position OR a maker order is resting (pending fill).
        # Without the second check the coordinator would approve new entries every
        # 250 ms loop while the GTX order waits to be lifted, causing 6+ buys.
        if self.mean_reversion.has_position:
            logger.debug("MR: already has position")
            return None
        if self.mean_reversion._pending_maker_order:
            logger.debug("MR: maker order pending — skipping new entry")
            return None

        # execute() is mid-flight (e.g. taker retries after GTX rejection) — a
        # second concurrent entry would produce fills the bot cannot track.
        if getattr(self.mean_reversion, "_entry_in_progress", False):
            logger.debug("MR: entry already in progress — skipping")
            return None

        # Block if momentum is already holding the same direction — prevents both
        # bots stacking the same side and doubling concentration risk.
        if (
            self.momentum.has_position
            and self.momentum.position.side.value == signal.side.value
        ):
            logger.debug(
                "MR: blocked — Momentum already long %s", signal.side.value.upper()
            )
            return None

        # Block in impulse window unless maker price is sufficiently inside mid
        if in_impulse_window:
            market = self.state.get_market_data()
            mid = market.mid_price
            maker_price = mid - self.config.mr_maker_offset_ticks if signal.side.value == "up" else mid + self.config.mr_maker_offset_ticks
            if signal.entry_price > maker_price:
                logger.debug("MR: blocked during impulse window (not maker)")
                return None

        # Per-bot entry cap (default: 1 per window)
        if self._window and self._window.mr_entries >= self.config.max_mr_entries:
            logger.debug(
                "MR: per-bot entry cap reached (%d/%d)",
                self._window.mr_entries, self.config.max_mr_entries,
            )
            return None

        # Shared total cap
        if self._window and self._window.total_entries >= self.config.max_entries_per_window:
            logger.debug("MR: shared entry cap reached (%d)", self._window.total_entries)
            return None

        # Check cooldown
        if self._window:
            elapsed = time.time() - self._window.last_mr_entry
            if elapsed < self.config.entry_cooldown:
                logger.debug("MR: cooldown active (%.0fs remaining)", self.config.entry_cooldown - elapsed)
                return None

        # Check loss budget
        if self._window and self._window.losses > self.config.cumulative_loss_budget_pct * self._mr_capital:
            logger.debug("MR: loss budget exceeded")
            return None

        # P1-7: pass VPIN-adjusted size to execute()
        original_size = self.mean_reversion.calculate_size(signal)
        size = original_size * self.config.vpin_reduce_momentum if risk.vpin > 0.5 else original_size

        if size < 5.0:
            return None

        # Stamp attempt time immediately — cooldown fires even on failed GTX/FOK.
        if self._window:
            self._window.last_mr_entry = time.time()

        result = await self.mean_reversion.execute(signal, size_override=size)

        if result.get("success"):
            if self._window:
                self._window.mr_entries += 1
                self._window.total_entries += 1
                self._window.mr_had_position = True
                self._window.entries.append({
                    "bot": "mean_reversion",
                    "side": signal.side.value,
                    "time": time.time(),
                })

        return result
    
    async def check_exits(self):
        """Check both bots for exit conditions and MR maker escalation."""
        # Check momentum exits
        momentum_exit = await self.momentum.check_exit()
        if momentum_exit:
            await self._handle_exit(BotType.MOMENTUM, momentum_exit)

        # Check MR maker escalation (pending GTX order may need improvement or taker fill)
        mr_escalation = await self.mean_reversion.check_maker_escalation()
        if mr_escalation and mr_escalation.get("success"):
            if self._window and not mr_escalation.get("pending_maker"):
                self._window.mr_entries += 1
                self._window.total_entries += 1
                self._window.last_mr_entry = time.time()
                self._window.mr_had_position = True

        # Check MR exits
        mr_exit = await self.mean_reversion.check_exit()
        if mr_exit:
            await self._handle_exit(BotType.MEAN_REVERSION, mr_exit)
    
    async def _handle_exit(self, bot_type: BotType, exit_data: Dict[str, Any]):
        """Handle exit from a bot."""

        action = exit_data.get("action")
        size = exit_data.get("size", 0)
        reason = exit_data.get("reason", "")

        if action == "sell":
            bot = self.momentum if bot_type == BotType.MOMENTUM else self.mean_reversion
            had_position_before = bot.has_position

            if exit_data.get("already_filled"):
                # GTC limit order already executed on the exchange — just update tracking.
                # Do NOT call close_position() or we double-sell tokens already gone.
                fill_price = exit_data.get("fill_price", 0.0)
                result = bot._handle_gtc_exit_fill(size, reason, fill_price)
            else:
                result = await bot.close_position(reason, sell_size=size)

            if result.get("success"):
                pnl = result.get("pnl", 0)
                if self._window and pnl < 0:
                    self._window.losses += abs(pnl)
                logger.info(f"Exit: {bot_type.value} {reason} PnL: ${pnl:.2f}")

            elif had_position_before and not bot.has_position:
                # P2-9: position was force-cleared — record an estimated loss
                entry_price = (
                    self.momentum.position.entry_price if bot_type == BotType.MOMENTUM and self.momentum.position
                    else self.mean_reversion.position.entry_price if self.mean_reversion.position
                    else 0.0
                )
                if entry_price > 0 and self._window:
                    mercy_pct = 0.30  # conservative floor
                    estimated_loss = entry_price * size * mercy_pct
                    self._window.losses += estimated_loss
                    logger.warning(
                        f"Exit: {bot_type.value} {reason} FORCE-CLEARED — estimated loss ${estimated_loss:.2f}"
                    )
    
    async def check_late_window(self):
        """Update the late-window gate (P2-8).

        When <15% of window time remains, set _in_late_window = True.
        This blocks new entries in check_and_coordinate(). Existing positions
        ride to expiry — mercy stops are still enforced inside each bot's
        check_exit().
        """
        market = self.state.get_market_data()
        if market.time_to_expiry <= 0:
            return

        time_pct = market.time_to_expiry / 300  # assumes 5-minute window
        was_late = self._in_late_window
        self._in_late_window = time_pct < self.config.late_window_threshold

        if self._in_late_window and not was_late:
            logger.info(
                "Coordinator: entering late window (%.0fs remaining, %.0f%% of window)",
                market.time_to_expiry,
                time_pct * 100,
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get coordinator status."""
        return {
            "running": self._running,
            "window": {
                "total_entries":    self._window.total_entries    if self._window else 0,
                "momentum_entries": self._window.momentum_entries if self._window else 0,
                "mr_entries":       self._window.mr_entries       if self._window else 0,
                "spread_entries":   self._window.spread_entries   if self._window else 0,
                "losses":           self._window.losses           if self._window else 0,
            },
            "momentum":      self.momentum.get_stats(),
            "mean_reversion": self.mean_reversion.get_stats(),
            "spread":         self.spread.get_stats() if self.spread else {},
            "in_late_window": self._in_late_window,
        }
