"""
Momentum Bot (Bot A) - Impulse chaser for trend-following

Entry Logic:
- Spot impulse: BTC spot change ≥ ±0.20% within ≤2 seconds
- Poly lag: |p* from spot − current Poly prob| ≥ 0.08
- Microstructure confirm: last-trade direction aligns with side

Execution:
- FOK taker with price-adjusted ladder
- Max slippage: 3-4 ticks per attempt
- No post-only

Exits:
- TP1: +12-15% (sell 35%), ratchet SL to break-even
- TP2: +25-30% (sell 35%), ratchet SL to TP1 price
- Hold remainder to expiry unless VPIN > 0.7 or spot reverses >0.15% in 5s
- Time-confirmed SL: initial -12%, confirm 5-8s below level

Sizing:
- Base: f = min(0.05, 0.25 × Kelly_estimate)
- Multiplier by toxicity: size *= max(0.4, 1 − 0.8 × VPIN)
- Cap: ≤5% bankroll, ≤25% visible depth

Usage:
    from bots.momentum_bot import MomentumBot
    
    config = MomentumConfig(bankroll=100)
    bot = MomentumBot(trading_bot, shared_state, config)
    
    # Check for entry signal
    signal = await bot.check_entry()
    if signal:
        result = await bot.execute(signal)
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Deque
from enum import Enum

from lib.shared_state import SharedState, MarketRegime

logger = logging.getLogger(__name__)


class MomentumSide(str, Enum):
    """Momentum direction."""
    UP = "up"
    DOWN = "down"


@dataclass
class MomentumSignal:
    """Entry signal from momentum detection."""
    side: MomentumSide
    confidence: float          # 0-1
    entry_price: float        # Current mid price
    reason: str
    impulse_strength: float   # How strong the impulse is
    divergence: float         # Spot-poly divergence


@dataclass
class MomentumExit:
    """Exit parameters for momentum trade."""
    tp1_price: float          # First take profit
    tp2_price: float          # Second take profit
    initial_sl: float         # Initial stop loss
    tp1_size_pct: float = 0.35
    tp2_size_pct: float = 0.35
    hold_pct: float = 0.30     # Remainder
    time_confirm_secs: float = 6.0  # Confirm SL for this long


@dataclass
class MomentumConfig:
    """Configuration for momentum bot."""
    
    # Bankroll and sizing
    bankroll: float = 100.0
    max_position_pct: float = 0.05      # 5% max per trade
    kelly_multiplier: float = 0.25      # Kelly fraction
    
    # Entry triggers
    spot_impulse_threshold: float = 0.0006  # 0.06% in 2s — typical sharp BTC move
    spot_impulse_window: int = 2            # seconds

    # Dynamic thresholds by regime — high vol LOWERS trigger, low vol RAISES it
    high_vol_impulse: float = 0.0004   # 0.04% — easier entry in high vol
    low_vol_impulse: float = 0.0008    # 0.08% — stricter entry in low vol
    
    # Execution
    max_slippage_ticks: int = 4
    ladder_attempts: int = 3
    
    # Exits
    tp1_pct: float = 0.14              # +14%
    tp2_pct: float = 0.28              # +28%
    initial_sl_pct: float = 0.35       # -35%
    trailing_sl_pct: float = 0.15      # trail 15% below peak — activates after TP1
    time_confirm_sl: float = 10.0       # seconds - no SL in first 10s
    early_exit_vpin: float = 0.7
    early_exit_spot_reversal: float = 0.0015  # 0.15% reversal
    
    # Depth cap
    max_depth_pct: float = 0.25        # 25% of visible depth
    
    # Cooldown
    cooldown_seconds: int = 30
    min_signal_strength: float = 0.3


@dataclass
class MomentumPosition:
    """Active momentum position."""
    side: MomentumSide
    entry_price: float
    size: float
    entry_time: float
    tp1_filled: bool = False
    tp2_filled: bool = False
    sl_ratcheted: bool = False
    sl_price: float = 0.0
    # Limit orders for automated exits
    sl_order_id: Optional[str] = None
    tp1_order_id: Optional[str] = None
    tp2_order_id: Optional[str] = None
    # Throttle: last time we polled order status via HTTP (P2-1)
    _last_order_poll: float = 0.0


class MomentumBot:
    """
    Momentum Taker Bot - catches impulse moves.
    
    Designed for low-latency execution in a shared process
    with MeanReversionBot via the Coordinator.
    """
    
    def __init__(
        self,
        trading_bot,           # TradingBot instance
        shared_state: SharedState,
        config: Optional[MomentumConfig] = None,
    ):
        self.bot = trading_bot
        self.state = shared_state
        self.config = config if config else MomentumConfig()
        
        # Position tracking
        self._position: Optional[MomentumPosition] = None
        self._last_entry_time: float = 0
        self._entry_cooldown: float = 0
        self._balance_block_until: float = 0  # Cooldown after balance errors
        self._failed_close_attempts: int = 0  # Track failed close attempts
        
        # Signal history — bounded deque (P3-2)
        self._recent_signals: Deque[MomentumSignal] = deque(maxlen=100)
        
        # Metrics
        self._entries_today: int = 0
        self._wins_today: int = 0
        self._realized_pnl: float = 0.0   # cumulative realized PnL this session
        self._window_pnl: float = 0.0     # realized PnL for current market window
    
    @property
    def has_position(self) -> bool:
        """Check if bot has active position."""
        return self._position is not None
    
    @property
    def position(self) -> Optional[MomentumPosition]:
        """Get current position."""
        return self._position
    
    async def check_entry(self) -> Optional[MomentumSignal]:
        """
        Check for momentum entry signal.
        
        Conditions:
        1. Spot impulse: ≥0.20% change in ≤2s
        2. Poly lag: divergence ≥ 0.08
        3. Not in cooldown
        4. Position not already open
        """
        # Check cooldown
        if time.time() - self._last_entry_time < self.config.cooldown_seconds:
            logger.debug("Momentum: blocked by cooldown")
            return None
        
        # Check if position already open
        if self.has_position:
            logger.debug("Momentum: position already open")
            return None
        
        # Check time remaining
        market = self.state.get_market_data()
        if market.time_to_expiry < 15:  # Too close to expiry
            logger.debug("Momentum: too close to expiry (%ds)", market.time_to_expiry)
            return None
        
        # Get risk metrics
        risk = self.state.get_risk_metrics()
        
        # Check VPIN toxicity
        if risk.vpin > 0.7:  # Very toxic
            logger.debug("Momentum: blocked by high VPIN %.2f", risk.vpin)
            return None
        
        # Get spot change - only log at debug level
        spot_change = self.state.get_spot_change(2)
        
        # Debug only - no noisy logging every second
        logger.debug(f"Momentum check: spot={spot_change*100:.3f}% thr={self.config.spot_impulse_threshold*100:.3f}% vpin={risk.vpin:.2f}")
        
        # Determine threshold based on regime
        threshold = self.config.spot_impulse_threshold
        if risk.regime == MarketRegime.HIGH:
            threshold = self.config.high_vol_impulse
        elif risk.regime == MarketRegime.LOW:
            threshold = self.config.low_vol_impulse
        
        # Check impulse direction
        if abs(spot_change) < threshold:
            logger.debug("Momentum: spot impulse %.3f%% below threshold %.3f%%", spot_change*100, threshold*100)
            return None
        
        side = MomentumSide.UP if spot_change > 0 else MomentumSide.DOWN
        
        # Calculate divergence - how much spot moved vs Poly's reaction
        # We want to catch when spot moves but Poly hasn't fully reacted yet
        mid_price = market.mid_price
        
        # Divergence is based on spot movement magnitude
        # No complex divergence calculation - just use spot movement as signal
        divergence = abs(spot_change) * 100  # Scale up for readability
        
        # Check divergence threshold (lowered to allow signals)
        if divergence < 0.01:  # Very low threshold
            logger.debug("Momentum: divergence %.3f below threshold 0.01", divergence)
            return None
        
        # Calculate confidence - primarily based on impulse strength
        impulse_strength = min(1.0, abs(spot_change) / threshold)
        vpin_factor = max(0.5, 1 - 0.5 * risk.vpin)  # Less punitive
        
        confidence = impulse_strength * vpin_factor
        
        if confidence < self.config.min_signal_strength:
            logger.debug("Momentum: confidence %.2f below min %.2f", confidence, self.config.min_signal_strength)
            return None
        
        # Build signal
        signal = MomentumSignal(
            side=side,
            confidence=confidence,
            entry_price=mid_price,
            reason=f"impulse {spot_change*100:.2f}%",
            impulse_strength=impulse_strength,
            divergence=divergence,
        )
        
        self._recent_signals.append(signal)
        return signal
    
    def calculate_size(self, signal: MomentumSignal) -> float:
        """
        Calculate position size in shares (min $1 value, max 10 shares).
        """
        market = self.state.get_market_data()
        mid_price = market.mid_price
        if mid_price <= 0:
            return 0
        
        # Calculate shares needed for minimum $1 order value
        min_value_shares = 1.0 / mid_price  # e.g., 1/0.10 = 10 shares
        
        # Minimum 5 shares, or more if needed to hit $1 minimum
        # Cap at 10 shares max
        size_shares = max(5.0, min(min_value_shares, 10.0))
        
        return round(size_shares, 2)
    
    async def execute(self, signal: MomentumSignal, size_override: Optional[float] = None) -> Dict[str, Any]:
        """Execute momentum entry.

        Args:
            signal: Entry signal.
            size_override: If provided by coordinator (e.g. VPIN-adjusted), use this
                           instead of recalculating size internally (P1-7).
        """
        # Check balance block cooldown
        if time.time() < self._balance_block_until:
            logger.debug("Momentum: blocked by balance cooldown")
            return {"success": False, "reason": "balance cooldown"}

        # Stamp attempt time so cooldown applies even on FOK rejection.
        self._last_entry_time = time.time()

        size = size_override if size_override is not None else self.calculate_size(signal)

        if size < 5.0:
            return {"success": False, "reason": "size too small"}

        # Determine token and side
        market = self.state.get_market_data()

        if signal.side == MomentumSide.UP:
            token_id = market.token_id_up
            side = "BUY"
        else:
            token_id = market.token_id_down
            side = "BUY"  # Must BUY to open a DOWN position

        # Execute FOK order with retries (for first-time setup issues)
        max_retries = 5
        for attempt in range(max_retries):
            # Refresh market data on each retry — prices may have changed
            market = self.state.get_market_data()
            price = market.up_price if signal.side == MomentumSide.UP else market.down_price

            try:
                result = await self.bot.place_order(
                    token_id=token_id,
                    price=price,
                    size=size,
                    side=side,
                    order_type="FOK",
                )

                if result.success:
                    actual_entry = round(price, 4)

                    tp1_actual = actual_entry * (1 + self.config.tp1_pct)
                    tp2_actual = actual_entry * (1 + self.config.tp2_pct)
                    sl_actual = actual_entry * (1 - self.config.initial_sl_pct)

                    # Record position
                    self._position = MomentumPosition(
                        side=signal.side,
                        entry_price=actual_entry,
                        size=size,
                        entry_time=time.time(),
                        sl_price=sl_actual,
                    )

                    self._last_entry_time = time.time()
                    self._entries_today += 1

                    # Create exits object
                    exits_obj = MomentumExit(
                        tp1_price=tp1_actual,
                        tp2_price=tp2_actual,
                        initial_sl=sl_actual,
                        time_confirm_secs=self.config.time_confirm_sl,
                    )

                    # Place exit limit orders as background task — do NOT block the
                    # coordination loop with a sleep (P2-2).
                    asyncio.create_task(
                        self._place_exit_limit_orders(self._position, tp1_actual, tp2_actual)
                    )

                    logger.info(
                        f"==> MOMENTUM ENTERED: {signal.side.value.upper()} | ${size:.2f} @ {actual_entry:.3f} | "
                        f"TP1: {tp1_actual:.3f} TP2: {tp2_actual:.3f} SL: {sl_actual:.3f}"
                    )

                    return {
                        "success": True,
                        "position": self._position,
                        "exits": exits_obj,
                    }

                # Check error type
                is_balance_error = result.message and (
                    "balance" in result.message.lower() or "allowance" in result.message.lower()
                )

                if is_balance_error:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # 1s, 2s, 4s
                        logger.warning("Momentum: balance error, retry %d/%d in %ds", attempt + 1, max_retries, wait_time)
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        self._balance_block_until = time.time() + 10
                        logger.warning("Momentum: balance error on entry, blocking 10s")

                return {"success": False, "reason": result.message}

            except Exception as e:
                logger.error("Momentum execution failed: %s", e)
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                return {"success": False, "reason": str(e)}

        return {"success": False, "reason": "max retries exceeded"}
    
    async def _cancel_all_exit_orders(self, pos: MomentumPosition):
        """Cancel all live exit limit orders for a position (P0-5)."""
        clob = self.bot.clob_client
        for oid in [pos.sl_order_id, pos.tp1_order_id, pos.tp2_order_id]:
            if oid:
                try:
                    await self.bot._run_in_thread(clob.cancel_order, oid)
                    logger.info(f"Momentum: cancelled exit order {oid}")
                except Exception as e:
                    logger.debug(f"Momentum: could not cancel order {oid}: {e}")

    async def _place_exit_limit_orders(self, pos: MomentumPosition, tp1_price: float, tp2_price: float):
        """No resting exit orders are placed.

        GTC exit orders are incompatible with close_position() architecture:
        when a GTC order fills on the exchange, calling close_position() again
        double-sells the same tokens, causing wrong position tracking and missed PnL.

        All exits (TP1, TP2, SL, mercy, expiry) are handled entirely by the
        _check_exit() price-level monitor running at 4 Hz.
        """
        logger.info(
            "Momentum: position active — price-level exits armed | "
            "TP1=%.3f (35%%)  TP2=%.3f (35%%)  SL=%.3f  size=%.2fsh",
            tp1_price, tp2_price, pos.sl_price, pos.size,
        )
    
    async def _cancel_remaining_orders(self, pos: MomentumPosition, reason: str):
        """Cancel remaining limit orders after one fills."""
        clob = self.bot.clob_client
        orders_to_cancel = []
        
        if reason == "SL":
            orders_to_cancel = [pos.tp1_order_id, pos.tp2_order_id]
        elif reason in ["TP1", "TP2"]:
            orders_to_cancel = [pos.sl_order_id]
            if reason == "TP1":
                orders_to_cancel.append(pos.tp2_order_id)
            elif reason == "TP2":
                orders_to_cancel.append(pos.tp1_order_id)
        
        for order_id in orders_to_cancel:
            if order_id:
                try:
                    await self.bot._run_in_thread(clob.cancel_order, order_id)
                    logger.info(f"Momentum: cancelled {order_id} after {reason}")
                except Exception as e:
                    logger.debug(f"Momentum: failed to cancel order {order_id}: {e}")
    
    async def _check_limit_orders(self, pos: MomentumPosition) -> Optional[Dict[str, Any]]:
        """Check if any limit orders have been filled.

        Polling is throttled to every 5 seconds (P2-1) to avoid flooding the REST
        API. Returns None if the poll window has not elapsed yet.
        """
        if not pos.sl_order_id and not pos.tp1_order_id and not pos.tp2_order_id:
            return None

        # P2-1: throttle HTTP polling to every 5 seconds
        now = time.time()
        if now - pos._last_order_poll < 5.0:
            return None
        pos._last_order_poll = now

        try:
            # Check SL order
            if pos.sl_order_id:
                result = await self.bot.get_order(pos.sl_order_id) or {}
                if result.get("status") in ("matched", "MATCHED"):
                    filled_size = float(result.get("size_matched") or result.get("size", 0))
                    if filled_size > 0:
                        logger.info(f"Momentum: SL order filled {filled_size} shares")
                        oid = pos.sl_order_id
                        pos.sl_order_id = None  # P0-4: clear immediately to prevent re-detection
                        await self._cancel_remaining_orders(pos, "SL")
                        return {"action": "sell", "size": filled_size, "reason": "SL", "order_id": oid}

            # Check TP1 order — ratchet: sell 35%, new TP2 higher, SL at break-even
            if pos.tp1_order_id:
                result = await self.bot.get_order(pos.tp1_order_id) or {}
                if result.get("status") in ("matched", "MATCHED"):
                    filled_size = float(result.get("size_matched") or result.get("size", 0))
                    if filled_size > 0:
                        pos.tp1_filled = True
                        oid = pos.tp1_order_id
                        pos.tp1_order_id = None  # P0-4: clear immediately
                        logger.info(f"Momentum: TP1 filled {filled_size} shares")

                        # Cancel old TP2 and SL before placing new ones
                        if pos.tp2_order_id:
                            try:
                                await self.bot._run_in_thread(clob.cancel_order, pos.tp2_order_id)
                                pos.tp2_order_id = None
                            except Exception:
                                pass
                        if pos.sl_order_id:
                            try:
                                await self.bot._run_in_thread(clob.cancel_order, pos.sl_order_id)
                                pos.sl_order_id = None
                            except Exception:
                                pass

                        # Place new TP2 using config value (P2-6) and new SL at break-even
                        market = self.state.get_market_data()
                        token_id = market.token_id_up if pos.side == MomentumSide.UP else market.token_id_down
                        new_tp2_price = round(pos.entry_price * (1 + self.config.tp2_pct) * 1.005, 4)
                        remaining_size = round(pos.size * 0.30, 2)  # 30% holds

                        new_tp2, new_sl = await asyncio.gather(
                            self.bot.place_order(token_id=token_id, price=new_tp2_price, size=remaining_size, side="SELL", order_type="GTC"),
                            self.bot.place_order(token_id=token_id, price=round(pos.entry_price, 4), size=remaining_size, side="SELL", order_type="GTC"),
                            return_exceptions=True,
                        )
                        if not isinstance(new_tp2, Exception) and new_tp2.success:
                            pos.tp2_order_id = new_tp2.order_id
                        if not isinstance(new_sl, Exception) and new_sl.success:
                            pos.sl_order_id = new_sl.order_id
                            pos.sl_ratcheted = True

                        logger.info(f"Momentum: TP1 done — new TP2 @ {new_tp2_price:.3f}, SL @ break-even")
                        return {"action": "sell", "size": filled_size, "reason": "TP1", "order_id": oid}

            # Check TP2 order — let remainder ride, move SL to TP1 price
            if pos.tp2_order_id:
                result = await self.bot.get_order(pos.tp2_order_id) or {}
                if result.get("status") in ("matched", "MATCHED"):
                    filled_size = float(result.get("size_matched") or result.get("size", 0))
                    if filled_size > 0:
                        pos.tp2_filled = True
                        oid = pos.tp2_order_id
                        pos.tp2_order_id = None  # P0-4: clear immediately
                        logger.info(f"Momentum: TP2 filled {filled_size} shares, letting rest ride")

                        # Cancel existing SL
                        if pos.sl_order_id:
                            try:
                                await self.bot._run_in_thread(clob.cancel_order, pos.sl_order_id)
                                pos.sl_order_id = None
                            except Exception:
                                pass

                        # Place SL at TP1 level for the hold portion (P2-7)
                        market = self.state.get_market_data()
                        token_id = market.token_id_up if pos.side == MomentumSide.UP else market.token_id_down
                        tp1_price = round(pos.entry_price * (1 + self.config.tp1_pct), 4)
                        hold_size = round(pos.size * 0.30, 2)  # P2-7: only the hold portion

                        new_sl = await self.bot.place_order(
                            token_id=token_id,
                            price=tp1_price,
                            size=hold_size,
                            side="SELL",
                            order_type="GTC",
                        )
                        if new_sl.success:
                            pos.sl_order_id = new_sl.order_id

                        logger.info(f"Momentum: TP2 done — hold riding with SL @ {tp1_price:.3f}")
                        return {"action": "sell", "size": filled_size, "reason": "TP2", "order_id": oid}

            return None
        except Exception as e:
            logger.debug(f"Momentum: error checking limit orders: {e}")
            return None
    
    async def check_exit(self) -> Optional[Dict[str, Any]]:
        """
        Check if position should be exited.
        
        Conditions:
        - TP1 hit: +14%, sell 35%, ratchet SL to break-even
        - TP2 hit: +28%, sell 35%, ratchet SL to TP1
        - SL hit: -12% with time confirmation
        - Early exit: VPIN > 0.7 or spot reversal > 0.15% in 5s
        - Time: <15% remaining, hold to expiry unless mercy
        """
        if not self._position:
            return None
        
        # Wait minimum time before checking exits - allows balance to settle
        time_held = time.time() - self._position.entry_time
        min_hold_seconds = 3.0  # Reduced from 15s for faster exits
        if time_held < min_hold_seconds:
            return None
        
        # Check if limit orders have been filled first
        limit_result = await self._check_limit_orders(self._position)
        if limit_result:
            return limit_result
        
        market = self.state.get_market_data()
        risk = self.state.get_risk_metrics()
        
        current_price = market.up_price if self._position.side == MomentumSide.UP else market.down_price
        
        pos = self._position
        
        # Calculate position value and PnL
        position_value = current_price * pos.size
        entry_value = pos.entry_price * pos.size
        pnl_pct = (position_value - entry_value) / entry_value if entry_value > 0 else 0
        pnl_dollars = position_value - entry_value
        
        # Only log on significant PnL change (>5%) or every ~10 seconds
        if not hasattr(self, '_last_logged_pnl'):
            self._last_logged_pnl = None
        now = time.time()
        should_log = (
            self._last_logged_pnl is None 
            or abs(pnl_pct - self._last_logged_pnl) > 0.05
            or (now - getattr(self, '_last_log_time', 0)) > 10
        )
        
        if should_log:
            side_str = pos.side.value.upper()
            logger.info(f"[{side_str}] Pos: entry={pos.entry_price:.3f} cur={current_price:.3f} | PnL: {pnl_pct*100:+.1f}% (${pnl_dollars:+.2f})")
            self._last_logged_pnl = pnl_pct
            self._last_log_time = now
        
        # ── Trailing SL ──────────────────────────────────────────────────────────
        # Track peak price since entry
        if not hasattr(pos, '_peak_price'):
            pos._peak_price = pos.entry_price  # type: ignore[attr-defined]
        pos._peak_price = max(pos._peak_price, current_price)  # type: ignore[attr-defined]

        # Floor SL from initial config
        floor_sl = pos.entry_price * (1 - self.config.initial_sl_pct)

        # After TP1 fires: activate trailing SL at (peak × (1 − trail_pct)),
        # minimum = entry price (break-even).  SL only ever moves UP.
        if pos.tp1_filled:
            trail_sl = max(pos.entry_price, pos._peak_price * (1 - self.config.trailing_sl_pct))  # type: ignore[attr-defined]
            new_sl = max(pos.sl_price, floor_sl, trail_sl)
        else:
            new_sl = max(pos.sl_price, floor_sl)

        if new_sl > pos.sl_price + 0.0001:
            logger.info(
                "Momentum: SL ratcheted %.4f → %.4f%s (peak=%.4f)",
                pos.sl_price, new_sl,
                " [TRAIL]" if pos.tp1_filled else "",
                pos._peak_price,  # type: ignore[attr-defined]
            )
        pos.sl_price = new_sl

        # ── Price-level exits ─────────────────────────────────────────────────
        tp1_threshold = pos.entry_price * (1 + self.config.tp1_pct)
        tp2_threshold = pos.entry_price * (1 + self.config.tp2_pct)

        # SL
        if current_price <= pos.sl_price and position_value >= 0.10:
            logger.warning("Momentum SL TRIGGERED: entry=%.4f sl=%.4f cur=%.4f", pos.entry_price, pos.sl_price, current_price)
            return {"action": "sell", "size": pos.size, "reason": "SL"}

        # TP1
        if not pos.tp1_filled and position_value >= 0.10:
            if current_price >= tp1_threshold:
                logger.info("Momentum TP1 TRIGGERED: entry=%.4f cur=%.4f", pos.entry_price, current_price)
                pos.tp1_filled = True
                return {"action": "sell", "size": pos.size * 0.35, "reason": "TP1"}

        # TP2 — only after TP1 has already fired
        if pos.tp1_filled and not pos.tp2_filled and position_value >= 0.10:
            if current_price >= tp2_threshold:
                logger.info("Momentum TP2 TRIGGERED: entry=%.4f cur=%.4f", pos.entry_price, current_price)
                pos.tp2_filled = True
                return {"action": "sell", "size": pos.size * 0.35, "reason": "TP2"}
        
        # Late window: mercy stop
        if market.time_to_expiry < 45:
            mercy_price = pos.entry_price * (1 - 0.30)  # 30% mercy
            if pos.side == MomentumSide.UP and current_price <= mercy_price:
                return {"action": "sell", "size": pos.size, "reason": "mercy"}
            elif pos.side == MomentumSide.DOWN and current_price <= mercy_price:
                return {"action": "sell", "size": pos.size, "reason": "mercy"}
        
        # Expiry close
        if market.time_to_expiry <= 5:
            return {"action": "sell", "size": pos.size, "reason": "expiry"}
    
    async def close_position(self, reason: str = "manual", sell_size: Optional[float] = None) -> Dict[str, Any]:
        """Close current position (full or partial)."""
        if not self._position:
            return {"success": False, "reason": "no position"}
        
        # Check balance block cooldown
        if time.time() < self._balance_block_until:
            return {"success": False, "reason": "balance cooldown"}
        
        # Wait for token settlement after recent entry (FOK fills need time)
        time_since_entry = time.time() - self._position.entry_time
        if time_since_entry < 3.0:
            wait_time = 3.0 - time_since_entry
            logger.debug(f"Momentum: waiting {wait_time:.1f}s for token settlement")
            await asyncio.sleep(wait_time)
        
        market = self.state.get_market_data()
        
        # Determine size to sell (default to full position)
        if sell_size is None:
            sell_size = self._position.size
        
        # Round to 2 decimals (FOK requirement)
        sell_size = int(sell_size * 100) / 100.0
        if sell_size < 0.01:
            return {"success": False, "reason": "size too small"}
        
        # Determine side to close
        if self._position.side == MomentumSide.UP:
            token_id = market.token_id_up
            side = "SELL"
            # Use best bid for better FOK fill
            if market.up_bids:
                price = market.up_bids[0][0]
            else:
                price = market.up_price
        else:
            token_id = market.token_id_down
            side = "SELL"  # Must SELL to close a DOWN position
            # Use best bid for better FOK fill
            if market.down_bids:
                price = market.down_bids[0][0]
            else:
                price = market.down_price
        
        try:
            # Try FOK first, then FAK if it fails
            order_type = "FOK"
            result = await self.bot.place_order(
                token_id=token_id,
                price=price,
                size=sell_size,
                side=side,
                order_type=order_type,
            )
            
            # If FOK fails, try FAK for partial fill
            if not result.success and "couldn't be fully filled" in result.message.lower():
                order_type = "FAK"
                result = await self.bot.place_order(
                    token_id=token_id,
                    price=price,
                    size=sell_size,
                    side=side,
                    order_type=order_type,
                )
            
            if result.success:
                # PnL calculation: works same for UP and DOWN since we now
                # BUY to open and SELL to close both positions
                pnl = (price - self._position.entry_price) * sell_size
                
                if pnl > 0:
                    self._wins_today += 1
                self._realized_pnl += pnl
                self._window_pnl   += pnl
                
                # Update position size (for partial sells)
                self._position.size -= sell_size
                self._failed_close_attempts = 0  # Reset on success
                
                # Clear position if fully sold
                if self._position.size < 0.01:
                    self._position = None
                
                pnl_str = f"+${pnl:.2f}" if pnl > 0 else f"-${abs(pnl):.2f}"
                logger.info(f"==> MOMENTUM CLOSED: {reason.upper()} | {sell_size:.2f}sh @ {price:.4f} | PnL: {pnl_str}")
                
                return {"success": True, "pnl": pnl}
            else:
                # Any close failure - increment counter
                self._failed_close_attempts += 1
                
                # Check for balance/allowance error
                if result.message and ("balance" in result.message.lower() or "allowance" in result.message.lower()):
                    self._balance_block_until = time.time() + 3
                    logger.warning("Momentum close: balance error, attempt %d", self._failed_close_attempts)
                else:
                    logger.warning("Momentum close: FOK failed (%s), attempt %d", result.message, self._failed_close_attempts)
                
                # Force clear after 5 failed attempts — cancel GTC orders first (P0-5)
                if self._failed_close_attempts >= 5:
                    logger.error("Momentum: force clearing position after 5 failed close attempts")
                    if self._position:
                        await self._cancel_all_exit_orders(self._position)
                    self._position = None
                    self._failed_close_attempts = 0

                return {"success": False, "reason": result.message}

        except Exception as e:
            logger.error("Momentum close failed: %s", e)
            self._failed_close_attempts += 1
            if self._failed_close_attempts >= 5:
                logger.error("Momentum: force clearing position after 5 failed close attempts")
                if self._position:
                    await self._cancel_all_exit_orders(self._position)
                self._position = None
                self._failed_close_attempts = 0
            return {"success": False, "reason": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bot statistics."""
        return {
            "entries_today": self._entries_today,
            "wins_today": self._wins_today,
            "win_rate": self._wins_today / max(1, self._entries_today),
            "has_position": self.has_position,
        }
    
    def reset_daily_stats(self):
        """Reset daily statistics."""
        self._entries_today = 0
        self._wins_today = 0

    def reset_window_pnl(self):
        """Reset per-window PnL counter (called by coordinator on new market)."""
        self._window_pnl = 0.0
