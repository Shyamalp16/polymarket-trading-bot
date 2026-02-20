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
    spot_impulse_threshold: float = 0.0003  # 0.03% in 2s — catches normal BTC moves
    spot_impulse_window: int = 2            # seconds

    # Dynamic thresholds by regime — high vol LOWERS trigger, low vol RAISES it
    high_vol_impulse: float = 0.0002   # 0.02% — easier entry in high vol
    low_vol_impulse: float = 0.0004    # 0.04% — stricter entry in low vol
    
    # Execution
    max_slippage_ticks: int = 4
    ladder_attempts: int = 3
    
    # Exits
    tp1_pct: float = 0.14              # +14%
    tp2_pct: float = 0.28              # +28%
    tp3_price: float = 0.90            # absolute price level — sell 100% at ≥90¢
    initial_sl_pct: float = 0.20       # -20%  (was 35% — never fired)
    trailing_sl_pct: float = 0.15      # trail 15% below peak — activates after TP1
    time_confirm_sl: float = 10.0       # seconds - no SL in first 10s
    early_exit_vpin: float = 0.7
    early_exit_spot_reversal: float = 0.0015  # 0.15% reversal
    
    # Depth cap
    max_depth_pct: float = 0.25        # 25% of visible depth
    
    # Cooldown
    cooldown_seconds: int = 30
    min_signal_strength: float = 0.30


@dataclass
class MomentumPosition:
    """Active momentum position."""
    side: MomentumSide
    entry_price: float
    size: float
    entry_time: float
    tp1_filled: bool = False
    tp2_filled: bool = False
    tp3_filled: bool = False
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

        # Entry lock — True while execute() is running (including taker retries).
        # Prevents the coordinator from starting a second concurrent entry during
        # the await sleeps inside execute().
        self._entry_in_progress: bool = False

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

        # Price ceiling: don't chase a side that's already priced in.
        # Above 0.85 there is little room to profit; the market has already moved.
        if side == MomentumSide.UP and market.up_price > 0.85:
            logger.debug("Momentum: UP price %.3f already high, skipping", market.up_price)
            return None
        if side == MomentumSide.DOWN and market.down_price > 0.85:
            logger.debug("Momentum: DOWN price %.3f already high, skipping", market.down_price)
            return None

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
        
        # Calculate confidence — impulse strength × VPIN toxicity penalty
        # impulse_strength: 1.0 at threshold, scales up with larger moves
        # vpin_factor: fully punitive — high toxicity blocks entries (was floored at 0.5)
        impulse_strength = min(1.0, abs(spot_change) / threshold)
        vpin_factor = max(0.1, 1 - risk.vpin)

        # Orderbook imbalance component (30% weight).
        # market.imbalance: 0 = bid-heavy (bullish UP), 1 = ask-heavy (bearish UP).
        # For an UP signal we want bid-heavy (imbalance → 0, ob_confirm → 1).
        # For a DOWN signal we want ask-heavy (imbalance → 1, ob_confirm → 1).
        ob_confirm = (1.0 - market.imbalance) if side == MomentumSide.UP else market.imbalance
        ob_factor = max(0.3, ob_confirm)  # floor 0.3 — OB alone can't kill a valid impulse

        confidence = (impulse_strength * vpin_factor) * 0.70 + ob_factor * 0.30
        
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
        """Calculate position size using half-Kelly criterion.

        Scales shares between the 5-share floor and a 30-share cap based on
        signal confidence and the trade's win/loss odds.  Uses TP1/SL ratio
        as the edge multiplier so strong impulses get proportionally larger size.
        """
        market = self.state.get_market_data()
        mid_price = market.mid_price or 0.5
        if mid_price <= 0:
            return 5.0

        # Kelly fraction: f = (b*p - q) / b, halved for safety
        b = self.config.tp1_pct / max(self.config.initial_sl_pct, 0.01)
        p = max(0.0, min(1.0, signal.confidence))
        q = 1.0 - p
        raw_kelly = (b * p - q) / b
        kelly_frac = max(0.0, raw_kelly) * 0.5   # half-Kelly

        capital = self.config.bankroll * self.config.max_position_pct
        dollar_size = capital * kelly_frac
        shares = dollar_size / mid_price

        return round(max(5.0, min(shares, 30.0)), 2)
    
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

        # Block concurrent calls — retry sleeps inside the FOK loop yield the event
        # loop back to the coordinator, which could approve another entry before the
        # first one completes, producing orphaned fills.
        if self._entry_in_progress:
            logger.debug("Momentum: entry already in progress — blocking concurrent execute()")
            return {"success": False, "reason": "entry in progress"}
        self._entry_in_progress = True

        try:
            return await self._execute_inner(signal, size_override)
        finally:
            self._entry_in_progress = False

    async def _execute_inner(self, signal: MomentumSignal, size_override: Optional[float] = None) -> Dict[str, Any]:
        """Inner execute — called exclusively from execute() which holds the entry lock."""
        # Stamp attempt time so cooldown applies even on FOK rejection.
        self._last_entry_time = time.time()

        size = size_override if size_override is not None else self.calculate_size(signal)

        if size < 5.0:
            return {"success": False, "reason": "size too small"}

        # Determine token and side
        market = self.state.get_market_data()

        if signal.side == MomentumSide.UP:
            token_id = market.token_id_up
        else:
            token_id = market.token_id_down
        side = "BUY"  # always BUY the token matching the direction

        # FAK with a +0.03 price sweep: fills whatever depth is available across
        # multiple ask levels and kills the rest.  Unlike FOK this never fully fails
        # on thin books — a partial fill is still a valid position.
        # 2 retries is enough; the sweep already covers several ticks.
        max_retries = 2
        for attempt in range(max_retries):
            # Refresh market data on each retry so the price is current
            market = self.state.get_market_data()

            # Always use best ask — FAK/FOK must be marketable (price >= ask)
            if signal.side == MomentumSide.UP:
                raw_ask = market.up_asks[0][0] if market.up_asks else market.up_price
            else:
                raw_ask = market.down_asks[0][0] if market.down_asks else market.down_price

            # Sweep 3 ticks above best ask to capture multiple depth levels
            exec_price = round(min(raw_ask + 0.03, 0.99), 4)

            try:
                result = await self.bot.place_order(
                    token_id=token_id,
                    price=exec_price,
                    size=size,
                    side=side,
                    order_type="FAK",
                )

                if result.success:
                    # FAK may report size_matched=0 even on partial fill — query settled order
                    actual_filled = size
                    if result.order_id:
                        try:
                            order_data = await self.bot.get_order(result.order_id)
                            raw = (
                                (order_data or {}).get("size_matched")
                                or (order_data or {}).get("sizeMatched")
                                or 0
                            )
                            queried = float(raw)
                            if queried > 0:
                                actual_filled = queried
                        except Exception:
                            pass  # keep actual_filled = size on query failure

                    if actual_filled < 5.0:
                        logger.warning(
                            "==> MOM ENTRY FAILED: FAK partial fill too small (%.2f sh < 5.0 min)",
                            actual_filled,
                        )
                        return {"success": False, "reason": f"partial fill too small: {actual_filled}"}

                    actual_entry = exec_price
                    tp1_actual = actual_entry * (1 + self.config.tp1_pct)
                    tp2_actual = actual_entry * (1 + self.config.tp2_pct)
                    sl_actual = actual_entry * (1 - self.config.initial_sl_pct)

                    self._position = MomentumPosition(
                        side=signal.side,
                        entry_price=actual_entry,
                        size=actual_filled,
                        entry_time=time.time(),
                        sl_price=sl_actual,
                    )

                    self._last_entry_time = time.time()
                    self._entries_today += 1

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
                        "==> MOMENTUM ENTERED: %s | $%.2f @ %.3f | TP1: %.3f TP2: %.3f SL: %.3f",
                        signal.side.value.upper(), actual_filled, actual_entry,
                        tp1_actual, tp2_actual, sl_actual,
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
                    self._balance_block_until = time.time() + 10
                    logger.error("==> MOM ENTRY FAILED: balance error — %s", result.message)
                    return {"success": False, "reason": result.message}

                logger.error(
                    "==> MOM ENTRY FAILED: FAK rejected (attempt %d/%d) — %s",
                    attempt + 1, max_retries, result.message,
                )
                # FAK rejection means no liquidity even with sweep — no point retrying immediately

            except Exception as e:
                logger.error("==> MOM ENTRY FAILED: exception (attempt %d/%d) — %s",
                             attempt + 1, max_retries, e)
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue

        return {"success": False, "reason": "all FAK attempts failed"}

        logger.error("==> MOM ENTRY FAILED: max retries exceeded")
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
        """Place GTC limit orders for SL, TP1, and TP2 immediately after entry fills.

        All three are posted as resting GTC sell orders so they fill naturally
        when price reaches each level — no FOK/FAK liquidity dependency.

        Sizes:
          SL  → full position (exits everything on a stop)
          TP1 → 35 % of position
          TP2 → 35 % of position
          (remaining 30 % hold is managed by ratcheted orders after TP1 fires)
        """
        await asyncio.sleep(3.0)  # Wait for tokens to settle after entry fill

        if not self._position or self._position is not pos:
            return  # position was cleared while waiting

        market = self.state.get_market_data()
        token_id = market.token_id_up if pos.side == MomentumSide.UP else market.token_id_down

        if not token_id:
            logger.warning("Momentum: cannot place exit orders — no token_id (market rotated?)")
            return

        tp1_size = round(pos.size * 0.35, 2)
        tp2_size = round(pos.size * 0.35, 2)
        sl_size  = round(pos.size, 2)

        tp1_price_r = max(0.01, min(0.99, round(tp1_price, 4)))
        tp2_price_r = max(0.01, min(0.99, round(tp2_price, 4)))
        sl_price_r  = max(0.01, min(0.99, round(pos.sl_price, 4)))

        logger.info(
            "Momentum: placing exit GTC orders | SL=%.4f x%.2f  TP1=%.4f x%.2f  TP2=%.4f x%.2f",
            sl_price_r, sl_size, tp1_price_r, tp1_size, tp2_price_r, tp2_size,
        )

        tp1_res, tp2_res, sl_res = await asyncio.gather(
            self.bot.place_order(token_id=token_id, price=tp1_price_r, size=tp1_size, side="SELL", order_type="GTC"),
            self.bot.place_order(token_id=token_id, price=tp2_price_r, size=tp2_size, side="SELL", order_type="GTC"),
            self.bot.place_order(token_id=token_id, price=sl_price_r,  size=sl_size,  side="SELL", order_type="GTC"),
            return_exceptions=True,
        )

        if not isinstance(tp1_res, Exception) and tp1_res.success:
            pos.tp1_order_id = tp1_res.order_id
            logger.info("Momentum: TP1 GTC resting @ %.4f x%.2f [%s]", tp1_price_r, tp1_size, tp1_res.order_id[:8])
        else:
            logger.warning("Momentum: TP1 GTC failed: %s", tp1_res)

        if not isinstance(tp2_res, Exception) and tp2_res.success:
            pos.tp2_order_id = tp2_res.order_id
            logger.info("Momentum: TP2 GTC resting @ %.4f x%.2f [%s]", tp2_price_r, tp2_size, tp2_res.order_id[:8])
        else:
            logger.warning("Momentum: TP2 GTC failed: %s", tp2_res)

        if not isinstance(sl_res, Exception) and sl_res.success:
            pos.sl_order_id = sl_res.order_id
            logger.info("Momentum: SL  GTC resting @ %.4f x%.2f [%s]", sl_price_r, sl_size, sl_res.order_id[:8])
        else:
            logger.warning("Momentum: SL GTC failed: %s", sl_res)
    
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

    def _handle_gtc_exit_fill(self, size: float, reason: str, fill_price: float) -> Dict[str, Any]:
        """Update position tracking after a GTC exit order confirmed filled on the exchange.

        Called by the coordinator when already_filled=True so we never call
        close_position() and double-sell tokens that are already gone.
        """
        if not self._position:
            return {"success": False, "reason": "no position"}

        pos = self._position
        pnl = (fill_price - pos.entry_price) * size
        self._realized_pnl += pnl
        self._window_pnl   += pnl
        if pnl > 0:
            self._wins_today += 1

        remaining = round(pos.size - size, 2)
        if remaining <= 0.01:
            self._position = None
            self._failed_close_attempts = 0
        else:
            pos.size = remaining

        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        logger.info(
            "==> MOMENTUM GTC EXIT: %s | %.2fsh @ %.4f | PnL: %s | remaining: %.2fsh",
            reason.upper(), size, fill_price, pnl_str, max(0.0, remaining),
        )
        return {"success": True, "pnl": pnl}

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
                        return {
                            "action": "sell", "size": filled_size, "reason": "SL", "order_id": oid,
                            "already_filled": True, "fill_price": pos.sl_price,
                        }

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
                        tp1_fill_price = round(pos.entry_price * (1 + self.config.tp1_pct), 4)
                        return {
                            "action": "sell", "size": filled_size, "reason": "TP1", "order_id": oid,
                            "already_filled": True, "fill_price": tp1_fill_price,
                        }

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
                        tp2_fill_price = round(pos.entry_price * (1 + self.config.tp2_pct), 4)
                        return {
                            "action": "sell", "size": filled_size, "reason": "TP2", "order_id": oid,
                            "already_filled": True, "fill_price": tp2_fill_price,
                        }

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

        # SL — skip if a resting GTC SL order is handling this level
        if not pos.sl_order_id and current_price <= pos.sl_price and position_value >= 0.10:
            logger.warning("Momentum SL TRIGGERED: entry=%.4f sl=%.4f cur=%.4f", pos.entry_price, pos.sl_price, current_price)
            return {"action": "sell", "size": pos.size, "reason": "SL"}

        # TP1 — skip if a resting GTC TP1 order is handling this level
        if not pos.tp1_order_id and not pos.tp1_filled and position_value >= 0.10:
            if current_price >= tp1_threshold:
                logger.info("Momentum TP1 TRIGGERED: entry=%.4f cur=%.4f", pos.entry_price, current_price)
                pos.tp1_filled = True
                return {"action": "sell", "size": pos.size * 0.35, "reason": "TP1"}

        # TP2 — skip if a resting GTC TP2 order is handling this level
        if not pos.tp2_order_id and pos.tp1_filled and not pos.tp2_filled and position_value >= 0.10:
            if current_price >= tp2_threshold:
                logger.info("Momentum TP2 TRIGGERED: entry=%.4f cur=%.4f", pos.entry_price, current_price)
                pos.tp2_filled = True
                return {"action": "sell", "size": pos.size * 0.35, "reason": "TP2"}

        # TP3 — absolute price level ≥ 0.90 — sell 100% immediately, no prerequisite
        if not pos.tp3_filled and position_value >= 0.10:
            if current_price >= self.config.tp3_price:
                logger.info(
                    "Momentum TP3 TRIGGERED (≥%.2f): cur=%.4f — selling 100%%",
                    self.config.tp3_price, current_price,
                )
                pos.tp3_filled = True
                return {"action": "sell", "size": pos.size, "reason": "TP3"}

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
        """
        Close current position (full or partial).

        Retry strategy (mirrors strategies/base.py execute_sell):
          1. Poll CLOB balance up to 3× before the first order attempt to
             confirm tokens have settled from the entry FOK fill.
          2. Retry the FOK up to MAX_CLOSE_ATTEMPTS times, dropping the
             sell price by PRICE_STEP each attempt to chase liquidity.
          3. On balance errors: refresh and wait progressively longer.
          4. Force-clear after FORCE_CLEAR_CALLS consecutive fully-failed calls.
        """
        MAX_CLOSE_ATTEMPTS = 5   # retries per close_position() call
        PRICE_STEP = 0.02        # drop sell price 2 ticks each retry
        BALANCE_POLLS = 3        # CLOB balance polls before first order
        FORCE_CLEAR_CALLS = 3    # fully-failed calls before force-clear

        if not self._position:
            return {"success": False, "reason": "no position"}

        # Cancel any resting GTC exit orders before FAK-selling to prevent double-sells
        # (mercy stop, expiry close, and TP3 all bypass GTC and use FAK directly)
        if self._position.sl_order_id or self._position.tp1_order_id or self._position.tp2_order_id:
            await self._cancel_all_exit_orders(self._position)

        # Wait for token settlement after recent entry (FOK fills need time)
        time_since_entry = time.time() - self._position.entry_time
        if time_since_entry < 3.0:
            wait_time = 3.0 - time_since_entry
            logger.debug("Momentum: waiting %.1fs for token settlement", wait_time)
            await asyncio.sleep(wait_time)

        market = self.state.get_market_data()

        # Determine size to sell (default to full position)
        if sell_size is None:
            sell_size = self._position.size

        # Round to 2 decimals (FOK requirement)
        sell_size = int(sell_size * 100) / 100.0
        if sell_size < 0.01:
            return {"success": False, "reason": "size too small"}

        if self._position.side == MomentumSide.UP:
            token_id = market.token_id_up
            if market.up_bids:
                initial_price = market.up_bids[0][0]
            else:
                initial_price = market.up_price
        else:
            token_id = market.token_id_down
            if market.down_bids:
                initial_price = market.down_bids[0][0]
            else:
                initial_price = market.down_price

        # Pre-flight: empty token_id means market rotated; bail immediately
        if not token_id:
            msg = f"no token_id for {self._position.side.value} (market rotated?)"
            logger.error("==> MOMENTUM CLOSE FAILED: %s | attempt %d | reason: %s",
                         reason.upper(), self._failed_close_attempts + 1, msg)
            self._failed_close_attempts += 1
            if self._failed_close_attempts >= FORCE_CLEAR_CALLS:
                if self._position:
                    await self._cancel_all_exit_orders(self._position)
                self._position = None
                self._failed_close_attempts = 0
            return {"success": False, "reason": msg}

        if initial_price <= 0:
            msg = f"price={initial_price:.4f} (no bids + stale mid)"
            logger.error("==> MOMENTUM CLOSE FAILED: %s | attempt %d | reason: %s",
                         reason.upper(), self._failed_close_attempts + 1, msg)
            self._failed_close_attempts += 1
            if self._failed_close_attempts >= FORCE_CLEAR_CALLS:
                if self._position:
                    await self._cancel_all_exit_orders(self._position)
                self._position = None
                self._failed_close_attempts = 0
            return {"success": False, "reason": msg}

        # ── Phase 1: poll CLOB balance until tokens are visible ──────────────
        for poll in range(BALANCE_POLLS):
            try:
                await self.bot.update_balance_allowance("CONDITIONAL", token_id)
                bal = await self.bot.get_balance_allowance("CONDITIONAL", token_id)
                if bal and float(bal.get("balance", 0)) > 0:
                    break
            except Exception:
                pass
            if poll < BALANCE_POLLS - 1:
                logger.debug("Momentum close: CLOB balance empty, waiting 3s (poll %d/%d)",
                             poll + 1, BALANCE_POLLS)
                await asyncio.sleep(3.0)

        # ── Phase 2: retry loop — FAK with partial-fill tracking ─────────────
        # FAK (Fill-And-Kill) fills whatever depth exists at the price level and
        # cancels the rest.  This guarantees progress every attempt even when the
        # book is thin — unlike FOK which fails all-or-nothing and leaves you
        # stuck at -50% while retrying the same full size each time.
        remaining = sell_size
        total_pnl = 0.0

        for attempt in range(MAX_CLOSE_ATTEMPTS):
            if remaining < 0.01:
                break  # Fully closed by earlier partial fills

            sell_price = max(0.01, initial_price - attempt * PRICE_STEP)

            try:
                result = await self.bot.place_order(
                    token_id=token_id,
                    price=sell_price,
                    size=remaining,
                    side="SELL",
                    order_type="FAK",  # Partial fills allowed — always makes progress
                )
            except Exception as e:
                logger.error("==> MOMENTUM CLOSE FAILED: %s | attempt %d | reason: exception — %s",
                             reason.upper(), attempt + 1, e)
                await asyncio.sleep(1.0)
                continue

            if result.success:
                # Query the confirmed fill amount via get_order().
                # Polymarket's FAK POST response often returns size_matched=0 or
                # omits it entirely even when shares were filled.  Using the POST
                # response directly (with `or remaining` as fallback) causes a
                # phantom close — bot clears _position but no trade happened.
                filled = 0.0
                if result.order_id:
                    try:
                        order_data = await self.bot.get_order(result.order_id)
                        raw = (order_data or {}).get("size_matched") or (order_data or {}).get("sizeMatched") or 0
                        filled = float(raw)
                    except Exception as e:
                        logger.warning("Momentum close: get_order failed after FAK (attempt %d): %s", attempt + 1, e)

                filled = min(filled, remaining)

                if filled < 0.01:
                    logger.debug("Momentum close: FAK 0 fill confirmed @ %.4f (attempt %d), retrying lower",
                                 sell_price, attempt + 1)
                    continue

                pnl = (sell_price - self._position.entry_price) * filled
                total_pnl += pnl
                if pnl > 0:
                    self._wins_today += 1
                self._realized_pnl += pnl
                self._window_pnl   += pnl
                self._position.size -= filled
                remaining -= filled

                logger.info(
                    "Momentum close: FAK filled %.2f/%.2f sh @ %.4f (attempt %d, remaining %.2f)",
                    filled, sell_size, sell_price, attempt + 1, remaining,
                )

                if remaining < 0.01:
                    self._failed_close_attempts = 0
                    if self._position.size < 0.01:
                        self._position = None
                    pnl_str = f"+${total_pnl:.2f}" if total_pnl >= 0 else f"-${abs(total_pnl):.2f}"
                    logger.info("==> MOMENTUM CLOSED: %s | %.2fsh | PnL: %s",
                                reason.upper(), sell_size, pnl_str)
                    return {"success": True, "pnl": total_pnl}

                continue

            err_msg = result.message or "unknown"

            if "balance" in err_msg.lower() or "allowance" in err_msg.lower():
                wait = 5.0 + attempt * 5.0
                logger.warning("Momentum close: balance not settled (attempt %d), refreshing + %.0fs wait",
                                attempt + 1, wait)
                try:
                    await self.bot.update_balance_allowance("CONDITIONAL", token_id)
                except Exception:
                    pass
                await asyncio.sleep(wait)
                continue

            logger.debug("Momentum close: FAK no fill @ %.4f (attempt %d), retrying lower",
                         sell_price, attempt + 1)

        # ── All attempts exhausted for this call ──────────────────────────────
        if remaining < sell_size:
            if self._position and self._position.size < 0.01:
                self._position = None
            pnl_str = f"+${total_pnl:.2f}" if total_pnl >= 0 else f"-${abs(total_pnl):.2f}"
            logger.error("==> MOMENTUM CLOSE PARTIAL: %s | sold %.2f/%.2f sh | PnL: %s | remaining %.2f",
                         reason.upper(), sell_size - remaining, sell_size, pnl_str, remaining)
            return {"success": False, "partial": True, "pnl": total_pnl, "remaining": remaining}

        self._failed_close_attempts += 1
        logger.error("==> MOMENTUM CLOSE FAILED: %s | attempt %d | reason: all %d retries exhausted (0 fills)",
                     reason.upper(), self._failed_close_attempts, MAX_CLOSE_ATTEMPTS)
        if self._failed_close_attempts >= FORCE_CLEAR_CALLS:
            logger.error("==> MOMENTUM CLOSE FAILED: %s | FORCE CLEAR after %d calls",
                         reason.upper(), FORCE_CLEAR_CALLS)
            if self._position:
                await self._cancel_all_exit_orders(self._position)
            self._position = None
            self._failed_close_attempts = 0
        return {"success": False, "reason": f"all {MAX_CLOSE_ATTEMPTS} retries exhausted"}
    
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
