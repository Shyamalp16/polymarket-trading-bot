"""
Mean Reversion Bot (Bot B) - Pullback eater for contrarian trading

Entry Logic:
- Flash detection: z-score or percent drop 0.08-0.12 in 5-10s
- z_threshold: 2.5 on 30-60s returns
- Cause filter: suppress if spot moved same direction >0.25% in window
- OB confirm: depth-weighted imbalance >0.7 favors revert direction
- Time gating: skip if <90s left on 5m

Execution:
- Maker-first: post-only at mid - r (r = 0.01-0.02)
- Escalate to taker if drop exceeds min_drop by 50% and OB confirms
- FOK with tight slippage

Exits:
- Wide initial SL: -18% early, -15% mid, -30% mercy
- Partial TP: +20-22% (sell 30%), +40-45% (sell 30%), hold 30%

Sizing:
- f = min(0.035, 0.20 × Kelly_estimate)
- Suppress if VPIN > 0.6 or fragility high

Usage:
    from bots.mean_reversion_bot import MeanReversionBot
    
    config = MeanReversionConfig(bankroll=100)
    bot = MeanReversionBot(trading_bot, shared_state, config)
    
    signal = await bot.check_entry()
    if signal:
        result = await bot.execute(signal)
"""

import asyncio
import logging
import time
import statistics
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Deque
from collections import deque
from enum import Enum

from lib.shared_state import SharedState, MarketRegime

logger = logging.getLogger(__name__)


class ReversionSide(str, Enum):
    """Mean reversion direction."""
    UP = "up"      # Buy the dip (price dropped, expect up)
    DOWN = "down"  # Sell the rip (price spiked, expect down)


@dataclass
class ReversionSignal:
    """Entry signal from mean reversion detection."""
    side: ReversionSide
    confidence: float
    entry_price: float
    reason: str
    drop_magnitude: float      # How much price moved
    z_score: float            # Statistical measure
    ob_confidence: float      # Orderbook support


@dataclass
class ReversionExit:
    """Exit parameters for mean reversion trade."""
    tp1_price: float
    tp2_price: float
    initial_sl: float
    tp1_size_pct: float = 0.30
    tp2_size_pct: float = 0.30
    hold_pct: float = 0.40
    early_sl_pct: float = 0.40
    mid_sl_pct: float = 0.35
    mercy_sl_pct: float = 0.30


@dataclass
class MeanReversionConfig:
    """Configuration for mean reversion bot."""
    
    # Bankroll and sizing
    bankroll: float = 100.0
    max_position_pct: float = 0.05
    kelly_multiplier: float = 0.20
    
    # Flash detection (P1-5: aligned with strategy spec)
    min_drop: float = 0.10          # 10% drop threshold
    min_drop_window: int = 8        # seconds
    z_threshold: float = 2.5        # Z-score threshold (strategy spec: 2.5)
    z_window: int = 60              # seconds for z-score

    # Dynamic thresholds by regime
    high_vol_drop: float = 0.12     # 12% in high vol (strategy spec)
    low_vol_drop: float = 0.08      # 8% in low vol (strategy spec)
    
    # Cause filter
    cause_filter_threshold: float = 0.0025  # 0.25% spot move blocks
    
    # OB confirmation
    ob_imbalance_threshold: float = 0.60   # Favor revert direction (lowered)
    decay: float = 0.70
    spoof_cancel_threshold: float = 0.6
    
    # Time gating
    min_time_remaining: int = 90     # Skip if <90s left
    
    # Execution
    maker_spread_offset: float = 0.015  # 1.5c from mid
    maker_improve_ticks: int = 1
    maker_improve_wait: float = 4.0   # seconds
    taker_escalation_threshold: float = 1.5  # 50% past min_drop
    
    # Exits
    tp1_pct: float = 0.22            # +22%
    tp2_pct: float = 0.45            # +45%
    early_sl_pct: float = 0.40       # -40% early window
    mid_sl_pct: float = 0.35         # -35% mid window
    mercy_sl_pct: float = 0.30       # -30% mercy
    time_confirm_sl: float = 10.0       # seconds - no SL in first 10s
    
    # Depth cap
    max_depth_pct: float = 0.25
    
    # VPIN suppression
    vpin_suppress_threshold: float = 0.6
    
    # Cooldown
    cooldown_seconds: int = 60
    min_signal_strength: float = 0.3


@dataclass
class MeanReversionPosition:
    """Active mean reversion position."""
    side: ReversionSide
    entry_price: float
    size: float
    entry_time: float
    window_start: int         # Time remaining at entry (seconds)
    sl_price: float = 0.0
    tp1_filled: bool = False
    tp2_filled: bool = False
    # Limit orders for automated exits
    sl_order_id: Optional[str] = None
    tp1_order_id: Optional[str] = None
    tp2_order_id: Optional[str] = None
    # Throttle: last time we polled order status via HTTP (P2-1)
    _last_order_poll: float = 0.0


class MeanReversionBot:
    """
    Mean Reversion Bot - catches pullbacks and reversals.
    
    Uses adaptive flash detection, OB confirmation, and
    post-only maker execution with taker escalation.
    """
    
    def __init__(
        self,
        trading_bot,
        shared_state: SharedState,
        config: Optional[MeanReversionConfig] = None,
    ):
        self.bot = trading_bot
        self.state = shared_state
        self.config = config if config else MeanReversionConfig()
        
        # Position tracking
        self._position: Optional[MeanReversionPosition] = None
        self._last_entry_time: float = 0
        self._balance_block_until: float = 0  # Cooldown after balance errors
        self._failed_close_attempts: int = 0  # Track failed close attempts

        # Price history for z-score
        self._price_history: Deque = deque(maxlen=120)  # 2 min at 1s

        # P3-4: per-side price history as explicit typed attributes (not dynamic hasattr)
        self._side_history_up: Deque = deque(maxlen=120)
        self._side_history_down: Deque = deque(maxlen=120)

        # Maker order tracking (P1-2: two-phase maker-first execution)
        self._pending_maker_order: Optional[Dict] = None
        self._maker_order_time: float = 0

        # Metrics
        self._entries_today: int = 0
        self._wins_today: int = 0
    
    @property
    def has_position(self) -> bool:
        return self._position is not None
    
    @property
    def position(self) -> Optional[MeanReversionPosition]:
        return self._position
    
    async def check_entry(self) -> Optional[ReversionSignal]:
        """Check for mean reversion entry signal."""
        # Check cooldown
        if time.time() - self._last_entry_time < self.config.cooldown_seconds:
            return None
        
        # Check if position already open
        if self.has_position:
            return None
        
        # Check time remaining
        market = self.state.get_market_data()
        if market.time_to_expiry < self.config.min_time_remaining:
            logger.debug("MR: too close to expiry (%ds)", market.time_to_expiry)
            return None
        
        risk = self.state.get_risk_metrics()
        
        # Check VPIN suppression
        if risk.vpin > self.config.vpin_suppress_threshold:
            logger.debug("MR: blocked by VPIN %.2f", risk.vpin)
            return None
        
        # Check fragility
        if risk.fragility > 0.6:
            logger.debug("MR: blocked by fragility %.2f", risk.fragility)
            return None
        
        # Get thresholds based on regime
        min_drop = self.config.min_drop
        if risk.regime == MarketRegime.HIGH:
            min_drop = self.config.high_vol_drop
        elif risk.regime == MarketRegime.LOW:
            min_drop = self.config.low_vol_drop
        
        # Calculate price changes
        up_price = market.up_price
        down_price = market.down_price
        
        # Check for UP reversion (price dropped, expect bounce)
        signal_up = await self._check_reversion_side(
            up_price, ReversionSide.UP, min_drop, risk
        )
        
        # Check for DOWN reversion (price spiked, expect dump)
        signal_down = await self._check_reversion_side(
            down_price, ReversionSide.DOWN, min_drop, risk
        )
        
        # Return the stronger signal
        if signal_up and signal_down:
            if signal_up.confidence > signal_down.confidence:
                return signal_up
            return signal_down
        elif signal_up:
            return signal_up
        elif signal_down:
            return signal_down
        
        return None
    
    async def _check_reversion_side(
        self,
        current_price: float,
        side: ReversionSide,
        min_drop: float,
        risk,
    ) -> Optional[ReversionSignal]:
        """Check for reversion signal on specific side."""

        # P3-4: use explicit typed deques instead of dynamic hasattr/setattr
        history: Deque = self._side_history_up if side == ReversionSide.UP else self._side_history_down
        history.append({'price': current_price, 'timestamp': time.time()})

        if len(history) < 5:
            return None

        # Calculate peak-to-trough drop within the detection window
        now = time.time()
        window_prices = [
            h['price'] for h in history
            if now - h['timestamp'] <= self.config.min_drop_window
        ]

        if len(window_prices) < 3:
            return None

        peak = max(window_prices)
        trough = min(window_prices)
        drop = (peak - trough) / peak if peak > 0 else 0

        if drop < min_drop:
            logger.debug(f"MR {side.value}: drop {drop*100:.1f}% < threshold {min_drop*100:.1f}%")
            return None

        # P3-3: directional cause filter — only suppress when spot moved in the SAME
        # direction as the Poly move (informed flow). Opposing spot move is healthy.
        spot_change_signed = self.state.get_spot_change(self.config.min_drop_window)
        if side == ReversionSide.UP and spot_change_signed < -self.config.cause_filter_threshold:
            # Spot also went down sharply — informed sellers, not a dislocation
            logger.debug("MR UP: suppressed — informed spot down %.4f", spot_change_signed)
            return None
        if side == ReversionSide.DOWN and spot_change_signed > self.config.cause_filter_threshold:
            # Spot also went up sharply — informed buyers, not a dislocation
            logger.debug("MR DOWN: suppressed — informed spot up %.4f", spot_change_signed)
            return None

        # Calculate z-score
        prices = [h['price'] for h in history if now - h['timestamp'] <= self.config.z_window]
        if len(prices) < 10:
            z_score = 0.0
        else:
            mean = statistics.mean(prices)
            stdev = statistics.stdev(prices) if len(prices) > 1 else 0.001
            z_score = abs((current_price - mean) / stdev) if stdev > 0 else 0.0

        if z_score < self.config.z_threshold:
            logger.debug(f"MR {side.value}: z={z_score:.1f} < {self.config.z_threshold}")
            return None

        # P1-6: OB confirmation — both sides use (1 - imbalance).
        # imbalance = ask_depth / total_depth on the UP token.
        # Lower imbalance = more bids = buyers present = price support.
        # This is valid for BOTH UP (buy the dip) and DOWN reversion (UP spike
        # exhausting = sellers overwhelming bids on UP token = DOWN token recovering).
        imbalance = self.state.get_imbalance()
        ob_support = 1 - imbalance

        if ob_support < self.config.ob_imbalance_threshold:
            logger.debug(f"MR {side.value}: ob={ob_support:.2f} < {self.config.ob_imbalance_threshold}")
            return None

        # Calculate confidence
        drop_strength = min(1.0, drop / (min_drop * 2))
        z_strength = min(1.0, z_score / (self.config.z_threshold * 2))
        ob_strength = min(1.0, ob_support / 0.9)
        vpin_factor = max(0.3, 1 - risk.vpin)

        confidence = (drop_strength * 0.3 + z_strength * 0.3 + ob_strength * 0.4) * vpin_factor

        if confidence < self.config.min_signal_strength:
            logger.debug(f"MR {side.value}: conf={confidence:.2f} < {self.config.min_signal_strength}")
            return None

        market = self.state.get_market_data()

        return ReversionSignal(
            side=side,
            confidence=confidence,
            entry_price=market.mid_price,
            reason=f"drop {drop*100:.1f}% z={z_score:.1f} ob={ob_support:.2f}",
            drop_magnitude=drop,
            z_score=z_score,
            ob_confidence=ob_support,
        )
    
    def calculate_size(self, signal: ReversionSignal) -> float:
        """Calculate position size in shares (min $1 value, max 10 shares)."""
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
    
    async def execute(self, signal: ReversionSignal, size_override: Optional[float] = None) -> Dict[str, Any]:
        """Execute mean reversion entry using maker-first, taker-escalation approach (P1-2).

        Phase 1: Post GTX (post-only) at maker_price.
        Phase 2: If drop exceeds taker_escalation_threshold and OB still confirms,
                 cancel maker and submit FOK (handled by check_maker_escalation()).

        Args:
            signal: Entry signal.
            size_override: VPIN-adjusted size from coordinator (P1-7).
        """
        if time.time() < self._balance_block_until:
            logger.debug("MR: blocked by balance cooldown")
            return {"success": False, "reason": "balance cooldown"}

        size = size_override if size_override is not None else self.calculate_size(signal)

        if size < 5.0:
            return {"success": False, "reason": "size too small"}

        market = self.state.get_market_data()

        if signal.side == ReversionSide.UP:
            token_id = market.token_id_up
            side = "BUY"
            maker_price = max(0.01, min(0.99, market.mid_price - self.config.maker_spread_offset))
        else:
            token_id = market.token_id_down
            side = "BUY"
            maker_price = max(0.01, min(0.99, market.mid_price + self.config.maker_spread_offset))

        window_remaining = market.time_to_expiry
        sl_pct = self.config.early_sl_pct if window_remaining > 180 else self.config.mid_sl_pct

        # Phase 1: attempt post-only (GTX) maker order
        try:
            maker_result = await self.bot.place_order(
                token_id=token_id,
                price=round(maker_price, 4),
                size=size,
                side=side,
                order_type="GTX",
            )

            if maker_result.success:
                # Maker order resting — store for escalation check
                self._pending_maker_order = {
                    "order_id": maker_result.order_id,
                    "token_id": token_id,
                    "side": side,
                    "size": size,
                    "price": maker_price,
                    "signal": signal,
                    "sl_pct": sl_pct,
                    "window_remaining": window_remaining,
                }
                self._maker_order_time = time.time()
                logger.info(
                    f"==> MR MAKER POSTED: {signal.side.value.upper()} | {size:.2f} shares @ {maker_price:.3f} | "
                    f"waiting for fill (escalation in {self.config.maker_improve_wait:.0f}s)"
                )
                return {"success": True, "pending_maker": True, "reason": "maker_posted"}

            # GTX rejected (e.g. would cross spread) — escalate directly to taker
            logger.info("MR: GTX rejected (%s), escalating to FOK taker", maker_result.message)

        except Exception as e:
            logger.error("MR: maker order failed: %s", e)

        # Phase 2 fallback: taker FOK
        return await self._execute_taker(signal, token_id, side, size, sl_pct, window_remaining)

    async def _execute_taker(
        self,
        signal: ReversionSignal,
        token_id: str,
        side: str,
        size: float,
        sl_pct: float,
        window_remaining: int,
    ) -> Dict[str, Any]:
        """Submit a FOK taker order and record the position on fill."""
        max_retries = 3
        for attempt in range(max_retries):
            market = self.state.get_market_data()
            exec_price = round(
                market.up_price if signal.side == ReversionSide.UP else market.down_price, 4
            )
            try:
                result = await self.bot.place_order(
                    token_id=token_id,
                    price=exec_price,
                    size=size,
                    side=side,
                    order_type="FOK",
                )

                if result.success:
                    return self._record_fill(signal, exec_price, size, sl_pct, window_remaining)

                is_balance_error = result.message and (
                    "balance" in result.message.lower() or "allowance" in result.message.lower()
                )
                is_fok_error = result.message and (
                    "couldn't be fully filled" in result.message.lower()
                    or "not enough liquidity" in result.message.lower()
                )

                if is_balance_error:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    self._balance_block_until = time.time() + 10
                    logger.warning("MR: balance error on taker entry, blocking 10s")

                if is_fok_error and attempt < max_retries - 1:
                    await asyncio.sleep(1 + attempt * 0.5)
                    continue

                return {"success": False, "reason": result.message}

            except Exception as e:
                logger.error("MR taker execution failed: %s", e)
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                return {"success": False, "reason": str(e)}

        return {"success": False, "reason": "max retries exceeded"}

    def _record_fill(
        self,
        signal: ReversionSignal,
        actual_entry: float,
        size: float,
        sl_pct: float,
        window_remaining: int,
    ) -> Dict[str, Any]:
        """Record a confirmed fill and schedule exit order placement."""
        tp1_actual = actual_entry * (1 + self.config.tp1_pct)
        tp2_actual = actual_entry * (1 + self.config.tp2_pct)
        sl_actual = actual_entry * (1 - sl_pct)

        self._position = MeanReversionPosition(
            side=signal.side,
            entry_price=actual_entry,
            size=size,
            entry_time=time.time(),
            window_start=window_remaining,
            sl_price=sl_actual,
        )

        self._last_entry_time = time.time()
        self._entries_today += 1
        self._pending_maker_order = None  # Clear any pending maker state

        exits_obj = ReversionExit(
            tp1_price=tp1_actual,
            tp2_price=tp2_actual,
            initial_sl=sl_actual,
            early_sl_pct=self.config.early_sl_pct,
            mid_sl_pct=self.config.mid_sl_pct,
            mercy_sl_pct=self.config.mercy_sl_pct,
        )

        # P2-2: schedule exit orders as background task — do NOT sleep here
        asyncio.create_task(self._place_exit_limit_orders(self._position, exits_obj))

        logger.info(
            f"==> MR ENTERED: {signal.side.value.upper()} | ${size:.2f} @ {actual_entry:.3f} | "
            f"TP1: {tp1_actual:.3f} TP2: {tp2_actual:.3f} SL: {sl_actual:.3f}"
        )

        return {"success": True, "position": self._position, "exits": exits_obj}

    async def check_maker_escalation(self) -> Optional[Dict[str, Any]]:
        """Check pending maker order — improve price or escalate to taker (P1-2).

        Called from the coordination loop alongside check_exit().
        """
        if not self._pending_maker_order or self.has_position:
            return None

        order = self._pending_maker_order
        elapsed = time.time() - self._maker_order_time

        clob = self.bot.clob_client

        # Check if the maker order already filled
        try:
            status = await self.bot._run_in_thread(clob.get_order, order["order_id"])
            if status.get("status") == "filled":
                filled_size = float(status.get("size_matched", status.get("size", 0)))
                signal = order["signal"]
                logger.info(f"MR: maker order filled {filled_size} shares")
                return self._record_fill(
                    signal, order["price"], filled_size, order["sl_pct"], order["window_remaining"]
                )
        except Exception as e:
            logger.debug(f"MR: error polling maker order: {e}")

        # Not filled yet — check if we should improve or escalate
        if elapsed < self.config.maker_improve_wait:
            return None  # Still waiting

        # Try improving by 1 tick first (one improvement attempt only)
        if elapsed < self.config.maker_improve_wait * 2:
            signal = order["signal"]
            improved_price = round(order["price"] + 0.01, 4) if signal.side == ReversionSide.UP else round(order["price"] - 0.01, 4)
            improved_price = max(0.01, min(0.99, improved_price))
            try:
                await self.bot._run_in_thread(clob.cancel_order, order["order_id"])
                improved = await self.bot.place_order(
                    token_id=order["token_id"],
                    price=improved_price,
                    size=order["size"],
                    side=order["side"],
                    order_type="GTX",
                )
                if improved.success:
                    self._pending_maker_order["order_id"] = improved.order_id
                    self._pending_maker_order["price"] = improved_price
                    self._maker_order_time = time.time()  # Reset timer for second wait
                    logger.info(f"MR: improved maker order to {improved_price:.3f}")
                    return None
            except Exception as e:
                logger.debug(f"MR: maker improve failed: {e}")

        # Escalate to taker: check if drop now exceeds escalation threshold and OB confirms
        signal = order["signal"]
        market = self.state.get_market_data()
        risk = self.state.get_risk_metrics()

        min_drop = self.config.min_drop
        if risk.regime.value == "high":
            min_drop = self.config.high_vol_drop
        elif risk.regime.value == "low":
            min_drop = self.config.low_vol_drop

        history: Deque = self._side_history_up if signal.side == ReversionSide.UP else self._side_history_down
        now = time.time()
        window_prices = [h['price'] for h in history if now - h['timestamp'] <= self.config.min_drop_window]
        if window_prices:
            peak = max(window_prices)
            trough = min(window_prices)
            current_drop = (peak - trough) / peak if peak > 0 else 0
            imbalance = self.state.get_imbalance()
            ob_support = 1 - imbalance

            if current_drop >= min_drop * self.config.taker_escalation_threshold and ob_support >= self.config.ob_imbalance_threshold:
                # Cancel maker and escalate to taker
                try:
                    await self.bot._run_in_thread(clob.cancel_order, order["order_id"])
                except Exception:
                    pass
                self._pending_maker_order = None
                logger.info(f"MR: escalating to taker (drop {current_drop*100:.1f}% ≥ {min_drop*self.config.taker_escalation_threshold*100:.1f}%)")
                sl_pct = order["sl_pct"]
                window_remaining = order["window_remaining"]
                return await self._execute_taker(signal, order["token_id"], order["side"], order["size"], sl_pct, window_remaining)

        # Drop not sufficient for escalation — cancel and give up
        try:
            await self.bot._run_in_thread(clob.cancel_order, order["order_id"])
        except Exception:
            pass
        self._pending_maker_order = None
        logger.info("MR: maker order expired, no escalation (drop insufficient)")
        return None
    
    async def _cancel_all_exit_orders(self, pos: MeanReversionPosition):
        """Cancel all live exit limit orders for a position (P0-5)."""
        clob = self.bot.clob_client
        for oid in [pos.sl_order_id, pos.tp1_order_id, pos.tp2_order_id]:
            if oid:
                try:
                    await self.bot._run_in_thread(clob.cancel_order, oid)
                    logger.info(f"MR: cancelled exit order {oid}")
                except Exception as e:
                    logger.debug(f"MR: could not cancel order {oid}: {e}")

    async def _place_exit_limit_orders(self, pos: MeanReversionPosition, exits: ReversionExit):
        """Place GTC limit orders for SL and TPs concurrently after entry (P0-3, P2-5)."""
        try:
            market = self.state.get_market_data()
            token_id = market.token_id_up if pos.side == ReversionSide.UP else market.token_id_down
            side = "SELL"

            sl_price_r = round(pos.sl_price, 4)
            tp1_price_r = round(pos.entry_price * (1 + self.config.tp1_pct + 0.01), 4)
            tp2_price_r = round(pos.entry_price * (1 + self.config.tp2_pct + 0.01), 4)

            logger.info(f"MR: placing exit orders for {pos.size} shares @ entry {pos.entry_price:.3f}")

            # P0-3: SL covers 100% of position; TP1/TP2 cover 30% each (40% hold rides)
            sl_result, tp1_result, tp2_result = await asyncio.gather(
                self.bot.place_order(token_id=token_id, price=sl_price_r,  size=pos.size,        side=side, order_type="GTC"),
                self.bot.place_order(token_id=token_id, price=tp1_price_r, size=pos.size * 0.30, side=side, order_type="GTC"),
                self.bot.place_order(token_id=token_id, price=tp2_price_r, size=pos.size * 0.30, side=side, order_type="GTC"),
                return_exceptions=True,
            )

            if not isinstance(sl_result, Exception) and sl_result.success:
                pos.sl_order_id = sl_result.order_id
                logger.info(f"MR: SL order {sl_result.order_id} @ {sl_price_r:.3f} (100%)")
            else:
                logger.warning(f"MR: SL order failed: {sl_result}")
            if not isinstance(tp1_result, Exception) and tp1_result.success:
                pos.tp1_order_id = tp1_result.order_id
                logger.info(f"MR: TP1 order {tp1_result.order_id} @ {tp1_price_r:.3f} (30%)")
            else:
                logger.warning(f"MR: TP1 order failed: {tp1_result}")
            if not isinstance(tp2_result, Exception) and tp2_result.success:
                pos.tp2_order_id = tp2_result.order_id
                logger.info(f"MR: TP2 order {tp2_result.order_id} @ {tp2_price_r:.3f} (30%)")
            else:
                logger.warning(f"MR: TP2 order failed: {tp2_result}")

        except Exception as e:
            logger.error(f"MR: failed to place exit limit orders: {e}", exc_info=True)
    
    async def _cancel_remaining_orders(self, pos: MeanReversionPosition, reason: str):
        """Cancel remaining limit orders after one fills."""
        clob = self.bot.clob_client
        orders_to_cancel = []
        
        if reason == "SL":
            # Cancel TPs if SL hit
            orders_to_cancel = [pos.tp1_order_id, pos.tp2_order_id]
        elif reason in ["TP1", "TP2"]:
            # Cancel SL and other TP if one TP hit
            orders_to_cancel = [pos.sl_order_id]
            if reason == "TP1":
                orders_to_cancel.append(pos.tp2_order_id)
            elif reason == "TP2":
                orders_to_cancel.append(pos.tp1_order_id)
        
        for order_id in orders_to_cancel:
            if order_id:
                try:
                    await self.bot._run_in_thread(clob.cancel_order, order_id)
                    logger.info(f"MR: cancelled {order_id} after {reason}")
                except Exception as e:
                    logger.debug(f"MR: failed to cancel order {order_id}: {e}")
    
    async def _check_limit_orders(self, pos: MeanReversionPosition) -> Optional[Dict[str, Any]]:
        """Check if any limit orders have been filled.

        Throttled to every 5 seconds (P2-1) to avoid flooding the REST API.
        """
        if not pos.sl_order_id and not pos.tp1_order_id and not pos.tp2_order_id:
            return None

        # P2-1: throttle HTTP polling to every 5 seconds
        now = time.time()
        if now - pos._last_order_poll < 5.0:
            return None
        pos._last_order_poll = now

        try:
            clob = self.bot.clob_client

            # Check SL order
            if pos.sl_order_id:
                result = await self.bot._run_in_thread(clob.get_order, pos.sl_order_id)
                if result.get("status") == "filled":
                    filled_size = float(result.get("size", 0))
                    if filled_size > 0:
                        logger.info(f"MR: SL filled {filled_size} shares")
                        oid = pos.sl_order_id
                        pos.sl_order_id = None  # P0-4: clear immediately
                        await self._cancel_remaining_orders(pos, "SL")
                        return {"action": "sell", "size": filled_size, "reason": "SL", "order_id": oid}

            # Check TP1 order — ratchet: sell 30%, new TP2 at config level, SL at break-even
            if pos.tp1_order_id:
                result = await self.bot._run_in_thread(clob.get_order, pos.tp1_order_id)
                if result.get("status") == "filled":
                    filled_size = float(result.get("size", 0))
                    if filled_size > 0:
                        pos.tp1_filled = True
                        oid = pos.tp1_order_id
                        pos.tp1_order_id = None  # P0-4: clear immediately
                        logger.info(f"MR: TP1 filled {filled_size} shares")

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

                        # P2-6: use config tp2_pct for ratcheted TP2; P2-5: gather
                        market = self.state.get_market_data()
                        token_id = market.token_id_up if pos.side == ReversionSide.UP else market.token_id_down
                        new_tp2_price = round(pos.entry_price * (1 + self.config.tp2_pct) * 1.005, 4)
                        remaining_size = round(pos.size * 0.40, 2)  # 40% hold portion

                        new_tp2, new_sl = await asyncio.gather(
                            self.bot.place_order(token_id=token_id, price=new_tp2_price, size=remaining_size, side="SELL", order_type="GTC"),
                            self.bot.place_order(token_id=token_id, price=round(pos.entry_price, 4), size=remaining_size, side="SELL", order_type="GTC"),
                            return_exceptions=True,
                        )
                        if not isinstance(new_tp2, Exception) and new_tp2.success:
                            pos.tp2_order_id = new_tp2.order_id
                        if not isinstance(new_sl, Exception) and new_sl.success:
                            pos.sl_order_id = new_sl.order_id

                        logger.info(f"MR: TP1 done — new TP2 @ {new_tp2_price:.3f}, SL @ break-even")
                        return {"action": "sell", "size": filled_size, "reason": "TP1", "order_id": oid}

            # Check TP2 order — let hold portion ride, SL to TP1 price
            if pos.tp2_order_id:
                result = await self.bot._run_in_thread(clob.get_order, pos.tp2_order_id)
                if result.get("status") == "filled":
                    filled_size = float(result.get("size", 0))
                    if filled_size > 0:
                        pos.tp2_filled = True
                        oid = pos.tp2_order_id
                        pos.tp2_order_id = None  # P0-4: clear immediately
                        logger.info(f"MR: TP2 filled {filled_size} shares, letting hold ride")

                        if pos.sl_order_id:
                            try:
                                await self.bot._run_in_thread(clob.cancel_order, pos.sl_order_id)
                                pos.sl_order_id = None
                            except Exception:
                                pass

                        # P2-7: SL at TP1 level for the 40% hold portion only
                        market = self.state.get_market_data()
                        token_id = market.token_id_up if pos.side == ReversionSide.UP else market.token_id_down
                        tp1_price = round(pos.entry_price * (1 + self.config.tp1_pct), 4)
                        hold_size = round(pos.size * 0.40, 2)

                        new_sl = await self.bot.place_order(
                            token_id=token_id,
                            price=tp1_price,
                            size=hold_size,
                            side="SELL",
                            order_type="GTC",
                        )
                        if new_sl.success:
                            pos.sl_order_id = new_sl.order_id

                        logger.info(f"MR: TP2 done — hold riding with SL @ {tp1_price:.3f}")
                        return {"action": "sell", "size": filled_size, "reason": "TP2", "order_id": oid}

            return None
        except Exception as e:
            logger.debug(f"MR: error checking limit orders: {e}")
            return None
    
    async def check_exit(self) -> Optional[Dict[str, Any]]:
        """Check if position should be exited."""
        if not self._position:
            return None
        
        # Wait minimum time before checking exits - allows balance to settle
        time_held = time.time() - self._position.entry_time
        min_hold_seconds = 3.0  # Reduced from 15s for faster exits
        if time_held < min_hold_seconds:
            logger.debug(f"MR: waiting for settlement ({time_held:.1f}s < {min_hold_seconds}s)")
            return None
        
        # Check if limit orders have been filled first
        limit_result = await self._check_limit_orders(self._position)
        if limit_result:
            return limit_result
        
        market = self.state.get_market_data()
        
        # Track last logged PnL to avoid spam
        if not hasattr(self, '_last_logged_pnl'):
            self._last_logged_pnl = None
        
        current_price = market.up_price if self._position.side == ReversionSide.UP else market.down_price
        
        pos = self._position
        
        # Calculate PnL
        position_value = current_price * pos.size
        entry_value = pos.entry_price * pos.size
        pnl_pct = (position_value - entry_value) / entry_value if entry_value > 0 else 0
        pnl_dollars = position_value - entry_value
        
        # Only log on significant PnL change (>5%) or every ~10 seconds
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
        
        # Recalculate position_value for exit checks
        position_value = current_price * pos.size
        
        # Determine SL based on window
        window_remaining = market.time_to_expiry
        if pos.window_start > 180:
            # Early window
            sl_pct = self.config.early_sl_pct
        else:
            # Mid window
            sl_pct = self.config.mid_sl_pct
        
        # Update SL in case window changed - both use same formula now
        # SL = entry * (1 - sl_pct) means price dropping below entry = loss
        pos.sl_price = pos.entry_price * (1 - sl_pct)
        
        # --- Fallback price-level exits (only when limit orders are NOT active) ---
        # P0-4: gate on order IDs being None to avoid double-sell races.

        # SL — fallback when no active SL limit order
        if pos.sl_order_id is None and position_value >= 0.10:
            if current_price <= pos.sl_price:
                return {"action": "sell", "size": pos.size, "reason": "SL"}

        # Partial TP1 — fallback when no active TP1 limit order
        if not pos.tp1_filled and pos.tp1_order_id is None and position_value >= 0.10:
            tp1_threshold = pos.entry_price * (1 + self.config.tp1_pct + 0.01)
            logger.debug(f"MR TP check: entry={pos.entry_price:.3f} cur={current_price:.3f} tp1_thresh={tp1_threshold:.3f}")
            if current_price >= tp1_threshold:
                logger.warning(f"MR TP1 TRIGGERED (fallback): entry={pos.entry_price:.3f} cur={current_price:.3f}")
                pos.tp1_filled = True
                return {"action": "sell", "size": pos.size * 0.30, "reason": "TP1"}

        # Partial TP2 — fallback when no active TP2 limit order
        if not pos.tp2_filled and pos.tp2_order_id is None and position_value >= 0.10:
            tp2_threshold = pos.entry_price * (1 + self.config.tp2_pct + 0.01)
            if current_price >= tp2_threshold:
                logger.warning(f"MR TP2 TRIGGERED (fallback): entry={pos.entry_price:.3f} cur={current_price:.3f}")
                pos.tp2_filled = True
                return {"action": "sell", "size": pos.size * 0.30, "reason": "TP2"}
        
        # Late window: mercy stop (always closes 100%)
        # Both UP and DOWN use same formula: price below entry = loss
        if window_remaining < 45:
            mercy_price = pos.entry_price * (1 - self.config.mercy_sl_pct)
            if current_price <= mercy_price:
                return {"action": "sell", "size": pos.size, "reason": "mercy"}
        
        # Expiry close - sell at market (always closes 100%)
        if window_remaining <= 5:
            logger.info("MR: expiry close at market")
            return {"action": "sell", "size": pos.size, "reason": "expiry"}
        
        return None
    
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
            logger.debug(f"MR: waiting {wait_time:.1f}s for token settlement")
            await asyncio.sleep(wait_time)
        
        market = self.state.get_market_data()
        
        # Determine size to sell (default to full position)
        if sell_size is None:
            sell_size = self._position.size
        
        # Round to 2 decimals (FOK requirement)
        sell_size = int(sell_size * 100) / 100.0
        if sell_size < 0.01:
            return {"success": False, "reason": "size too small"}
        
        if self._position.side == ReversionSide.UP:
            token_id = market.token_id_up
            side = "SELL"
            # Use best bid for better FOK fill, fallback to mid
            if market.up_bids:
                price = market.up_bids[0][0]  # Best bid
            else:
                price = market.up_price
        else:
            token_id = market.token_id_down
            side = "SELL"  # Must SELL to close a DOWN position
            # Use best bid for better FOK fill, fallback to mid
            if market.down_bids:
                price = market.down_bids[0][0]  # Best bid
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
                
                # Update position size (for partial sells)
                self._position.size -= sell_size
                self._failed_close_attempts = 0  # Reset on success
                
                # Clear position if fully sold
                if self._position.size < 0.01:
                    self._position = None
                
                pnl_str = f"+${pnl:.2f}" if pnl > 0 else f"-${abs(pnl):.2f}"
                logger.info(f"==> MR CLOSED: {reason.upper()} | {sell_size:.2f} shares | PnL: {pnl_str}")
                
                return {"success": True, "pnl": pnl}
            else:
                # Any close failure - increment counter
                self._failed_close_attempts += 1
                
                # Check for balance/allowance error
                if result.message and ("balance" in result.message.lower() or "allowance" in result.message.lower()):
                    self._balance_block_until = time.time() + 3
                    logger.warning("MR close: balance error, attempt %d", self._failed_close_attempts)
                else:
                    logger.warning("MR close: FOK failed (%s), attempt %d", result.message, self._failed_close_attempts)
                
                # Force clear after 5 failed attempts — cancel GTC orders first (P0-5)
                if self._failed_close_attempts >= 5:
                    logger.error("MR: force clearing position after 5 failed close attempts")
                    if self._position:
                        await self._cancel_all_exit_orders(self._position)
                    self._position = None
                    self._failed_close_attempts = 0

                return {"success": False, "reason": result.message}

        except Exception as e:
            logger.error("MeanReversion close failed: %s", e)
            self._failed_close_attempts += 1
            if self._failed_close_attempts >= 5:
                logger.error("MR: force clearing position after 5 failed close attempts")
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
