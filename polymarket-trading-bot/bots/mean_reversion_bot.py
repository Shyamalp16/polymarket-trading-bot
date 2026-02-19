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
    early_sl_pct: float = 0.20
    mid_sl_pct: float = 0.15
    mercy_sl_pct: float = 0.10


@dataclass
class MeanReversionConfig:
    """Configuration for mean reversion bot."""
    
    # Bankroll and sizing
    bankroll: float = 100.0
    max_position_pct: float = 0.05
    kelly_multiplier: float = 0.20
    
    # Flash detection — thresholds calibrated to observed Poly token dislocations
    min_drop: float = 0.05          # 5% NORMAL (was 10% — missed 7.1% moves)
    min_drop_window: int = 8        # seconds
    z_threshold: float = 1.8        # was 2.5 — 5m windows have thin history
    z_window: int = 60              # seconds for z-score

    # Dynamic thresholds by regime
    high_vol_drop: float = 0.07     # 7% HIGH vol (was 12%)
    low_vol_drop: float = 0.04      # 4% LOW vol  (was 8%)
    
    # Cause filter
    cause_filter_threshold: float = 0.0025  # 0.25% spot move blocks
    
    # OB confirmation — (1 − imbalance) must exceed this.
    # Default imbalance = 0.5, so ob_support = 0.5. Old threshold 0.60 always blocked.
    ob_imbalance_threshold: float = 0.40
    decay: float = 0.70
    spoof_cancel_threshold: float = 0.6

    # Time gating
    min_time_remaining: int = 60     # Skip if <60s left (was 90s)
    
    # Execution
    maker_spread_offset: float = 0.015  # 1.5c from mid
    maker_improve_ticks: int = 1
    maker_improve_wait: float = 4.0   # seconds
    taker_escalation_threshold: float = 1.5  # 50% past min_drop
    
    # Exits
    tp1_pct: float = 0.22            # +22%
    tp2_pct: float = 0.45            # +45%
    tp3_price: float = 0.90          # absolute price level — sell 100% at ≥90¢
    trailing_sl_pct: float = 0.15    # trail 15% below peak — activates after TP1
    early_sl_pct: float = 0.20       # -20% early window  (was 40% — never fired)
    mid_sl_pct: float = 0.15         # -15% mid window    (was 35% — never fired)
    mercy_sl_pct: float = 0.10       # -10% last 45 s     (was 30% — never fired)
    time_confirm_sl: float = 10.0       # seconds - no SL in first 10s
    
    # Depth cap
    max_depth_pct: float = 0.25
    
    # VPIN suppression
    vpin_suppress_threshold: float = 0.6

    # Cooldown
    cooldown_seconds: int = 60
    min_signal_strength: float = 0.45


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
    tp3_filled: bool = False
    # Token ID at the time of entry.  After a market rotation the shared-state
    # token_id changes, so close_position() MUST use this stored value — not the
    # live market token — to avoid selling the wrong (new-market) token.
    entry_token_id: str = ""
    # Snipe flag: if True, this position was opened by expiry_snipe and should
    # NEVER be force-closed by the mercy-SL or expiry-close logic — it holds
    # to natural on-chain resolution at $1.00.
    is_snipe: bool = False
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

        # Entry lock — True while execute() is running (including taker retries).
        # Prevents the coordinator from starting a second concurrent entry during
        # the await sleeps inside _execute_taker().
        self._entry_in_progress: bool = False

        # Metrics
        self._entries_today: int = 0
        self._wins_today: int = 0
        self._realized_pnl: float = 0.0   # cumulative realized PnL this session
        self._window_pnl: float = 0.0     # realized PnL for current market window
    
    @property
    def has_position(self) -> bool:
        # Treat a pending (resting) maker order as "having a position" so the
        # coordinator never approves a second entry while the GTX is waiting.
        return self._position is not None or self._pending_maker_order is not None
    
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

        # Measure peak-to-CURRENT drop within the detection window.
        # Using current_price instead of trough ensures we only enter while
        # the price is STILL at the dislocation level, not after it has already
        # bounced — trough-based measurement would fire stale signals.
        now = time.time()
        window_prices = [
            h['price'] for h in history
            if now - h['timestamp'] <= self.config.min_drop_window
        ]

        if len(window_prices) < 3:
            return None

        peak = max(window_prices)
        drop = (peak - current_price) / peak if peak > 0 else 0

        if drop < min_drop:
            logger.debug(f"MR {side.value}: drop {drop*100:.1f}% < threshold {min_drop*100:.1f}%")
            return None

        # Price zone filter — only trade mid-range tokens.
        # Tokens near 0 or 1 are resolving correctly (not dislocated); MR there is
        # fighting the market's actual assessment, not a flash crash.
        if current_price < 0.18 or current_price > 0.82:
            logger.debug(f"MR {side.value}: price {current_price:.3f} outside tradeable zone [0.18, 0.82]")
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

        # OB confirmation — uses UP-token imbalance = ask_depth / total_depth.
        # UP reversion:   ob_support = 1 - imbalance = UP bid fraction.
        #                 High value → lots of UP buyers → UP price will bounce.
        # DOWN reversion: ob_support = imbalance = UP ask fraction.
        #                 High value → lots of UP sellers (= DOWN buyers) → DOWN price will bounce.
        # The two cases are OPPOSITE because in binary markets selling YES = buying NO.
        imbalance = self.state.get_imbalance()
        ob_support = (1 - imbalance) if side == ReversionSide.UP else imbalance

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

        # Block concurrent calls — the coordinator loop runs every 250ms and can
        # fire a second execute() during the await sleeps inside _execute_taker(),
        # producing duplicate fills that the bot only partially tracks.
        if self._entry_in_progress:
            logger.debug("MR: entry already in progress — blocking concurrent execute()")
            return {"success": False, "reason": "entry in progress"}
        self._entry_in_progress = True

        try:
            return await self._execute_inner(signal, size_override)
        finally:
            self._entry_in_progress = False

    async def _execute_inner(self, signal: ReversionSignal, size_override: Optional[float] = None) -> Dict[str, Any]:
        """Inner execute — called exclusively from execute() which holds the entry lock."""
        # Stamp attempt time NOW — ensures the 60s cooldown applies even if the
        # order fails, preventing instant re-entry on FOK/GTX rejection.
        self._last_entry_time = time.time()

        size = size_override if size_override is not None else self.calculate_size(signal)

        if size < 5.0:
            return {"success": False, "reason": "size too small"}

        market = self.state.get_market_data()

        if signal.side == ReversionSide.UP:
            token_id = market.token_id_up
            side = "BUY"
            # Bid just below UP token's current price to post as maker
            maker_price = max(0.01, min(0.99, market.up_price - self.config.maker_spread_offset))
        else:
            token_id = market.token_id_down
            side = "BUY"
            # Bid just below DOWN token's current price to post as maker
            # (was market.mid_price + offset which used the UP price — always GTX-rejected)
            maker_price = max(0.01, min(0.99, market.down_price - self.config.maker_spread_offset))

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
            # FOK orders must be marketable: BUY needs price >= best ask,
            # SELL needs price <= best bid.  Using the mid price causes every
            # FOK to be rejected with FOK_ORDER_NOT_FILLED_ERROR because the
            # signed makerAmount never crosses the resting offer.
            is_up = signal.side == ReversionSide.UP
            if side == "BUY":
                raw_price = (
                    (market.up_asks[0][0]   if market.up_asks   else market.up_price)
                    if is_up else
                    (market.down_asks[0][0] if market.down_asks else market.down_price)
                )
            else:
                raw_price = (
                    (market.up_bids[0][0]   if market.up_bids   else market.up_price)
                    if is_up else
                    (market.down_bids[0][0] if market.down_bids else market.down_price)
                )
            exec_price = round(min(raw_price, 0.99), 4)
            try:
                result = await self.bot.place_order(
                    token_id=token_id,
                    price=exec_price,
                    size=size,
                    side=side,
                    order_type="FOK",
                )

                if result.success:
                    return self._record_fill(signal, exec_price, size, sl_pct, window_remaining, token_id)

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
        token_id: str = "",
    ) -> Dict[str, Any]:
        """Record a confirmed fill and schedule exit order placement."""
        tp1_actual = actual_entry * (1 + self.config.tp1_pct)
        tp2_actual = actual_entry * (1 + self.config.tp2_pct)
        sl_actual = actual_entry * (1 - sl_pct)
        is_snipe = (signal.reason == "expiry_snipe")

        self._position = MeanReversionPosition(
            side=signal.side,
            entry_price=actual_entry,
            size=size,
            entry_time=time.time(),
            window_start=window_remaining,
            sl_price=sl_actual,
            entry_token_id=token_id,
            is_snipe=is_snipe,
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

    async def _poll_order_status(self, order_id: str) -> Optional[str]:
        """
        Poll order status using two methods:
          1. get_order(order_id)  — specific order lookup
          2. get_open_orders()   — fallback if (1) fails or returns None

        Returns lowercase status string ("matched", "cancelled", "live", …)
        or None if both methods fail.
        """
        # Method 1: direct order lookup
        try:
            data = await self.bot.get_order(order_id)   # wrapper returns None on error
            if data is not None:
                raw = (data.get("status") or "").lower()
                logger.info("MR: get_order status=%s id=%s…", raw, order_id[:8])
                return raw
        except Exception as e:
            logger.warning("MR: get_order exception: %s", e)

        # Method 2: check open orders — if order_id absent, it's no longer live
        try:
            open_orders = await self.bot.get_open_orders()
            open_ids: set = {
                str(o.get("id") or o.get("order_id") or o.get("orderID", ""))
                for o in (open_orders or [])
            }
            logger.info("MR: open_orders check — %d live orders, looking for %s…", len(open_ids), order_id[:8])
            if order_id not in open_ids:
                # Order is gone from open orders — presume matched (safer than ignoring)
                return "matched"
            return "live"
        except Exception as e:
            logger.warning("MR: get_open_orders exception: %s", e)

        return None

    async def _gtx_partial_fill(self, order: Dict) -> Optional[Dict[str, Any]]:
        """
        Check if the GTX maker order has been partially or fully filled.

        Called before canceling a maker order for improvement or escalation.
        If any shares were already lifted by a taker, we record them as the
        position and cancel the remainder — rather than firing a new full-size
        FOK on top, which would orphan the already-filled shares.

        Returns:
            The _record_fill() result dict if partial/full fill detected,
            None if the order has zero fills.
        """
        try:
            data = await self.bot.get_order(order["order_id"])
            filled = float((data or {}).get("size_matched") or 0)
        except Exception as e:
            logger.warning("MR: _gtx_partial_fill: get_order failed (%s) — assuming no fill", e)
            return None

        if filled < 0.01:
            return None

        # Some shares were already on the exchange — record them now and cancel
        # the unfilled remainder so we don't end up with orphaned shares.
        logger.info(
            "MR: GTX partially filled %.2f/%.2f shares before cancel — recording position",
            filled, order["size"],
        )
        clob = self.bot.clob_client
        try:
            await self.bot._run_in_thread(clob.cancel_order, order["order_id"])
        except Exception:
            pass
        self._pending_maker_order = None
        return self._record_fill(
            order["signal"], order["price"], filled,
            order["sl_pct"], order["window_remaining"],
            order["token_id"],
        )

    async def check_maker_escalation(self) -> Optional[Dict[str, Any]]:
        """Check pending maker order — improve price or escalate to taker (P1-2).

        Called from the coordination loop alongside check_exit().
        """
        # Guard: only skip if there's no pending order OR a filled position already exists.
        # Do NOT use has_position here — it returns True when _pending_maker_order is set,
        # which would cause this function to return immediately every time (self-defeating).
        if not self._pending_maker_order or self._position is not None:
            return None

        order = self._pending_maker_order
        now   = time.time()
        elapsed = now - self._maker_order_time

        # ── Throttled fill check (every 5 s to avoid API rate-limits) ──────────
        last_check = order.get("_last_fill_check", 0.0)
        if now - last_check >= 5.0:
            self._pending_maker_order["_last_fill_check"] = now

            order_status = await self._poll_order_status(order["order_id"])

            if order_status is not None and "match" in order_status:
                # ORDER FILLED — record the position
                # Prefer size_matched from get_order; fall back to the size we originally requested.
                try:
                    data = await self.bot.get_order(order["order_id"])
                    filled_size = float((data or {}).get("size_matched") or (data or {}).get("size") or order["size"])
                except Exception:
                    filled_size = float(order["size"])
                logger.info("MR: maker order filled %.2f shares @ %.4f", filled_size, order["price"])
                return self._record_fill(
                    order["signal"], order["price"], filled_size,
                    order["sl_pct"], order["window_remaining"],
                    order["token_id"],
                )

            if order_status is not None and "cancel" in order_status:
                # ORDER CANCELLED by exchange (GTX cross) — nothing to track
                logger.info("MR: maker order cancelled by exchange (GTX cross) — clearing")
                self._pending_maker_order = None
                return None

        # ── Improvement / escalation window ─────────────────────────────────────
        if elapsed < self.config.maker_improve_wait:
            return None  # Still in initial wait window

        clob = self.bot.clob_client

        # Try improving by 1 tick (once only)
        if elapsed < self.config.maker_improve_wait * 2:
            signal = order["signal"]
            improved_price = round(
                order["price"] + 0.01 if signal.side == ReversionSide.UP
                else order["price"] - 0.01,
                4
            )
            improved_price = max(0.01, min(0.99, improved_price))
            try:
                # Check for partial fills BEFORE canceling — if any shares were
                # already lifted from the maker order, record them and skip improvement.
                partial = await self._gtx_partial_fill(order)
                if partial is not None:
                    return partial

                await self.bot._run_in_thread(clob.cancel_order, order["order_id"])

                # Only post the improved order if the cancel actually succeeded.
                # Verify by confirming the old order_id is no longer live; if the
                # cancel failed silently we'd end up with two resting GTX orders.
                try:
                    old_status = await self._poll_order_status(order["order_id"])
                    if old_status not in (None,) and "live" in (old_status or ""):
                        logger.warning("MR: cancel did not clear old order %s… — skipping improve", order["order_id"][:8])
                        return None
                except Exception:
                    pass  # Poll failure — proceed cautiously

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
                    self._pending_maker_order["_last_fill_check"] = 0.0  # force check on next cycle
                    self._maker_order_time = time.time()
                    logger.info("MR: improved maker order to %.3f", improved_price)
                    return None
            except Exception as e:
                logger.warning("MR: maker improve failed: %s", e)

        # Escalate to taker if the drop is still strong enough
        signal = order["signal"]
        market = self.state.get_market_data()
        risk = self.state.get_risk_metrics()

        min_drop = self.config.min_drop
        if risk.regime.value == "high":
            min_drop = self.config.high_vol_drop
        elif risk.regime.value == "low":
            min_drop = self.config.low_vol_drop

        history: Deque = self._side_history_up if signal.side == ReversionSide.UP else self._side_history_down
        window_prices = [h['price'] for h in history if now - h['timestamp'] <= self.config.min_drop_window]
        if window_prices:
            peak  = max(window_prices)
            trough = min(window_prices)
            current_drop = (peak - trough) / peak if peak > 0 else 0
            ob_support = 1 - self.state.get_imbalance()

            if current_drop >= min_drop * self.config.taker_escalation_threshold and ob_support >= self.config.ob_imbalance_threshold:
                # Check for partial fills BEFORE canceling — any shares already lifted
                # must be accounted for; otherwise the subsequent FOK orphans them.
                partial = await self._gtx_partial_fill(order)
                if partial is not None:
                    return partial

                try:
                    await self.bot._run_in_thread(clob.cancel_order, order["order_id"])
                except Exception:
                    pass
                self._pending_maker_order = None
                logger.info(
                    "MR: escalating to taker (drop %.1f%% ≥ %.1f%%)",
                    current_drop * 100, min_drop * self.config.taker_escalation_threshold * 100
                )
                return await self._execute_taker(
                    signal, order["token_id"], order["side"], order["size"],
                    order["sl_pct"], order["window_remaining"]
                )

        # Drop insufficient for escalation — one final fill-check before giving up.
        # We do NOT blindly cancel: if the order silently filled and we cancel, we
        # lose track of the position entirely.
        final_status = await self._poll_order_status(order["order_id"])
        if final_status is not None and "match" in final_status:
            try:
                data = await self.bot.get_order(order["order_id"])
                filled_size = float((data or {}).get("size_matched") or (data or {}).get("size") or order["size"])
            except Exception:
                filled_size = float(order["size"])
            logger.info("MR: maker order (late-fill) %.2f shares @ %.4f", filled_size, order["price"])
            return self._record_fill(
                order["signal"], order["price"], filled_size,
                order["sl_pct"], order["window_remaining"],
                order["token_id"],
            )

        # Truly not filled — safe to cancel
        if final_status not in (None,) and "live" not in final_status:
            pass  # already cancelled or unknown — nothing to cancel
        else:
            try:
                await self.bot._run_in_thread(clob.cancel_order, order["order_id"])
            except Exception:
                pass
        self._pending_maker_order = None
        logger.info("MR: maker order expired, no fill, no escalation")
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
        """Place GTC limit orders for TP1 and TP2 immediately after entry fills.

        Only TP orders are posted as resting GTC sell orders — they sit above
        the current market and fill naturally when price rises to the target.

        SL is intentionally NOT placed as a GTC order: Polymarket GTC orders
        are marketable, so a sell limit placed below the current bid crosses
        immediately at market price instead of resting.  The software
        check_exit() loop monitors the SL level and fires a FAK sell when
        current_price <= pos.sl_price.

        Sizes:
          TP1 → 30 % of position
          TP2 → 30 % of position
          (remaining 40 % hold is managed by ratcheted TP2 after TP1 fires)
        """
        await asyncio.sleep(3.0)  # Wait for tokens to settle after entry fill

        if not self._position or self._position is not pos:
            return  # position was cleared while waiting

        market = self.state.get_market_data()
        token_id = market.token_id_up if pos.side == ReversionSide.UP else market.token_id_down

        if not token_id:
            logger.warning("MR: cannot place exit orders — no token_id (market rotated?)")
            return

        tp1_size = round(pos.size * 0.30, 2)
        tp2_size = round(pos.size * 0.30, 2)

        tp1_price = max(0.01, min(0.99, round(exits.tp1_price, 4)))
        tp2_price = max(0.01, min(0.99, round(exits.tp2_price, 4)))

        logger.info(
            "MR: placing exit GTC orders | TP1=%.4f x%.2f  TP2=%.4f x%.2f  (SL=%.4f via software)",
            tp1_price, tp1_size, tp2_price, tp2_size, exits.initial_sl,
        )

        tp1_res, tp2_res = await asyncio.gather(
            self.bot.place_order(token_id=token_id, price=tp1_price, size=tp1_size, side="SELL", order_type="GTC"),
            self.bot.place_order(token_id=token_id, price=tp2_price, size=tp2_size, side="SELL", order_type="GTC"),
            return_exceptions=True,
        )

        if not isinstance(tp1_res, Exception) and tp1_res.success:
            pos.tp1_order_id = tp1_res.order_id
            logger.info("MR: TP1 GTC resting @ %.4f x%.2f [%s]", tp1_price, tp1_size, tp1_res.order_id[:8])
        else:
            logger.warning("MR: TP1 GTC failed: %s", tp1_res)

        if not isinstance(tp2_res, Exception) and tp2_res.success:
            pos.tp2_order_id = tp2_res.order_id
            logger.info("MR: TP2 GTC resting @ %.4f x%.2f [%s]", tp2_price, tp2_size, tp2_res.order_id[:8])
        else:
            logger.warning("MR: TP2 GTC failed: %s", tp2_res)
    
    async def _cancel_remaining_orders(self, pos: MeanReversionPosition, reason: str):
        """Cancel remaining GTC TP orders after an exit fires.

        SL is software-monitored (no GTC order), so there is never a GTC SL
        order to cancel here.  close_position() independently cancels all
        outstanding GTC orders before submitting the FAK sell.
        """
        clob = self.bot.clob_client
        orders_to_cancel: list[Optional[str]] = []

        if reason == "SL":
            orders_to_cancel = [pos.tp1_order_id, pos.tp2_order_id]
        elif reason == "TP1":
            orders_to_cancel = [pos.tp2_order_id]
        elif reason == "TP2":
            orders_to_cancel = [pos.tp1_order_id]

        for order_id in orders_to_cancel:
            if order_id:
                try:
                    await self.bot._run_in_thread(clob.cancel_order, order_id)
                    logger.info(f"MR: cancelled {order_id} after {reason}")
                except Exception as e:
                    logger.debug(f"MR: failed to cancel order {order_id}: {e}")

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
            "==> MR GTC EXIT: %s | %.2fsh @ %.4f | PnL: %s | remaining: %.2fsh",
            reason.upper(), size, fill_price, pnl_str, max(0.0, remaining),
        )
        return {"success": True, "pnl": pnl}

    async def _check_limit_orders(self, pos: MeanReversionPosition) -> Optional[Dict[str, Any]]:
        """Check if any TP limit orders have been filled.

        Throttled to every 5 seconds (P2-1) to avoid flooding the REST API.

        SL is handled entirely by the software check_exit() loop — no GTC SL
        order is placed, so there is nothing to poll here for the stop.
        """
        if not pos.tp1_order_id and not pos.tp2_order_id:
            return None

        # P2-1: throttle HTTP polling to every 5 seconds
        now = time.time()
        if now - pos._last_order_poll < 5.0:
            return None
        pos._last_order_poll = now

        try:
            clob = self.bot.clob_client

            # Check TP1 order — ratchet: sell 30%, new TP2 at config level
            if pos.tp1_order_id:
                result = await self.bot.get_order(pos.tp1_order_id) or {}
                if result.get("status") in ("matched", "MATCHED"):
                    filled_size = float(result.get("size_matched") or result.get("size", 0))
                    if filled_size > 0:
                        pos.tp1_filled = True
                        oid = pos.tp1_order_id
                        pos.tp1_order_id = None  # P0-4: clear immediately
                        logger.info(f"MR: TP1 filled {filled_size} shares")

                        # Cancel old resting TP2 before placing a tighter ratcheted one
                        if pos.tp2_order_id:
                            try:
                                await self.bot._run_in_thread(clob.cancel_order, pos.tp2_order_id)
                                pos.tp2_order_id = None
                            except Exception:
                                pass

                        # P2-6: ratcheted TP2 for the 40% hold portion
                        market = self.state.get_market_data()
                        token_id = market.token_id_up if pos.side == ReversionSide.UP else market.token_id_down
                        new_tp2_price = round(pos.entry_price * (1 + self.config.tp2_pct) * 1.005, 4)
                        remaining_size = round(pos.size * 0.40, 2)  # 40% hold portion

                        new_tp2 = await self.bot.place_order(
                            token_id=token_id, price=new_tp2_price,
                            size=remaining_size, side="SELL", order_type="GTC",
                        )
                        if not isinstance(new_tp2, Exception) and new_tp2.success:
                            pos.tp2_order_id = new_tp2.order_id

                        logger.info(
                            "MR: TP1 done — new TP2 @ %.3f  (SL trailing via software)",
                            new_tp2_price,
                        )
                        tp1_fill_price = round(pos.entry_price * (1 + self.config.tp1_pct), 4)
                        return {
                            "action": "sell", "size": filled_size, "reason": "TP1", "order_id": oid,
                            "already_filled": True, "fill_price": tp1_fill_price,
                        }

            # Check TP2 order — let hold portion ride, SL to TP1 price
            if pos.tp2_order_id:
                result = await self.bot.get_order(pos.tp2_order_id) or {}
                if result.get("status") in ("matched", "MATCHED"):
                    filled_size = float(result.get("size_matched") or result.get("size", 0))
                    if filled_size > 0:
                        pos.tp2_filled = True
                        oid = pos.tp2_order_id
                        pos.tp2_order_id = None  # P0-4: clear immediately
                        logger.info(
                            "MR: TP2 filled %.2f shares — hold portion riding, SL trailing via software",
                            filled_size,
                        )
                        tp2_fill_price = round(pos.entry_price * (1 + self.config.tp2_pct), 4)
                        return {
                            "action": "sell", "size": filled_size, "reason": "TP2", "order_id": oid,
                            "already_filled": True, "fill_price": tp2_fill_price,
                        }

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
        
        # Determine SL pct based on *current* time remaining — tightens as expiry nears
        window_remaining = market.time_to_expiry
        if window_remaining > 180:
            sl_pct = self.config.early_sl_pct   # -20%: early, more room to breathe
        else:
            sl_pct = self.config.mid_sl_pct     # -15%: mid/late, tighter
        
        # ── Trailing SL ──────────────────────────────────────────────────────────
        # Track peak price since entry
        if not hasattr(pos, '_peak_price'):
            pos._peak_price = pos.entry_price  # type: ignore[attr-defined]
        pos._peak_price = max(pos._peak_price, current_price)  # type: ignore[attr-defined]

        # Floor SL — window-based (tightens toward entry as expiry approaches)
        floor_sl = pos.entry_price * (1 - sl_pct)

        # After TP1 fires: activate trailing SL at (peak × (1 − trail_pct)),
        # minimum = entry price (break-even).  SL only ever moves UP.
        if pos.tp1_filled:
            trail_sl = max(pos.entry_price, pos._peak_price * (1 - self.config.trailing_sl_pct))  # type: ignore[attr-defined]
            new_sl = max(pos.sl_price, floor_sl, trail_sl)
        else:
            new_sl = max(pos.sl_price, floor_sl)

        if new_sl > pos.sl_price + 0.0001:
            logger.info(
                "MR: SL ratcheted %.4f → %.4f%s (peak=%.4f)",
                pos.sl_price, new_sl,
                " [TRAIL]" if pos.tp1_filled else "",
                pos._peak_price,  # type: ignore[attr-defined]
            )
        pos.sl_price = new_sl

        # ── Price-level exits ─────────────────────────────────────────────────
        tp1_threshold = pos.entry_price * (1 + self.config.tp1_pct)
        tp2_threshold = pos.entry_price * (1 + self.config.tp2_pct)

        _now = time.time()

        # SL — skip if a resting GTC SL order is handling this level
        if not pos.sl_order_id and current_price <= pos.sl_price and position_value >= 0.10:
            return {"action": "sell", "size": pos.size, "reason": "SL"}

        # TP1 — skip if a resting GTC TP1 order is handling this level
        if not pos.tp1_order_id and not pos.tp1_filled and position_value >= 0.10:
            if current_price >= tp1_threshold:
                logger.info("MR TP1 TRIGGERED: entry=%.4f cur=%.4f", pos.entry_price, current_price)
                pos.tp1_filled = True
                return {"action": "sell", "size": pos.size * 0.30, "reason": "TP1"}

        # TP2 — skip if a resting GTC TP2 order is handling this level
        if not pos.tp2_order_id and pos.tp1_filled and not pos.tp2_filled and position_value >= 0.10:
            if current_price >= tp2_threshold:
                logger.info("MR TP2 TRIGGERED: entry=%.4f cur=%.4f", pos.entry_price, current_price)
                pos.tp2_filled = True
                return {"action": "sell", "size": pos.size * 0.30, "reason": "TP2"}

        # TP3 — absolute price level ≥ 0.90 — sell 100% immediately, no prerequisite
        # Skip for snipe positions: the whole point is to hold to $1.00 on-chain
        # resolution, and snipe entries are already above 0.90, so TP3 would fire
        # immediately and cause a rapid buy/sell loop.
        if not pos.is_snipe and not pos.tp3_filled and position_value >= 0.10:
            if current_price >= self.config.tp3_price:
                logger.info(
                    "MR TP3 TRIGGERED (≥%.2f): cur=%.4f — selling 100%%",
                    self.config.tp3_price, current_price,
                )
                pos.tp3_filled = True
                return {"action": "sell", "size": pos.size, "reason": "TP3"}

        logger.debug("MR TP check: entry=%.3f cur=%.3f tp1=%.3f tp2=%.3f", pos.entry_price, current_price, tp1_threshold, tp2_threshold)
        
        # Snipe positions hold to on-chain resolution — never mercy-close or
        # expiry-close them.  The whole point is to capture the $1.00 payout.
        if not pos.is_snipe:
            # Late window: mercy stop (always closes 100%)
            if window_remaining < 45:
                mercy_price = pos.entry_price * (1 - self.config.mercy_sl_pct)
                if current_price <= mercy_price:
                    return {"action": "sell", "size": pos.size, "reason": "mercy"}

            # Expiry close - sell at market (always closes 100%)
            if window_remaining <= 5:
                logger.info("MR: expiry close at market")
                return {"action": "sell", "size": pos.size, "reason": "expiry"}
        
        return None
    
    def _force_clear_position(self, cause: str) -> None:
        """
        Wipe the local position after repeated close failures.
        Records estimated PnL at the current market mid so session/window
        totals reflect the real loss — no trades are sent to Polymarket.
        """
        if not self._position:
            return
        pos = self._position
        try:
            market = self.state.get_market_data()
            est_price = (
                market.up_price if pos.side == ReversionSide.UP else market.down_price
            )
            est_pnl = (est_price - pos.entry_price) * pos.size
            self._realized_pnl += est_pnl
            self._window_pnl   += est_pnl
            pnl_str = f"+${est_pnl:.2f}" if est_pnl >= 0 else f"-${abs(est_pnl):.2f}"
            logger.error(
                "MR: force-clear (%s) — est exit @ %.4f  est PnL %s",
                cause, est_price, pnl_str,
            )
        except Exception:
            logger.error("MR: force-clear (%s) — could not estimate PnL", cause)
        self._position = None
        self._failed_close_attempts = 0

    async def close_position(self, reason: str = "manual", sell_size: Optional[float] = None) -> Dict[str, Any]:
        """
        Close current position (full or partial).

        Retry strategy (mirrors strategies/base.py execute_sell):
          1. Poll CLOB balance up to 3× before the first order attempt to
             confirm tokens have settled from the entry FOK fill.
          2. Retry the FOK up to MAX_CLOSE_ATTEMPTS times, dropping the
             sell price by PRICE_STEP each attempt to chase liquidity.
          3. On "balance not settled" errors: refresh balance and wait
             progressively longer before the next attempt.
          4. Hard abort (force-clear) after FORCE_CLEAR_CALLS consecutive
             fully-exhausted call sequences.
        """
        MAX_CLOSE_ATTEMPTS = 8   # retries per close_position() call
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
            logger.debug("MR: waiting %.1fs for token settlement", wait_time)
            await asyncio.sleep(wait_time)

        market = self.state.get_market_data()

        # Determine size to sell (default to full position)
        if sell_size is None:
            sell_size = self._position.size

        # Round to 2 decimals (FOK requirement)
        sell_size = int(sell_size * 100) / 100.0
        if sell_size < 0.01:
            return {"success": False, "reason": "size too small"}

        # Use the token_id captured at entry — after a market rotation the live
        # market token changes to the new window, which we don't own.
        token_id = self._position.entry_token_id
        if not token_id:
            # Fallback for positions recorded before this field was added
            token_id = market.token_id_up if self._position.side == ReversionSide.UP else market.token_id_down

        if self._position.side == ReversionSide.UP:
            if market.up_bids:
                initial_price = market.up_bids[0][0]
            else:
                initial_price = market.up_price
        else:
            if market.down_bids:
                initial_price = market.down_bids[0][0]
            else:
                initial_price = market.down_price

        # Pre-flight: empty token_id means we have no idea what to sell
        if not token_id:
            msg = f"no token_id for {self._position.side.value} (market rotated?)"
            logger.error("==> MR CLOSE FAILED: %s | attempt %d | reason: %s",
                         reason.upper(), self._failed_close_attempts + 1, msg)
            self._failed_close_attempts += 1
            if self._failed_close_attempts >= FORCE_CLEAR_CALLS:
                self._force_clear_position(msg)
            return {"success": False, "reason": msg}

        if initial_price <= 0:
            msg = f"price={initial_price:.4f} (no bids + stale mid)"
            logger.error("==> MR CLOSE FAILED: %s | attempt %d | reason: %s",
                         reason.upper(), self._failed_close_attempts + 1, msg)
            self._failed_close_attempts += 1
            if self._failed_close_attempts >= FORCE_CLEAR_CALLS:
                self._force_clear_position(msg)
            return {"success": False, "reason": msg}

        # ── Phase 1: poll CLOB balance until tokens are visible ──────────────
        # Tokens from the entry FOK take a few seconds to appear in the CLOB
        # cache.  Sending a sell before they settle causes "not enough balance".
        for poll in range(BALANCE_POLLS):
            try:
                await self.bot.update_balance_allowance("CONDITIONAL", token_id)
                bal = await self.bot.get_balance_allowance("CONDITIONAL", token_id)
                if bal and float(bal.get("balance", 0)) > 0:
                    break
            except Exception:
                pass
            if poll < BALANCE_POLLS - 1:
                logger.debug("MR close: CLOB balance empty, waiting 3s (poll %d/%d)",
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

            # Drop sell price each retry to chase liquidity lower in the book
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
                logger.error("==> MR CLOSE FAILED: %s | attempt %d | reason: exception — %s",
                             reason.upper(), attempt + 1, e)
                await asyncio.sleep(1.0)
                continue

            if result.success:
                # Query the confirmed fill amount via get_order().
                #
                # NEVER use `size_matched from the POST response as the source of
                # truth for FAK fills: Polymarket often returns size_matched=0 (or
                # omits it entirely) in the immediate POST reply even when shares
                # were filled.  Using `or remaining` as a fallback causes a phantom
                # close — the bot records a full close, clears _position, but no
                # trade actually happened on the exchange.
                filled = 0.0
                if result.order_id:
                    try:
                        order_data = await self.bot.get_order(result.order_id)
                        raw = (order_data or {}).get("size_matched") or (order_data or {}).get("sizeMatched") or 0
                        filled = float(raw)
                    except Exception as e:
                        logger.warning("MR close: get_order failed after FAK (attempt %d): %s", attempt + 1, e)

                filled = min(filled, remaining)

                if filled < 0.01:
                    # FAK returned success but nothing was actually filled at this
                    # price level — treat as no-fill and retry lower.
                    logger.debug("MR close: FAK 0 fill confirmed @ %.4f (attempt %d), retrying lower",
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
                    "MR close: FAK filled %.2f/%.2f sh @ %.4f (attempt %d, remaining %.2f)",
                    filled, sell_size, sell_price, attempt + 1, remaining,
                )

                if remaining < 0.01:
                    self._failed_close_attempts = 0
                    if self._position.size < 0.01:
                        self._position = None
                    pnl_str = f"+${total_pnl:.2f}" if total_pnl >= 0 else f"-${abs(total_pnl):.2f}"
                    logger.info("==> MR CLOSED: %s | %.2fsh | PnL: %s",
                                reason.upper(), sell_size, pnl_str)
                    return {"success": True, "pnl": total_pnl}

                # Partial fill — continue loop for remaining shares
                continue

            err_msg = result.message or "unknown"

            if "balance" in err_msg.lower() or "allowance" in err_msg.lower():
                # Tokens still settling — refresh and wait progressively longer
                wait = 5.0 + attempt * 5.0
                logger.warning("MR close: balance not settled (attempt %d), refreshing + %.0fs wait",
                                attempt + 1, wait)
                try:
                    await self.bot.update_balance_allowance("CONDITIONAL", token_id)
                except Exception:
                    pass
                await asyncio.sleep(wait)
                continue

            # No fill at this price — lower price on next attempt, no extra wait
            logger.debug("MR close: FAK no fill @ %.4f (attempt %d), retrying lower",
                         sell_price, attempt + 1)

        # ── All attempts exhausted for this call ──────────────────────────────
        if remaining < sell_size:
            # At least partially closed — log as ERROR so it appears in the
            # TUI's sticky CLOSE FAILURES panel (warning is filtered out).
            if self._position and self._position.size < 0.01:
                self._position = None
            pnl_str = f"+${total_pnl:.2f}" if total_pnl >= 0 else f"-${abs(total_pnl):.2f}"
            logger.error("==> MR CLOSE PARTIAL: %s | sold %.2f/%.2f sh | PnL: %s | remaining %.2f",
                         reason.upper(), sell_size - remaining, sell_size, pnl_str, remaining)
            return {"success": False, "partial": True, "pnl": total_pnl, "remaining": remaining}

        self._failed_close_attempts += 1
        logger.error("==> MR CLOSE FAILED: %s | attempt %d | reason: all %d retries exhausted (0 fills)",
                     reason.upper(), self._failed_close_attempts, MAX_CLOSE_ATTEMPTS)
        if self._failed_close_attempts >= FORCE_CLEAR_CALLS:
            self._force_clear_position(f"{FORCE_CLEAR_CALLS} fully-exhausted close calls")
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

    async def snipe_entry(
        self,
        side: ReversionSide,
        token_id: str,
        price: float,
        size: float,
        window_remaining: int,
    ) -> Dict[str, Any]:
        """
        Execute an immediate FOK taker entry for the expiry-snipe strategy.

        Bypasses the GTX maker step — with ≤30 s remaining there is no time
        to wait for a limit-order fill.

        SL is scaled by time remaining:
        - 22–30 s left: 10% SL (more time = more room for a reversal before exit)
        - 10–22 s left:  7% SL
        - < 10 s left:   5% SL (barely any time — if it's moving wrong get out fast)

        Called exclusively by Coordinator._check_expiry_snipe().
        """
        if self._entry_in_progress:
            return {"success": False, "reason": "entry in progress"}
        self._entry_in_progress = True
        try:
            signal = ReversionSignal(
                side=side,
                confidence=0.90,
                entry_price=price,
                reason="expiry_snipe",
                drop_magnitude=0.0,
                z_score=0.0,
                ob_confidence=0.5,
            )
            self._last_entry_time = time.time()
            if window_remaining >= 22:
                sl_pct = 0.10
            elif window_remaining >= 10:
                sl_pct = 0.07
            else:
                sl_pct = 0.05
            return await self._execute_taker(
                signal, token_id, "BUY", size, sl_pct, window_remaining
            )
        finally:
            self._entry_in_progress = False

    async def cancel_pending_maker(self) -> None:
        """Cancel any resting GTX order and clear pending state.

        Called by the coordinator on market rotation.  Without this, a GTX
        posted for the previous window's token_id stays open in the CLOB
        forever — and once the window state resets (mr_had_position = False),
        the bot posts a second GTX for the new window on top of the first.
        """
        order = self._pending_maker_order
        if not order:
            return

        order_id = order.get("order_id", "")
        try:
            clob = self.bot.clob_client
            await self.bot._run_in_thread(clob.cancel_order, order_id)
            logger.info("MR: cancelled stale maker order %s… on market reset", order_id[:8])
        except Exception as e:
            logger.warning("MR: could not cancel stale maker order %s…: %s", order_id[:8], e)
        finally:
            self._pending_maker_order = None
