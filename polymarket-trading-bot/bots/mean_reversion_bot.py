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
from typing import Optional, Dict, Any, List, Deque
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
    
    # Flash detection
    min_drop: float = 0.05          # 5% drop triggers (lowered for more signals)
    min_drop_window: int = 8        # seconds
    z_threshold: float = 1.5         # Z-score threshold (lowered further)
    z_window: int = 60              # seconds for z-score
    
    # Dynamic thresholds by regime
    high_vol_drop: float = 0.08     # Higher threshold in high vol
    low_vol_drop: float = 0.04      # Lower threshold in low vol
    
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
        
        # Maker order tracking
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
        
        # Track price for this side
        if not hasattr(self, f'_side_history_{side.value}'):
            setattr(self, f'_side_history_{side.value}', deque(maxlen=120))
        
        history: Deque = getattr(self, f'_side_history_{side.value}')
        history.append({
            'price': current_price,
            'timestamp': time.time()
        })
        
        if len(history) < 5:
            return None
        
        # Calculate drop in window
        now = time.time()
        window_prices = [
            h['price'] for h in history
            if now - h['timestamp'] <= self.config.min_drop_window
        ]
        
        if len(window_prices) < 3:
            return None
        
        peak = max(window_prices)
        trough = min(window_prices)
        
        if side == ReversionSide.UP:
            # Price dropped, looking for bounce
            drop = (peak - trough) / peak if peak > 0 else 0
            expected_direction = "down"  # Spot should have gone down
        else:
            # Price spiked, looking for dump
            drop = (peak - trough) / peak if peak > 0 else 0
            expected_direction = "up"  # Spot should have gone up
        
        # Check drop threshold
        if drop < min_drop:
            logger.debug(f"MR {side.value}: drop {drop*100:.1f}% < threshold {min_drop*100:.1f}%")
            return None
        
        # Cause filter: check if spot moved same direction
        spot_change = abs(self.state.get_spot_change(self.config.min_drop_window))
        if spot_change > self.config.cause_filter_threshold:
            # Informed flow - suppress
            logger.debug("MR: suppressed due to spot flow %.4f", spot_change)
            return None
        
        # Calculate z-score
        prices = [h['price'] for h in history if now - h['timestamp'] <= self.config.z_window]
        if len(prices) < 10:
            z_score = 0
        else:
            mean = statistics.mean(prices)
            stdev = statistics.stdev(prices) if len(prices) > 1 else 0.001
            z_score = abs((current_price - mean) / stdev) if stdev > 0 else 0
        
        if z_score < self.config.z_threshold:
            logger.debug(f"MR {side.value}: z={z_score:.1f} < {self.config.z_threshold}")
            return None
        
        # OB confirmation
        imbalance = self.state.get_imbalance()
        
        # For UP reversion (buy dip), we want ask imbalance (sells getting exhausted)
        # For DOWN reversion (sell rip), we want bid imbalance (buys getting exhausted)
        if side == ReversionSide.UP:
            # Want: more bids than asks at lower prices = buyers stepping in
            ob_support = 1 - imbalance
        else:
            ob_support = imbalance
        
        if ob_support < self.config.ob_imbalance_threshold:
            logger.debug(f"MR {side.value}: ob={ob_support:.2f} < {self.config.ob_imbalance_threshold}")
            return None
        
        # Calculate confidence
        drop_strength = min(1.0, drop / (min_drop * 2))
        z_strength = min(1.0, z_score / (self.config.z_threshold * 2))
        ob_strength = min(1.0, ob_support / 0.9)
        
        # Reduce for VPIN
        vpin_factor = max(0.3, 1 - risk.vpin)
        
        confidence = (drop_strength * 0.3 + z_strength * 0.3 + ob_strength * 0.4) * vpin_factor
        
        if confidence < self.config.min_signal_strength:
            logger.debug(f"MR {side.value}: conf={confidence:.2f} < {self.config.min_signal_strength}")
            return None
        
        # Get market data
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
    
    async def execute(self, signal: ReversionSignal) -> Dict[str, Any]:
        """Execute mean reversion entry."""
        # Check balance block cooldown
        if time.time() < self._balance_block_until:
            logger.debug("MR: blocked by balance cooldown")
            return {"success": False, "reason": "balance cooldown"}
        
        size = self.calculate_size(signal)
        
        if size < 5.0:
            return {"success": False, "reason": "size too small"}
        
        market = self.state.get_market_data()
        
        # Determine token and price
        if signal.side == ReversionSide.UP:
            token_id = market.token_id_up
            side = "BUY"
            mid = market.mid_price
            # Post at mid - offset
            maker_price = mid - self.config.maker_spread_offset
        else:
            token_id = market.token_id_down
            side = "BUY"
            mid = market.mid_price
            maker_price = mid + self.config.maker_spread_offset
        
        maker_price = max(0.01, min(0.99, maker_price))
        
        # Determine window for SL
        window_remaining = market.time_to_expiry
        
        if window_remaining > 180:  # Early window (>60%)
            sl_pct = self.config.early_sl_pct
        else:  # Mid window
            sl_pct = self.config.mid_sl_pct
        
        # Calculate exits
        if signal.side == ReversionSide.UP:
            entry = maker_price
            tp1 = entry * (1 + self.config.tp1_pct)
            tp2 = entry * (1 + self.config.tp2_pct)
            sl = entry * (1 - sl_pct)
        else:
            entry = maker_price
            tp1 = entry * (1 + self.config.tp1_pct)
            tp2 = entry * (1 + self.config.tp2_pct)
            sl = entry * (1 - sl_pct)
        
        # Use FOK with retries - wait for liquidity
        max_retries = 5
        for attempt in range(max_retries):
            # Refresh market data on each retry - prices may have changed
            market = self.state.get_market_data()
            
            try:
                # Use current market price for immediate fill
                exec_price = round(market.up_price if signal.side == ReversionSide.UP else market.down_price, 4)
                
                result = await self.bot.place_order(
                    token_id=token_id,
                    price=exec_price,
                    size=size,
                    side=side,
                    order_type="FOK",
                )
                
                if result.success:
                    actual_entry = exec_price
                    
                    # Record position
                    self._position = MeanReversionPosition(
                        side=signal.side,
                        entry_price=actual_entry,  # Use actual fill price, not intended price
                        size=size,
                        entry_time=time.time(),
                        window_start=window_remaining,
                        sl_price=actual_entry * (1 - sl_pct),  # Recalculate SL based on actual entry
                    )

                    self._last_entry_time = time.time()
                    self._entries_today += 1
                    
                    # Recalculate TP based on actual entry price
                    if signal.side == ReversionSide.UP:
                        tp1_actual = actual_entry * (1 + self.config.tp1_pct)
                        tp2_actual = actual_entry * (1 + self.config.tp2_pct)
                    else:
                        tp1_actual = actual_entry * (1 + self.config.tp1_pct)
                        tp2_actual = actual_entry * (1 + self.config.tp2_pct)
                    sl_actual = actual_entry * (1 - sl_pct)
                    
                    logger.info(
                        f"==> MR ENTERED: {signal.side.value.upper()} | ${size:.2f} @ {actual_entry:.3f} | "
                        f"TP1: {tp1_actual:.3f} TP2: {tp2_actual:.3f} SL: {sl_actual:.3f}"
                    )
                    
                    return {
                        "success": True,
                        "position": self._position,
                        "exits": ReversionExit(
                            tp1_price=tp1_actual,
                            tp2_price=tp2_actual,
                            initial_sl=sl_actual,
                            early_sl_pct=self.config.early_sl_pct,
                            mid_sl_pct=self.config.mid_sl_pct,
                            mercy_sl_pct=self.config.mercy_sl_pct,
                        ),
                    }
                
                # Check error type
                is_balance_error = result.message and ("balance" in result.message.lower() or "allowance" in result.message.lower())
                
                # Check if FOK couldn't fill (no liquidity)
                is_fok_error = result.message and ("couldn't be fully filled" in result.message.lower() or "not enough liquidity" in result.message.lower())
                
                if is_balance_error:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # 1s, 2s, 4s
                        logger.warning("MR: balance error, retry %d/%d in %ds", attempt + 1, max_retries, wait_time)
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        self._balance_block_until = time.time() + 10
                        logger.warning("MR: balance error on entry, blocking 10s")
                
                # FOK failed - wait and retry
                if is_fok_error:
                    if attempt < max_retries - 1:
                        wait_time = 1 + attempt * 0.5  # 1s, 1.5s, 2s, 2.5s
                        logger.info("MR: FOK couldn't fill, retry %d/%d in %ss", attempt + 1, max_retries, wait_time)
                        await asyncio.sleep(wait_time)
                        continue
                
                return {"success": False, "reason": result.message}
                
            except Exception as e:
                logger.error("MeanReversion execution failed: %s", e)
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                return {"success": False, "reason": str(e)}
        
        return {"success": False, "reason": "max retries exceeded"}
    
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
        
        # SL always closes 100% of position
        if position_value >= 0.10:
            if pos.side == ReversionSide.UP and current_price <= pos.sl_price:
                if time.time() - pos.entry_time > self.config.time_confirm_sl:
                    return {"action": "sell", "size": pos.size, "reason": "SL"}
            elif pos.side == ReversionSide.DOWN and current_price <= pos.sl_price:
                if time.time() - pos.entry_time > self.config.time_confirm_sl:
                    return {"action": "sell", "size": pos.size, "reason": "SL"}
        
        # Partial TP - sell 30% at TP1
        if not pos.tp1_filled and position_value >= 0.10:
            if pos.side == ReversionSide.UP and current_price >= pos.entry_price * (1 + self.config.tp1_pct):
                pos.tp1_filled = True
                return {"action": "sell", "size": pos.size * 0.3, "reason": "TP1"}
            elif pos.side == ReversionSide.DOWN and current_price >= pos.entry_price * (1 + self.config.tp1_pct):
                pos.tp1_filled = True
                return {"action": "sell", "size": pos.size * 0.3, "reason": "TP1"}
        
        # Partial TP - sell 30% at TP2
        if not pos.tp2_filled and position_value >= 0.10:
            if pos.side == ReversionSide.UP and current_price >= pos.entry_price * (1 + self.config.tp2_pct):
                pos.tp2_filled = True
                return {"action": "sell", "size": pos.size * 0.3, "reason": "TP2"}
            elif pos.side == ReversionSide.DOWN and current_price >= pos.entry_price * (1 + self.config.tp2_pct):
                pos.tp2_filled = True
                return {"action": "sell", "size": pos.size * 0.3, "reason": "TP2"}
        
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
            result = await self.bot.place_order(
                token_id=token_id,
                price=price,
                size=sell_size,
                side=side,
                order_type="FOK",
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
                # Check for balance/allowance error
                if result.message and ("balance" in result.message.lower() or "allowance" in result.message.lower()):
                    self._balance_block_until = time.time() + 3  # 3s cooldown for exits
                    self._failed_close_attempts += 1
                    logger.warning("MR close: balance error, attempt %d", self._failed_close_attempts)
                    
                    # Force clear after 5 failed attempts
                    if self._failed_close_attempts >= 5:
                        logger.error("MR: force clearing position after 5 failed close attempts")
                        self._position = None
                        self._failed_close_attempts = 0
                
                return {"success": False, "reason": result.message}
                
        except Exception as e:
            logger.error("MeanReversion close failed: %s", e)
            self._failed_close_attempts += 1
            if self._failed_close_attempts >= 5:
                logger.error("MR: force clearing position after 5 failed close attempts")
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
