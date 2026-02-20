"""
Spread Bot (Bot C) - Market-neutral liquidity provider

Strategy (PBot1-style):
  Post GTX bids on both UP and DOWN legs simultaneously at market open,
  slightly below the current best ask.  If both legs fill, total cost is
  ≈ $0.94–0.98 and the guaranteed payout at expiry is $1.00 — locking in
  a 2–6% edge with zero directional view.

Four fill cases handled:
  1. Neither fills         → cancel both at the late-window threshold
  2. Both fill             → hold to expiry (guaranteed profit)
  3. Only one fills        → single-leg position; apply directional SL
  4. Single-leg + reversal → close immediately via FAK

When to enter:
  - ≥ 120 s remaining (enough time for both legs to fill)
  - UP price and DOWN price both in [0.44, 0.56] (market genuinely uncertain)
  - Neither leg already filled/pending

Single-leg SL:
  If only one side fills and the other is cancelled, the bot holds a
  directional binary at ~48 ¢.  A 15% SL exits at ~41 ¢ to cap the
  downside while preserving the chance the position resolves at $1.00.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, TYPE_CHECKING

from lib.shared_state import SharedState

if TYPE_CHECKING:
    from src.websocket_client import UserWebSocket

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class SpreadConfig:
    """Configuration for SpreadBot."""

    # Sizing
    bankroll: float = 100.0
    size_per_leg: float = 5.0          # shares per side

    # Entry conditions
    min_time_remaining: int = 120      # seconds — need time for both legs to fill
    target_low: float = 0.44          # only spread when mid price in this band
    target_high: float = 0.56         # (markets near 50¢ are genuinely uncertain)
    bid_offset: float = 0.02          # how far below best ask to post our bid
    min_locked_profit: float = 0.10   # minimum $ locked per leg-size (e.g. 0.10 = 2¢/share on 5sh)
    max_spreads_per_session: int = 0   # 0 = unlimited; set >0 to cap total spreads

    # Timing
    fill_poll_interval: float = 5.0   # seconds between order status checks
    single_leg_cancel_window: int = 90 # seconds after first fill to wait for second
    cancel_at_remaining: int = 60     # cancel unfilled orders if < this many secs left

    # Single-leg risk management
    single_leg_sl_pct: float = 0.15   # SL 15% below entry for directional exposure

    # Overpriced sell-spread
    # When up_ask + down_ask > sell_spread_threshold the market is overpriced:
    # selling both tokens at those prices guarantees (sum − 1.00) × size profit
    # regardless of outcome.  1.04 provides a 2% buffer above the 1.02 break-even
    # to absorb fees and slippage.
    sell_spread_threshold: float = 1.04

    # Execution
    max_close_attempts: int = 8
    price_step: float = 0.02          # lower price by this much per FAK retry


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SpreadLegs:
    """Tracks the two GTX orders posted at window open."""
    up_order_id: str
    down_order_id: str
    up_token_id: str
    down_token_id: str
    up_bid: float
    down_bid: float
    size: float
    posted_at: float = field(default_factory=time.time)
    _last_poll: float = 0.0


@dataclass
class SpreadPosition:
    """Active position resulting from spread fills."""

    # Which legs actually filled
    up_filled: bool = False
    down_filled: bool = False
    up_size: float = 0.0
    down_size: float = 0.0
    up_entry: float = 0.0
    down_entry: float = 0.0
    up_token_id: str = ""
    down_token_id: str = ""

    # If only one leg filled, we manage it as a directional position
    status: str = "spread"   # "spread" | "long_up" | "long_down"
    sl_price: float = 0.0    # only set for single-leg directional
    entry_time: float = field(default_factory=time.time)

    @property
    def is_both_filled(self) -> bool:
        return self.up_filled and self.down_filled

    @property
    def cost(self) -> float:
        cost = 0.0
        if self.up_filled:
            cost += self.up_entry * self.up_size
        if self.down_filled:
            cost += self.down_entry * self.down_size
        return cost

    @property
    def guaranteed_profit(self) -> float:
        """Locked-in edge when both legs are filled."""
        if not self.is_both_filled:
            return 0.0
        payout = max(self.up_size, self.down_size)  # one resolves at $1
        return payout - self.cost


# ── SpreadBot ─────────────────────────────────────────────────────────────────

class SpreadBot:
    """
    Market-neutral spread bot (Bot C).

    Lifecycle per window:
      check_spread()  → called from coordinator; decides whether to post
      check_fills()   → polls pending GTX orders for fill confirmation
      check_exit()    → monitors directional single-leg for SL / expiry
      close_leg()     → FAK-based close of a single directional leg
    """

    def __init__(
        self,
        trading_bot,
        shared_state: SharedState,
        config: Optional[SpreadConfig] = None,
        user_ws: Optional["UserWebSocket"] = None,
    ):
        self.bot   = trading_bot
        self.state = shared_state
        self.config = config or SpreadConfig()
        self._user_ws = user_ws

        # State
        self._legs: Optional[SpreadLegs] = None           # pending GTX orders
        self._position: Optional[SpreadPosition] = None   # filled position
        self._entry_in_progress: bool = False

        # Metrics
        self._spreads_completed: int = 0   # both legs filled → expiry
        self._single_legs_today: int = 0   # single leg directional plays
        self._wins_today: int = 0
        self._realized_pnl: float = 0.0
        self._window_pnl: float = 0.0

        # Window cooldown — only one spread attempt per market window
        self._spread_attempted_this_window: bool = False

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def has_position(self) -> bool:
        return self._position is not None or self._legs is not None

    @property
    def position(self) -> Optional[SpreadPosition]:
        return self._position

    @property
    def legs(self) -> Optional[SpreadLegs]:
        return self._legs

    # ── Entry: post both GTX bids ─────────────────────────────────────────────

    async def check_spread(self) -> Optional[Dict[str, Any]]:
        """
        Evaluate entry conditions and post dual GTX bids if met.

        Returns a result dict on entry, None if conditions not met.
        Called from the coordinator's check_and_coordinate().
        """
        if self._spread_attempted_this_window:
            return None
        if self.has_position:
            return None
        if self._entry_in_progress:
            return None

        # Session cap — stop entering new spreads once the limit is reached.
        # Set max_spreads_per_session = 0 in SpreadConfig to disable the cap.
        if (self.config.max_spreads_per_session > 0 and
                self._spreads_completed >= self.config.max_spreads_per_session):
            logger.debug(
                "Spread: session cap reached (%d/%d) — no new entries",
                self._spreads_completed, self.config.max_spreads_per_session,
            )
            return None

        market = self.state.get_market_data()
        if not market:
            return None

        tte = market.time_to_expiry
        if tte < self.config.min_time_remaining:
            logger.debug("Spread: skipping — only %ds remaining (need %ds)", tte, self.config.min_time_remaining)
            return None

        # Compute the actual bid prices that will be posted (ask − offset).
        # Validate THESE against the target band — not the mid prices.
        # Mid prices can be inside the band while the computed bids land outside it
        # (e.g. UP ask=0.480, bid=0.440 sits at the floor even though mid=0.470).
        cfg = self.config
        up_ask   = market.up_asks[0][0]   if market.up_asks   else market.up_price
        down_ask = market.down_asks[0][0] if market.down_asks else market.down_price

        up_bid   = round(max(0.01, up_ask   - cfg.bid_offset), 4)
        down_bid = round(max(0.01, down_ask - cfg.bid_offset), 4)

        if not (cfg.target_low <= up_bid <= cfg.target_high):
            logger.debug(
                "Spread: UP bid %.3f outside target band [%.2f, %.2f] (ask=%.3f)",
                up_bid, cfg.target_low, cfg.target_high, up_ask,
            )
            return None
        if not (cfg.target_low <= down_bid <= cfg.target_high):
            logger.debug(
                "Spread: DOWN bid %.3f outside target band [%.2f, %.2f] (ask=%.3f)",
                down_bid, cfg.target_low, cfg.target_high, down_ask,
            )
            return None

        # Minimum locked profit — combined bids must be far enough below $1.00
        # to make the trade worth the execution risk and spread cost.
        locked_profit = (1.0 - (up_bid + down_bid)) * cfg.size_per_leg
        if locked_profit < cfg.min_locked_profit:
            logger.debug(
                "Spread: locked profit $%.2f < min $%.2f (UP=%.3f DN=%.3f) — skip",
                locked_profit, cfg.min_locked_profit, up_bid, down_bid,
            )
            return None

        return await self._post_spread(market, up_bid=up_bid, down_bid=down_bid)

    async def _post_one_leg(
        self,
        token_id: str,
        bid_price: float,
        size: float,
        leg_name: str,
    ):
        """Post a single spread leg with one retry if the order crosses the book.

        A 'crosses book' rejection means the ask moved down to meet our bid
        between the time we read the orderbook and the time the order landed.
        We back off by one tick and retry exactly once.
        """
        res = await self.bot.place_order(
            token_id=token_id,
            price=bid_price,
            size=size,
            side="BUY",
            order_type="GTC",
            post_only=True,
        )
        if not res.success and "crosses book" in (res.message or ""):
            adjusted = round(max(0.01, bid_price - 0.01), 2)
            logger.info(
                "SPR: %s crosses book at %.3f — retrying at %.3f",
                leg_name, bid_price, adjusted,
            )
            res = await self.bot.place_order(
                token_id=token_id,
                price=adjusted,
                size=size,
                side="BUY",
                order_type="GTC",
                post_only=True,
            )
        return res

    async def _post_sell_leg(
        self,
        token_id: str,
        ask_price: float,
        size: float,
        leg_name: str,
    ):
        """Post a single SELL spread leg with one retry if the order crosses the book."""
        res = await self.bot.place_order(
            token_id=token_id,
            price=ask_price,
            size=size,
            side="SELL",
            order_type="GTC",
            post_only=True,
        )
        if not res.success and "crosses book" in (res.message or ""):
            adjusted = round(min(0.99, ask_price + 0.01), 2)
            logger.info(
                "SPR SELL: %s crosses book at %.3f — retrying at %.3f",
                leg_name, ask_price, adjusted,
            )
            res = await self.bot.place_order(
                token_id=token_id,
                price=adjusted,
                size=size,
                side="SELL",
                order_type="GTC",
                post_only=True,
            )
        return res

    async def _check_sell_spread(self, market) -> Optional[Dict[str, Any]]:
        """Post SELL GTC orders on both legs when both asks sum above the threshold.

        When ``up_ask + down_ask > sell_spread_threshold`` the market is
        overpriced: selling both at those prices guarantees a locked profit of
        ``(sum − 1.00) × size`` irrespective of the binary outcome.

        Reuses the same ``_spread_attempted_this_window`` gate and the
        ``has_position`` / ``_entry_in_progress`` guards checked by
        ``check_spread`` before calling here.
        """
        cfg = self.config
        up_ask   = market.up_asks[0][0]   if market.up_asks   else market.up_price
        down_ask = market.down_asks[0][0] if market.down_asks else market.down_price

        asks_sum = up_ask + down_ask
        if asks_sum <= cfg.sell_spread_threshold:
            return None

        locked_profit = (asks_sum - 1.0) * cfg.size_per_leg
        if locked_profit < cfg.min_locked_profit:
            logger.debug(
                "Spread SELL: locked profit $%.2f < min $%.2f (UP=%.3f DN=%.3f) — skip",
                locked_profit, cfg.min_locked_profit, up_ask, down_ask,
            )
            return None

        logger.info(
            "==> SPREAD SELL: UP ask=%.3f + DN ask=%.3f = %.3f > threshold %.3f | locked=$%.2f",
            up_ask, down_ask, asks_sum, cfg.sell_spread_threshold, locked_profit,
        )

        self._entry_in_progress = True
        self._spread_attempted_this_window = True
        try:
            size = cfg.size_per_leg
            up_res, down_res = await asyncio.gather(
                self._post_sell_leg(market.token_id_up,   up_ask,   size, "UP"),
                self._post_sell_leg(market.token_id_down, down_ask, size, "DOWN"),
            )

            if not up_res.success and not down_res.success:
                logger.warning("Spread SELL: both legs rejected — %s / %s",
                               up_res.message, down_res.message)
                self._spread_attempted_this_window = False
                return {"success": False, "reason": "both SELL legs rejected"}

            if not up_res.success:
                logger.warning("Spread SELL: UP leg rejected (%s), cancelling DOWN", up_res.message)
                if down_res.order_id:
                    try:
                        await self.bot._run_in_thread(self.bot.clob_client.cancel_order, down_res.order_id)
                    except Exception:
                        pass
                self._spread_attempted_this_window = False
                return {"success": False, "reason": f"SELL UP rejected: {up_res.message}"}

            if not down_res.success:
                logger.warning("Spread SELL: DOWN leg rejected (%s), cancelling UP", down_res.message)
                if up_res.order_id:
                    try:
                        await self.bot._run_in_thread(self.bot.clob_client.cancel_order, up_res.order_id)
                    except Exception:
                        pass
                self._spread_attempted_this_window = False
                return {"success": False, "reason": f"SELL DOWN rejected: {down_res.message}"}

            # Both legs posted — track as a filled SELL position (locked profit on exit)
            logger.info(
                "==> SPREAD SELL POSTED: UP order_id=%s | DOWN order_id=%s | locked=$%.2f",
                up_res.order_id, down_res.order_id, locked_profit,
            )
            self._position = {
                "type": "sell_spread",
                "up_ask": up_ask,
                "down_ask": down_ask,
                "size": size,
                "locked_profit": locked_profit,
                "up_order_id": up_res.order_id,
                "down_order_id": down_res.order_id,
                "entry_time": time.time(),
            }
            return {
                "success": True,
                "type": "sell_spread",
                "up_ask": up_ask,
                "down_ask": down_ask,
                "locked_profit": locked_profit,
            }
        finally:
            self._entry_in_progress = False

    async def _post_spread(
        self,
        market,
        up_bid: float,
        down_bid: float,
    ) -> Optional[Dict[str, Any]]:
        """Post GTC post-only bids on both legs simultaneously.

        up_bid / down_bid are pre-computed and pre-validated by check_spread().
        """
        self._entry_in_progress = True
        self._spread_attempted_this_window = True
        try:
            cfg  = self.config
            size = cfg.size_per_leg

            # Sanity: combined cost must be < $1 for guaranteed profit
            combined = (up_bid + down_bid) * size
            payout   = size * 1.0
            if combined >= payout:
                logger.debug("Spread: combined cost %.3f ≥ payout %.3f — skip", combined, payout)
                return None

            # Post both legs concurrently. Each leg retries once on "crosses book".
            up_res, down_res = await asyncio.gather(
                self._post_one_leg(market.token_id_up,   up_bid,   size, "UP"),
                self._post_one_leg(market.token_id_down, down_bid, size, "DOWN"),
            )

            # Handle partial posting failures
            if not up_res.success and not down_res.success:
                logger.warning("Spread: both legs rejected — %s / %s",
                               up_res.message, down_res.message)
                self._spread_attempted_this_window = False  # allow retry later this window
                return {"success": False, "reason": "both legs rejected"}

            if not up_res.success:
                logger.warning("Spread: UP leg rejected (%s), cancelling DOWN", up_res.message)
                if down_res.order_id:
                    try:
                        await self.bot._run_in_thread(self.bot.clob_client.cancel_order, down_res.order_id)
                    except Exception:
                        pass
                self._spread_attempted_this_window = False
                return {"success": False, "reason": f"UP rejected: {up_res.message}"}

            if not down_res.success:
                logger.warning("Spread: DOWN leg rejected (%s), cancelling UP", down_res.message)
                if up_res.order_id:
                    try:
                        await self.bot._run_in_thread(self.bot.clob_client.cancel_order, up_res.order_id)
                    except Exception:
                        pass
                self._spread_attempted_this_window = False
                return {"success": False, "reason": f"DOWN rejected: {down_res.message}"}

            # Both legs posted — log the confirmed entry now
            logger.info(
                "==> SPREAD POSTED: UP @ %.3f  DOWN @ %.3f | %d sh/leg | "
                "cost=%.2f  locked=+%.2f",
                up_bid, down_bid, size, combined, payout - combined,
            )

            self._legs = SpreadLegs(
                up_order_id=up_res.order_id,
                down_order_id=down_res.order_id,
                up_token_id=market.token_id_up,
                down_token_id=market.token_id_down,
                up_bid=up_bid,
                down_bid=down_bid,
                size=size,
            )

            return {"success": True, "pending_spread": True, "up_bid": up_bid, "down_bid": down_bid}

        finally:
            self._entry_in_progress = False

    # ── Fill monitoring ───────────────────────────────────────────────────────

    async def check_fills(self) -> Optional[Dict[str, Any]]:
        """
        Detect fill confirmation for pending GTC legs.
        Handles all four fill cases.

        Fast path: if the user-channel WebSocket has delivered a MATCHED event
        for either order_id, we detect the fill immediately without waiting for
        the HTTP poll interval.  The HTTP poll runs as a fallback on its own
        5-second cadence in case the WS is disconnected or the event was missed.

        Called from coordinator every loop tick.
        """
        if not self._legs or self._position is not None:
            return None

        legs = self._legs
        now  = time.time()
        market = self.state.get_market_data()
        tte = market.time_to_expiry if market else 999

        # Cancel remaining orders if the window is almost over
        if tte < self.config.cancel_at_remaining:
            logger.info("Spread: late-window — cancelling unfilled orders (%ds left)", tte)
            await self._cancel_legs()
            return None

        # ── Fast path: user-channel WebSocket fill events ─────────────────────
        ws = self._user_ws
        ws_up   = ws.is_filled(legs.up_order_id)   if ws else False
        ws_down = ws.is_filled(legs.down_order_id) if ws else False

        if ws_up or ws_down:
            logger.info(
                "Spread: WS fill detected — UP=%s DOWN=%s (no HTTP poll needed)",
                ws_up, ws_down,
            )

        # ── Throttle HTTP polling to fill_poll_interval ───────────────────────
        # Still run poll if neither WS fill fired (may be all-WS path or mixed).
        need_poll = (now - legs._last_poll >= self.config.fill_poll_interval)

        if not ws_up and not ws_down and not need_poll:
            return None  # nothing to do yet

        # Determine status for each leg: WS result takes priority, HTTP as fallback
        if ws_up:
            up_status = "matched"
        elif need_poll:
            up_status = await self._poll_order(legs.up_order_id)
        else:
            up_status = None  # only WS-driven path reached here

        if ws_down:
            down_status = "matched"
        elif need_poll:
            down_status = await self._poll_order(legs.down_order_id)
        else:
            down_status = None

        if need_poll:
            legs._last_poll = now

        up_filled   = up_status   in ("matched", "filled")
        down_filled = down_status in ("matched", "filled")
        up_dead     = up_status   in ("cancelled", "expired")
        down_dead   = down_status in ("cancelled", "expired")

        logger.debug("Spread fill check: UP=%s DOWN=%s", up_status, down_status)

        # ── Case 2: both filled → guaranteed profit ───────────────────────────
        if up_filled and down_filled:
            return await self._record_both_fills()

        # ── Case 3a: only UP filled ───────────────────────────────────────────
        if up_filled and not down_filled:
            elapsed_since_up_fill = now - legs.posted_at  # approximate
            if elapsed_since_up_fill > self.config.single_leg_cancel_window or down_dead:
                logger.info("Spread: only UP filled — cancelling DOWN and exiting flat")
                await self._cancel_single_order(legs.down_order_id)
                return await self._exit_flat(
                    token_id=legs.up_token_id,
                    size=legs.size,
                    entry_price=legs.up_bid,
                    side="UP",
                )
            return None  # still waiting for DOWN to fill

        # ── Case 3b: only DOWN filled ─────────────────────────────────────────
        if down_filled and not up_filled:
            elapsed_since_down_fill = now - legs.posted_at
            if elapsed_since_down_fill > self.config.single_leg_cancel_window or up_dead:
                logger.info("Spread: only DOWN filled — cancelling UP and exiting flat")
                await self._cancel_single_order(legs.up_order_id)
                return await self._exit_flat(
                    token_id=legs.down_token_id,
                    size=legs.size,
                    entry_price=legs.down_bid,
                    side="DOWN",
                )
            return None  # still waiting for UP to fill

        # Both still live (or unknown)
        return None

    async def _poll_order(self, order_id: str) -> Optional[str]:
        """HTTP fallback: poll a single order's status. Returns lowercase status string."""
        try:
            data = await self.bot.get_order(order_id)
            if data:
                return (data.get("status") or "").lower()
        except Exception as e:
            logger.debug("Spread: get_order %s… failed: %s", order_id[:8], e)

        # Second fallback: absence from open-orders list implies filled/cancelled
        try:
            open_orders = await self.bot.get_open_orders()
            open_ids = {
                str(o.get("id") or o.get("order_id") or o.get("orderID", ""))
                for o in (open_orders or [])
            }
            return "live" if order_id in open_ids else "matched"
        except Exception:
            return None

    async def _record_both_fills(self) -> Dict[str, Any]:
        """Record a complete spread — both legs filled."""
        legs = self._legs
        size = legs.size
        cost = (legs.up_bid + legs.down_bid) * size
        locked = size * 1.0 - cost

        self._position = SpreadPosition(
            up_filled=True,
            down_filled=True,
            up_size=size,
            down_size=size,
            up_entry=legs.up_bid,
            down_entry=legs.down_bid,
            up_token_id=legs.up_token_id,
            down_token_id=legs.down_token_id,
            status="spread",
        )
        self._legs = None
        self._spreads_completed += 1

        logger.info(
            "==> SPREAD FILLED: UP @ %.3f  DOWN @ %.3f | cost $%.2f | locked +$%.2f",
            legs.up_bid, legs.down_bid, cost, locked,
        )
        return {"success": True, "both_filled": True, "locked_profit": locked}

    async def _record_single_fill(self, up: bool) -> Dict[str, Any]:
        """Record a single-leg fill — now a directional position."""
        legs = self._legs
        size = legs.size

        if up:
            entry   = legs.up_bid
            token   = legs.up_token_id
            sl      = entry * (1 - self.config.single_leg_sl_pct)
            status  = "long_up"
            side_s  = "UP"
        else:
            entry   = legs.down_bid
            token   = legs.down_token_id
            sl      = entry * (1 - self.config.single_leg_sl_pct)
            status  = "long_down"
            side_s  = "DOWN"

        self._position = SpreadPosition(
            up_filled=up,
            down_filled=not up,
            up_size=size if up else 0.0,
            down_size=size if not up else 0.0,
            up_entry=entry if up else 0.0,
            down_entry=entry if not up else 0.0,
            up_token_id=legs.up_token_id if up else "",
            down_token_id=legs.down_token_id if not up else "",
            status=status,
            sl_price=sl,
            entry_time=time.time(),
        )
        self._legs = None
        self._single_legs_today += 1

        logger.info(
            "==> SPREAD SINGLE LEG: %s @ %.3f | %d sh | SL %.3f",
            side_s, entry, size, sl,
        )
        return {"success": True, "both_filled": False, "side": side_s, "entry": entry}

    async def _exit_flat(
        self,
        token_id: str,
        size: float,
        entry_price: float,
        side: str,
    ) -> Dict[str, Any]:
        """FAK-close the one filled leg when the second leg never fills.

        Exits the position flat rather than taking on directional risk.
        Uses the same retry ladder as _close_leg.
        """
        market = self.state.get_market_data()
        self._legs = None  # release legs tracking before executing close

        if not market:
            self._single_legs_today += 1
            logger.warning("Spread exit flat: no market data — position may be orphaned")
            return {"success": False, "reason": "no market data for flat exit"}

        initial_price = (
            market.up_bids[0][0]   if side == "UP"   and market.up_bids   else
            market.down_bids[0][0] if side == "DOWN" and market.down_bids else
            (market.up_price if side == "UP" else market.down_price)
        )

        cfg       = self.config
        remaining = size
        total_pnl = 0.0

        for attempt in range(cfg.max_close_attempts):
            if remaining < 0.01:
                break

            sell_price = max(0.01, initial_price - attempt * cfg.price_step)
            try:
                result = await self.bot.place_order(
                    token_id=token_id,
                    price=sell_price,
                    size=remaining,
                    side="SELL",
                    order_type="FAK",
                )
            except Exception as e:
                logger.error("Spread flat exit %s: attempt %d failed — %s", side, attempt + 1, e)
                await asyncio.sleep(1.0)
                continue

            if result.success:
                filled = 0.0
                if result.order_id:
                    try:
                        order_data = await self.bot.get_order(result.order_id)
                        raw = (
                            (order_data or {}).get("size_matched")
                            or (order_data or {}).get("sizeMatched")
                            or 0
                        )
                        filled = float(raw)
                    except Exception:
                        pass
                filled = min(filled, remaining)
                if filled < 0.01:
                    logger.debug("Spread flat exit: FAK 0 fill @ %.4f (attempt %d)", sell_price, attempt + 1)
                    continue
                pnl = (sell_price - entry_price) * filled
                total_pnl += pnl
                self._realized_pnl += pnl
                self._window_pnl   += pnl
                remaining -= filled
                if remaining < 0.01:
                    break

        self._single_legs_today += 1
        pnl_str = f"+${total_pnl:.2f}" if total_pnl >= 0 else f"-${abs(total_pnl):.2f}"

        if remaining < 0.01:
            logger.info(
                "==> SPREAD EXIT FLAT: %s filled @ %.3f, sold @ %.3f | PnL: %s",
                side, entry_price, sell_price, pnl_str,
            )
            return {"success": True, "exit_flat": True, "pnl": total_pnl}
        else:
            logger.error(
                "==> SPREAD EXIT FLAT INCOMPLETE: %s | sold %.2f/%.2f sh | PnL: %s",
                side, size - remaining, size, pnl_str,
            )
            return {"success": False, "exit_flat": True, "pnl": total_pnl}

    # ── Exit monitoring (single-leg directional) ──────────────────────────────

    async def check_exit(self) -> Optional[Dict[str, Any]]:
        """
        Monitor a single-leg directional position for SL or expiry.

        Full-spread positions (status="spread") are left to resolve at
        expiry automatically — no action needed.
        """
        if not self._position:
            return None
        pos = self._position

        if pos.status == "spread":
            # Both legs filled — nothing to do until expiry resolves
            return None

        market = self.state.get_market_data()
        if not market:
            return None

        tte = market.time_to_expiry

        # Determine current price for the held side
        if pos.status == "long_up":
            current_price = market.up_price
            token_id      = pos.up_token_id
            size          = pos.up_size
            entry         = pos.up_entry
        else:
            current_price = market.down_price
            token_id      = pos.down_token_id
            size          = pos.down_size
            entry         = pos.down_entry

        # SL check
        if current_price < pos.sl_price:
            logger.info(
                "Spread SL hit: %.4f < %.4f — closing single leg",
                current_price, pos.sl_price,
            )
            return await self._close_leg(token_id, size, entry, reason="SL")

        # Expiry: if < 10 s remaining and position not resolved, close manually
        if tte <= 10 and tte > 0:
            logger.info("Spread: < 10s to expiry — closing single leg (mercy)")
            return await self._close_leg(token_id, size, entry, reason="expiry")

        return None

    # ── Position close (FAK with partial-fill tracking) ───────────────────────

    async def _close_leg(
        self,
        token_id: str,
        size: float,
        entry_price: float,
        reason: str = "manual",
    ) -> Dict[str, Any]:
        """Close a single directional leg via FAK with partial-fill tracking."""
        market = self.state.get_market_data()
        if not market:
            return {"success": False, "reason": "no market data"}

        if self._position and self._position.status == "long_up":
            initial_price = market.up_bids[0][0] if market.up_bids else market.up_price
        else:
            initial_price = market.down_bids[0][0] if market.down_bids else market.down_price

        remaining  = size
        total_pnl  = 0.0
        cfg        = self.config

        for attempt in range(cfg.max_close_attempts):
            if remaining < 0.01:
                break

            sell_price = max(0.01, initial_price - attempt * cfg.price_step)

            try:
                result = await self.bot.place_order(
                    token_id=token_id,
                    price=sell_price,
                    size=remaining,
                    side="SELL",
                    order_type="FAK",
                )
            except Exception as e:
                logger.error("==> SPREAD CLOSE FAILED: %s | attempt %d | %s",
                             reason.upper(), attempt + 1, e)
                await asyncio.sleep(1.0)
                continue

            if result.success:
                filled = 0.0
                if result.order_id:
                    try:
                        order_data = await self.bot.get_order(result.order_id)
                        raw = (
                            (order_data or {}).get("size_matched")
                            or (order_data or {}).get("sizeMatched")
                            or 0
                        )
                        filled = float(raw)
                    except Exception as e:
                        logger.warning("Spread close: get_order failed (attempt %d): %s", attempt + 1, e)

                filled = min(filled, remaining)
                if filled < 0.01:
                    logger.debug("Spread close: FAK 0 fill @ %.4f (attempt %d), retrying lower",
                                 sell_price, attempt + 1)
                    continue

                pnl = (sell_price - entry_price) * filled
                total_pnl += pnl
                if pnl > 0:
                    self._wins_today += 1
                self._realized_pnl += pnl
                self._window_pnl   += pnl
                remaining -= filled

                if remaining < 0.01:
                    self._position = None
                    pnl_str = f"+${total_pnl:.2f}" if total_pnl >= 0 else f"-${abs(total_pnl):.2f}"
                    logger.info("==> SPREAD CLOSED: %s | %.2f sh | PnL: %s",
                                reason.upper(), size, pnl_str)
                    return {"success": True, "pnl": total_pnl}
                continue

            err_msg = result.message or "unknown"
            if "balance" in err_msg.lower() or "allowance" in err_msg.lower():
                await asyncio.sleep(5.0)
                continue
            logger.debug("Spread close: no fill @ %.4f attempt %d", sell_price, attempt + 1)

        pnl_str = f"+${total_pnl:.2f}" if total_pnl >= 0 else f"-${abs(total_pnl):.2f}"
        if remaining < size:
            logger.error("==> SPREAD CLOSE PARTIAL: %s | sold %.2f/%.2f | PnL: %s",
                         reason.upper(), size - remaining, size, pnl_str)
        else:
            logger.error("==> SPREAD CLOSE FAILED: %s | 0 fills after %d attempts",
                         reason.upper(), cfg.max_close_attempts)
        return {"success": False, "pnl": total_pnl}

    # ── Window reset ──────────────────────────────────────────────────────────

    async def on_window_reset(self) -> None:
        """
        Called by the coordinator when a new market window is detected.
        Cancels any unfilled GTX orders from the previous window and releases
        any fully-locked spread position (both legs filled) so the bot can
        enter a new spread in the next window.  Locked tokens resolve on-chain
        automatically — no close action is needed.
        """
        if self._legs:
            logger.info("Spread: new market — cancelling stale GTX orders")
            await self._cancel_legs()

        if self._position and self._position.status == "spread":
            logger.info(
                "==> SPREAD RELEASED: UP %.3f + DOWN %.3f | locked +$%.2f | "
                "resolves on-chain at market expiry",
                self._position.up_entry,
                self._position.down_entry,
                self._position.guaranteed_profit,
            )
            self._position = None

        self._spread_attempted_this_window = False

    async def _cancel_legs(self) -> None:
        """Cancel both pending GTX orders."""
        legs = self._legs
        if not legs:
            return
        clob = self.bot.clob_client
        for oid, label in [(legs.up_order_id, "UP"), (legs.down_order_id, "DOWN")]:
            try:
                await self.bot._run_in_thread(clob.cancel_order, oid)
                logger.info("Spread: cancelled %s order %s…", label, oid[:8])
            except Exception as e:
                logger.warning("Spread: could not cancel %s order %s…: %s", label, oid[:8], e)
        self._legs = None

    async def _cancel_single_order(self, order_id: str) -> None:
        """Cancel one GTX order."""
        clob = self.bot.clob_client
        try:
            await self.bot._run_in_thread(clob.cancel_order, order_id)
        except Exception as e:
            logger.debug("Spread: cancel_single %s… failed: %s", order_id[:8], e)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            "spreads_completed": self._spreads_completed,
            "single_legs_today": self._single_legs_today,
            "wins_today": self._wins_today,
            "realized_pnl": self._realized_pnl,
        }

    def reset_daily_stats(self) -> None:
        self._spreads_completed = 0
        self._single_legs_today = 0
        self._wins_today = 0

    def reset_window_pnl(self) -> None:
        self._window_pnl = 0.0
