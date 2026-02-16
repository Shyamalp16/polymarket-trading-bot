"""
Strategy Base Class - Foundation for Trading Strategies

Provides:
- Base class for all trading strategies
- Common lifecycle methods (start, stop, run)
- Integration with lib components (MarketManager, PriceTracker, PositionManager)
- Logging and status display utilities

Usage:
    from strategies.base import BaseStrategy, StrategyConfig

    class MyStrategy(BaseStrategy):
        async def on_book_update(self, snapshot):
            # Handle orderbook updates
            pass

        async def on_tick(self, prices):
            # Called each strategy tick
            pass
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, List, Any

from lib.console import LogBuffer, log
from lib.market_manager import MarketManager, MarketInfo
from lib.price_tracker import PriceTracker
from lib.position_manager import PositionManager, Position
from lib.trade_telemetry import TradeTelemetry
from lib.latency_metrics import record_latency
from src.bot import TradingBot
from src.websocket_client import OrderbookSnapshot


@dataclass
class StrategyConfig:
    """Base strategy configuration."""

    coin: str = "ETH"
    size: float = 5.0  # USDC size per trade
    max_positions: int = 1
    take_profit: float = 0.10
    stop_loss: float = 0.20
    early_stop_loss: float = 0.30   # Wider SL during the first minute of the market
    early_sl_window: int = 60       # Seconds from market start for the wider SL
    use_time_phase_exits: bool = True
    min_hold_before_exit_seconds: float = 8.0
    partial_tp_enabled: bool = True
    partial_tp_time_force_enabled: bool = True
    partial_tp_late_hold_threshold: float = 0.15
    partial_tp_force_elapsed_threshold: float = 0.30
    partial_tp_force_min_profit_pct: float = 0.08
    partial_tp_force_sell_pct: float = 0.40
    partial_tp_depth_levels: int = 3
    partial_tp_max_depth_fraction: float = 0.25
    sell_retry_price_step: float = 0.01
    sell_retry_max_slippage: float = 0.03
    sell_retry_no_wait_on_fok: bool = True

    # Market settings
    market_duration: int = 15  # Market duration in minutes (5 or 15)
    market_check_interval: float = 30.0
    auto_switch_market: bool = True

    # Price tracking
    price_lookback_seconds: int = 10
    price_history_size: int = 100

    # Display settings
    update_interval: float = 0.1
    render_interval: float = 0.25  # seconds between TUI redraws
    order_refresh_interval: float = 30.0  # Seconds between order refreshes
    wallet_sync_interval: float = 3.0     # Seconds between wallet/position reconciliation

    # Telemetry / analytics
    trade_telemetry_enabled: bool = True
    trade_telemetry_dir: str = "logs/trades"


class BaseStrategy(ABC):
    """
    Base class for trading strategies.

    Provides common infrastructure:
    - MarketManager for WebSocket and market discovery
    - PriceTracker for price history
    - PositionManager for positions and TP/SL
    - Logging and status display
    """

    def __init__(self, bot: TradingBot, config: StrategyConfig):
        """
        Initialize base strategy.

        Args:
            bot: TradingBot instance for order execution
            config: Strategy configuration
        """
        self.bot = bot
        self.config = config

        # Core components
        self.market = MarketManager(
            coin=config.coin,
            market_check_interval=config.market_check_interval,
            auto_switch_market=config.auto_switch_market,
            market_duration=config.market_duration,
        )

        self.prices = PriceTracker(
            lookback_seconds=config.price_lookback_seconds,
            max_history=config.price_history_size,
        )

        self.positions = PositionManager(
            take_profit=config.take_profit,
            stop_loss=config.stop_loss,
            max_positions=config.max_positions,
        )
        self.telemetry = TradeTelemetry(
            enabled=config.trade_telemetry_enabled,
            base_dir=config.trade_telemetry_dir,
            strategy_name=self.__class__.__name__,
            coin=config.coin,
            market_duration_min=config.market_duration,
        )

        # State
        self.running = False
        self._status_mode = False

        # Logging
        self._log_buffer = LogBuffer(max_size=5)
        self._last_render_time: float = 0.0

        # Open orders cache (refreshed in background)
        self._cached_orders: List[dict] = []
        self._last_order_refresh: float = 0
        self._order_refresh_task: Optional[asyncio.Task] = None
        self._last_wallet_sync: float = 0.0
        # Recovery state for empty-book periods after market rotation.
        self._no_data_since: float = 0.0
        self._last_data_recovery_at: float = 0.0
        # One-and-done market policy: once we exit in a market window,
        # do not enter again until the market slug changes.
        self._completed_market_slugs: set[str] = set()
        self._last_blocked_buy_slug: Optional[str] = None

        # Allowance-error cooldown: avoids spamming the same error every tick.
        # When a persistent issue is detected (allowance==0, insufficient USDC),
        # we record the timestamp and suppress further checks for a cooldown
        # period so the log isn't flooded.
        self._balance_block_until: float = 0.0  # time.time() when next re-check allowed
        self._BALANCE_COOLDOWN: float = 60.0    # seconds between re-checks
        self._partial_tp_state: Dict[str, Dict[str, Any]] = {}

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self.market.is_connected

    @property
    def current_market(self) -> Optional[MarketInfo]:
        """Get current market info."""
        return self.market.current_market

    @property
    def token_ids(self) -> Dict[str, str]:
        """Get current token IDs."""
        return self.market.token_ids

    @property
    def open_orders(self) -> List[dict]:
        """Get cached open orders."""
        return self._cached_orders

    def _refresh_orders_sync(self) -> List[dict]:
        """Refresh open orders synchronously (called via to_thread)."""
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.bot.get_open_orders())
            finally:
                loop.close()
        except Exception:
            return []

    async def _do_order_refresh(self) -> None:
        """Background task to refresh orders without blocking."""
        try:
            orders = await asyncio.to_thread(self._refresh_orders_sync)
            self._cached_orders = orders
        except Exception:
            pass
        finally:
            self._order_refresh_task = None

    def _maybe_refresh_orders(self) -> None:
        """Schedule order refresh if interval has passed (fire-and-forget)."""
        now = time.time()
        if now - self._last_order_refresh > self.config.order_refresh_interval:
            # Don't start new refresh if one is already running
            if self._order_refresh_task is not None and not self._order_refresh_task.done():
                return
            self._last_order_refresh = now
            # Fire and forget - doesn't block main loop
            self._order_refresh_task = asyncio.create_task(self._do_order_refresh())

    def log(self, msg: str, level: str = "info") -> None:
        """
        Log a message.

        Args:
            msg: Message to log
            level: Log level (info, success, warning, error, trade)
        """
        if self._status_mode:
            self._log_buffer.add(msg, level)
        else:
            log(msg, level)

    def _get_trade_context(self) -> Dict[str, object]:
        """
        Strategy-specific telemetry context hook.

        Subclasses can override to attach signal state, raw features, etc.
        """
        return {}

    def _time_remaining_secs(self) -> Optional[int]:
        market = self.current_market
        if not market:
            return None
        mins, secs = market.get_countdown()
        if mins < 0:
            return None
        return mins * 60 + secs

    def _market_slug(self) -> str:
        market = self.current_market
        return market.slug if market else ""

    def _telemetry_log(self, payload: Dict[str, object]) -> None:
        try:
            self.telemetry.log_event(payload)
        except Exception:
            # Telemetry must never break live trading.
            pass

    async def start(self) -> bool:
        """
        Start the strategy.

        Returns:
            True if started successfully
        """
        self.running = True

        # Register callbacks on market manager
        @self.market.on_book_update
        async def handle_book(snapshot: OrderbookSnapshot):  # pyright: ignore[reportUnusedFunction]
            # Record price
            for side, token_id in self.token_ids.items():
                if token_id == snapshot.asset_id:
                    self.prices.record(side, snapshot.mid_price)
                    break

            # Delegate to subclass
            await self.on_book_update(snapshot)

        @self.market.on_market_change
        def handle_market_change(old_slug: str, new_slug: str):  # pyright: ignore[reportUnusedFunction]
            self.log(f"Market changed: {old_slug} -> {new_slug}", "warning")
            # Keep old-market positions tracked through resolution.
            # They will be cleared by wallet sync when conditional token
            # balances settle to zero after resolution/redemption.
            open_positions = self.positions.get_all_positions()
            if open_positions:
                self.log(
                    "Keeping open positions for settlement; not force-closing on market switch.",
                    "warning",
                )
            self.prices.clear()
            self._log_buffer.clear()
            self._last_blocked_buy_slug = None
            self._balance_block_until = 0.0  # reset cooldown for new market
            self.on_market_change(old_slug, new_slug)

        @self.market.on_connect
        def handle_connect():  # pyright: ignore[reportUnusedFunction]
            self.log("WebSocket connected", "success")
            self.on_connect()

        @self.market.on_disconnect
        def handle_disconnect():  # pyright: ignore[reportUnusedFunction]
            self.log("WebSocket disconnected", "warning")
            self.on_disconnect()

        # Start market manager
        if not await self.market.start():
            self.running = False
            return False

        # Wait for initial data
        if not await self.market.wait_for_data(timeout=5.0):
            self.log("Timeout waiting for market data", "warning")

        return True

    async def stop(self) -> None:
        """Stop the strategy."""
        self.running = False

        # Cancel order refresh task if running
        if self._order_refresh_task is not None:
            self._order_refresh_task.cancel()
            try:
                await self._order_refresh_task
            except asyncio.CancelledError:
                pass
            self._order_refresh_task = None

        await self.market.stop()

    async def run(self) -> None:
        """Main strategy loop."""
        try:
            if not await self.start():
                self.log("Failed to start strategy", "error")
                return

            self._status_mode = True

            while self.running:
                # Get current prices
                prices = self._get_current_prices()
                await self._recover_if_no_data(prices)

                # Call tick handler
                await self.on_tick(prices)

                # Reconcile tracked positions with wallet balances so
                # manual/external sells do not leave stale open positions.
                await self._sync_positions_with_wallet(prices)

                # Check position exits
                await self._check_exits(prices)

                # Refresh orders in background (fire-and-forget)
                self._maybe_refresh_orders()

                # Update display at a slower cadence to reduce TUI flicker.
                now = time.time()
                if (
                    self.config.render_interval <= 0
                    or (now - self._last_render_time) >= self.config.render_interval
                ):
                    self.render_status(prices)
                    self._last_render_time = now

                await asyncio.sleep(self.config.update_interval)

        except KeyboardInterrupt:
            self.log("Strategy stopped by user")
        finally:
            await self.stop()
            self._print_summary()

    def _get_current_prices(self) -> Dict[str, float]:
        """Get current prices from market manager."""
        prices = {}
        for side in ["up", "down"]:
            price = self.market.get_mid_price(side)
            if price > 0:
                prices[side] = price
        return prices

    async def _recover_if_no_data(self, prices: Dict[str, float]) -> None:
        """
        Recover from empty orderbooks by forcing a market/WS refresh.

        This protects against occasional stale subscriptions after
        market rotation where prices stay at zero.
        """
        now = time.time()
        if prices:
            self._no_data_since = 0.0
            return

        if self._no_data_since == 0.0:
            self._no_data_since = now
            return

        no_data_for = now - self._no_data_since
        # Avoid aggressive refresh loops.
        if no_data_for < 4.0 or (now - self._last_data_recovery_at) < 8.0:
            return

        self._last_data_recovery_at = now
        self.log("No live book data; forcing market/websocket refresh", "warning")
        try:
            await self.market.refresh_market()
        except Exception:
            pass

    def _get_market_elapsed_secs(self) -> Optional[float]:
        """Return seconds elapsed since the current market opened, or None."""
        market = self.current_market
        if not market:
            return None
        mins, secs = market.get_countdown()
        if mins < 0:
            return None
        remaining_secs = mins * 60 + secs
        market_duration_secs = self.config.market_duration * 60
        elapsed = market_duration_secs - remaining_secs
        # elapsed could be negative if clock skew; clamp to 0
        return max(0.0, elapsed)

    def _get_time_remaining_secs(self) -> Optional[int]:
        """Return seconds remaining in current market, or None."""
        market = self.current_market
        if not market:
            return None
        mins, secs = market.get_countdown()
        if mins < 0:
            return None
        return max(0, mins * 60 + secs)

    def _phase_exit_params(self, time_remaining_secs: int) -> tuple[Optional[float], float, bool, str]:
        """
        Return (tp_delta, sl_delta, hold_to_expiry, phase_label).

        Percent values are deltas on entry price (e.g. 0.30 => +30% TP).
        """
        total = max(1, self.config.market_duration * 60)

        # 5m profile
        if total <= 300:
            if time_remaining_secs > 180:
                return (0.25, 0.15, False, "early")
            if time_remaining_secs > 60:
                return (0.35, 0.12, False, "mid")
            return (None, 0.25, True, "late")

        # 15m (and default longer-window fallback) profile
        if time_remaining_secs > 600:
            return (0.30, 0.20, False, "early")
        if time_remaining_secs > 180:
            return (0.40, 0.15, False, "mid")
        return (None, 0.30, True, "late")

    def _late_phase_cutoff_secs(self) -> int:
        """
        Return late-phase cutoff where binary positions are usually held.

        For 5m markets this starts in the final 60s.
        For 15m markets this starts in the final 180s (3m).
        """
        total = max(1, self.config.market_duration * 60)
        return 60 if total <= 300 else 180

    @staticmethod
    def _partial_tp_profile(entry_price: float) -> Dict[str, object]:
        """Adaptive partial TP profile by entry price."""
        if entry_price < 0.40:
            return {
                "steps": [{"gain": 0.40, "sell_pct": 0.25}, {"gain": 0.90, "sell_pct": 0.35}],
                "hold_remainder": 0.40,
                "label": "deep_value",
            }
        if entry_price < 0.65:
            return {
                "steps": [{"gain": 0.22, "sell_pct": 0.30}, {"gain": 0.45, "sell_pct": 0.30}],
                "hold_remainder": 0.40,
                "label": "mid_range",
            }
        return {
            "steps": [{"gain": 0.12, "sell_pct": 0.35}, {"gain": 0.25, "sell_pct": 0.35}],
            "hold_remainder": 0.30,
            "label": "high_conviction",
        }

    def _ensure_partial_tp_state(self, position: Position) -> Dict[str, Any]:
        state = self._partial_tp_state.get(position.id)
        if state:
            return state
        state = {
            "step": 0,
            "initial_size": float(position.size),
            "realized_proceeds": 0.0,
            "profile": self._partial_tp_profile(position.entry_price),
        }
        self._partial_tp_state[position.id] = state
        return state

    def _can_partial_sell(self, side: str, sell_shares: float) -> Dict[str, object]:
        """Thin-book guard for partial TP exits."""
        ob = self.market.get_orderbook(side)
        if not ob or not ob.bids:
            return {"executable": False, "suggestion": 0.0, "method": "none"}
        best_bid_size = float(ob.bids[0].size)
        depth_levels = max(1, int(self.config.partial_tp_depth_levels))
        depth = sum(float(level.size) for level in ob.bids[:depth_levels])
        max_safe = depth * max(0.0, min(1.0, float(self.config.partial_tp_max_depth_fraction)))
        if sell_shares <= best_bid_size:
            return {"executable": True, "method": "single", "suggestion": sell_shares}
        if sell_shares <= max_safe:
            return {"executable": True, "method": "split_2", "suggestion": sell_shares}
        suggestion = max(0.0, min(sell_shares, max_safe))
        return {"executable": suggestion >= 0.01, "method": "reduced", "suggestion": suggestion}

    async def _execute_partial_sell(
        self,
        position: Position,
        current_price: float,
        shares_to_sell: float,
        label: str,
    ) -> bool:
        """Sell part of a position and keep the remainder open."""
        sell_size = int(max(0.0, float(shares_to_sell)) * 100) / 100.0
        if sell_size < 0.01:
            return False
        if sell_size >= position.size:
            return await self.execute_sell(position, current_price, exit_reason=label)

        best_bid = self.market.get_best_bid(position.side)
        base_price = max(current_price - 0.05, 0.01)
        sell_price = max(0.01, min(base_price, best_bid)) if best_bid > 0 else base_price
        result = await self.bot.place_order(
            token_id=position.token_id,
            price=sell_price,
            size=sell_size,
            side="SELL",
            order_type="FOK",
        )
        if not result.success:
            self.log(f"Partial TP skipped ({label}): {result.message}", "warning")
            return False

        realized_pnl = (sell_price - position.entry_price) * sell_size
        old_size = position.size
        position.size = max(0.01, old_size - sell_size)
        self.positions.total_pnl += realized_pnl
        state = self._partial_tp_state.get(position.id)
        if state is not None:
            state["realized_proceeds"] = float(state.get("realized_proceeds", 0.0)) + (sell_price * sell_size)
        self.log(
            f"PARTIAL TP ({label}): sold {sell_size:.2f} {position.side.upper()} @ {sell_price:.4f} "
            f"PnL=${realized_pnl:+.2f} remaining={position.size:.2f}",
            "success",
        )
        self._telemetry_log(
            {
                "event_type": "partial_exit",
                "size": round(position.entry_price * sell_size, 6),
                "entry_price": position.entry_price,
                "exit_price": sell_price,
                "size_shares": sell_size,
                "hold_seconds": round(position.get_hold_time(), 3),
            }
        )
        await self.bot.refresh_balances(token_id=position.token_id)
        return True

    async def _maybe_partial_tp(
        self,
        position: Position,
        current_price: float,
        time_remaining_secs: int,
    ) -> bool:
        """Check and execute laddered partial TP step(s)."""
        if not self.config.partial_tp_enabled:
            return False
        if position.entry_price <= 0 or current_price <= 0:
            return False

        state = self._ensure_partial_tp_state(position)
        profile = state.get("profile", {})
        steps = profile.get("steps", [])
        step_idx = int(state.get("step", 0))
        if step_idx >= len(steps):
            return False

        total_window = max(1, int(self.config.market_duration * 60))
        time_frac = max(0.0, min(1.0, float(time_remaining_secs) / float(total_window)))
        pnl_pct = (current_price - position.entry_price) / position.entry_price
        if time_frac < self.config.partial_tp_late_hold_threshold:
            return False

        if (
            self.config.partial_tp_time_force_enabled
            and step_idx == 0
            and time_frac < self.config.partial_tp_force_elapsed_threshold
            and pnl_pct > self.config.partial_tp_force_min_profit_pct
        ):
            sell_amount = float(state["initial_size"]) * self.config.partial_tp_force_sell_pct
            sell_amount = min(sell_amount, position.size)
            liq = self._can_partial_sell(position.side, sell_amount)
            if not liq.get("executable", False):
                return False
            sell_amount = float(liq.get("suggestion", sell_amount))
            if await self._execute_partial_sell(position, current_price, sell_amount, "TP1_time_force"):
                state["step"] = 1
                return True
            return False

        target = steps[step_idx]
        if pnl_pct < float(target.get("gain", 0.0)):
            return False
        sell_amount = float(state["initial_size"]) * float(target.get("sell_pct", 0.0))
        sell_amount = min(sell_amount, position.size)
        liq = self._can_partial_sell(position.side, sell_amount)
        if not liq.get("executable", False):
            return False
        sell_amount = float(liq.get("suggestion", sell_amount))
        method = str(liq.get("method", "single"))
        if method == "split_2" and sell_amount >= 0.02:
            half = int((sell_amount / 2.0) * 100) / 100.0
            rem = max(0.0, sell_amount - half)
            ok1 = await self._execute_partial_sell(position, current_price, half, f"TP{step_idx+1}_a")
            ok2 = await self._execute_partial_sell(position, current_price, rem, f"TP{step_idx+1}_b") if ok1 else False
            if ok1 and ok2:
                state["step"] = step_idx + 1
                return True
            return False
        if await self._execute_partial_sell(position, current_price, sell_amount, f"TP{step_idx+1}"):
            state["step"] = step_idx + 1
            return True
        return False

    def _update_dynamic_stop_loss(self) -> None:
        """
        Dynamically adjust stop-loss on open positions based on market age.

        During the first ``early_sl_window`` seconds of a market, use a wider
        stop-loss (``early_stop_loss``) to ride out the initial volatility.
        After that window, revert to the normal ``stop_loss``.
        """
        elapsed = self._get_market_elapsed_secs()
        if elapsed is None:
            return

        in_early_window = elapsed < self.config.early_sl_window

        for pos in self.positions.get_all_positions():
            if in_early_window:
                if pos.stop_loss_delta != self.config.early_stop_loss:
                    pos.stop_loss_delta = self.config.early_stop_loss
            else:
                if pos.stop_loss_delta != self.config.stop_loss:
                    pos.stop_loss_delta = self.config.stop_loss

    async def _check_exits(self, prices: Dict[str, float]) -> None:
        """Check and execute exits for all positions."""
        if not self.config.use_time_phase_exits:
            # Legacy behavior
            self._update_dynamic_stop_loss()
            exits = self.positions.check_all_exits(prices)
            for position, exit_type, _pnl in exits:
                if exit_type == "take_profit":
                    self.log(
                        f"TAKE PROFIT: {position.side.upper()} exiting",
                        "success"
                    )
                elif exit_type == "stop_loss":
                    self.log(
                        f"STOP LOSS: {position.side.upper()} exiting",
                        "warning"
                    )
                await self.execute_sell(
                    position,
                    prices.get(position.side, 0),
                    exit_reason=exit_type,
                )
            return

        time_remaining = self._get_time_remaining_secs()
        if time_remaining is None:
            return

        for position in self.positions.get_all_positions():
            current_slug = self._market_slug()
            if current_slug and getattr(position, "market_slug", "") and position.market_slug != current_slug:
                # Do not evaluate exits for an old market against the new market's prices.
                continue
            current_price = prices.get(position.side, 0.0)
            if current_price <= 0:
                continue
            if position.get_hold_time() < self.config.min_hold_before_exit_seconds:
                continue
            if await self._maybe_partial_tp(position, current_price, time_remaining):
                continue

            # When partial TP is managing this position, suppress the
            # phase-based full TP so the remainder rides to expiry.
            # Only stop-loss is kept as a safety net.
            partial_tp_managing = (
                self.config.partial_tp_enabled
                and position.id in self._partial_tp_state
            )

            tp_delta, sl_delta, hold_to_expiry, phase = self._phase_exit_params(time_remaining)
            position.stop_loss_delta = sl_delta
            if tp_delta is not None:
                position.take_profit_delta = tp_delta

            pnl_pct = 0.0
            if position.entry_price > 0:
                pnl_pct = (current_price - position.entry_price) / position.entry_price

            exit_type: Optional[str] = None
            if partial_tp_managing:
                # Partial TP ladder owns take-profit for this position.
                # Only allow stop-loss; the remainder holds to expiry.
                if pnl_pct <= -sl_delta:
                    exit_type = "stop_loss"
            elif hold_to_expiry:
                # Late phase: hold by default, only mercy-stop severe losers.
                if pnl_pct <= -sl_delta:
                    exit_type = "stop_loss"
            else:
                if tp_delta is not None and pnl_pct >= tp_delta:
                    exit_type = "take_profit"
                elif pnl_pct <= -sl_delta:
                    exit_type = "stop_loss"

            if exit_type is None:
                continue

            if exit_type == "take_profit":
                self.log(
                    f"TAKE PROFIT ({phase}): {position.side.upper()} exiting",
                    "success",
                )
            else:
                self.log(
                    f"STOP LOSS ({phase}): {position.side.upper()} exiting",
                    "warning",
                )

            await self.execute_sell(position, current_price, exit_reason=exit_type)

    async def _sync_positions_with_wallet(self, prices: Dict[str, float]) -> None:
        """
        Reconcile in-memory positions with wallet balances.

        This catches manual UI sells (or any external wallet action) and keeps
        local position state aligned with actual token balances.
        """
        if self.config.wallet_sync_interval <= 0:
            return

        now = time.time()
        if (now - self._last_wallet_sync) < self.config.wallet_sync_interval:
            return
        self._last_wallet_sync = now

        open_positions = self.positions.get_all_positions()
        if not open_positions:
            return

        for pos in list(open_positions):
            try:
                await self.bot.update_balance_allowance("CONDITIONAL", pos.token_id)
                tradable = await self._get_tradable_shares(pos.token_id)
            except Exception:
                continue

            if tradable is None:
                continue

            tracked_size = max(0.0, int(pos.size * 100) / 100.0)
            wallet_size = max(0.0, int(tradable * 100) / 100.0)
            size_delta = tracked_size - wallet_size

            # Ignore tiny rounding differences.
            if size_delta < 0.01:
                continue

            if wallet_size < 0.01:
                current_slug = self._market_slug()
                if current_slug and getattr(pos, "market_slug", "") and pos.market_slug != current_slug:
                    current_price = 0.0
                else:
                    current_price = prices.get(pos.side, 0.0)
                self._close_position_as_external_exit(
                    pos,
                    current_price=current_price,
                    tracked_size=tracked_size,
                    reason=f"wallet sync (tracked={tracked_size:.2f}, wallet={wallet_size:.2f})",
                )
                continue

            pos.size = wallet_size
            self.log(
                f"Wallet sync: adjusted {pos.side.upper()} size "
                f"{tracked_size:.2f} -> {wallet_size:.2f} (external partial sell)",
                "warning",
            )

    @staticmethod
    def _normalize_shares(value: Optional[object]) -> Optional[float]:
        """
        Convert amount value to shares.

        CLOB responses may return either share values (e.g. 5.23)
        or raw 1e6-scaled integers (e.g. 5230000). This helper
        normalizes both into shares.
        """
        if value is None:
            return None
        try:
            amount = float(value)
        except (TypeError, ValueError):
            return None
        # Raw 1e6-scaled amounts are typically much larger than
        # any share count we place in this strategy.
        if amount >= 10000:
            return amount / 1_000_000
        return amount

    def _filled_size_from_result(self, side: str, result_data: Dict[str, object], fallback: float) -> float:
        """
        Extract filled share size from order response.

        For BUY, shares are usually in takingAmount. For SELL, shares
        are usually in makingAmount. We also accept size_matched when present.
        """
        # Common order detail fields
        for key in ("size_matched", "sizeMatched"):
            parsed = self._normalize_shares(result_data.get(key))
            if parsed and parsed > 0:
                return parsed

        # Fallback to side-specific amount fields
        side_key = "takingAmount" if side.upper() == "BUY" else "makingAmount"
        parsed = self._normalize_shares(result_data.get(side_key))
        if parsed and parsed > 0:
            return parsed

        return fallback

    async def _get_tradable_shares(self, token_id: str) -> Optional[float]:
        """
        Get maximum tradable share balance for a token.

        Uses CLOB cached balance/allowance and returns the lower of the two.
        """
        bal = await self.bot.get_balance_allowance(
            asset_type="CONDITIONAL",
            token_id=token_id,
        )
        if not bal:
            return None

        balance_shares = self._normalize_shares(bal.get("balance"))
        allowance_shares = self._normalize_shares(bal.get("allowance"))

        if balance_shares is not None and allowance_shares is not None:
            return max(0.0, min(balance_shares, allowance_shares))
        if balance_shares is not None:
            return max(0.0, balance_shares)
        return None

    def _close_position_as_external_exit(
        self,
        position: Position,
        current_price: float,
        tracked_size: Optional[float] = None,
        reason: str = "external sell detected",
    ) -> None:
        """
        Close in-memory position when wallet indicates it was sold externally.

        Realized PnL is recorded as 0.0 because external execution price is
        unknown to the strategy process.
        """
        shares = tracked_size if tracked_size is not None else max(0.0, position.size)
        shares = max(0.0, int(shares * 100) / 100.0)
        entry_notional = position.entry_price * shares
        self._partial_tp_state.pop(position.id, None)
        self.positions.close_position(position.id, realized_pnl=0.0)
        self.log(
            f"Wallet sync: external sell closed {position.side.upper()} position "
            f"({reason})",
            "warning",
        )
        self._telemetry_log(
            {
                "event_type": "exit",
                "size": round(entry_notional, 6),
                "entry_price": position.entry_price,
                "exit_price": current_price if current_price > 0 else "",
                "size_shares": shares,
                "hold_seconds": round(position.get_hold_time(), 3),
            }
        )
        market = self.current_market
        current_slug = market.slug if market else ""
        if current_slug:
            self._completed_market_slugs.add(current_slug)

    async def execute_buy(self, side: str, current_price: float) -> bool:
        """
        Execute market buy order.

        Args:
            side: "up" or "down"
            current_price: Current market price

        Returns:
            True if order placed successfully
        """
        # Guard: never buy when already at max positions
        if not self.positions.can_open_position:
            return False

        token_id = self.token_ids.get(side)
        if not token_id:
            self.log(f"No token ID for {side}", "error")
            return False

        market = self.current_market
        current_slug = market.slug if market else ""
        if current_slug and current_slug in self._completed_market_slugs:
            if self._last_blocked_buy_slug != current_slug:
                self.log(
                    f"Skipping BUY in {current_slug}: already traded and exited this market",
                    "warning",
                )
                self._last_blocked_buy_slug = current_slug
            return False

        size = self.config.size / current_price
        # Polymarket enforces a minimum of 5 shares per order
        if size < 5.0:
            size = 5.0
        # Keep share size on 2 decimals so SELL can close the full position
        # under FOK maker precision constraints.
        size = round(size, 2)
        # Use aggressive price (+5%) to ensure fill, with FOK for guaranteed execution
        buy_price = min(current_price + 0.05, 0.99)

        # Cooldown gate: if a recent order failed with a balance/allowance
        # error we pause before retrying to avoid spamming the CLOB.
        now = time.time()
        if now < self._balance_block_until:
            return False  # still in cooldown from a previous order failure

        # Advisory pre-check: try to detect insufficient USDC before
        # sending the order.  The /balance-allowance endpoint doesn't
        # reliably report allowance for all wallet configurations, so
        # we only hard-block when we're confident the *balance* is too
        # low.  Allowance issues are caught by the actual order attempt.
        cost = size * buy_price
        await self.bot.update_balance_allowance("COLLATERAL")
        bal = await self.bot.get_balance_allowance("COLLATERAL")
        if bal:
            balance_raw = float(bal.get("balance", "0"))
            allowance_raw = float(bal.get("allowance", "0"))

            # Convert from raw (6 decimals) to USDC
            usdc_balance = balance_raw / 1_000_000

            # Only block on balance when we have a positive allowance
            # (confirms the endpoint is returning real data).
            if allowance_raw > 0:
                usdc_allowance = allowance_raw / 1_000_000
                usdc_available = min(usdc_balance, usdc_allowance)
                if usdc_available < cost:
                    self.log(
                        f"Insufficient USDC: ${usdc_available:.2f} available, "
                        f"${cost:.2f} needed — skipping BUY "
                        f"[next retry in {self._BALANCE_COOLDOWN:.0f}s]",
                        "warning",
                    )
                    self._balance_block_until = now + self._BALANCE_COOLDOWN
                    return False
            elif usdc_balance > 0 and usdc_balance < cost:
                # Allowance unknown but balance clearly too low
                self.log(
                    f"Insufficient USDC balance: ${usdc_balance:.2f} "
                    f"available, ${cost:.2f} needed — skipping BUY "
                    f"[next retry in {self._BALANCE_COOLDOWN:.0f}s]",
                    "warning",
                )
                self._balance_block_until = now + self._BALANCE_COOLDOWN
                return False

        self.log(f"BUY {side.upper()} @ {current_price:.4f} size={size:.2f}", "trade")

        submit_started = time.perf_counter()
        result = await self.bot.place_order(
            token_id=token_id,
            price=buy_price,
            size=size,
            side="BUY",
            order_type="FOK"
        )
        record_latency(
            "order_submit_ms",
            (time.perf_counter() - submit_started) * 1000.0,
            {"side": "BUY", "price": round(buy_price, 6)},
        )

        if result.success:
            # Keep tracked size aligned to what actually filled.
            # This avoids SELL attempts for more shares than we own.
            filled_size = self._filled_size_from_result("BUY", result.data, size)
            if result.order_id:
                # Prefer persisted order fields when available.
                order_details = await self.bot.get_order(result.order_id)
                if order_details:
                    filled_size = self._filled_size_from_result("BUY", order_details, filled_size)

            # Safety: if _filled_size_from_result returned the fallback
            # (= requested size) but no fill evidence exists, the FOK may
            # have been killed silently. Check for orderHashes or
            # size_matched to confirm actual fill.
            has_fill_evidence = bool(
                result.data.get("orderHashes")
                or result.data.get("transactionsHashes")
            )
            if not has_fill_evidence and order_details:
                sm = order_details.get("size_matched") or order_details.get("sizeMatched")
                has_fill_evidence = sm is not None and float(sm) > 0

            if not has_fill_evidence:
                self.log(
                    f"BUY accepted but no fill evidence found (FOK may have been killed). "
                    f"Response: {result.data}",
                    "warning",
                )
                return False

            tracked_size = max(0.01, int(filled_size * 100) / 100.0)

            # Wait for on-chain settlement before refreshing cache.
            # On Polygon, FOK fills need a few blocks (~2s each) to
            # settle AND the CLOB's internal balance cache needs time to
            # pick up the new tokens.  For signature_type=2 (browser
            # proxy) the manual refresh endpoint often returns {} so we
            # rely on the CLOB's automatic cache — 15s gives enough
            # headroom for ~5-7 Polygon blocks.
            self.log("Waiting for on-chain settlement (15s)...", "info")
            await asyncio.sleep(15.0)

            # Pre-warm the CLOB balance cache so subsequent sells don't hit
            # the stale "not enough balance" error.  Refresh both
            # CONDITIONAL (for sells) and COLLATERAL (for next buy check).
            await self.bot.refresh_balances(token_id=token_id)
            self._balance_block_until = 0.0  # clear cooldown on success
            self._last_blocked_buy_slug = None
            self.log(f"Order filled: {result.order_id} size={tracked_size:.2f}", "success")
            position = self.positions.open_position(
                side=side,
                token_id=token_id,
                market_slug=current_slug,
                entry_price=current_price,
                size=tracked_size,
                order_id=result.order_id,
            )
            if position:
                entry_notional = position.entry_price * position.size
                self._partial_tp_state[position.id] = {
                    "step": 0,
                    "initial_size": float(position.size),
                    "realized_proceeds": 0.0,
                    "profile": self._partial_tp_profile(position.entry_price),
                }
                self._telemetry_log(
                    {
                        "event_type": "entry",
                        "size": round(entry_notional, 6),
                        "entry_price": position.entry_price,
                        "exit_price": "",
                        "size_shares": position.size,
                        "hold_seconds": 0.0,
                    }
                )
            return True
        else:
            msg = (result.message or "").lower()
            if "not enough balance" in msg:
                self.log(
                    f"Order rejected: not enough balance/allowance "
                    f"[next retry in {self._BALANCE_COOLDOWN:.0f}s]",
                    "error",
                )
                self._balance_block_until = time.time() + self._BALANCE_COOLDOWN
            else:
                self.log(f"Order failed: {result.message}", "error")
            return False

    async def execute_sell(
        self,
        position: Position,
        current_price: float,
        exit_reason: Optional[str] = None,
    ) -> bool:
        """
        Execute sell order to close position.

        Simple and robust: sell tracked size with wait-and-retry on
        balance errors (tokens may still be settling from FOK buy).

        Args:
            position: Position to close
            current_price: Current price

        Returns:
            True if fully sold
        """
        # SELL FOK maker amount supports max 2 decimals in shares.
        # Round DOWN to avoid over-selling.
        sell_size = int(position.size * 100) / 100.0
        if sell_size < 0.01:
            sell_size = 0.01

        # Advisory pre-check: try to detect share balance before entering
        # the retry loop.  The /balance-allowance endpoint may return 0
        # for allowance in some wallet configurations even when trading
        # works, so we only use this to *adjust* sell_size downward, never
        # to hard-block the sell.
        tradable = await self._get_tradable_shares(position.token_id)
        if tradable is not None and tradable < 0.01:
            # Tokens may still be settling from FOK buy; refresh and recheck.
            await self.bot.update_balance_allowance(
                "CONDITIONAL", position.token_id
            )
            await asyncio.sleep(1.0)
            tradable = await self._get_tradable_shares(position.token_id)
            if tradable is not None and tradable < 0.01:
                # Position appears already sold outside this process.
                self._close_position_as_external_exit(
                    position,
                    current_price=current_price,
                    reason="sell requested but wallet has zero shares",
                )
                return True

        if tradable is not None and tradable >= 0.01 and tradable < sell_size:
            old_size = sell_size
            sell_size = max(0.01, int(tradable * 100) / 100.0)
            self.log(
                f"Adjusted sell size {old_size:.2f} -> {sell_size:.2f} "
                f"(available: {tradable:.2f})",
                "warning",
            )

        # Attempt up to 6 times with increasing wait (token settlement) and
        # widening price (liquidity).  For signature_type=2 accounts, the
        # manual balance refresh often returns {} so we first poll the
        # balance endpoint until we see actual data before burning an
        # order attempt.
        max_attempts = 6
        initial_sell_price = 0.0
        for attempt in range(max_attempts):
            # --- Phase 1: wait until the CLOB cache shows a balance ---
            # On the first attempt we poll aggressively (up to 8×3s = 24s)
            # because the tokens may still be settling.  On retries we
            # already waited via the retry delay so a single refresh is enough.
            max_polls = 8 if attempt == 0 else 1
            bal = None
            balance_visible = False
            for poll in range(max_polls):
                await self.bot.update_balance_allowance(
                    asset_type="CONDITIONAL",
                    token_id=position.token_id,
                )
                bal = await self.bot.get_balance_allowance(
                    "CONDITIONAL", position.token_id
                )
                if bal and bal.get("balance"):
                    raw_bal = float(bal.get("balance", "0"))
                    if raw_bal > 0:
                        balance_visible = True
                        break
                if poll < max_polls - 1:
                    self.log(
                        f"Sell attempt {attempt+1}: CLOB cache empty, "
                        f"waiting for settlement (poll {poll+1}/{max_polls})...",
                        "info",
                    )
                    await asyncio.sleep(3.0)

            # Diagnostic: log what the CLOB sees for this token
            if bal and (bal.get("balance") or bal.get("allowance")):
                self.log(
                    f"Sell attempt {attempt+1}: CLOB sees balance={bal.get('balance','?')} "
                    f"allowance={bal.get('allowance','?')}",
                    "info",
                )
            elif attempt == 0 or not balance_visible:
                self.log(
                    f"Sell attempt {attempt+1}: CLOB balance cache still empty",
                    "warning",
                )

            # --- Phase 2: submit the sell order ---
            # Price: start at best bid or mid-5%, widen 2c per retry.
            best_bid = self.market.get_best_bid(position.side)
            base_price = max(current_price - 0.05, 0.01)
            if best_bid > 0:
                sell_price = max(0.01, min(base_price, best_bid))
            else:
                sell_price = base_price
            if attempt == 0:
                initial_sell_price = sell_price
            step = max(0.0, float(self.config.sell_retry_price_step))
            sell_price = max(0.01, initial_sell_price - (attempt * step))
            realized_slippage = max(0.0, initial_sell_price - sell_price)
            if (
                self.config.sell_retry_max_slippage > 0
                and realized_slippage > self.config.sell_retry_max_slippage
            ):
                self.log(
                    f"Sell retries halted: slippage cap hit "
                    f"({realized_slippage:.3f} > {self.config.sell_retry_max_slippage:.3f})",
                    "warning",
                )
                return False
            # Realized PnL should be based on the actual submitted exit price/size,
            # not on pre-trade mark price snapshots.
            realized_pnl = (sell_price - position.entry_price) * sell_size

            submit_started = time.perf_counter()
            result = await self.bot.place_order(
                token_id=position.token_id,
                price=sell_price,
                size=sell_size,
                side="SELL",
                order_type="FOK"
            )
            record_latency(
                "order_submit_ms",
                (time.perf_counter() - submit_started) * 1000.0,
                {"side": "SELL", "attempt": attempt + 1, "price": round(sell_price, 6)},
            )

            if result.success:
                self.log(f"Sell order: full exit PnL: ${realized_pnl:+.2f}", "success")
                if attempt > 0:
                    self.log(
                        f"Sell filled after {attempt+1} attempts "
                        f"(slippage={realized_slippage:.3f})",
                        "info",
                    )
                entry_notional = position.entry_price * position.size
                self._telemetry_log(
                    {
                        "event_type": "exit",
                        "size": round(entry_notional, 6),
                        "entry_price": position.entry_price,
                        "exit_price": sell_price,
                        "size_shares": sell_size,
                        "hold_seconds": round(position.get_hold_time(), 3),
                    }
                )
                self._partial_tp_state.pop(position.id, None)
                self.positions.close_position(position.id, realized_pnl=realized_pnl)
                # Refresh both caches so next buy sees updated USDC balance
                await self.bot.refresh_balances(token_id=position.token_id)
                market = self.current_market
                current_slug = market.slug if market else ""
                if current_slug:
                    self._completed_market_slugs.add(current_slug)
                return True

            msg = (result.message or "").lower()

            if "not enough balance" in msg:
                # Re-check tradable shares after rejection. If wallet now shows
                # no shares, treat this as an external/manual close and clear
                # local state to avoid repeated failing sell attempts.
                await self.bot.update_balance_allowance(
                    "CONDITIONAL", position.token_id
                )
                post_err_tradable = await self._get_tradable_shares(position.token_id)
                if post_err_tradable is not None and post_err_tradable < 0.01:
                    self._close_position_as_external_exit(
                        position,
                        current_price=current_price,
                        reason="sell rejected and wallet shows zero shares",
                    )
                    return True
                if attempt < max_attempts - 1:
                    # Progressive waits: 5, 10, 15, 20, 25s.
                    wait = 5.0 + attempt * 5.0
                    self.log(
                        f"Sell attempt {attempt+1}/{max_attempts}: balance not yet "
                        f"settled, waiting {wait:.0f}s before retry",
                        "warning",
                    )
                    await asyncio.sleep(wait)
                    continue
            elif "fok" in msg and "not" in msg and "fill" in msg:
                if attempt < max_attempts - 1:
                    self.log(
                        f"Sell attempt {attempt+1}/{max_attempts}: FOK not filled, widening price",
                        "warning",
                    )
                    if not self.config.sell_retry_no_wait_on_fok:
                        await asyncio.sleep(0.5)
                    continue

            # Unknown error or final attempt — give up for this tick.
            self.log(f"Sell failed: {result.message}", "error")
            return False

        self.log("Sell failed after all retries", "error")
        return False

    def _print_summary(self) -> None:
        """Print session summary."""
        self._status_mode = False
        print()
        stats = self.positions.get_stats()
        self.log("Session Summary:")
        self.log(f"  Trades: {stats['trades_closed']}")
        self.log(f"  Total PnL: ${stats['total_pnl']:+.2f}")
        self.log(f"  Win rate: {stats['win_rate']:.1f}%")

    # Abstract methods to implement in subclasses

    @abstractmethod
    async def on_book_update(self, snapshot: OrderbookSnapshot) -> None:
        """
        Handle orderbook update.

        Called when new orderbook data is received.

        Args:
            snapshot: OrderbookSnapshot from WebSocket
        """
        pass

    @abstractmethod
    async def on_tick(self, prices: Dict[str, float]) -> None:
        """
        Handle strategy tick.

        Called on each iteration of the main loop.

        Args:
            prices: Current prices {side: price}
        """
        pass

    @abstractmethod
    def render_status(self, prices: Dict[str, float]) -> None:
        """
        Render status display.

        Called on each tick to update the display.

        Args:
            prices: Current prices
        """
        pass

    # Optional hooks (override as needed)

    def on_market_change(self, old_slug: str, new_slug: str) -> None:
        """Called when market changes."""
        pass

    def on_connect(self) -> None:
        """Called when WebSocket connects."""
        pass

    def on_disconnect(self) -> None:
        """Called when WebSocket disconnects."""
        pass
