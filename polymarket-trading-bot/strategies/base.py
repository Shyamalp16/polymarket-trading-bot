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
from typing import Optional, Dict, List

from lib.console import LogBuffer, log
from lib.market_manager import MarketManager, MarketInfo
from lib.price_tracker import PriceTracker
from lib.position_manager import PositionManager, Position
from src.bot import TradingBot
from src.websocket_client import OrderbookSnapshot


@dataclass
class StrategyConfig:
    """Base strategy configuration."""

    coin: str = "ETH"
    size: float = 5.0  # USDC size per trade
    max_positions: int = 1
    take_profit: float = 0.10
    stop_loss: float = 0.05
    early_stop_loss: float = 0.30   # Wider SL during the first minute of the market
    early_sl_window: int = 60       # Seconds from market start for the wider SL

    # Market settings
    market_duration: int = 15  # Market duration in minutes (5 or 15)
    market_check_interval: float = 30.0
    auto_switch_market: bool = True

    # Price tracking
    price_lookback_seconds: int = 10
    price_history_size: int = 100

    # Display settings
    update_interval: float = 0.1
    order_refresh_interval: float = 30.0  # Seconds between order refreshes


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

        # State
        self.running = False
        self._status_mode = False

        # Logging
        self._log_buffer = LogBuffer(max_size=5)

        # Open orders cache (refreshed in background)
        self._cached_orders: List[dict] = []
        self._last_order_refresh: float = 0
        self._order_refresh_task: Optional[asyncio.Task] = None
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
            # Force-close orphaned positions from the expired market.
            # The old market's tokens will auto-resolve on-chain;
            # we just need to clear internal tracking so the bot
            # can trade fresh in the new market window.
            open_positions = self.positions.get_all_positions()
            for pos in open_positions:
                self.log(
                    f"Auto-closing position {pos.side.upper()} "
                    f"(market expired)",
                    "warning",
                )
                self.positions.close_position(pos.id, realized_pnl=0.0)
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

                # Check position exits
                await self._check_exits(prices)

                # Refresh orders in background (fire-and-forget)
                self._maybe_refresh_orders()

                # Update display
                self.render_status(prices)

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
        # Adjust SL width based on how far into the market we are
        self._update_dynamic_stop_loss()

        exits = self.positions.check_all_exits(prices)

        for position, exit_type, pnl in exits:
            if exit_type == "take_profit":
                self.log(
                    f"TAKE PROFIT: {position.side.upper()} PnL: +${pnl:.2f}",
                    "success"
                )
            elif exit_type == "stop_loss":
                self.log(
                    f"STOP LOSS: {position.side.upper()} PnL: ${pnl:.2f}",
                    "warning"
                )

            # Execute sell
            await self.execute_sell(position, prices.get(position.side, 0))

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

        result = await self.bot.place_order(
            token_id=token_id,
            price=buy_price,
            size=size,
            side="BUY",
            order_type="FOK"
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
            self.positions.open_position(
                side=side,
                token_id=token_id,
                entry_price=current_price,
                size=tracked_size,
                order_id=result.order_id,
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

    async def execute_sell(self, position: Position, current_price: float) -> bool:
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
        pnl = position.get_pnl(current_price)

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
            sell_price = max(0.01, sell_price - (attempt * 0.02))

            result = await self.bot.place_order(
                token_id=position.token_id,
                price=sell_price,
                size=sell_size,
                side="SELL",
                order_type="FOK"
            )

            if result.success:
                self.log(f"Sell order: full exit PnL: ${pnl:+.2f}", "success")
                self.positions.close_position(position.id, realized_pnl=pnl)
                # Refresh both caches so next buy sees updated USDC balance
                await self.bot.refresh_balances(token_id=position.token_id)
                market = self.current_market
                current_slug = market.slug if market else ""
                if current_slug:
                    self._completed_market_slugs.add(current_slug)
                return True

            msg = (result.message or "").lower()

            if "not enough balance" in msg:
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
