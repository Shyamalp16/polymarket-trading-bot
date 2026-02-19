"""
BTC Price Tracker — Real-time price feed for the dual-bot system

Price source priority:
  1. Coinbase Exchange WebSocket  wss://ws-feed.exchange.coinbase.com
       → same feed Polymarket BTC markets settle against; pushed on every tick
  2. Binance WebSocket            wss://stream.binance.com:9443/ws/btcusdt@bookTicker
       → real-time best bid/ask (mid computed here); no auth needed
  3. Coinbase REST fallback       api.exchange.coinbase.com/products/BTC-USD/ticker
  4. Binance REST fallback        api.binance.com/api/v3/ticker/bookTicker?symbol=BTCUSDT

Usage:
    tracker = BTCPriceTracker()
    await tracker.start()
    price  = tracker.get_price()          # latest mid price
    change = tracker.get_change(2)        # % change over last 2 s
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from typing import Deque, Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_CB_WS_URL  = "wss://ws-feed.exchange.coinbase.com"
_BN_WS_URL  = "wss://stream.binance.com:9443/ws/btcusdt@bookTicker"
_CB_REST    = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"
_BN_REST    = "https://api.binance.com/api/v3/ticker/bookTicker?symbol=BTCUSDT"

_WS_RECONNECT_DELAY = 3.0   # seconds between reconnect attempts
_REST_POLL_INTERVAL = 1.0   # fallback polling cadence
_STALE_THRESHOLD    = 5.0   # seconds before a price is considered stale


class BTCPriceTracker:
    """
    Tracks BTC/USD price in real time.

    Tries WebSocket sources first for sub-100 ms latency.  Falls back to
    REST polling if both WebSocket connections fail.  Thread-safe for
    asyncio single-threaded use.
    """

    def __init__(self, poll_interval: float = _REST_POLL_INTERVAL):
        self.poll_interval = poll_interval

        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Price state
        self._current_price: float = 0.0
        self._last_update:   float = 0.0

        # History for change / EWMA calculations
        self._history: Deque[dict] = deque(maxlen=600)  # 10 min at 1 Hz
        self._ewma:   float = 0.0
        self._alpha:  float = 0.1

        # Diagnostics
        self._source:   str = "none"
        self._ws_fails: int = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info("BTC price tracker started (Coinbase WS → Binance WS → REST)")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("BTC price tracker stopped")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_price(self) -> float:
        return self._current_price

    def get_source(self) -> str:
        return self._source

    def get_change(self, window_seconds: int) -> float:
        """% price change over the last ``window_seconds`` seconds."""
        if len(self._history) < 2:
            return 0.0
        now = time.time()
        for snap in reversed(self._history):
            if now - snap["ts"] >= window_seconds:
                ref = snap["price"]
                return (self._current_price - ref) / ref if ref > 0 else 0.0
        # Not enough history yet — use oldest available
        ref = self._history[0]["price"]
        return (self._current_price - ref) / ref if ref > 0 else 0.0

    def get_ewma(self) -> float:
        return self._ewma

    def get_last_update(self) -> float:
        return self._last_update

    def is_stale(self, max_age: float = _STALE_THRESHOLD) -> bool:
        return time.time() - self._last_update > max_age

    # Compat shims (used by run_dual_bot.py)
    def set_poly_mid_price(self, mid_price: float) -> None:
        pass

    def get_divergence(self) -> float:
        return 0.0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _record(self, price: float, source: str) -> None:
        """Store a new price tick and update EWMA."""
        now = time.time()
        if self._current_price > 0:
            ret = abs((price - self._current_price) / self._current_price)
            self._ewma = self._alpha * ret + (1 - self._alpha) * self._ewma

        self._current_price = price
        self._last_update   = now
        self._source        = source
        self._history.append({"price": price, "ts": now})
        logger.debug("BTC $%.2f  [%s]", price, source)

    # ── Main driver ───────────────────────────────────────────────────────────

    async def _run(self) -> None:
        """Try Coinbase WS → Binance WS → REST, cycling on failure."""
        while self._running:
            try:
                await self._coinbase_ws()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Coinbase WS failed: %s", e)

            if not self._running:
                break

            try:
                await self._binance_ws()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Binance WS failed: %s", e)

            if not self._running:
                break

            self._ws_fails += 1
            if self._ws_fails >= 3:
                logger.warning(
                    "Both WebSocket sources failed %d times — falling back to REST polling. "
                    "Prices will have ~1 s latency.",
                    self._ws_fails,
                )
            await self._rest_fallback_round()

    # ── Coinbase WebSocket ────────────────────────────────────────────────────

    async def _coinbase_ws(self) -> None:
        """
        Stream BTC-USD ticks from the Coinbase Exchange WebSocket feed.
        Sends a ticker subscription and records every price update.
        """
        import aiohttp

        sub = json.dumps({
            "type": "subscribe",
            "product_ids": ["BTC-USD"],
            "channels": ["ticker"],
        })

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                _CB_WS_URL,
                heartbeat=20,
                timeout=aiohttp.ClientWSTimeout(ws_close=10),
            ) as ws:
                await ws.send_str(sub)
                logger.info("BTC feed: Coinbase WebSocket connected")
                self._ws_fails = 0

                async for msg in ws:
                    if not self._running:
                        return
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        self._handle_coinbase_msg(msg.data)
                    elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                        logger.debug("Coinbase WS closed/error")
                        return

    def _handle_coinbase_msg(self, raw: str) -> None:
        try:
            d = json.loads(raw)
            if d.get("type") != "ticker":
                return
            # Use bid/ask mid when available for cleaner signal; fall back to last price
            bid = d.get("best_bid")
            ask = d.get("best_ask")
            if bid and ask:
                price = (float(bid) + float(ask)) / 2
            elif d.get("price"):
                price = float(d["price"])
            else:
                return
            self._record(price, "coinbase-ws")
        except Exception:
            pass

    # ── Binance WebSocket ─────────────────────────────────────────────────────

    async def _binance_ws(self) -> None:
        """
        Stream BTC best bid/ask from Binance bookTicker.
        Mid = (best_bid + best_ask) / 2.
        """
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                _BN_WS_URL,
                heartbeat=20,
                timeout=aiohttp.ClientWSTimeout(ws_close=10),
            ) as ws:
                logger.info("BTC feed: Binance WebSocket connected")
                self._ws_fails = 0

                async for msg in ws:
                    if not self._running:
                        return
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        self._handle_binance_msg(msg.data)
                    elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                        logger.debug("Binance WS closed/error")
                        return

    def _handle_binance_msg(self, raw: str) -> None:
        try:
            d = json.loads(raw)
            bid = float(d["b"])
            ask = float(d["a"])
            self._record((bid + ask) / 2, "binance-ws")
        except Exception:
            pass

    # ── REST fallback ─────────────────────────────────────────────────────────

    async def _rest_fallback_round(self) -> None:
        """
        Poll REST APIs for one ``_WS_RECONNECT_DELAY`` period, then let the
        caller retry WebSocket sources.
        """
        deadline = time.time() + _WS_RECONNECT_DELAY * 4  # ~12 s before retry
        while self._running and time.time() < deadline:
            fetched = await self._fetch_rest()
            if not fetched:
                logger.debug("All REST sources failed — price may be stale")
            await asyncio.sleep(self.poll_interval)

    async def _fetch_rest(self) -> bool:
        """Try Coinbase REST then Binance REST. Returns True on success."""
        import aiohttp
        timeout = aiohttp.ClientTimeout(total=4)

        # Coinbase REST (preferred — same exchange as settlement)
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(_CB_REST, timeout=timeout) as r:
                    if r.status == 200:
                        d = await r.json()
                        bid = float(d.get("bid", 0))
                        ask = float(d.get("ask", 0))
                        price = float(d["price"]) if not (bid and ask) else (bid + ask) / 2
                        self._record(price, "coinbase-rest")
                        return True
        except Exception as e:
            logger.debug("Coinbase REST error: %s", e)

        # Binance REST fallback
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(_BN_REST, timeout=timeout) as r:
                    if r.status == 200:
                        d = await r.json()
                        bid = float(d["bidPrice"])
                        ask = float(d["askPrice"])
                        self._record((bid + ask) / 2, "binance-rest")
                        return True
        except Exception as e:
            logger.debug("Binance REST error: %s", e)

        return False
