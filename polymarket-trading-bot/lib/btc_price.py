"""
Simple BTC Price Tracker - Uses public APIs instead of Chainlink

Provides real-time BTC price from multiple sources:
1. Binance API (primary)
2. Coinbase API (fallback)

Usage:
    from lib.btc_price import BTCPriceTracker
    
    tracker = BTCPriceTracker()
    await tracker.start()
    
    price = tracker.get_price()
    change_2s = tracker.get_change(2)
"""

import asyncio
import logging
import time
from typing import Optional, Deque
from collections import deque

logger = logging.getLogger(__name__)


class BTCPriceTracker:
    """
    Tracks BTC price from public exchanges.
    
    No API keys needed - uses public REST endpoints.
    """
    
    def __init__(
        self,
        poll_interval: float = 1.0,
    ):
        self.poll_interval = poll_interval
        
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        # Price history
        self._history: Deque = deque(maxlen=300)  # 5 min at 1s
        self._current_price: float = 0.0
        self._last_update: float = 0.0
        
        # EWMA for volatility
        self._ewma: float = 0.0
        self._ewma_alpha: float = 0.1
    
    async def start(self):
        """Start tracking price."""
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("BTC price tracker started (Binance/Coinbase)")
    
    async def stop(self):
        """Stop tracking."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("BTC price tracker stopped")
    
    async def _poll_loop(self):
        """Poll for price updates."""
        while self._running:
            try:
                await self._fetch_price()
            except Exception as e:
                logger.debug(f"Price fetch error: {e}")
            
            await asyncio.sleep(self.poll_interval)
    
    async def _fetch_price(self):
        """Fetch price from Binance or Coinbase."""
        import aiohttp
        
        # Try Binance first
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        price = float(data['price'])
                        logger.debug(f"BTC price: ${price}")
                        self._update_price(price)
                        return
        except Exception as e:
            logger.debug(f"Binance error: {e}")
        
        # Fallback to Coinbase
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.coinbase.com/v2/prices/BTC-USD/spot",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        price = float(data['data']['amount'])
                        logger.debug(f"BTC price (Coinbase): ${price}")
                        self._update_price(price)
                        return
        except Exception as e:
            logger.debug(f"Coinbase error: {e}")
        
        logger.debug("All price sources failed")
    
    def _update_price(self, price: float):
        """Update price and calculate changes."""
        now = time.time()
        
        # Calculate 1s return for EWMA
        if self._current_price > 0:
            ret_1s = abs((price - self._current_price) / self._current_price)
            if self._ewma == 0:
                self._ewma = ret_1s
            else:
                self._ewma = self._ewma_alpha * ret_1s + (1 - self._ewma_alpha) * self._ewma
        
        self._current_price = price
        self._last_update = now
        self._history.append({
            'price': price,
            'timestamp': now,
        })
    
    def get_price(self) -> float:
        """Get current BTC price."""
        return self._current_price
    
    def get_change(self, window_seconds: int) -> float:
        """Get price change over window as percentage."""
        if not self._history or len(self._history) < 2:
            return 0.0
        
        now = time.time()
        for snapshot in reversed(self._history):
            if now - snapshot['timestamp'] >= window_seconds:
                if snapshot['price'] > 0:
                    return (self._current_price - snapshot['price']) / snapshot['price']
                break
        
        if self._history:
            earliest = self._history[0]
            if earliest['price'] > 0:
                return (self._current_price - earliest['price']) / earliest['price']
        
        return 0.0
    
    def get_ewma(self) -> float:
        """Get EWMA of 1-second returns (volatility)."""
        return self._ewma
    
    def get_last_update(self) -> float:
        """Get timestamp of last update."""
        return self._last_update
    
    def is_stale(self, max_age: float = 5.0) -> bool:
        """Check if price data is stale."""
        return time.time() - self._last_update > max_age
    
    def set_poly_mid_price(self, mid_price: float):
        """Set Polymarket mid price for divergence calculation (unused in this tracker)."""
        pass
    
    def get_divergence(self) -> float:
        """Get divergence (not applicable for exchange price)."""
        return 0.0
