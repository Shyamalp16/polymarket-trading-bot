"""
Spot Price Tracker - Chainlink BTC/USD price feed on Polygon

Provides real-time BTC spot price tracking via Chainlink oracle.
Uses Web3.py to read from the AggregatorV3Interface.

Chainlink BTC/USD on Polygon:
    Address: 0xc907E6A6C8Ec11008C6147579E83Db4B90C4dF6f

Usage:
    from lib.spot_tracker import SpotTracker
    
    tracker = SpotTracker(rpc_url="https://polygon-rpc.com")
    await tracker.start()
    
    price = tracker.get_price()
    change_2s = tracker.get_change(2)  # % change in 2s
    
    await tracker.stop()
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional, Deque
from collections import deque

logger = logging.getLogger(__name__)

# Chainlink BTC/USD on Polygon mainnet
CHAINLINK_BTC_USD_POLYGON = "0xc907E6A6C8Ec11008C6147579E83Db4B90C4dF6f"

# AggregatorV3Interface ABI (only what we need)
AGGREGATOR_ABI = [
    {
        "inputs": [],
        "name": "latestRoundData",
        "outputs": [
            {"name": "roundId", "type": "uint80"},
            {"name": "answer", "type": "int256"},
            {"name": "startedAt", "type": "uint256"},
            {"name": "updatedAt", "type": "uint256"},
            {"name": "answeredInRound", "type": "uint80"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function",
    },
]


@dataclass
class SpotSnapshot:
    """Spot price snapshot."""
    price: float
    timestamp: float


class SpotTracker:
    """
    Tracks BTC spot price from Chainlink oracle.
    
    Polls the oracle at regular intervals and maintains
    price history for change calculations.
    """
    
    def __init__(
        self,
        rpc_url: str,
        feed_address: str = CHAINLINK_BTC_USD_POLYGON,
        poll_interval: float = 1.0,
    ):
        self.rpc_url = rpc_url
        self.feed_address = feed_address
        self.poll_interval = poll_interval
        
        self._web3 = None
        self._contract = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        # Price history
        self._history: Deque[SpotSnapshot] = deque(maxlen=300)  # 5 min at 1s
        self._current_price: float = 0.0
        self._last_update: float = 0.0
        
        # EWMA for volatility
        self._ewma: float = 0.0
        self._ewma_alpha: float = 0.1
        
        # For divergence calculation
        self._poly_mid_price: float = 0.5
    
    async def start(self):
        """Start tracking spot price."""
        from web3 import Web3
        
        self._web3 = Web3(Web3.HTTPProvider(self.rpc_url))
        
        if not self._web3.is_connected():
            raise ConnectionError(f"Failed to connect to RPC: {self.rpc_url}")
        
        self._contract = self._web3.eth.contract(
            address=self._web3.to_checksum_address(self.feed_address),
            abi=AGGREGATOR_ABI,
        )
        
        # Get initial price
        await self._fetch_price()
        
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(f"Spot tracker started: {self.feed_address}")
    
    async def stop(self):
        """Stop tracking."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Spot tracker stopped")
    
    async def _poll_loop(self):
        """Poll Chainlink for price updates."""
        while self._running:
            try:
                await self._fetch_price()
            except Exception as e:
                logger.warning(f"Failed to fetch spot price: {e}")
            
            await asyncio.sleep(self.poll_interval)
    
    async def _fetch_price(self):
        """Fetch latest price from Chainlink."""
        try:
            # Call synchronously in executor to avoid blocking
            loop = asyncio.get_event_loop()
            round_data = await loop.run_in_executor(
                None,
                self._contract.functions.latestRoundData().call
            )
            
            # Price is returned as int256 with 8 decimals
            price = round_data[1] / 1e8
            
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
            self._history.append(SpotSnapshot(price=price, timestamp=now))
            
        except Exception as e:
            logger.debug(f"Chainlink price fetch error (non-fatal): {e}")
            # Don't raise - just log and continue
    
    def get_price(self) -> float:
        """Get current BTC price."""
        return self._current_price
    
    def get_change(self, window_seconds: int) -> float:
        """
        Get price change over window.
        
        Args:
            window_seconds: Number of seconds to look back
            
        Returns:
            Percentage change (e.g., 0.002 = 0.2%)
        """
        if not self._history or len(self._history) < 2:
            return 0.0
        
        now = time.time()
        for snapshot in reversed(self._history):
            if now - snapshot.timestamp >= window_seconds:
                if snapshot.price > 0:
                    return (self._current_price - snapshot.price) / snapshot.price
                break
        
        # If not enough history, return from earliest
        earliest = self._history[0]
        if earliest.price > 0:
            return (self._current_price - earliest.price) / earliest.price
        
        return 0.0
    
    def get_ewma(self) -> float:
        """Get EWMA of 1-second returns (volatility metric)."""
        return self._ewma
    
    def set_poly_mid_price(self, mid_price: float):
        """Set Polymarket mid price for divergence calculation."""
        self._poly_mid_price = mid_price
    
    def get_divergence(self) -> float:
        """
        Get divergence between spot implied probability and Poly price.
        
        For BTC markets, we use spot as a proxy for implied probability.
        Returns: spot_implied - poly_mid
        """
        if self._poly_mid_price <= 0:
            return 0.0
        
        # Convert spot to probability-like (normalize around $50k baseline)
        # This is a simplified version - in practice you'd calibrate
        baseline = 50000.0
        spot_implied = min(0.99, max(0.01, self._current_price / (baseline * 2)))
        
        return spot_implied - self._poly_mid_price
    
    def get_last_update(self) -> float:
        """Get timestamp of last price update."""
        return self._last_update
    
    def is_stale(self, max_age: float = 5.0) -> bool:
        """Check if price data is stale."""
        return time.time() - self._last_update > max_age
