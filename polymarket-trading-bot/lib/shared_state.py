"""
Shared State Service - Shared market data and risk metrics for dual-bot system

Provides:
- Market data synchronization between bots
- VPIN (toxicity) calculation
- Orderbook fragility metrics
- Spot price tracking via Chainlink

Usage:
    from lib.shared_state import SharedState, MarketData
    
    state = SharedState()
    await state.start()
    
    # Both bots access shared data
    market_data = state.get_market_data()
    vpin = state.get_vpin()
    fragility = state.get_fragility()
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Deque
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    """Volatility regime classification."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


@dataclass
class MarketData:
    """Current market state shared between bots."""
    
    # Polymarket prices
    up_price: float = 0.5
    down_price: float = 0.5
    mid_price: float = 0.5
    spread: float = 0.0
    
    # Orderbook data
    up_bids: list = field(default_factory=list)  # [(price, size), ...]
    up_asks: list = field(default_factory=list)
    down_bids: list = field(default_factory=list)
    down_asks: list = field(default_factory=list)
    
    # Orderbook metrics
    bid_depth: float = 0.0    # Total bid depth (5 levels, weighted)
    ask_depth: float = 0.0    # Total ask depth (5 levels, weighted)
    imbalance: float = 0.5    # 0 = bid heavy, 1 = ask heavy
    
    # Market info
    token_id_up: str = ""
    token_id_down: str = ""
    market_slug: str = ""
    time_to_expiry: int = 300  # seconds
    
    # Timestamps
    last_update: float = field(default_factory=time.time)


@dataclass
class RiskMetrics:
    """Risk metrics shared between bots."""
    
    # VPIN (Volume-Synchronized Probability of Informed Trading)
    vpin: float = 0.5              # 0-1, >0.7 = high toxicity
    vpin_buckets: Deque = field(default_factory=lambda: deque(maxlen=10))
    vpin_last_update: float = 0.0
    
    # Orderbook fragility
    fragility: float = 0.0         # 0-1, >0.6 = fragile
    cancel_rate: float = 0.0       # Recent cancel rate
    depth_3tick_coverage: float = 1.0  # % of orders in top 3 ticks
    
    # Spot price
    btc_spot_price: float = 0.0
    btc_spot_change_2s: float = 0.0  # % change in last 2 seconds
    btc_spot_change_60s: float = 0.0  # % change in last 60 seconds
    
    # Regime
    regime: MarketRegime = MarketRegime.NORMAL
    volatility_ewma: float = 0.0
    
    # Last update
    last_update: float = field(default_factory=time.time)


class SharedState:
    """
    Shared state service for dual-bot coordination.
    
    Provides synchronized access to market data and risk metrics
    for both bots running in the same process.
    """
    
    def __init__(self):
        self._market = MarketData()
        self._risk = RiskMetrics()
        self._lock = asyncio.Lock()

        # Price history for calculations
        self._price_history: Deque = deque(maxlen=1000)
        self._volume_buckets: Deque = deque(maxlen=10)  # For VPIN
        self._cancel_history: Deque = deque(maxlen=50)

        # Spot price history
        self._spot_history: Deque = deque(maxlen=120)  # 2 minutes at 1s intervals
        self._spot_ewma: float = 0.0
        self._ewma_alpha: float = 0.1

        # Edge signal (set by EdgeAgent, read by coordinator)
        self._edge_signal = None   # Optional[EdgeSignal] — avoid circular import

        # Running state
        self._running = False
    
    @property
    def market(self) -> MarketData:
        """Get current market data."""
        return self._market
    
    @property
    def risk(self) -> RiskMetrics:
        """Get current risk metrics."""
        return self._risk
    
    def get_edge_signal(self):
        """Return the latest EdgeSignal from the EdgeAgent, or None."""
        return self._edge_signal

    def set_edge_signal(self, signal) -> None:
        """Called by EdgeAgent to publish its latest finding."""
        self._edge_signal = signal

    async def start(self):
        """Start the shared state service."""
        self._running = True
        logger.info("Shared state service started")

    async def stop(self):
        """Stop the shared state service."""
        self._running = False
        logger.info("Shared state service stopped")
    
    async def update_market(
        self,
        up_price: float,
        down_price: float,
        up_bids: list,
        up_asks: list,
        down_bids: list,
        down_asks: list,
        token_id_up: str = "",
        token_id_down: str = "",
        market_slug: str = "",
        time_to_expiry: int = 300,
    ):
        """Update market data from WebSocket."""
        async with self._lock:
            self._market.up_price = up_price
            self._market.down_price = down_price
            self._market.mid_price = (up_price + down_price) / 2
            self._market.spread = up_price - down_price
            
            self._market.up_bids = up_bids
            self._market.up_asks = up_asks
            self._market.down_bids = down_bids
            self._market.down_asks = down_asks
            
            # Calculate depth
            self._market.bid_depth = self._calculate_depth(up_bids, down_bids)
            self._market.ask_depth = self._calculate_depth(up_asks, down_asks)
            
            # Calculate imbalance (0 = bid heavy, 1 = ask heavy)
            total_depth = self._market.bid_depth + self._market.ask_depth
            if total_depth > 0:
                self._market.imbalance = self._market.ask_depth / total_depth
            else:
                self._market.imbalance = 0.5
            
            self._market.token_id_up = token_id_up
            self._market.token_id_down = token_id_down
            self._market.market_slug = market_slug
            self._market.time_to_expiry = time_to_expiry
            self._market.last_update = time.time()
            
            # Record for price history
            self._price_history.append({
                'timestamp': time.time(),
                'up_price': up_price,
                'down_price': down_price,
            })
    
    async def update_expiry(self, time_to_expiry: int) -> None:
        """Update only the time_to_expiry field without touching orderbook data."""
        async with self._lock:
            self._market.time_to_expiry = time_to_expiry

    def _calculate_depth(self, side_a: list, side_b: list, decay: float = 0.7) -> float:
        """Calculate depth-weighted orderbook depth combining both token sides."""
        depth = 0.0
        for i, (price, size) in enumerate(side_a[:5]):
            depth += size * (decay ** i)
        for i, (price, size) in enumerate(side_b[:5]):
            depth += size * (decay ** i)
        return depth
    
    async def update_spot(self, price: float):
        """Update BTC spot price and calculate changes."""
        async with self._lock:
            now = time.time()
            
            # Calculate 2s change - find oldest entry within 2s window
            change_2s = 0.0
            oldest_age = 0.0
            oldest_price = None
            for entry in self._spot_history:
                age = now - entry['timestamp']
                if age <= 2 and age > oldest_age:
                    oldest_age = age
                    oldest_price = entry['price']
            
            if oldest_price and oldest_price > 0:
                change_2s = (price - oldest_price) / oldest_price
            
            # Calculate 60s change — find oldest entry still within the 60s window
            change_60s = 0.0
            oldest_60s_age = 0.0
            oldest_60s_price = None
            for entry in self._spot_history:
                age = now - entry['timestamp']
                if age <= 60 and age > oldest_60s_age:
                    oldest_60s_age = age
                    oldest_60s_price = entry['price']
            if oldest_60s_price and oldest_60s_price > 0:
                change_60s = (price - oldest_60s_price) / oldest_60s_price
            
            # Update EWMA for volatility
            if self._spot_ewma == 0:
                self._spot_ewma = abs(change_60s)
            else:
                self._spot_ewma = self._ewma_alpha * abs(change_60s) + (1 - self._ewma_alpha) * self._spot_ewma
            
            # Update regime
            if self._spot_ewma > 0.003:  # >0.3% per minute
                self._risk.regime = MarketRegime.HIGH
            elif self._spot_ewma < 0.001:  # <0.1% per minute
                self._risk.regime = MarketRegime.LOW
            else:
                self._risk.regime = MarketRegime.NORMAL
            
            self._spot_history.append({
                'timestamp': now,
                'price': price,
            })
            
            self._risk.btc_spot_price = price
            self._risk.btc_spot_change_2s = change_2s
            self._risk.btc_spot_change_60s = change_60s
            self._risk.volatility_ewma = self._spot_ewma
            self._risk.last_update = now
    
    async def record_volume(self, side: str, volume: float, timestamp: float):
        """Record trade volume for VPIN calculation."""
        self._volume_buckets.append({
            'timestamp': timestamp,
            'side': side,
            'volume': volume,
        })
        await self._calculate_vpin()
    
    async def _calculate_vpin(self):
        """Calculate VPIN from recent volume buckets."""
        if len(self._volume_buckets) < 5:
            return
        
        # Calculate volume-weighted buy/sell
        buy_volume = 0.0
        sell_volume = 0.0
        
        for bucket in self._volume_buckets:
            if bucket['side'] == 'up':
                buy_volume += bucket['volume']
            else:
                sell_volume += bucket['volume']
        
        total = buy_volume + sell_volume
        if total > 0:
            # VPIN = |buy - sell| / total
            vpin = abs(buy_volume - sell_volume) / total
            self._risk.vpin_buckets.append(vpin)
            
            # Average VPIN over buckets
            self._risk.vpin = sum(self._risk.vpin_buckets) / len(self._risk.vpin_buckets)
            self._risk.vpin_last_update = time.time()
    
    async def record_cancel(self, canceled_volume: float, total_volume: float):
        """Record order cancellation for fragility calculation."""
        if total_volume > 0:
            rate = canceled_volume / total_volume
            self._cancel_history.append(rate)
            
            # Average cancel rate
            self._risk.cancel_rate = sum(self._cancel_history) / len(self._cancel_history)
            
            # Calculate fragility
            # Fragile if: high cancel rate OR low depth in top 3 ticks
            depth_coverage = self._calculate_depth_coverage()
            self._risk.depth_3tick_coverage = depth_coverage
            
            self._risk.fragility = max(
                self._risk.cancel_rate * 0.8,
                (1 - depth_coverage) * 0.5,
            )
    
    def _calculate_depth_coverage(self) -> float:
        """Calculate what % of depth is in top 3 ticks across both UP and DOWN tokens."""
        has_up = self._market.up_bids or self._market.up_asks
        has_down = self._market.down_bids or self._market.down_asks
        if not has_up and not has_down:
            return 1.0

        top3_up_bid = sum(size for _, size in list(self._market.up_bids)[:3])
        top3_up_ask = sum(size for _, size in list(self._market.up_asks)[:3])
        top3_dn_bid = sum(size for _, size in list(self._market.down_bids)[:3])
        top3_dn_ask = sum(size for _, size in list(self._market.down_asks)[:3])

        total_up_bid = sum(size for _, size in self._market.up_bids)
        total_up_ask = sum(size for _, size in self._market.up_asks)
        total_dn_bid = sum(size for _, size in self._market.down_bids)
        total_dn_ask = sum(size for _, size in self._market.down_asks)

        total_all = total_up_bid + total_up_ask + total_dn_bid + total_dn_ask
        if total_all == 0:
            return 1.0

        top3_all = top3_up_bid + top3_up_ask + top3_dn_bid + top3_dn_ask
        return top3_all / total_all
    
    def get_market_data(self) -> MarketData:
        """Get current market data (thread-safe)."""
        return self._market
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics (thread-safe)."""
        return self._risk
    
    def get_vpin(self) -> float:
        """Get current VPIN."""
        return self._risk.vpin
    
    def get_fragility(self) -> float:
        """Get current orderbook fragility."""
        return self._risk.fragility
    
    def get_regime(self) -> MarketRegime:
        """Get current market regime."""
        return self._risk.regime
    
    def get_imbalance(self) -> float:
        """Get orderbook imbalance (0=bid heavy, 1=ask heavy)."""
        return self._market.imbalance
    
    def get_spot_change(self, window: int = 2) -> float:
        """Get spot price change over window in seconds."""
        if window == 2:
            return self._risk.btc_spot_change_2s
        elif window == 60:
            return self._risk.btc_spot_change_60s
        return 0.0
