"""
Dual-Bot Trading System - Main Entry Point

Coordinates momentum and mean-reversion bots for Polymarket 5m BTC markets.

Usage:
    python -m bots.run_dual_bot
    
    # Or with custom config:
    python -m bots.run_dual_bot --bankroll 100 --coin BTC

Environment Variables:
    POLY_PRIVATE_KEY: Your private key
    POLY_SAFE_ADDRESS: Your proxy wallet address
    POLY_RPC_URL: Polygon RPC URL
    POLY_BUILDER_API_KEY: Builder API key (optional)
    POLY_BUILDER_API_SECRET: Builder API secret (optional)
    POLY_BUILDER_API_PASSPHRASE: Builder passphrase (optional)
"""

import asyncio
import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env file
from dotenv import load_dotenv
load_dotenv()

from src.config import Config
from src.bot import TradingBot
from src.gamma_client import GammaClient
from src.websocket_client import MarketWebSocket

from lib.shared_state import SharedState
from lib.btc_price import BTCPriceTracker
from lib.market_manager import MarketManager

from bots.momentum_bot import MomentumBot, MomentumConfig
from bots.mean_reversion_bot import MeanReversionBot, MeanReversionConfig
from bots.coordinator import Coordinator, CoordinatorConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Silence noisy loggers
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("src.websocket_client").setLevel(logging.ERROR)
logging.getLogger("lib.market_manager").setLevel(logging.ERROR)
logging.getLogger("src.client").setLevel(logging.ERROR)
logging.getLogger("src.bot").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class DualBotRunner:
    """Main runner for the dual-bot trading system."""
    
    def __init__(self, args):
        self.args = args
        
        # Components
        self.config: Config = None
        self.trading_bot: TradingBot = None
        self.shared_state: SharedState = None
        self.btc_price: BTCPriceTracker = None
        self.market_manager: MarketManager = None
        
        # Bots
        self.momentum_bot: MomentumBot = None
        self.mean_reversion_bot: MeanReversionBot = None
        self.coordinator: Coordinator = None
        
        # Tasks
        self._ws_task: asyncio.Task = None
        self._btc_task: asyncio.Task = None
        self._coordination_task: asyncio.Task = None
        self._running = False
    
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing dual-bot system...")
        
        # Load config
        self.config = Config.from_env()
        
        # Get private key from env directly
        private_key = os.environ.get("POLY_PRIVATE_KEY")
        
        # Validate required config
        if not private_key:
            raise ValueError("POLY_PRIVATE_KEY environment variable is required")
        if not self.config.safe_address:
            raise ValueError("POLY_SAFE_ADDRESS environment variable is required")
        
        # Get RPC URL
        rpc_url = os.environ.get("POLY_RPC_URL") or self.config.rpc_url
        if not rpc_url:
            logger.warning("POLY_RPC_URL not set, spot tracking may not work")
        
        # Initialize trading bot
        self.trading_bot = TradingBot(
            private_key=private_key,
            safe_address=self.config.safe_address,
            builder_creds=self.config.builder if self.config.builder else None,
        )
        
        # Initialize shared state
        self.shared_state = SharedState()
        
        # Initialize BTC price tracker (from Binance/Coinbase)
        self.btc_price = BTCPriceTracker(poll_interval=1.0)
        
        # Initialize bots
        bankroll = self.args.bankroll
        
        momentum_config = MomentumConfig(
            bankroll=bankroll,
            max_position_pct=0.05,
            spot_impulse_threshold=0.0020,  # P1-4: 0.20% (strategy spec)
            high_vol_impulse=0.0018,        # 0.18% in high vol
            low_vol_impulse=0.0025,         # 0.25% in low vol
            cooldown_seconds=60,            # P1-9: unified 60s cooldown
            min_signal_strength=0.4,
            tp1_pct=0.14,
            tp2_pct=0.28,
            initial_sl_pct=0.35,
        )
        
        mr_config = MeanReversionConfig(
            bankroll=bankroll,
            max_position_pct=0.05,
            min_drop=0.10,           # 10% (strategy spec)
            z_threshold=2.5,         # P1-5: strategy spec
            tp1_pct=0.22,
            tp2_pct=0.45,
            early_sl_pct=0.40,
            mid_sl_pct=0.35,
            cooldown_seconds=60,     # P1-9: unified 60s cooldown
        )
        
        coord_config = CoordinatorConfig(
            total_bankroll=bankroll,
            momentum_allocation=0.40,
            mean_reversion_allocation=0.40,
            buffer_allocation=0.20,
        )
        
        self.momentum_bot = MomentumBot(
            self.trading_bot,
            self.shared_state,
            momentum_config,
        )
        
        self.mean_reversion_bot = MeanReversionBot(
            self.trading_bot,
            self.shared_state,
            mr_config,
        )
        
        self.coordinator = Coordinator(
            self.momentum_bot,
            self.mean_reversion_bot,
            self.shared_state,
            coord_config,
        )
        
        # Initialize market manager (5-minute markets)
        self.market_manager = MarketManager(coin=self.args.coin, market_duration=5)
        
        logger.info("Initialization complete")
    
    async def start(self):
        """Start all components."""
        self._running = True
        
        # Start shared state
        await self.shared_state.start()
        
        # Start spot tracker
        if self.btc_price:
            await self.btc_price.start()
        
        # Start market manager
        await self.market_manager.start()
        
        # Wait for market data
        await self.market_manager.wait_for_data()
        
        # Wait for API credentials to fully propagate (prevents initial order failures)
        logger.info("Waiting for API to settle...")
        await asyncio.sleep(5)  # Increased from 2s to 5s
        
        # Warm up the relayer by checking balance (forces connection)
        logger.info("Warming up relayer connection...")
        try:
            await self.trading_bot.update_balance_allowance("COLLATERAL")
            bal = await self.trading_bot.get_balance_allowance("COLLATERAL")
            if bal:
                balance = float(bal.get("balance", 0)) / 1_000_000
                logger.info(f"Balance check: ${balance:.2f} USDC available")
        except Exception as e:
            logger.warning(f"Balance check failed (non-fatal): {e}")
        
        # Extra delay to ensure everything is ready
        await asyncio.sleep(2)
        
        # Start coordinator
        await self.coordinator.start()
        
        # Start background tasks
        self._ws_task = asyncio.create_task(self._market_update_loop())
        
        if self.btc_price:
            self._btc_task = asyncio.create_task(self._btc_update_loop())
        
        self._coordination_task = asyncio.create_task(self._coordination_loop())
        
        logger.info("Dual-bot system started")
    
    async def stop(self):
        """Stop all components."""
        self._running = False
        
        # Cancel tasks
        if self._ws_task:
            self._ws_task.cancel()
        if self._btc_task:
            self._btc_task.cancel()
        if self._coordination_task:
            self._coordination_task.cancel()
        
        # Stop components
        if self.market_manager:
            await self.market_manager.stop()
        
        if self.btc_price:
            await self.btc_price.stop()
        await self.shared_state.stop()
        await self.coordinator.stop()
        
        logger.info("Dual-bot system stopped")
    
    async def _market_update_loop(self):
        """Update shared state from market data."""
        while self._running:
            try:
                # Get orderbook
                up_book = self.market_manager.get_orderbook("up")
                down_book = self.market_manager.get_orderbook("down")
                
                if up_book and down_book:
                    market = self.market_manager.current_market
                    
                    # Convert orderbook to list of tuples (price, size)
                    up_bids = [(b.price, b.size) for b in up_book.bids[:10]]
                    up_asks = [(a.price, a.size) for a in up_book.asks[:10]]
                    down_bids = [(b.price, b.size) for b in down_book.bids[:10]]
                    down_asks = [(a.price, a.size) for a in down_book.asks[:10]]
                    
                    # Update shared state
                    # P2-10: call get_countdown() once
                    if market:
                        mins, secs = market.get_countdown()
                        time_to_expiry = mins * 60 + secs if mins >= 0 else 300
                    else:
                        time_to_expiry = 300

                    await self.shared_state.update_market(
                        up_price=up_book.mid_price,
                        down_price=down_book.mid_price,
                        up_bids=up_bids,
                        up_asks=up_asks,
                        down_bids=down_bids,
                        down_asks=down_asks,
                        token_id_up=market.up_token if market else "",
                        token_id_down=market.down_token if market else "",
                        market_slug=market.slug if market else "",
                        time_to_expiry=time_to_expiry,
                    )
                
                await asyncio.sleep(0.5)  # market update: 2 Hz

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Market update error: %s", e)
                await asyncio.sleep(0.5)
    
    async def _btc_update_loop(self):
        """Update shared state from BTC price."""
        _stale_warned = False
        while self._running:
            try:
                if self.btc_price:
                    price = self.btc_price.get_price()
                    if price > 0:
                        await self.shared_state.update_spot(price)
                        _stale_warned = False
                        market = self.shared_state.get_market_data()
                        self.btc_price.set_poly_mid_price(market.mid_price)
                    elif self.btc_price.is_stale(max_age=5.0) and not _stale_warned:
                        logger.warning(
                            "BTC price is 0 or stale — spot_change will be 0. "
                            "Momentum bot cannot fire. Check Binance/Coinbase connectivity "
                            "and that aiohttp is installed (pip install aiohttp)."
                        )
                        _stale_warned = True

                await asyncio.sleep(0.3)  # P3-5: BTC price: ~3 Hz (staggered from market update)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Spot update error: %s", e)
                await asyncio.sleep(0.3)
    
    async def _coordination_loop(self):
        """Main coordination loop."""
        while self._running:
            try:
                # Check for entry signals
                entry_result = await self.coordinator.check_and_coordinate()
                
                if entry_result and entry_result.get("success"):
                    logger.info("Entry executed: %s", entry_result)
                
                # Check for exits
                await self.coordinator.check_exits()
                
                # Check late window
                await self.coordinator.check_late_window()
                
                await asyncio.sleep(0.25)  # P3-5: coordination: 4 Hz (faster than data feeds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Coordination error: %s", e)
                await asyncio.sleep(0.25)
    
    async def run(self):
        """Run the dual-bot system."""
        try:
            await self.initialize()
            await self.start()
            
            # Keep running
            while self._running:
                await asyncio.sleep(10)
                
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        except Exception as e:
            logger.error("Fatal error: %s", e)
            raise
        finally:
            await self.stop()


import time


def main():
    parser = argparse.ArgumentParser(description="Dual-Bot Trading System")
    parser.add_argument("--coin", default="BTC", help="Coin to trade (default: BTC)")
    parser.add_argument("--bankroll", type=float, default=100, help="Bankroll in USD (default: 100)")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Bot loggers - respect --log-level argument
    bot_log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger("bots.coordinator").setLevel(bot_log_level)
    logging.getLogger("bots.momentum_bot").setLevel(bot_log_level)
    logging.getLogger("bots.mean_reversion_bot").setLevel(bot_log_level)
    
    # Run
    runner = DualBotRunner(args)
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()
