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
import logging.handlers
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
from src.websocket_client import MarketWebSocket, UserWebSocket
from src.data_api_client import DataApiClient

from lib.shared_state import SharedState
from lib.btc_price import BTCPriceTracker
from lib.market_manager import MarketManager
from bots.momentum_bot import MomentumBot, MomentumConfig
from bots.mean_reversion_bot import MeanReversionBot, MeanReversionConfig
from bots.coordinator import Coordinator, CoordinatorConfig
from bots.spread_bot import SpreadBot, SpreadConfig

# basicConfig may already have been called by an imported library, so configure
# the root logger explicitly to guarantee our level and format take effect.
_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=_LOG_FORMAT)          # install handler if none exist
_root = logging.getLogger()
_root.setLevel(logging.INFO)
for _h in _root.handlers:
    _h.setFormatter(logging.Formatter(_LOG_FORMAT))

# ── Persistent error log: logs/errors.log ─────────────────────────────────────
# Captures every ERROR+ message from all loggers (including src.client HTTP 400s).
# Written regardless of whether TUI is active; rotate at 1 MB, keep 5 backups.
_logs_dir = project_root / "logs"
_logs_dir.mkdir(exist_ok=True)
_err_handler = logging.handlers.RotatingFileHandler(
    _logs_dir / "errors.log",
    maxBytes=1_000_000,
    backupCount=5,
    encoding="utf-8",
)
_err_handler.setLevel(logging.ERROR)
_err_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
_root.addHandler(_err_handler)

# Silence noisy loggers
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("src.websocket_client").setLevel(logging.ERROR)
logging.getLogger("lib.market_manager").setLevel(logging.ERROR)
logging.getLogger("src.client").setLevel(logging.ERROR)
logging.getLogger("src.bot").setLevel(logging.WARNING)

# Bot loggers default to INFO so DEBUG spam is hidden unless --log-level DEBUG
logging.getLogger("bots.coordinator").setLevel(logging.INFO)
logging.getLogger("bots.momentum_bot").setLevel(logging.INFO)
logging.getLogger("bots.mean_reversion_bot").setLevel(logging.INFO)
logging.getLogger("bots.spread_bot").setLevel(logging.INFO)

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
        self.spread_bot: SpreadBot = None
        self.coordinator: Coordinator = None

        # User-channel WebSocket (real-time fill notifications)
        self.user_ws: UserWebSocket = None

        # Tasks
        self._ws_task: asyncio.Task = None
        self._btc_task: asyncio.Task = None
        self._coordination_task: asyncio.Task = None
        self._user_ws_task: asyncio.Task = None
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
            # Spot impulse — original 0.20% required a $130+ move in 2s (flash-crash
            # level). Calibrated to typical "sharp" intraday BTC moves:
            spot_impulse_threshold=0.0006,  # 0.06% NORMAL  (~$40 on $66k BTC)
            high_vol_impulse=0.0004,        # 0.04% HIGH vol (easier entry)
            low_vol_impulse=0.0008,         # 0.08% LOW vol  (stricter)
            cooldown_seconds=60,
            min_signal_strength=0.3,        # was 0.4 — confidence builds from spot strength
            tp1_pct=0.14,
            tp2_pct=0.28,
            initial_sl_pct=0.35,
        )

        mr_config = MeanReversionConfig(
            bankroll=bankroll,
            max_position_pct=0.05,
            # Drop threshold — logs showed 7.1% peak drop vs 8% LOW threshold.
            # Lowered to catch typical Poly dislocations (3-6% spikes are common):
            min_drop=0.05,           # 5% NORMAL  (was 10%)
            high_vol_drop=0.07,      # 7% HIGH vol (was 12%)
            low_vol_drop=0.04,       # 4% LOW vol  (was 8%)
            # z-score — 2.5 too strict for short 5m windows with thin history.
            z_threshold=1.8,         # was 2.5
            # OB support — (1 − imbalance) ≥ 0.40 instead of 0.60.
            # Default imbalance = 0.5 → ob_support = 0.5, which used to always block.
            ob_imbalance_threshold=0.40,
            # Time gate — don't waste the last 60s of a window.
            min_time_remaining=60,   # was 90s
            tp1_pct=0.22,
            tp2_pct=0.45,
            early_sl_pct=0.15,   # -15% early window  (was 40% — too wide, poor R:R)
            mid_sl_pct=0.10,     # -10% mid/late window (tighten as expiry nears)
            min_signal_strength=0.25,  # was 0.3
            cooldown_seconds=60,
        )
        
        coord_config = CoordinatorConfig(
            total_bankroll=bankroll,
            momentum_allocation=0.30,
            mean_reversion_allocation=0.30,
            spread_allocation=0.30,
            buffer_allocation=0.10,
        )

        spread_config = SpreadConfig(
            bankroll=bankroll,
            size_per_leg=15.0,        # 3× increase — SPR is near-risk-free once locked
            min_time_remaining=120,   # only spread if ≥2 min left in window
            target_low=0.44,          # spread when both sides near 50¢
            target_high=0.56,
            bid_offset=0.02,          # bid 2¢ below ask → maker pricing
            min_locked_profit=0.40,   # require ≥2.7¢/share locked (15sh × $0.027)
            single_leg_cancel_window=90,
            single_leg_sl_pct=0.15,
        )

        # Build the user-channel WebSocket from the derived CLOB API credentials.
        # The credentials are guaranteed to exist after TradingBot.__init__ calls
        # _derive_api_creds() synchronously in its constructor.
        _api_creds = self.trading_bot._api_creds
        if _api_creds and _api_creds.is_valid():
            self.user_ws = UserWebSocket(
                api_key=_api_creds.api_key,
                secret=_api_creds.secret,
                passphrase=_api_creds.passphrase,
            )
            logger.info("UserWebSocket created (apiKey=%s…)", _api_creds.api_key[:8])
        else:
            logger.warning(
                "CLOB API credentials not available — user-channel WS disabled. "
                "Fill detection will fall back to HTTP polling."
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
            user_ws=self.user_ws,
        )

        self.spread_bot = SpreadBot(
            self.trading_bot,
            self.shared_state,
            spread_config,
            user_ws=self.user_ws,
        )

        self.coordinator = Coordinator(
            self.momentum_bot,
            self.mean_reversion_bot,
            self.shared_state,
            coord_config,
            spread_bot=self.spread_bot,
            data_api=DataApiClient(),
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

        # Register a real-time WS callback so every book event immediately
        # pushes to shared_state — no 500 ms polling delay.
        @self.market_manager.on_book_update
        async def _on_ws_book(snapshot) -> None:  # pyright: ignore[reportUnusedFunction]
            try:
                up_book = self.market_manager.get_orderbook("up")
                down_book = self.market_manager.get_orderbook("down")
                if not up_book or not down_book:
                    return
                market = self.market_manager.current_market
                if not market:
                    return
                mins, secs = market.get_countdown()
                tte = mins * 60 + secs if mins >= 0 else 300
                await self.shared_state.update_market(
                    up_price=up_book.mid_price,
                    down_price=down_book.mid_price,
                    up_bids=[(b.price, b.size) for b in up_book.bids[:10]],
                    up_asks=[(a.price, a.size) for a in up_book.asks[:10]],
                    down_bids=[(b.price, b.size) for b in down_book.bids[:10]],
                    down_asks=[(a.price, a.size) for a in down_book.asks[:10]],
                    token_id_up=market.up_token,
                    token_id_down=market.down_token,
                    market_slug=market.slug,
                    time_to_expiry=tte,
                )
            except Exception as e:
                logger.debug("WS book push error: %s", e)

        # Seed neg_risk cache for the initial market so place_order never
        # needs to make a network call.  Also re-seed on every market rotation.
        if self.market_manager.current_market:
            self._seed_neg_risk_cache(self.market_manager.current_market)

        @self.market_manager.on_market_change
        def _on_market_change(market) -> None:  # pyright: ignore[reportUnusedFunction]
            self._seed_neg_risk_cache(market)

        # Start background tasks
        # _ws_task now runs a lightweight 1-second ticker that updates only
        # time_to_expiry (countdown changes with wall clock, not WS events).
        # Orderbook data is pushed in real-time by the callback above.
        self._ws_task = asyncio.create_task(self._expiry_ticker_loop())

        if self.btc_price:
            self._btc_task = asyncio.create_task(self._btc_update_loop())

        self._coordination_task = asyncio.create_task(self._coordination_loop())

        if self.user_ws:
            self._user_ws_task = asyncio.create_task(self.user_ws.run())
            logger.info("UserWebSocket task started")

        logger.info("Dual-bot system started")
    
    async def stop(self):
        """Stop all components."""
        self._running = False
        
        # Cancel tasks
        for task in (
            self._ws_task,
            self._btc_task,
            self._coordination_task,
            self._user_ws_task,
        ):
            if task:
                task.cancel()

        # Stop components
        if self.market_manager:
            await self.market_manager.stop()

        if self.btc_price:
            await self.btc_price.stop()
        await self.shared_state.stop()
        await self.coordinator.stop()
        
        logger.info("Dual-bot system stopped")
    
    def _seed_neg_risk_cache(self, market) -> None:
        """Pre-populate clob_client._neg_risk_cache from Gamma market data.

        Called at startup and on every market rotation so get_neg_risk()
        returns instantly from cache without making any HTTP request.
        BTC/ETH/SOL crypto markets are always neg_risk=False.
        """
        clob = self.trading_bot.clob_client
        for token_id in market.token_ids.values():
            if token_id:
                clob.set_neg_risk(token_id, market.neg_risk)
        logger.debug(
            "neg_risk cache seeded for %s: neg_risk=%s",
            market.slug, market.neg_risk,
        )

    async def _expiry_ticker_loop(self):
        """Keep time_to_expiry current via a 1-second wall-clock tick.

        Orderbook data is pushed in real-time by the on_book_update WS callback
        registered in start().  This loop only needs to advance the countdown
        because time_to_expiry changes with the clock, not with market events.
        """
        while self._running:
            try:
                market = self.market_manager.current_market
                if market:
                    mins, secs = market.get_countdown()
                    tte = mins * 60 + secs if mins >= 0 else 300
                    await self.shared_state.update_expiry(tte)
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Expiry ticker error: %s", e)
                await asyncio.sleep(1.0)
    
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
        _last_heartbeat = 0.0
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

                # Heartbeat every 30 s — one-line status summary at INFO
                now = time.time()
                if now - _last_heartbeat >= 30:
                    _last_heartbeat = now
                    risk   = self.shared_state.get_risk_metrics()
                    market = self.shared_state.get_market_data()
                    spot2  = risk.btc_spot_change_2s
                    regime = risk.regime.value if hasattr(risk.regime, "value") else str(risk.regime)
                    btc    = self.btc_price.get_price() if self.btc_price else 0
                    mom_pos = "LONG" if self.momentum_bot.has_position else "idle"
                    mr_pos  = "LONG" if self.mean_reversion_bot.has_position else "idle"
                    spr_pos = "ACTIVE" if self.spread_bot.has_position else "idle"

                    logger.info(
                        "Heartbeat | BTC $%.0f spot2s=%+.3f%% regime=%s "
                        "VPIN=%.2f frag=%.2f expiry=%ds | MOM=%s MR=%s SPR=%s",
                        btc, spot2 * 100, regime,
                        risk.vpin, risk.fragility, market.time_to_expiry,
                        mom_pos, mr_pos, spr_pos,
                    )

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
    parser.add_argument("--tui", action="store_true", help="Enable trading TUI (signal/PnL dashboard)")

    args = parser.parse_args()

    # Honour --log-level flag: apply to root + bot loggers
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(level)
    for _name in ("bots.coordinator", "bots.momentum_bot", "bots.mean_reversion_bot", "bots.spread_bot"):
        logging.getLogger(_name).setLevel(level)

    # Run
    runner = DualBotRunner(args)
    if args.tui:
        _run_with_tui(runner)
    else:
        asyncio.run(runner.run())


def _run_with_tui(runner: DualBotRunner) -> None:
    """Entry point that injects the TUI render loop alongside the runner."""
    import asyncio as _aio

    async def _main():
        from apps.trading_tui import TradingEventCapture, TradingTUI, install_tui_handler

        await runner.initialize()
        await runner.start()

        # Install event capture AFTER startup so setup logs print normally
        capture = TradingEventCapture()
        install_tui_handler(capture)

        tui = TradingTUI(
            capture=capture,
            shared_state=runner.shared_state,
            coordinator=runner.coordinator,
            momentum_bot=runner.momentum_bot,
            mr_bot=runner.mean_reversion_bot,
            spread_bot=runner.spread_bot,
            btc_tracker=runner.btc_price,
            refresh_interval=0.5,
        )
        tui_task = _aio.create_task(tui.run())

        try:
            while runner._running:
                await _aio.sleep(10)
        except KeyboardInterrupt:
            logger.info("Shutting down…")
        finally:
            tui.stop()
            tui_task.cancel()
            await runner.stop()

    _aio.run(_main())


if __name__ == "__main__":
    main()
