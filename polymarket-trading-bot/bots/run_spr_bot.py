"""
SPR-Only Trading System — Standalone Spread Bot

Runs only the Spread Bot (Bot C) against Polymarket 5-minute markets.
All infrastructure (connections, WebSockets, market refresh, TUI) is
identical to run_dual_bot.py.  MR and Momentum bots are not created.

Usage:
    python -m bots.run_spr_bot
    python -m bots.run_spr_bot --tui
    python -m bots.run_spr_bot --bankroll 500 --coin ETH --tui

Environment Variables:
    POLY_PRIVATE_KEY:          Your private key
    POLY_SAFE_ADDRESS:         Your proxy wallet address
    POLY_RPC_URL:              Polygon RPC URL (optional — spread bot doesn't need spot)
    POLY_BUILDER_API_KEY:      Builder API key (optional)
    POLY_BUILDER_API_SECRET:   Builder API secret (optional)
    POLY_BUILDER_API_PASSPHRASE: Builder passphrase (optional)
"""

import asyncio
import argparse
import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from src.config import Config
from src.bot import TradingBot
from src.websocket_client import UserWebSocket

from lib.shared_state import SharedState
from lib.market_manager import MarketManager
from bots.spread_bot import SpreadBot, SpreadConfig

# ── Logging setup ─────────────────────────────────────────────────────────────
_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=_LOG_FORMAT)
_root = logging.getLogger()
_root.setLevel(logging.INFO)
for _h in _root.handlers:
    _h.setFormatter(logging.Formatter(_LOG_FORMAT))

_logs_dir = project_root / "logs"
_logs_dir.mkdir(exist_ok=True)
_err_handler = logging.handlers.RotatingFileHandler(
    _logs_dir / "spr_errors.log",
    maxBytes=1_000_000,
    backupCount=5,
    encoding="utf-8",
)
_err_handler.setLevel(logging.ERROR)
_err_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
_root.addHandler(_err_handler)

logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("src.websocket_client").setLevel(logging.ERROR)
logging.getLogger("lib.market_manager").setLevel(logging.ERROR)
logging.getLogger("src.client").setLevel(logging.ERROR)
logging.getLogger("src.bot").setLevel(logging.WARNING)
logging.getLogger("bots.spread_bot").setLevel(logging.INFO)

logger = logging.getLogger(__name__)


# ── Minimal coordinator stub (for TUI get_status compatibility) ───────────────

class _SprCoordinatorStub:
    """Returns a minimal status dict so the TUI can read window/late state."""

    def __init__(self, spread_bot: SpreadBot):
        self._spr = spread_bot

    def get_status(self) -> dict:
        return {
            "window": {
                "total_entries": self._spr._spreads_completed,
                "losses": 0.0,
            },
            "in_late_window": False,
        }


# ── Main runner ───────────────────────────────────────────────────────────────

class SprBotRunner:
    """Standalone SPR-only trading system."""

    def __init__(self, args):
        self.args = args

        self.config: Config = None
        self.trading_bot: TradingBot = None
        self.shared_state: SharedState = None
        self.market_manager: MarketManager = None
        self.spread_bot: SpreadBot = None
        self.user_ws: UserWebSocket = None

        self._ws_task: asyncio.Task = None
        self._spread_task: asyncio.Task = None
        self._user_ws_task: asyncio.Task = None
        self._running = False

    async def initialize(self):
        logger.info("Initializing SPR-only bot...")

        self.config = Config.from_env()
        private_key = os.environ.get("POLY_PRIVATE_KEY")
        if not private_key:
            raise ValueError("POLY_PRIVATE_KEY environment variable is required")
        if not self.config.safe_address:
            raise ValueError("POLY_SAFE_ADDRESS environment variable is required")

        self.trading_bot = TradingBot(
            private_key=private_key,
            safe_address=self.config.safe_address,
            builder_creds=self.config.builder if self.config.builder else None,
        )

        self.shared_state = SharedState()

        bankroll = self.args.bankroll
        spread_config = SpreadConfig(
            bankroll=bankroll,
            size_per_leg=15.0,
            min_time_remaining=120,
            target_low=0.44,
            target_high=0.56,
            bid_offset=0.02,
            min_locked_profit=0.40,
            single_leg_cancel_window=90,
            single_leg_sl_pct=0.15,
        )

        # User-channel WebSocket for real-time fill notifications
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

        self.spread_bot = SpreadBot(
            self.trading_bot,
            self.shared_state,
            spread_config,
            user_ws=self.user_ws,
        )

        self.market_manager = MarketManager(
            coin=self.args.coin, market_duration=5
        )
        logger.info("SPR-only bot initialized")

    async def start(self):
        self._running = True

        await self.shared_state.start()
        await self.market_manager.start()
        await self.market_manager.wait_for_data()

        logger.info("Waiting for API to settle…")
        await asyncio.sleep(5)

        logger.info("Warming up relayer connection…")
        try:
            await self.trading_bot.update_balance_allowance("COLLATERAL")
            bal = await self.trading_bot.get_balance_allowance("COLLATERAL")
            if bal:
                balance = float(bal.get("balance", 0)) / 1_000_000
                logger.info("Balance: $%.2f USDC", balance)
        except Exception as e:
            logger.warning("Balance check failed (non-fatal): %s", e)

        await asyncio.sleep(2)

        # Real-time orderbook push from WebSocket events
        @self.market_manager.on_book_update
        async def _on_ws_book(snapshot) -> None:  # pyright: ignore[reportUnusedFunction]
            try:
                up_book   = self.market_manager.get_orderbook("up")
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

        # Seed neg_risk cache
        if self.market_manager.current_market:
            self._seed_neg_risk_cache(self.market_manager.current_market)

        @self.market_manager.on_market_change
        def _on_market_change(market) -> None:  # pyright: ignore[reportUnusedFunction]
            self._seed_neg_risk_cache(market)

        self._ws_task     = asyncio.create_task(self._expiry_ticker_loop())
        self._spread_task = asyncio.create_task(self._spread_loop())

        if self.user_ws:
            self._user_ws_task = asyncio.create_task(self.user_ws.run())
            logger.info("UserWebSocket task started")

        logger.info("SPR-only bot started")

    async def stop(self):
        self._running = False
        for task in (self._ws_task, self._spread_task, self._user_ws_task):
            if task:
                task.cancel()
        if self.market_manager:
            await self.market_manager.stop()
        await self.shared_state.stop()
        logger.info("SPR-only bot stopped")

    def _seed_neg_risk_cache(self, market) -> None:
        clob = self.trading_bot.clob_client
        for token_id in market.token_ids.values():
            if token_id:
                clob.set_neg_risk(token_id, market.neg_risk)
        logger.debug(
            "neg_risk cache seeded for %s: neg_risk=%s", market.slug, market.neg_risk
        )

    async def _expiry_ticker_loop(self):
        """Advance time_to_expiry every second via wall clock."""
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

    async def _spread_loop(self):
        """Main SPR loop: check entry, check exits, heartbeat."""
        _last_heartbeat = 0.0
        _last_token_id_up: str = ""
        while self._running:
            try:
                # Detect market rotation via token_id change (same logic as coordinator)
                market_data = self.shared_state.get_market_data()
                current_token = market_data.token_id_up if market_data else ""
                if current_token and _last_token_id_up and current_token != _last_token_id_up:
                    logger.info(
                        "=== NEW MARKET: %s… → %s… ===",
                        _last_token_id_up[:8], current_token[:8],
                    )
                    await self.spread_bot.on_window_reset()
                    self.spread_bot.reset_window_pnl()
                if current_token:
                    _last_token_id_up = current_token

                # Entry check
                result = await self.spread_bot.check_spread()
                if result and result.get("success"):
                    logger.info("SPR entry: %s", result)

                # Fill monitoring (detects when legs are filled)
                await self.spread_bot.check_fills()
                # Exit check (single-leg SL, expiry close)
                await self.spread_bot.check_exit()

                # Heartbeat every 30 s
                now = time.time()
                if now - _last_heartbeat >= 30:
                    _last_heartbeat = now
                    market  = self.shared_state.get_market_data()
                    spr_pos = "ACTIVE" if self.spread_bot.has_position else "idle"
                    legs    = "PENDING" if self.spread_bot._legs else "none"
                    logger.info(
                        "SPR Heartbeat | expiry=%ds | pos=%s legs=%s done=%d",
                        market.time_to_expiry, spr_pos, legs,
                        self.spread_bot._spreads_completed,
                    )

                await asyncio.sleep(0.25)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Spread loop error: %s", e)
                await asyncio.sleep(0.25)

    async def run(self):
        try:
            await self.initialize()
            await self.start()
            while self._running:
                await asyncio.sleep(10)
        except KeyboardInterrupt:
            logger.info("Shutting down…")
        except Exception as e:
            logger.error("Fatal error: %s", e)
            raise
        finally:
            await self.stop()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SPR-Only Spread Bot")
    parser.add_argument("--coin",      default="BTC",  help="Coin to trade (default: BTC)")
    parser.add_argument("--bankroll",  type=float, default=100, help="Bankroll in USD (default: 100)")
    parser.add_argument("--log-level", default="INFO",  help="Log level")
    parser.add_argument("--tui",       action="store_true", help="Enable TUI dashboard")
    args = parser.parse_args()

    level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(level)
    logging.getLogger("bots.spread_bot").setLevel(level)

    runner = SprBotRunner(args)
    if args.tui:
        _run_with_tui(runner)
    else:
        asyncio.run(runner.run())


def _run_with_tui(runner: SprBotRunner) -> None:
    import asyncio as _aio

    async def _main():
        from apps.trading_tui import TradingEventCapture, TradingTUI, install_tui_handler

        await runner.initialize()
        await runner.start()

        capture = TradingEventCapture()
        install_tui_handler(capture)

        # Build a minimal coordinator stub so the TUI can call get_status()
        coord_stub = _SprCoordinatorStub(runner.spread_bot)

        tui = TradingTUI(
            capture=capture,
            shared_state=runner.shared_state,
            coordinator=coord_stub,
            # momentum_bot and mr_bot are omitted (None) — TUI handles this
            spread_bot=runner.spread_bot,
            btc_tracker=None,       # SPR doesn't need spot price
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
