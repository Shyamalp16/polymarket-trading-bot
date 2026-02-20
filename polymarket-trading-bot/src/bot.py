"""
Trading Bot Module - Main Trading Interface

A production-ready trading bot for Polymarket with:
- Gasless transactions via Builder Program
- Encrypted private key storage
- Modular strategy support
- Comprehensive order management

Example:
    from src.bot import TradingBot

    # Initialize with config
    bot = TradingBot(config_path="config.yaml")

    # Or manually
    bot = TradingBot(
        safe_address="0x...",
        builder_creds=builder_creds,
        private_key="0x..."  # or use encrypted key
    )

    # Place an order
    result = await bot.place_order(
        token_id="123...",
        price=0.65,
        size=10,
        side="BUY"
    )
"""

import os
import asyncio
import logging
import time
from typing import Optional, Dict, Any, List, Callable, TypeVar
from dataclasses import dataclass, field
from enum import Enum

from .config import Config, BuilderConfig
from .signer import OrderSigner, Order
from .client import ClobClient, RelayerClient, ApiCredentials, BalanceError, DuplicateOrderError
from .crypto import KeyManager, CryptoError, InvalidPasswordError
from lib.latency_metrics import record_latency


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

T = TypeVar("T")

class OrderSide(str, Enum):
    """Order side constants."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type constants."""
    GTC = "GTC"  # Good Till Cancelled
    GTD = "GTD"  # Good Till Date
    FOK = "FOK"  # Fill Or Kill


@dataclass
class OrderResult:
    """Result of an order operation."""
    success: bool
    order_id: Optional[str] = None
    status: Optional[str] = None
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_response(cls, response: Dict[str, Any]) -> "OrderResult":
        """Create from API response.

        IMPORTANT: The Polymarket CLOB returns ``success: true`` even for
        errors like "not enough balance / allowance" or "FOK order not filled".
        The authoritative signal is ``errorMsg`` — if it's non-empty the order
        was NOT successfully placed/filled regardless of the ``success`` flag.
        """
        raw_success = response.get("success", False)
        error_msg = response.get("errorMsg", "")

        # errorMsg is authoritative: if non-empty, the order failed.
        actually_ok = raw_success and not error_msg

        return cls(
            success=actually_ok,
            order_id=response.get("orderID") or response.get("orderId"),  # API may use orderID or orderId
            status=response.get("status"),
            message=error_msg if error_msg else ("Order placed successfully" if actually_ok else "Unknown error"),
            data=response
        )


class TradingBotError(Exception):
    """Base exception for trading bot errors."""
    pass


class NotInitializedError(TradingBotError):
    """Raised when bot is not initialized."""
    pass


class TradingBot:
    """
    Main trading bot class for Polymarket.

    Provides a high-level interface for:
    - Order placement and cancellation
    - Position management
    - Trade history
    - Gasless transactions (with Builder Program)

    Attributes:
        config: Bot configuration
        signer: Order signer instance
        clob_client: CLOB API client
        relayer_client: Relayer API client (if gasless enabled)
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Config] = None,
        safe_address: Optional[str] = None,
        builder_creds: Optional[BuilderConfig] = None,
        private_key: Optional[str] = None,
        encrypted_key_path: Optional[str] = None,
        password: Optional[str] = None,
        api_creds_path: Optional[str] = None,
        log_level: int = logging.INFO
    ):
        """
        Initialize trading bot.

        Can be initialized in multiple ways:

        1. From config file:
           bot = TradingBot(config_path="config.yaml")

        2. From Config object:
           bot = TradingBot(config=my_config)

        3. With manual parameters:
           bot = TradingBot(
               safe_address="0x...",
               builder_creds=builder_creds,
               private_key="0x..."
           )

        4. With encrypted key:
           bot = TradingBot(
               safe_address="0x...",
               encrypted_key_path="credentials/key.enc",
               password="mypassword"
           )

        Args:
            config_path: Path to config YAML file
            config: Config object
            safe_address: Safe/Proxy wallet address
            builder_creds: Builder Program credentials
            private_key: Raw private key (with 0x prefix)
            encrypted_key_path: Path to encrypted key file
            password: Password for encrypted key
            api_creds_path: Path to API credentials file
            log_level: Logging level
        """
        # Set log level
        logger.setLevel(log_level)

        # Load configuration
        if config_path:
            self.config = Config.load(config_path)
        elif config:
            self.config = config
        else:
            self.config = Config()

        # Override with provided parameters
        if safe_address:
            self.config.safe_address = safe_address
        if builder_creds:
            self.config.builder = builder_creds
            self.config.use_gasless = True

        # Initialize components
        self.signer: Optional[OrderSigner] = None
        self.clob_client: Optional[ClobClient] = None
        self.relayer_client: Optional[RelayerClient] = None
        self._api_creds: Optional[ApiCredentials] = None

        # Load private key
        if private_key:
            self.signer = OrderSigner(private_key)
        elif encrypted_key_path and password:
            self._load_encrypted_key(encrypted_key_path, password)

        # Load API credentials
        if api_creds_path:
            self._load_api_creds(api_creds_path)

        # Initialize API clients
        self._init_clients()

        # Auto-derive API credentials if we have a signer but no API creds
        if self.signer and not self._api_creds:
            self._derive_api_creds()

        logger.info(f"TradingBot initialized (gasless: {self.config.use_gasless})")

    def _load_encrypted_key(self, filepath: str, password: str) -> None:
        """Load and decrypt private key from encrypted file."""
        try:
            manager = KeyManager()
            private_key = manager.load_and_decrypt(password, filepath)
            self.signer = OrderSigner(private_key)
            logger.info(f"Loaded encrypted key from {filepath}")
        except FileNotFoundError:
            raise TradingBotError(f"Encrypted key file not found: {filepath}")
        except InvalidPasswordError:
            raise TradingBotError("Invalid password for encrypted key")
        except CryptoError as e:
            raise TradingBotError(f"Failed to load encrypted key: {e}")

    def _load_api_creds(self, filepath: str) -> None:
        """Load API credentials from file."""
        if os.path.exists(filepath):
            try:
                self._api_creds = ApiCredentials.load(filepath)
                logger.info(f"Loaded API credentials from {filepath}")
            except Exception as e:
                logger.warning(f"Failed to load API credentials: {e}")

    def _derive_api_creds(self) -> None:
        """Derive L2 API credentials from signer."""
        if not self.signer or not self.clob_client:
            return

        try:
            logger.info("Deriving L2 API credentials...")
            self._api_creds = self.clob_client.create_or_derive_api_key(self.signer)
            self.clob_client.set_api_creds(self._api_creds, address=self.signer.address)
            logger.info(f"L2 API credentials derived successfully (address: {self.signer.address})")
        except Exception as e:
            logger.warning(f"Failed to derive API credentials: {e}")
            logger.warning("Some API endpoints may not be accessible")

    def _init_clients(self) -> None:
        """Initialize API clients."""
        # Validate signature_type + funder combination.
        # A mismatch is the #1 cause of "not enough balance / allowance".
        sig_type = self.config.clob.signature_type
        funder = self.config.safe_address

        sig_type_labels = {0: "EOA", 1: "Email/Magic proxy", 2: "Browser wallet proxy"}
        sig_label = sig_type_labels.get(sig_type, f"unknown({sig_type})")

        if sig_type in (1, 2) and not funder:
            logger.warning(
                f"signature_type={sig_type} ({sig_label}) requires a funder/safe_address, "
                f"but none is set.  Orders will fail with 'not enough balance / allowance'. "
                f"Set safe_address in config.yaml or POLY_SAFE_ADDRESS env var."
            )
        elif sig_type == 0 and funder:
            logger.warning(
                f"signature_type=0 (EOA) does not use a funder address, "
                f"but safe_address is set to {funder}. "
                f"If you use a proxy wallet, change signature_type to 1 or 2."
            )

        logger.info(
            f"CLOB client: signature_type={sig_type} ({sig_label}), "
            f"funder={funder or '(none)'}"
        )

        # CLOB client
        self.clob_client = ClobClient(
            host=self.config.clob.host,
            chain_id=self.config.clob.chain_id,
            signature_type=self.config.clob.signature_type,
            funder=self.config.safe_address,
            api_creds=self._api_creds,
            builder_creds=self.config.builder if self.config.use_gasless else None,
        )

        # Relayer client (for gasless)
        if self.config.use_gasless:
            self.relayer_client = RelayerClient(
                host=self.config.relayer.host,
                chain_id=self.config.clob.chain_id,
                builder_creds=self.config.builder,
                tx_type=self.config.relayer.tx_type,
            )
            logger.info("Relayer client initialized (gasless enabled)")

    async def _run_in_thread(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run a blocking call in a worker thread to avoid event loop stalls."""
        return await asyncio.to_thread(func, *args, **kwargs)

    def is_initialized(self) -> bool:
        """Check if bot is properly initialized."""
        return (
            self.signer is not None and
            self.config.safe_address and
            self.clob_client is not None
        )

    def require_signer(self) -> OrderSigner:
        """Get signer or raise if not initialized."""
        if not self.signer:
            raise NotInitializedError(
                "Signer not initialized. Provide private_key or encrypted_key."
            )
        return self.signer

    async def place_order(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str,
        order_type: str = "GTC",
        tick_size: float = 0.01,
        post_only: bool = False,
    ) -> OrderResult:
        """
        Place an order.

        Args:
            token_id: Market token ID
            price: Price per share (0-1)
            size: Number of shares
            side: 'BUY' or 'SELL'
            order_type: Order type (GTC, GTD, FOK, FAK)
            tick_size: Market tick size (from WS tick_size_change events)
            post_only: If True, reject if order would cross the spread (GTC/GTD only)

        Returns:
            OrderResult with order status
        """
        signer = self.require_signer()

        try:
            # Query neg_risk for this token (cached after first call).
            # Neg-risk markets use a different exchange contract for signing.
            neg_risk = await self._run_in_thread(
                self.clob_client.get_neg_risk, token_id
            )

            # Polymarket validates feeRateBps server-side and rejects orders
            # where it doesn't match the market's configured fee.
            # Current markets charge 1000 bps (10%) for both maker and taker.
            fee_rate_bps = 1000

            # Clamp price to SDK maximum of 0.99
            price = min(price, 0.99)

            # Create order — tick_size ensures price is snapped to the market's
            # current grid (avoids INVALID_ORDER_MIN_TICK_SIZE rejections when
            # the tick changes after a price crosses 0.96 or 0.04).
            order = Order(
                token_id=token_id,
                price=price,
                size=size,
                side=side,
                maker=self.config.safe_address,
                fee_rate_bps=fee_rate_bps,
                order_type=order_type,
                tick_size=tick_size,
            )

            # Sign order with correct exchange domain
            signed = signer.sign_order(order, neg_risk=neg_risk)

            # Submit to CLOB
            submit_started = time.perf_counter()
            response = await self._run_in_thread(
                self.clob_client.post_order,
                signed,
                order_type,
                post_only,
            )
            record_latency(
                "clob_post_order_ms",
                (time.perf_counter() - submit_started) * 1000.0,
                {"side": side, "order_type": order_type},
            )

            result = OrderResult.from_response(response)
            if result.success:
                logger.info(
                    "Order placed: %s %.2f@%.4f (token: %s… neg_risk=%s)",
                    side, size, price, token_id[:16], neg_risk,
                )
            else:
                logger.warning(
                    "Order rejected by CLOB: %s %.2f@%.4f type=%s — %s",
                    side, size, price, order_type, result.message,
                )
            return result

        except BalanceError as e:
            logger.warning(f"Balance/allowance error: {e}")
            return OrderResult(
                success=False,
                message=str(e)
            )
        except DuplicateOrderError as e:
            logger.warning(
                f"Duplicate order rejected by CLOB: {side} {size}@{price} "
                f"token={token_id[:16]}..."
            )
            return OrderResult(
                success=False,
                message=str(e),
                data={"duplicate": True},
            )
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return OrderResult(
                success=False,
                message=str(e)
            )

    async def place_orders(
        self,
        orders: List[Dict[str, Any]],
        order_type: str = "GTC"
    ) -> List[OrderResult]:
        """
        Place multiple orders.

        Args:
            orders: List of order dictionaries with keys:
                - token_id: Market token ID
                - price: Price per share
                - size: Number of shares
                - side: 'BUY' or 'SELL'
            order_type: Order type (GTC, GTD, FOK)

        Returns:
            List of OrderResults
        """
        results = []
        for order_data in orders:
            result = await self.place_order(
                token_id=order_data["token_id"],
                price=order_data["price"],
                size=order_data["size"],
                side=order_data["side"],
                order_type=order_type,
            )
            results.append(result)

            # Small delay between orders to avoid rate limits
            await asyncio.sleep(0.1)

        return results

    async def cancel_order(self, order_id: str) -> OrderResult:
        """
        Cancel a specific order.

        Args:
            order_id: Order ID to cancel

        Returns:
            OrderResult with cancellation status
        """
        try:
            response = await self._run_in_thread(self.clob_client.cancel_order, order_id)
            logger.info(f"Order cancelled: {order_id}")
            return OrderResult(
                success=True,
                order_id=order_id,
                message="Order cancelled",
                data=response
            )
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return OrderResult(
                success=False,
                order_id=order_id,
                message=str(e)
            )

    async def cancel_all_orders(self) -> OrderResult:
        """
        Cancel all open orders.

        Returns:
            OrderResult with cancellation status
        """
        try:
            response = await self._run_in_thread(self.clob_client.cancel_all_orders)
            logger.info("All orders cancelled")
            return OrderResult(
                success=True,
                message="All orders cancelled",
                data=response
            )
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")
            return OrderResult(success=False, message=str(e))

    async def cancel_market_orders(
        self,
        market: Optional[str] = None,
        asset_id: Optional[str] = None
    ) -> OrderResult:
        """
        Cancel orders for a specific market.

        Args:
            market: Condition ID of the market (optional)
            asset_id: Token/asset ID (optional)

        Returns:
            OrderResult with cancellation status
        """
        try:
            response = await self._run_in_thread(
                self.clob_client.cancel_market_orders,
                market,
                asset_id,
            )
            logger.info(f"Market orders cancelled (market: {market or 'all'}, asset: {asset_id or 'all'})")
            return OrderResult(
                success=True,
                message=f"Orders cancelled for market {market or 'all'}",
                data=response
            )
        except Exception as e:
            logger.error(f"Failed to cancel market orders: {e}")
            return OrderResult(success=False, message=str(e))

    async def get_open_orders(self) -> List[Dict[str, Any]]:
        """
        Get all open orders.

        Returns:
            List of open orders
        """
        try:
            orders = await self._run_in_thread(self.clob_client.get_open_orders)
            logger.debug(f"Retrieved {len(orders)} open orders")
            return orders
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    async def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get order details.

        Args:
            order_id: Order ID

        Returns:
            Order details or None
        """
        try:
            return await self._run_in_thread(self.clob_client.get_order, order_id)
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None

    async def get_balance_allowance(
        self,
        asset_type: str = "COLLATERAL",
        token_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached balance and allowance from CLOB.

        Args:
            asset_type: "COLLATERAL" or "CONDITIONAL"
            token_id: token id when using CONDITIONAL

        Returns:
            Balance/allowance dictionary or None on failure
        """
        try:
            return await self._run_in_thread(
                self.clob_client.get_balance_allowance,
                asset_type,
                token_id,
            )
        except Exception as e:
            logger.error(f"Failed to get balance/allowance: {e}")
            return None

    async def update_balance_allowance(
        self,
        asset_type: str = "CONDITIONAL",
        token_id: Optional[str] = None,
    ) -> None:
        """
        Refresh the CLOB's cached balance for a token.

        Call before selling after a FOK buy so the CLOB recognizes
        the newly acquired conditional tokens.
        """
        try:
            await self._run_in_thread(
                self.clob_client.update_balance_allowance,
                asset_type,
                token_id,
            )
        except Exception:
            pass  # Best-effort

    async def refresh_balances(self, token_id: Optional[str] = None) -> None:
        """
        Refresh CLOB balance cache for both COLLATERAL (USDC) and CONDITIONAL tokens.

        Should be called after any trade to ensure subsequent balance
        checks / order submissions use fresh data.

        Args:
            token_id: Conditional token ID to refresh (optional)
        """
        await self.update_balance_allowance("COLLATERAL")
        if token_id:
            await self.update_balance_allowance("CONDITIONAL", token_id)

    async def get_trades(
        self,
        token_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get trade history.

        Args:
            token_id: Optional token ID to filter
            limit: Maximum number of trades

        Returns:
            List of trades
        """
        try:
            trades = await self._run_in_thread(self.clob_client.get_trades, token_id, limit)
            logger.debug(f"Retrieved {len(trades)} trades")
            return trades
        except Exception as e:
            logger.error(f"Failed to get trades: {e}")
            return []

    async def get_order_book(self, token_id: str) -> Dict[str, Any]:
        """
        Get order book for a token.

        Args:
            token_id: Market token ID

        Returns:
            Order book data
        """
        try:
            return await self._run_in_thread(self.clob_client.get_order_book, token_id)
        except Exception as e:
            logger.error(f"Failed to get order book: {e}")
            return {}

    async def get_market_price(self, token_id: str) -> Dict[str, Any]:
        """
        Get current market price for a token.

        Args:
            token_id: Market token ID

        Returns:
            Price data
        """
        try:
            return await self._run_in_thread(self.clob_client.get_market_price, token_id)
        except Exception as e:
            logger.error(f"Failed to get market price: {e}")
            return {}

    async def deploy_safe_if_needed(self) -> bool:
        """
        Deploy Safe proxy wallet if not already deployed.

        Returns:
            True if deployment was needed or successful
        """
        if not self.config.use_gasless or not self.relayer_client:
            logger.debug("Gasless not enabled, skipping Safe deployment")
            return False

        try:
            response = await self._run_in_thread(
                self.relayer_client.deploy_safe,
                self.config.safe_address,
            )
            logger.info(f"Safe deployment initiated: {response}")
            return True
        except Exception as e:
            logger.warning(f"Safe deployment failed (may already be deployed): {e}")
            return False

    def create_order_dict(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str
    ) -> Dict[str, Any]:
        """
        Create an order dictionary for batch processing.

        Args:
            token_id: Market token ID
            price: Price per share
            size: Number of shares
            side: 'BUY' or 'SELL'

        Returns:
            Order dictionary
        """
        return {
            "token_id": token_id,
            "price": price,
            "size": size,
            "side": side.upper(),
        }


# Convenience function for quick initialization
def create_bot(
    config_path: str = "config.yaml",
    private_key: Optional[str] = None,
    encrypted_key_path: Optional[str] = None,
    password: Optional[str] = None,
    **kwargs
) -> TradingBot:
    """
    Create a TradingBot instance with common options.

    Args:
        config_path: Path to config file
        private_key: Private key (with 0x prefix)
        encrypted_key_path: Path to encrypted key file
        password: Password for encrypted key
        **kwargs: Additional arguments for TradingBot

    Returns:
        Configured TradingBot instance
    """
    return TradingBot(
        config_path=config_path,
        private_key=private_key,
        encrypted_key_path=encrypted_key_path,
        password=password,
        **kwargs
    )
