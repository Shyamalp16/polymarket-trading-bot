"""
Client Module - API Clients for Polymarket

Provides clients for interacting with:
- CLOB (Central Limit Order Book) API
- Builder Relayer API

Features:
- Gasless transactions via Builder Program
- HMAC authentication for Builder APIs
- Automatic retry and error handling

Example:
    from src.client import ClobClient, RelayerClient

    clob = ClobClient(
        host="https://clob.polymarket.com",
        chain_id=137,
        signature_type=2,
        funder="0x..."
    )

    relayer = RelayerClient(
        host="https://relayer-v2.polymarket.com",
        chain_id=137,
        builder_creds=builder_creds
    )
"""

import time
import hmac
import hashlib
import base64
import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import requests

from .config import BuilderConfig
from .http import ThreadLocalSessionMixin

logger = logging.getLogger(__name__)


class ApiError(Exception):
    """Base exception for API errors."""
    pass


class AuthenticationError(ApiError):
    """Raised when authentication fails."""
    pass


class OrderError(ApiError):
    """Raised when order operations fail."""
    pass


class BalanceError(OrderError):
    """Raised when balance or allowance is insufficient.

    This error is NOT retryable — the caller must either:
    1. Refresh the CLOB balance cache (update_balance_allowance), or
    2. Set on-chain token allowances (scripts/approve_tokens.py), or
    3. Deposit more USDC.
    """
    pass


class DuplicateOrderError(OrderError):
    """Raised when the CLOB rejects an order as duplicated.

    This error is NOT retryable because replaying the same signed payload
    will produce the same duplicate rejection.
    """
    pass


@dataclass
class ApiCredentials:
    """User-level API credentials for CLOB."""
    api_key: str
    secret: str
    passphrase: str

    @classmethod
    def load(cls, filepath: str) -> "ApiCredentials":
        """Load credentials from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(
            api_key=data.get("apiKey", ""),
            secret=data.get("secret", ""),
            passphrase=data.get("passphrase", ""),
        )

    def is_valid(self) -> bool:
        """Check if credentials are valid."""
        return bool(self.api_key and self.secret and self.passphrase)


class ApiClient(ThreadLocalSessionMixin):
    """
    Base HTTP client with common functionality.

    Provides:
    - Automatic JSON handling
    - Request/response logging
    - Error handling
    """

    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
        retry_count: int = 3
    ):
        """
        Initialize API client.

        Args:
            base_url: Base URL for all requests
            timeout: Request timeout in seconds
            retry_count: Number of retries on failure
        """
        super().__init__()
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.retry_count = retry_count

    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Any] = None,
        headers: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request with error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request body data
            headers: Additional headers
            params: Query parameters

        Returns:
            Response JSON data

        Raises:
            ApiError: On request failure
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        request_headers = {"Content-Type": "application/json"}

        if headers:
            request_headers.update(headers)

        # POST /order must NEVER be retried with the same signed payload.
        # If the first attempt succeeded on the server but the response was
        # lost (timeout/reset), a retry sends the identical order hash and
        # triggers "Duplicated" — the bot then treats a successful fill as
        # a failure.  We detect this case and skip retries entirely.
        is_order_post = method.upper() == "POST" and endpoint.rstrip("/") == "/order"
        effective_retries = 1 if is_order_post else self.retry_count

        last_error = None
        for attempt in range(effective_retries):
            try:
                session = self.session
                if method.upper() == "GET":
                    response = session.get(
                        url, headers=request_headers,
                        params=params, timeout=self.timeout
                    )
                elif method.upper() == "POST":
                    if isinstance(data, str):
                        response = session.post(
                            url, headers=request_headers,
                            data=data, params=params, timeout=self.timeout
                        )
                    else:
                        response = session.post(
                            url, headers=request_headers,
                            json=data, params=params, timeout=self.timeout
                        )
                elif method.upper() == "DELETE":
                    if isinstance(data, str):
                        response = session.delete(
                            url, headers=request_headers,
                            data=data, params=params, timeout=self.timeout
                        )
                    else:
                        response = session.delete(
                            url, headers=request_headers,
                            json=data, params=params, timeout=self.timeout
                        )
                else:
                    raise ApiError(f"Unsupported method: {method}")

                if not response.ok:
                    # Extract the actual error message from the response body
                    try:
                        error_body = response.text
                    except Exception:
                        error_body = "(could not read response body)"
                    logger.error(
                        f"HTTP {response.status_code} {method} {endpoint}: {error_body}"
                    )

                    body_lc = error_body.lower()

                    # Balance/allowance errors are NOT retryable — break
                    # out immediately instead of wasting retries.
                    if response.status_code == 400 and "not enough balance" in body_lc:
                        raise BalanceError(
                            f"{response.status_code} {method} {endpoint}: {error_body}"
                        )

                    # Duplicate order errors are also NOT retryable.
                    # Retrying replays the same signed payload and fails again.
                    if response.status_code == 400 and "duplicated" in body_lc:
                        raise DuplicateOrderError(
                            f"{response.status_code} {method} {endpoint}: {error_body}"
                        )

                    # Raise with the real error detail so it surfaces in the TUI
                    raise ApiError(
                        f"{response.status_code} {method} {endpoint}: {error_body}"
                    )

                return response.json() if response.text else {}

            except (BalanceError, DuplicateOrderError):
                raise  # NOT retryable — propagate immediately
            except ApiError:
                raise  # Server responded with a clear rejection — propagate as-is
            except requests.exceptions.RequestException as e:
                last_error = e
                # For order POSTs, transport errors are ambiguous — the server
                # may have already accepted the order.  Do NOT retry.
                if is_order_post:
                    raise ApiError(
                        f"Order submission failed (transport error, order may "
                        f"have been accepted): {last_error}"
                    )
                if attempt < effective_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        raise ApiError(f"Request failed after {effective_retries} attempts: {last_error}")


class ClobClient(ApiClient):
    """
    Client for Polymarket CLOB (Central Limit Order Book) API.

    Features:
    - Order placement and cancellation
    - Order book queries
    - Trade history
    - Builder attribution support

    Example:
        client = ClobClient(
            host="https://clob.polymarket.com",
            chain_id=137,
            signature_type=2,
            funder="0x..."
        )
    """

    def __init__(
        self,
        host: str = "https://clob.polymarket.com",
        chain_id: int = 137,
        signature_type: int = 2,
        funder: str = "",
        api_creds: Optional[ApiCredentials] = None,
        builder_creds: Optional[BuilderConfig] = None,
        timeout: int = 30
    ):
        """
        Initialize CLOB client.

        Args:
            host: CLOB API host
            chain_id: Chain ID (137 for Polygon mainnet)
            signature_type: Signature type (2 = Gnosis Safe)
            funder: Funder/Safe address
            api_creds: User API credentials (optional)
            builder_creds: Builder credentials for attribution (optional)
            timeout: Request timeout
        """
        super().__init__(base_url=host, timeout=timeout)
        self.host = host
        self.chain_id = chain_id
        self.signature_type = signature_type
        self.funder = funder
        self.api_creds = api_creds
        self.builder_creds = builder_creds
        # Address associated with L2 API credentials (EOA address)
        # Falls back to funder if not explicitly set
        self._l2_address = ""

    def _build_headers(
        self,
        method: str,
        path: str,
        body: str = ""
    ) -> Dict[str, str]:
        """
        Build authentication headers.

        Supports both user API credentials and Builder credentials.

        Args:
            method: HTTP method
            path: Request path
            body: Request body

        Returns:
            Dictionary of headers
        """
        headers = {}

        # Builder HMAC authentication
        if self.builder_creds and self.builder_creds.is_configured():
            timestamp = str(int(time.time()))

            message = f"{timestamp}{method}{path}{body}"
            signature = hmac.new(
                self.builder_creds.api_secret.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()

            headers.update({
                "POLY_BUILDER_API_KEY": self.builder_creds.api_key,
                "POLY_BUILDER_TIMESTAMP": timestamp,
                "POLY_BUILDER_PASSPHRASE": self.builder_creds.api_passphrase,
                "POLY_BUILDER_SIGNATURE": signature,
            })

        # User API credentials (L2 authentication)
        if self.api_creds and self.api_creds.is_valid():
            timestamp = str(int(time.time()))

            # Build message: timestamp + method + path + body
            message = f"{timestamp}{method}{path}"
            if body:
                message += body

            # Decode base64 secret and create HMAC signature
            try:
                base64_secret = base64.urlsafe_b64decode(self.api_creds.secret)
                h = hmac.new(base64_secret, message.encode("utf-8"), hashlib.sha256)
                signature = base64.urlsafe_b64encode(h.digest()).decode("utf-8")
            except Exception:
                # Fallback: use secret directly if not base64 encoded
                signature = hmac.new(
                    self.api_creds.secret.encode(),
                    message.encode(),
                    hashlib.sha256
                ).hexdigest()

            # POLY_ADDRESS must be the signer's EOA address (the address
            # that derived the API key), matching the official py-clob-client.
            # The CLOB resolves signer → funder/Safe internally using
            # the signature_type parameter on balance endpoints.
            headers.update({
                "POLY_ADDRESS": self._l2_address or self.funder,
                "POLY_API_KEY": self.api_creds.api_key,
                "POLY_TIMESTAMP": timestamp,
                "POLY_PASSPHRASE": self.api_creds.passphrase,
                "POLY_SIGNATURE": signature,
            })

        return headers

    def derive_api_key(self, signer: "OrderSigner", nonce: int = 0) -> ApiCredentials:
        """
        Derive L2 API credentials using L1 EIP-712 authentication.

        This is required to access authenticated endpoints like
        /orders and /trades.

        Args:
            signer: OrderSigner instance with private key
            nonce: Nonce for the auth message (default 0)

        Returns:
            ApiCredentials with api_key, secret, and passphrase
        """
        timestamp = str(int(time.time()))

        # Sign the auth message using EIP-712
        auth_signature = signer.sign_auth_message(timestamp=timestamp, nonce=nonce)

        # L1 headers
        headers = {
            "POLY_ADDRESS": signer.address,
            "POLY_SIGNATURE": auth_signature,
            "POLY_TIMESTAMP": timestamp,
            "POLY_NONCE": str(nonce),
        }

        response = self._request("GET", "/auth/derive-api-key", headers=headers)

        return ApiCredentials(
            api_key=response.get("apiKey", ""),
            secret=response.get("secret", ""),
            passphrase=response.get("passphrase", ""),
        )

    def create_api_key(self, signer: "OrderSigner", nonce: int = 0) -> ApiCredentials:
        """
        Create new L2 API credentials using L1 EIP-712 authentication.

        Use this if derive_api_key fails (first time setup).

        Args:
            signer: OrderSigner instance with private key
            nonce: Nonce for the auth message (default 0)

        Returns:
            ApiCredentials with api_key, secret, and passphrase
        """
        timestamp = str(int(time.time()))

        # Sign the auth message using EIP-712
        auth_signature = signer.sign_auth_message(timestamp=timestamp, nonce=nonce)

        # L1 headers
        headers = {
            "POLY_ADDRESS": signer.address,
            "POLY_SIGNATURE": auth_signature,
            "POLY_TIMESTAMP": timestamp,
            "POLY_NONCE": str(nonce),
        }

        response = self._request("POST", "/auth/api-key", headers=headers)

        return ApiCredentials(
            api_key=response.get("apiKey", ""),
            secret=response.get("secret", ""),
            passphrase=response.get("passphrase", ""),
        )

    def create_or_derive_api_key(self, signer: "OrderSigner", nonce: int = 0) -> ApiCredentials:
        """
        Create API credentials if not exists, otherwise derive them.

        Args:
            signer: OrderSigner instance with private key
            nonce: Nonce for the auth message (default 0)

        Returns:
            ApiCredentials with api_key, secret, and passphrase
        """
        try:
            return self.create_api_key(signer, nonce)
        except Exception:
            return self.derive_api_key(signer, nonce)

    def set_api_creds(self, creds: ApiCredentials, address: str = "") -> None:
        """Set API credentials for authenticated requests.

        Args:
            creds: API credentials (api_key, secret, passphrase)
            address: The address the credentials were derived for (EOA address).
                     If empty, falls back to self.funder.
        """
        self.api_creds = creds
        if address:
            self._l2_address = address

    def get_order_book(self, token_id: str) -> Dict[str, Any]:
        """
        Get order book for a token.

        Args:
            token_id: Market token ID

        Returns:
            Order book data
        """
        return self._request(
            "GET",
            "/book",
            params={"token_id": token_id}
        )

    def get_market_price(self, token_id: str) -> Dict[str, Any]:
        """
        Get current market price for a token.

        Args:
            token_id: Market token ID

        Returns:
            Price data
        """
        return self._request(
            "GET",
            "/price",
            params={"token_id": token_id}
        )

    def get_open_orders(self) -> List[Dict[str, Any]]:
        """
        Get all open orders for the funder.

        Returns:
            List of open orders
        """
        endpoint = "/data/orders"

        headers = self._build_headers("GET", endpoint)

        result = self._request(
            "GET",
            endpoint,
            headers=headers
        )

        # Handle paginated response
        if isinstance(result, dict) and "data" in result:
            return result.get("data", [])
        return result if isinstance(result, list) else []

    def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Get order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order details
        """
        endpoint = f"/data/order/{order_id}"
        headers = self._build_headers("GET", endpoint)
        return self._request("GET", endpoint, headers=headers)

    def get_trades(
        self,
        token_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get trade history.

        Args:
            token_id: Filter by token (optional)
            limit: Maximum number of trades

        Returns:
            List of trades
        """
        endpoint = "/data/trades"
        headers = self._build_headers("GET", endpoint)
        params: Dict[str, Any] = {"limit": limit}
        if token_id:
            params["token_id"] = token_id

        result = self._request(
            "GET",
            endpoint,
            headers=headers,
            params=params
        )

        # Handle paginated response
        if isinstance(result, dict) and "data" in result:
            return result.get("data", [])
        return result if isinstance(result, list) else []

    def get_balance_allowance(
        self,
        asset_type: str = "COLLATERAL",
        token_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get cached balance and allowance for collateral/conditional tokens.

        Args:
            asset_type: "COLLATERAL" or "CONDITIONAL"
            token_id: Required for CONDITIONAL asset type

        Returns:
            {"balance": "...", "allowance": "..."} (string values)
        """
        endpoint = "/balance-allowance"
        headers = self._build_headers("GET", endpoint)
        params: Dict[str, Any] = {
            "asset_type": asset_type,
            "signature_type": self.signature_type,
        }
        if token_id:
            params["token_id"] = token_id
        return self._request("GET", endpoint, headers=headers, params=params)

    def get_neg_risk(self, token_id: str) -> bool:
        """
        Check if a token belongs to a neg-risk market.

        Neg-risk markets use a different exchange contract for signing.
        Result is cached per token_id to avoid repeated API calls.

        Args:
            token_id: The conditional token ID

        Returns:
            True if the market is neg-risk
        """
        if not hasattr(self, "_neg_risk_cache"):
            self._neg_risk_cache: Dict[str, bool] = {}

        if token_id in self._neg_risk_cache:
            return self._neg_risk_cache[token_id]

        endpoint = "/neg-risk"
        try:
            result = self._request(
                "GET", endpoint, params={"token_id": token_id}
            )
            neg_risk = bool(result.get("neg_risk", False))
        except Exception:
            neg_risk = False  # Assume non-neg-risk on error
            logger.warning(f"Failed to query neg_risk for {token_id[:16]}..., defaulting to False")

        self._neg_risk_cache[token_id] = neg_risk
        logger.info(f"Token {token_id[:16]}... neg_risk={neg_risk}")
        return neg_risk

    def update_balance_allowance(
        self,
        asset_type: str = "CONDITIONAL",
        token_id: Optional[str] = None,
    ) -> None:
        """
        Refresh the CLOB server's cached balance/allowance for a token.

        Must be called before selling after a FOK buy, otherwise
        the CLOB may reject the sell with "not enough balance / allowance"
        because its cache hasn't registered the newly acquired tokens.

        Uses GET /balance-allowance/update with query params (matching
        the official py-clob-client SDK).

        Args:
            asset_type: "COLLATERAL" or "CONDITIONAL"
            token_id: Token ID to refresh (required for CONDITIONAL)
        """
        endpoint = "/balance-allowance/update"
        headers = self._build_headers("GET", endpoint)
        params: Dict[str, Any] = {
            "asset_type": asset_type,
            "signature_type": self.signature_type,
        }
        if token_id:
            params["token_id"] = token_id
        try:
            result = self._request("GET", endpoint, headers=headers, params=params)
            logger.info(
                f"Balance cache refresh OK ({asset_type}): {result}"
            )
        except Exception as e:
            logger.warning(f"Balance cache refresh FAILED ({asset_type}): {e}")

    def post_order(
        self,
        signed_order: Dict[str, Any],
        order_type: str = "GTC"
    ) -> Dict[str, Any]:
        """
        Submit a signed order.

        Args:
            signed_order: Order with signature
            order_type: Order type (GTC, GTD, FOK)

        Returns:
            Response with order ID and status
        """
        endpoint = "/order"

        # owner must be the L2 API key (order owner), not the funder address
        owner = self.api_creds.api_key if (self.api_creds and self.api_creds.is_valid()) else self.funder

        # Build request body — signature is already inside the order dict
        body = {
            "order": signed_order.get("order", signed_order),
            "owner": owner,
            "orderType": order_type,
        }

        body_json = json.dumps(body, separators=(',', ':'))
        headers = self._build_headers("POST", endpoint, body_json)

        # Send exact body string used for L2 HMAC so server verification succeeds
        return self._request(
            "POST",
            endpoint,
            data=body_json,
            headers=headers
        )

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            Cancellation response
        """
        endpoint = "/order"
        body = {"orderID": order_id}
        body_json = json.dumps(body, separators=(',', ':'))
        headers = self._build_headers("DELETE", endpoint, body_json)

        # Send exact body string used for HMAC so server verification succeeds
        return self._request(
            "DELETE",
            endpoint,
            data=body_json,
            headers=headers
        )

    def cancel_orders(self, order_ids: List[str]) -> Dict[str, Any]:
        """
        Cancel multiple orders by their IDs.

        Args:
            order_ids: List of order IDs to cancel

        Returns:
            Cancellation response with canceled and not_canceled lists
        """
        endpoint = "/orders"
        body_json = json.dumps(order_ids, separators=(',', ':'))
        headers = self._build_headers("DELETE", endpoint, body_json)

        # Send exact body string used for HMAC so server verification succeeds
        return self._request(
            "DELETE",
            endpoint,
            data=body_json,
            headers=headers
        )

    def cancel_all_orders(self) -> Dict[str, Any]:
        """
        Cancel all open orders.

        Returns:
            Cancellation response with canceled and not_canceled lists
        """
        endpoint = "/cancel-all"
        headers = self._build_headers("DELETE", endpoint)

        return self._request(
            "DELETE",
            endpoint,
            headers=headers
        )

    def cancel_market_orders(
        self,
        market: Optional[str] = None,
        asset_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Cancel orders for a specific market.

        Args:
            market: Condition ID of the market (optional)
            asset_id: Token/asset ID (optional)

        Returns:
            Cancellation response with canceled and not_canceled lists
        """
        endpoint = "/cancel-market-orders"
        body = {}

        if market:
            body["market"] = market
        if asset_id:
            body["asset_id"] = asset_id

        body_json = json.dumps(body, separators=(',', ':')) if body else ""
        headers = self._build_headers("DELETE", endpoint, body_json)

        # Send exact body string used for HMAC so server verification succeeds
        return self._request(
            "DELETE",
            endpoint,
            data=body_json if body_json else None,
            headers=headers
        )


class RelayerClient(ApiClient):
    """
    Client for Builder Relayer API.

    Provides gasless transactions through Polymarket's
    relayer infrastructure.

    Example:
        client = RelayerClient(
            host="https://relayer-v2.polymarket.com",
            chain_id=137,
            builder_creds=builder_creds
        )
    """

    def __init__(
        self,
        host: str = "https://relayer-v2.polymarket.com",
        chain_id: int = 137,
        builder_creds: Optional[BuilderConfig] = None,
        tx_type: str = "SAFE",
        timeout: int = 60
    ):
        """
        Initialize Relayer client.

        Args:
            host: Relayer API host
            chain_id: Chain ID (137 for Polygon)
            builder_creds: Builder credentials
            tx_type: Transaction type (SAFE or PROXY)
            timeout: Request timeout
        """
        super().__init__(base_url=host, timeout=timeout)
        self.chain_id = chain_id
        self.builder_creds = builder_creds
        self.tx_type = tx_type

    def _build_headers(
        self,
        method: str,
        path: str,
        body: str = ""
    ) -> Dict[str, str]:
        """Build Builder HMAC authentication headers."""
        if not self.builder_creds or not self.builder_creds.is_configured():
            raise AuthenticationError("Builder credentials required for relayer")

        timestamp = str(int(time.time()))

        message = f"{timestamp}{method}{path}{body}"
        signature = hmac.new(
            self.builder_creds.api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        return {
            "POLY_BUILDER_API_KEY": self.builder_creds.api_key,
            "POLY_BUILDER_TIMESTAMP": timestamp,
            "POLY_BUILDER_PASSPHRASE": self.builder_creds.api_passphrase,
            "POLY_BUILDER_SIGNATURE": signature,
        }

    def deploy_safe(self, safe_address: str) -> Dict[str, Any]:
        """
        Deploy a Safe proxy wallet.

        Args:
            safe_address: The Safe address to deploy

        Returns:
            Deployment transaction response
        """
        endpoint = "/deploy"
        body = {"safeAddress": safe_address}
        body_json = json.dumps(body, separators=(',', ':'))
        headers = self._build_headers("POST", endpoint, body_json)

        return self._request(
            "POST",
            endpoint,
            data=body_json,
            headers=headers
        )

    def approve_usdc(
        self,
        safe_address: str,
        spender: str,
        amount: int
    ) -> Dict[str, Any]:
        """
        Approve USDC spending.

        Args:
            safe_address: Safe address
            spender: Spender address
            amount: Amount to approve

        Returns:
            Approval transaction response
        """
        endpoint = "/approve-usdc"
        body = {
            "safeAddress": safe_address,
            "spender": spender,
            "amount": str(amount),
        }
        body_json = json.dumps(body, separators=(',', ':'))
        headers = self._build_headers("POST", endpoint, body_json)

        return self._request(
            "POST",
            endpoint,
            data=body_json,
            headers=headers
        )

    def approve_token(
        self,
        safe_address: str,
        token_id: str,
        spender: str,
        amount: int
    ) -> Dict[str, Any]:
        """
        Approve an ERC-1155 token.

        Args:
            safe_address: Safe address
            token_id: Token ID
            spender: Spender address
            amount: Amount to approve

        Returns:
            Approval transaction response
        """
        endpoint = "/approve-token"
        body = {
            "safeAddress": safe_address,
            "tokenId": token_id,
            "spender": spender,
            "amount": str(amount),
        }
        body_json = json.dumps(body, separators=(',', ':'))
        headers = self._build_headers("POST", endpoint, body_json)

        return self._request(
            "POST",
            endpoint,
            data=body_json,
            headers=headers
        )
