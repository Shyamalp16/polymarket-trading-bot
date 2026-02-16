"""
Signer Module - EIP-712 Order Signing

Provides EIP-712 signature functionality for Polymarket orders
and authentication messages.

EIP-712 is a standard for structured data hashing and signing
that provides better security and user experience than plain
message signing.

Example:
    from src.signer import OrderSigner

    signer = OrderSigner(private_key)
    signature = signer.sign_order(
        token_id="123...",
        price=0.65,
        size=10,
        side="BUY",
        maker="0x..."
    )
"""

import time
import random
import json
import os
import queue
import subprocess
import threading
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from eth_account import Account
from eth_account.messages import encode_typed_data
from eth_utils import to_checksum_address

from lib.latency_metrics import record_latency

logger = logging.getLogger(__name__)


# USDC has 6 decimal places
USDC_DECIMALS = 6


@dataclass
class Order:
    """
    Represents a Polymarket order.

    Attributes:
        token_id: The ERC-1155 token ID for the market outcome
        price: Price per share (0-1, e.g., 0.65 = 65%)
        size: Number of shares
        side: Order side ('BUY' or 'SELL')
        maker: The maker's wallet address (Safe/Proxy)
        nonce: Unique order nonce (usually timestamp)
        fee_rate_bps: Fee rate in basis points (usually 0)
        signature_type: Signature type (2 = Gnosis Safe)
    """
    token_id: str
    price: float
    size: float
    side: str
    maker: str
    nonce: Optional[int] = None
    fee_rate_bps: int = 1000
    signature_type: int = 2
    order_type: str = "GTC"  # GTC, FOK, GTD, FAK

    def __post_init__(self):
        """Validate and normalize order parameters."""
        self.side = self.side.upper()
        if self.side not in ("BUY", "SELL"):
            raise ValueError(f"Invalid side: {self.side}")

        if not 0 < self.price <= 0.99:
            raise ValueError(f"Invalid price: {self.price} (SDK max is 0.99)")

        if self.size <= 0:
            raise ValueError(f"Invalid size: {self.size}")

        if self.nonce is None:
            self.nonce = int(time.time())

        is_market = self.order_type in ("FOK", "FAK")

        if is_market:
            # FOK/FAK (market) orders precision rules (same for BUY & SELL):
            #   makerAmount: max 2 decimal accuracy → divisible by 10,000
            #   takerAmount: max 4 decimal accuracy → divisible by 100
            # The price acts as a limit (worst acceptable), not exact placement.
            if self.side == "BUY":
                # maker = USDC (2dp), taker = shares (4dp)
                usdc_amount = round(self.price * self.size * 100) * 10000
                share_amount = round(self.size * 10000) * 100
            else:
                # maker = shares (2dp), taker = USDC (4dp)
                share_amount = round(self.size * 100) * 10000
                usdc_amount = round(self.price * self.size * 10000) * 100
        else:
            # GTC/GTD (limit) orders: price must be on 0.01 tick.
            #   BUY:  makerAmount (USDC) → divisible by 100
            #         takerAmount (shares) → divisible by 10,000
            #   SELL: makerAmount (shares) → divisible by 10,000
            #         takerAmount (USDC) → divisible by 100
            price_cents = round(self.price * 100)
            size_hundredths = round(self.size * 100)
            usdc_amount = price_cents * size_hundredths * 100   # divisible by 100
            share_amount = size_hundredths * 10000              # divisible by 10,000

        if self.side == "BUY":
            self.maker_amount = str(usdc_amount)
            self.taker_amount = str(share_amount)
        else:
            self.maker_amount = str(share_amount)
            self.taker_amount = str(usdc_amount)

        self.side_value = 0 if self.side == "BUY" else 1


class SignerError(Exception):
    """Base exception for signer operations."""
    pass


class NodeSignerBridge:
    """Persistent Node subprocess bridge for EIP-712 order signing."""

    def __init__(
        self,
        private_key: str,
        chain_id: int,
        enabled: bool = True,
        timeout_sec: float = 1.5,
        script_path: Optional[str] = None,
    ) -> None:
        self.enabled = enabled
        self.chain_id = int(chain_id)
        self.timeout_sec = max(0.1, float(timeout_sec))
        self.script_path = (
            Path(script_path)
            if script_path
            else Path(__file__).resolve().parent.parent / "scripts" / "js_order_signer.mjs"
        )
        self._private_key = private_key if private_key.startswith("0x") else f"0x{private_key}"
        self._proc: Optional[subprocess.Popen[str]] = None
        self._io_lock = threading.Lock()
        self._rpc_id = 0
        self._disabled_reason: str = ""

    def _readline_with_timeout(self) -> str:
        if not self._proc or not self._proc.stdout:
            raise SignerError("JS signer stdout unavailable")
        q: "queue.Queue[str]" = queue.Queue(maxsize=1)

        def _reader() -> None:
            try:
                line = self._proc.stdout.readline()
            except Exception:
                line = ""
            try:
                q.put_nowait(line)
            except Exception:
                pass

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        try:
            return q.get(timeout=self.timeout_sec)
        except queue.Empty as exc:
            raise SignerError("JS signer timed out") from exc

    def _spawn(self) -> None:
        if not self.enabled:
            raise SignerError("JS signer disabled")
        if self._proc and self._proc.poll() is None:
            return
        if not self.script_path.exists():
            raise SignerError(f"JS signer script not found: {self.script_path}")

        env = os.environ.copy()
        env["POLY_SIGNER_PRIVATE_KEY"] = self._private_key
        env["POLY_SIGNER_CHAIN_ID"] = str(self.chain_id)
        self._proc = subprocess.Popen(
            ["node", str(self.script_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=env,
        )
        self.health_check()

    def _rpc(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self._spawn()
        if not self._proc or not self._proc.stdin or not self._proc.stdout:
            raise SignerError("JS signer process unavailable")
        with self._io_lock:
            self._rpc_id += 1
            req = {"id": self._rpc_id, "method": method, "params": params}
            self._proc.stdin.write(json.dumps(req) + "\n")
            self._proc.stdin.flush()
            line = self._readline_with_timeout().strip()
        if not line:
            raise SignerError("JS signer returned empty response")
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SignerError(f"Invalid JS signer response: {line[:200]}") from exc
        if msg.get("error"):
            raise SignerError(str(msg.get("error")))
        return msg

    def health_check(self) -> bool:
        resp = self._rpc("health", {})
        return bool(resp.get("ok", False))

    def sign_order(self, payload: Dict[str, Any]) -> str:
        resp = self._rpc("sign_order", payload)
        sig = str(resp.get("signature", ""))
        if not sig.startswith("0x"):
            raise SignerError("JS signer did not return hex signature")
        return sig

    def mark_disabled(self, reason: str) -> None:
        self.enabled = False
        self._disabled_reason = reason
        logger.warning("JS signer disabled: %s", reason)
        self.close()

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if not proc:
            return
        try:
            proc.terminate()
            proc.wait(timeout=0.5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


class OrderSigner:
    """
    Signs Polymarket orders using EIP-712.

    This signer handles:
    - Authentication messages (L1)
    - Order messages (for CLOB submission)

    Attributes:
        wallet: The Ethereum wallet instance
        address: The signer's address
        domain: EIP-712 domain separator
    """

    # EIP-712 domain for L1 authentication (API key create/derive)
    AUTH_DOMAIN = {
        "name": "ClobAuthDomain",
        "version": "1",
        "chainId": 137,  # Polygon mainnet
    }

    # EIP-712 domain for order signing (CTF Exchange contract)
    # https://polygonscan.com/address/0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E
    CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
    NEG_RISK_CTF_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"

    ORDER_DOMAIN = {
        "name": "Polymarket CTF Exchange",
        "version": "1",
        "chainId": 137,
        "verifyingContract": CTF_EXCHANGE,
    }

    NEG_RISK_ORDER_DOMAIN = {
        "name": "Polymarket CTF Exchange",
        "version": "1",
        "chainId": 137,
        "verifyingContract": NEG_RISK_CTF_EXCHANGE,
    }

    # Order type definition for EIP-712
    ORDER_TYPES = {
        "Order": [
            {"name": "salt", "type": "uint256"},
            {"name": "maker", "type": "address"},
            {"name": "signer", "type": "address"},
            {"name": "taker", "type": "address"},
            {"name": "tokenId", "type": "uint256"},
            {"name": "makerAmount", "type": "uint256"},
            {"name": "takerAmount", "type": "uint256"},
            {"name": "expiration", "type": "uint256"},
            {"name": "nonce", "type": "uint256"},
            {"name": "feeRateBps", "type": "uint256"},
            {"name": "side", "type": "uint8"},
            {"name": "signatureType", "type": "uint8"},
        ]
    }

    ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

    def __init__(self, private_key: str):
        """
        Initialize signer with a private key.

        Args:
            private_key: Private key (with or without 0x prefix)

        Raises:
            ValueError: If private key is invalid
        """
        if private_key.startswith("0x"):
            private_key = private_key[2:]

        try:
            self.wallet = Account.from_key(f"0x{private_key}")
        except Exception as e:
            raise ValueError(f"Invalid private key: {e}")

        self.address = self.wallet.address
        self._chain_id = int(self.ORDER_DOMAIN["chainId"])
        self._maker_checksum_cache: Dict[str, str] = {}
        self._order_payload_cache: Dict[bool, Dict[str, Any]] = {
            False: {"domain_data": self.ORDER_DOMAIN, "message_types": self.ORDER_TYPES},
            True: {"domain_data": self.NEG_RISK_ORDER_DOMAIN, "message_types": self.ORDER_TYPES},
        }
        js_enabled = os.environ.get("POLY_JS_SIGNER_ENABLED", "1").lower() not in {"0", "false", "no"}
        js_timeout = float(os.environ.get("POLY_JS_SIGNER_TIMEOUT_SEC", "1.5"))
        js_script = os.environ.get("POLY_JS_SIGNER_SCRIPT", "")
        self._js_bridge = NodeSignerBridge(
            private_key=f"0x{private_key}",
            chain_id=self._chain_id,
            enabled=js_enabled,
            timeout_sec=js_timeout,
            script_path=js_script or None,
        )

    def _checksum_maker(self, maker: str) -> str:
        if maker in self._maker_checksum_cache:
            return self._maker_checksum_cache[maker]
        checksum = to_checksum_address(maker)
        self._maker_checksum_cache[maker] = checksum
        return checksum

    def _order_message(self, order: Order, salt: int) -> Dict[str, Any]:
        return {
            "salt": int(salt),
            "maker": self._checksum_maker(order.maker),
            "signer": self.address,
            "taker": self.ZERO_ADDRESS,
            "tokenId": int(order.token_id),
            "makerAmount": int(order.maker_amount),
            "takerAmount": int(order.taker_amount),
            "expiration": 0,
            "nonce": 0,
            "feeRateBps": int(order.fee_rate_bps),
            "side": int(order.side_value),
            "signatureType": int(order.signature_type),
        }

    def _sign_order_python(
        self,
        order_message: Dict[str, Any],
        neg_risk: bool,
    ) -> str:
        signable = encode_typed_data(
            domain_data=self._order_payload_cache[neg_risk]["domain_data"],
            message_types=self._order_payload_cache[neg_risk]["message_types"],
            message_data=order_message,
        )
        signed = self.wallet.sign_message(signable)
        return "0x" + signed.signature.hex()

    def _sign_order_js(
        self,
        order_message: Dict[str, Any],
        neg_risk: bool,
    ) -> str:
        payload = {
            "negRisk": bool(neg_risk),
            "message": {
                "salt": str(order_message["salt"]),
                "maker": str(order_message["maker"]),
                "signer": str(order_message["signer"]),
                "taker": str(order_message["taker"]),
                "tokenId": str(order_message["tokenId"]),
                "makerAmount": str(order_message["makerAmount"]),
                "takerAmount": str(order_message["takerAmount"]),
                "expiration": str(order_message["expiration"]),
                "nonce": str(order_message["nonce"]),
                "feeRateBps": str(order_message["feeRateBps"]),
                "side": int(order_message["side"]),
                "signatureType": int(order_message["signatureType"]),
            },
        }
        return self._js_bridge.sign_order(payload)

    @classmethod
    def from_encrypted(
        cls,
        encrypted_data: dict,
        password: str
    ) -> "OrderSigner":
        """
        Create signer from encrypted private key.

        Args:
            encrypted_data: Encrypted key data
            password: Decryption password

        Returns:
            Configured OrderSigner instance

        Raises:
            InvalidPasswordError: If password is incorrect
        """
        from .crypto import KeyManager, InvalidPasswordError

        manager = KeyManager()
        private_key = manager.decrypt(encrypted_data, password)
        return cls(private_key)

    def sign_auth_message(
        self,
        timestamp: Optional[str] = None,
        nonce: int = 0
    ) -> str:
        """
        Sign an authentication message for L1 authentication.

        This signature is used to create or derive API credentials.

        Args:
            timestamp: Message timestamp (defaults to current time)
            nonce: Message nonce (usually 0)

        Returns:
            Hex-encoded signature
        """
        if timestamp is None:
            timestamp = str(int(time.time()))

        # Auth message types
        auth_types = {
            "ClobAuth": [
                {"name": "address", "type": "address"},
                {"name": "timestamp", "type": "string"},
                {"name": "nonce", "type": "uint256"},
                {"name": "message", "type": "string"},
            ]
        }

        message_data = {
            "address": self.address,
            "timestamp": timestamp,
            "nonce": nonce,
            "message": "This message attests that I control the given wallet",
        }

        signable = encode_typed_data(
            domain_data=self.AUTH_DOMAIN,
            message_types=auth_types,
            message_data=message_data
        )

        signed = self.wallet.sign_message(signable)
        return "0x" + signed.signature.hex()

    def sign_order(self, order: Order, neg_risk: bool = False) -> Dict[str, Any]:
        """
        Sign a Polymarket order.

        Args:
            order: Order instance to sign
            neg_risk: If True, use the Neg Risk CTF Exchange domain

        Returns:
            Dictionary containing order (with signature inside) matching
            the CLOB API payload format.

        Raises:
            SignerError: If signing fails
        """
        try:
            salt = random.randint(1, 2**31)
            order_message = self._order_message(order, salt)
            signature = ""
            started = time.perf_counter()
            use_js = self._js_bridge.enabled
            if use_js:
                try:
                    signature = self._sign_order_js(order_message, neg_risk=neg_risk)
                except Exception as js_exc:
                    logger.warning("JS signer failed, using Python fallback: %s", js_exc)
                    self._js_bridge.mark_disabled(str(js_exc))
            if not signature:
                signature = self._sign_order_python(order_message, neg_risk=neg_risk)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            record_latency(
                "sign_order_ms",
                elapsed_ms,
                {
                    "backend": "js" if use_js and self._js_bridge.enabled else "python",
                    "neg_risk": bool(neg_risk),
                    "order_type": order.order_type,
                },
            )

            # Return the full order in CLOB API format
            # (all numeric fields as strings, side as "BUY"/"SELL",
            #  signature inside the order dict)
            return {
                "order": {
                    "salt": salt,
                    "maker": order_message["maker"],
                    "signer": self.address,
                    "taker": self.ZERO_ADDRESS,
                    "tokenId": order.token_id,
                    "makerAmount": str(int(order.maker_amount)),
                    "takerAmount": str(int(order.taker_amount)),
                    "expiration": "0",
                    "nonce": "0",
                    "feeRateBps": str(order.fee_rate_bps),
                    "side": order.side,
                    "signatureType": order.signature_type,
                    "signature": signature,
                },
            }

        except Exception as e:
            raise SignerError(f"Failed to sign order: {e}")

    def sign_order_dict(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str,
        maker: str,
        nonce: Optional[int] = None,
        fee_rate_bps: int = 0
    ) -> Dict[str, Any]:
        """
        Sign an order from dictionary parameters.

        Args:
            token_id: Market token ID
            price: Price per share
            size: Number of shares
            side: 'BUY' or 'SELL'
            maker: Maker's wallet address
            nonce: Order nonce (defaults to timestamp)
            fee_rate_bps: Fee rate in basis points

        Returns:
            Dictionary containing order and signature
        """
        order = Order(
            token_id=token_id,
            price=price,
            size=size,
            side=side,
            maker=maker,
            nonce=nonce,
            fee_rate_bps=fee_rate_bps,
        )
        return self.sign_order(order)

    def sign_message(self, message: str) -> str:
        """
        Sign a plain text message (for API key derivation).

        Args:
            message: Plain text message to sign

        Returns:
            Hex-encoded signature
        """
        from eth_account.messages import encode_defunct

        signable = encode_defunct(text=message)
        signed = self.wallet.sign_message(signable)
        return "0x" + signed.signature.hex()


# Alias for backwards compatibility
WalletSigner = OrderSigner
