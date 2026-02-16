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
from typing import Optional, Dict, Any
from dataclasses import dataclass
from eth_account import Account
from eth_account.messages import encode_typed_data
from eth_utils import to_checksum_address


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

        if not 0 < self.price <= 1:
            raise ValueError(f"Invalid price: {self.price}")

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

            # Build order message for EIP-712 signing (all ints for uint256)
            order_message = {
                "salt": salt,
                "maker": to_checksum_address(order.maker),
                "signer": self.address,
                "taker": "0x0000000000000000000000000000000000000000",
                "tokenId": int(order.token_id),
                "makerAmount": int(order.maker_amount),
                "takerAmount": int(order.taker_amount),
                "expiration": 0,
                "nonce": 0,
                "feeRateBps": order.fee_rate_bps,
                "side": order.side_value,
                "signatureType": order.signature_type,
            }

            # Use the correct exchange domain for signing
            domain = self.NEG_RISK_ORDER_DOMAIN if neg_risk else self.ORDER_DOMAIN

            # Sign the order using EIP-712 with Exchange domain
            signable = encode_typed_data(
                domain_data=domain,
                message_types=self.ORDER_TYPES,
                message_data=order_message
            )

            signed = self.wallet.sign_message(signable)
            signature = "0x" + signed.signature.hex()

            # Return the full order in CLOB API format
            # (all numeric fields as strings, side as "BUY"/"SELL",
            #  signature inside the order dict)
            return {
                "order": {
                    "salt": salt,
                    "maker": to_checksum_address(order.maker),
                    "signer": self.address,
                    "taker": "0x0000000000000000000000000000000000000000",
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
