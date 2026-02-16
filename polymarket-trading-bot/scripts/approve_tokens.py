#!/usr/bin/env python3
"""
One-time setup: approve all tokens needed for trading.

Sets the following approvals via the Polymarket Builder relayer (gasless):
  1. USDC -> CTF Exchange          (for BUY orders)
  2. USDC -> Neg Risk CTF Exchange (for BUY on neg-risk markets)
  3. CTF tokens -> CTF Exchange          (for SELL orders)
  4. CTF tokens -> Neg Risk CTF Exchange (for SELL on neg-risk markets)

Usage:
    python scripts/approve_tokens.py
"""

import os
import sys
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from py_builder_relayer_client.client import RelayClient
from py_builder_relayer_client.models import SafeTransaction, OperationType
from py_builder_signing_sdk.config import BuilderConfig
from py_builder_signing_sdk.sdk_types import BuilderApiKeyCreds

# ---------------------------------------------------------------------------
# Contract addresses (Polygon mainnet)
# ---------------------------------------------------------------------------
USDC = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
CTF = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
NEG_RISK_CTF_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
NEG_RISK_ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"

MAX_UINT256 = "0x" + "f" * 64  # 2^256 - 1


def encode_erc20_approve(spender: str, amount: str) -> str:
    """Encode ERC-20 approve(address,uint256) call."""
    # Function selector: keccak256("approve(address,uint256)")[:4]
    selector = "095ea7b3"
    spender_padded = spender.lower().replace("0x", "").zfill(64)
    amount_padded = amount.replace("0x", "").zfill(64)
    return "0x" + selector + spender_padded + amount_padded


def encode_erc1155_set_approval_for_all(operator: str, approved: bool) -> str:
    """Encode ERC-1155 setApprovalForAll(address,bool) call."""
    # Function selector: keccak256("setApprovalForAll(address,bool)")[:4]
    selector = "a22cb465"
    operator_padded = operator.lower().replace("0x", "").zfill(64)
    approved_padded = ("1" if approved else "0").zfill(64)
    return "0x" + selector + operator_padded + approved_padded


def main():
    # Load credentials from environment
    private_key = os.environ.get("POLY_PRIVATE_KEY")
    if not private_key:
        print("ERROR: POLY_PRIVATE_KEY not set in environment / .env")
        sys.exit(1)

    api_key = os.environ.get("POLY_BUILDER_API_KEY", "")
    api_secret = os.environ.get("POLY_BUILDER_API_SECRET", "")
    api_passphrase = os.environ.get("POLY_BUILDER_API_PASSPHRASE", "")

    if not all([api_key, api_secret, api_passphrase]):
        # Try loading from config.yaml
        try:
            import yaml
            with open("config.yaml") as f:
                cfg = yaml.safe_load(f)
            builder = cfg.get("builder", {})
            api_key = api_key or builder.get("api_key", "")
            api_secret = api_secret or builder.get("api_secret", "")
            api_passphrase = api_passphrase or builder.get("api_passphrase", "")
        except Exception:
            pass

    if not all([api_key, api_secret, api_passphrase]):
        print("ERROR: Builder API credentials not found in env or config.yaml")
        sys.exit(1)

    local_creds = BuilderApiKeyCreds(
        key=api_key,
        secret=api_secret,
        passphrase=api_passphrase,
    )
    builder_config = BuilderConfig(local_builder_creds=local_creds)

    print("Setting up Polymarket trading approvals...")
    print(f"  Relayer: https://relayer-v2.polymarket.com")

    client = RelayClient(
        relayer_url="https://relayer-v2.polymarket.com",
        chain_id=137,
        private_key=private_key,
        builder_config=builder_config,
    )

    safe_address = client.get_expected_safe()
    print(f"  Safe:    {safe_address}")

    # Check if Safe is deployed
    if not client.get_deployed(safe_address):
        print("\n  Safe not deployed yet. Deploying...")
        resp = client.deploy()
        result = resp.wait()
        if result:
            print(f"  Safe deployed! TX: {result.get('transactionHash', 'N/A')}")
        else:
            print("  ERROR: Safe deployment failed")
            sys.exit(1)

    # Build all approval transactions
    transactions = []

    # 1. USDC approvals (ERC-20 approve)
    usdc_spenders = [
        (CTF, "CTF"),
        (NEG_RISK_ADAPTER, "Neg Risk Adapter"),
        (CTF_EXCHANGE, "CTF Exchange"),
        (NEG_RISK_CTF_EXCHANGE, "Neg Risk Exchange"),
    ]
    for spender, name in usdc_spenders:
        transactions.append(SafeTransaction(
            to=USDC,
            operation=OperationType.Call,
            data=encode_erc20_approve(spender, MAX_UINT256),
            value="0",
        ))
        print(f"  + USDC approve -> {name}")

    # 2. CTF token approvals (ERC-1155 setApprovalForAll)
    token_spenders = [
        (CTF_EXCHANGE, "CTF Exchange"),
        (NEG_RISK_CTF_EXCHANGE, "Neg Risk Exchange"),
        (NEG_RISK_ADAPTER, "Neg Risk Adapter"),
    ]
    for operator, name in token_spenders:
        transactions.append(SafeTransaction(
            to=CTF,
            operation=OperationType.Call,
            data=encode_erc1155_set_approval_for_all(operator, True),
            value="0",
        ))
        print(f"  + CTF setApprovalForAll -> {name}")

    # Execute all approvals in a single batched transaction
    print(f"\nSubmitting {len(transactions)} approvals via relayer...")
    try:
        # Build Safe transaction request using the SDK internals
        from py_builder_relayer_client.builder.safe import build_safe_transaction_request
        from py_builder_relayer_client.models import SafeTransactionArgs, TransactionType

        from_address = client.signer.address()
        nonce_payload = client.get_nonce(from_address, TransactionType.SAFE.value)
        nonce = nonce_payload.get("nonce", 0) if nonce_payload else 0

        safe_args = SafeTransactionArgs(
            from_address=from_address,
            nonce=nonce,
            chain_id=137,
            transactions=transactions,
        )

        txn_request = build_safe_transaction_request(
            signer=client.signer,
            args=safe_args,
            config=client.contract_config,
            metadata="Approve all tokens for trading",
        ).to_dict()

        # Submit with proper JSON HMAC (bypass buggy str(dict) serialization)
        import json, hmac as hmac_mod, hashlib, base64, time as time_mod, requests
        body_json = json.dumps(txn_request, separators=(',', ':'))
        ts = str(int(time_mod.time()))
        method = "POST"
        path = "/submit"
        msg = ts + method + path + body_json
        base64_secret = base64.urlsafe_b64decode(api_secret)
        sig = base64.urlsafe_b64encode(
            hmac_mod.new(base64_secret, msg.encode(), hashlib.sha256).digest()
        ).decode()
        headers = {
            "POLY_BUILDER_API_KEY": api_key,
            "POLY_BUILDER_TIMESTAMP": ts,
            "POLY_BUILDER_PASSPHRASE": api_passphrase,
            "POLY_BUILDER_SIGNATURE": sig,
            "Content-Type": "application/json",
        }
        r = requests.post(
            "https://relayer-v2.polymarket.com/submit",
            data=body_json,
            headers=headers,
        )
        print(f"  Status: {r.status_code}")
        print(f"  Response: {r.text}")

        if r.status_code == 200:
            resp_data = r.json()
            tx_id = resp_data.get("transactionID", "")
            print(f"  Transaction ID: {tx_id}")
            print("  Waiting for confirmation...")
            # Poll for completion
            for i in range(15):
                time_mod.sleep(3)
                txn = client.get_transaction(tx_id)
                if txn:
                    t = txn[0] if isinstance(txn, list) else txn
                    state = t.get("state", "")
                    print(f"  State: {state}")
                    if state in ("STATE_CONFIRMED", "STATE_MINED"):
                        tx_hash = t.get("transactionHash", "N/A")
                        print(f"\n  All approvals confirmed!")
                        print(f"  TX: https://polygonscan.com/tx/{tx_hash}")
                        break
                    if state in ("STATE_FAILED", "STATE_INVALID"):
                        print(f"\n  Transaction failed: {t}")
                        break
        else:
            print(f"\n  ERROR: Relayer returned {r.status_code}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n  ERROR: {e}")
        sys.exit(1)

    print("\nDone! You can now run the trading bot.")


if __name__ == "__main__":
    main()
