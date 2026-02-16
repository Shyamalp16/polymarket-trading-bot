# Polymarket Trading Bot - Code Audit Report

**Date:** February 2025  
**Scope:** Full codebase review, Polymarket API alignment, loose-end identification  
**Strategies:** Not modified (per audit scope)

---

## Executive Summary

The codebase was audited against the official Polymarket API (py-clob-client, Gamma API, CLOB docs). Several issues were identified and fixed. Remaining items are documented for future consideration.

---

## Fixes Applied

### 1. **client.py – HMAC Body Consistency (Critical)**

**Issue:** `cancel_order`, `cancel_orders`, `cancel_market_orders`, and RelayerClient methods (`deploy_safe`, `approve_usdc`, `approve_token`) built HMAC headers from `body_json` but sent `data=body` (dict) or `data=order_ids` (list). The requests library serializes these differently (e.g. spaces in JSON), so the server’s HMAC verification could fail.

**Fix:** All affected methods now send `data=body_json` (the exact string used for HMAC signing), matching the pattern used in `post_order`.

### 2. **config.py – POLY_SIGNATURE_TYPE Environment Variable**

**Issue:** `signature_type` could not be overridden via environment variables. Users with EOA (0) or Email/Magic (1) wallets had to edit YAML.

**Fix:** Added `POLY_SIGNATURE_TYPE` support in `Config.from_env()` and `Config.load_with_env()`.

---

## API Alignment Verified

| Component | Status | Notes |
|-----------|--------|-------|
| **WebSocket subscribe** | OK | Uses `assets_ids` per Polymarket docs |
| **Gamma API fields** | OK | `acceptingOrders` → `accepting_orders`, `endDate` → `end_date` |
| **CLOB endpoints** | OK | `/book`, `/price`, `/data/orders`, `/data/trades`, `/order`, `/balance-allowance`, `/neg-risk` |
| **Order signing** | OK | EIP-712, CTF Exchange + Neg Risk domains |
| **L2 auth** | OK | HMAC with timestamp + method + path + body |

---

## Potential Issues (Not Fixed)

### 1. **Pagination for get_open_orders and get_trades**

The official py-clob-client paginates with `next_cursor` until `END_CURSOR`. This implementation returns only the first page. For typical usage (few orders/trades) this is usually fine, but high-volume users may miss data.

**Recommendation:** Add pagination when needed.

### 2. **Pre-existing Test Failures**

Five tests fail (unrelated to this audit):

- `test_from_response_success` – fixed: `OrderResult.from_response` now accepts both `orderID` and `orderId`
- `test_sign_order_dict_basic` – expects `result["signature"]` but signer returns `result["order"]["signature"]`
- `test_sign_order_with_nonce` – nonce is string `"0"` in order, not int
- `test_sign_order_with_fee` – `feeRateBps` is string `"100"`, not int
- `test_sign_order_generates_valid_signature` – same structure mismatch

**Recommendation:** Update tests to match current implementation or adjust implementation to match tests.

### 3. **flash_crash.py render_status – TP/SL Display**

Lines 172–173 show TP/SL as `+$0.10` / `-$0.05` even though `take_profit_delta` and `stop_loss_delta` are percentages (e.g. 0.10 = 10%). Display should be `+10%` / `-5%` for clarity. Not changed per “do not change strategies” constraint.

### 4. **fee_rate_bps Default**

`Order` uses `fee_rate_bps=1000` by default. The official client resolves fee rate from the market via `get_fee_rate_bps(token_id)`. Using a fixed default may be incorrect for some markets.

**Recommendation:** Consider querying `GET /fee-rate?token_id=...` and using the market fee when available.

### 5. **Tick Size Validation**

The official py-clob-client validates price against `get_tick_size(token_id)`. This implementation does not. Invalid tick sizes can cause order rejection.

**Recommendation:** Add tick-size validation before placing orders.

---

## Reference Repositories

- [Polymarket/py-clob-client](https://github.com/Polymarket/py-clob-client) – CLOB client
- [Polymarket/py-builder-relayer-client](https://github.com/Polymarket/py-builder-relayer-client) – Relayer client
- [Polymarket/agents](https://github.com/Polymarket/agents) – AI agent framework
- [polymarket-apis (PyPI)](https://pypi.org/project/polymarket-apis/) – Unified API client (Python 3.12+)

---

## File Change Summary

| File | Changes |
|------|---------|
| `src/client.py` | Use `body_json` for cancel_order, cancel_orders, cancel_market_orders, deploy_safe, approve_usdc, approve_token |
| `src/config.py` | Add `POLY_SIGNATURE_TYPE` env support in from_env and load_with_env |
