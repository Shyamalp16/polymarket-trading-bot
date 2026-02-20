"""
Polymarket Data API Client

Provides access to the public, unauthenticated Polymarket Data API
(data-api.polymarket.com) for auxiliary market intelligence such as
open-interest and trader positions.

Usage:
    from src.data_api_client import DataApiClient

    client = DataApiClient()
    oi = client.get_token_open_interest(token_id)
"""

import logging
from typing import List, Dict, Any

from .http import ThreadLocalSessionMixin

logger = logging.getLogger(__name__)


class DataApiClient(ThreadLocalSessionMixin):
    """
    Unauthenticated client for data-api.polymarket.com.

    All methods are synchronous and thread-safe via the
    ThreadLocalSessionMixin.  Callers inside async contexts should
    run them via ``asyncio.to_thread``.
    """

    BASE_URL = "https://data-api.polymarket.com"

    def __init__(self, timeout: int = 5):
        super().__init__()
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: Dict[str, Any] | None = None) -> List[Dict]:
        """GET ``path`` and return the JSON body as a list.

        Returns an empty list on any error so callers can always iterate
        without extra error handling.
        """
        url = f"{self.BASE_URL}{path}"
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                # Some endpoints wrap the list in a "data" key
                return data.get("data", [data])
            return []
        except Exception as exc:
            logger.debug("DataApiClient GET %s failed: %s", path, exc)
            return []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_token_positions(self, token_id: str) -> List[Dict]:
        """Return all open positions for *token_id*.

        Each position dict from the API typically contains:
          ``size``        — shares held (float)
          ``outcome``     — "Yes" / "No"
          ``avg_price``   — average entry price (float)
        """
        return self._get("/positions", params={"token_id": token_id})

    def get_token_open_interest(self, token_id: str) -> float:
        """Return the total open interest (shares held) for *token_id*.

        Sums the ``size`` field across all positions returned by the
        API.  Returns 0.0 on error so the caller can decide whether to
        fall back to allowing the trade.
        """
        positions = self.get_token_positions(token_id)
        return sum(float(p.get("size", 0)) for p in positions)
