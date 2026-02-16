"""
Trade Telemetry - persistent trade/event logging.

Stores structured telemetry to both JSONL and CSV so strategy behavior can be
analyzed and thresholds tuned from real executions.
"""

from __future__ import annotations

import csv
import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{os.getpid()}"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=_json_default)
    return str(value)


@dataclass
class TradeTelemetry:
    """
    Lightweight file logger for strategy telemetry.
    """

    enabled: bool = True
    base_dir: str = "logs/trades"
    strategy_name: str = ""
    coin: str = ""
    market_duration_min: int = 0
    session_id: str = field(default_factory=_session_id)

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _jsonl_path: Optional[Path] = field(default=None, init=False)
    _csv_path: Optional[Path] = field(default=None, init=False)
    _csv_initialized: bool = field(default=False, init=False)

    _CSV_FIELDS = [
        "ts_usc",
        "session",
        "strat",
        "coin",
        "market_duration_min",
        "event_type",
        "size",
        "entry_price",
        "exit_price",
        "size_shares",
        "hold_seconds",
    ]

    def _ensure_paths(self) -> None:
        if not self.enabled:
            return
        if self._jsonl_path and self._csv_path:
            return
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        base = Path(self.base_dir)
        base.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = base / f"trade_events_{day}.jsonl"
        self._csv_path = base / f"trade_events_{day}.csv"
        self._csv_initialized = self._csv_path.exists() and self._csv_path.stat().st_size > 0

    def log_event(self, payload: Dict[str, Any]) -> None:
        """
        Persist one telemetry event to JSONL + CSV.
        """
        if not self.enabled:
            return
        self._ensure_paths()
        assert self._jsonl_path is not None
        assert self._csv_path is not None

        # Keep telemetry intentionally minimal and stable.
        event: Dict[str, Any] = {
            "ts_usc": _utc_iso(),
            "session": self.session_id,
            "strat": self.strategy_name,
            "coin": self.coin,
            "market_duration_min": self.market_duration_min,
            "event_type": payload.get("event_type", ""),
            "size": payload.get("size", ""),
            "entry_price": payload.get("entry_price", ""),
            "exit_price": payload.get("exit_price", ""),
            "size_shares": payload.get("size_shares", ""),
            "hold_seconds": payload.get("hold_seconds", ""),
        }

        with self._lock:
            with self._jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, sort_keys=True, default=_json_default))
                f.write("\n")

            row = {k: "" for k in self._CSV_FIELDS}
            for key in row.keys():
                if key in event:
                    row[key] = _stringify(event.get(key))

            with self._csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._CSV_FIELDS)
                if not self._csv_initialized:
                    writer.writeheader()
                    self._csv_initialized = True
                writer.writerow(row)

