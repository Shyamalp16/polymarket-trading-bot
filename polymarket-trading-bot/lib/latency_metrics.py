"""
Lightweight JSONL latency metrics recorder.

Enable with POLY_LATENCY_METRICS=1.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

_LOCK = threading.Lock()
_SESSION = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
_BASE_DIR = Path("logs/latency")
_FILE_PATH = _BASE_DIR / f"latency_{_SESSION}.jsonl"


def _enabled() -> bool:
    return os.environ.get("POLY_LATENCY_METRICS", "0").lower() in {"1", "true", "yes", "on"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def record_latency(stage: str, elapsed_ms: float, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Record one latency datapoint to JSONL."""
    if not _enabled():
        return
    payload: Dict[str, Any] = {
        "ts_utc": _utc_now(),
        "session": _SESSION,
        "stage": str(stage),
        "elapsed_ms": round(float(elapsed_ms), 4),
    }
    if metadata:
        payload["meta"] = metadata
    line = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    try:
        with _LOCK:
            _BASE_DIR.mkdir(parents=True, exist_ok=True)
            with _FILE_PATH.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception:
        # Metrics should never affect trading path.
        return

