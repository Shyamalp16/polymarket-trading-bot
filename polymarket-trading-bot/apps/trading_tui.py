"""
Trading TUI — Real-time Signal & PnL Monitor for the Dual-Bot System

Renders live in-place after startup completes.
Setup logs print normally; the TUI takes over once trading begins.

Enable with:
    python -m bots.run_dual_bot --bankroll 100 --coin BTC --tui
"""

from __future__ import annotations

import logging
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Deque, TYPE_CHECKING

if TYPE_CHECKING:
    from bots.momentum_bot import MomentumBot
    from bots.mean_reversion_bot import MeanReversionBot
    from bots.coordinator import Coordinator
    from lib.shared_state import SharedState
    from lib.btc_price import BTCPriceTracker

# ── ANSI palette ──────────────────────────────────────────────────────────────

class C:
    RST = "\033[0m"
    B   = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GRN = "\033[92m"
    YLW = "\033[93m"
    BLU = "\033[94m"
    MAG = "\033[95m"
    CYN = "\033[96m"
    WHT = "\033[97m"


def _c(code: str, s: str) -> str:
    return f"{code}{s}{C.RST}"


def _pnl(v: float) -> str:
    col = C.GRN if v >= 0 else C.RED
    return _c(col, f"{'+'if v>=0 else ''}{v:.2f}")


def _pct(v: float, d: int = 3) -> str:
    col = C.GRN if v >= 0 else C.RED
    return _c(col, f"{'+'if v>=0 else ''}{v*100:.{d}f}%")


def _ts_now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _elapsed(t: float) -> str:
    s = int(time.time() - t)
    return f"{s}s" if s < 60 else f"{s//60}m{s%60:02d}s"


def _vis(s: str) -> str:
    """Strip ANSI codes to get visible length."""
    return re.sub(r"\033\[[0-9;]*m", "", s)


def _pad(s: str, width: int) -> str:
    """Right-pad a (possibly ANSI-colored) string to visible width."""
    return s + " " * max(0, width - len(_vis(s)))


def _countdown(secs: int) -> str:
    if secs <= 0:
        return _c(C.RED, "ENDED")
    m, s = divmod(secs, 60)
    col = C.RED if secs < 60 else (C.YLW if secs < 180 else C.GRN)
    return _c(col, f"{m:02d}:{s:02d}")


def _regime(r: str) -> str:
    col = {"high": C.RED, "normal": C.GRN, "low": C.BLU}.get(r.lower(), C.WHT)
    return _c(col, r.upper())


# ── Event model ───────────────────────────────────────────────────────────────

@dataclass
class TEvent:
    ts: float = field(default_factory=time.time)
    etype: str = ""      # SIGNAL ENTRY EXIT MARKET MAKER ESCALATE RATCHET SYSTEM
    bot: str   = ""      # MOM MR SYS
    side: str  = ""      # UP DOWN --
    detail: str = ""
    pnl: Optional[float] = None


# ── Log-message patterns ──────────────────────────────────────────────────────
#
# Bots emit structured log lines prefixed with "==>" so we can parse them
# without touching the bot logic.  Any line not matching a pattern is ignored.

_PATS: list[tuple] = [
    # ==> MOMENTUM ENTERED: UP | $10.00 @ 0.420 | TP1: 0.484 TP2: 0.543 SL: 0.273
    (re.compile(
        r"==> MOMENTUM ENTERED: (\w+) \| \$([\d.]+) @ ([\d.]+)"
        r" \| TP1: ([\d.]+) TP2: ([\d.]+) SL: ([\d.]+)"
     ), "MOM", "ENTRY"),

    # ==> MR ENTERED: DOWN | $10.00 @ 0.460 | TP1: 0.567 TP2: 0.674 SL: 0.276
    (re.compile(
        r"==> MR ENTERED: (\w+) \| \$([\d.]+) @ ([\d.]+)"
        r" \| TP1: ([\d.]+) TP2: ([\d.]+) SL: ([\d.]+)"
     ), "MR", "ENTRY"),

    # ==> MOMENTUM CLOSED: TP1 | 3.50sh @ 0.4840 | PnL: +$0.22
    (re.compile(
        r"==> MOMENTUM CLOSED: (\w+) \| ([\d.]+)sh @ ([\d.]+) \| PnL: ([+\-\$\d.]+)"
     ), "MOM", "EXIT"),

    # ==> MR CLOSED: SL | 10.00sh @ 0.2910 | PnL: -$1.12
    (re.compile(
        r"==> MR CLOSED: (\w+) \| ([\d.]+)sh @ ([\d.]+) \| PnL: ([+\-\$\d.]+)"
     ), "MR", "EXIT"),

    # ==> MOMENTUM SIGNAL: UP | conf=85% | impulse 0.20%
    (re.compile(
        r"==> MOMENTUM SIGNAL: (\w+) \| conf=([\d.]+%) \| (.+)"
     ), "MOM", "SIGNAL"),

    # ==> MR SIGNAL: DOWN | conf=42% | drop 11.2% z=2.73 ob=0.62
    (re.compile(
        r"==> MR SIGNAL: (\w+) \| conf=([\d.]+%) \| (.+)"
     ), "MR", "SIGNAL"),

    # === NEW MARKET: slug → slug ===
    (re.compile(r"=== NEW MARKET: (.+) ==="), "SYS", "MARKET"),

    # Coordinator: entering late window (45s remaining, 15.0% of window)
    (re.compile(r"Coordinator: entering late window \((.+)\)"), "SYS", "SYSTEM"),

    # Ratchet / trailing SL notifications
    (re.compile(r"Momentum: TP1 done"), "MOM", "RATCHET"),
    (re.compile(r"Momentum: TP2 done"), "MOM", "RATCHET"),
    (re.compile(r"MR: TP1 done"),       "MR",  "RATCHET"),
    (re.compile(r"MR: TP2 done"),       "MR",  "RATCHET"),
    # SL ratcheted (trailing SL update)
    (re.compile(
        r"(?:Momentum|MR): SL ratcheted ([\d.]+) → ([\d.]+)(\s*\[TRAIL\])?"
     ), "BOT", "RATCHET"),

    # ==> MR MAKER POSTED: DOWN | 10.00 shares @ 0.445
    (re.compile(
        r"==> MR MAKER POSTED: (\w+) \| ([\d.]+) shares @ ([\d.]+)"
     ), "MR", "MAKER"),

    # MR escalating to taker
    (re.compile(r"MR: escalating to taker"), "MR", "ESCALATE"),

    # ==> MR ENTRY FAILED: FOK rejected (attempt 1/3) — INVALID_ORDER...
    (re.compile(r"==> MR ENTRY FAILED: (.+)"), "MR", "ERROR"),

    # ==> MR MAKER REJECTED: post_only crossed spread — ...
    (re.compile(r"==> MR MAKER REJECTED: (.+)"), "MR", "ERROR"),

    # ==> MOM ENTRY FAILED: FAK rejected (attempt 1/2) — ...
    (re.compile(r"==> MOM ENTRY FAILED: (.+)"), "MOM", "ERROR"),

    # Order rejected by CLOB: BUY 5.00@0.0950 type=FOK — INVALID_ORDER_...
    (re.compile(
        r"Order rejected by CLOB: (\w+) ([\d.]+)@([\d.]+) type=(\w+) — (.+)"
     ), "BOT", "ERROR"),

    # ==> MR CLOSE FAILED: SL | attempt 2 | reason: no token_id ...
    (re.compile(
        r"==> MR CLOSE FAILED: (\w+) \| attempt (\d+) \| reason: (.+)"
     ), "MR", "ERROR"),

    # ==> MR CLOSE PARTIAL: SL | sold 2.00/5.00 sh | PnL: -$0.60 | remaining 3.00
    (re.compile(
        r"==> MR CLOSE PARTIAL: (\w+) \| sold ([\d.]+)/([\d.]+) sh \| PnL: ([+\-\$\d.]+) \| remaining ([\d.]+)"
     ), "MR", "ERROR"),

    # ==> MOMENTUM CLOSE FAILED: SL | attempt 2 | reason: ...
    (re.compile(
        r"==> MOMENTUM CLOSE FAILED: (\w+) \| attempt (\d+) \| reason: (.+)"
     ), "MOM", "ERROR"),

    # ==> SPREAD POSTED: UP @ 0.480  DOWN @ 0.480 | 5 sh/leg | cost=4.80  locked=+0.20
    (re.compile(
        r"==> SPREAD POSTED: UP @ ([\d.]+)\s+DOWN @ ([\d.]+) \| (\d+) sh/leg"
     ), "SPR", "ENTRY"),

    # ==> SPREAD FILLED: UP @ 0.480  DOWN @ 0.480 | cost $4.80 | locked +$0.20
    # Both legs filled = spread locked, NOT an exit.  Use "LOCKED" etype.
    (re.compile(
        r"==> SPREAD FILLED: UP @ ([\d.]+)\s+DOWN @ ([\d.]+) \| cost \$([\d.]+) \| locked \+\$([\d.]+)"
     ), "SPR", "LOCKED"),

    # ==> SPREAD RELEASED: UP 0.480 + DOWN 0.480 | locked +$0.20 | resolves on-chain…
    (re.compile(
        r"==> SPREAD RELEASED: UP ([\d.]+) \+ DOWN ([\d.]+) \| locked \+\$([\d.]+)"
     ), "SPR", "LOCKED"),

    # ==> SPREAD SINGLE LEG: UP @ 0.480 | 15 sh | SL 0.408 | TP1 0.552 | TP2 0.600
    (re.compile(
        r"==> SPREAD SINGLE LEG: (\w+) @ ([\d.]+) \| ([\d.]+) sh \| SL ([\d.]+) \| TP1 ([\d.]+) \| TP2 ([\d.]+)"
     ), "SPR", "SIGNAL"),

    # ==> SPREAD TP1 HIT: 0.552 >= 0.552 — closing 50% ...
    (re.compile(
        r"==> SPREAD TP1 HIT: ([\d.]+) >= ([\d.]+)"
     ), "SPR", "EXIT"),

    # ==> SPREAD TP2 HIT: 0.600 >= 0.600 — closing remaining ...
    (re.compile(
        r"==> SPREAD TP2 HIT: ([\d.]+) >= ([\d.]+)"
     ), "SPR", "EXIT"),

    # ==> SPREAD SL HIT: 0.374 < 0.408 — closing N sh
    (re.compile(
        r"==> SPREAD SL HIT: ([\d.]+) < ([\d.]+)"
     ), "SPR", "EXIT"),

    # ==> SPREAD CLOSED: SL | 5.00 sh | PnL: -$0.35
    (re.compile(
        r"==> SPREAD CLOSED: (\w+) \| ([\d.]+) sh \| PnL: ([+\-\$\d.]+)"
     ), "SPR", "EXIT"),

    # ==> SPR ENTRY FAILED / LEG FAILED
    (re.compile(r"==> SPR ENTRY FAILED: (.+)"), "SPR", "ERROR"),
    (re.compile(r"==> SPR LEG FAILED: (.+)"), "SPR", "ERROR"),
    (re.compile(r"==> SPR SELL FAILED: (.+)"), "SPR", "ERROR"),

    # ==> SPREAD CLOSE FAILED / PARTIAL
    (re.compile(r"==> SPREAD CLOSE FAILED: (.+)"), "SPR", "ERROR"),
    (re.compile(r"==> SPREAD CLOSE PARTIAL: (.+)"), "SPR", "ERROR"),

    # ==> EXPIRY SNIPE: UP @ 0.920 | 20s remaining | expect $1.00 resolution
    (re.compile(
        r"==> EXPIRY SNIPE: (\w+) @ ([\d.]+) \| (\d+)s remaining"
     ), "SPR", "ENTRY"),

    # ==> MOMENTUM CLOSE PARTIAL: SL | sold 2.00/5.00 sh | PnL: -$0.60 | remaining 3.00
    (re.compile(
        r"==> MOMENTUM CLOSE PARTIAL: (\w+) \| sold ([\d.]+)/([\d.]+) sh \| PnL: ([+\-\$\d.]+) \| remaining ([\d.]+)"
     ), "MOM", "ERROR"),

    # ==> MOMENTUM CLOSE FAILED: SL | FORCE CLEAR after 5 attempts
    (re.compile(
        r"==> MOMENTUM CLOSE FAILED: (\w+) \| (FORCE CLEAR.+)"
     ), "MOM", "ERROR"),

    # src.client HTTP error: HTTP 400 POST /order: {"errorMsg":"...",...}
    # Also matches 401, 403, 422, 500, etc.
    (re.compile(r"HTTP (\d+) (\w+) ([^ :]+): (.+)"), "BOT", "ERROR"),

    # ==> CIRCUIT BREAKER: trading halted — session loss limit ($15.00 > $15.00)
    (re.compile(r"==> CIRCUIT BREAKER: trading halted — (.+)"), "SYS", "HALT"),

    # ==> SPREAD SELL: UP ask=0.540 + DN ask=0.540 = 1.080 > threshold 1.040 | locked=$0.40
    (re.compile(
        r"==> SPREAD SELL: UP ask=([\d.]+) \+ DN ask=([\d.]+) = ([\d.]+) > threshold ([\d.]+) \| locked=\$([\d.]+)"
     ), "SPR", "SIGNAL"),

    # ==> SPREAD SELL POSTED: UP order_id=0x... | DOWN order_id=0x... | locked=$0.40
    (re.compile(
        r"==> SPREAD SELL POSTED: UP order_id=\S+ \| DOWN order_id=\S+ \| locked=\$([\d.]+)"
     ), "SPR", "LOCKED"),
]


# ── Event capture (logging.Handler) ──────────────────────────────────────────

class TradingEventCapture(logging.Handler):
    """
    Intercepts bot/coordinator log records and converts them into
    structured TEvent objects.  Tracks running session & window PnL.

    Events accumulate for the lifetime of the current market window.
    When a NEW MARKET event arrives the window log is cleared so the
    display always shows the full history of the *current* window.
    """

    def __init__(self):
        super().__init__()
        # No maxlen — we keep every event until the window resets.
        # A 5-minute window typically produces <100 events so memory is fine.
        self.events: list[TEvent] = []
        self.session_pnl: float = 0.0
        self.window_pnl: float = 0.0
        # Errors persist across market resets so they're never lost
        self.errors: list[TEvent] = []

    # ── logging.Handler interface ─────────────────────────────────────────────

    def emit(self, record: logging.LogRecord) -> None:
        try:
            ev = self._parse(record.getMessage())
        except Exception:
            return  # never let a parse error crash the caller's thread
        if ev:
            if ev.etype == "MARKET":
                # New window → clear the per-window log and reset window PnL
                self.events.clear()
                self.window_pnl = 0.0
            # Errors and HALT events go into a separate sticky list that never clears
            if ev.etype in ("ERROR", "HALT"):
                self.errors.insert(0, ev)
                if len(self.errors) > 20:   # cap at 20 to avoid unbounded growth
                    self.errors.pop()
            # Prepend so index-0 is always the most recent event
            self.events.insert(0, ev)
            if ev.pnl is not None:
                self.session_pnl += ev.pnl
                self.window_pnl  += ev.pnl

    # ── Parser ────────────────────────────────────────────────────────────────

    def _parse(self, msg: str) -> Optional[TEvent]:
        for pat, bot, etype in _PATS:
            m = pat.search(msg)
            if not m:
                continue
            g = m.groups()

            if etype == "ENTRY":
                if len(g) == 6:
                    # Standard bot entry: (side, cost, entry, tp1, tp2, sl)
                    side, cost, entry, tp1, tp2, sl = g
                    detail = f"{side:<4} {cost}sh @ {entry}  SL={sl}  TP1={tp1}  TP2={tp2}"
                    return TEvent(etype="ENTRY", bot=bot, side=side, detail=detail)
                elif len(g) == 3 and "EXPIRY SNIPE" in msg:
                    # Expiry snipe: (side, price, tte_secs)
                    snap_side, snap_price, snap_tte = g
                    detail = f"{snap_side:<4} snipe @ {snap_price}  {snap_tte}s left"
                    return TEvent(etype="ENTRY", bot="MR", side=snap_side, detail=detail)
                elif len(g) == 3:
                    # Spread posted: (up_bid, down_bid, size_per_leg)
                    up_bid, dn_bid, size = g
                    detail = f"UP @ {up_bid}  DN @ {dn_bid}  {size}sh/leg"
                    return TEvent(etype="ENTRY", bot=bot, side="--", detail=detail)

            if etype == "EXIT":
                if len(g) == 4:
                    # Bot exit: (reason, shares, price, pnl_raw)
                    reason, shares, price, pnl_raw = g
                elif len(g) == 3:
                    # Spread closed: (reason, shares, pnl_raw) — no fill price
                    reason, shares, pnl_raw = g
                    price = "—"
                else:
                    reason, pnl_raw = str(g[0]), str(g[-1])
                    shares, price = "?", "?"
                try:
                    pnl = float(pnl_raw.replace("$", "").replace("+", ""))
                except ValueError:
                    pnl = 0.0
                detail = f"{reason:<6} {shares}sh @ {price}  pnl={pnl_raw}"
                return TEvent(etype="EXIT", bot=bot, side="--", detail=detail, pnl=pnl)

            if etype == "SIGNAL":
                if len(g) == 3:
                    side, conf, info = g
                    detail = f"{side:<4} conf={conf}  {info.strip()}"
                    return TEvent(etype="SIGNAL", bot=bot, side=side, detail=detail)
                elif len(g) == 4:
                    # SPR single leg: (side, price, shares, sl)
                    s_side, s_price, s_shares, s_sl = g
                    detail = f"{s_side:<4} single @ {s_price}  {s_shares}sh  SL={s_sl}"
                    return TEvent(etype="SIGNAL", bot=bot, side=s_side, detail=detail)
                elif len(g) == 5 and bot == "SPR":
                    # SPREAD SELL detected: (up_ask, dn_ask, sum, threshold, locked)
                    up_a, dn_a, _sum, _thr, locked = g
                    detail = f"SELL UP@{up_a} + DN@{dn_a}  locked +${locked}"
                    return TEvent(etype="SIGNAL", bot="SPR", side="--", detail=detail)
                else:
                    detail = "  ".join(str(x) for x in g)
                    return TEvent(etype="SIGNAL", bot=bot, side="--", detail=detail)

            if etype == "MARKET":
                return TEvent(etype="MARKET", bot="SYS", side="--", detail=f"NEW MARKET  {g[0]}")

            if etype == "SYSTEM":
                return TEvent(etype="SYSTEM", bot="SYS", side="--", detail=f"late window  {g[0]}")

            if etype == "MAKER":
                side, shares, price = g
                detail = f"{side:<4} maker @ {price}  {shares}sh"
                return TEvent(etype="MAKER", bot="MR", side=side, detail=detail)

            if etype == "ESCALATE":
                return TEvent(etype="ESCALATE", bot="MR", side="--", detail="maker → taker")

            if etype == "LOCKED":
                if len(g) == 4:
                    # SPREAD FILLED: (up_bid, down_bid, cost, locked)
                    up_b, dn_b, cost, locked = g
                    detail = f"UP {up_b} + DN {dn_b}  cost ${cost}  locked +${locked}"
                    try:
                        pnl = float(locked)
                    except ValueError:
                        pnl = None
                    return TEvent(etype="LOCKED", bot="SPR", side="--", detail=detail, pnl=pnl)
                elif len(g) == 3:
                    # SPREAD RELEASED: (up_entry, down_entry, locked)
                    up_e, dn_e, locked = g
                    detail = f"released UP {up_e} + DN {dn_e}  locked +${locked} → on-chain"
                    return TEvent(etype="LOCKED", bot="SPR", side="--", detail=detail)
                elif len(g) == 1:
                    # SPREAD SELL POSTED: (locked)
                    locked = g[0]
                    try:
                        pnl = float(locked)
                    except ValueError:
                        pnl = None
                    detail = f"SELL both legs posted  locked +${locked}"
                    return TEvent(etype="LOCKED", bot="SPR", side="--", detail=detail, pnl=pnl)
                else:
                    return TEvent(etype="LOCKED", bot="SPR", side="--",
                                  detail="  ".join(str(x) for x in g))

            if etype == "HALT":
                # CIRCUIT BREAKER triggered — sticky error-level event
                reason = g[0] if g else "unknown"
                return TEvent(etype="HALT", bot="SYS", side="--",
                              detail=f"TRADING HALTED — {reason.strip()}")

            if etype == "RATCHET":
                if len(g) == 3 and g[0]:
                    # Trailing SL ratchet: groups = (old_sl, new_sl, "[TRAIL]"?)
                    trail_tag = " [TSL]" if g[2] else ""
                    label = f"SL {g[0]} → {g[1]}{trail_tag}"
                else:
                    label = "TP1 ratcheted → BE" if "TP1" in msg else "TP2 ratcheted → TP1"
                return TEvent(etype="RATCHET", bot=bot, side="--", detail=label)

            if etype == "ERROR":
                if len(g) == 5:
                    # CLOB rejection: (side, size, price, order_type, reason)
                    e_side, e_size, e_price, e_type, e_reason = g
                    detail = f"{e_side} {e_size}@{e_price} [{e_type}] {e_reason.strip()}"
                elif len(g) == 4:
                    # HTTP error from src.client: (status, method, endpoint, body)
                    h_status, h_method, h_ep, h_body = g
                    detail = f"HTTP {h_status} {h_method} {h_ep}: {h_body.strip()}"
                elif len(g) == 3:
                    trigger, attempt, reason = g
                    detail = f"{trigger} attempt {attempt}: {reason.strip()}"
                elif len(g) == 2:
                    trigger, detail_raw = g
                    detail = f"{trigger} {detail_raw.strip()}"
                elif len(g) == 1:
                    detail = str(g[0]).strip()
                else:
                    detail = " ".join(str(x) for x in g)
                return TEvent(etype="ERROR", bot=bot, side="--", detail=detail)

        return None


# ── TUI renderer ──────────────────────────────────────────────────────────────

_W = 80   # terminal columns
_HALF = 38  # columns per bot panel (excluding left margin)

_ICONS = {
    "SIGNAL":   ("◈", C.CYN),
    "ENTRY":    ("▶", C.GRN),
    "EXIT":     ("✓", C.YLW),
    "LOCKED":   ("◉", C.GRN),   # spread both-legs-filled / released
    "MARKET":   ("⟳", C.MAG),
    "SYSTEM":   ("ℹ", C.BLU),
    "MAKER":    ("◷", C.CYN),
    "ESCALATE": ("↑", C.YLW),
    "RATCHET":  ("↗", C.GRN),
    "ERROR":    ("✖", C.RED),
    "HALT":     ("⛔", C.RED),  # session circuit breaker triggered
}


def _ruler(ch: str = "─") -> str:
    return ch * _W


def _section(title: str) -> str:
    return _c(C.B, f" {title}")


class TradingTUI:
    """
    Renders the dual-bot dashboard in-place every ``refresh_interval``
    seconds.  All live data is read directly from the shared objects —
    no extra queues needed.
    """

    def __init__(
        self,
        capture: TradingEventCapture,
        shared_state: "SharedState",
        coordinator: "Coordinator",
        momentum_bot: Optional["MomentumBot"] = None,
        mr_bot: Optional["MeanReversionBot"] = None,
        spread_bot: Optional["SpreadBot"] = None,
        btc_tracker: Optional["BTCPriceTracker"] = None,
        refresh_interval: float = 0.5,
    ):
        self.capture  = capture
        self.state    = shared_state
        self.coord    = coordinator
        self.mom      = momentum_bot
        self.mr       = mr_bot
        self.spr      = spread_bot
        self.btc      = btc_tracker
        self.interval = refresh_interval
        self._running = False
        self._kb_thread: Optional[threading.Thread] = None
        # Brief status message shown in the footer after a local-clear action
        self._notify_msg: str = ""
        self._notify_ts:  float = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        import asyncio
        self._running = True
        self._kb_thread = threading.Thread(
            target=self._keyboard_loop, daemon=True, name="tui-keyboard"
        )
        self._kb_thread.start()
        while self._running:
            try:
                self._render()
            except Exception:
                pass
            await asyncio.sleep(self.interval)

    def stop(self) -> None:
        self._running = False

    # ── Keyboard listener (runs in background thread) ─────────────────────────

    def _keyboard_loop(self) -> None:
        """
        Poll for single-character keypresses without blocking the render loop.
        No terminal mode is changed on Windows; on Unix setcbreak is used.
        Errors are swallowed — the keyboard listener is purely optional.
        """
        try:
            if sys.platform == "win32":
                import msvcrt
                while self._running:
                    if msvcrt.kbhit():
                        raw = msvcrt.getch()
                        # Extended / arrow keys arrive as two bytes; skip both.
                        if raw in (b"\x00", b"\xe0"):
                            msvcrt.getch()
                            continue
                        try:
                            key = raw.decode("utf-8", errors="ignore").lower()
                        except Exception:
                            continue
                        self._handle_key(key)
                    time.sleep(0.05)
            else:
                import select
                import termios
                import tty
                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                try:
                    tty.setcbreak(fd)
                    while self._running:
                        r, _, _ = select.select([sys.stdin], [], [], 0.1)
                        if r:
                            self._handle_key(sys.stdin.read(1).lower())
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass  # keyboard listener is optional

    def _handle_key(self, key: str) -> None:
        if key == "1":
            self._local_clear("A")
        elif key == "2":
            self._local_clear("B")
        elif key == "3":
            self._local_clear("C")
        elif key == "0":
            self._local_clear("A")
            self._local_clear("B")
            self._local_clear("C")
        elif key == "c":
            # Hard-clear the terminal buffer so stale output is fully gone.
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
            self._notify_msg = "Screen cleared"
            self._notify_ts  = time.time()
        elif key == "e":
            self.capture.errors.clear()
            self._notify_msg = "Error log cleared"
            self._notify_ts  = time.time()

    def _local_clear(self, which: str) -> None:
        """
        Wipe the bot's in-memory position state.

        *** NO orders are sent to Polymarket. ***
        This is purely a local state reset — useful when you manually
        closed a trade on Polymarket and want the bot to stop tracking it.
        """
        cleared: list[str] = []

        if which == "A":
            if self.mom is not None and self.mom._position is not None:
                side = self.mom._position.side.value.upper()
                self.mom._position = None
                cleared.append(f"BotA pos ({side})")

        elif which == "B":
            if self.mr is not None and self.mr._position is not None:
                side = self.mr._position.side.value.upper()
                self.mr._position = None
                cleared.append(f"BotB pos ({side})")
            if self.mr is not None and self.mr._pending_maker_order is not None:
                self.mr._pending_maker_order = None
                cleared.append("BotB maker order")

        elif which == "C" and self.spr is not None:
            if self.spr._position is not None:
                self.spr._position = None
                cleared.append("BotC position")
            if self.spr._legs is not None:
                self.spr._legs = None
                cleared.append("BotC pending legs")

        msg = ("LOCAL CLEAR: " + ", ".join(cleared)) if cleared else "LOCAL CLEAR: nothing to clear"
        self._notify_msg = msg
        self._notify_ts  = time.time()

        self.capture.events.insert(0, TEvent(
            etype="SYSTEM",
            bot="SYS",
            side="--",
            detail=msg,
        ))

    # ── Top-level render ──────────────────────────────────────────────────────

    def _render(self) -> None:
        market = self.state.get_market_data()
        risk   = self.state.get_risk_metrics()
        status = self.coord.get_status()
        window = status.get("window", {})
        late   = status.get("in_late_window", False)

        btc_price  = self.btc.get_price()   if self.btc else 0.0
        btc_src    = self.btc.get_source()  if self.btc else "—"
        btc_stale  = self.btc.is_stale(3)  if self.btc else True
        spot_2s    = risk.btc_spot_change_2s
        regime     = risk.regime.value if hasattr(risk.regime, "value") else str(risk.regime)
        total_ent  = window.get("total_entries", 0)

        lines: list[str] = []

        # ── Header bar ───────────────────────────────────────────────────────
        ts_str  = _ts_now()
        title   = "POLYMARKET DUAL BOT"
        inner   = f" {title} "
        pad_l   = "═" * ((_W - len(inner) - len(ts_str) - 1) // 2)
        pad_r   = "═" * (_W - 1 - len(pad_l) - len(inner) - len(ts_str))
        lines.append(_c(C.B, f"═{pad_l}{inner}{pad_r}") + _c(C.CYN, ts_str))

        # ── Market info line ──────────────────────────────────────────────────
        slug    = (market.market_slug or "—")[:38]
        if btc_price and not btc_stale:
            src_tag = _c(C.DIM, f"[{btc_src}]")
            btc_str = f"BTC ${btc_price:>9,.2f} {src_tag}"
        elif btc_stale:
            btc_str = _c(C.YLW, "BTC STALE")
        else:
            btc_str = _c(C.DIM, "BTC ———")
        lines.append(
            f" {_c(C.B,'Market:')} {slug:<38}  "
            f"{btc_str}  {_pct(spot_2s,3)}(2s)"
        )

        # ── Status bar ───────────────────────────────────────────────────────
        late_tag = f"  {_c(C.B+C.YLW, '⚠ LATE WINDOW')}" if late else ""
        lines.append(
            f" Expiry {_countdown(market.time_to_expiry)}  "
            f"Regime {_regime(regime)}  "
            f"VPIN {_c(C.RED if risk.vpin>0.7 else C.WHT, f'{risk.vpin:.2f}')}  "
            f"Frag {_c(C.RED if risk.fragility>0.6 else C.WHT, f'{risk.fragility:.2f}')}  "
            f"Entries {_c(C.YLW if total_ent>=2 else C.GRN, f'{total_ent}/2')}"
            f"{late_tag}"
        )
        lines.append(_ruler())

        # ── Bot panels (side-by-side) ─────────────────────────────────────────
        mom_lines = self._mom_panel(market, risk, regime)
        mr_lines  = self._mr_panel(market, risk, regime)
        if mom_lines or mr_lines:
            h = max(len(mom_lines), len(mr_lines))
            mom_lines += [""] * (h - len(mom_lines))
            mr_lines  += [""] * (h - len(mr_lines))
            for ml, rl in zip(mom_lines, mr_lines):
                lines.append(" " + _pad(ml, _HALF) + "  " + rl)
            lines.append(_ruler())

        # ── Spread panel (full width) ─────────────────────────────────────────
        if self.spr is not None:
            for l in self._spr_panel(market, risk):
                lines.append(l)
            lines.append(_ruler())

        # ── Market snapshot ───────────────────────────────────────────────────
        up_bid  = market.up_bids[0][0]   if market.up_bids   else 0.0
        up_ask  = market.up_asks[0][0]   if market.up_asks   else 0.0
        dn_bid  = market.down_bids[0][0] if market.down_bids else 0.0
        dn_ask  = market.down_asks[0][0] if market.down_asks else 0.0

        # PnL read directly from bot objects (reliable — not log-parsed)
        spr_rpnl    = self.spr._realized_pnl if self.spr else 0.0
        spr_wpnl    = self.spr._window_pnl   if self.spr else 0.0
        mom_rpnl    = self.mom._realized_pnl if self.mom else 0.0
        mom_wpnl    = self.mom._window_pnl   if self.mom else 0.0
        mr_rpnl     = self.mr._realized_pnl  if self.mr  else 0.0
        mr_wpnl     = self.mr._window_pnl    if self.mr  else 0.0
        session_pnl = mom_rpnl + mr_rpnl + spr_rpnl
        window_pnl  = mom_wpnl + mr_wpnl  + spr_wpnl

        # Unrealized: only open positions contribute
        mom_pos = self.mom.position if self.mom else None
        mr_pos  = self.mr.position  if self.mr  else None
        spr_pos = self.spr.position if self.spr else None
        unreal = 0.0
        if mom_pos:
            mpx = market.up_price if mom_pos.side.value == "up" else market.down_price
            unreal += (mpx - mom_pos.entry_price) * mom_pos.size
        if mr_pos:
            mpx = market.up_price if mr_pos.side.value == "up" else market.down_price
            unreal += (mpx - mr_pos.entry_price) * mr_pos.size
        if spr_pos and spr_pos.status != "spread":
            # Single directional leg — calculate unrealized
            if spr_pos.status == "long_up" and spr_pos.up_size > 0:
                unreal += (market.up_price - spr_pos.up_entry) * spr_pos.up_size
            elif spr_pos.status == "long_down" and spr_pos.down_size > 0:
                unreal += (market.down_price - spr_pos.down_entry) * spr_pos.down_size

        lines.append(
            f" {_c(C.GRN,'UP')}  mid {market.up_price:.3f}"
            f"  bid {up_bid:.3f} / ask {up_ask:.3f}"
            f"    │  Session {_pnl(session_pnl)}  Unreal {_pnl(unreal)}"
        )
        lines.append(
            f" {_c(C.RED,'DN')}  mid {market.down_price:.3f}"
            f"  bid {dn_bid:.3f} / ask {dn_ask:.3f}"
            f"    │  Window  {_pnl(window_pnl)}"
            + (f"  mom {_pnl(mom_wpnl)}  mr {_pnl(mr_wpnl)}" if (self.mom or self.mr) else "")
            + (f"  spr {_pnl(spr_wpnl)}" if self.spr else "")
        )
        lines.append(_ruler())

        # ── Sticky error panel — persists across market resets ────────────────────
        if self.capture.errors:
            lines.append(
                _c(C.B + C.RED, " ✖ ORDER FAILURES")
                + _c(C.DIM, "  (persist across markets — most recent first)")
            )
            # Each error wraps across up to 3 lines so the full JSON body is visible.
            line_w   = _W - 14   # usable chars per continuation line
            prefix_w = 14        # "  HH:MM:SS  ✖ BOT " header
            for ev in self.capture.errors[:5]:
                ts_s    = datetime.fromtimestamp(ev.ts).strftime("%H:%M:%S")
                bot_col = C.CYN if ev.bot == "MOM" else (C.MAG if ev.bot == "MR" else C.YLW)
                bot_str = _c(bot_col, f"{ev.bot:<3}")
                raw     = ev.detail
                # Split into chunks of line_w visible chars
                chunks: list[str] = []
                while _vis(raw):
                    chunks.append(raw[:line_w])
                    raw = raw[line_w:]
                    if len(chunks) >= 3:
                        if raw:
                            chunks[-1] = chunks[-1][:-1] + "…"
                        break
                header = f"  {_c(C.DIM, ts_s)}  {_c(C.RED, '✖')} {bot_str} {_c(C.RED, chunks[0])}"
                lines.append(header)
                indent = " " * prefix_w
                for chunk in chunks[1:]:
                    lines.append(f"{indent}{_c(C.RED, chunk)}")
            lines.append(_ruler())

        # ── Event log — all events for this window ────────────────────────────────
        n_ev = len(self.capture.events)
        ev_plural = "s" if n_ev != 1 else ""
        lines.append(
            f" {_c(C.B, 'EVENT LOG  (this window)')}"
            f"{_c(C.DIM, f'  {n_ev} event{ev_plural}  ·  clears on new market')}"
        )

        if not self.capture.events:
            lines.append(_c(C.DIM, "   no events yet — waiting for signals…"))
        else:
            # Show ALL events for this window (newest first); terminal scrolls naturally.
            max_d    = _W - 34
            max_err  = _W * 4    # ERROR details can be long JSON — allow up to 4× width
            for ev in self.capture.events:
                ts_s    = datetime.fromtimestamp(ev.ts).strftime("%H:%M:%S")
                icon_ch, icon_col = _ICONS.get(ev.etype, ("·", C.DIM))
                icon    = _c(icon_col, icon_ch)
                bot_col = C.CYN if ev.bot == "MOM" else (C.MAG if ev.bot == "MR" else (C.YLW if ev.bot == "SPR" else C.DIM))
                bot_str = _c(bot_col, f"{ev.bot:<3}")
                etype_s = _c(_ICONS.get(ev.etype, ("·", C.DIM))[1], f"{ev.etype:<8}")
                pnl_tag = f"  {_pnl(ev.pnl)}" if ev.pnl is not None else ""
                detail  = ev.detail
                limit   = max_err if ev.etype == "ERROR" else max_d
                if len(_vis(detail)) > limit:
                    detail = detail[:limit] + "…"
                lines.append(
                    f"  {_c(C.DIM, ts_s)}  {icon} {bot_str} {etype_s} {detail}{pnl_tag}"
                )

        lines.append(_ruler("═"))

        # Show brief flash notification for 3 s after a local-clear action
        if self._notify_msg and (time.time() - self._notify_ts) < 3.0:
            lines.append(
                _c(C.YLW + C.B, f"  ⚡ {self._notify_msg}")
            )
        else:
            self._notify_msg = ""

        lines.append(
            _c(C.DIM, "  Ctrl+C stop")
            + "  "
            + _c(C.B + C.YLW, "[1]") + _c(C.DIM, " BotA  ")
            + _c(C.B + C.YLW, "[2]") + _c(C.DIM, " BotB  ")
            + _c(C.B + C.YLW, "[3]") + _c(C.DIM, " BotC  ")
            + _c(C.B + C.YLW, "[0]") + _c(C.DIM, " all  ")
            + _c(C.DIM, "← clear display only, no cancel/sell sent  ")
            + _c(C.B + C.CYN, "[c]") + _c(C.DIM, " screen  ")
            + _c(C.B + C.CYN, "[e]") + _c(C.DIM, " errors")
        )

        sys.stdout.write("\033[H\033[J" + "\n".join(lines) + "\n")
        sys.stdout.flush()

    # ── Bot panel builders ────────────────────────────────────────────────────

    def _mom_panel(self, market, risk, regime: str) -> list[str]:
        if self.mom is None:
            return []
        pos    = self.mom.position
        stats  = self.mom.get_stats()
        spot2  = risk.btc_spot_change_2s
        cool   = time.time() - self.mom._last_entry_time >= self.mom.config.cooldown_seconds

        thr = {"high": 0.0018, "normal": 0.0020, "low": 0.0025}.get(regime, 0.0020)

        lines = [_c(C.B + C.CYN, "── BOT A · MOMENTUM ──────────────────")]

        if pos:
            mkt_px  = market.up_price if pos.side.value == "up" else market.down_price
            unreal  = (mkt_px - pos.entry_price) * pos.size
            unreal_pct = (mkt_px - pos.entry_price) / pos.entry_price if pos.entry_price else 0
            status_str = _c(C.GRN, f"LONG {pos.side.value.upper()}")
            lines.append(f"{'Status':<10} {status_str}  ({_elapsed(pos.entry_time)})")
            lines.append(f"{'Entry':<10} {pos.entry_price:.4f}  size {pos.size:.1f}sh")
            tp1 = pos.entry_price * (1 + self.mom.config.tp1_pct)
            tp2 = pos.entry_price * (1 + self.mom.config.tp2_pct)
            tp3 = self.mom.config.tp3_price
            lines.append(
                f"{'TP1':<10} {tp1:.4f} {'✓' if pos.tp1_filled else '○'}"
                f"  TP2 {tp2:.4f} {'✓' if pos.tp2_filled else '○'}"
                f"  TP3 {tp3:.2f} {'✓' if pos.tp3_filled else '○'}"
            )
            peak = getattr(pos, '_peak_price', pos.entry_price)
            sl_trailed = pos.tp1_filled and pos.sl_price > pos.entry_price
            sl_tag = _c(C.YLW, " [TSL]") if sl_trailed else ""
            lines.append(f"{'SL':<10} {pos.sl_price:.4f}{sl_tag}  peak {peak:.4f}")
            lines.append(f"{'Unreal PnL':<10} {_pnl(unreal)}  ({_pct(unreal_pct)})")
            lines.append(f"{'Real PnL':<10} {_pnl(self.mom._realized_pnl)}  (win {_pnl(self.mom._window_pnl)})")
        else:
            lines.append(f"{'Status':<10} {_c(C.DIM, 'WATCHING')}")
            lines.append(f"{'Spot 2s':<10} {_pct(spot2, 3)}  (thr {thr*100:.2f}%)")
            lines.append(f"{'Cooldown':<10} {_c(C.GRN,'READY') if cool else _c(C.YLW,'waiting')}")
            lines.append(f"{'Entries':<10} {stats['entries_today']}  wins {stats['wins_today']}")
            lines.append(f"{'Real PnL':<10} {_pnl(self.mom._realized_pnl)}  (win {_pnl(self.mom._window_pnl)})")
            lines.append(_c(C.DIM, "  — no position —"))

        return lines

    def _mr_panel(self, market, risk, regime: str) -> list[str]:
        if self.mr is None:
            return []
        pos     = self.mr.position
        stats   = self.mr.get_stats()
        pending = self.mr._pending_maker_order
        cool    = time.time() - self.mr._last_entry_time >= self.mr.config.cooldown_seconds

        min_drop = {"high": 0.12, "normal": 0.10, "low": 0.08}.get(regime, 0.10)

        lines = [_c(C.B + C.MAG, "── BOT B · MEAN REVERSION ────────────")]

        if pos:
            mkt_px  = market.up_price if pos.side.value == "up" else market.down_price
            unreal  = (mkt_px - pos.entry_price) * pos.size
            unreal_pct = (mkt_px - pos.entry_price) / pos.entry_price if pos.entry_price else 0
            status_str = _c(C.GRN, f"LONG {pos.side.value.upper()}")
            lines.append(f"{'Status':<10} {status_str}  ({_elapsed(pos.entry_time)})")
            lines.append(f"{'Entry':<10} {pos.entry_price:.4f}  size {pos.size:.1f}sh")
            tp1 = pos.entry_price * (1 + self.mr.config.tp1_pct)
            tp2 = pos.entry_price * (1 + self.mr.config.tp2_pct)
            tp3 = self.mr.config.tp3_price
            lines.append(
                f"{'TP1':<10} {tp1:.4f} {'✓' if pos.tp1_filled else '○'}"
                f"  TP2 {tp2:.4f} {'✓' if pos.tp2_filled else '○'}"
                f"  TP3 {tp3:.2f} {'✓' if pos.tp3_filled else '○'}"
            )
            peak = getattr(pos, '_peak_price', pos.entry_price)
            sl_trailed = pos.tp1_filled and pos.sl_price > pos.entry_price
            sl_tag = _c(C.YLW, " [TSL]") if sl_trailed else ""
            lines.append(f"{'SL':<10} {pos.sl_price:.4f}{sl_tag}  peak {peak:.4f}")
            lines.append(f"{'Unreal PnL':<10} {_pnl(unreal)}  ({_pct(unreal_pct)})")
            lines.append(f"{'Real PnL':<10} {_pnl(self.mr._realized_pnl)}  (win {_pnl(self.mr._window_pnl)})")
        elif pending:
            lines.append(f"{'Status':<10} {_c(C.YLW, 'MAKER RESTING')}")
            p_price = pending.get("price", 0.0)
            p_size  = pending.get("size", 0.0)
            lines.append(f"{'Order':<10} @ {p_price:.4f}  {p_size:.1f}sh")
            lines.append(f"{'Entries':<10} {stats['entries_today']}  wins {stats['wins_today']}")
            lines.append(f"{'Real PnL':<10} {_pnl(self.mr._realized_pnl)}  (win {_pnl(self.mr._window_pnl)})")
            lines.append(_c(C.DIM, "  — waiting for fill —"))
        else:
            lines.append(f"{'Status':<10} {_c(C.DIM, 'WATCHING')}")
            lines.append(f"{'Drop thr':<10} {min_drop*100:.0f}%  z≥2.5")
            lines.append(f"{'Cooldown':<10} {_c(C.GRN,'READY') if cool else _c(C.YLW,'waiting')}")
            lines.append(f"{'Entries':<10} {stats['entries_today']}  wins {stats['wins_today']}")
            lines.append(f"{'Real PnL':<10} {_pnl(self.mr._realized_pnl)}  (win {_pnl(self.mr._window_pnl)})")
            lines.append(_c(C.DIM, "  — no position —"))

        return lines


    def _spr_panel(self, market, risk) -> list[str]:
        """Full-width panel for the Spread Bot (Bot C)."""
        spr   = self.spr
        pos   = spr.position
        legs  = spr.legs
        stats = spr.get_stats()

        lines = [_c(C.B + C.YLW, "── BOT C · SPREAD ─────────────────────────────────────────────────────────────")]

        if pos and pos.status == "spread":
            cost   = pos.cost
            locked = pos.guaranteed_profit
            lines.append(
                f"{'Status':<12} {_c(C.GRN, 'SPREAD LOCKED')}  "
                f"UP {pos.up_entry:.3f} + DOWN {pos.down_entry:.3f}  "
                f"cost ${cost:.2f}  "
                f"locked {_pnl(locked)}"
            )
            lines.append(f"{'Outcome':<12} holds to expiry — one leg resolves at $1.00")
            lines.append(f"{'Session':<12} {_pnl(spr._realized_pnl)}  "
                         f"window {_pnl(spr._window_pnl)}  "
                         f"spreads_done={stats['spreads_completed']}")

        elif pos and pos.status in ("long_up", "long_down"):
            side_s = "UP" if pos.status == "long_up" else "DOWN"
            entry  = pos.up_entry if pos.status == "long_up" else pos.down_entry
            size   = pos.up_size  if pos.status == "long_up" else pos.down_size
            cur_px = market.up_price if pos.status == "long_up" else market.down_price
            unreal = (cur_px - entry) * size
            lines.append(
                f"{'Status':<12} {_c(C.YLW, f'DIRECTIONAL {side_s}')}  "
                f"entry {entry:.3f}  {size:.1f}sh  "
                f"cur {cur_px:.3f}  unreal {_pnl(unreal)}"
            )
            sl_pct  = (pos.sl_price  - entry) / entry if entry else 0
            tp1_pct = (pos.tp1_price - entry) / entry if entry and pos.tp1_price else 0
            tp2_pct = (pos.tp2_price - entry) / entry if entry and pos.tp2_price else 0
            tp1_tag = _c(C.DIM, f"TP1 {pos.tp1_price:.3f} ({_pct(tp1_pct)}) ✓") if pos.tp1_hit else \
                      f"TP1 {pos.tp1_price:.3f} ({_pct(tp1_pct)})"
            lines.append(
                f"{'TP1/TP2/SL':<12} {tp1_tag}  "
                f"TP2 {pos.tp2_price:.3f} ({_pct(tp2_pct)})  "
                f"SL {pos.sl_price:.3f} ({_pct(sl_pct)})  "
                f"({_elapsed(pos.entry_time)} held)"
            )
            rem = pos.remaining_size if pos.remaining_size > 0 else size
            lines.append(
                f"{'Open':<12} {rem:.1f} sh remaining  "
                + (_c(C.GRN, "TP1 filled — riding to TP2") if pos.tp1_hit else "waiting for TP1")
            )
            lines.append(f"{'Session':<12} {_pnl(spr._realized_pnl)}  window {_pnl(spr._window_pnl)}")

        elif legs:
            age = time.time() - legs.posted_at
            lines.append(
                f"{'Status':<12} {_c(C.CYN, 'PENDING MAKER')}  "
                f"UP @ {legs.up_bid:.3f}  DOWN @ {legs.down_bid:.3f}  "
                f"{legs.size:.0f}sh/leg  posted {_elapsed(legs.posted_at)}"
            )
            combined = (legs.up_bid + legs.down_bid) * legs.size
            locked   = legs.size * 1.0 - combined
            lines.append(
                f"{'Edge':<12} cost ${combined:.2f} → locked +${locked:.2f}  "
                f"(if both fill at quoted prices)"
            )
            lines.append(f"{'Session':<12} {_pnl(spr._realized_pnl)}  window {_pnl(spr._window_pnl)}")

        else:
            lines.append(
                f"{'Status':<12} {_c(C.DIM, 'WATCHING')}  "
                f"attempted={spr._spread_attempted_this_window}  "
                f"done={stats['spreads_completed']}  singles={stats['single_legs_today']}"
            )
            up_p   = market.up_price if market else 0.0
            down_p = market.down_price if market else 0.0
            in_band = (
                spr.config.target_low <= up_p <= spr.config.target_high
                and spr.config.target_low <= down_p <= spr.config.target_high
            )
            band_tag = _c(C.GRN, "IN BAND") if in_band else _c(C.DIM, f"need {spr.config.target_low:.2f}–{spr.config.target_high:.2f}")
            tte      = market.time_to_expiry if market else 0
            time_ok  = tte >= spr.config.min_time_remaining
            time_tag = _c(C.GRN, f"{tte}s OK") if time_ok else _c(C.YLW, f"{tte}s (need {spr.config.min_time_remaining}s)")

            # Potential locked profit: compute the same way check_spread does
            _up_ask  = market.up_asks[0][0]   if market.up_asks   else up_p
            _dn_ask  = market.down_asks[0][0] if market.down_asks else down_p
            _up_bid  = round(max(0.01, _up_ask  - spr.config.bid_offset), 4)
            _dn_bid  = round(max(0.01, _dn_ask  - spr.config.bid_offset), 4)
            _pot     = (1.0 - (_up_bid + _dn_bid)) * spr.config.size_per_leg
            if _pot >= spr.config.min_locked_profit:
                _pot_tag = _c(C.GRN,  f"pot +${_pot:.2f}")
            elif _pot > 0:
                _pot_tag = _c(C.YLW,  f"pot +${_pot:.2f} (< min ${spr.config.min_locked_profit:.2f})")
            else:
                _pot_tag = _c(C.DIM,  f"pot ${_pot:.2f}")

            lines.append(
                f"{'Condition':<12} mid_UP={up_p:.3f}  mid_DN={down_p:.3f}  {band_tag}  {_pot_tag}  time {time_tag}"
            )
            lines.append(f"{'Session':<12} {_pnl(spr._realized_pnl)}  window {_pnl(spr._window_pnl)}")

        return lines


# ── Helper: attach handler to bot loggers ─────────────────────────────────────

def install_tui_handler(capture: TradingEventCapture) -> None:
    """
    Attach the event-capture handler to the relevant bot/coordinator
    loggers and stop them propagating to the root handler while TUI
    is active (so trading noise doesn't scroll past the TUI).

    src.client keeps propagation ON so HTTP errors still reach the root
    file handler (logs/errors.log) even when TUI is running.
    """
    # Bot loggers: attach capture and silence console (TUI owns output)
    for name in (
        "bots.momentum_bot",
        "bots.mean_reversion_bot",
        "bots.coordinator",
        "bots.spread_bot",
        "src.bot",    # captures "Order rejected by CLOB: ..." warnings
    ):
        lg = logging.getLogger(name)
        lg.addHandler(capture)
        lg.propagate = False  # TUI owns trading output

    # src.client: attach capture but keep propagation so file handler still gets it
    _cl = logging.getLogger("src.client")
    _cl.addHandler(capture)
    # propagate stays True → errors also flow to logs/errors.log via root handler

    capture.setLevel(logging.DEBUG)
