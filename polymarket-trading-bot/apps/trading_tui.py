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

    # ── logging.Handler interface ─────────────────────────────────────────────

    def emit(self, record: logging.LogRecord) -> None:
        ev = self._parse(record.getMessage())
        if ev:
            if ev.etype == "MARKET":
                # New window → clear the per-window log and reset window PnL
                self.events.clear()
                self.window_pnl = 0.0
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
                side, cost, entry, tp1, tp2, sl = g
                detail = f"{side:<4} {cost}sh @ {entry}  SL={sl}  TP1={tp1}  TP2={tp2}"
                return TEvent(etype="ENTRY", bot=bot, side=side, detail=detail)

            if etype == "EXIT":
                reason, shares, price, pnl_raw = g
                try:
                    pnl = float(pnl_raw.replace("$", "").replace("+", ""))
                except ValueError:
                    pnl = 0.0
                detail = f"{reason:<6} {shares}sh @ {price}  pnl={pnl_raw}"
                return TEvent(etype="EXIT", bot=bot, side="--", detail=detail, pnl=pnl)

            if etype == "SIGNAL":
                side, conf, info = g
                detail = f"{side:<4} conf={conf}  {info.strip()}"
                return TEvent(etype="SIGNAL", bot=bot, side=side, detail=detail)

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

            if etype == "RATCHET":
                if len(g) == 3 and g[0]:
                    # Trailing SL ratchet: groups = (old_sl, new_sl, "[TRAIL]"?)
                    trail_tag = " [TSL]" if g[2] else ""
                    label = f"SL {g[0]} → {g[1]}{trail_tag}"
                else:
                    label = "TP1 ratcheted → BE" if "TP1" in msg else "TP2 ratcheted → TP1"
                return TEvent(etype="RATCHET", bot=bot, side="--", detail=label)

        return None


# ── TUI renderer ──────────────────────────────────────────────────────────────

_W = 80   # terminal columns
_HALF = 38  # columns per bot panel (excluding left margin)

_ICONS = {
    "SIGNAL":   ("◈", C.CYN),
    "ENTRY":    ("▶", C.GRN),
    "EXIT":     ("✓", C.YLW),
    "MARKET":   ("⟳", C.MAG),
    "SYSTEM":   ("ℹ", C.BLU),
    "MAKER":    ("◷", C.CYN),
    "ESCALATE": ("↑", C.YLW),
    "RATCHET":  ("↗", C.GRN),
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
        momentum_bot: "MomentumBot",
        mr_bot: "MeanReversionBot",
        btc_tracker: Optional["BTCPriceTracker"] = None,
        refresh_interval: float = 0.5,
    ):
        self.capture  = capture
        self.state    = shared_state
        self.coord    = coordinator
        self.mom      = momentum_bot
        self.mr       = mr_bot
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
        elif key == "0":
            self._local_clear("A")
            self._local_clear("B")

    def _local_clear(self, which: str) -> None:
        """
        Wipe the bot's in-memory position state.

        *** NO orders are sent to Polymarket. ***
        This is purely a local state reset — useful when you manually
        closed a trade on Polymarket and want the bot to stop tracking it.
        """
        cleared: list[str] = []

        if which == "A":
            if self.mom._position is not None:
                side = self.mom._position.side.value.upper()
                self.mom._position = None
                cleared.append(f"BotA pos ({side})")

        elif which == "B":
            if self.mr._position is not None:
                side = self.mr._position.side.value.upper()
                self.mr._position = None
                cleared.append(f"BotB pos ({side})")
            if self.mr._pending_maker_order is not None:
                self.mr._pending_maker_order = None
                cleared.append("BotB maker order")

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
        h = max(len(mom_lines), len(mr_lines))
        mom_lines += [""] * (h - len(mom_lines))
        mr_lines  += [""] * (h - len(mr_lines))

        for ml, rl in zip(mom_lines, mr_lines):
            lines.append(" " + _pad(ml, _HALF) + "  " + rl)

        lines.append(_ruler())

        # ── Market snapshot ───────────────────────────────────────────────────
        up_bid  = market.up_bids[0][0]   if market.up_bids   else 0.0
        up_ask  = market.up_asks[0][0]   if market.up_asks   else 0.0
        dn_bid  = market.down_bids[0][0] if market.down_bids else 0.0
        dn_ask  = market.down_asks[0][0] if market.down_asks else 0.0

        # PnL read directly from bot objects (reliable — not log-parsed)
        session_pnl = self.mom._realized_pnl + self.mr._realized_pnl
        window_pnl  = self.mom._window_pnl   + self.mr._window_pnl

        # Unrealized: only open positions contribute
        mom_pos = self.mom.position
        mr_pos  = self.mr.position
        unreal = 0.0
        if mom_pos:
            mpx = market.up_price if mom_pos.side.value == "up" else market.down_price
            unreal += (mpx - mom_pos.entry_price) * mom_pos.size
        if mr_pos:
            mpx = market.up_price if mr_pos.side.value == "up" else market.down_price
            unreal += (mpx - mr_pos.entry_price) * mr_pos.size

        lines.append(
            f" {_c(C.GRN,'UP')}  mid {market.up_price:.3f}"
            f"  bid {up_bid:.3f} / ask {up_ask:.3f}"
            f"    │  Session {_pnl(session_pnl)}  Unreal {_pnl(unreal)}"
        )
        lines.append(
            f" {_c(C.RED,'DN')}  mid {market.down_price:.3f}"
            f"  bid {dn_bid:.3f} / ask {dn_ask:.3f}"
            f"    │  Window  {_pnl(window_pnl)}"
            f"  mom {_pnl(self.mom._window_pnl)}  mr {_pnl(self.mr._window_pnl)}"
        )
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
            max_d = _W - 34
            for ev in self.capture.events:
                ts_s    = datetime.fromtimestamp(ev.ts).strftime("%H:%M:%S")
                icon_ch, icon_col = _ICONS.get(ev.etype, ("·", C.DIM))
                icon    = _c(icon_col, icon_ch)
                bot_col = C.CYN if ev.bot == "MOM" else (C.MAG if ev.bot == "MR" else C.DIM)
                bot_str = _c(bot_col, f"{ev.bot:<3}")
                etype_s = _c(_ICONS.get(ev.etype, ("·", C.DIM))[1], f"{ev.etype:<8}")
                pnl_tag = f"  {_pnl(ev.pnl)}" if ev.pnl is not None else ""
                detail  = ev.detail
                if len(_vis(detail)) > max_d:
                    detail = detail[:max_d] + "…"
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
            + _c(C.B + C.YLW, "[1]") + _c(C.DIM, " clear BotA  ")
            + _c(C.B + C.YLW, "[2]") + _c(C.DIM, " clear BotB  ")
            + _c(C.B + C.YLW, "[0]") + _c(C.DIM, " clear All")
            + _c(C.DIM, "  (local only — no trades sent)")
        )

        sys.stdout.write("\033[H\033[J" + "\n".join(lines) + "\n")
        sys.stdout.flush()

    # ── Bot panel builders ────────────────────────────────────────────────────

    def _mom_panel(self, market, risk, regime: str) -> list[str]:
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
            lines.append(
                f"{'TP1':<10} {tp1:.4f} {'✓' if pos.tp1_filled else '○'}"
                f"  TP2 {tp2:.4f} {'✓' if pos.tp2_filled else '○'}"
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
            lines.append(
                f"{'TP1':<10} {tp1:.4f} {'✓' if pos.tp1_filled else '○'}"
                f"  TP2 {tp2:.4f} {'✓' if pos.tp2_filled else '○'}"
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


# ── Helper: attach handler to bot loggers ─────────────────────────────────────

def install_tui_handler(capture: TradingEventCapture) -> None:
    """
    Attach the event-capture handler to the relevant bot/coordinator
    loggers and stop them propagating to the root handler while TUI
    is active (so trading noise doesn't scroll past the TUI).
    """
    for name in ("bots.momentum_bot", "bots.mean_reversion_bot", "bots.coordinator"):
        lg = logging.getLogger(name)
        lg.addHandler(capture)
        lg.propagate = False  # TUI owns trading output
    capture.setLevel(logging.DEBUG)
