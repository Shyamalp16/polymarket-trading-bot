#!/usr/bin/env python3
"""
Analyze trade telemetry and print threshold-tuning guidance.

Usage:
  python scripts/analyze_trade_logs.py
  python scripts/analyze_trade_logs.py --dir logs/trades --limit 5000
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    idx = int((len(vals) - 1) * p)
    return vals[idx]


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_events(log_dir: Path, limit: int) -> List[Dict[str, Any]]:
    files = sorted(log_dir.glob("trade_events_*.jsonl"))
    events: List[Dict[str, Any]] = []
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if limit > 0:
        events = events[-limit:]
    return events


def _parse_context(raw_context: Any) -> Dict[str, Any]:
    if isinstance(raw_context, dict):
        return raw_context
    if isinstance(raw_context, str) and raw_context:
        try:
            decoded = json.loads(raw_context)
            if isinstance(decoded, dict):
                return decoded
        except json.JSONDecodeError:
            return {}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Polybot trade telemetry")
    parser.add_argument("--dir", type=str, default="logs/trades", help="Telemetry directory")
    parser.add_argument("--limit", type=int, default=0, help="Max most-recent events to read (0=all)")
    args = parser.parse_args()

    log_dir = Path(args.dir)
    if not log_dir.exists():
        print(f"No telemetry directory found: {log_dir}")
        return

    events = _read_events(log_dir, args.limit)
    if not events:
        print("No trade telemetry events found.")
        return

    entries = [e for e in events if e.get("event_type") == "entry"]
    exits = [e for e in events if e.get("event_type") == "exit"]

    exit_by_pos: Dict[str, Dict[str, Any]] = {}
    for e in exits:
        pos_id = str(e.get("position_id", ""))
        if pos_id:
            exit_by_pos[pos_id] = e

    closed = 0
    wins = 0
    total_pnl = 0.0
    pnl_values: List[float] = []
    pnl_pct_values: List[float] = []
    hold_secs: List[float] = []
    reason_count: Dict[str, int] = defaultdict(int)

    for e in exits:
        entry_price = _to_float(e.get("entry_price")) or 0.0
        exit_price = _to_float(e.get("exit_price")) or 0.0
        size_shares = _to_float(e.get("size_shares")) or 0.0
        pnl = (exit_price - entry_price) * size_shares
        total_pnl += pnl
        pnl_values.append(pnl)
        closed += 1
        if pnl >= 0:
            wins += 1
        entry_notional = entry_price * size_shares
        if entry_notional > 0:
            pnl_pct_values.append((pnl / entry_notional) * 100)
        hold = _to_float(e.get("hold_seconds"))
        if hold is not None:
            hold_secs.append(hold)
        reason_count[str(e.get("reason", "unknown"))] += 1

    win_rate = (wins / closed * 100.0) if closed else 0.0
    avg_pnl = (sum(pnl_values) / len(pnl_values)) if pnl_values else 0.0
    avg_pnl_pct = (sum(pnl_pct_values) / len(pnl_pct_values)) if pnl_pct_values else 0.0
    median_hold = _percentile(hold_secs, 0.50) or 0.0

    # Map entry -> outcome and extract raw context for threshold recommendations.
    winning_entries: List[Dict[str, Any]] = []
    all_entries: List[Dict[str, Any]] = []
    for e in entries:
        pos_id = str(e.get("position_id", ""))
        if not pos_id:
            continue
        ctx = _parse_context(e.get("context_json"))
        row = {"entry": e, "ctx": ctx}
        all_entries.append(row)
        ex = exit_by_pos.get(pos_id)
        if ex and (_to_float(ex.get("pnl_usdc")) or 0.0) > 0:
            winning_entries.append(row)

    sample = winning_entries if winning_entries else all_entries

    flash_vals: List[float] = []
    mom_vals: List[float] = []
    ob_ratio_vals: List[float] = []
    comp_vals: List[float] = []

    for item in sample:
        ctx = item["ctx"]
        raw = ctx.get("raw", {}) if isinstance(ctx.get("raw"), dict) else {}
        gate = ctx.get("entry_gate", {}) if isinstance(ctx.get("entry_gate"), dict) else {}
        f = _to_float(raw.get("max_10s_drop"))
        m = _to_float(raw.get("momentum_30s"))
        r = _to_float(raw.get("ob_ratio"))
        c = _to_float(gate.get("composite"))
        if f is not None:
            flash_vals.append(abs(f))
        if m is not None:
            mom_vals.append(abs(m))
        if r is not None:
            ob_ratio_vals.append(r)
        if c is not None:
            comp_vals.append(c)

    rec_flash = _percentile(flash_vals, 0.70)
    rec_mom = _percentile(mom_vals, 0.70)
    rec_ratio = _percentile(ob_ratio_vals, 0.70)
    rec_base = _percentile(comp_vals, 0.35)
    rec_max = _percentile(comp_vals, 0.75)

    print("\n=== Trade Telemetry Summary ===")
    print(f"events: {len(events)}  entries: {len(entries)}  exits: {len(exits)}")
    print(f"closed trades: {closed}")
    print(f"win rate: {win_rate:.1f}%")
    print(f"total pnl: ${total_pnl:+.2f}")
    print(f"avg pnl/trade: ${avg_pnl:+.2f}")
    print(f"avg pnl % notional: {avg_pnl_pct:+.2f}%")
    print(f"median hold seconds: {median_hold:.1f}s")
    if reason_count:
        print("exit reasons:")
        for reason, n in sorted(reason_count.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  - {reason}: {n}")

    print("\n=== Suggested Threshold Tightening (from telemetry) ===")
    print("(requires context_json in logs; may be unavailable in minimal schema)")
    print(f"sample size used: {len(sample)} entries")
    if rec_flash is not None:
        print(f"flash crash drop_threshold ~ {rec_flash:.4f}")
    if rec_mom is not None:
        print(f"momentum_threshold ~ {rec_mom:.4f}")
    if rec_ratio is not None:
        print(f"orderbook raw ratio p70 ~ {rec_ratio:.3f}")
    if rec_base is not None:
        print(f"dynamic_threshold_base ~ {rec_base:.3f}")
    if rec_max is not None:
        print(f"dynamic_threshold_max ~ {max(0.0, rec_max):.3f}")
    if all(v is None for v in [rec_flash, rec_mom, rec_ratio, rec_base, rec_max]):
        print("insufficient context fields for threshold recommendations")


if __name__ == "__main__":
    main()

