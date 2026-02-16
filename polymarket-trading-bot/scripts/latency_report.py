#!/usr/bin/env python3
"""
Latency metrics report (p50/p95) from JSONL metrics.

Enable metric emission during bot runs with:
  POLY_LATENCY_METRICS=1
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def _pct(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = int(round((len(xs) - 1) * q))
    idx = max(0, min(idx, len(xs) - 1))
    return xs[idx]


def _load_events(base_dir: Path, session: Optional[str]) -> List[dict]:
    events: List[dict] = []
    files = sorted(base_dir.glob("latency_*.jsonl"))
    for path in files:
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if session and obj.get("session") != session:
                        continue
                    events.append(obj)
        except Exception:
            continue
    return events


def _summarize(events: List[dict]) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[float]] = defaultdict(list)
    for e in events:
        stage = str(e.get("stage", ""))
        try:
            ms = float(e.get("elapsed_ms", 0.0))
        except (TypeError, ValueError):
            continue
        if not stage:
            continue
        buckets[stage].append(ms)
    out: Dict[str, Dict[str, float]] = {}
    for stage, vals in buckets.items():
        out[stage] = {
            "count": float(len(vals)),
            "p50_ms": round(_pct(vals, 0.50), 4),
            "p95_ms": round(_pct(vals, 0.95), 4),
            "avg_ms": round(sum(vals) / len(vals), 4) if vals else 0.0,
        }
    return out


def _print_summary(title: str, summary: Dict[str, Dict[str, float]]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not summary:
        print("(no data)")
        return
    print(f"{'stage':26} {'count':>8} {'p50_ms':>10} {'p95_ms':>10} {'avg_ms':>10}")
    for stage in sorted(summary.keys()):
        row = summary[stage]
        print(
            f"{stage:26} {int(row['count']):>8} "
            f"{row['p50_ms']:>10.4f} {row['p95_ms']:>10.4f} {row['avg_ms']:>10.4f}"
        )


def _print_delta(base: Dict[str, Dict[str, float]], cand: Dict[str, Dict[str, float]]) -> None:
    print("\nDelta (candidate - baseline)")
    print("---------------------------")
    stages = sorted(set(base.keys()) | set(cand.keys()))
    print(f"{'stage':26} {'p50_delta':>10} {'p95_delta':>10}")
    for stage in stages:
        b = base.get(stage, {"p50_ms": 0.0, "p95_ms": 0.0})
        c = cand.get(stage, {"p50_ms": 0.0, "p95_ms": 0.0})
        print(
            f"{stage:26} "
            f"{(c['p50_ms'] - b['p50_ms']):>10.4f} "
            f"{(c['p95_ms'] - b['p95_ms']):>10.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize latency metrics JSONL")
    parser.add_argument("--dir", type=str, default="logs/latency", help="Latency metrics directory")
    parser.add_argument("--session", type=str, default="", help="Single session to summarize")
    parser.add_argument("--baseline-session", type=str, default="", help="Baseline session for comparison")
    parser.add_argument("--candidate-session", type=str, default="", help="Candidate session for comparison")
    args = parser.parse_args()

    base_dir = Path(args.dir)
    if not base_dir.exists():
        print(f"No latency directory found: {base_dir}")
        return

    if args.baseline_session and args.candidate_session:
        baseline_events = _load_events(base_dir, args.baseline_session)
        candidate_events = _load_events(base_dir, args.candidate_session)
        base_summary = _summarize(baseline_events)
        cand_summary = _summarize(candidate_events)
        _print_summary(f"Baseline session={args.baseline_session}", base_summary)
        _print_summary(f"Candidate session={args.candidate_session}", cand_summary)
        _print_delta(base_summary, cand_summary)
        return

    events = _load_events(base_dir, args.session or None)
    if args.session:
        _print_summary(f"Session={args.session}", _summarize(events))
    else:
        _print_summary("All sessions", _summarize(events))


if __name__ == "__main__":
    main()

