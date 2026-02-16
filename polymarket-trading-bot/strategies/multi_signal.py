"""
Multi-Signal Strategy - Combined Signal Trading for 5m and 15m Markets

This strategy combines four independent signal detectors and aggregates
their scores to make high-confidence entry decisions:

1. Flash Crash (Mean Reversion)
   - Detects sudden probability drops and buys the crash
   - Best in volatile/panicky markets

2. Momentum (Trend Following)
   - Detects consistent directional price movement
   - Rides trends when probability moves steadily in one direction

3. Orderbook Imbalance
   - Reads buy/sell pressure from the orderbook depth
   - Predicts short-term direction from volume asymmetry

4. Time Decay (Convergence)
   - Near expiry, binary outcomes converge to 0 or 1
   - Exploits slow-to-update prices when outcome is likely decided

Signal Combination:
    Each detector produces TradeSignal(side, raw_score, confidence, reason, source).
    Scores are weighted, decorrelated, confluence-adjusted, then compared
    against a dynamic threshold before entry/size decisions.

Usage:
    from strategies.multi_signal import MultiSignalStrategy, MultiSignalConfig

    config = MultiSignalConfig(coin="ETH", market_duration=5, ...)
    strategy = MultiSignalStrategy(bot, config)
    await strategy.run()
"""

import time
import math
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Tuple, Any

from lib.console import Colors, format_countdown
from lib.latency_metrics import record_latency
from strategies.base import BaseStrategy, StrategyConfig
from src.bot import TradingBot
from src.websocket_client import OrderbookSnapshot


# ---------------------------------------------------------------------------
# Signal data
# ---------------------------------------------------------------------------

@dataclass
class TradeSignal:
    """A directional trading signal from a detector."""

    side: str       # "up" or "down"
    raw_score: float    # 0.0 to 1.0 signal strength
    confidence: float   # 0.0 to 1.0 detector confidence
    reason: str     # Human-readable reason
    source: str     # detector source (flash|momentum|imbalance|decay)

    @property
    def score(self) -> float:
        """Composite score contribution from this signal."""
        return self.raw_score * self.confidence


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MultiSignalConfig(StrategyConfig):
    """
    Configuration for the multi-signal strategy.

    All thresholds are tuned separately for 5m vs 15m markets.
    See apps/run_5m.py and apps/run_15m.py for recommended defaults.
    """

    # --- Flash crash signal ---
    flash_crash_enabled: bool = True
    drop_threshold: float = 0.20        # Absolute probability drop to trigger
    flash_weight: float = 0.35

    # --- Momentum signal ---
    momentum_enabled: bool = True
    momentum_window: int = 30           # Seconds to measure trend
    momentum_threshold: float = 0.08    # Min absolute price move
    momentum_min_ticks: int = 4         # Min data points in window
    momentum_consistency: float = 0.65  # Min % of ticks in same direction
    momentum_weight: float = 0.20

    # --- Orderbook imbalance signal ---
    imbalance_enabled: bool = True
    imbalance_depth: int = 5                # Order book levels to consider
    imbalance_weight: float = 0.25
    imbalance_decay: float = 0.70
    imbalance_signal_threshold: float = 0.20
    imbalance_min_total_size: float = 40.0
    imbalance_max_top_level_share: float = 0.75
    spoof_min_snapshots: int = 5
    spoof_large_order_multiple: float = 3.0
    spoof_cancel_rate_threshold: float = 0.60

    # --- Time decay / convergence signal ---
    time_decay_enabled: bool = True
    time_decay_threshold_pct: float = 0.20  # Activate when this % of time remains
    time_decay_weight: float = 0.20
    time_decay_steepness_15m: float = 5.0
    time_decay_steepness_5m: float = 8.0
    time_decay_midpoint: float = 0.20
    time_decay_neutral_distance: float = 0.30
    time_decay_edge_distance: float = 0.15
    time_decay_urgency_neutral_min: float = 0.70
    time_decay_urgency_edge_min: float = 0.50
    time_decay_spot_trend_window_sec: float = 30.0
    spot_trend_provider: Optional[Callable[[float], Optional[float]]] = None
    time_decay_cross_side_support: float = 0.10

    # --- Spot divergence signal ---
    spot_divergence_enabled: bool = True
    spot_divergence_weight: float = 0.15
    spot_divergence_provider: Optional[Callable[[float], Optional[float]]] = None
    spot_divergence_window_sec: float = 30.0
    spot_divergence_min_abs: float = 0.20

    # --- Flow toxicity signal ---
    flow_toxicity_enabled: bool = True
    flow_toxicity_weight: float = 0.00
    flow_toxicity_spread_threshold: float = 0.08
    flow_toxicity_min_imbalance: float = 0.15
    flow_toxicity_strength_scale: float = 0.50
    flow_toxicity_veto_threshold: float = 0.60

    # --- Signal combination ---
    min_signal_score: float = 0.7       # Min combined score to enter
    dynamic_threshold_base: float = 0.35
    dynamic_threshold_vol_adjustment: float = 0.15
    dynamic_threshold_enabled: bool = True
    dynamic_threshold_max: float = 0.40
    dynamic_threshold_warmup_samples: int = 60
    threshold_volatility_window: float = 45.0
    volatility_history_size: int = 180
    signal_cooldown: float = 12.0       # Seconds between entries
    pair_penalty_flash_imbalance: float = 0.40
    pair_penalty_flash_momentum: float = 0.25
    pair_penalty_momentum_imbalance: float = 0.15
    pair_penalty_time_decay_any: float = 0.05
    pair_penalty_spot_time_decay: float = 0.15
    pair_penalty_flash_spot: float = 0.35
    pair_penalty_momentum_spot: float = 0.25
    pair_penalty_imbalance_spot: float = 0.20
    pair_penalty_flash_flow: float = 0.40
    pair_penalty_momentum_flow: float = 0.20
    pair_penalty_imbalance_flow: float = 0.55
    regime_low_vol_percentile: float = 0.25
    regime_high_vol_percentile: float = 0.75
    conflict_min_strength: float = 0.30
    conflict_balance_ratio: float = 0.60
    conflict_skip_enabled: bool = True

    # --- Safety ---
    min_time_remaining: int = 60        # Don't open positions with less time (seconds)
    spread_block_threshold: float = 0.08
    depth_within_1pct_min: float = 20.0
    low_activity_trade_interval_sec: float = 30.0
    low_activity_size_cap: float = 10.0
    max_depth_usage: float = 0.25

    # --- Confluence / sizing ---
    cross_timeframe_enabled: bool = True
    cross_timeframe_align_boost: float = 1.10
    cross_timeframe_conflict_penalty: float = 0.90
    toxicity_size_impact: float = 1.00
    min_size_multiplier: float = 0.30

    # --- Re-entry controls ---
    max_entries_per_window: int = 2
    reentry_cooldown_sec: float = 60.0
    reentry_loss_budget: float = 0.10
    reentry_min_improvement: float = 1.20

    # --- Optional Kelly sizing ---
    bankroll_provider: Optional[Callable[[], Optional[float]]] = None
    kelly_fraction: float = 0.25
    kelly_max_pct: float = 0.05

    # --- Diagnostics ---
    signal_state_logging_enabled: bool = False
    signal_state_log_interval_sec: float = 1.0
    incremental_signals_enabled: bool = False
    incremental_parity_logging_enabled: bool = False
    incremental_parity_interval_sec: float = 5.0


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class MultiSignalStrategy(BaseStrategy):
    """
    Multi-signal trading strategy for Polymarket Up/Down markets.

    Combines flash crash, momentum, orderbook imbalance, and time decay
    signals to make high-confidence trading decisions. Designed for both
    5-minute and 15-minute market durations with tunable parameters.
    """

    def __init__(self, bot: TradingBot, config: MultiSignalConfig):
        """Initialize multi-signal strategy."""
        super().__init__(bot, config)
        self.signal_config = config

        # Update price tracker with flash crash threshold
        self.prices.drop_threshold = config.drop_threshold

        # Signal state
        self._last_signal_time: float = 0.0
        self._last_signals: List[TradeSignal] = []
        self._total_signals_fired: int = 0
        self._volatility_samples: Deque[float] = deque(
            maxlen=max(20, self.signal_config.volatility_history_size)
        )
        self._book_snapshots: Dict[str, Deque[dict]] = {
            "up": deque(maxlen=max(10, self.signal_config.spoof_min_snapshots * 3)),
            "down": deque(maxlen=max(10, self.signal_config.spoof_min_snapshots * 3)),
        }
        self._recent_trade_times: Deque[float] = deque(maxlen=10)
        self._window_entries: Dict[str, List[dict]] = {}
        self._window_total_pnl: Dict[str, float] = {}
        self._last_signal_state_log_time: float = 0.0
        self._last_signal_state_lines: List[str] = []
        self._last_raw_signal_state: Dict[str, float] = {}
        self._last_conflict_reason: str = ""
        self._screen_initialized: bool = False
        self._last_render_line_count: int = 0
        self._last_entry_gate: Dict[str, object] = {
            "ts": 0.0,
            "side": "",
            "composite": 0.0,
            "threshold": 0.0,
            "pass": False,
        }
        self._signal_cache: Dict[str, List[TradeSignal]] = {
            "flash": [],
            "momentum": [],
            "imbalance": [],
            "decay": [],
            "spot_divergence": [],
        }
        self._last_price_fingerprint: Tuple[float, float] = (-1.0, -1.0)
        self._last_book_fingerprint: Tuple[str, str] = ("", "")
        self._last_incremental_parity_check_ts: float = 0.0

    # ------------------------------------------------------------------
    # BaseStrategy hooks
    # ------------------------------------------------------------------

    async def on_book_update(self, snapshot: OrderbookSnapshot) -> None:
        """Track update cadence for liquidity/activity estimation."""
        ts = float(snapshot.timestamp) / 1000.0 if snapshot.timestamp > 0 else time.time()
        self._recent_trade_times.append(ts)

    async def on_tick(self, prices: Dict[str, float]) -> None:
        """
        Main tick handler - collect signals and decide whether to trade.

        Called on every iteration of the strategy loop (~100ms).
        """
        if not self.positions.can_open_position:
            return

        # Cooldown check
        now = time.time()
        if now - self._last_signal_time < self.signal_config.signal_cooldown:
            return

        # Safety: don't enter too close to market expiry
        if self._is_too_close_to_expiry():
            return

        started_compute = time.perf_counter()
        signals = self._collect_signals(prices)
        record_latency(
            "signal_compute_ms",
            (time.perf_counter() - started_compute) * 1000.0,
            {"incremental": bool(self.signal_config.incremental_signals_enabled)},
        )

        self._last_signals = signals

        if not signals:
            return

        conflict, conflict_reason = self._detect_conflict(signals)
        self._last_conflict_reason = conflict_reason
        if conflict and self.signal_config.conflict_skip_enabled:
            return

        # Aggregate scores per side with decorrelation penalties
        side_scores = self._compute_composite_scores(signals)
        side_reasons: Dict[str, List[str]] = {"up": [], "down": []}

        for side in ["up", "down"]:
            side_signals = [s for s in signals if s.side == side]
            side_reasons[side] = [f"{s.source}:{s.reason}" for s in side_signals]

        # Pick best side
        best_side = max(side_scores, key=lambda s: side_scores[s])
        best_score = side_scores[best_side]
        best_score *= self._cross_timeframe_multiplier(best_side)
        min_score = self._entry_threshold()
        self._last_raw_signal_state = self._build_raw_signal_state(prices, best_side)
        self._maybe_log_signal_state(signals, side_scores, best_score, min_score)

        if self._entry_check(best_score, min_score, best_side):
            if not self._can_enter_window(now, best_score):
                return

            reasons = ", ".join(side_reasons[best_side])
            current_price = prices.get(best_side, 0)
            if current_price > 0:
                if (
                    self.signal_config.spread_block_threshold > 0
                    and self._max_spread() > self.signal_config.spread_block_threshold
                    and not any(s.source == "flash" and s.side == best_side for s in signals)
                ):
                    return

                liquidity = self._liquidity_filter(best_side, current_price)
                if not liquidity["tradeable"]:
                    return

                base_size = self.config.size
                size_cap = liquidity.get("size_cap")
                if size_cap is not None:
                    base_size = min(base_size, float(size_cap))

                toxicity = self._flow_toxicity_score(best_side, current_price)
                if (
                    self.signal_config.flow_toxicity_enabled
                    and toxicity >= self.signal_config.flow_toxicity_veto_threshold
                ):
                    return
                toxicity_mult = max(
                    self.signal_config.min_size_multiplier,
                    1.0 - (self.signal_config.toxicity_size_impact * toxicity),
                )
                kelly_mult = self._kelly_size_multiplier(best_score, min_score)
                size_to_use = max(0.0, base_size * toxicity_mult * kelly_mult)
                if size_to_use <= 0:
                    return

                success = await self._execute_buy_with_size(best_side, current_price, size_to_use)
                if success:
                    self._record_window_entry(now, best_score)
                    # Only log a signal event when it actually resulted
                    # in an executed BUY attempt/position open.
                    self.log(
                        f"SIGNAL: {best_side.upper()} score={best_score:.2f} thr={min_score:.2f} "
                        f"size=${size_to_use:.2f} [{reasons}]",
                        "trade",
                    )
                    self._last_signal_time = now
                    self._total_signals_fired += 1
        else:
            return

    # ------------------------------------------------------------------
    # Signal detectors
    # ------------------------------------------------------------------

    def _price_fingerprint(self, prices: Dict[str, float]) -> Tuple[float, float]:
        return (
            round(float(prices.get("up", 0.0)), 6),
            round(float(prices.get("down", 0.0)), 6),
        )

    def _book_fingerprint(self) -> Tuple[str, str]:
        up = self.market.get_orderbook("up")
        down = self.market.get_orderbook("down")
        up_hash = up.hash if up and up.hash else f"{up.best_bid:.4f}-{up.best_ask:.4f}" if up else ""
        down_hash = (
            down.hash if down and down.hash else f"{down.best_bid:.4f}-{down.best_ask:.4f}" if down else ""
        )
        return (up_hash, down_hash)

    def _collect_signals_full(self, prices: Dict[str, float]) -> List[TradeSignal]:
        """Compute all enabled detectors from scratch."""
        signals: List[TradeSignal] = []
        if self.signal_config.flash_crash_enabled:
            signals.extend(self._check_flash_crash(prices))
        if self.signal_config.momentum_enabled:
            signals.extend(self._check_momentum(prices))
        if self.signal_config.imbalance_enabled:
            signals.extend(self._check_orderbook_imbalance())
        if self.signal_config.time_decay_enabled:
            signals.extend(self._check_time_decay(prices))
        if self.signal_config.spot_divergence_enabled:
            signals.extend(self._check_spot_divergence(prices))
        return signals

    def _collect_signals(self, prices: Dict[str, float]) -> List[TradeSignal]:
        """
        Collect detector outputs.

        Incremental mode updates only detector groups affected by data changes.
        """
        cfg = self.signal_config
        if not cfg.incremental_signals_enabled:
            return self._collect_signals_full(prices)

        price_fp = self._price_fingerprint(prices)
        book_fp = self._book_fingerprint()
        price_changed = price_fp != self._last_price_fingerprint
        book_changed = book_fp != self._last_book_fingerprint

        if not cfg.flash_crash_enabled:
            self._signal_cache["flash"] = []
        elif price_changed:
            self._signal_cache["flash"] = self._check_flash_crash(prices)

        if not cfg.momentum_enabled:
            self._signal_cache["momentum"] = []
        elif price_changed:
            self._signal_cache["momentum"] = self._check_momentum(prices)

        if not cfg.imbalance_enabled:
            self._signal_cache["imbalance"] = []
        elif book_changed:
            self._signal_cache["imbalance"] = self._check_orderbook_imbalance()

        if not cfg.time_decay_enabled:
            self._signal_cache["decay"] = []
        elif price_changed:
            self._signal_cache["decay"] = self._check_time_decay(prices)

        if not cfg.spot_divergence_enabled:
            self._signal_cache["spot_divergence"] = []
        elif price_changed:
            self._signal_cache["spot_divergence"] = self._check_spot_divergence(prices)

        self._last_price_fingerprint = price_fp
        self._last_book_fingerprint = book_fp

        signals: List[TradeSignal] = []
        for key in ["flash", "momentum", "imbalance", "decay", "spot_divergence"]:
            signals.extend(self._signal_cache[key])

        if (
            cfg.incremental_parity_logging_enabled
            and (time.time() - self._last_incremental_parity_check_ts) >= cfg.incremental_parity_interval_sec
        ):
            self._last_incremental_parity_check_ts = time.time()
            full = self._collect_signals_full(prices)
            inc_sig = sorted((s.source, s.side, round(s.raw_score, 3), round(s.confidence, 3)) for s in signals)
            full_sig = sorted((s.source, s.side, round(s.raw_score, 3), round(s.confidence, 3)) for s in full)
            if inc_sig != full_sig:
                self.log("Incremental signal parity mismatch detected; check detector cache gating.", "warning")

        return signals

    def _check_flash_crash(self, prices: Dict[str, float]) -> List[TradeSignal]:
        """
        Flash crash detector with continuous scoring.
        """
        signals: List[TradeSignal] = []
        window = float(self.config.price_lookback_seconds)
        for side in ["up", "down"]:
            drop = self._flash_drop(side, window)
            # Continuous normalization: non-zero input produces non-zero score.
            strength = self._sigmoid_normalize(drop, self.signal_config.drop_threshold)
            confidence = min(1.0, 0.55 + 0.45 * strength)
            # Contrarian mean-reversion mapping:
            # price drop on one side votes for the opposite side.
            target_side = "down" if side == "up" else "up"
            signals.append(
                TradeSignal(
                    side=target_side,
                    raw_score=strength,
                    confidence=confidence,
                    reason=f"crash:{side}->{target_side}:{drop:.3f}",
                    source="flash",
                )
            )

        return signals

    def _check_momentum(self, prices: Dict[str, float]) -> List[TradeSignal]:
        """
        Momentum detector with canonical direction.

        We compute a single signed momentum value from the UP side when
        available (fallback to DOWN inverted), then map:
        +momentum -> UP, -momentum -> DOWN.
        """
        now = time.time()
        cutoff = now - self.signal_config.momentum_window
        up_recent = [p for p in self.prices.get_history("up") if p.timestamp >= cutoff]
        down_recent = [p for p in self.prices.get_history("down") if p.timestamp >= cutoff]

        # Prefer UP-side momentum as canonical direction; fallback to DOWN inverted.
        recent = up_recent
        source_side = "up"
        invert_sign = 1.0
        if len(recent) < self.signal_config.momentum_min_ticks and len(down_recent) >= self.signal_config.momentum_min_ticks:
            recent = down_recent
            source_side = "down"
            invert_sign = -1.0
        if len(recent) < self.signal_config.momentum_min_ticks:
            return []

        raw_move = recent[-1].price - recent[0].price
        canonical_move = raw_move * invert_sign
        if abs(canonical_move) <= 1e-9:
            return []

        up_ticks = 0
        down_ticks = 0
        for i in range(1, len(recent)):
            diff = (recent[i].price - recent[i - 1].price) * invert_sign
            if diff > 0.001:
                up_ticks += 1
            elif diff < -0.001:
                down_ticks += 1
        total_ticks = up_ticks + down_ticks
        if total_ticks == 0:
            return []

        move_strength = self._sigmoid_normalize(
            abs(canonical_move),
            self.signal_config.momentum_threshold,
        )
        consistency = (up_ticks / total_ticks) if canonical_move > 0 else (down_ticks / total_ticks)
        consistency_strength = self._sigmoid_normalize(
            consistency,
            self.signal_config.momentum_consistency,
        )
        confidence = min(1.0, max(0.0, consistency_strength))
        target_side = "up" if canonical_move > 0 else "down"

        return [
            TradeSignal(
                side=target_side,
                raw_score=move_strength,
                confidence=confidence,
                reason=f"mom:{source_side}:{canonical_move:+.3f}",
                source="momentum",
            )
        ]

    def _check_orderbook_imbalance(self) -> List[TradeSignal]:
        """
        Depth-weighted imbalance with spoof-risk suppression.
        """
        signals: List[TradeSignal] = []
        depth = self.signal_config.imbalance_depth

        for side in ["up", "down"]:
            ob = self.market.get_orderbook(side)
            if not ob or not ob.bids or not ob.asks:
                continue

            bid_levels = ob.bids[:depth]
            ask_levels = ob.asks[:depth]
            total_bid_size = sum(level.size for level in bid_levels)
            total_ask_size = sum(level.size for level in ask_levels)

            if total_ask_size <= 0:
                continue
            if (total_bid_size + total_ask_size) < self.signal_config.imbalance_min_total_size:
                continue

            top_bid_share = bid_levels[0].size / max(total_bid_size, 1e-9)
            top_ask_share = ask_levels[0].size / max(total_ask_size, 1e-9)
            if (
                top_bid_share > self.signal_config.imbalance_max_top_level_share
                or top_ask_share > self.signal_config.imbalance_max_top_level_share
            ):
                continue

            imbalance = self._depth_weighted_imbalance(bid_levels, ask_levels, depth)

            self._record_book_snapshot(side, bid_levels, ask_levels)
            if self._detect_spoof(side):
                continue

            normalized = self._sigmoid_normalize(
                max(0.0, imbalance),
                self.signal_config.imbalance_signal_threshold,
            )
            confidence = min(1.0, 0.55 + 0.45 * normalized)
            signals.append(TradeSignal(
                side=side,
                raw_score=normalized,
                confidence=confidence,
                reason=f"imbal:{imbalance:+.2f}",
                source="imbalance",
            ))

        return signals

    def _check_time_decay(self, prices: Dict[str, float]) -> List[TradeSignal]:
        """
        Time Decay v2 with sigmoid urgency and optional spot-trend tie-break.
        """
        signals: List[TradeSignal] = []
        market = self.current_market
        if not market:
            return signals

        mins, secs = market.get_countdown()
        if mins < 0:
            return signals

        total_secs_remaining = mins * 60 + secs
        market_duration_secs = self.signal_config.market_duration * 60

        if market_duration_secs <= 0:
            return signals

        time_pct_remaining = total_secs_remaining / market_duration_secs

        # Only activate when time remaining falls below threshold
        if time_pct_remaining > self.signal_config.time_decay_threshold_pct:
            return signals

        spot_trend = self._spot_trend()

        for side in ["up", "down"]:
            price = prices.get(side, 0)
            sig = self._time_decay_signal_for_side(
                side=side,
                price=price,
                time_remaining_sec=float(total_secs_remaining),
                total_window_sec=float(market_duration_secs),
                spot_trend=spot_trend,
            )
            if sig:
                signals.append(sig)

        return signals

    def _check_spot_divergence(self, prices: Dict[str, float]) -> List[TradeSignal]:
        """Spot-divergence detector using optional provider."""
        provider = self.signal_config.spot_divergence_provider
        if not provider:
            return []
        try:
            divergence = provider(self.signal_config.spot_divergence_window_sec)
        except Exception:
            return []
        if divergence is None:
            return []
        value = float(divergence)
        if abs(value) < self.signal_config.spot_divergence_min_abs:
            return []

        side = "up" if value > 0 else "down"
        raw = min(1.0, abs(value))
        confidence = min(1.0, 0.50 + 0.50 * raw)
        return [
            TradeSignal(
                side=side,
                raw_score=raw,
                confidence=confidence,
                reason=f"spot_div:{value:+.2f}",
                source="spot_divergence",
            )
        ]

    def _check_flow_toxicity(self, prices: Dict[str, float]) -> List[TradeSignal]:
        """
        Flow-toxicity directional signal.

        High toxicity contributes a weak directional signal and primarily
        acts later as a position-size reducer.
        """
        out: List[TradeSignal] = []
        for side in ["up", "down"]:
            p = prices.get(side, 0.0)
            if p <= 0:
                continue
            tox = self._flow_toxicity_score(side, p)
            if tox <= 0:
                continue
            out.append(
                TradeSignal(
                    side=side,
                    raw_score=min(1.0, tox),
                    confidence=min(1.0, 0.40 + 0.60 * tox),
                    reason=f"flow_tox:{tox:.2f}",
                    source="flow_toxicity",
                )
            )
        return out

    def _detect_conflict(self, signals: List[TradeSignal]) -> tuple[bool, str]:
        """
        Detect balanced up/down pressure conflict and optionally skip entries.
        """
        up_strength = 0.0
        down_strength = 0.0
        for s in signals:
            eff = s.raw_score * s.confidence * self._source_weight(s.source)
            if s.side == "up":
                up_strength += eff
            elif s.side == "down":
                down_strength += eff

        if up_strength <= 0.0 and down_strength <= 0.0:
            return (False, "")
        if (
            up_strength > self.signal_config.conflict_min_strength
            and down_strength > self.signal_config.conflict_min_strength
        ):
            ratio = min(up_strength, down_strength) / max(up_strength, down_strength)
            if ratio > self.signal_config.conflict_balance_ratio:
                return (
                    True,
                    f"conflict up={up_strength:.3f} down={down_strength:.3f} ratio={ratio:.2f}",
                )
        return (False, "")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid_normalize(raw_value: float, soft_threshold: float) -> float:
        """
        Sigmoid normalization that remains continuous around threshold.
        """
        t = max(soft_threshold, 1e-9)
        x = (raw_value / t) - 1.0
        val = 1.0 / (1.0 + math.exp(-4.0 * x))
        return max(0.0, min(1.0, val))

    def _flash_drop(self, side: str, window_sec: float) -> float:
        """
        Max drop over a lookback window for one side.
        """
        history = self.prices.get_history(side)
        if len(history) < 2:
            return 0.0
        cutoff = time.time() - window_sec
        recent = [p for p in history if p.timestamp >= cutoff]
        if len(recent) < 2:
            return 0.0
        current = recent[-1].price
        peak = max(p.price for p in recent)
        return max(0.0, peak - current)

    def _orderbook_bid_ask_ratio(self, side: str, levels: int) -> float:
        """
        Weighted bid/ask pressure ratio for diagnostics.
        """
        ob = self.market.get_orderbook(side)
        if not ob or not ob.bids or not ob.asks:
            return 0.0
        decay = self.signal_config.imbalance_decay
        bid_pressure = sum(level.size * (decay ** i) for i, level in enumerate(ob.bids[:levels]))
        ask_pressure = sum(level.size * (decay ** i) for i, level in enumerate(ob.asks[:levels]))
        if ask_pressure <= 0:
            return 0.0
        return bid_pressure / ask_pressure

    def _momentum_value(self, side: str, window_sec: float) -> float:
        """Return signed momentum over window for diagnostics."""
        history = self.prices.get_history(side)
        if len(history) < 2:
            return 0.0
        cutoff = time.time() - window_sec
        recent = [p for p in history if p.timestamp >= cutoff]
        if len(recent) < 2:
            return 0.0
        return recent[-1].price - recent[0].price

    def _build_raw_signal_state(self, prices: Dict[str, float], best_side: str) -> Dict[str, float]:
        """
        Build raw-feature snapshot for diagnostics.
        """
        max_drop = max(
            self._flash_drop("up", 10.0),
            self._flash_drop("down", 10.0),
        )
        momentum_up_30s = self._momentum_value("up", 30.0)
        momentum_down_30s = self._momentum_value("down", 30.0)
        # Canonical momentum: UP-side move (fallback to inverted DOWN-side move).
        momentum_30s = (
            momentum_up_30s
            if abs(momentum_up_30s) > 1e-9
            else (-momentum_down_30s)
        )
        ob_ratio = max(
            self._orderbook_bid_ask_ratio("up", self.signal_config.imbalance_depth),
            self._orderbook_bid_ask_ratio("down", self.signal_config.imbalance_depth),
        )
        market = self.current_market
        time_remaining = 0.0
        if market:
            mins, secs = market.get_countdown()
            if mins >= 0:
                time_remaining = float(mins * 60 + secs)
        current_prob = prices.get(best_side, max(prices.values()) if prices else 0.0)
        vpin_proxy = max(
            self._flow_toxicity_score("up", prices.get("up", 0.0)),
            self._flow_toxicity_score("down", prices.get("down", 0.0)),
        )
        return {
            "max_10s_drop": max_drop,
            "momentum_30s": momentum_30s,
            "momentum_up_30s": momentum_up_30s,
            "momentum_down_30s": momentum_down_30s,
            "ob_ratio": ob_ratio,
            "time_remaining_s": time_remaining,
            "current_prob": current_prob,
            "vpin_proxy": vpin_proxy,
        }

    def _get_trade_context(self) -> Dict[str, object]:
        """
        Attach strategy-specific context to telemetry events.
        """
        signal_map: Dict[str, float] = {}
        for sig in self._last_signals:
            key = f"{sig.source}_{sig.side}"
            signal_map[key] = max(signal_map.get(key, 0.0), sig.score)

        gate = {
            "side": str(self._last_entry_gate.get("side", "")),
            "composite": float(self._last_entry_gate.get("composite", 0.0)),
            "threshold": float(self._last_entry_gate.get("threshold", 0.0)),
            "pass": bool(self._last_entry_gate.get("pass", False)),
        }

        return {
            "signals": signal_map,
            "raw": dict(self._last_raw_signal_state),
            "entry_gate": gate,
            "conflict": self._last_conflict_reason,
            "detectors_enabled": {
                "flash_crash": self.signal_config.flash_crash_enabled,
                "momentum": self.signal_config.momentum_enabled,
                "imbalance": self.signal_config.imbalance_enabled,
                "time_decay": self.signal_config.time_decay_enabled,
                "spot_divergence": self.signal_config.spot_divergence_enabled,
                "flow_toxicity": self.signal_config.flow_toxicity_enabled,
            },
        }

    def _source_weight(self, source: str) -> float:
        cfg = self.signal_config
        weights = {
            "flash": cfg.flash_weight,
            "momentum": cfg.momentum_weight,
            "imbalance": cfg.imbalance_weight,
            "decay": cfg.time_decay_weight,
            "spot_divergence": cfg.spot_divergence_weight,
            "flow_toxicity": cfg.flow_toxicity_weight,
        }
        base = weights.get(source, 1.0)
        p = self._current_vol_percentile()
        if p >= cfg.regime_high_vol_percentile:
            mult = {
                "flash": 1.05,
                "momentum": 0.95,
                "imbalance": 0.90,
                "decay": 1.10,
                "spot_divergence": 1.00,
                "flow_toxicity": 1.05,
            }.get(source, 1.0)
            return base * mult
        if p <= cfg.regime_low_vol_percentile:
            mult = {
                "flash": 0.95,
                "momentum": 1.00,
                "imbalance": 1.00,
                "decay": 1.05,
                "spot_divergence": 1.00,
                "flow_toxicity": 0.95,
            }.get(source, 1.0)
            return base * mult
        return base

    def _current_vol_percentile(self) -> float:
        """Estimate current volatility percentile from rolling samples."""
        cfg = self.signal_config
        up_vol = self.prices.get_volatility("up", cfg.threshold_volatility_window)
        down_vol = self.prices.get_volatility("down", cfg.threshold_volatility_window)
        current_vol = max(up_vol, down_vol)
        if not self._volatility_samples:
            return 0.5
        count_le = sum(1 for v in self._volatility_samples if v <= current_vol)
        return count_le / len(self._volatility_samples)

    def _pair_penalty(self, s1: str, s2: str) -> float:
        """
        Return configured correlation penalty for a signal-source pair.
        """
        pair = tuple(sorted((s1, s2)))
        cfg = self.signal_config
        direct: Dict[Tuple[str, str], float] = {
            ("flash", "imbalance"): cfg.pair_penalty_flash_imbalance,
            ("flash", "momentum"): cfg.pair_penalty_flash_momentum,
            ("imbalance", "momentum"): cfg.pair_penalty_momentum_imbalance,
            ("decay", "spot_divergence"): cfg.pair_penalty_spot_time_decay,
            ("flash", "spot_divergence"): cfg.pair_penalty_flash_spot,
            ("momentum", "spot_divergence"): cfg.pair_penalty_momentum_spot,
            ("imbalance", "spot_divergence"): cfg.pair_penalty_imbalance_spot,
            ("flash", "flow_toxicity"): cfg.pair_penalty_flash_flow,
            ("momentum", "flow_toxicity"): cfg.pair_penalty_momentum_flow,
            ("flow_toxicity", "imbalance"): cfg.pair_penalty_imbalance_flow,
        }
        if pair in direct:
            return direct[pair]
        if "decay" in pair:
            return cfg.pair_penalty_time_decay_any
        return 0.0

    def _compute_composite_scores(self, signals: List[TradeSignal]) -> Dict[str, float]:
        """
        Weighted sum with pairwise correlation penalties.
        """
        out = {"up": 0.0, "down": 0.0}
        for direction in ["up", "down"]:
            dir_signals = [s for s in signals if s.side == direction]
            if not dir_signals:
                continue

            # Compute effective contribution first (raw * confidence * weight).
            # Correlation penalties should be applied on effective overlap,
            # otherwise a zero-weight source can still suppress the score.
            effective_scores = [
                s.raw_score * s.confidence * self._source_weight(s.source)
                for s in dir_signals
            ]
            raw_weighted = sum(effective_scores)
            penalty = 0.0
            for i, s1 in enumerate(dir_signals):
                for j, s2 in enumerate(dir_signals[i + 1:], start=i + 1):
                    pair_pen = self._pair_penalty(s1.source, s2.source)
                    if pair_pen <= 0:
                        continue
                    overlap = min(effective_scores[i], effective_scores[j])
                    penalty += pair_pen * overlap

            out[direction] = max(0.0, raw_weighted - penalty)
        return out

    def _depth_weighted_imbalance(self, bids, asks, levels: int) -> float:
        """
        Exponentially weighted imbalance in [-1, 1].
        """
        decay = self.signal_config.imbalance_decay
        bid_pressure = sum(level.size * (decay ** i) for i, level in enumerate(bids[:levels]))
        ask_pressure = sum(level.size * (decay ** i) for i, level in enumerate(asks[:levels]))
        total = bid_pressure + ask_pressure
        if total <= 0:
            return 0.0
        return (bid_pressure - ask_pressure) / total

    @staticmethod
    def _median(values: List[float]) -> float:
        if not values:
            return 0.0
        vals = sorted(values)
        mid = len(vals) // 2
        if len(vals) % 2 == 1:
            return vals[mid]
        return (vals[mid - 1] + vals[mid]) / 2.0

    def _record_book_snapshot(self, side: str, bids, asks) -> None:
        """Store lightweight snapshots for spoof-risk estimation."""
        bid_sizes = [level.size for level in bids]
        ask_sizes = [level.size for level in asks]
        median_size = self._median(bid_sizes + ask_sizes)
        snap = {
            "bids": [(f"{level.price:.4f}", level.size) for level in bids],
            "median_size": median_size,
        }
        self._book_snapshots[side].append(snap)

    def _detect_spoof(self, side: str) -> bool:
        """
        Spoof detector proxy using large bid-level vanish rate.
        """
        snaps = list(self._book_snapshots[side])
        min_snaps = self.signal_config.spoof_min_snapshots
        if len(snaps) < min_snaps:
            return False
        recent = snaps[-min_snaps:]
        large_orders_seen: set[str] = set()
        large_orders_cancelled: set[str] = set()
        multiple = self.signal_config.spoof_large_order_multiple

        for snap in recent:
            active_ids = {order_id for order_id, _ in snap["bids"]}
            for order_id, size in snap["bids"]:
                if snap["median_size"] > 0 and size > snap["median_size"] * multiple:
                    large_orders_seen.add(order_id)
            for oid in large_orders_seen:
                if oid not in active_ids:
                    large_orders_cancelled.add(oid)

        if not large_orders_seen:
            return False
        cancel_rate = len(large_orders_cancelled) / len(large_orders_seen)
        return cancel_rate > self.signal_config.spoof_cancel_rate_threshold

    def _time_decay_steepness(self) -> float:
        """Pick a default steepness by market duration."""
        if self.signal_config.market_duration <= 5:
            return self.signal_config.time_decay_steepness_5m
        return self.signal_config.time_decay_steepness_15m

    def _spot_trend(self) -> float:
        """Return BTC spot trend proxy; defaults to 0 when unavailable."""
        provider = self.signal_config.spot_trend_provider
        if not provider:
            return 0.0
        try:
            value = provider(self.signal_config.time_decay_spot_trend_window_sec)
            return float(value) if value is not None else 0.0
        except Exception:
            return 0.0

    def _time_decay_signal_for_side(
        self,
        side: str,
        price: float,
        time_remaining_sec: float,
        total_window_sec: float,
        spot_trend: float,
    ) -> Optional[TradeSignal]:
        """Nonlinear convergence score with optional spot trend directionality."""
        if price <= 0 or total_window_sec <= 0:
            return None

        t_frac = time_remaining_sec / total_window_sec
        urgency = 1.0 / (
            1.0 + math.exp(self._time_decay_steepness() * (t_frac - self.signal_config.time_decay_midpoint))
        )
        distance_from_edge = min(price, 1.0 - price)

        if (
            distance_from_edge > self.signal_config.time_decay_neutral_distance
            and urgency > self.signal_config.time_decay_urgency_neutral_min
        ):
            if spot_trend == 0:
                return None
            expected_side = "up" if spot_trend > 0 else "down"
            if expected_side != side:
                return None
            score = urgency * min(1.0, distance_from_edge / 0.5)
            return TradeSignal(
                side=side,
                raw_score=min(score, 1.0),
                confidence=min(1.0, 0.50 + 0.50 * urgency),
                reason=f"decay:neutral@{time_remaining_sec:.0f}s u={urgency:.2f}",
                source="decay",
            )

        if (
            distance_from_edge < self.signal_config.time_decay_edge_distance
            and urgency > self.signal_config.time_decay_urgency_edge_min
        ):
            expected_side = "up" if price > 0.5 else "down"
            if expected_side != side:
                return None
            score = urgency * (1.0 - distance_from_edge / self.signal_config.time_decay_edge_distance) * 0.6
            return TradeSignal(
                side=side,
                raw_score=min(max(score, 0.0), 1.0),
                confidence=min(1.0, 0.55 + 0.45 * urgency),
                reason=f"decay:edge@{time_remaining_sec:.0f}s u={urgency:.2f}",
                source="decay",
            )

        return None

    def _current_slug(self) -> str:
        market = self.current_market
        return market.slug if market else ""

    def _cross_timeframe_multiplier(self, side: str) -> float:
        """
        Cross-timeframe confluence proxy from short vs base momentum windows.
        """
        if not self.signal_config.cross_timeframe_enabled:
            return 1.0
        base_window = max(10, self.signal_config.momentum_window)
        short_window = max(5, int(base_window / 3))
        short_dir = self._momentum_direction(side, short_window)
        base_dir = self._momentum_direction(side, base_window)
        if short_dir == 0 or base_dir == 0:
            return 1.0
        if short_dir == base_dir:
            return self.signal_config.cross_timeframe_align_boost
        return self.signal_config.cross_timeframe_conflict_penalty

    def _momentum_direction(self, side: str, window_sec: int) -> int:
        """
        Return +1, -1, or 0 momentum direction for a side.
        """
        now = time.time()
        history = self.prices.get_history(side)
        recent = [p for p in history if p.timestamp >= (now - window_sec)]
        if len(recent) < 2:
            return 0
        move = recent[-1].price - recent[0].price
        if move > 0.002:
            return 1
        if move < -0.002:
            return -1
        return 0

    def _max_spread(self) -> float:
        return max(self.market.get_spread("up"), self.market.get_spread("down"))

    def _depth_within_pct(self, side: str, pct: float = 0.01) -> float:
        """
        Sum size within +/- pct from mid price.
        """
        ob = self.market.get_orderbook(side)
        if not ob:
            return 0.0
        mid = ob.mid_price
        if mid <= 0:
            return 0.0
        low = mid * (1.0 - pct)
        high = mid * (1.0 + pct)
        size = 0.0
        for level in ob.bids:
            if level.price >= low:
                size += level.size
        for level in ob.asks:
            if level.price <= high:
                size += level.size
        return size

    def _avg_trade_interval(self) -> float:
        if len(self._recent_trade_times) < 2:
            return 999.0
        times = list(self._recent_trade_times)
        deltas = [times[i] - times[i - 1] for i in range(1, len(times))]
        if not deltas:
            return 999.0
        return sum(deltas) / len(deltas)

    def _liquidity_filter(self, side: str, current_price: float) -> Dict[str, object]:
        """
        Liquidity gate + size cap recommendation.
        """
        spread = self.market.get_spread(side)
        depth_1pct = self._depth_within_pct(side, 0.01)
        avg_interval = self._avg_trade_interval()

        if spread > self.signal_config.spread_block_threshold:
            return {"tradeable": False, "reason": "spread_too_wide"}

        if depth_1pct < self.signal_config.depth_within_1pct_min:
            size_cap = min(5.0, depth_1pct * self.signal_config.max_depth_usage)
            size_cap = min(size_cap, self.config.size)
            return {
                "tradeable": size_cap > 0,
                "size_cap": size_cap,
                "reason": "thin_book_small_size",
            }

        if avg_interval > self.signal_config.low_activity_trade_interval_sec:
            return {
                "tradeable": True,
                "size_cap": min(self.config.size, self.signal_config.low_activity_size_cap),
                "reason": "low_activity",
                "slippage_warning": True,
            }

        return {"tradeable": True, "size_cap": None}

    def _flow_toxicity_score(self, side: str, current_price: float) -> float:
        """
        Side-level toxicity proxy in [0, 1].
        """
        ob = self.market.get_orderbook(side)
        if not ob or not ob.bids or not ob.asks or current_price <= 0:
            return 0.0
        spread = self.market.get_spread(side)
        depth = self.signal_config.imbalance_depth
        imbalance = self._depth_weighted_imbalance(ob.bids[:depth], ob.asks[:depth], depth)
        spread_term = max(0.0, spread - self.signal_config.flow_toxicity_spread_threshold) / max(
            1e-9, self.signal_config.flow_toxicity_strength_scale
        )
        imbalance_term = max(0.0, abs(imbalance) - self.signal_config.flow_toxicity_min_imbalance)
        spoof_term = 0.35 if self._detect_spoof(side) else 0.0
        return min(1.0, spread_term + imbalance_term + spoof_term)

    def _kelly_size_multiplier(self, score: float, threshold: float) -> float:
        """
        Optional Kelly fraction; defaults to 1.0 if bankroll missing.
        """
        provider = self.signal_config.bankroll_provider
        if not provider:
            return 1.0
        bankroll = provider()
        if not bankroll or bankroll <= 0:
            return 1.0

        # Map signal score margin to win probability estimate.
        edge = max(0.0, score - threshold)
        win_prob = min(0.85, 0.50 + edge * 0.35)
        win_return = max(1e-9, self.config.take_profit)
        loss_return = -max(1e-9, self.config.stop_loss)
        kelly_pct = self._kelly_fraction(
            win_prob=win_prob,
            win_return=win_return,
            loss_return=loss_return,
            fraction=self.signal_config.kelly_fraction,
            max_pct=self.signal_config.kelly_max_pct,
        )
        if kelly_pct <= 0:
            return 0.0
        target_usdc = bankroll * kelly_pct
        return min(1.0, target_usdc / max(self.config.size, 1e-9))

    @staticmethod
    def _kelly_fraction(
        win_prob: float,
        win_return: float,
        loss_return: float,
        fraction: float,
        max_pct: float,
    ) -> float:
        b = win_return / max(abs(loss_return), 1e-9)
        q = 1.0 - win_prob
        full_kelly = (win_prob * b - q) / max(b, 1e-9)
        if full_kelly <= 0:
            return 0.0
        return min(full_kelly * fraction, max_pct)

    async def _execute_buy_with_size(self, side: str, current_price: float, size_usdc: float) -> bool:
        """
        Execute buy with temporary size override.
        """
        gate_ts = float(self._last_entry_gate.get("ts", 0.0))
        gate_side = str(self._last_entry_gate.get("side", ""))
        gate_pass = bool(self._last_entry_gate.get("pass", False))
        if (not gate_pass) or gate_side != side or (time.time() - gate_ts) > 2.0:
            return False

        old_size = self.config.size
        self.config.size = max(0.0, float(size_usdc))
        try:
            return await self.execute_buy(side, current_price)
        finally:
            self.config.size = old_size

    def _entry_check(self, composite: float, threshold: float, side: str) -> bool:
        """
        Single source of truth for threshold gating.
        """
        passed = composite >= threshold
        self._last_entry_gate = {
            "ts": time.time(),
            "side": side,
            "composite": composite,
            "threshold": threshold,
            "pass": passed,
        }
        return passed

    def _can_enter_window(self, now: float, signal_score: float) -> bool:
        """
        Conditional re-entry manager (max entries, cooldown, loss budget, stronger signal).
        """
        slug = self._current_slug()
        if not slug:
            return False
        entries = self._window_entries.setdefault(slug, [])
        total_pnl = self._window_total_pnl.get(slug, 0.0)

        if len(entries) >= self.signal_config.max_entries_per_window:
            return False
        if total_pnl < -self.signal_config.reentry_loss_budget:
            return False
        if entries:
            last_exit = entries[-1].get("exit_time")
            if last_exit and (now - float(last_exit) < self.signal_config.reentry_cooldown_sec):
                return False
            last_score = float(entries[-1].get("signal_score", 0.0))
            if signal_score <= (last_score * self.signal_config.reentry_min_improvement):
                return False

        # Allow re-entry by lifting base one-and-done lock for this slug.
        if slug in self._completed_market_slugs:
            self._completed_market_slugs.discard(slug)
        return True

    def _record_window_entry(self, now: float, signal_score: float) -> None:
        slug = self._current_slug()
        if not slug:
            return
        entries = self._window_entries.setdefault(slug, [])
        entries.append(
            {
                "entry_time": now,
                "exit_time": None,
                "signal_score": signal_score,
                "pnl": None,
            }
        )

    def _entry_threshold(self, update_cache: bool = True) -> float:
        """Regime-aware threshold using volatility percentile."""
        cfg = self.signal_config
        if not cfg.dynamic_threshold_enabled:
            return cfg.min_signal_score

        up_vol = self.prices.get_volatility("up", cfg.threshold_volatility_window)
        down_vol = self.prices.get_volatility("down", cfg.threshold_volatility_window)
        current_vol = max(up_vol, down_vol)
        if update_cache:
            self._volatility_samples.append(current_vol)
        if not self._volatility_samples:
            return cfg.dynamic_threshold_base
        if len(self._volatility_samples) < cfg.dynamic_threshold_warmup_samples:
            return cfg.dynamic_threshold_base

        count_le = sum(1 for v in self._volatility_samples if v <= current_vol)
        volatility_percentile = count_le / len(self._volatility_samples)
        dynamic = cfg.dynamic_threshold_base + (
            cfg.dynamic_threshold_vol_adjustment * volatility_percentile
        )
        return min(cfg.dynamic_threshold_max, dynamic)

    def _maybe_log_signal_state(
        self,
        signals: List[TradeSignal],
        side_scores: Dict[str, float],
        composite: float,
        threshold: float,
    ) -> None:
        """Capture score diagnostics and render in persistent TUI section."""
        cfg = self.signal_config
        if not cfg.signal_state_logging_enabled:
            return
        now = time.time()
        if now - self._last_signal_state_log_time < cfg.signal_state_log_interval_sec:
            return
        self._last_signal_state_log_time = now

        lines: List[str] = [
            f"[{time.strftime('%H:%M:%S')}] Composite: {composite:.3f} "
            f"(threshold: {threshold:.3f}) "
            f"[UP={side_scores.get('up', 0):.3f} DOWN={side_scores.get('down', 0):.3f}]"
        ]
        per_signal: Dict[str, float] = {}
        for s in signals:
            key = f"{s.source}_{s.side}"
            per_signal[key] = max(per_signal.get(key, 0.0), s.score)
        for name in sorted(per_signal.keys()):
            score = per_signal[name]
            bar = "#" * int(max(0.0, min(1.0, score)) * 20)
            lines.append(f"  {name:20s}: {score:.3f} {bar}")
        raw = self._last_raw_signal_state
        if raw:
            lines.append(
                f"  [raw] 10s price drop   : {raw.get('max_10s_drop', 0.0):.4f}"
            )
            lines.append(
                f"  [raw] 30s momentum     : {raw.get('momentum_30s', 0.0):.4f}"
            )
            lines.append(
                f"  [raw] 30s mom up/down  : {raw.get('momentum_up_30s', 0.0):+.4f} / "
                f"{raw.get('momentum_down_30s', 0.0):+.4f}"
            )
            lines.append(
                f"  [raw] OB bid/ask ratio : {raw.get('ob_ratio', 0.0):.2f}"
            )
            lines.append(
                f"  [raw] time remaining   : {raw.get('time_remaining_s', 0.0):.0f}s"
            )
            lines.append(
                f"  [raw] current prob     : {raw.get('current_prob', 0.0):.3f}"
            )
            lines.append(
                f"  [raw] VPIN proxy       : {raw.get('vpin_proxy', 0.0):.3f}"
            )
        self._last_signal_state_lines = lines

    def _is_too_close_to_expiry(self) -> bool:
        """Check if market is too close to expiry for new entries."""
        market = self.current_market
        if not market:
            return True

        mins, secs = market.get_countdown()
        if mins < 0:
            return True

        total_secs = mins * 60 + secs
        # Never open a new position inside the late "hold-to-expiry" zone.
        # This prevents late entries from being immediately mercy-stopped.
        min_entry_secs = max(
            int(self.signal_config.min_time_remaining),
            int(self._late_phase_cutoff_secs()),
        )
        return total_secs < min_entry_secs

    async def execute_sell(
        self,
        position,
        current_price: float,
        exit_reason: Optional[str] = None,
        sl_anchor_price: Optional[float] = None,
    ) -> bool:
        """
        Extend base sell to track re-entry budget state.
        """
        pnl = position.get_pnl(current_price)
        success = await super().execute_sell(
            position,
            current_price,
            exit_reason=exit_reason,
            sl_anchor_price=sl_anchor_price,
        )
        if success:
            slug = self._current_slug()
            if slug:
                entries = self._window_entries.setdefault(slug, [])
                if entries and entries[-1].get("exit_time") is None:
                    entries[-1]["exit_time"] = time.time()
                    entries[-1]["pnl"] = pnl
                self._window_total_pnl[slug] = self._window_total_pnl.get(slug, 0.0) + pnl
        return success

    def _get_countdown_str(self) -> str:
        """Get formatted countdown string."""
        market = self.current_market
        if not market:
            return "--:--"
        mins, secs = market.get_countdown()
        return format_countdown(mins, secs)

    def on_market_change(self, old_slug: str, new_slug: str) -> None:
        """Handle market change - reset signal state."""
        self.prices.clear()
        self._last_signal_time = 0.0
        self._last_signals = []
        self._volatility_samples.clear()
        self._book_snapshots["up"].clear()
        self._book_snapshots["down"].clear()
        self._recent_trade_times.clear()
        self._window_entries.clear()
        self._window_total_pnl.clear()
        self._last_signal_state_lines = []
        self._last_raw_signal_state = {}
        self._last_conflict_reason = ""
        self._screen_initialized = False
        self._last_render_line_count = 0
        self._last_entry_gate = {
            "ts": 0.0,
            "side": "",
            "composite": 0.0,
            "threshold": 0.0,
            "pass": False,
        }
        self._signal_cache = {
            "flash": [],
            "momentum": [],
            "imbalance": [],
            "decay": [],
            "spot_divergence": [],
        }
        self._last_price_fingerprint = (-1.0, -1.0)
        self._last_book_fingerprint = ("", "")
        self._last_incremental_parity_check_ts = 0.0

    # ------------------------------------------------------------------
    # TUI Rendering
    # ------------------------------------------------------------------

    def render_status(self, prices: Dict[str, float]) -> None:
        """Render TUI status display with signal information."""
        lines: List[str] = []
        cfg = self.signal_config
        duration_label = f"{cfg.market_duration}m"

        # Header
        ws_status = (
            f"{Colors.GREEN}WS{Colors.RESET}"
            if self.is_connected
            else f"{Colors.RED}REST{Colors.RESET}"
        )
        countdown = self._get_countdown_str()
        stats = self.positions.get_stats()

        lines.append(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        lines.append(
            f"{Colors.CYAN}[{cfg.coin} {duration_label}]{Colors.RESET} "
            f"[{ws_status}] "
            f"Ends: {countdown} | "
            f"Trades: {stats['trades_closed']} | "
            f"PnL: ${stats['total_pnl']:+.2f} | "
            f"Signals: {self._total_signals_fired}"
        )
        lines.append(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")

        # Orderbook display
        up_ob = self.market.get_orderbook("up")
        down_ob = self.market.get_orderbook("down")

        lines.append(
            f"{Colors.GREEN}{'UP':^39}{Colors.RESET}|"
            f"{Colors.RED}{'DOWN':^39}{Colors.RESET}"
        )
        lines.append(
            f"{'Bid':>9} {'Size':>9} | {'Ask':>9} {'Size':>9}|"
            f"{'Bid':>9} {'Size':>9} | {'Ask':>9} {'Size':>9}"
        )
        lines.append("-" * 80)

        up_bids = up_ob.bids[:5] if up_ob else []
        up_asks = up_ob.asks[:5] if up_ob else []
        down_bids = down_ob.bids[:5] if down_ob else []
        down_asks = down_ob.asks[:5] if down_ob else []

        for i in range(5):
            ub = (
                f"{up_bids[i].price:>9.4f} {up_bids[i].size:>9.1f}"
                if i < len(up_bids) else f"{'--':>9} {'--':>9}"
            )
            ua = (
                f"{up_asks[i].price:>9.4f} {up_asks[i].size:>9.1f}"
                if i < len(up_asks) else f"{'--':>9} {'--':>9}"
            )
            db = (
                f"{down_bids[i].price:>9.4f} {down_bids[i].size:>9.1f}"
                if i < len(down_bids) else f"{'--':>9} {'--':>9}"
            )
            da = (
                f"{down_asks[i].price:>9.4f} {down_asks[i].size:>9.1f}"
                if i < len(down_asks) else f"{'--':>9} {'--':>9}"
            )
            lines.append(f"{ub} | {ua}|{db} | {da}")

        lines.append("-" * 80)

        # Price summary
        up_mid = up_ob.mid_price if up_ob else prices.get("up", 0)
        down_mid = down_ob.mid_price if down_ob else prices.get("down", 0)
        up_spread = self.market.get_spread("up")
        down_spread = self.market.get_spread("down")

        lines.append(
            f"Mid: {Colors.GREEN}{up_mid:.4f}{Colors.RESET}  "
            f"Spread: {up_spread:.4f}           |"
            f"Mid: {Colors.RED}{down_mid:.4f}{Colors.RESET}  "
            f"Spread: {down_spread:.4f}"
        )

        # Active signals
        lines.append(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        enabled = []
        if cfg.flash_crash_enabled:
            enabled.append(f"Crash(>{cfg.drop_threshold:.2f})")
        if cfg.momentum_enabled:
            enabled.append(f"Mom({cfg.momentum_window}s)")
        if cfg.imbalance_enabled:
            enabled.append(f"Imbal(>{cfg.imbalance_signal_threshold:.2f})")
        if cfg.time_decay_enabled:
            enabled.append(f"Decay(<{cfg.time_decay_threshold_pct:.0%})")

        lines.append(
            f"{Colors.BOLD}Detectors:{Colors.RESET} {' | '.join(enabled)} | "
            f"Entry: >={self._entry_threshold(update_cache=False):.2f} | "
            f"TP: +{cfg.take_profit:.0%} | SL: -{cfg.stop_loss:.0%}"
        )

        # Current signals
        if self._last_signals:
            sig_parts = []
            for sig in self._last_signals:
                color = Colors.GREEN if sig.side == "up" else Colors.RED
                sig_parts.append(
                    f"{color}{sig.side.upper()}{Colors.RESET} "
                    f"{sig.score:.2f} ({sig.source}:{sig.reason})"
                )
            lines.append(f"{Colors.BOLD}Live Signals:{Colors.RESET} {' | '.join(sig_parts)}")
        else:
            lines.append(
                f"{Colors.BOLD}Live Signals:{Colors.RESET} "
                f"{Colors.DIM}(waiting for signals...){Colors.RESET}"
            )

        if cfg.signal_state_logging_enabled:
            lines.append(f"{Colors.BOLD}Signal Diagnostics:{Colors.RESET}")
            if self._last_signal_state_lines:
                for entry in self._last_signal_state_lines:
                    lines.append(f"  {entry}")
                if self._last_conflict_reason:
                    lines.append(f"  conflict: {self._last_conflict_reason}")
                gate = self._last_entry_gate
                lines.append(
                    f"  entry_gate: side={str(gate.get('side','')).upper() or '?'} "
                    f"composite={float(gate.get('composite', 0.0)):.3f} "
                    f"threshold={float(gate.get('threshold', 0.0)):.3f} "
                    f"pass={bool(gate.get('pass', False))}"
                )
            else:
                lines.append(f"  {Colors.DIM}(awaiting first diagnostic sample){Colors.RESET}")

        # Open Orders section
        lines.append(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        lines.append(f"{Colors.BOLD}Open Orders:{Colors.RESET}")
        if self.open_orders:
            for order in self.open_orders[:5]:
                side = order.get("side", "?")
                price = float(order.get("price", 0))
                size = float(order.get("original_size", order.get("size", 0)))
                filled = float(order.get("size_matched", 0))
                order_id = order.get("id", "")[:8]
                token = order.get("asset_id", "")
                token_side = (
                    "UP" if token == self.token_ids.get("up")
                    else "DOWN" if token == self.token_ids.get("down")
                    else "?"
                )
                color = Colors.GREEN if side == "BUY" else Colors.RED
                lines.append(
                    f"  {color}{side:4}{Colors.RESET} {token_side:4} "
                    f"@ {price:.4f} Size: {size:.1f} "
                    f"Filled: {filled:.1f} ID: {order_id}..."
                )
        else:
            lines.append(f"  {Colors.CYAN}(no open orders){Colors.RESET}")

        # Positions
        lines.append(f"{Colors.BOLD}Positions:{Colors.RESET}")
        all_positions = self.positions.get_all_positions()
        if all_positions:
            for pos in all_positions:
                current = prices.get(pos.side, 0)
                pnl = pos.get_pnl(current)
                pnl_pct = pos.get_pnl_percent(current)
                hold_time = pos.get_hold_time()
                color = Colors.GREEN if pnl >= 0 else Colors.RED

                lines.append(
                    f"  {Colors.BOLD}{pos.side.upper():4}{Colors.RESET} "
                    f"Entry: {pos.entry_price:.4f} | "
                    f"Current: {current:.4f} | "
                    f"Size: ${pos.size:.2f} | "
                    f"PnL: {color}${pnl:+.2f} ({pnl_pct:+.1f}%){Colors.RESET} | "
                    f"Hold: {hold_time:.0f}s"
                )
                if pos.stop_loss_delta <= 0:
                    sl_label = f"{Colors.GREEN}SL: {pos.stop_loss_price:.4f} (locked +{-pos.stop_loss_delta:.0%}){Colors.RESET}"
                else:
                    sl_label = f"SL: {pos.stop_loss_price:.4f} (-{pos.stop_loss_delta:.0%})"
                lines.append(
                    f"       TP: {pos.take_profit_price:.4f} "
                    f"(+{pos.take_profit_delta:.0%}) | "
                    f"{sl_label}"
                )
        else:
            lines.append(f"  {Colors.CYAN}(no open positions){Colors.RESET}")

        # Recent logs
        if self._log_buffer.messages:
            lines.append("-" * 80)
            lines.append(f"{Colors.BOLD}Recent Events:{Colors.RESET}")
            for msg in self._log_buffer.get_messages():
                lines.append(f"  {msg}")

        # Render in place. Clear once, then keep redrawing from the top.
        # This is much less visually noisy than clearing the full screen every frame.
        pad_lines = max(0, self._last_render_line_count - len(lines))
        if pad_lines:
            lines.extend([""] * pad_lines)
        self._last_render_line_count = len(lines)
        prefix = "\033[2J\033[H" if not self._screen_initialized else "\033[H"
        self._screen_initialized = True
        output = prefix + "\n".join(lines)
        print(output, end="", flush=True)
