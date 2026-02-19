"""
Edge Finding Agent - Detects mispriced contracts by comparing model probability
against market-implied probability.

How it works:
    1. Estimate "true" probability of BTC going UP by expiry using:
       - Recent spot momentum (60s and 2s BTC price change)
       - Orderbook imbalance (bid-heavy = bullish pressure)
       - Volatility regime (high vol → probability closer to 50%)
    2. Compare against current market price (which IS the implied probability)
    3. Edge = model_prob - market_price
       - Positive edge = market underpricing this side → opportunity
       - Negative edge = market overpricing this side → fade opportunity
    4. Apply longshot bias correction (from calibration research):
       Contracts priced below ~12¢ historically win far less often than
       implied, so we haircut our model estimate when market is in that zone.

The agent does NOT replace momentum/mean-reversion bots. It layers on top:
    - Coordinator can check get_current_signal() before approving entries
    - Signals above min_edge threshold surface as actionable opportunities
    - All signals are logged for post-trade analysis

Usage:
    from lib.edge_agent import EdgeAgent, EdgeConfig

    config = EdgeConfig(min_edge=0.05)
    agent = EdgeAgent(shared_state, btc_tracker, config)

    await agent.start()

    signal = agent.get_current_signal()
    if signal:
        print(f"Edge found: {signal.side} @ {signal.edge:.1%}")

    await agent.stop()
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class EdgeConfig:
    """Configuration for the edge finding agent."""

    # Minimum edge (model vs market) to surface a signal
    min_edge: float = 0.05              # 5% mispricing threshold

    # Minimum model confidence to surface a signal
    min_confidence: float = 0.35

    # How often to run a scan (seconds)
    scan_interval: float = 3.0

    # Weight given to 60s BTC momentum in probability estimate
    momentum_weight: float = 0.18

    # Weight given to 2s BTC momentum (short-burst signal)
    impulse_weight: float = 0.08

    # Weight given to orderbook imbalance
    ob_weight: float = 0.10

    # Below this market price, apply longshot bias correction
    # (research shows sub-15¢ contracts win ~43% as often as implied)
    longshot_threshold: float = 0.15

    # Haircut applied to model probability when market is in longshot zone
    # E.g. 0.5 means: estimated win rate is 50% of what market implies
    longshot_haircut: float = 0.55

    # Suppress signals when VPIN exceeds this (informed flow dominates)
    max_vpin: float = 0.70

    # Suppress signals when fewer than this many seconds remain
    min_time_remaining: int = 45

    # Rolling history of edge estimates kept for logging/analysis
    history_maxlen: int = 200


# ── Signal dataclass ──────────────────────────────────────────────────────────

@dataclass
class EdgeSignal:
    """
    A single edge-finding output.

    Fields mirror what you'd log for post-trade TCA:
        model_prob   — agent's estimated true probability
        market_prob  — current market price (implied probability)
        edge         — model_prob - market_prob  (positive = buy signal)
        confidence   — 0-1 score of model confidence
        reason       — human-readable breakdown of contributing factors
    """
    side: str               # "up" or "down"
    model_prob: float       # agent's P(outcome resolves YES)
    market_prob: float      # market-implied probability (current price)
    edge: float             # model_prob - market_prob
    confidence: float       # 0-1
    reason: str             # factor breakdown for logging
    timestamp: float = field(default_factory=time.time)
    market_slug: str = ""
    time_to_expiry: int = 0

    @property
    def is_actionable(self) -> bool:
        """True when edge and confidence both exceed configured thresholds."""
        return self.edge > 0 and self.confidence > 0


# ── Agent ─────────────────────────────────────────────────────────────────────

class EdgeAgent:
    """
    Continuously scans for mispriced contracts.

    Runs as a background asyncio task. Call get_current_signal() from the
    coordinator or any other component to retrieve the latest finding.
    """

    def __init__(self, shared_state, btc_tracker, config: Optional[EdgeConfig] = None):
        """
        Args:
            shared_state: SharedState instance (market data + risk metrics)
            btc_tracker:  BTCPriceTracker instance (spot price + history)
            config:       EdgeConfig, defaults to EdgeConfig() if None
        """
        self.shared_state = shared_state
        self.btc_tracker = btc_tracker
        self.config = config or EdgeConfig()

        self._current_signal: Optional[EdgeSignal] = None
        self._history: list[dict] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._scan_count = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background scan loop."""
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info("EdgeAgent started (interval=%.1fs, min_edge=%.0f%%)",
                    self.config.scan_interval, self.config.min_edge * 100)

    async def stop(self) -> None:
        """Stop the background scan loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("EdgeAgent stopped after %d scans", self._scan_count)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_current_signal(self) -> Optional[EdgeSignal]:
        """
        Return the most recent edge signal, or None if no edge was found.

        The signal is replaced each scan cycle. A non-None return means the
        agent found edge above min_edge AND min_confidence thresholds.
        """
        return self._current_signal

    def get_history(self) -> list[dict]:
        """Return the full scan history as a list of dicts (for logging/analysis)."""
        return list(self._history)

    def get_scan_count(self) -> int:
        return self._scan_count

    # ── Core logic ────────────────────────────────────────────────────────────

    def estimate_probability(self) -> tuple[float, float, str]:
        """
        Estimate the true probability of the UP contract resolving YES.

        Returns:
            (up_prob, confidence, reason_string)

        Model:
            base = 0.50  (fair prior)
            + momentum_factor  (60s BTC spot direction, regime-scaled)
            + impulse_factor   (2s BTC spot burst)
            + ob_factor        (orderbook bid/ask imbalance)
            clamped to [0.05, 0.95]

        Confidence decays when:
            - VPIN is high (informed flow, our model is unreliable)
            - We're very close to 50% (no clear edge to estimate)
            - BTC spot data is stale
        """
        cfg = self.config
        market = self.shared_state.get_market_data()
        risk   = self.shared_state.get_risk_metrics()

        # ── Spot momentum factor ──────────────────────────────────────────────
        change_60s = risk.btc_spot_change_60s   # fractional, e.g. 0.003 = 0.3%
        change_2s  = risk.btc_spot_change_2s

        # Scale by volatility so strong-vol regimes don't over-fire
        vol_scale = max(risk.volatility_ewma, 0.0005)   # avoid div/0

        # tanh squashes to (-1, +1), then weight down to a fraction of 50%
        momentum_factor = math.tanh(change_60s / (vol_scale * 10)) * cfg.momentum_weight
        impulse_factor  = math.tanh(change_2s  / (vol_scale * 5))  * cfg.impulse_weight

        # ── Orderbook imbalance factor ────────────────────────────────────────
        # market.imbalance: 0 = bid-heavy (bullish), 1 = ask-heavy (bearish)
        # (0.5 - imbalance) → positive when bids dominate → bullish
        ob_factor = (0.5 - market.imbalance) * cfg.ob_weight

        # ── Combine ───────────────────────────────────────────────────────────
        up_prob = 0.50 + momentum_factor + impulse_factor + ob_factor
        up_prob = max(0.05, min(0.95, up_prob))

        # ── Confidence ───────────────────────────────────────────────────────
        # Distance from 50% — the further we are, the more confident
        prob_signal_strength = abs(up_prob - 0.50) / 0.45   # 0-1

        # VPIN penalty: high toxicity means informed traders dominate;
        # our model is reading noise, not signal
        vpin_penalty = min(risk.vpin, 1.0)

        # Stale BTC data penalty
        stale_penalty = 0.3 if self.btc_tracker.is_stale(max_age=5.0) else 0.0

        confidence = prob_signal_strength * (1.0 - vpin_penalty * 0.6) - stale_penalty
        confidence = max(0.0, min(1.0, confidence))

        reason = (
            f"mom60={change_60s*100:+.3f}% ({momentum_factor:+.3f}) "
            f"imp2s={change_2s*100:+.3f}% ({impulse_factor:+.3f}) "
            f"ob_imb={market.imbalance:.2f} ({ob_factor:+.3f}) "
            f"→ P(up)={up_prob:.3f} conf={confidence:.2f} "
            f"vpin={risk.vpin:.2f}"
        )

        return up_prob, confidence, reason

    def _apply_longshot_correction(
        self, model_prob: float, market_prob: float
    ) -> float:
        """
        Apply calibration-based longshot bias correction.

        When the market prices a contract below longshot_threshold, the
        research shows realized win rate is ~55% of implied. We haircut
        our model estimate accordingly so we don't chase overpriced longshots.
        """
        cfg = self.config
        if market_prob >= cfg.longshot_threshold:
            return model_prob

        # How deep into longshot territory are we?
        depth = (cfg.longshot_threshold - market_prob) / cfg.longshot_threshold
        haircut = 1.0 - (1.0 - cfg.longshot_haircut) * depth
        corrected = model_prob * haircut

        logger.debug(
            "Longshot correction: market=%.3f model %.3f → %.3f (haircut=%.2f)",
            market_prob, model_prob, corrected, haircut
        )
        return corrected

    async def scan(self) -> Optional[EdgeSignal]:
        """
        Run one full scan cycle.

        Returns an EdgeSignal if actionable edge is found, else None.
        The internal _current_signal is always updated (to None or a signal).
        """
        cfg = self.config
        self._scan_count += 1

        market = self.shared_state.get_market_data()
        risk   = self.shared_state.get_risk_metrics()

        # ── Gate checks ───────────────────────────────────────────────────────
        if market.time_to_expiry < cfg.min_time_remaining:
            self._current_signal = None
            if hasattr(self.shared_state, "set_edge_signal"):
                self.shared_state.set_edge_signal(None)
            return None

        if risk.vpin > cfg.max_vpin:
            logger.debug("EdgeAgent suppressed: VPIN=%.2f > %.2f", risk.vpin, cfg.max_vpin)
            self._current_signal = None
            if hasattr(self.shared_state, "set_edge_signal"):
                self.shared_state.set_edge_signal(None)
            return None

        # ── Estimate probability ──────────────────────────────────────────────
        model_up, confidence, reason = self.estimate_probability()

        # DOWN contract is complementary
        model_down = 1.0 - model_up

        # Market prices (Polymarket: price IS the probability, 0-1)
        market_up   = market.up_price    # P(UP resolves YES) per market
        market_down = market.down_price  # P(DOWN resolves YES) per market

        # Apply longshot corrections
        model_up_adj   = self._apply_longshot_correction(model_up,   market_up)
        model_down_adj = self._apply_longshot_correction(model_down, market_down)

        # ── Edge calculation ──────────────────────────────────────────────────
        edge_up   = model_up_adj   - market_up
        edge_down = model_down_adj - market_down

        # Pick the side with more edge
        if abs(edge_up) >= abs(edge_down):
            best_side    = "up"
            best_edge    = edge_up
            best_model   = model_up_adj
            best_market  = market_up
        else:
            best_side    = "down"
            best_edge    = edge_down
            best_model   = model_down_adj
            best_market  = market_down

        # Record for history (always, even below threshold)
        record = {
            "ts": time.time(),
            "slug": market.market_slug,
            "expiry": market.time_to_expiry,
            "side": best_side,
            "model": round(best_model, 4),
            "market": round(best_market, 4),
            "edge": round(best_edge, 4),
            "conf": round(confidence, 3),
            "vpin": round(risk.vpin, 3),
            "reason": reason,
        }
        self._history.append(record)
        if len(self._history) > cfg.history_maxlen:
            self._history.pop(0)

        # ── Threshold check ───────────────────────────────────────────────────
        if best_edge < cfg.min_edge or confidence < cfg.min_confidence:
            self._current_signal = None
            if hasattr(self.shared_state, "set_edge_signal"):
                self.shared_state.set_edge_signal(None)
            logger.debug(
                "EdgeAgent no signal: edge=%.3f conf=%.2f (thresholds %.3f/%.2f)",
                best_edge, confidence, cfg.min_edge, cfg.min_confidence
            )
            return None

        # ── Signal found ──────────────────────────────────────────────────────
        signal = EdgeSignal(
            side=best_side,
            model_prob=best_model,
            market_prob=best_market,
            edge=best_edge,
            confidence=confidence,
            reason=reason,
            market_slug=market.market_slug,
            time_to_expiry=market.time_to_expiry,
        )
        self._current_signal = signal

        # Publish to shared state so coordinator/bots can read without
        # needing a direct reference to the agent
        if hasattr(self.shared_state, "set_edge_signal"):
            self.shared_state.set_edge_signal(signal)

        logger.info(
            "EdgeAgent SIGNAL: %s | model=%.3f market=%.3f edge=%.1f%% conf=%.2f | %s",
            best_side.upper(), best_model, best_market,
            best_edge * 100, confidence, reason
        )
        return signal

    # ── Background loop ───────────────────────────────────────────────────────

    async def _run(self) -> None:
        """Background scan loop."""
        while self._running:
            try:
                await self.scan()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("EdgeAgent scan error: %s", e)

            try:
                await asyncio.sleep(self.config.scan_interval)
            except asyncio.CancelledError:
                break
