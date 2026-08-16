"""TradeOrchestrator — the main trading loop coordinator.

Ties together all domain components:
  market_data → pipeline → signal_processor → risk_manager → order_executor

The orchestrator is designed for use in both:
1. Live/paper trading (called by ahf-trade entrypoint)
2. RL training (called by the gym environment step() function)

Design reference: design/pre_upgrade_v2_analysis/option_ab_architecture.md
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from ahf.core.enums import TradeAction
from ahf.domain.order_executor import OrderExecutor
from ahf.domain.position_tracker import PositionTracker
from ahf.domain.risk.risk_manager import RiskManager
from ahf.domain.risk.risk_types import RiskVerdict
from ahf.domain.signal_processor import SignalProcessor
from ahf.signals.signal_aggregator import SignalAggregator
from ahf.signals.signal_audit import SignalAuditLog
from ahf.signals.signal_logger import SignalLogger
from ahf.signals.signal_producer import SignalProducer
from ahf.signals.signal_types import SignalOutput
from ahf.signals.timeout import produce_with_timeout

logger = logging.getLogger(__name__)


class StepResult:
    """Result of a single orchestrator step."""

    __slots__ = (
        "step", "action", "signal", "risk_verdict",
        "order", "skipped", "skip_reason", "elapsed_ms"
    )

    def __init__(
        self,
        step: int,
        action: TradeAction,
        signal: Optional[SignalOutput],
        risk_verdict: RiskVerdict,
        order: Optional[dict],
        skipped: bool,
        skip_reason: str,
        elapsed_ms: float,
    ) -> None:
        self.step = step
        self.action = action
        self.signal = signal
        self.risk_verdict = risk_verdict
        self.order = order
        self.skipped = skipped
        self.skip_reason = skip_reason
        self.elapsed_ms = elapsed_ms


class TradeOrchestrator:
    """Main trading loop coordinator.

    Args:
        producers: Layer-1 signal producers (from load_pipeline()).
        aggregator: Layer-2 signal aggregator (from load_pipeline()).
        signal_processor: Converts continuous signal → discrete action.
        risk_manager: Pre-trade risk gate.
        order_executor: Translates action → exchange order.
        position_tracker: Tracks portfolio state.
        min_valid_signals: Minimum non-None signals required to act.
        audit_log: Optional JSONL audit logger.
    """

    def __init__(
        self,
        producers: list[SignalProducer],
        aggregator: SignalAggregator,
        signal_processor: SignalProcessor,
        risk_manager: RiskManager,
        order_executor: OrderExecutor,
        position_tracker: PositionTracker,
        min_valid_signals: int = 1,
        audit_log: Optional[SignalAuditLog] = None,
    ) -> None:
        self._producers = producers
        self._aggregator = aggregator
        self._signal_processor = signal_processor
        self._risk_manager = risk_manager
        self._order_executor = order_executor
        self._position_tracker = position_tracker
        self._min_valid_signals = min_valid_signals
        self._audit_log = audit_log
        self._signal_logger = SignalLogger()
        self._step = 0
        self._producer_ids = [p.name for p in producers]

    def step(self, market_data: dict, context: dict | None = None) -> StepResult:
        """Execute one trading step.

        Args:
            market_data: Multi-timeframe OHLCV dict.
            context: Optional extra context (injected into producers/aggregator).

        Returns:
            StepResult with all step details for logging and audit.
        """
        t_start = time.monotonic()
        ctx = context or {}
        ctx["step"] = self._step
        ctx["producer_ids"] = self._producer_ids

        # 1. Collect signals from all producers (with timeout)
        raw_signals: list[Optional[SignalOutput]] = []
        producer_pairs: list[tuple[str, Optional[SignalOutput]]] = []

        for producer in self._producers:
            t0 = time.monotonic()
            # Per-producer timeout from config is not stored here — use 5s default
            sig, timed_out, error = produce_with_timeout(producer, market_data, ctx, 5.0)
            latency_ms = (time.monotonic() - t0) * 1000
            raw_signals.append(sig)
            producer_pairs.append((producer.name, sig))
            self._signal_logger.log_producer_result(producer.name, sig, latency_ms, error, timed_out)

        # 2. Check minimum valid signals
        valid_count = sum(1 for s in raw_signals if s is not None)
        if valid_count < self._min_valid_signals:
            self._signal_logger.log_step_skipped(
                "insufficient_signals", valid_count, self._min_valid_signals
            )
            return self._make_result(TradeAction.HOLD, None, RiskVerdict.ALLOW, None,
                                     skipped=True, skip_reason="insufficient_signals",
                                     t_start=t_start)

        # 3. Aggregate
        t_agg_start = time.monotonic()
        aggregated = self._aggregator.aggregate(raw_signals, ctx)
        agg_latency_ms = (time.monotonic() - t_agg_start) * 1000
        self._signal_logger.log_aggregation_result(aggregated, valid_count, len(raw_signals), agg_latency_ms)

        # 4. Discretise signal → action
        action = self._signal_processor.process(aggregated)

        # 5. Risk gate
        portfolio = self._position_tracker.snapshot()
        risk_verdict, effective_size, risk_results = self._risk_manager.evaluate(
            portfolio,
            proposed_action=float(action),
            proposed_size=abs(aggregated.action),
        )

        # 6. Execute order (or skip if VETOED / HOLD)
        order: Optional[dict] = None
        if risk_verdict != RiskVerdict.VETO and action != TradeAction.HOLD:
            order = self._order_executor.execute(action, effective_size)

        # 7. Audit log
        if self._audit_log is not None:
            self._audit_log.write_step(self._step, producer_pairs, aggregated, {
                "action": int(action),
                "risk_verdict": risk_verdict.value,
                "valid_signals": valid_count,
            })

        # 8. Advance tracker
        self._position_tracker.tick()
        self._step += 1

        return self._make_result(action, aggregated, risk_verdict, order,
                                 skipped=False, skip_reason="", t_start=t_start)

    def _make_result(
        self,
        action: TradeAction,
        signal: Optional[SignalOutput],
        risk_verdict: RiskVerdict,
        order: Optional[dict],
        skipped: bool,
        skip_reason: str,
        t_start: float,
    ) -> StepResult:
        elapsed_ms = (time.monotonic() - t_start) * 1000
        return StepResult(
            step=self._step,
            action=action,
            signal=signal,
            risk_verdict=risk_verdict,
            order=order,
            skipped=skipped,
            skip_reason=skip_reason,
            elapsed_ms=elapsed_ms,
        )
