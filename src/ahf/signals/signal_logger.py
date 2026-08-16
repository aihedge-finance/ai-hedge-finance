"""Structured JSON telemetry logger for the signal pipeline.

Emits one JSON log line per event. Can be consumed by log aggregation
tools (Datadog, Grafana Loki, CloudWatch) without further parsing.

Design reference: design/pre_upgrade_v2_analysis/option_ab_production_readiness.md
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from ahf.signals.signal_types import SignalOutput

_logger = logging.getLogger("ahf.signal_pipeline")


class SignalLogger:
    """Structured JSON logger for signal pipeline events."""

    def __init__(self, logger: logging.Logger = _logger) -> None:
        self._log = logger

    def log_producer_result(
        self,
        producer_name: str,
        signal: Optional[SignalOutput],
        latency_ms: float,
        error: Optional[Exception] = None,
        timed_out: bool = False,
    ) -> None:
        entry: dict = {
            "event": "producer_result",
            "producer": producer_name,
            "latency_ms": round(latency_ms, 2),
            "success": signal is not None and error is None,
            "timed_out": timed_out,
        }
        if signal is not None:
            entry["action"] = signal.action
            entry["confidence"] = signal.confidence
        if error is not None:
            entry["error"] = str(error)
        self._log.info(json.dumps(entry))

    def log_aggregation_result(
        self,
        aggregated: SignalOutput,
        valid_count: int,
        total_count: int,
        latency_ms: float,
    ) -> None:
        entry = {
            "event": "aggregation_result",
            "action": aggregated.action,
            "confidence": aggregated.confidence,
            "valid_producers": valid_count,
            "total_producers": total_count,
            "latency_ms": round(latency_ms, 2),
        }
        self._log.info(json.dumps(entry))

    def log_step_skipped(self, reason: str, valid_count: int, required: int) -> None:
        entry = {
            "event": "step_skipped",
            "reason": reason,
            "valid_producers": valid_count,
            "min_required": required,
        }
        self._log.warning(json.dumps(entry))
