"""SignalProcessor — converts aggregated SignalOutput → discrete TradeAction.

This is the boundary between the signal pipeline (continuous floats) and
the domain layer (discrete, executable actions). It applies:
1. Dead-zone filtering: actions near zero become HOLD
2. TradeAction discretisation: BUY / HOLD / SELL

Design reference: design/pre_upgrade_v2_analysis/functional_preservation_report.md
"""
from __future__ import annotations

import logging

from ahf.core.enums import TradeAction
from ahf.signals.signal_types import SignalOutput

logger = logging.getLogger(__name__)


class SignalProcessor:
    """Converts SignalOutput → TradeAction with configurable dead-zone.

    Args:
        buy_threshold: Minimum action value to trigger a BUY.
            Actions between dead-zone boundaries → HOLD.
        sell_threshold: Maximum (most negative) action value to trigger SELL.
        confidence_floor: Minimum confidence required to act.
            Signals below this confidence are forced to HOLD.
    """

    def __init__(
        self,
        buy_threshold: float = 0.1,
        sell_threshold: float = -0.1,
        confidence_floor: float = 0.0,
    ) -> None:
        if buy_threshold <= 0:
            raise ValueError(f"buy_threshold must be > 0, got {buy_threshold}")
        if sell_threshold >= 0:
            raise ValueError(f"sell_threshold must be < 0, got {sell_threshold}")
        self._buy_threshold = buy_threshold
        self._sell_threshold = sell_threshold
        self._confidence_floor = confidence_floor

    def process(self, signal: SignalOutput) -> TradeAction:
        """Convert a SignalOutput to a discrete TradeAction.

        Args:
            signal: The aggregated signal from the pipeline.

        Returns:
            TradeAction.BUY, TradeAction.HOLD, or TradeAction.SELL.
        """
        if signal.confidence < self._confidence_floor:
            logger.debug(
                "Signal below confidence floor — forcing HOLD",
                extra={
                    "action": signal.action,
                    "confidence": signal.confidence,
                    "floor": self._confidence_floor,
                },
            )
            return TradeAction.HOLD

        if signal.action >= self._buy_threshold:
            return TradeAction.BUY
        if signal.action <= self._sell_threshold:
            return TradeAction.SELL
        return TradeAction.HOLD

    def process_to_int(self, signal: SignalOutput) -> int:
        """Convenience: returns -1, 0, or 1."""
        return int(self.process(signal))
