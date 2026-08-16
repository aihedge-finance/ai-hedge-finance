"""TechIndicatorProducer — wraps TradingStrategy (double_kf, RSI_MACD, etc.)

The strategy's run_step() / get_signal() method is called each bar.
The output is a direction enum which maps to action ∈ [-1.0, 1.0].

If the strategy is not configured or raises, returns HOLD (confidence=0.0).

Phase 4 implementation — strategy loaded lazily on first produce() call.
"""
from __future__ import annotations

import logging
from typing import Any

from ahf.signals.signal_producer import SignalProducer
from ahf.signals.signal_types import SignalOutput

logger = logging.getLogger(__name__)

# TradeAction int values from v1 TradeEnum
_SELL = -1
_HOLD = 0
_BUY = 1


class TechIndicatorProducer(SignalProducer):
    """Technical strategy-based signal producer.

    Wraps a TradingStrategy instance. The strategy must expose:
        strategy.get_signal(market_data) -> int  (-1, 0, or 1)
    OR
        strategy.step(market_data) -> dict with "action" key

    Args:
        name: Producer identifier.
        strategy: Pre-built TradingStrategy instance (double_kf, rsi_macd, etc.)
        strategy_name: Name for logging (e.g. "double_kf").
        confidence: Fixed confidence value (default: 0.65).
            Will be replaced with a dynamic confidence score in Phase 5.
    """

    def __init__(
        self,
        name: str,
        strategy: Any = None,
        strategy_name: str = "double_kf",
        confidence: float = 0.65,
    ) -> None:
        self._name = name
        self._strategy = strategy
        self._strategy_name = strategy_name
        self._confidence = confidence

    @property
    def name(self) -> str:
        return self._name

    def produce(self, market_data: dict, context: dict) -> SignalOutput:
        """Run one step of the technical strategy.

        If strategy is None → HOLD stub (allows pipeline to run without it).
        """
        if self._strategy is None:
            return SignalOutput(signal_name=self._name, action=0.0, confidence=0.5)

        try:
            # Try get_signal() first (v2 interface)
            if hasattr(self._strategy, "get_signal"):
                raw = self._strategy.get_signal(market_data)
                action = float(max(-1, min(1, int(raw))))
            # Fallback: step() interface (v1 Strategy classes)
            elif hasattr(self._strategy, "step"):
                result = self._strategy.step(market_data)
                if isinstance(result, dict):
                    action = float(result.get("action", 0))
                else:
                    action = float(max(-1, min(1, int(result))))
            else:
                logger.warning(
                    "TechIndicatorProducer: strategy has no get_signal/step — HOLD",
                    extra={"producer": self._name, "strategy": self._strategy_name},
                )
                return SignalOutput(signal_name=self._name, action=0.0, confidence=0.0)

            return SignalOutput(
                signal_name=self._name,
                action=action,
                confidence=self._confidence,
                metadata={"strategy": self._strategy_name},
            )

        except Exception as e:
            logger.warning(
                "TechIndicatorProducer strategy error — returning HOLD",
                extra={"producer": self._name, "strategy": self._strategy_name, "error": str(e)},
            )
            return SignalOutput(signal_name=self._name, action=0.0, confidence=0.0)

    def health_check(self) -> None:
        if self._strategy is None:
            logger.warning(
                f"[{self._name}] TechIndicatorProducer: no strategy loaded — will return stub HOLD signals"
            )

    @classmethod
    def from_config(cls, name: str, config: dict, runtime_deps: dict) -> "TechIndicatorProducer":
        """Build from pipeline.json config.

        Expected runtime_deps keys:
            "{name}_strategy" or "tech_strategy": Pre-built strategy instance.

        Example pipeline.json config:
            {
                "id": "tech_kf",
                "type": "tech_indicator",
                "config": {
                    "strategy": "double_kf",
                    "confidence": 0.65
                }
            }
        """
        strategy_name = config.get("strategy", "double_kf")
        strategy = (
            runtime_deps.get(f"{name}_strategy")
            or runtime_deps.get("tech_strategy")
            or runtime_deps.get(strategy_name)
        )
        return cls(
            name=name,
            strategy=strategy,
            strategy_name=strategy_name,
            confidence=float(config.get("confidence", 0.65)),
        )
