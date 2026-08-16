"""TechIndicatorProducer — stub (Phase 2).

Phase 4 will replace this stub with the real TradingStrategy wrapper
(double_kf, RSI_MACD, etc.).
"""
from __future__ import annotations

from ahf.signals.signal_producer import SignalProducer
from ahf.signals.signal_types import SignalOutput


class TechIndicatorProducer(SignalProducer):
    """Stub: Phase 4 will wire in the real TradingStrategy inference."""

    def __init__(self, name: str, strategy: str = "double_kf") -> None:
        self._name = name
        self._strategy = strategy

    @property
    def name(self) -> str:
        return self._name

    def produce(self, market_data: dict, context: dict) -> SignalOutput:
        # STUB: Phase 4 replaces this with real strategy computation
        return SignalOutput(signal_name=self.name, action=0.0, confidence=0.5)

    def health_check(self) -> None:
        # STUB: Phase 4 will verify strategy config file exists
        pass

    @classmethod
    def from_config(cls, name: str, config: dict, runtime_deps: dict) -> "TechIndicatorProducer":
        return cls(
            name=name,
            strategy=config.get("strategy", "double_kf"),
        )
