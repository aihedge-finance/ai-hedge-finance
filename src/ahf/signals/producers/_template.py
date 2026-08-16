"""SignalProducer template — copy this file to create a new producer.

Instructions:
1. Copy this file: cp _template.py my_producer.py
2. Rename the class: MyProducer
3. Implement produce() with your signal logic
4. Implement from_config() for JSON pipeline config wiring
5. Register in pipeline_loader.py PRODUCER_REGISTRY:
       "my_type": "ahf.signals.producers.my_producer.MyProducer"
6. Add "type": "my_type" entry to pipeline.json

See: docs/guides/CONTRIBUTING_SIGNALS.md for the full tutorial.
"""
from __future__ import annotations

from ahf.signals.signal_producer import SignalProducer
from ahf.signals.signal_types import SignalOutput


class TemplateProducer(SignalProducer):
    """Template: replace with your signal logic."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def produce(self, market_data: dict, context: dict) -> SignalOutput:
        """TODO: implement your signal logic here.

        Args:
            market_data: Multi-timeframe OHLCV dict.
                Access the 5m OHLCV DataFrame: market_data["5m"]["df"]
                Access latest close price:       market_data["5m"]["last_close"]
            context: Runtime context from the orchestrator.
                Access current step:             context.get("step")
                Access portfolio snapshot:       context.get("portfolio")

        Returns:
            SignalOutput(signal_name=self.name, action=..., confidence=...)
            action    ∈ [-1.0, 1.0]  — negative=sell, zero=hold, positive=buy
            confidence ∈ [0.0, 1.0]  — how confident you are in the action
        """
        # Replace this with your logic:
        action = 0.0
        confidence = 0.0

        return SignalOutput(
            signal_name=self.name,
            action=action,
            confidence=confidence,
        )

    def health_check(self) -> None:
        """Optional: verify your dependencies at startup. Raise on failure."""
        pass

    @classmethod
    def from_config(cls, name: str, config: dict, runtime_deps: dict) -> "TemplateProducer":
        """Build from pipeline.json config + runtime dependencies.

        Example config in pipeline.json:
            {
                "id": "my_producer",
                "type": "my_type",
                "timeout_seconds": 5.0,
                "config": {
                    "my_param": "value"
                }
            }
        """
        return cls(name=name)
