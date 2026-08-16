"""RLSignalProducer — stub (Phase 2).

Phase 4 will replace this stub with the real AgentPPO/DRLAgent wrapper.
Until then, returns a fixed neutral signal so the full pipeline can run.
"""
from __future__ import annotations

from ahf.signals.signal_producer import SignalProducer
from ahf.signals.signal_types import SignalOutput


class RLSignalProducer(SignalProducer):
    """Stub: Phase 4 will wire in the real AgentPPO inference."""

    def __init__(self, name: str, model_path: str = "", agent_type: str = "ppo") -> None:
        self._name = name
        self._model_path = model_path
        self._agent_type = agent_type

    @property
    def name(self) -> str:
        return self._name

    def produce(self, market_data: dict, context: dict) -> SignalOutput:
        # STUB: Phase 4 replaces this with real RL agent inference
        return SignalOutput(signal_name=self.name, action=0.0, confidence=0.5)

    def health_check(self) -> None:
        # STUB: Phase 4 will verify model file exists and loads
        pass

    @classmethod
    def from_config(cls, name: str, config: dict, runtime_deps: dict) -> "RLSignalProducer":
        return cls(
            name=name,
            model_path=config.get("model_path", ""),
            agent_type=config.get("agent", "ppo"),
        )
