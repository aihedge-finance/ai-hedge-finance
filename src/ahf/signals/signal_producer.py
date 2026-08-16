"""SignalProducer ABC — Layer 1 of the signal pipeline.

Every signal source (RL agent, technical indicator, LLM, rule engine)
must implement this interface. The orchestrator calls produce() each step
and passes the result to the aggregator.

Design reference: design/pre_upgrade_v2_analysis/option_ab_architecture.md
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ahf.signals.signal_types import SignalOutput


class SignalProducer(ABC):
    """Abstract base class for all Layer-1 signal sources.

    Implementing a new producer requires:
    1. Subclass SignalProducer
    2. Implement `name` property
    3. Implement `produce(market_data, context) -> SignalOutput`
    4. Implement `from_config(name, config, runtime_deps)` classmethod
    5. Optionally override `health_check()` for startup validation

    See: src/ahf/signals/producers/_template.py for a copy-paste starter.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this producer instance.

        Must match the 'id' field in pipeline.json.
        Used in audit logs, timeout messages, and aggregator weight lookup.
        """
        ...

    @abstractmethod
    def produce(self, market_data: dict, context: dict) -> SignalOutput:
        """Compute and return a signal for the current market state.

        Args:
            market_data: Multi-timeframe OHLCV data dict.
                Keys are interval strings: "5m", "1h", "1d", etc.
                Each value is a dict with "df", "last_close", "last_volume", "updated_at".
                The execution timeframe (e.g. "5m") is always present.
                Higher timeframes are present only if configured.
            context: Shared runtime context injected by the orchestrator.
                Keys may include: "step", "portfolio", "regime", "timestamp".

        Returns:
            SignalOutput with action ∈ [-1.0, 1.0] and confidence ∈ [0.0, 1.0].

        Raises:
            Any exception: The orchestrator's try/except will catch it and
            treat this producer as failed for this step (equivalent to None).
            Do NOT swallow exceptions silently — let them propagate.
        """
        ...

    def health_check(self) -> None:
        """Verify this producer's dependencies at startup. Raise on failure.

        Called once before the trading loop starts. Override to check:
        - RL: model file exists and loads without error
        - LLM: API key is set and endpoint responds
        - Tech: strategy config file exists and parses correctly

        Default implementation: no-op (always healthy).

        Raises:
            RuntimeError: If a required dependency is unavailable.
        """
        pass

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        name: str,
        config: dict,
        runtime_deps: dict,
    ) -> "SignalProducer":
        """Factory method: build a producer from JSON config + runtime objects.

        Args:
            name: Producer ID from pipeline.json (e.g. "rl_ppo").
            config: The producer's 'config' dict from pipeline.json.
            runtime_deps: Dict of runtime objects that can't be in JSON
                (e.g. {"env": env, "agent_ppo": agent, "gemini-2.0-flash": client}).

        Returns:
            A configured, ready-to-use SignalProducer instance.
        """
        ...
