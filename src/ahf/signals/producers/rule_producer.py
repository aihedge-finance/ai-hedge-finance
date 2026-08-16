"""RuleBasedProducer — deterministic rule engine.

Evaluates a list of named rules against current market context.
Each rule is a Python callable: (market_data, context) -> float (in [-1, 1]).
Rules are averaged with equal weight to produce the final signal.

Useful for:
- Price > 200d MA → BUY bias
- VIX spike → SELL veto
- Funding rate > X% → avoid longs
"""
from __future__ import annotations

from typing import Callable

from ahf.signals.signal_producer import SignalProducer
from ahf.signals.signal_types import SignalOutput

RuleFunc = Callable[[dict, dict], float]


class RuleBasedProducer(SignalProducer):
    """Deterministic rule engine — equal-weighted average of rule outputs."""

    def __init__(self, name: str, rules: list[tuple[str, RuleFunc]] | None = None) -> None:
        self._name = name
        self._rules: list[tuple[str, RuleFunc]] = rules or []

    @property
    def name(self) -> str:
        return self._name

    def add_rule(self, rule_name: str, fn: RuleFunc) -> None:
        """Register a rule function. fn(market_data, context) -> float in [-1, 1]."""
        self._rules.append((rule_name, fn))

    def produce(self, market_data: dict, context: dict) -> SignalOutput:
        if not self._rules:
            return SignalOutput(signal_name=self.name, action=0.0, confidence=0.0)

        results: list[float] = []
        for rule_name, fn in self._rules:
            try:
                val = float(fn(market_data, context))
                results.append(max(-1.0, min(1.0, val)))
            except Exception:
                pass  # Skip failed rules — they are deterministic, failure is a bug

        if not results:
            return SignalOutput(signal_name=self.name, action=0.0, confidence=0.0)

        action = sum(results) / len(results)
        # Confidence = fraction of rules that agree on direction
        if action > 0:
            agreeing = sum(1 for r in results if r > 0)
        elif action < 0:
            agreeing = sum(1 for r in results if r < 0)
        else:
            agreeing = sum(1 for r in results if r == 0)
        confidence = agreeing / len(results)

        return SignalOutput(
            signal_name=self.name,
            action=action,
            confidence=confidence,
            metadata={"n_rules": len(self._rules), "n_results": len(results)},
        )

    def health_check(self) -> None:
        pass

    @classmethod
    def from_config(cls, name: str, config: dict, runtime_deps: dict) -> "RuleBasedProducer":
        # Rules are registered programmatically — can't be JSON-serialised.
        # Subclass and override from_config() to register domain-specific rules.
        return cls(name=name)
