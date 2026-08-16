"""FixedWeightAggregator — static config-defined weights per producer.

Weights are defined in pipeline.json under aggregator.config.weights.
Missing producers (failed / timed out) have their weight redistributed
proportionally to the remaining valid producers.
"""
from __future__ import annotations

from typing import Optional

from ahf.signals.signal_aggregator import SignalAggregator
from ahf.signals.signal_types import SignalOutput

_HOLD_SIGNAL = SignalOutput(signal_name="fixed_weight", action=0.0, confidence=0.1)


class FixedWeightAggregator(SignalAggregator):
    """Explicit static-weight aggregator.

    Config example in pipeline.json:
        "aggregator": {
            "type": "fixed_weight",
            "config": {
                "weights": {
                    "rl_ppo":   0.50,
                    "tech_kf":  0.30,
                    "llm_gemini": 0.20
                }
            }
        }

    If ALL signals are None → return HOLD with confidence=0.1.
    """

    def __init__(self, weights: dict[str, float]) -> None:
        self._weights = weights  # {producer_id: weight}

    def aggregate(
        self,
        signals: list[Optional[SignalOutput]],
        context: dict,
    ) -> SignalOutput:
        producer_ids: list[str] = context.get("producer_ids", [])

        # Build (weight, signal) pairs for valid signals
        valid_pairs: list[tuple[float, SignalOutput]] = []
        for i, sig in enumerate(signals):
            if sig is None:
                continue
            pid = producer_ids[i] if i < len(producer_ids) else sig.signal_name
            w = self._weights.get(pid, 0.0)
            if w > 0.0:
                valid_pairs.append((w, sig))

        if not valid_pairs:
            return _HOLD_SIGNAL

        total_w = sum(w for w, _ in valid_pairs)
        if total_w == 0.0:
            return _HOLD_SIGNAL

        # Re-normalise weights over valid signals
        action = sum((w / total_w) * s.action for w, s in valid_pairs)
        confidence = sum((w / total_w) * s.confidence for w, s in valid_pairs)

        return SignalOutput(
            signal_name="fixed_weight",
            action=action,
            confidence=confidence,
        )

    @classmethod
    def from_config(cls, config: dict, runtime_deps: dict) -> "FixedWeightAggregator":
        weights = config.get("weights", {})
        if not weights:
            raise ValueError("FixedWeightAggregator requires 'weights' in config")
        return cls(weights=weights)
