"""WeightedVoteAggregator — confidence-weighted average of all valid signals.

Each producer's contribution is proportional to its confidence score.
Failed / timed-out producers (None) are excluded and weights are
re-normalised over the remaining valid signals.
"""
from __future__ import annotations

from typing import Optional

from ahf.signals.signal_aggregator import SignalAggregator
from ahf.signals.signal_types import SignalOutput

_HOLD_SIGNAL = SignalOutput(signal_name="weighted_vote", action=0.0, confidence=0.1)


class WeightedVoteAggregator(SignalAggregator):
    """Confidence-weighted average aggregator.

    Formula:
        action = Σ(action_i * confidence_i) / Σ(confidence_i)
        confidence = mean(confidence_i) of valid signals

    If ALL signals are None → return HOLD with confidence=0.1.
    """

    def aggregate(
        self,
        signals: list[Optional[SignalOutput]],
        context: dict,
    ) -> SignalOutput:
        valid = [s for s in signals if s is not None]
        if not valid:
            return _HOLD_SIGNAL

        total_confidence = sum(s.confidence for s in valid)
        if total_confidence == 0.0:
            # All valid signals have confidence=0 — equal weight
            action = sum(s.action for s in valid) / len(valid)
            confidence = 0.0
        else:
            action = sum(s.action * s.confidence for s in valid) / total_confidence
            confidence = total_confidence / len(valid)

        return SignalOutput(
            signal_name="weighted_vote",
            action=action,
            confidence=confidence,
        )

    @classmethod
    def from_config(cls, config: dict, runtime_deps: dict) -> "WeightedVoteAggregator":
        return cls()
