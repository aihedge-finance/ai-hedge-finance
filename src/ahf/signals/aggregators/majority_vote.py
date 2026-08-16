"""MajorityVoteAggregator — directional consensus voting.

Each valid signal casts a directional vote (BUY / HOLD / SELL).
The direction with the most votes wins. On a tie between BUY and SELL,
the result is HOLD.

The final action magnitude is the mean of the action values of the
winning-direction signals. The confidence is the mean confidence of
all valid signals.
"""
from __future__ import annotations

from typing import Optional

from ahf.signals.signal_aggregator import SignalAggregator
from ahf.signals.signal_types import SignalOutput

_HOLD_SIGNAL = SignalOutput(signal_name="majority_vote", action=0.0, confidence=0.1)

_BUY_THRESHOLD = 0.05   # action > threshold → BUY vote
_SELL_THRESHOLD = -0.05  # action < threshold → SELL vote


class MajorityVoteAggregator(SignalAggregator):
    """Simple directional majority vote.

    Config options (all optional):
        buy_threshold  (float, default=0.05):  action > this → BUY vote
        sell_threshold (float, default=-0.05): action < this → SELL vote
    """

    def __init__(
        self,
        buy_threshold: float = _BUY_THRESHOLD,
        sell_threshold: float = _SELL_THRESHOLD,
    ) -> None:
        self._buy_threshold = buy_threshold
        self._sell_threshold = sell_threshold

    def aggregate(
        self,
        signals: list[Optional[SignalOutput]],
        context: dict,
    ) -> SignalOutput:
        valid = [s for s in signals if s is not None]
        if not valid:
            return _HOLD_SIGNAL

        buys = [s for s in valid if s.action > self._buy_threshold]
        sells = [s for s in valid if s.action < self._sell_threshold]
        holds = [s for s in valid if self._sell_threshold <= s.action <= self._buy_threshold]

        mean_confidence = sum(s.confidence for s in valid) / len(valid)

        # Resolve winner
        if len(buys) > len(sells) and len(buys) > len(holds):
            # BUY wins
            action = sum(s.action for s in buys) / len(buys)
        elif len(sells) > len(buys) and len(sells) > len(holds):
            # SELL wins
            action = sum(s.action for s in sells) / len(sells)
        else:
            # Tie or HOLD plurality → HOLD
            action = 0.0

        return SignalOutput(
            signal_name="majority_vote",
            action=action,
            confidence=mean_confidence,
        )

    @classmethod
    def from_config(cls, config: dict, runtime_deps: dict) -> "MajorityVoteAggregator":
        return cls(
            buy_threshold=config.get("buy_threshold", _BUY_THRESHOLD),
            sell_threshold=config.get("sell_threshold", _SELL_THRESHOLD),
        )
