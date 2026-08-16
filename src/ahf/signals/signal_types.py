"""SignalOutput — the typed contract between signal producers and the aggregator.

Every SignalProducer.produce() must return exactly one SignalOutput.
The aggregator receives a list of SignalOutput (or None for timed-out/failed producers).

Design reference: design/pre_upgrade_v2_analysis/option_ab_signal_contract.md
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class SignalOutput(BaseModel):
    """Typed, validated output from a single signal producer.

    Fields
    ------
    signal_name : str
        Identifier of the producer that generated this signal.
        Used in audit logs and aggregator weight lookup.
    action : float
        Normalised continuous action in [-1.0, 1.0].
        Semantics:
          -1.0  =  maximum SELL conviction
           0.0  =  HOLD / no opinion
          +1.0  =  maximum BUY conviction
    confidence : float
        Model's self-assessed confidence in [0.0, 1.0].
        Used as the weight in WeightedVote and FixedWeight aggregators.
    timestamp : datetime
        UTC wall-clock time when this signal was produced.
        Auto-populated if not supplied.
    metadata : dict
        Optional producer-specific diagnostic data (not used in aggregation).
        Examples: {"raw_5m_signal": 0.6, "trend_1d": 0.2}
    """

    model_config = {"frozen": True}  # Immutable after creation

    signal_name: str = Field(..., min_length=1)
    action: float = Field(..., ge=-1.0, le=1.0, description="Normalised action in [-1, 1]")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Self-assessed confidence in [0, 1]")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC time of signal generation",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("action", mode="before")
    @classmethod
    def clamp_action(cls, v: float) -> float:
        """Clamp values that are VERY slightly outside [-1, 1] due to floating-point.

        Only clamps if the value is within a 1e-6 tolerance of the boundary.
        Values far outside [-1, 1] (e.g. 5.0, -2.0) are NOT clamped and will
        fail the ge/le constraint below, raising a ValidationError.
        """
        fv = float(v)
        _TOL = 1e-6
        if -1.0 - _TOL <= fv <= -1.0:
            return -1.0
        if 1.0 <= fv <= 1.0 + _TOL:
            return 1.0
        return fv

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        """Clamp confidence values that are VERY slightly outside [0, 1] due to floating-point.

        Only clamps within 1e-6 tolerance. Values far outside (e.g. -0.5, 2.0)
        will fail the ge/le constraint, raising a ValidationError.
        """
        fv = float(v)
        _TOL = 1e-6
        if -_TOL <= fv <= 0.0:
            return 0.0
        if 1.0 <= fv <= 1.0 + _TOL:
            return 1.0
        return fv

    @model_validator(mode="after")
    def validate_action_confidence_consistency(self) -> "SignalOutput":
        """Zero-action signals should not have high confidence.

        This is a soft warning encoded as a constraint: if action == 0.0
        (pure HOLD), confidence should reflect uncertainty, not certainty.
        We don't hard-reject, but we cap confidence at 0.5 for HOLD signals.

        Producers that intentionally issue a high-confidence HOLD (e.g., a
        regime-detection rule that vetoes all trades) should set action to a
        tiny non-zero value like 0.001 to bypass this cap.
        """
        if self.action == 0.0 and self.confidence > 0.5:
            # Use object.__setattr__ because the model is frozen
            object.__setattr__(self, "confidence", 0.5)
        return self

    def is_bullish(self) -> bool:
        """True if the signal leans towards a BUY."""
        return self.action > 0.0

    def is_bearish(self) -> bool:
        """True if the signal leans towards a SELL."""
        return self.action < 0.0

    def is_hold(self) -> bool:
        """True if the signal is neutral."""
        return self.action == 0.0

    def weighted_action(self) -> float:
        """Return action weighted by confidence. Used in WeightedVote aggregator."""
        return self.action * self.confidence

    def to_audit_dict(self) -> dict[str, Any]:
        """Serialise to a flat dict for audit log writing."""
        return {
            "signal_name": self.signal_name,
            "action": self.action,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            **self.metadata,
        }
