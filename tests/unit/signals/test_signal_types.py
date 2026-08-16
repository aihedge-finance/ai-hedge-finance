"""Tests for signals/signal_types.py — the core contract."""
import pytest
from pydantic import ValidationError

from ahf.signals.signal_types import SignalOutput

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_valid_buy_signal():
    s = SignalOutput(signal_name="test", action=0.8, confidence=0.9)
    assert s.action == 0.8
    assert s.confidence == 0.9
    assert s.signal_name == "test"


def test_valid_sell_signal():
    s = SignalOutput(signal_name="test", action=-0.5, confidence=0.7)
    assert s.action == -0.5


def test_valid_hold_signal():
    s = SignalOutput(signal_name="test", action=0.0, confidence=0.3)
    assert s.action == 0.0


def test_boundary_action_plus_one():
    s = SignalOutput(signal_name="test", action=1.0, confidence=1.0)
    assert s.action == 1.0


def test_boundary_action_minus_one():
    s = SignalOutput(signal_name="test", action=-1.0, confidence=0.5)
    assert s.action == -1.0


def test_boundary_confidence_zero():
    s = SignalOutput(signal_name="test", action=0.5, confidence=0.0)
    assert s.confidence == 0.0


# ---------------------------------------------------------------------------
# Clamping (action and confidence slightly outside bounds are clamped)
# ---------------------------------------------------------------------------


def test_action_slightly_over_one_is_clamped():
    """Floating point overflow 1.0000000001 should clamp to 1.0."""
    s = SignalOutput(signal_name="test", action=1.0000000001, confidence=0.5)
    assert s.action == 1.0


def test_action_slightly_under_minus_one_is_clamped():
    s = SignalOutput(signal_name="test", action=-1.0000000001, confidence=0.5)
    assert s.action == -1.0


def test_confidence_slightly_over_one_is_clamped():
    s = SignalOutput(signal_name="test", action=0.5, confidence=1.0000000001)
    assert s.confidence == 1.0


# ---------------------------------------------------------------------------
# Validation errors (values far outside bounds should still fail)
# ---------------------------------------------------------------------------


def test_action_far_out_of_range_raises():
    with pytest.raises(ValidationError):
        SignalOutput(signal_name="test", action=5.0, confidence=0.5)


def test_action_far_negative_out_of_range_raises():
    with pytest.raises(ValidationError):
        SignalOutput(signal_name="test", action=-2.0, confidence=0.5)


def test_confidence_negative_raises():
    with pytest.raises(ValidationError):
        SignalOutput(signal_name="test", action=0.5, confidence=-0.1)


def test_empty_signal_name_raises():
    with pytest.raises(ValidationError):
        SignalOutput(signal_name="", action=0.5, confidence=0.5)


# ---------------------------------------------------------------------------
# HOLD confidence cap
# ---------------------------------------------------------------------------


def test_hold_signal_caps_confidence_at_0_5():
    """action=0.0 with confidence > 0.5 should be capped to 0.5."""
    s = SignalOutput(signal_name="test", action=0.0, confidence=0.95)
    assert s.confidence == 0.5


def test_non_hold_signal_keeps_high_confidence():
    s = SignalOutput(signal_name="test", action=0.001, confidence=0.95)
    assert s.confidence == 0.95


# ---------------------------------------------------------------------------
# Immutability (frozen model)
# ---------------------------------------------------------------------------


def test_signal_is_immutable():
    s = SignalOutput(signal_name="test", action=0.5, confidence=0.8)
    with pytest.raises(Exception):
        s.action = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Helper methods
# ---------------------------------------------------------------------------


def test_is_bullish():
    assert SignalOutput(signal_name="t", action=0.1, confidence=0.5).is_bullish()
    assert not SignalOutput(signal_name="t", action=-0.1, confidence=0.5).is_bullish()


def test_is_bearish():
    assert SignalOutput(signal_name="t", action=-0.1, confidence=0.5).is_bearish()
    assert not SignalOutput(signal_name="t", action=0.1, confidence=0.5).is_bearish()


def test_is_hold():
    assert SignalOutput(signal_name="t", action=0.0, confidence=0.1).is_hold()


def test_weighted_action():
    s = SignalOutput(signal_name="t", action=0.8, confidence=0.5)
    assert abs(s.weighted_action() - 0.4) < 1e-9


def test_to_audit_dict_keys():
    s = SignalOutput(signal_name="t", action=0.5, confidence=0.7, metadata={"raw": 0.6})
    d = s.to_audit_dict()
    assert "signal_name" in d
    assert "action" in d
    assert "confidence" in d
    assert "timestamp" in d
    assert "raw" in d


def test_metadata_default_is_empty_dict():
    s = SignalOutput(signal_name="t", action=0.0, confidence=0.1)
    assert s.metadata == {}
