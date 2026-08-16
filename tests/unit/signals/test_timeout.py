"""Tests for signals/timeout.py."""
import time

from ahf.signals.signal_types import SignalOutput
from ahf.signals.timeout import produce_with_timeout


class _FastProducer:
    name = "fast"

    def produce(self, market_data, context):
        return SignalOutput(signal_name="fast", action=0.5, confidence=0.8)


class _SlowProducer:
    name = "slow"

    def produce(self, market_data, context):
        time.sleep(10)  # Will be abandoned by timeout
        return SignalOutput(signal_name="slow", action=0.5, confidence=0.8)


class _ErrorProducer:
    name = "error"

    def produce(self, market_data, context):
        raise RuntimeError("Simulated producer failure")


def test_fast_producer_returns_signal():
    signal, timed_out, error = produce_with_timeout(_FastProducer(), {}, {}, timeout_seconds=5.0)
    assert signal is not None
    assert signal.action == 0.5
    assert timed_out is False
    assert error is None


def test_slow_producer_times_out():
    signal, timed_out, error = produce_with_timeout(_SlowProducer(), {}, {}, timeout_seconds=0.2)
    assert signal is None
    assert timed_out is True
    assert error is None


def test_error_producer_returns_none_with_error():
    signal, timed_out, error = produce_with_timeout(_ErrorProducer(), {}, {}, timeout_seconds=5.0)
    assert signal is None
    assert timed_out is False
    assert error is not None
    assert "Simulated producer failure" in str(error)
