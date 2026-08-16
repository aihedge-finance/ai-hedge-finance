"""Tests for domain/signal_processor.py."""
import pytest

from ahf.core.enums import TradeAction
from ahf.domain.signal_processor import SignalProcessor
from ahf.signals.signal_types import SignalOutput


def _sig(action: float, confidence: float = 0.8) -> SignalOutput:
    return SignalOutput(signal_name="test", action=action, confidence=confidence)


class TestSignalProcessor:
    def setup_method(self):
        self.sp = SignalProcessor(buy_threshold=0.1, sell_threshold=-0.1)

    def test_strong_buy(self):
        assert self.sp.process(_sig(0.8)) == TradeAction.BUY

    def test_boundary_buy(self):
        assert self.sp.process(_sig(0.1)) == TradeAction.BUY

    def test_just_below_buy_threshold(self):
        assert self.sp.process(_sig(0.09)) == TradeAction.HOLD

    def test_strong_sell(self):
        assert self.sp.process(_sig(-0.8)) == TradeAction.SELL

    def test_boundary_sell(self):
        assert self.sp.process(_sig(-0.1)) == TradeAction.SELL

    def test_just_above_sell_threshold(self):
        assert self.sp.process(_sig(-0.09)) == TradeAction.HOLD

    def test_zero_is_hold(self):
        assert self.sp.process(_sig(0.0, confidence=0.3)) == TradeAction.HOLD

    def test_confidence_floor_forces_hold(self):
        sp = SignalProcessor(buy_threshold=0.1, sell_threshold=-0.1, confidence_floor=0.5)
        assert sp.process(_sig(0.9, confidence=0.3)) == TradeAction.HOLD

    def test_confidence_above_floor_acts(self):
        sp = SignalProcessor(buy_threshold=0.1, sell_threshold=-0.1, confidence_floor=0.5)
        assert sp.process(_sig(0.9, confidence=0.8)) == TradeAction.BUY

    def test_process_to_int_buy(self):
        assert self.sp.process_to_int(_sig(0.8)) == 1

    def test_process_to_int_sell(self):
        assert self.sp.process_to_int(_sig(-0.8)) == -1

    def test_process_to_int_hold(self):
        assert self.sp.process_to_int(_sig(0.0, confidence=0.3)) == 0

    def test_invalid_buy_threshold_raises(self):
        with pytest.raises(ValueError):
            SignalProcessor(buy_threshold=-0.1, sell_threshold=-0.2)

    def test_invalid_sell_threshold_raises(self):
        with pytest.raises(ValueError):
            SignalProcessor(buy_threshold=0.1, sell_threshold=0.1)
