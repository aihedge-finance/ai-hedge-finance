"""Tests for domain/order_executor.py + DummyAdapter."""
from decimal import Decimal

from ahf.adapters.exchange.dummy_adapter import DummyAdapter
from ahf.core.enums import TradeAction
from ahf.domain.order_executor import OrderExecutor


def _make_executor(balance: float = 1000.0, price: float = 67000.0) -> tuple[OrderExecutor, DummyAdapter]:
    adapter = DummyAdapter(
        initial_balance=Decimal(str(balance)),
        initial_price=Decimal(str(price)),
        slippage_pct=0.0,  # No slippage for deterministic tests
    )
    executor = OrderExecutor(adapter, symbol="BTCUSDT", max_position_fraction=0.95, quantity_precision=5)
    return executor, adapter


class TestOrderExecutor:
    def test_hold_returns_none(self):
        executor, _ = _make_executor()
        result = executor.execute(TradeAction.HOLD, signal_strength=0.8)
        assert result is None

    def test_buy_reduces_balance(self):
        executor, adapter = _make_executor(balance=1000.0, price=1000.0)
        executor.execute(TradeAction.BUY, signal_strength=1.0)
        # 95% of 1000 = 950 → buy 0.95 BTC at 1000 → balance ≈ 50
        assert adapter.get_balance() < Decimal("100")

    def test_buy_increases_position(self):
        executor, adapter = _make_executor(balance=1000.0, price=1000.0)
        executor.execute(TradeAction.BUY, signal_strength=1.0)
        position = adapter.get_position("BTCUSDT")
        assert position > Decimal("0")

    def test_sell_increases_balance(self):
        executor, adapter = _make_executor(balance=1000.0, price=1000.0)
        executor.execute(TradeAction.BUY, signal_strength=1.0)
        balance_after_buy = adapter.get_balance()
        executor.execute(TradeAction.SELL, signal_strength=1.0)
        assert adapter.get_balance() > balance_after_buy

    def test_sell_with_no_position_returns_none(self):
        executor, adapter = _make_executor()
        result = executor.execute(TradeAction.SELL, signal_strength=1.0)
        assert result is None

    def test_buy_with_tiny_size_skipped(self):
        executor, adapter = _make_executor(balance=1000.0, price=1000.0)
        # signal_strength=0.00001 → target_value = 1000*0.95*0.00001 = 0.0095 < min_notional(10)
        result = executor.execute(TradeAction.BUY, signal_strength=0.00001)
        assert result is None

    def test_dummy_adapter_is_simulated(self):
        adapter = DummyAdapter()
        assert adapter.is_simulated() is True

    def test_dummy_adapter_order_status(self):
        executor, adapter = _make_executor(balance=1000.0, price=1000.0)
        order = executor.execute(TradeAction.BUY, signal_strength=1.0)
        assert order is not None
        status = adapter.get_order_status("BTCUSDT", order["orderId"])
        assert status["status"] == "FILLED"

    def test_dummy_adapter_cancel_order(self):
        executor, adapter = _make_executor(balance=1000.0, price=1000.0)
        order = executor.execute(TradeAction.BUY, signal_strength=1.0)
        assert order is not None
        ok = adapter.cancel_order("BTCUSDT", order["orderId"])
        assert ok is True
        status = adapter.get_order_status("BTCUSDT", order["orderId"])
        assert status["status"] == "CANCELED"
