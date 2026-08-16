"""Tests for PositionTracker — covers sell/PnL/drawdown paths (Issue 8)."""
from __future__ import annotations

from decimal import Decimal

import pytest

from ahf.adapters.exchange.dummy_adapter import DummyAdapter
from ahf.domain.position_tracker import PositionTracker


@pytest.fixture()
def adapter() -> DummyAdapter:
    return DummyAdapter(initial_balance=Decimal("5000"), initial_price=Decimal("100"))


@pytest.fixture()
def tracker(adapter: DummyAdapter) -> PositionTracker:
    return PositionTracker(adapter, "BTCUSDT", Decimal("5000"))


# ---------------------------------------------------------------------------
# Basic snapshot — no position
# ---------------------------------------------------------------------------


def test_initial_snapshot_all_cash(tracker: PositionTracker, adapter: DummyAdapter) -> None:
    snap = tracker.snapshot()
    assert snap.cash == Decimal("5000")
    assert snap.position_value == Decimal("0")
    assert snap.current_drawdown_pct == pytest.approx(0.0)
    assert snap.step == 0


def test_initial_peak_equals_initial_cash(tracker: PositionTracker) -> None:
    snap = tracker.snapshot()
    assert snap.peak_portfolio_value == Decimal("5000")


# ---------------------------------------------------------------------------
# Snapshot after BUY
# ---------------------------------------------------------------------------


def test_snapshot_after_buy_reflects_position(
    tracker: PositionTracker, adapter: DummyAdapter
) -> None:
    adapter.place_order("BTCUSDT", "BUY", Decimal("5"))  # 5 units @ ~100
    snap = tracker.snapshot()
    assert snap.position_value > Decimal("0")
    assert snap.cash < Decimal("5000")


# ---------------------------------------------------------------------------
# Snapshot after SELL — the critical uncovered path (lines 60, 71)
# ---------------------------------------------------------------------------


def test_snapshot_after_sell_reflects_balance_increase(
    tracker: PositionTracker, adapter: DummyAdapter
) -> None:
    adapter.place_order("BTCUSDT", "BUY", Decimal("5"))
    balance_after_buy = adapter.get_balance()
    adapter.place_order("BTCUSDT", "SELL", Decimal("5"))
    snap = tracker.snapshot()
    # After selling, cash increases and position drops to near zero
    assert adapter.get_balance() > balance_after_buy
    assert snap.position_value == pytest.approx(0, abs=Decimal("0.01"))


def test_sell_partial_position(
    tracker: PositionTracker, adapter: DummyAdapter
) -> None:
    adapter.place_order("BTCUSDT", "BUY", Decimal("10"))
    adapter.place_order("BTCUSDT", "SELL", Decimal("4"))
    position = adapter.get_position("BTCUSDT")
    assert position == pytest.approx(Decimal("6"), abs=Decimal("0.01"))
    snap = tracker.snapshot()
    assert snap.position_value > Decimal("0")


# ---------------------------------------------------------------------------
# PnL / unrealised_pnl calculation (lines 90, 95-111)
# ---------------------------------------------------------------------------


def test_unrealised_pnl_zero_at_start(tracker: PositionTracker) -> None:
    snap = tracker.snapshot()
    assert snap.unrealised_pnl == Decimal("0")


def test_unrealised_pnl_positive_when_price_rises(
    tracker: PositionTracker, adapter: DummyAdapter
) -> None:
    adapter.place_order("BTCUSDT", "BUY", Decimal("5"))
    adapter.set_price(Decimal("200"))  # price doubled
    snap = tracker.snapshot()
    # position_value = 5 * 200 = 1000; spent ~500+slippage; pnl should be positive
    assert snap.unrealised_pnl > Decimal("0")


def test_unrealised_pnl_negative_when_price_falls(
    tracker: PositionTracker, adapter: DummyAdapter
) -> None:
    adapter.place_order("BTCUSDT", "BUY", Decimal("5"))
    adapter.set_price(Decimal("50"))  # price halved
    snap = tracker.snapshot()
    assert snap.unrealised_pnl < Decimal("0")


# ---------------------------------------------------------------------------
# Drawdown calculation (lines 95-111)
# ---------------------------------------------------------------------------


def test_drawdown_zero_at_peak(tracker: PositionTracker) -> None:
    snap = tracker.snapshot()
    assert snap.current_drawdown_pct == pytest.approx(0.0)


def test_drawdown_increases_as_value_falls(
    tracker: PositionTracker, adapter: DummyAdapter
) -> None:
    adapter.place_order("BTCUSDT", "BUY", Decimal("5"))
    adapter.set_price(Decimal("200"))
    tracker.tick()  # updates peak
    adapter.set_price(Decimal("50"))
    snap = tracker.snapshot()
    assert snap.current_drawdown_pct > 0.0


def test_tick_advances_step_counter(tracker: PositionTracker) -> None:
    tracker.tick()
    tracker.tick()
    snap = tracker.snapshot()
    assert snap.step == 2


# ---------------------------------------------------------------------------
# RL env path (lines 95-111 via _snapshot_from_rl_env)
# ---------------------------------------------------------------------------


def test_rl_env_snapshot_path(adapter: DummyAdapter) -> None:
    """Exercise the rl_env snapshot path with a mock datastore."""

    class MockDS:
        def get(self, key: str, default=None):
            data = {"cash": "800", "position": "2", "price": "100", "unrealised_pnl": "50"}
            return data.get(key, default)

    class MockEnv:
        class MockExchEnv:
            ds = MockDS()
        exch_env = MockExchEnv()

    tracker = PositionTracker(adapter, "BTCUSDT", Decimal("1000"), rl_env=MockEnv())
    snap = tracker.snapshot()
    assert snap.cash == Decimal("800")
    assert snap.position_value == Decimal("200")  # 2 * 100
    assert snap.unrealised_pnl == Decimal("50")
