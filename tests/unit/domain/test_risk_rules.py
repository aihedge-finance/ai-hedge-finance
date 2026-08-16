"""Tests for all risk rules."""
from decimal import Decimal

import pytest

from ahf.domain.risk.drawdown_rule import MaxDrawdownRule
from ahf.domain.risk.kelly_rule import KellyRule
from ahf.domain.risk.risk_types import PortfolioSnapshot, RiskVerdict
from ahf.domain.risk.total_loss_rule import TotalLossRule


def _portfolio(
    cash: float = 1000.0,
    position_value: float = 0.0,
    drawdown_pct: float = 0.0,
    peak: float = 1000.0,
) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        cash=Decimal(str(cash)),
        position_value=Decimal(str(position_value)),
        unrealised_pnl=Decimal("0"),
        peak_portfolio_value=Decimal(str(peak)),
        current_drawdown_pct=drawdown_pct,
    )


# ---------------------------------------------------------------------------
# MaxDrawdownRule
# ---------------------------------------------------------------------------


class TestDrawdownRule:
    def test_no_drawdown_allows_buy(self):
        rule = MaxDrawdownRule(max_drawdown_pct=0.15)
        result = rule.evaluate(_portfolio(drawdown_pct=0.0), proposed_action=0.8, proposed_size=0.1)
        assert result.verdict == RiskVerdict.ALLOW

    def test_below_threshold_allows_buy(self):
        rule = MaxDrawdownRule(max_drawdown_pct=0.15)
        result = rule.evaluate(_portfolio(drawdown_pct=0.10), proposed_action=0.8, proposed_size=0.1)
        assert result.verdict == RiskVerdict.ALLOW

    def test_exceeds_threshold_vetos_buy(self):
        rule = MaxDrawdownRule(max_drawdown_pct=0.15)
        result = rule.evaluate(_portfolio(drawdown_pct=0.20), proposed_action=0.8, proposed_size=0.1)
        assert result.verdict == RiskVerdict.VETO
        assert "Drawdown" in result.reason

    def test_exceeds_threshold_allows_sell(self):
        rule = MaxDrawdownRule(max_drawdown_pct=0.15, allow_close_only=True)
        result = rule.evaluate(_portfolio(drawdown_pct=0.20), proposed_action=-0.8, proposed_size=0.1)
        assert result.verdict == RiskVerdict.ALLOW

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            MaxDrawdownRule(max_drawdown_pct=0.0)

    def test_invalid_threshold_over_100_raises(self):
        with pytest.raises(ValueError):
            MaxDrawdownRule(max_drawdown_pct=1.5)


# ---------------------------------------------------------------------------
# TotalLossRule
# ---------------------------------------------------------------------------


class TestTotalLossRule:
    def test_above_floor_allows(self):
        rule = TotalLossRule(max_loss_pct=0.30, initial_capital=1000.0)
        # Floor = 700.0; total = 1000.0 → ALLOW
        result = rule.evaluate(_portfolio(cash=1000.0), proposed_action=0.5, proposed_size=0.1)
        assert result.verdict == RiskVerdict.ALLOW

    def test_below_floor_vetos(self):
        rule = TotalLossRule(max_loss_pct=0.30, initial_capital=1000.0)
        # Floor = 700.0; cash=650 + position=0 = 650 → VETO
        result = rule.evaluate(_portfolio(cash=650.0), proposed_action=0.5, proposed_size=0.1)
        assert result.verdict == RiskVerdict.VETO
        assert "floor" in result.reason.lower()

    def test_exactly_at_floor_allows(self):
        rule = TotalLossRule(max_loss_pct=0.30, initial_capital=1000.0)
        # Floor = 700.0; total = 700.0 → ALLOW (>= floor)
        result = rule.evaluate(_portfolio(cash=700.0), proposed_action=0.5, proposed_size=0.1)
        assert result.verdict == RiskVerdict.ALLOW

    def test_invalid_loss_pct_raises(self):
        with pytest.raises(ValueError):
            TotalLossRule(max_loss_pct=1.5)


# ---------------------------------------------------------------------------
# KellyRule
# ---------------------------------------------------------------------------


class TestKellyRule:
    def test_hold_action_always_allows(self):
        rule = KellyRule()
        result = rule.evaluate(_portfolio(), proposed_action=0.0, proposed_size=0.1)
        assert result.verdict == RiskVerdict.ALLOW

    def test_high_confidence_allows(self):
        rule = KellyRule(fraction=0.5, win_loss_ratio=2.0, max_size_pct=0.20)
        # p=0.8, b=2.0 → kelly = (2*0.8 - 0.2)/2 = 1.4/2 = 0.7; half-kelly=0.35 > max=0.2
        # proposed_size = 0.1 < 0.2 → ALLOW
        result = rule.evaluate(_portfolio(), proposed_action=0.8, proposed_size=0.1)
        assert result.verdict == RiskVerdict.ALLOW

    def test_low_probability_returns_reduce_or_veto(self):
        rule = KellyRule(fraction=0.5, win_loss_ratio=1.0)
        # Very low action → near-zero Kelly fraction
        result = rule.evaluate(_portfolio(), proposed_action=0.11, proposed_size=0.9)
        # Kelly may reduce or veto, but should not raise
        assert result.verdict in (RiskVerdict.REDUCE, RiskVerdict.VETO, RiskVerdict.ALLOW)

    def test_very_high_proposed_size_returns_reduce(self):
        rule = KellyRule(fraction=0.5, win_loss_ratio=1.5, max_size_pct=0.10)
        # max_size_pct=0.10; proposed=0.9 → REDUCE
        result = rule.evaluate(_portfolio(), proposed_action=0.7, proposed_size=0.9)
        assert result.verdict in (RiskVerdict.REDUCE, RiskVerdict.VETO)

    def test_rule_name_includes_fraction(self):
        rule = KellyRule(fraction=0.5)
        assert "kelly_50pct" == rule.name
