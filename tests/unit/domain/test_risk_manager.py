"""Tests for the RiskManager orchestration."""
from decimal import Decimal

from ahf.domain.risk.drawdown_rule import MaxDrawdownRule
from ahf.domain.risk.kelly_rule import KellyRule
from ahf.domain.risk.risk_manager import RiskManager
from ahf.domain.risk.risk_types import PortfolioSnapshot, RiskRuleResult, RiskVerdict
from ahf.domain.risk.total_loss_rule import TotalLossRule


def _portfolio(cash: float = 1000.0, drawdown_pct: float = 0.0) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        cash=Decimal(str(cash)),
        position_value=Decimal("0"),
        unrealised_pnl=Decimal("0"),
        peak_portfolio_value=Decimal(str(cash)),
        current_drawdown_pct=drawdown_pct,
    )


def test_empty_manager_allows_everything():
    rm = RiskManager()
    verdict, size, results = rm.evaluate(_portfolio(), 0.5, 0.1)
    assert verdict == RiskVerdict.ALLOW
    assert size == 0.1
    assert results == []


def test_first_veto_stops_evaluation():
    """When first rule vetoes, remaining rules are not evaluated."""
    rm = RiskManager()
    rm.add_rule(MaxDrawdownRule(max_drawdown_pct=0.05))  # Will VETO at 10% drawdown
    rm.add_rule(TotalLossRule(max_loss_pct=0.30, initial_capital=1000.0))  # Would ALLOW

    verdict, size, results = rm.evaluate(_portfolio(drawdown_pct=0.10), 0.8, 0.1)
    assert verdict == RiskVerdict.VETO
    assert size == 0.0
    assert len(results) == 1  # Only the first rule was evaluated


def test_reduce_returns_minimum_suggestion():
    """When multiple REDUCE rules conflict, use the most conservative size."""

    class ReduceRule1(MaxDrawdownRule.__mro__[0]):
        @property
        def name(self): return "reduce_to_05"
        def evaluate(self, p, a, s):
            return RiskRuleResult(self.name, RiskVerdict.REDUCE, suggested_size=Decimal("0.05"))

    class ReduceRule2(MaxDrawdownRule.__mro__[0]):
        @property
        def name(self): return "reduce_to_03"
        def evaluate(self, p, a, s):
            return RiskRuleResult(self.name, RiskVerdict.REDUCE, suggested_size=Decimal("0.03"))

    from ahf.domain.risk.risk_rule import RiskRule

    class _R1(RiskRule):
        @property
        def name(self): return "r1"
        def evaluate(self, p, a, s):
            return RiskRuleResult("r1", RiskVerdict.REDUCE, suggested_size=Decimal("0.05"))

    class _R2(RiskRule):
        @property
        def name(self): return "r2"
        def evaluate(self, p, a, s):
            return RiskRuleResult("r2", RiskVerdict.REDUCE, suggested_size=Decimal("0.03"))

    rm = RiskManager()
    rm.add_rule(_R1())
    rm.add_rule(_R2())

    verdict, size, results = rm.evaluate(_portfolio(), 0.5, 0.1)
    assert verdict == RiskVerdict.REDUCE
    assert abs(size - 0.03) < 1e-9


def test_all_rules_allow():
    rm = RiskManager()
    rm.add_rule(MaxDrawdownRule(max_drawdown_pct=0.50))
    rm.add_rule(TotalLossRule(max_loss_pct=0.50, initial_capital=1000.0))
    verdict, size, results = rm.evaluate(_portfolio(), 0.5, 0.1)
    assert verdict == RiskVerdict.ALLOW
    assert size == 0.1


def test_evaluate_to_bool_veto_returns_false():
    rm = RiskManager()
    rm.add_rule(MaxDrawdownRule(max_drawdown_pct=0.05))
    ok, size = rm.evaluate_to_bool(_portfolio(drawdown_pct=0.10), 0.8, 0.1)
    assert ok is False
    assert size == 0.0


def test_evaluate_to_bool_allow_returns_true():
    rm = RiskManager()
    ok, size = rm.evaluate_to_bool(_portfolio(), 0.5, 0.1)
    assert ok is True
    assert size == 0.1


def test_buggy_rule_is_skipped_not_vetoed():
    """A rule that raises an exception should not veto the trade."""
    from ahf.domain.risk.risk_rule import RiskRule

    class BuggyRule(RiskRule):
        @property
        def name(self): return "buggy"
        def evaluate(self, p, a, s):
            raise RuntimeError("I'm broken")

    rm = RiskManager()
    rm.add_rule(BuggyRule())
    verdict, size, results = rm.evaluate(_portfolio(), 0.5, 0.1)
    # Buggy rule → error is caught, treated as ALLOW
    assert verdict == RiskVerdict.ALLOW


def test_chaining_api():
    rm = (
        RiskManager()
        .add_rule(MaxDrawdownRule(0.15))
        .add_rule(TotalLossRule(0.30, 1000.0))
        .add_rule(KellyRule(0.5))
    )
    # 3 rules registered
    assert len(rm._rules) == 3
