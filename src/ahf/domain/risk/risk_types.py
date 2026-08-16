"""Risk domain types.

These types flow through the risk manager and are consumed by
the order executor and position tracker.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum


class RiskVerdict(str, Enum):
    """Outcome of a risk rule evaluation."""

    ALLOW  = "ALLOW"   # Rule passes — trade is permitted
    VETO   = "VETO"    # Rule fails  — trade is blocked
    REDUCE = "REDUCE"  # Rule recommends reducing position size


@dataclass(frozen=True)
class RiskRuleResult:
    """Result from a single RiskRule.evaluate() call."""

    rule_name: str
    verdict: RiskVerdict
    reason: str = ""
    suggested_size: Decimal = Decimal("0")  # Only meaningful when verdict=REDUCE


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Read-only portfolio state passed into risk rules."""

    cash: Decimal
    position_value: Decimal      # Mark-to-market value of current position
    unrealised_pnl: Decimal
    peak_portfolio_value: Decimal
    current_drawdown_pct: float  # 0.0 = no drawdown, 0.5 = -50%
    step: int = 0

    @property
    def total_value(self) -> Decimal:
        return self.cash + self.position_value

    @property
    def is_in_drawdown(self) -> bool:
        return self.current_drawdown_pct > 0.0
