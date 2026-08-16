"""TotalLossRule — hard stop: veto all new trades if total loss % is exceeded."""
from __future__ import annotations

from ahf.domain.risk.risk_rule import RiskRule
from ahf.domain.risk.risk_types import PortfolioSnapshot, RiskRuleResult, RiskVerdict


class TotalLossRule(RiskRule):
    """Absolute loss limit: compare current portfolio value to initial capital.

    Unlike MaxDrawdownRule (which tracks peak → current), TotalLossRule
    compares absolute current value to the initial cash at bot start.
    Useful as a hard account-level stop-loss.

    Args:
        max_loss_pct: Fraction of initial capital that can be lost.
            Example: 0.30 = stop all trading if total account is down 30%.
        initial_capital: The starting portfolio value for comparison.
    """

    def __init__(self, max_loss_pct: float = 0.30, initial_capital: float = 1000.0) -> None:
        if not 0.0 < max_loss_pct <= 1.0:
            raise ValueError(f"max_loss_pct must be in (0, 1], got {max_loss_pct}")
        self._max_loss_pct = max_loss_pct
        self._initial_capital = initial_capital
        self._floor = initial_capital * (1.0 - max_loss_pct)

    @property
    def name(self) -> str:
        return f"total_loss_{int(self._max_loss_pct * 100)}pct"

    def evaluate(
        self,
        portfolio: PortfolioSnapshot,
        proposed_action: float,
        proposed_size: float,
    ) -> RiskRuleResult:
        total = float(portfolio.total_value)

        if total >= self._floor:
            return RiskRuleResult(rule_name=self.name, verdict=RiskVerdict.ALLOW)

        return RiskRuleResult(
            rule_name=self.name,
            verdict=RiskVerdict.VETO,
            reason=(
                f"Total portfolio value {total:.2f} is below floor "
                f"{self._floor:.2f} (max loss {self._max_loss_pct:.0%} of "
                f"initial {self._initial_capital:.2f})"
            ),
        )
