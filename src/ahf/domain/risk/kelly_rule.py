"""KellyRule — size trades using the Kelly Criterion.

The Kelly Criterion determines the optimal fraction of capital to bet
given a win probability p and win/loss ratio b:
    f* = (b*p - (1-p)) / b

In practice we use a fractional Kelly (half-Kelly or quarter-Kelly)
to reduce variance while retaining most of the expected growth.

The rule scales `proposed_size` using the signal confidence as a proxy
for win probability. If the Kelly-adjusted size is significantly smaller
than the proposed size, it returns REDUCE.

References:
- design/pre_upgrade_v2_analysis/risk_sizing_stoploss_analysis.md
"""
from __future__ import annotations

from decimal import Decimal

from ahf.domain.risk.risk_rule import RiskRule
from ahf.domain.risk.risk_types import PortfolioSnapshot, RiskRuleResult, RiskVerdict


class KellyRule(RiskRule):
    """Kelly Criterion position sizer.

    Args:
        fraction: Fractional Kelly multiplier. 1.0 = full Kelly, 0.5 = half-Kelly.
        win_loss_ratio: Assumed average win / average loss ratio.
            Conservative default: 1.5 (win = 1.5x the average loss).
        max_size_pct: Hard cap on position size as fraction of portfolio.
            Prevents Kelly from sizing too aggressively in high-confidence scenarios.
    """

    def __init__(
        self,
        fraction: float = 0.5,
        win_loss_ratio: float = 1.5,
        max_size_pct: float = 0.20,
    ) -> None:
        self._fraction = fraction
        self._win_loss_ratio = win_loss_ratio
        self._max_size_pct = max_size_pct

    @property
    def name(self) -> str:
        return f"kelly_{int(self._fraction * 100)}pct"

    def evaluate(
        self,
        portfolio: PortfolioSnapshot,
        proposed_action: float,
        proposed_size: float,
    ) -> RiskRuleResult:
        if proposed_action == 0.0:
            return RiskRuleResult(rule_name=self.name, verdict=RiskVerdict.ALLOW)

        # Use |action| as win probability proxy (RL outputs ∈ [-1, 1])
        p = min(max(abs(proposed_action), 0.01), 0.99)
        b = self._win_loss_ratio
        kelly_f = (b * p - (1.0 - p)) / b
        kelly_f = max(0.0, kelly_f)  # Negative Kelly → 0 (no trade)
        adjusted = min(kelly_f * self._fraction, self._max_size_pct)

        if adjusted <= 0.0:
            return RiskRuleResult(
                rule_name=self.name,
                verdict=RiskVerdict.VETO,
                reason=f"Kelly fraction non-positive (p={p:.2f}, b={b:.2f}) — skip trade",
            )

        if adjusted < proposed_size * 0.8:
            # Kelly recommends meaningfully smaller size
            return RiskRuleResult(
                rule_name=self.name,
                verdict=RiskVerdict.REDUCE,
                reason=f"Kelly({self._fraction}x) = {adjusted:.3f} < proposed {proposed_size:.3f}",
                suggested_size=Decimal(str(round(adjusted, 6))),
            )

        return RiskRuleResult(rule_name=self.name, verdict=RiskVerdict.ALLOW)
