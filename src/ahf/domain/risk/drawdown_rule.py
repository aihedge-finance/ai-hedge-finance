"""MaxDrawdownRule — veto trades when drawdown exceeds threshold."""
from __future__ import annotations

from ahf.domain.risk.risk_rule import RiskRule
from ahf.domain.risk.risk_types import PortfolioSnapshot, RiskRuleResult, RiskVerdict


class MaxDrawdownRule(RiskRule):
    """Halt all new buy/sell trades when portfolio drawdown exceeds a threshold.

    When in breach:
    - BUY trades are VETOED (don't add exposure during a drawdown)
    - SELL trades are ALLOWED (closing/reducing positions is safe)
    - HOLD trades are always ALLOWED

    Args:
        max_drawdown_pct: Drawdown threshold in [0, 1].
            Example: 0.15 = halt trading if down 15% from peak.
        allow_close_only: If True (default), allow sells even during drawdown.
    """

    def __init__(self, max_drawdown_pct: float = 0.15, allow_close_only: bool = True) -> None:
        if not 0.0 < max_drawdown_pct <= 1.0:
            raise ValueError(f"max_drawdown_pct must be in (0, 1], got {max_drawdown_pct}")
        self._threshold = max_drawdown_pct
        self._allow_close_only = allow_close_only

    @property
    def name(self) -> str:
        return f"max_drawdown_{int(self._threshold * 100)}pct"

    def evaluate(
        self,
        portfolio: PortfolioSnapshot,
        proposed_action: float,
        proposed_size: float,
    ) -> RiskRuleResult:
        if portfolio.current_drawdown_pct <= self._threshold:
            return RiskRuleResult(rule_name=self.name, verdict=RiskVerdict.ALLOW)

        # Drawdown breach
        if proposed_action > 0:
            # BUY during drawdown → VETO
            return RiskRuleResult(
                rule_name=self.name,
                verdict=RiskVerdict.VETO,
                reason=(
                    f"Drawdown {portfolio.current_drawdown_pct:.1%} exceeds "
                    f"threshold {self._threshold:.1%} — blocking new buys"
                ),
            )

        if proposed_action < 0 and self._allow_close_only:
            # SELL during drawdown → ALLOW (close / reduce exposure)
            return RiskRuleResult(rule_name=self.name, verdict=RiskVerdict.ALLOW)

        return RiskRuleResult(rule_name=self.name, verdict=RiskVerdict.ALLOW)
