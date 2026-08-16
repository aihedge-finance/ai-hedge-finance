"""RiskManager — runs all registered rules and returns a consolidated verdict.

Design:
- Rules run in registration order.
- First VETO wins — trade is blocked, remaining rules are skipped.
- REDUCE returns the minimum suggested size across all REDUCE rules.
- If all rules ALLOW → trade proceeds with original proposed size.
"""
from __future__ import annotations

import logging
from decimal import Decimal

from ahf.domain.risk.risk_rule import RiskRule
from ahf.domain.risk.risk_types import PortfolioSnapshot, RiskRuleResult, RiskVerdict

logger = logging.getLogger(__name__)


class RiskManager:
    """Orchestrates pre-trade risk rule evaluation.

    Usage:
        rm = RiskManager()
        rm.add_rule(MaxDrawdownRule(0.15))
        rm.add_rule(TotalLossRule(0.30))
        rm.add_rule(KellyRule(fraction=0.5))

        verdict, size, results = rm.evaluate(portfolio, action=0.7, size=0.1)
        if verdict == RiskVerdict.VETO:
            # Skip trade
        elif verdict == RiskVerdict.REDUCE:
            # Use `size` (adjusted)
    """

    def __init__(self) -> None:
        self._rules: list[RiskRule] = []

    def add_rule(self, rule: RiskRule) -> "RiskManager":
        """Register a risk rule. Returns self for chaining."""
        self._rules.append(rule)
        return self

    def evaluate(
        self,
        portfolio: PortfolioSnapshot,
        proposed_action: float,
        proposed_size: float,
    ) -> tuple[RiskVerdict, float, list[RiskRuleResult]]:
        """Evaluate all rules against the proposed trade.

        Returns:
            (verdict, effective_size, rule_results)
            - verdict: ALLOW, VETO, or REDUCE
            - effective_size: The (possibly reduced) position size to use.
              Same as proposed_size unless one or more rules returned REDUCE.
            - rule_results: All individual rule results for audit logging.
        """
        results: list[RiskRuleResult] = []
        reduce_suggestions: list[Decimal] = []

        for rule in self._rules:
            try:
                result = rule.evaluate(portfolio, proposed_action, proposed_size)
                results.append(result)

                if result.verdict == RiskVerdict.VETO:
                    logger.warning(
                        "Trade VETOED by risk rule",
                        extra={"rule": rule.name, "reason": result.reason},
                    )
                    return RiskVerdict.VETO, 0.0, results

                if result.verdict == RiskVerdict.REDUCE:
                    reduce_suggestions.append(result.suggested_size)
                    logger.info(
                        "Trade size REDUCED by risk rule",
                        extra={
                            "rule": rule.name,
                            "suggested": float(result.suggested_size),
                            "reason": result.reason,
                        },
                    )

            except Exception as e:
                logger.error(
                    "Risk rule evaluation error — skipping rule",
                    extra={"rule": rule.name, "error": str(e)},
                )
                # Don't veto on rule error — add a failed result and continue
                results.append(
                    RiskRuleResult(
                        rule_name=rule.name,
                        verdict=RiskVerdict.ALLOW,
                        reason=f"Rule evaluation error: {e}",
                    )
                )

        if reduce_suggestions:
            # Use the most conservative (smallest) suggestion
            effective_size = float(min(reduce_suggestions))
            return RiskVerdict.REDUCE, effective_size, results

        return RiskVerdict.ALLOW, proposed_size, results

    def evaluate_to_bool(
        self,
        portfolio: PortfolioSnapshot,
        proposed_action: float,
        proposed_size: float,
    ) -> tuple[bool, float]:
        """Simplified interface: returns (ok, effective_size).

        ok=False means VETO. ok=True means proceed (with effective_size).
        """
        verdict, size, _ = self.evaluate(portfolio, proposed_action, proposed_size)
        return verdict != RiskVerdict.VETO, size
