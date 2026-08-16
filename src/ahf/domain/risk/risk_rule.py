"""RiskRule ABC — pluggable pre-trade check."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ahf.domain.risk.risk_types import PortfolioSnapshot, RiskRuleResult


class RiskRule(ABC):
    """Abstract base class for pre-trade risk rules.

    A rule receives a portfolio snapshot and the proposed trade size/direction
    and returns a verdict: ALLOW, VETO, or REDUCE.

    Rules are stateless: all necessary context is passed as arguments.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for logging and audit."""
        ...

    @abstractmethod
    def evaluate(
        self,
        portfolio: PortfolioSnapshot,
        proposed_action: float,
        proposed_size: float,
    ) -> RiskRuleResult:
        """Evaluate pre-trade risk.

        Args:
            portfolio: Current portfolio state (read-only).
            proposed_action: Normalised action in [-1, 1].
                Negative = sell/short, positive = buy/long.
            proposed_size: Proposed trade size as a fraction of portfolio
                value, in (0, 1]. 1.0 = full portfolio.

        Returns:
            RiskRuleResult with ALLOW / VETO / REDUCE verdict.
        """
        ...
