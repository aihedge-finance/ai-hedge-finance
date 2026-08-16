"""SignalAggregator ABC — Layer 2 of the signal pipeline.

The aggregator receives all SignalOutput values from Layer 1 producers
(including None for failed/timed-out producers) and produces a single
consensus SignalOutput that the orchestrator uses to make a trade decision.

Design reference: design/pre_upgrade_v2_analysis/option_ab_architecture.md
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ahf.signals.signal_types import SignalOutput


class SignalAggregator(ABC):
    """Abstract base class for all Layer-2 aggregators.

    The aggregator must handle None values gracefully — these represent
    producers that failed or timed out. Re-normalise weights over the
    remaining valid signals.
    """

    @abstractmethod
    def aggregate(
        self,
        signals: list[Optional[SignalOutput]],
        context: dict,
    ) -> SignalOutput:
        """Combine multiple producer signals into one consensus signal.

        Args:
            signals: List of SignalOutput from all configured producers.
                None entries represent failed or timed-out producers.
                The list order matches the order of producers in pipeline.json.
            context: Shared runtime context from the orchestrator.
                May contain "producer_ids" (list[str]) in the same order.

        Returns:
            A single SignalOutput representing the aggregated view.
            signal_name should be set to this aggregator's identifier.

        Note:
            If ALL signals are None, implementations should return a
            HOLD signal with low confidence rather than raising.
        """
        ...

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        config: dict,
        runtime_deps: dict,
    ) -> "SignalAggregator":
        """Factory method: build an aggregator from JSON config + runtime objects.

        Args:
            config: The aggregator's 'config' dict from pipeline.json.
            runtime_deps: Runtime objects (e.g. LLM client for MetaLLM).

        Returns:
            A configured, ready-to-use SignalAggregator instance.
        """
        ...
