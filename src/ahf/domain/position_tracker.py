"""PositionTracker — tracks portfolio state and computes drawdown.

In v2, the PositionTracker is a thin layer that reads state from the
ExchangeAdapter (for live/paper) or from env.exch_env.ds (for RL
simulation). It never owns state directly — it delegates to its source.

CRITICAL CONSTRAINT (from functional_preservation_report.md):
  "PositionTracker in v2 must delegate state access to env.exch_env.ds
   to remain compatible with existing RL environment logic."

For the live/paper path, ExchangeAdapter is used.
For the RL simulation path, set `rl_env` and state is read from there.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from ahf.adapters.exchange.exchange_adapter import ExchangeAdapter
from ahf.core.types import d
from ahf.domain.risk.risk_types import PortfolioSnapshot


class PositionTracker:
    """Portfolio state tracker with drawdown computation.

    Args:
        exchange: Live/paper exchange adapter (used when rl_env is None).
        symbol: Trading pair to track.
        initial_cash: Starting portfolio value (for TotalLossRule baseline).
        rl_env: Optional RL environment object.
            If set, state is read from env.exch_env.ds (v1 compatibility).
    """

    def __init__(
        self,
        exchange: ExchangeAdapter,
        symbol: str,
        initial_cash: Decimal,
        rl_env: Any = None,
    ) -> None:
        self._exchange = exchange
        self._symbol = symbol
        self._initial_cash = initial_cash
        self._rl_env = rl_env
        self._peak_value: Decimal = initial_cash
        self._step: int = 0

    def snapshot(self) -> PortfolioSnapshot:
        """Return a read-only portfolio snapshot for risk rule evaluation."""
        if self._rl_env is not None:
            return self._snapshot_from_rl_env()
        return self._snapshot_from_exchange()

    def tick(self) -> None:
        """Advance the step counter and update peak portfolio value."""
        self._step += 1
        snap = self.snapshot()
        if snap.total_value > self._peak_value:
            self._peak_value = snap.total_value

    def _snapshot_from_exchange(self) -> PortfolioSnapshot:
        """Read state from the ExchangeAdapter (live/paper path)."""
        price = self._exchange.get_price(self._symbol)
        balance = self._exchange.get_balance()
        position = self._exchange.get_position(self._symbol)
        position_value = position * price

        total = balance + position_value
        if self._peak_value < total:
            self._peak_value = total

        drawdown_pct = (
            float((self._peak_value - total) / self._peak_value)
            if self._peak_value > Decimal("0")
            else 0.0
        )

        return PortfolioSnapshot(
            cash=balance,
            position_value=position_value,
            unrealised_pnl=position_value - (self._initial_cash - balance),
            peak_portfolio_value=self._peak_value,
            current_drawdown_pct=drawdown_pct,
            step=self._step,
        )

    def _snapshot_from_rl_env(self) -> PortfolioSnapshot:
        """Read state from env.exch_env.ds (RL environment path).

        Accesses the data structure that BrunhildEnv_v11 uses internally.
        This preserves v1 compatibility — RL environment state is the
        single source of truth during RL training/inference.
        """
        ds = self._rl_env.exch_env.ds  # v1 data store access pattern
        cash = d(ds.get("cash", self._initial_cash))
        position = d(ds.get("position", 0))
        price = d(ds.get("price", 0))
        position_value = position * price
        total = cash + position_value

        if self._peak_value < total:
            self._peak_value = total

        drawdown_pct = (
            float((self._peak_value - total) / self._peak_value)
            if self._peak_value > Decimal("0")
            else 0.0
        )

        return PortfolioSnapshot(
            cash=cash,
            position_value=position_value,
            unrealised_pnl=d(ds.get("unrealised_pnl", 0)),
            peak_portfolio_value=self._peak_value,
            current_drawdown_pct=drawdown_pct,
            step=self._step,
        )
