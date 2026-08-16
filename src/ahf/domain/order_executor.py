"""OrderExecutor — translates TradeAction → exchange orders.

Applies position sizing and delegates actual order submission to the
ExchangeAdapter. This module owns the buy/sell logic but has no
direct dependency on any exchange SDK.

Design reference: design/pre_upgrade_v2_analysis/functional_preservation_report.md
"""
from __future__ import annotations

import logging
from decimal import Decimal

from ahf.adapters.exchange.exchange_adapter import ExchangeAdapter
from ahf.core.enums import TradeAction
from ahf.core.types import d_round

logger = logging.getLogger(__name__)

_MIN_NOTIONAL = Decimal("10.00")  # Minimum order value in quote currency


class OrderExecutor:
    """Translates TradeAction → exchange API calls.

    Args:
        exchange: The exchange adapter to use for order submission.
        symbol: Trading pair (e.g. "BTCUSDT").
        max_position_fraction: Maximum fraction of portfolio to hold in one position.
            0.95 = can hold at most 95% of portfolio in the asset.
        quantity_precision: Number of decimal places for quantity rounding.
    """

    def __init__(
        self,
        exchange: ExchangeAdapter,
        symbol: str,
        max_position_fraction: float = 0.95,
        quantity_precision: int = 5,
    ) -> None:
        self._exchange = exchange
        self._symbol = symbol
        self._max_position_fraction = Decimal(str(max_position_fraction))
        self._quantity_precision = quantity_precision

    def execute(
        self,
        action: TradeAction,
        signal_strength: float,
    ) -> dict | None:
        """Execute a trade action.

        Args:
            action: BUY, SELL, or HOLD.
            signal_strength: Action magnitude from signal (|action|), used to
                scale position size. In [0.0, 1.0].

        Returns:
            Order response dict from exchange, or None if no order was placed.
        """
        if action == TradeAction.HOLD:
            return None

        price = self._exchange.get_price(self._symbol)
        position = self._exchange.get_position(self._symbol)

        if price <= Decimal("0"):
            logger.error("Invalid price — skipping order", extra={"price": str(price)})
            return None

        if action == TradeAction.BUY:
            return self._execute_buy(price, signal_strength)
        else:
            return self._execute_sell(price, position, signal_strength)

    def _execute_buy(self, price: Decimal, signal_strength: float) -> dict | None:
        balance = self._exchange.get_balance()
        # Scale order value by signal strength * max_position_fraction
        target_value = balance * self._max_position_fraction * Decimal(str(signal_strength))

        if target_value < _MIN_NOTIONAL:
            logger.info(
                "Buy order too small — skipping",
                extra={"target_value": str(target_value), "min_notional": str(_MIN_NOTIONAL)},
            )
            return None

        quantity = d_round(target_value / price, self._quantity_precision)
        if quantity <= Decimal("0"):
            return None

        logger.info(
            "Placing BUY order",
            extra={"symbol": self._symbol, "quantity": str(quantity), "price": str(price)},
        )
        return self._exchange.place_order(self._symbol, "BUY", quantity)

    def _execute_sell(
        self, price: Decimal, position: Decimal, signal_strength: float
    ) -> dict | None:
        if position <= Decimal("0"):
            logger.info("No position to sell — skipping")
            return None

        # Sell a fraction of position proportional to signal strength
        sell_qty = d_round(position * Decimal(str(signal_strength)), self._quantity_precision)
        if sell_qty <= Decimal("0"):
            return None

        logger.info(
            "Placing SELL order",
            extra={"symbol": self._symbol, "quantity": str(sell_qty), "price": str(price)},
        )
        return self._exchange.place_order(self._symbol, "SELL", sell_qty)
