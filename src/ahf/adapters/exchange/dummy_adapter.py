"""DummyAdapter — in-memory exchange simulation for backtesting and tests.

Simulates order execution with configurable slippage and fill latency.
State is stored in-memory only. Suitable for:
- Unit tests
- Paper trading (PAPER mode)
- Backtesting (SIMULATION mode)

Design reference: design/pre_upgrade_v2_analysis/functional_preservation_report.md
"""
from __future__ import annotations

import uuid
from decimal import Decimal

from ahf.adapters.exchange.exchange_adapter import ExchangeAdapter

_DEFAULT_PRICE = Decimal("67000.00")


class DummyAdapter(ExchangeAdapter):
    """Simulated exchange with configurable slippage.

    All orders are filled instantly at current_price * (1 ± slippage).
    """

    def __init__(
        self,
        initial_balance: Decimal = Decimal("1000.00"),
        initial_price: Decimal = _DEFAULT_PRICE,
        slippage_pct: float = 0.0005,  # 0.05% default
    ) -> None:
        self._balance = initial_balance
        self._price = initial_price
        self._slippage = Decimal(str(slippage_pct))
        self._positions: dict[str, Decimal] = {}
        self._orders: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------

    def set_price(self, price: Decimal) -> None:
        """Update the simulated market price."""
        self._price = price

    def credit(self, amount: Decimal) -> None:
        """Add funds to the simulated balance."""
        self._balance += amount

    # ------------------------------------------------------------------
    # ExchangeAdapter interface
    # ------------------------------------------------------------------

    def get_balance(self) -> Decimal:
        return self._balance

    def get_position(self, symbol: str) -> Decimal:
        return self._positions.get(symbol, Decimal("0"))

    def get_price(self, symbol: str) -> Decimal:
        return self._price

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        order_type: str = "MARKET",
        price: Decimal | None = None,
    ) -> dict:
        order_id = str(uuid.uuid4())[:8]
        fill_price = self._price

        if side.upper() == "BUY":
            fill_price = fill_price * (Decimal("1") + self._slippage)
            cost = quantity * fill_price
            if cost > self._balance:
                return {"orderId": order_id, "status": "REJECTED", "reason": "Insufficient balance"}
            self._balance -= cost
            self._positions[symbol] = self._positions.get(symbol, Decimal("0")) + quantity

        elif side.upper() == "SELL":
            fill_price = fill_price * (Decimal("1") - self._slippage)
            current_position = self._positions.get(symbol, Decimal("0"))
            if quantity > current_position:
                quantity = current_position  # Partial fill to position size
            proceeds = quantity * fill_price
            self._balance += proceeds
            self._positions[symbol] = current_position - quantity

        order = {
            "orderId": order_id,
            "symbol": symbol,
            "side": side.upper(),
            "quantity": str(quantity),
            "fillPrice": str(fill_price),
            "status": "FILLED",
        }
        self._orders[order_id] = order
        return order

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        if order_id in self._orders:
            self._orders[order_id]["status"] = "CANCELED"
            return True
        return False

    def get_order_status(self, symbol: str, order_id: str) -> dict:
        return self._orders.get(order_id, {"orderId": order_id, "status": "UNKNOWN"})

    def is_simulated(self) -> bool:
        return True
