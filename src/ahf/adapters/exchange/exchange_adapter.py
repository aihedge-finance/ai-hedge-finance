"""ExchangeAdapter ABC — unified interface for live and simulated exchanges."""
from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal


class ExchangeAdapter(ABC):
    """Abstract exchange interface.

    All exchange-specific implementations (Binance, Kraken, DummyAdapter)
    must implement this interface. The domain layer only uses this ABC —
    it has no direct dependency on any exchange SDK.
    """

    @abstractmethod
    def get_balance(self) -> Decimal:
        """Return the current available USDT/quote-currency balance."""
        ...

    @abstractmethod
    def get_position(self, symbol: str) -> Decimal:
        """Return the current position size for the given symbol (base currency units)."""
        ...

    @abstractmethod
    def get_price(self, symbol: str) -> Decimal:
        """Return the current market price for the given symbol."""
        ...

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        order_type: str = "MARKET",
        price: Decimal | None = None,
    ) -> dict:
        """Submit an order to the exchange.

        Args:
            symbol: Trading pair e.g. "BTCUSDT".
            side: "BUY" or "SELL".
            quantity: Order quantity in base currency units.
            order_type: "MARKET" or "LIMIT".
            price: Limit price (only for LIMIT orders).

        Returns:
            Exchange-specific order response dict.
            Must include at least: {"orderId": str, "status": str}.
        """
        ...

    @abstractmethod
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an open order. Returns True on success."""
        ...

    @abstractmethod
    def get_order_status(self, symbol: str, order_id: str) -> dict:
        """Return the current status of an order."""
        ...

    def is_simulated(self) -> bool:
        """Return True if this adapter does NOT submit real orders."""
        return False
