"""Core enums for ahf.

Ported from v1: envs/TradeEnum.py, app/enums.py
"""
from __future__ import annotations

from enum import Enum, IntEnum

# ---------------------------------------------------------------------------
# Trading Action
# ---------------------------------------------------------------------------


class TradeAction(IntEnum):
    """Discrete trade actions produced by the signal processor."""

    SELL = -1
    HOLD = 0
    BUY = 1


# ---------------------------------------------------------------------------
# Trading / Bot Modes
# ---------------------------------------------------------------------------


class TradingMode(str, Enum):
    """Execution mode for the tradebot.

    - LIVE: Real exchange orders with real capital
    - PAPER: Real market data, simulated orders (DummyAdapter)
    - SIMULATION: Synthetic/backtested data, simulated orders
    """

    LIVE = "LIVE"
    PAPER = "PAPER"
    SIMULATION = "SIMULATION"


class AppEnv(str, Enum):
    """Application runtime environment."""

    DEV = "DEV"
    STAGE = "STAGE"
    PRODUCTION = "PRODUCTION"


class NodeEnv(str, Enum):
    """Deployment container type (mirrors v1 NODE_ENV)."""

    DEV = "DEV"
    DOCKER = "DOCKER"
    K8S = "K8s"


class SaasEnv(str, Enum):
    """SaaS operation mode (mirrors v1 SaasEnv)."""

    API = "API"
    STANDALONE = "STANDALONE"
