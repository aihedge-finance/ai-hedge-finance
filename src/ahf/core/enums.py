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
# v2 Trading / Bot Modes
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


# ---------------------------------------------------------------------------
# v1-compatible enums (preserved for BrunhildEnv_v11 / RL code compatibility)
# ---------------------------------------------------------------------------


class AppEnv(str, Enum):
    """Application runtime environment (v1).

    決定交易 data_source 是到交易所還是本地紀錄(backtest_data)而已.
    """

    TRAIN = "TRAIN"
    TRADE = "TRADE"
    SIMULATION = "SIMULATION"
    BOT = "BOT"


class PriceEnv(str, Enum):
    """Determines how prices are fetched (v1)."""

    TRAIN = "TRAIN"
    TRADE = "TRADE"
    WS = "WS"


class BotEnv(str, Enum):
    """Overall bot operating mode (v1)."""

    TRADE = "TRADE"
    MOCKING = "MOCKING"
    SIMULATION = "SIMULATION"
    TRAIN = "TRAIN"
    BACKTESTING = "BACKTESTING"


class MLLevel(str, Enum):
    """Training depth level (v1)."""

    FIX = "FIX"
    LV2 = "LV2"
    DEEP1 = "DEEP1"
    DEEP2 = "DEEP2"


# ---------------------------------------------------------------------------
# v2 deployment-level enums
# ---------------------------------------------------------------------------


class DeployEnv(str, Enum):
    """Infrastructure deployment environment (v2)."""

    DEV = "DEV"
    STAGE = "STAGE"
    PRODUCTION = "PRODUCTION"


class NodeEnv(str, Enum):
    """Container runtime (mirrors v1 NODE_ENV)."""

    DEV = "DEV"
    DOCKER = "DOCKER"
    K8S = "K8s"


class SaasEnv(str, Enum):
    """SaaS operation mode."""

    API = "API"
    STANDALONE = "STANDALONE"


# ---------------------------------------------------------------------------
# v1 BOT_MODE mapping (preserved for RL env compatibility)
# ---------------------------------------------------------------------------

BOT_MODE: dict[BotEnv, dict] = {
    BotEnv.TRADE: {
        "bot_env": BotEnv.TRADE,
        "app_env": AppEnv.TRADE,
        "exch_mode": "SpotAPI",
        "price_env": PriceEnv.TRADE,
    },
    BotEnv.SIMULATION: {
        "bot_env": BotEnv.SIMULATION,
        "app_env": AppEnv.TRAIN,
        "exch_mode": "SpotAPI",
        "price_env": PriceEnv.TRADE,
    },
    BotEnv.MOCKING: {
        "bot_env": BotEnv.MOCKING,
        "app_env": AppEnv.TRADE,
        "exch_mode": "SpotTest",
        "price_env": PriceEnv.TRADE,
    },
}


def init_bot(bot_env: BotEnv, trade_args: dict) -> tuple[dict, dict]:
    """Initialise trade_args from a BotEnv (v1-compatible)."""
    bot_env_args = BOT_MODE[bot_env]
    trade_args["app_env"] = bot_env_args["app_env"]
    trade_args["price_env"] = bot_env_args["price_env"]
    trade_args["exch_mode"] = bot_env_args["exch_mode"]
    if trade_args.get("saas_env") is None:
        raise ValueError("trade_args.saas_env is required, got None")
    return BOT_MODE[bot_env], trade_args
