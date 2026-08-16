"""Tests for core/enums.py."""
from ahf.core.enums import AppEnv, NodeEnv, SaasEnv, TradeAction, TradingMode


def test_trade_action_values():
    assert TradeAction.BUY == 1
    assert TradeAction.HOLD == 0
    assert TradeAction.SELL == -1


def test_trade_action_comparison():
    assert TradeAction.BUY > TradeAction.HOLD > TradeAction.SELL


def test_trading_mode_values():
    assert TradingMode.LIVE == "LIVE"
    assert TradingMode.PAPER == "PAPER"
    assert TradingMode.SIMULATION == "SIMULATION"


def test_app_env_values():
    assert AppEnv.DEV == "DEV"
    assert AppEnv.PRODUCTION == "PRODUCTION"


def test_node_env_docker():
    assert NodeEnv.DOCKER == "DOCKER"


def test_saas_env_api():
    assert SaasEnv.API == "API"
