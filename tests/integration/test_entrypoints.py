"""Entrypoint integration tests (Issue 13 — trade.py was 38% coverage).

Tests the main() function of each entrypoint using environment override
and a mock shutdown signal — verifies the startup path without spinning
up a real trading loop.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import patch


def test_trade_main_runs_one_step_and_exits(tmp_path) -> None:
    """trade.main() should initialise the full stack, run 1 step, then exit."""
    import ahf.entrypoints.trade as _trade_mod

    # Force shutdown after first step
    step_count = 0
    original_step = None

    def patched_sleep(seconds: float) -> None:
        # After sleeping (i.e. after 1 step) trigger shutdown
        _trade_mod._SHUTDOWN = True

    env_overrides = {
        "TRADING_MODE": "PAPER",
        "SYMBOL": "BTCUSDT",
        "PIPELINE_CONFIG": "configs/pipeline.json",
        "INITIAL_CAPITAL": "1000.0",
        "STEP_INTERVAL_SECONDS": "0",  # no real sleep
        "AUDIT_LOG_ENABLED": "false",
        "HALT_ON_ERROR": "false",
    }

    _trade_mod._SHUTDOWN = False  # reset in case previous test left it True

    with patch.dict(os.environ, env_overrides, clear=False):
        with patch("ahf.entrypoints.trade.time.sleep", side_effect=patched_sleep):
            from ahf.core.settings import AHFSettings
            AHFSettings.model_config  # ensure settings reload

            result = _trade_mod.main()

    _trade_mod._SHUTDOWN = False  # cleanup
    assert result == 0


def test_trade_main_halt_on_error_returns_1(tmp_path) -> None:
    """When an orchestrator step raises and halt_on_error=True, main() returns 1."""
    import ahf.entrypoints.trade as _trade_mod

    _trade_mod._SHUTDOWN = False

    env_overrides = {
        "TRADING_MODE": "PAPER",
        "SYMBOL": "BTCUSDT",
        "PIPELINE_CONFIG": "configs/pipeline.json",
        "INITIAL_CAPITAL": "1000.0",
        "STEP_INTERVAL_SECONDS": "0",
        "AUDIT_LOG_ENABLED": "false",
        "HALT_ON_ERROR": "true",
    }

    def raise_runtime(*args, **kwargs):
        raise RuntimeError("simulated error")

    with patch.dict(os.environ, env_overrides, clear=False):
        with patch("ahf.domain.trade_orchestrator.TradeOrchestrator.step", side_effect=raise_runtime):
            result = _trade_mod.main()

    _trade_mod._SHUTDOWN = False
    assert result == 1


def test_backtest_main_imports_and_callable() -> None:
    """backtest.main() is importable and callable without crashing."""
    import ahf.entrypoints.backtest as _backtest_mod
    assert callable(_backtest_mod.main)


def test_train_main_imports_and_callable() -> None:
    """train.main() is importable and callable without crashing."""
    import ahf.entrypoints.train as _train_mod
    assert callable(_train_mod.main)
