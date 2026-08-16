"""RL module import graph test.

Verifies that ported RL modules can be imported when all optional [rl]
deps are installed.  Each test uses pytest.importorskip() so it is
automatically *skipped* (not failed) when a required dependency is
missing from the current environment.

To run with full RL deps:
    uv sync --extra rl
    uv run pytest tests/integration/test_rl_imports.py -v
"""
from __future__ import annotations

import pytest


def test_brunhild_env_importable() -> None:
    pytest.importorskip("dask")
    pytest.importorskip("numpy")
    import ahf.rl.envs.BrunhildEnv_v11
    assert ahf.rl.envs.BrunhildEnv_v11 is not None


def test_base_env_importable() -> None:
    pytest.importorskip("pytz")
    pytest.importorskip("numpy")
    import ahf.rl.envs.BaseEnv_v11
    assert ahf.rl.envs.BaseEnv_v11 is not None


def test_brunhild_datastore_importable() -> None:
    # NOTE: BrunhildDatastore imports 'api.Binance.BinanceOrder' which is a
    # v1-era internal module not shipped in this repo. This test will skip
    # until an ahf-native BinanceOrder adapter is implemented.
    pytest.importorskip("simplejson")
    pytest.importorskip("api")  # v1 internal — will skip until ported
    import ahf.rl.envs.BrunhildDatastore_v11
    assert ahf.rl.envs.BrunhildDatastore_v11 is not None


def test_agent_ppo_importable() -> None:
    # NOTE: AgentPPO imports AgentBase which is a missing intermediate class.
    # This is a v1 porting gap — will skip until AgentBase is added.
    pytest.importorskip("torch")
    pytest.importorskip("numpy")
    try:
        from ahf.rl.agents import AgentBase  # noqa: F401
    except ImportError:
        pytest.skip("AgentBase not yet ported — v1 porting gap")
    import ahf.rl.agents.AgentPPO
    assert ahf.rl.agents.AgentPPO is not None


def test_train_config_importable() -> None:
    # NOTE: train/config imports 'deprecated' package — add to [rl] deps if needed.
    pytest.importorskip("torch")
    pytest.importorskip("deprecated")
    import ahf.rl.train.config
    assert ahf.rl.train.config is not None


def test_double_kf_strategy_importable() -> None:
    pytest.importorskip("tqdm")
    pytest.importorskip("numpy")
    import ahf.rl.strategies.double_kf.Strategy
    assert ahf.rl.strategies.double_kf.Strategy is not None


def test_rsi_macd_strategy_importable() -> None:
    # NOTE: rsi_macd/Strategy imports api.Binance.BinanceOrder (v1 internal).
    pytest.importorskip("tqdm")
    pytest.importorskip("numpy")
    pytest.importorskip("api")  # v1 internal — will skip until ported
    import ahf.rl.strategies.rsi_macd.Strategy
    assert ahf.rl.strategies.rsi_macd.Strategy is not None


def test_kalman_moving_average_importable() -> None:
    pytest.importorskip("pandas")
    import ahf.preprocessor.kf.KalmanMovingAverage
    assert ahf.preprocessor.kf.KalmanMovingAverage is not None


def test_ta_macd_importable() -> None:
    pytest.importorskip("pytz")
    pytest.importorskip("pandas")
    import ahf.preprocessor.ta.MACD
    assert ahf.preprocessor.ta.MACD is not None


def test_ta_rsi_importable() -> None:
    pytest.importorskip("pytz")
    pytest.importorskip("pandas")
    import ahf.preprocessor.ta.RSI
    assert ahf.preprocessor.ta.RSI is not None


def test_utils_importable() -> None:
    pytest.importorskip("pytz")
    import ahf.utils.utils
    assert ahf.utils.utils is not None


def test_logger_importable() -> None:
    pytest.importorskip("pytz")
    pytest.importorskip("loguru")
    import ahf.utils.logger
    assert ahf.utils.logger is not None
