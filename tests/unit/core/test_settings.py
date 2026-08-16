"""Tests for core/settings.py."""

import pytest

from ahf.core.settings import Settings, get_settings


def test_settings_defaults():
    """Settings loads with defaults when no .env is present."""
    s = Settings()
    assert s.loglevel == "INFO"
    assert s.debug is False
    assert s.redis_port == 6379
    assert s.init_trade_cash == 1000.0


def test_settings_loglevel_uppercased():
    s = Settings(loglevel="debug")
    assert s.loglevel == "DEBUG"


def test_settings_invalid_loglevel():
    with pytest.raises(Exception, match="loglevel"):
        Settings(loglevel="VERBOSE")


def test_settings_live_mode_requires_pod_dir():
    with pytest.raises(Exception, match="pod_dir"):
        Settings(exch_mode="REAL", pod_dir="")


def test_settings_live_mode_with_pod_dir_ok():
    s = Settings(exch_mode="REAL", pod_dir="data/models/pod_000001")
    assert s.pod_dir == "data/models/pod_000001"


def test_get_settings_returns_same_instance():
    get_settings.cache_clear()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    get_settings.cache_clear()


def test_settings_optuna_default():
    s = Settings()
    assert s.optuna_storage.startswith("sqlite:///")
