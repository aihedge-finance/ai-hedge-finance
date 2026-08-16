"""
Shared pytest fixtures for ahf test suite.
Fixtures grow with each phase — stubs are added here as placeholders.
"""
import pytest

# ---------------------------------------------------------------------------
# Market data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_market_data() -> dict:
    """Minimal OHLCV market snapshot for unit tests."""
    return {
        "5m": {
            "df": None,  # Will be a pandas DataFrame in Phase 4+
            "last_close": 67_432.50,
            "last_volume": 1_234.5,
            "updated_at": None,
        }
    }


@pytest.fixture
def dummy_pipeline_config_dict() -> dict:
    """Minimal pipeline config dict for single RL producer (stub)."""
    return {
        "producers": [
            {
                "id": "rl_ppo",
                "type": "rl",
                "timeout_seconds": 5.0,
                "config": {"model_path": "data/models/pod_000000", "agent": "ppo"},
            }
        ],
        "aggregator": {"type": "weighted_vote", "config": {}},
        "settings": {"min_valid_signals": 1, "audit_log_enabled": False},
    }


@pytest.fixture
def multi_producer_config_dict() -> dict:
    """Three-producer config for aggregator tests."""
    return {
        "producers": [
            {"id": "rl_ppo", "type": "rl", "timeout_seconds": 5.0, "config": {}},
            {"id": "tech_kf", "type": "tech_indicator", "timeout_seconds": 2.0, "config": {}},
            {"id": "llm_gemini", "type": "llm", "timeout_seconds": 10.0, "config": {}},
        ],
        "aggregator": {
            "type": "fixed_weight",
            "config": {"weights": {"rl_ppo": 0.5, "tech_kf": 0.3, "llm_gemini": 0.2}},
        },
        "settings": {"min_valid_signals": 2, "audit_log_enabled": False},
    }
