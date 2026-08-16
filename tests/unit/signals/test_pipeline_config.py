"""Tests for signals/pipeline_config.py — Pydantic schema validation."""
import pytest
from pydantic import ValidationError

from ahf.signals.pipeline_config import PipelineConfig


def _base_config(**overrides) -> dict:
    cfg = {
        "producers": [
            {"id": "rl_ppo", "type": "rl", "timeout_seconds": 5.0, "config": {}},
        ],
        "aggregator": {"type": "weighted_vote", "config": {}},
        "settings": {"min_valid_signals": 1},
    }
    cfg.update(overrides)
    return cfg


def test_valid_minimal_config():
    cfg = PipelineConfig.model_validate(_base_config())
    assert len(cfg.producers) == 1
    assert cfg.aggregator.type == "weighted_vote"


def test_duplicate_producer_ids_raise():
    cfg = _base_config()
    cfg["producers"] = [
        {"id": "rl_ppo", "type": "rl", "config": {}},
        {"id": "rl_ppo", "type": "llm", "config": {}},
    ]
    with pytest.raises(ValidationError, match="Duplicate producer ids"):
        PipelineConfig.model_validate(cfg)


def test_min_valid_signals_exceeds_producers_raises():
    cfg = _base_config()
    cfg["settings"] = {"min_valid_signals": 5}
    with pytest.raises(ValidationError, match="min_valid_signals"):
        PipelineConfig.model_validate(cfg)


def test_invalid_producer_type_raises():
    cfg = _base_config()
    cfg["producers"][0]["type"] = "unknown_type"
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(cfg)


def test_invalid_aggregator_type_raises():
    cfg = _base_config()
    cfg["aggregator"]["type"] = "bad_aggregator"
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(cfg)


def test_negative_timeout_raises():
    cfg = _base_config()
    cfg["producers"][0]["timeout_seconds"] = -1.0
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(cfg)


def test_zero_timeout_raises():
    cfg = _base_config()
    cfg["producers"][0]["timeout_seconds"] = 0.0
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(cfg)


def test_fixed_weight_missing_producer_weight_raises():
    cfg = {
        "producers": [
            {"id": "rl_ppo", "type": "rl", "config": {}},
            {"id": "tech_kf", "type": "tech_indicator", "config": {}},
        ],
        "aggregator": {
            "type": "fixed_weight",
            "config": {"weights": {"rl_ppo": 0.6}},  # missing tech_kf
        },
    }
    with pytest.raises(ValidationError, match="missing weights for producers"):
        PipelineConfig.model_validate(cfg)


def test_fixed_weight_extra_producer_weight_raises():
    cfg = {
        "producers": [
            {"id": "rl_ppo", "type": "rl", "config": {}},
        ],
        "aggregator": {
            "type": "fixed_weight",
            "config": {"weights": {"rl_ppo": 0.6, "ghost_producer": 0.4}},
        },
    }
    with pytest.raises(ValidationError, match="unknown producers"):
        PipelineConfig.model_validate(cfg)


def test_fixed_weight_all_weights_valid():
    cfg = {
        "producers": [
            {"id": "rl_ppo", "type": "rl", "config": {}},
            {"id": "tech_kf", "type": "tech_indicator", "config": {}},
        ],
        "aggregator": {
            "type": "fixed_weight",
            "config": {"weights": {"rl_ppo": 0.6, "tech_kf": 0.4}},
        },
    }
    parsed = PipelineConfig.model_validate(cfg)
    assert parsed.aggregator.config["weights"]["rl_ppo"] == 0.6


def test_empty_producers_list_raises():
    cfg = _base_config()
    cfg["producers"] = []
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(cfg)


def test_settings_defaults():
    cfg = PipelineConfig.model_validate(_base_config())
    assert cfg.settings.min_valid_signals == 1
    assert cfg.settings.audit_log_enabled is True
