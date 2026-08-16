"""Tests for signals/pipeline_loader.py — registry and factory."""

import pytest

from ahf.signals.pipeline_loader import (
    AGGREGATOR_REGISTRY,
    PRODUCER_REGISTRY,
    _import_class,
)


def test_producer_registry_has_all_types():
    expected = {"rl", "tech_indicator", "llm", "rule_based", "replay"}
    assert expected == set(PRODUCER_REGISTRY.keys())


def test_aggregator_registry_has_all_types():
    expected = {"weighted_vote", "fixed_weight", "majority_vote", "meta_llm"}
    assert expected == set(AGGREGATOR_REGISTRY.keys())


def test_import_class_pydantic_basemodel():
    """_import_class should work on any valid dotted path."""
    cls = _import_class("pydantic.BaseModel")
    from pydantic import BaseModel
    assert cls is BaseModel


def test_import_class_unknown_module_raises():
    with pytest.raises(ModuleNotFoundError):
        _import_class("ahf.nonexistent.module.SomeClass")


def test_load_pipeline_from_dict_weighted_vote(dummy_pipeline_config_dict):
    """Phase 2 stubs are needed for this test — for now just verify validation."""
    # This will raise ImportError because stub producers don't exist yet in Phase 1.
    # That's expected — we just test the config validation path here.
    cfg = dummy_pipeline_config_dict.copy()
    from ahf.signals.pipeline_config import PipelineConfig
    parsed = PipelineConfig.model_validate(cfg)
    assert len(parsed.producers) == 1
    assert parsed.aggregator.type == "weighted_vote"


def test_load_pipeline_file_not_found():
    from ahf.signals.pipeline_loader import load_pipeline
    with pytest.raises(FileNotFoundError):
        load_pipeline("/nonexistent/path/pipeline.json", {})
