"""Pipeline loader: parses pipeline.json and wires producers + aggregator.

Design reference: design/pre_upgrade_v2_analysis/option_ab_architecture.md
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from ahf.signals.pipeline_config import PipelineConfig
from ahf.signals.signal_aggregator import SignalAggregator
from ahf.signals.signal_producer import SignalProducer

# ---------------------------------------------------------------------------
# Registry: maps type string → dotted module path (updated as phases complete)
# ---------------------------------------------------------------------------

PRODUCER_REGISTRY: dict[str, str] = {
    "rl":             "ahf.signals.producers.rl_producer.RLSignalProducer",
    "tech_indicator": "ahf.signals.producers.tech_producer.TechIndicatorProducer",
    "llm":            "ahf.signals.producers.llm_producer.LLMSignalProducer",
    "rule_based":     "ahf.signals.producers.rule_producer.RuleBasedProducer",
    "replay":         "ahf.signals.producers.replay_producer.ReplayProducer",
}

AGGREGATOR_REGISTRY: dict[str, str] = {
    "weighted_vote": "ahf.signals.aggregators.weighted_vote.WeightedVoteAggregator",
    "fixed_weight":  "ahf.signals.aggregators.fixed_weight.FixedWeightAggregator",
    "majority_vote": "ahf.signals.aggregators.majority_vote.MajorityVoteAggregator",
    "meta_llm":      "ahf.signals.aggregators.meta_llm.MetaLLMAggregator",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_pipeline(
    config_path: str | Path,
    runtime_deps: dict,
) -> tuple[list[SignalProducer], SignalAggregator]:
    """Parse pipeline.json and instantiate all producers + aggregator.

    Args:
        config_path: Path to pipeline.json (absolute or relative to cwd).
        runtime_deps: Runtime objects that can't be serialised in JSON.
            Examples: {"env": env, "agent_ppo": agent, "gemini-2.0-flash": client}

    Returns:
        (producers, aggregator): Ready-to-use instances.

    Raises:
        FileNotFoundError: If config_path doesn't exist.
        pydantic.ValidationError: If pipeline.json fails schema validation.
        ImportError: If a producer/aggregator class can't be imported.
        KeyError: If a type string is not registered.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {path.resolve()}")

    with path.open() as f:
        raw = json.load(f)

    config = PipelineConfig.model_validate(raw)
    producers = [_build_producer(p_cfg, runtime_deps) for p_cfg in config.producers]
    aggregator = _build_aggregator(config.aggregator, runtime_deps)
    return producers, aggregator


def load_pipeline_from_dict(
    config_dict: dict,
    runtime_deps: dict,
) -> tuple[list[SignalProducer], SignalAggregator]:
    """Same as load_pipeline() but accepts a dict directly (useful in tests)."""
    config = PipelineConfig.model_validate(config_dict)
    producers = [_build_producer(p_cfg, runtime_deps) for p_cfg in config.producers]
    aggregator = _build_aggregator(config.aggregator, runtime_deps)
    return producers, aggregator


def run_health_checks(producers: list[SignalProducer]) -> None:
    """Run health_check() on all producers. Raise immediately on first failure.

    Call this once after load_pipeline() and before starting the trading loop.
    """
    for producer in producers:
        producer.health_check()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_producer(p_cfg: Any, runtime_deps: dict) -> SignalProducer:
    if p_cfg.type not in PRODUCER_REGISTRY:
        raise KeyError(
            f"Unknown producer type: {p_cfg.type!r}. "
            f"Registered types: {list(PRODUCER_REGISTRY.keys())}"
        )
    cls = _import_class(PRODUCER_REGISTRY[p_cfg.type])
    return cls.from_config(p_cfg.id, p_cfg.config, runtime_deps)  # type: ignore[no-any-return]


def _build_aggregator(agg_cfg: Any, runtime_deps: dict) -> SignalAggregator:
    if agg_cfg.type not in AGGREGATOR_REGISTRY:
        raise KeyError(
            f"Unknown aggregator type: {agg_cfg.type!r}. "
            f"Registered types: {list(AGGREGATOR_REGISTRY.keys())}"
        )
    cls = _import_class(AGGREGATOR_REGISTRY[agg_cfg.type])
    return cls.from_config(agg_cfg.config, runtime_deps)  # type: ignore[no-any-return]


def _import_class(dotted_path: str) -> Any:
    """Import a class from a dotted module path string."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
