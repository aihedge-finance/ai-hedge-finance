"""Pydantic schema for pipeline.json.

Validated on load — catches config errors at startup, not at trade time.

Design reference: design/pre_upgrade_v2_analysis/option_ab_production_readiness.md
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Producer types supported by the built-in registry
# ---------------------------------------------------------------------------

ProducerType = Literal["rl", "tech_indicator", "llm", "rule_based", "replay"]
AggregatorType = Literal["weighted_vote", "fixed_weight", "majority_vote", "meta_llm"]


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class ProducerConfig(BaseModel):
    """Configuration for a single signal producer."""

    id: str = Field(..., min_length=1, description="Unique producer identifier")
    type: ProducerType
    timeout_seconds: float = Field(default=5.0, gt=0, description="Hard timeout per produce() call")
    config: dict[str, Any] = Field(default_factory=dict, description="Producer-specific config")


class AggregatorConfig(BaseModel):
    """Configuration for the signal aggregator."""

    type: AggregatorType
    config: dict[str, Any] = Field(default_factory=dict, description="Aggregator-specific config")


class PipelineSettings(BaseModel):
    """Global pipeline behaviour settings."""

    min_valid_signals: int = Field(
        default=1,
        ge=1,
        description="Minimum valid (non-None) signals required to proceed with a trade step.",
    )
    audit_log_enabled: bool = Field(
        default=True,
        description="If True, write a JSONL audit entry for every pipeline step.",
    )
    audit_log_path: str = Field(
        default="data/logs/signal_audit.jsonl",
        description="Path to the append-only JSONL audit log file.",
    )


# ---------------------------------------------------------------------------
# Root config model
# ---------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    """Root schema for pipeline.json.

    Cross-field validators run after field-level validation to catch
    logical inconsistencies that span multiple fields.
    """

    producers: list[ProducerConfig] = Field(
        ...,
        min_length=1,
        description="At least one producer is required",
    )
    aggregator: AggregatorConfig
    settings: PipelineSettings = Field(default_factory=PipelineSettings)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "PipelineConfig":
        ids = [p.id for p in self.producers]
        duplicates = {x for x in ids if ids.count(x) > 1}
        if duplicates:
            raise ValueError(f"Duplicate producer ids: {duplicates}")
        return self

    @model_validator(mode="after")
    def validate_min_signals(self) -> "PipelineConfig":
        n_producers = len(self.producers)
        if self.settings.min_valid_signals > n_producers:
            raise ValueError(
                f"min_valid_signals ({self.settings.min_valid_signals}) cannot exceed "
                f"number of producers ({n_producers})"
            )
        return self

    @model_validator(mode="after")
    def validate_fixed_weights(self) -> "PipelineConfig":
        """If aggregator is fixed_weight, all producer ids must have a weight."""
        if self.aggregator.type == "fixed_weight":
            weights: dict = self.aggregator.config.get("weights", {})
            producer_ids = {p.id for p in self.producers}
            missing = producer_ids - set(weights.keys())
            extra = set(weights.keys()) - producer_ids
            if missing:
                raise ValueError(
                    f"fixed_weight aggregator is missing weights for producers: {missing}"
                )
            if extra:
                raise ValueError(
                    f"fixed_weight aggregator has weights for unknown producers: {extra}"
                )
        return self
