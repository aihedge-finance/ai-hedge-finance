# Option A(b): Production Readiness

**Parent document**: [Option A(b) Architecture](option_ab_architecture.md)  
**Related**: [Signal Contract](option_ab_signal_contract.md) | [Audience Analysis](option_ab_audience_analysis.md)

---

## What A(b) Has Today (As Designed)

| Capability | Status | Notes |
|---|---|---|
| Multiple signal sources | ✅ | Any N producers via JSON config |
| Signal contract enforcement | ✅ | Pydantic validation at boundary |
| Aggregation strategies | ✅ | Weighted vote, fixed weight, majority vote, meta-LLM |
| JSON-driven pipeline config | ✅ | Edit JSON, not Python |
| Config loader + registry | ✅ | `pipeline_loader.py` with `from_config` pattern |
| Graceful degradation | ✅ | try/except per producer + timeout per producer |
| Backward compatibility | ✅ | Single-producer config = current behavior |
| Growth path to Option B | ✅ | All ABCs map 1:1 to Option B interfaces |
| Timeout per producer | ✅ | `timeout.py` + `timeout_seconds` in JSON config |
| JSON schema validation | ✅ | `pipeline_config.py` — Pydantic model with cross-field validation |
| Startup health check | ✅ | `health_check()` on every producer, fail-fast |
| Structured logging | ✅ | `signal_logger.py` — JSON-structured log entries |
| Min-valid-signals safety | ✅ | `min_valid_signals` in config → orchestrator skips step if too few valid |
| Signal audit log | ✅ | `signal_audit.py` — JSONL append-only log |
| Backtest replay | ✅ | `replay_producer.py` — replays signals from audit log |
| Producer template | ✅ | `_template.py` — copy-paste starter with inline docs |

---

## Phased Implementation Roadmap

### Phase 1: Ship-Blocking (Add Before First Deploy)

These features prevent the system from hanging, crashing on bad config, or trading blind.

```
Phase 1 (Ship-blocking):
  ├── ⏱️  Timeout per producer (threading-based, ~30 lines)
  │     └─ Prevents LLM calls from hanging the entire trading step
  ├── 🔍 JSON schema validation for pipeline.json (~90 lines)
  │     └─ Catches typos, invalid types, duplicate IDs at load time
  ├── 🚀 Eager startup validation / health check (~40 lines)
  │     └─ Verifies model files, API keys, strategy configs before trading
  └── 🛑 min_valid_signals policy (~20 lines)
        └─ Prevents trading on degraded signal set (e.g., only 1 of 3 producers alive)
```

### Phase 2: First Week in Production

These features enable debugging, post-trade analysis, and operational visibility.

```
Phase 2 (First-week-in-production):
  ├── 📊 Structured JSON logging (~60 lines)
  │     └─ producer_result, aggregation_result, step_skipped events for dashboards
  ├── 📝 Signal audit log to JSONL (~50 lines)
  │     └─ Every step's full signal map persisted for post-trade forensics
  └── 🧪 Backtest replay producer (~60 lines)
        └─ Replay LLM signals from audit log — deterministic, free, fast
```

### Phase 3: Maturity

These features are for scaling, community, and operational excellence.

```
Phase 3 (Maturity):
  ├── 📈 Prometheus metrics hooks
  │     └─ Producer success/failure rates, latency percentiles, signal drift
  ├── 🔄 Config hot-reload
  │     └─ File watcher or Redis signal → update pipeline without restart
  ├── 📖 Contributor guide + producer template
  │     └─ CONTRIBUTING_SIGNALS.md + _template.py (already done)
  └── 🛠️ CLI for config management
        └─ `diewalkure pipeline validate`, `diewalkure pipeline list-producers`
```

---

## Complete Implementations

All Phase 1 and Phase 2 features are implemented below.

---

### Pipeline Config Model (JSON Schema Validation) — Phase 1

A Pydantic model that validates `pipeline.json` at load time. Typos and invalid values produce clear, human-readable errors instead of cryptic `KeyError`s.

```python
# Trade/signals/pipeline_config.py
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Dict, Any, Literal


class ProducerConfig(BaseModel):
    """Schema for a single producer entry in pipeline.json."""
    id: str = Field(..., description="Unique name for this producer")
    type: Literal["rl", "tech_indicator", "llm", "rule_based", "replay"] = Field(
        ..., description="Producer type — must match a registered type"
    )
    timeout_seconds: float = Field(
        default=10.0, gt=0, le=300,
        description="Max seconds to wait for this producer before treating as failed"
    )
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Type-specific configuration"
    )

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Producer id cannot be empty")
        if " " in v:
            raise ValueError(f"Producer id cannot contain spaces: '{v}'")
        return v


class AggregatorConfig(BaseModel):
    """Schema for the aggregator entry in pipeline.json."""
    type: Literal["weighted_vote", "fixed_weight", "majority_vote", "meta_llm"] = Field(
        ..., description="Aggregator type — must match a registered type"
    )
    config: Dict[str, Any] = Field(default_factory=dict)


class PipelineSettings(BaseModel):
    """Top-level pipeline safety and behavior settings."""
    min_valid_signals: int = Field(
        default=1, ge=1,
        description=(
            "Minimum number of producers that must return valid signals "
            "for the aggregator to run. If fewer succeed, the step is skipped "
            "and a HOLD is returned. Set to 1 for maximum availability, "
            "set to N for maximum safety."
        )
    )
    audit_log_enabled: bool = Field(
        default=True,
        description="If true, every step's signals are written to the signal audit log"
    )
    audit_log_path: str = Field(
        default="logs/signal_audit.jsonl",
        description="Path for the signal audit log (JSONL format)"
    )


class PipelineConfig(BaseModel):
    """Root schema for pipeline.json. Validated on load."""
    producers: List[ProducerConfig] = Field(
        ..., min_length=1,
        description="At least one producer is required"
    )
    aggregator: AggregatorConfig
    settings: PipelineSettings = Field(default_factory=PipelineSettings)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "PipelineConfig":
        ids = [p.id for p in self.producers]
        duplicates = [x for x in ids if ids.count(x) > 1]
        if duplicates:
            raise ValueError(f"Duplicate producer ids: {set(duplicates)}")
        return self

    @model_validator(mode="after")
    def validate_min_signals(self) -> "PipelineConfig":
        if self.settings.min_valid_signals > len(self.producers):
            raise ValueError(
                f"min_valid_signals ({self.settings.min_valid_signals}) cannot exceed "
                f"number of producers ({len(self.producers)})"
            )
        return self

    @model_validator(mode="after")
    def validate_fixed_weights(self) -> "PipelineConfig":
        """If aggregator is fixed_weight, ensure all producer ids have weights."""
        if self.aggregator.type == "fixed_weight":
            weights = self.aggregator.config.get("weights", {})
            producer_ids = {p.id for p in self.producers}
            missing = producer_ids - set(weights.keys())
            extra = set(weights.keys()) - producer_ids
            if missing:
                raise ValueError(
                    f"fixed_weight aggregator missing weights for: {missing}"
                )
            if extra:
                raise ValueError(
                    f"fixed_weight aggregator has weights for unknown producers: {extra}"
                )
        return self
```

**What this catches at load time (not at trade time):**

```
❌ pipeline.json has "typ": "rl"
→ "Input should be 'rl', 'tech_indicator', 'llm', 'rule_based' or 'replay'"

❌ pipeline.json has duplicate producer ids
→ "Duplicate producer ids: {'rl_ppo'}"

❌ min_valid_signals = 5 but only 3 producers
→ "min_valid_signals (5) cannot exceed number of producers (3)"

❌ fixed_weight aggregator missing weight for "llm_gemini"
→ "fixed_weight aggregator missing weights for: {'llm_gemini'}"

❌ timeout_seconds = -1
→ "Input should be greater than 0"
```

---

### Health Check (Startup Validation) — Phase 1

Each producer implements an optional `health_check()` that verifies its dependencies are available before the bot starts trading.

```python
# Added to Trade/signals/signal_producer.py
class SignalProducer(ABC):
    """Layer 1: Any source that produces a trading signal."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def produce(self, market_data: dict, context: dict) -> SignalOutput: ...

    def health_check(self) -> None:
        """Verify this producer's dependencies at startup. Raise on failure.
        
        Override this in producers that depend on external resources:
        - RL: check model file exists and loads
        - LLM: check API key is set and endpoint responds
        - Tech: check strategy config file exists
        
        Default: no-op (always healthy).
        """
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, name: str, config: dict, runtime_deps: dict) -> "SignalProducer":
        """Build from JSON config + runtime dependencies."""
        ...
```

```python
# Example health checks in concrete producers:

class RLSignalProducer(SignalProducer):
    def health_check(self) -> None:
        if self._agent is None:
            raise RuntimeError(f"[{self.name}] RL agent not loaded")
        if self._env is None:
            raise RuntimeError(f"[{self.name}] Environment not initialized")

class LLMSignalProducer(SignalProducer):
    def health_check(self) -> None:
        if self._client is None:
            raise RuntimeError(f"[{self.name}] LLM API client not configured")
        try:
            self._client.generate("health check ping", max_tokens=5)
        except Exception as e:
            raise RuntimeError(f"[{self.name}] LLM API unreachable: {e}")

class TechIndicatorProducer(SignalProducer):
    def health_check(self) -> None:
        if self._strategy is None:
            raise RuntimeError(f"[{self.name}] Strategy not loaded")
```

---

### Timeout per Producer — Phase 1

```python
# Trade/signals/timeout.py
import threading
import logging
from typing import Optional
from Trade.signals.signal_types import SignalOutput

logger = logging.getLogger(__name__)


def produce_with_timeout(
    producer, market_data: dict, context: dict, timeout_seconds: float
) -> Optional[SignalOutput]:
    """Run a producer with a hard timeout. Returns None on timeout."""
    result = [None]
    error = [None]

    def target():
        try:
            result[0] = producer.produce(market_data, context)
        except Exception as e:
            error[0] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        logger.warning(f"[{producer.name}] timed out after {timeout_seconds}s")
        return None

    if error[0]:
        raise error[0]

    return result[0]
```

JSON config gains a `timeout_seconds` field per producer:

```json
{
  "id": "llm_gemini",
  "type": "llm",
  "timeout_seconds": 5.0,
  "config": { "model": "gemini-2.0-flash", "prompt_template": "market_analysis_v2" }
}
```

---

### Structured JSON Logging — Phase 2

```python
# Trade/signals/signal_logger.py
import json
import logging
from typing import Optional
from Trade.signals.signal_types import SignalOutput


class SignalLogger:
    """Structured JSON logger for signal pipeline events."""

    def __init__(self, logger_name: str = "signal_pipeline"):
        self._logger = logging.getLogger(logger_name)

    def log_producer_result(
        self,
        producer_name: str,
        signal: Optional[SignalOutput],
        latency_ms: float,
        error: Optional[str] = None,
        timed_out: bool = False,
    ) -> None:
        entry = {
            "event": "producer_result",
            "producer": producer_name,
            "latency_ms": round(latency_ms, 2),
            "success": signal is not None and error is None,
            "timed_out": timed_out,
        }
        if signal:
            entry["action"] = signal.action
            entry["confidence"] = signal.confidence
        if error:
            entry["error"] = error
        self._logger.info(json.dumps(entry))

    def log_aggregation_result(
        self,
        aggregated: SignalOutput,
        valid_count: int,
        total_count: int,
        latency_ms: float,
    ) -> None:
        entry = {
            "event": "aggregation_result",
            "action": aggregated.action,
            "confidence": aggregated.confidence,
            "valid_producers": valid_count,
            "total_producers": total_count,
            "latency_ms": round(latency_ms, 2),
        }
        self._logger.info(json.dumps(entry))

    def log_step_skipped(self, reason: str, valid_count: int, required: int) -> None:
        entry = {
            "event": "step_skipped",
            "reason": reason,
            "valid_producers": valid_count,
            "min_required": required,
        }
        self._logger.warning(json.dumps(entry))
```

---

### Signal Audit Log — Phase 2

```python
# Trade/signals/signal_audit.py
import json
import os
from datetime import datetime, timezone
from typing import Optional, Dict
from Trade.signals.signal_types import SignalOutput


class SignalAuditLog:
    """Append-only JSONL audit log of all signals per trading step."""

    def __init__(self, log_path: str = "logs/signal_audit.jsonl"):
        self._log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def record_step(
        self,
        step_id: int,
        producer_signals: Dict[str, Optional[SignalOutput]],
        aggregated: Optional[SignalOutput],
        regime: str,
        skipped: bool = False,
        skip_reason: str = None,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step_id": step_id,
            "regime": regime,
            "skipped": skipped,
            "skip_reason": skip_reason,
            "producers": {},
            "aggregated": None,
        }
        for name, sig in producer_signals.items():
            if sig is not None:
                record["producers"][name] = {
                    "action": sig.action,
                    "confidence": sig.confidence,
                    "metadata": sig.metadata,
                }
            else:
                record["producers"][name] = None
        if aggregated:
            record["aggregated"] = {
                "action": aggregated.action,
                "confidence": aggregated.confidence,
            }
        with open(self._log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
```

**Example audit log entry:**

```json
{
  "timestamp": "2026-07-28T06:00:00Z",
  "step_id": 1042,
  "regime": "TRENDING_BULL",
  "skipped": false,
  "skip_reason": null,
  "producers": {
    "rl_ppo": {"action": 0.73, "confidence": 0.85, "metadata": {"raw_action": [0.73]}},
    "tech_kalman": {"action": 0.4, "confidence": 0.6, "metadata": {"strategy": "double_kf"}},
    "llm_gemini": null
  },
  "aggregated": {"action": 0.58, "confidence": 0.73}
}
```

---

### Backtest Replay Producer — Phase 2

```python
# Trade/signals/producers/replay_producer.py
import json
from Trade.signals.signal_producer import SignalProducer
from Trade.signals.signal_types import SignalOutput


class ReplayProducer(SignalProducer):
    """Replays pre-recorded signals from a JSONL file instead of computing live.
    
    Use in backtest mode to replay LLM signals without API calls.
    """

    def __init__(self, name: str, replay_file: str):
        self._name = name
        self._signals = self._load_replay(replay_file, name)
        self._step = 0

    @property
    def name(self) -> str:
        return self._name

    def produce(self, market_data: dict, context: dict) -> SignalOutput:
        if self._step >= len(self._signals):
            return SignalOutput(
                signal_name=self._name, action=0.0, confidence=0.0,
                metadata={"replay": True, "exhausted": True}
            )
        entry = self._signals[self._step]
        self._step += 1
        return SignalOutput(
            signal_name=self._name,
            action=entry["action"],
            confidence=entry["confidence"],
            metadata={"replay": True, "original_metadata": entry.get("metadata")},
        )

    def health_check(self) -> None:
        if not self._signals:
            raise RuntimeError(f"[{self._name}] Replay file is empty or producer not found")

    @classmethod
    def from_config(cls, name: str, config: dict, runtime_deps: dict) -> "ReplayProducer":
        return cls(name=name, replay_file=config["replay_file"])

    @staticmethod
    def _load_replay(filepath: str, producer_name: str) -> list:
        signals = []
        with open(filepath) as f:
            for line in f:
                record = json.loads(line)
                entry = record.get("producers", {}).get(producer_name)
                if entry is not None:
                    signals.append(entry)
        return signals
```

---

### Producer Template (Copy-Paste Starter)

```python
# Trade/signals/producers/_template.py
"""
Copy this file to create a new signal producer.

Steps:
  1. Copy this file: cp _template.py my_producer.py
  2. Rename the class and update `name`
  3. Implement produce() — translate your native output to SignalOutput
  4. Implement from_config() — map JSON config to constructor args
  5. Implement health_check() — verify dependencies at startup
  6. Register in Trade/signals/pipeline_loader.py PRODUCER_REGISTRY
  7. Add to pipeline.json
"""
from Trade.signals.signal_producer import SignalProducer
from Trade.signals.signal_types import SignalOutput


class MyCustomProducer(SignalProducer):
    """One-line description of what this producer does."""

    def __init__(self, name: str, my_param: str):
        self._name = name
        self._my_param = my_param

    @property
    def name(self) -> str:
        return self._name

    def produce(self, market_data: dict, context: dict) -> SignalOutput:
        """Generate a trading signal from market data.
        
        IMPORTANT: You MUST normalize your output to:
          - action:     float in [-1.0, +1.0]  (-1 = sell, 0 = hold, +1 = buy)
          - confidence: float in [ 0.0,  1.0]  (0 = no confidence, 1 = certain)
        """
        raw_score = 0.0  # Replace with your computation
        action = max(-1.0, min(1.0, raw_score))
        confidence = 0.5

        return SignalOutput(
            signal_name=self._name,
            action=action,
            confidence=confidence,
            metadata={"raw_score": raw_score, "my_param": self._my_param},
        )

    def health_check(self) -> None:
        if not self._my_param:
            raise RuntimeError(f"[{self.name}] my_param is not configured")

    @classmethod
    def from_config(cls, name: str, config: dict, runtime_deps: dict) -> "MyCustomProducer":
        return cls(name=name, my_param=config.get("my_param", "default_value"))
```

---

### Complete Orchestrator (All Features Integrated)

```python
# Trade/trade_orchestrator.py — Production-ready Option A(b)
import time
import logging
from typing import List, Optional, Dict
from Trade.signals.signal_producer import SignalProducer
from Trade.signals.signal_aggregator import SignalAggregator
from Trade.signals.signal_types import SignalOutput
from Trade.signals.timeout import produce_with_timeout
from Trade.signals.signal_logger import SignalLogger
from Trade.signals.signal_audit import SignalAuditLog
from Trade.signals.pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)


class TradeOrchestrator:
    def __init__(
        self,
        producers: List[SignalProducer],
        aggregator: SignalAggregator,
        pipeline_config: PipelineConfig,
        exchange, risk, signal_processor,
        notifier, publisher, state_logger,
        regime_detector=None,
    ):
        self.producers = producers
        self.aggregator = aggregator
        self.config = pipeline_config
        self.exchange = exchange
        self.risk = risk
        self.signal_processor = signal_processor
        self.notifier = notifier
        self.publisher = publisher
        self.state_logger = state_logger
        self.regime_detector = regime_detector

        self._timeouts: Dict[str, float] = {
            p.id: p.timeout_seconds for p in pipeline_config.producers
        }
        self._signal_logger = SignalLogger()
        self._audit_log = (
            SignalAuditLog(pipeline_config.settings.audit_log_path)
            if pipeline_config.settings.audit_log_enabled
            else None
        )
        self._step_count = 0

    def step(self, env, agent, user_input=False) -> dict:
        self._step_count += 1
        market_data = self._get_market_data(env)
        context = {"regime": "UNKNOWN"}
        if self.regime_detector:
            context["regime"] = self.regime_detector.detect(env)

        # ── Layer 1: Collect signals with timeout ─────
        signals: List[Optional[SignalOutput]] = []
        producer_signals: Dict[str, Optional[SignalOutput]] = {}

        for producer in self.producers:
            timeout = self._timeouts.get(producer.name, 10.0)
            start = time.monotonic()
            sig, error_msg, timed_out = None, None, False
            try:
                sig = produce_with_timeout(producer, market_data, context, timeout)
                if sig is None:
                    timed_out = True
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"[{producer.name}] failed: {e}")

            elapsed_ms = (time.monotonic() - start) * 1000
            signals.append(sig)
            producer_signals[producer.name] = sig
            self._signal_logger.log_producer_result(
                producer_name=producer.name, signal=sig,
                latency_ms=elapsed_ms, error=error_msg, timed_out=timed_out,
            )

        # ── Min-valid-signals check ───────────────────
        valid_signals = [s for s in signals if s is not None]
        min_required = self.config.settings.min_valid_signals

        if len(valid_signals) < min_required:
            self._signal_logger.log_step_skipped(
                reason=f"Only {len(valid_signals)}/{len(signals)} valid, need {min_required}",
                valid_count=len(valid_signals), required=min_required,
            )
            if self._audit_log:
                self._audit_log.record_step(
                    step_id=self._step_count, producer_signals=producer_signals,
                    aggregated=None, regime=context["regime"], skipped=True,
                    skip_reason=f"insufficient signals: {len(valid_signals)}/{min_required}",
                )
            return {"action": "HOLD", "reason": "insufficient_valid_signals"}

        # ── Layer 2: Aggregate ────────────────────────
        agg_start = time.monotonic()
        aggregated = self.aggregator.aggregate(signals, context)
        agg_elapsed_ms = (time.monotonic() - agg_start) * 1000
        self._signal_logger.log_aggregation_result(
            aggregated=aggregated, valid_count=len(valid_signals),
            total_count=len(signals), latency_ms=agg_elapsed_ms,
        )

        if self._audit_log:
            self._audit_log.record_step(
                step_id=self._step_count, producer_signals=producer_signals,
                aggregated=aggregated, regime=context["regime"],
            )

        # ── Post-aggregation: same as Option A ────────
        action_signal = self.signal_processor.to_trade_action(aggregated)
        action_signal = self.risk.check(action_signal, env)
        action = agent.predict(env) if not user_input else None
        state, reward, done, result = env.step(action, action_signal)

        self.state_logger.log(state, result)
        self.publisher.publish(result)
        if self.risk.should_alert(env):
            self.notifier.alert(self.risk.alert_message)
        return result

    def _get_market_data(self, env) -> dict:
        return {"close": env.current_price, "volume": env.current_volume}
```

---

### Updated Pipeline Loader (Complete)

```python
# Trade/signals/pipeline_loader.py
import json
import logging
import importlib
from typing import List, Tuple
from Trade.signals.signal_producer import SignalProducer
from Trade.signals.signal_aggregator import SignalAggregator
from Trade.signals.pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)

PRODUCER_REGISTRY = {
    "rl": "Trade.signals.producers.rl_producer.RLSignalProducer",
    "tech_indicator": "Trade.signals.producers.tech_indicator_producer.TechIndicatorProducer",
    "llm": "Trade.signals.producers.llm_producer.LLMSignalProducer",
    "rule_based": "Trade.signals.producers.rule_producer.RuleBasedProducer",
    "replay": "Trade.signals.producers.replay_producer.ReplayProducer",
}

AGGREGATOR_REGISTRY = {
    "weighted_vote": "Trade.signals.aggregators.weighted_vote.WeightedVoteAggregator",
    "fixed_weight": "Trade.signals.aggregators.fixed_weight.FixedWeightAggregator",
    "majority_vote": "Trade.signals.aggregators.majority_vote.MajorityVoteAggregator",
    "meta_llm": "Trade.signals.aggregators.meta_llm.MetaLLMAggregator",
}


def load_and_validate_pipeline(
    config_path: str, runtime_deps: dict = None
) -> Tuple[List[SignalProducer], SignalAggregator, PipelineConfig]:
    """Load, schema-validate, build, and health-check the full pipeline."""
    runtime_deps = runtime_deps or {}

    with open(config_path) as f:
        raw = json.load(f)

    pipeline_section = raw.get("pipeline", raw)
    config = PipelineConfig(**pipeline_section)
    logger.info(
        f"Config validated: {len(config.producers)} producers, "
        f"aggregator={config.aggregator.type}, "
        f"min_valid_signals={config.settings.min_valid_signals}"
    )

    producers = []
    for node in config.producers:
        cls = _import_class(PRODUCER_REGISTRY[node.type])
        producer = cls.from_config(node.id, node.config, runtime_deps)
        producers.append(producer)

    agg_cls = _import_class(AGGREGATOR_REGISTRY[config.aggregator.type])
    aggregator = agg_cls.from_config(config.aggregator.config, runtime_deps)

    errors = []
    for producer in producers:
        try:
            producer.health_check()
            logger.info(f"  ✅ [{producer.name}] healthy")
        except Exception as e:
            errors.append(f"  ❌ [{producer.name}] {e}")
            logger.error(f"  ❌ [{producer.name}] health check failed: {e}")

    if errors:
        raise RuntimeError(
            f"Pipeline startup failed — {len(errors)} producer(s) unhealthy:\n"
            + "\n".join(errors)
        )

    logger.info("Pipeline ready.")
    return producers, aggregator, config


def _import_class(dotted_path: str):
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
```

---

### Updated Wiring in RL_TradeBot.py

```python
from Trade.signals.pipeline_loader import load_and_validate_pipeline
from Trade.trade_orchestrator import TradeOrchestrator

runtime_deps = {
    "env": env, "agent_ppo": agent, "strategy": strategy,
    "gemini-2.0-flash": gemini_client,
    "default_llm_client": gemini_client,
}

producers, aggregator, pipeline_config = load_and_validate_pipeline(
    "pipeline.json", runtime_deps
)

orchestrator = TradeOrchestrator(
    producers=producers, aggregator=aggregator,
    pipeline_config=pipeline_config,
    exchange=binance_adapter, risk=risk_manager,
    signal_processor=signal_processor,
    notifier=email_notifier, publisher=redis_publisher,
    state_logger=csv_logger,
)
```

---

### Complete pipeline.json (Production Default)

```json
{
  "pipeline": {
    "producers": [
      {
        "id": "rl_ppo",
        "type": "rl",
        "timeout_seconds": 1.0,
        "config": { "model_path": "pod_000042/", "agent": "ppo" }
      }
    ],
    "aggregator": {
      "type": "weighted_vote",
      "config": {}
    },
    "settings": {
      "min_valid_signals": 1,
      "audit_log_enabled": true,
      "audit_log_path": "logs/signal_audit.jsonl"
    }
  }
}
```

---

## Updated File Table (Complete)

| New File | Lines (est.) | Phase | Purpose |
|----------|-------------|-------|---------|
| `Trade/signals/__init__.py` | ~5 | 1 | Package init |
| `Trade/signals/signal_types.py` | ~50 | 1 | `SignalOutput` Pydantic model with validation |
| `Trade/signals/signal_producer.py` | ~40 | 1 | `SignalProducer` ABC with `health_check()` + `from_config()` |
| `Trade/signals/signal_aggregator.py` | ~25 | 1 | `SignalAggregator` ABC with `from_config()` |
| `Trade/signals/pipeline_config.py` | ~90 | 1 | Pydantic config model for `pipeline.json` schema validation |
| `Trade/signals/pipeline_loader.py` | ~90 | 1 | Load → validate → build → health check factory |
| `Trade/signals/timeout.py` | ~30 | 1 | `produce_with_timeout()` threading wrapper |
| `Trade/signals/signal_logger.py` | ~60 | 2 | Structured JSON logging for signal events |
| `Trade/signals/signal_audit.py` | ~50 | 2 | Append-only JSONL audit log |
| `Trade/signals/producers/__init__.py` | ~5 | 1 | Package init |
| `Trade/signals/producers/rl_producer.py` | ~60 | 1 | RL agent wrapper |
| `Trade/signals/producers/tech_indicator_producer.py` | ~55 | 1 | Strategy wrapper |
| `Trade/signals/producers/llm_producer.py` | ~80 | 1 | LLM signal source |
| `Trade/signals/producers/rule_producer.py` | ~55 | 1 | Hardcoded rules |
| `Trade/signals/producers/replay_producer.py` | ~60 | 2 | Backtest signal replay from audit log |
| `Trade/signals/producers/_template.py` | ~55 | 1 | Copy-paste starter for new producers |
| `Trade/signals/aggregators/__init__.py` | ~5 | 1 | Package init |
| `Trade/signals/aggregators/weighted_vote.py` | ~45 | 1 | Confidence-weighted average |
| `Trade/signals/aggregators/fixed_weight.py` | ~60 | 1 | Explicit weight map |
| `Trade/signals/aggregators/meta_llm.py` | ~75 | 1 | LLM as meta-reasoner |
| `pipeline.json` | ~20 | 1 | Default pipeline config |
| **Total new code** | **~1,080** | — | — |

---

## Resolved Gap Matrix

| Gap | Phase | Status | Implementation |
|---|---|---|---|
| ~~No timeout per producer~~ | 1 | ✅ Resolved | `timeout.py` + `timeout_seconds` in JSON config |
| ~~No JSON schema validation~~ | 1 | ✅ Resolved | `pipeline_config.py` — Pydantic model with cross-field validation |
| ~~No config validation at startup~~ | 1 | ✅ Resolved | `health_check()` on every producer, fail-fast in `load_and_validate_pipeline()` |
| ~~No min-producers policy~~ | 1 | ✅ Resolved | `min_valid_signals` in config → orchestrator skips step if too few valid |
| ~~No structured logging~~ | 2 | ✅ Resolved | `signal_logger.py` — JSON-structured log entries |
| ~~No signal audit log~~ | 2 | ✅ Resolved | `signal_audit.py` — JSONL append-only log |
| ~~No backtest integration~~ | 2 | ✅ Resolved | `replay_producer.py` — replays signals from audit log |
| ~~No producer template~~ | 1 | ✅ Resolved | `_template.py` — copy-paste starter with inline docs |
| No metrics / observability | 3 | ⚠️ Deferred | Hooks exist in `SignalLogger`; Prometheus adapter is Phase 3 |
| No config hot-reload | 3 | ⚠️ Deferred | File watcher or Redis signal — Phase 3 |
