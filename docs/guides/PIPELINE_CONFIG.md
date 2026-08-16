# Pipeline Configuration Guide

In AI Hedge Finance (AHF) v2, the signal pipeline is fully declarative. You define your alpha sources, weights, timeouts, and aggregation algorithm in a single JSON file.

---

## 1. Pipeline Schema Overview

A pipeline configuration file consists of:
1. `producers`: List of signal producers (Layer 1).
2. `aggregator`: Aggregation strategy and parameters (Layer 2).
3. `settings`: Global execution policies.

### Example: Multi-Signal Ensemble (`configs/pipeline.multi_signal.json`)

```json
{
  "producers": [
    {
      "id": "rl_ppo",
      "type": "rl",
      "timeout_seconds": 5.0,
      "config": {
        "model_path": "data/models/pod_000000",
        "agent": "ppo"
      }
    },
    {
      "id": "tech_kf",
      "type": "tech_indicator",
      "timeout_seconds": 2.0,
      "config": {
        "strategy": "double_kf"
      }
    },
    {
      "id": "llm_gemini",
      "type": "llm",
      "timeout_seconds": 15.0,
      "config": {
        "model": "gemini-2.0-flash",
        "client_key": "gemini-2.0-flash"
      }
    }
  ],
  "aggregator": {
    "type": "fixed_weight",
    "config": {
      "weights": {
        "rl_ppo":     0.50,
        "tech_kf":    0.30,
        "llm_gemini": 0.20
      }
    }
  },
  "settings": {
    "min_valid_signals": 2,
    "audit_log_enabled": true,
    "audit_log_path": "data/logs/signal_audit.jsonl"
  }
}
```

> **Note:** There is no top-level `"version"` field. Producer-level `"weight"` fields are only used by the `weighted_vote` aggregator; `fixed_weight` reads weights from `aggregator.config.weights` keyed by producer `id`.

---

## 2. Producer Schema

Each producer entry has:

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | string | ✅ | Unique identifier (must match weight keys for `fixed_weight`) |
| `type` | string | ✅ | Registry key — see supported types below |
| `timeout_seconds` | float | ✅ (default: `5.0`) | Hard per-producer execution timeout |
| `config` | dict | ✅ | Producer-specific configuration |

---

## 3. Producer Types

| Producer Type | Registry Key | Description | Key `config` Options |
|---|---|---|---|
| **Reinforcement Learning** | `"rl"` | PPO policy network inference via `BrunhildEnv_v11` | `model_path`, `agent` (`"ppo"`), `confidence` |
| **Technical Strategy** | `"tech_indicator"` | Kalman / RSI-MACD algorithmic strategies | `strategy` (`"double_kf"`, `"rsi_macd"`, `"double_ukf"`, `"squeeze_momentum"`), `confidence` |
| **LLM Sentiment** | `"llm"` | LLM market reasoning and sentiment | `model`, `client_key`, `temperature` |
| **Rule-Based** | `"rule_based"` | Deterministic technical rule evaluator | `rules` list, `confidence` |
| **Audit Replay** | `"replay"` | Historical signal audit log replay | `log_path`, `producer_id` |

---

## 4. Aggregator Types

| Registry Key | Algorithm | When to Use |
|---|---|---|
| `"fixed_weight"` | Static per-producer weights defined in `aggregator.config.weights` | Tuned production ensembles with known optimal weights |
| `"weighted_vote"` | Action weighted by `w_config × confidence` dynamically | When you trust confidence scores to self-regulate influence |
| `"majority_vote"` | BUY/HOLD/SELL directional democratic vote by weight | Conflict-tolerant ensembles; resistant to outliers |
| `"meta_llm"` | LLM arbiter synthesises all signals and rationale | Conflicting signals that require contextual reasoning |

### `fixed_weight` Details
The `weights` dict under `aggregator.config.weights` must list **all** producer `id`s — missing or extra keys raise a validation error at load time.

---

## 5. Built-in Configurations

| Config Path | Architecture | Aggregator | Primary Use Case |
|---|---|---|---|
| `configs/pipeline.json` | Single RL producer | `weighted_vote` | Standard RL paper/live trading |
| `configs/pipeline.multi_signal.json` | RL (0.50) + Tech (0.30) + LLM (0.20) | `fixed_weight` | Production ensemble |
| `configs/pipeline.llm_ensemble.json` | Multi-LLM producers | `meta_llm` | News / macro regime trading |
| `configs/pipeline.replay.json` | Replay producer | `weighted_vote` | Backtesting & audit regression |

---

## 6. Loading Pipelines Programmatically

```python
from ahf.signals.pipeline_loader import load_pipeline

# Load pipeline with optional runtime dependencies (agent, env objects)
runtime_deps: dict = {}
producers, aggregator = load_pipeline("configs/pipeline.json", runtime_deps)

# Execute one pipeline step
market_data = {"5m": {"last_close": 67000.0}}
signals = [p.produce(market_data, {}) for p in producers]
final_signal = aggregator.aggregate(signals)
```
