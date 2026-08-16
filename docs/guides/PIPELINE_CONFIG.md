# Pipeline Configuration Guide

In AI Hedge Finance (AHF) v2, the signal pipeline is fully declarative. You define your alpha sources, weights, timeouts, and aggregation algorithm in a single JSON file.

---

## 1. Pipeline Schema Overview

A pipeline configuration file consists of:
1. `version`: Schema version (e.g. `"2.0"`).
2. `producers`: List of signal producers (Layer 1).
3. `aggregator`: Aggregation strategy and parameters (Layer 2).
4. `settings`: Global timeout and execution policies.

### Example: Multi-Signal Ensemble (`configs/pipeline.multi_signal.json`)

```json
{
  "version": "2.0",
  "pipeline_id": "multi_signal_ensemble",
  "description": "RL (PPO) + Tech (Kalman Filter) + Sentiment (LLM)",
  "settings": {
    "timeout_ms": 500,
    "min_valid_signals": 1,
    "on_timeout": "skip"
  },
  "producers": [
    {
      "id": "rl_ppo",
      "type": "rl",
      "weight": 0.50,
      "enabled": true,
      "timeout_ms": 300,
      "config": {
        "model_path": "data/models/pod_000000/PROD/BTCUSDT_BrunhildEnv-v11_double_kf_dual_8h_00_PPO_CPU_CODING",
        "agent": "ppo",
        "confidence": 0.70
      }
    },
    {
      "id": "tech_kf",
      "type": "tech_indicator",
      "weight": 0.35,
      "enabled": true,
      "timeout_ms": 100,
      "config": {
        "strategy": "double_kf",
        "confidence": 0.65
      }
    },
    {
      "id": "llm_sentiment",
      "type": "llm",
      "weight": 0.15,
      "enabled": true,
      "timeout_ms": 800,
      "config": {
        "model": "gemini-2.5-flash",
        "temperature": 0.1,
        "prompt_template": "sentiment_v1"
      }
    }
  ],
  "aggregator": {
    "type": "fixed_weight",
    "params": {
      "normalize_weights": true
    }
  }
}
```

---

## 2. Producer Types

| Producer Type | Key | Description | Key Config Options |
|---|---|---|---|
| **Reinforcement Learning** | `"rl"` | PPO policy network inference | `model_path`, `agent`, `confidence` |
| **Technical Strategy** | `"tech_indicator"` | Algorithmic strategies (Kalman, RSI/MACD) | `strategy` (`"double_kf"`, `"rsi_macd"`), `confidence` |
| **LLM Sentiment/Macro** | `"llm"` | LLM market reasoning | `model`, `prompt_template`, `temperature` |
| **Rule-Based** | `"rule_based"` | Deterministic technical rules | `rules` list, `confidence` |
| **Audit Replay** | `"replay"` | Historical signal audit log replay | `log_path`, `producer_id` |

---

## 3. Aggregator Types

### 1. `fixed_weight`
Weights are fixed in the JSON configuration. The aggregated action is:
$$A_{agg} = \frac{\sum w_i \cdot A_i}{\sum w_i}$$
Confidence is the weighted average of individual confidences.

### 2. `weighted_vote`
Weights are dynamically adjusted by the producer's returned `confidence`:
$$W_i^{effective} = w_i^{config} \times \text{confidence}_i$$

### 3. `majority_vote`
Directional voting (BUY / HOLD / SELL). Direction with $>50\%$ total weight wins.

### 4. `meta_llm`
Passes all producer signals, confidences, and rationale into an LLM arbiter to resolve conflicting signals.

---

## 4. Built-in Configurations

| Config Path | Architecture | Primary Use Case |
|---|---|---|
| `configs/pipeline.json` | Single RL Producer (`rl_ppo`) + Weighted Vote | Standard RL trading |
| `configs/pipeline.multi_signal.json` | RL (0.5) + Tech (0.35) + LLM (0.15) | Production ensemble |
| `configs/pipeline.llm_ensemble.json` | Multi-LLM (Gemini, Claude, GPT) + Meta LLM | Macro & news trading |
| `configs/pipeline.replay.json` | Replay Producer | Backtesting & audit regression |

---

## 5. Loading Pipelines Programmatically

```python
from ahf.signals.pipeline_loader import load_pipeline

# Load pipeline with runtime dependencies
runtime_deps = {"rl_agent": loaded_agent, "rl_env": active_env}
producers, aggregator = load_pipeline("configs/pipeline.json", runtime_deps)

# Execute pipeline
market_data = {"5m": {"last_close": 67000.0, ...}}
signals = [p.produce(market_data, {}) for p in producers]
final_signal = aggregator.aggregate(signals)
```
