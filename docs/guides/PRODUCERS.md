# Signal Producers Reference

Signal Producers represent **Layer 1** of the Option A(b) signal pipeline. Each producer is an independent alpha generator running in isolated execution with strict timeouts.

---

## 1. `RLSignalProducer` (`type: "rl"`)

Wraps a Deep Reinforcement Learning agent (`AgentPPO` / `DRLAgent`) interacting with `BrunhildEnv_v11`.

### Capabilities:
- Extracts observations from continuous multi-timeframe price states.
- Executes forward inference via Actor network: `agent.act_no_exploration(obs)`.
- Normalizes raw discrete or continuous action to $[-1.0, 1.0]$.
- Falls back to `HOLD` signal with `confidence=0.5` if agent is offline.

### Configuration:
```json
{
  "id": "rl_ppo",
  "type": "rl",
  "weight": 0.50,
  "config": {
    "model_path": "data/models/pod_000000/PROD/BTCUSDT_BrunhildEnv-v11_...",
    "agent": "ppo",
    "confidence": 0.70
  }
}
```

---

## 2. `TechIndicatorProducer` (`type: "tech_indicator"`)

Wraps quantitative technical strategies ported from `diewalkure` (`TradingStrategy`).

### Built-in Strategies:
- **`double_kf`**: Dual Kalman Filter alpha processor with adaptive delta observation covariance.
- **`rsi_macd`**: Multi-timeframe RSI momentum filter with MACD histogram crossover.
- **`double_ukf`**: Dual Unscented Kalman Filter for non-linear state estimation.
- **`squeeze_momentum`**: Volatility squeeze breakout indicator.

### Configuration:
```json
{
  "id": "tech_kf",
  "type": "tech_indicator",
  "weight": 0.35,
  "config": {
    "strategy": "double_kf",
    "confidence": 0.65
  }
}
```

---

## 3. `LLMSignalProducer` (`type: "llm"`)

Connects to large language models (Gemini 2.5 Flash, Claude 3.5 Sonnet, GPT-4o) for qualitative market reasoning, news sentiment, and macro context analysis.

### Configuration:
```json
{
  "id": "llm_sentiment",
  "type": "llm",
  "weight": 0.15,
  "config": {
    "model": "gemini-2.5-flash",
    "temperature": 0.1,
    "prompt_template": "sentiment_v1"
  }
}
```

---

## 4. `RuleBasedProducer` (`type: "rule_based"`)

Evaluates deterministic boolean technical rules (e.g. SMA crossovers, RSI oversold/overbought).

### Configuration:
```json
{
  "id": "rule_rsi",
  "type": "rule_based",
  "weight": 0.20,
  "config": {
    "rules": ["rsi_oversold_buy", "rsi_overbought_sell"],
    "confidence": 0.60
  }
}
```

---

## 5. `ReplayProducer` (`type: "replay"`)

Reads historical signals from an append-only `signal_audit.jsonl` log. Enables exact regression testing and parameter re-weighting without re-running ML inference.

### Configuration:
```json
{
  "id": "replay_signal",
  "type": "replay",
  "config": {
    "log_path": "data/logs/signal_audit.jsonl",
    "producer_id": "rl_ppo"
  }
}
```
