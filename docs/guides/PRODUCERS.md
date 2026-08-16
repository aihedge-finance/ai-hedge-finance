# Signal Producers Reference

Signal Producers are **Layer 1** of the Option A(b) pipeline. Each runs with an isolated threading timeout. All return an immutable `SignalOutput`.

---

## 1. `RLSignalProducer` (`type: "rl"`)

Wraps a PPO Actor-Critic agent (`AgentPPO`) interacting with `BrunhildEnv_v11`.

**Constructor:**
```python
RLSignalProducer(name, agent=None, env=None, confidence=0.7, model_path="", agent_type="ppo")
```

**Behaviour:**
- With a loaded agent: calls `agent.act_no_exploration(obs)` via the env's current state, normalises to `[-1.0, 1.0]`.
- Without agent/env: returns `HOLD` with `confidence=0.5` (graceful degradation).
- `health_check()` raises `RuntimeError` if agent or env is absent (used at startup to prevent silent failures).

**Pipeline Config:**
```json
{
  "id": "rl_ppo",
  "type": "rl",
  "timeout_seconds": 5.0,
  "config": {
    "model_path": "data/models/pod_000000",
    "agent": "ppo",
    "confidence": 0.70
  }
}
```

---

## 2. `TechIndicatorProducer` (`type: "tech_indicator"`)

Wraps quantitative technical strategies ported from the v1 codebase.

**Built-in Strategies:**
- **`double_kf`**: Dual Kalman Filter alpha processor with adaptive buy/sell delta covariance.
- **`rsi_macd`**: Multi-timeframe RSI momentum + MACD histogram crossover.
- **`double_ukf`**: Dual Unscented Kalman Filter for non-linear state estimation.
- **`squeeze_momentum`**: Volatility squeeze breakout indicator.

**Pipeline Config:**
```json
{
  "id": "tech_kf",
  "type": "tech_indicator",
  "timeout_seconds": 2.0,
  "config": {
    "strategy": "double_kf",
    "confidence": 0.65
  }
}
```

---

## 3. `LLMSignalProducer` (`type: "llm"`)

Connects to large language models for qualitative market reasoning, news sentiment, and macro analysis. Falls back to `HOLD` on any API failure.

**Pipeline Config:**
```json
{
  "id": "llm_gemini",
  "type": "llm",
  "timeout_seconds": 15.0,
  "config": {
    "model": "gemini-2.0-flash",
    "client_key": "gemini-2.0-flash",
    "temperature": 0.1
  }
}
```

---

## 4. `RuleBasedProducer` (`type: "rule_based"`)

Evaluates deterministic named boolean rules registered in the rule engine. Confidence scales with the fraction of agreeing rules.

**Pipeline Config:**
```json
{
  "id": "rule_rsi",
  "type": "rule_based",
  "timeout_seconds": 1.0,
  "config": {
    "rules": ["rsi_oversold_buy", "rsi_overbought_sell"],
    "confidence": 0.60
  }
}
```

---

## 5. `ReplayProducer` (`type: "replay"`)

Reads historical signals from an append-only `signal_audit.jsonl` log. Enables exact regression testing and parameter re-weighting without re-running ML inference.

**Pipeline Config:**
```json
{
  "id": "replay_signal",
  "type": "replay",
  "timeout_seconds": 1.0,
  "config": {
    "log_path": "data/logs/signal_audit.jsonl",
    "producer_id": "rl_ppo"
  }
}
```

When the replay log is exhausted, `ReplayProducer` returns a `HOLD` signal for all remaining steps.
