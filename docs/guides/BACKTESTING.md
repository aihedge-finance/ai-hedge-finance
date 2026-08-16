# Backtesting & Replay Guide

AI Hedge Finance (AHF) v2 provides two decoupled modes of backtesting:

1. **Signal Replay Backtesting**: Instantaneous simulation replaying past audited signals from `signal_audit.jsonl` through the risk engine and order executor.
2. **Environment Simulation Backtesting**: Full bar-by-bar market simulation using `BrunhildEnv_v11` with technical indicators and RL policy execution.

---

## 1. Signal Replay Backtesting

Every live or paper trading run appends complete signal and telemetry data to `data/logs/signal_audit.jsonl`. You can replay this log with modified risk parameters or aggregation weights without re-running models.

### Step 1: Configure Replay Pipeline (`configs/pipeline.replay.json`)

```json
{
  "version": "2.0",
  "pipeline_id": "replay_backtest",
  "producers": [
    {
      "id": "replay_rl",
      "type": "replay",
      "config": {
        "log_path": "data/logs/signal_audit.jsonl",
        "producer_id": "rl_ppo"
      }
    }
  ],
  "aggregator": {
    "type": "fixed_weight"
  }
}
```

### Step 2: Run Backtest
```bash
AHF_PIPELINE_CONFIG=configs/pipeline.replay.json uv run ahf-backtest
```

---

## 2. Environment Simulation Backtesting

To run a historical backtest through the Kalman Filter + PPO reinforcement learning engine:

```bash
# Run backtest script
uv run python -m ahf.rl.train.run --mode backtest --symbol BTCUSDT --strategy double_kf
```

---

## 3. Comparing Strategy PnL

The simulated portfolio tracker logs performance metrics on each step:
- **Cumulative Return (%)**
- **Max Drawdown (%)**
- **Sharpe / Sortino Ratio**
- **Kelly Sizing Utilization**
- **Veto Frequency by Risk Manager**

Audit logs can be analyzed using pandas:
```python
import pandas as pd
df = pd.read_json("data/logs/signal_audit.jsonl", lines=True)
print(df[["timestamp", "action", "confidence", "risk_verdict"]].head())
```
