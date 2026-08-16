# Migration Guide: v1 (`diewalkure`) to v2 (`ai-hedge-finance`)

This document records the architectural improvements, breaking changes, and migration mappings between the legacy `diewalkure` repository and `ai-hedge-finance` v2.

---

## 1. Summary of Major Changes

| Area | v1 (`diewalkure`) | v2 (`ai-hedge-finance`) |
|---|---|---|
| **Package Management** | Poetry / `requirements.txt` | `uv` with PEP 621 `pyproject.toml` |
| **Python Version** | 3.8 / 3.9 | **3.11+** |
| **Architecture** | Monolithic trading loop + RL coupling | **Option A(b) 2-Layer Signal Hierarchy** |
| **Configuration** | `config.py` + `sys_config.py` + Kickstarter scripts | Declarative Pydantic `Settings` + `pipeline.json` |
| **Multi-Signal** | Single RL agent or single strategy | **Dynamic Ensemble**: RL + Kalman + LLM + Rules |
| **Risk Management** | Ad-hoc stop-loss in strategy loops | **Chain of Responsibility `RiskManager` Gate** |
| **Telemetry** | Unstructured text logging | Structured JSON + Append-only `signal_audit.jsonl` |
| **Docker** | Kickstarter wrapper containers | Single multi-stage production Docker image |

---

## 2. Namespace & Module Mapping

All old imports have been migrated to the clean `ahf.*` top-level namespace:

| v1 Import Path | v2 New Import Path |
|---|---|
| `from envs.BrunhildEnv_v11 import ...` | `from ahf.rl.envs.BrunhildEnv_v11 import ...` |
| `from agents.AgentPPO import ...` | `from ahf.rl.agents.AgentPPO import ...` |
| `from train.run import ...` | `from ahf.rl.train.run import ...` |
| `from TradingStrategy.double_kf.Strategy import ...` | `from ahf.rl.strategies.double_kf.Strategy import ...` |
| `from preprocessor.kf.*` | `from ahf.preprocessor.kf.*` |
| `from app.utils import ...` | `from ahf.utils.utils import ...` |
| `from app.enums import ...` | `from ahf.core.enums import ...` |

---

## 3. Removal of Kickstarter Mechanism

In v1, running the bot required `RL_Kickstarter.py` or complex shell scripts. In v2:
- **Direct CLI Execution**: `uv run ahf-trade`, `uv run ahf-train`, `uv run ahf-backtest`.
- **Environment Variables**: Configure everything via `.env` or container environment flags.
- **Docker Compose**: Pre-configured services for `trade`, `train`, and `backtest`.

---

## 4. Preservation of Mathematical Parity

- **`BrunhildEnv_v11`**: Observation vector calculations, reward functions, and step physics preserved identically.
- **`AgentPPO`**: Neural network weights and policy distributions ported verbatim.
- **`KalmanMovingAverage`**: Dual Kalman filter alpha formulas and delta parameters preserved with exact arithmetic.
