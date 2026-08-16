# AI Hedge Finance (AHF) v2

> Production-grade Multi-Signal Trading System: Deep Reinforcement Learning (PPO) + Kalman Filter Alpha + LLM Ensemble on a 2-Layer Hierarchical Signal Architecture.

[![CI](https://github.com/HamiltonAIArtIO/ai-hedge-finance/actions/workflows/ci.yml/badge.svg)](https://github.com/HamiltonAIArtIO/ai-hedge-finance/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/managed%20by-uv-7C3AED)](https://github.com/astral-sh/uv)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)

---

## 🌟 Highlights

- **Hierarchical Signal Architecture (Option A-b)**: Decouples alpha generation (Layer 1 Producers) from ensemble synthesis (Layer 2 Aggregators).
- **Deep RL + Kalman Alpha**: PPO Actor-Critic agent trained with dual Kalman Filter alpha features extracted from multi-timeframe price feeds.
- **Deterministic Risk Gate**: Supervised risk management using Chain of Responsibility (`MaxDrawdownRule`, `TotalLossRule`, `KellyRule`).
- **Zero Kickstarter Scripts**: Clean, environment-driven CLI and Docker workflows managed with `uv`.
- **Append-Only Signal Audit**: Every decision, confidence score, and risk evaluation is logged to `signal_audit.jsonl` for instant replay and regression testing.

---

## 📐 Architecture Overview

```
Layer 1: Producers (Isolated)        Layer 2: Aggregator        Domain Execution Gate
┌──────────────────────────┐        ┌───────────────────┐      ┌─────────────────────────┐
│ RLSignalProducer (PPO)   │───────►│                   │      │ SignalProcessor         │
│ TechIndicatorProducer    │───────►│  WeightedVote /   │─────►│        ▼                │
│ LLMSignalProducer        │───────►│  FixedWeight /    │      │ RiskManager Gate        │
│ RuleBasedProducer        │───────►│  MetaLLM          │      │        ▼                │
└──────────────────────────┘        └───────────────────┘      │ OrderExecutor → Adapter │
                                                               └─────────────────────────┘
```

For full details, see the [Architecture Documentation](docs/architecture/ARCHITECTURE.md) and [Signal Contract](docs/architecture/SIGNAL_CONTRACT.md).

---

## 🚀 Quick Start

### 1. Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/HamiltonAIArtIO/ai-hedge-finance.git
cd ai-hedge-finance
cp .env.example .env

# Install dependencies and dev tools
uv sync --extra dev
```

### 2. Verify Environment

```bash
# Run unit & integration tests (170+ tests)
uv run pytest

# Check code style & types
uv run ruff check src/ tests/
uv run mypy src/
```

### 3. Run Trading Bot

```bash
# Start paper trading (simulated execution)
uv run ahf-trade

# Run with custom pipeline config
AHF_SYMBOL=ETHUSDT AHF_PIPELINE_CONFIG=configs/pipeline.multi_signal.json uv run ahf-trade

# Replay historical signal audit for backtesting
AHF_PIPELINE_CONFIG=configs/pipeline.replay.json uv run ahf-backtest
```

---

## 🐳 Docker Deployment

Run the system without local Python dependencies using Docker Compose:

```bash
# Start paper trading bot
docker compose up trade

# Run in background
docker compose up -d trade

# View logs
docker compose logs -f trade
```

---

## 📚 Documentation Index

- **Guides**:
  - [Quickstart Guide](docs/guides/QUICKSTART.md)
  - [Pipeline Configuration](docs/guides/PIPELINE_CONFIG.md)
  - [Contributing Alpha Signals](docs/guides/CONTRIBUTING_SIGNALS.md)
  - [Producers Reference](docs/guides/PRODUCERS.md)
  - [Aggregators Reference](docs/guides/AGGREGATORS.md)
  - [Backtesting & Replay](docs/guides/BACKTESTING.md)
- **Architecture**:
  - [Option A(b) Architecture Overview](docs/architecture/ARCHITECTURE.md)
  - [Signal Contract Specification](docs/architecture/SIGNAL_CONTRACT.md)
  - [Migration from v1 (diewalkure)](docs/architecture/MIGRATION_V1_TO_V2.md)

---

## 🛠 Project Status

| Phase | Milestone | Status |
|---|---|---|
| **Phase 0** | Scaffold, `uv`, `pyproject.toml`, CI/CD | ✅ Complete |
| **Phase 1** | Core Signal Pipeline Contracts (`SignalOutput`, ABCs) | ✅ Complete |
| **Phase 2** | Aggregators (`WeightedVote`, `FixedWeight`, `MajorityVote`, `MetaLLM`) | ✅ Complete |
| **Phase 3** | Domain Modules (`RiskManager`, `OrderExecutor`, `TradeOrchestrator`) | ✅ Complete |
| **Phase 4** | RL Engine Port (`BrunhildEnv_v11`, `AgentPPO`, `KalmanMovingAverage`) | ✅ Complete |
| **Phase 5** | Entrypoints (`ahf-trade`, `ahf-train`, `ahf-backtest`) & Docker | ✅ Complete |
| **Phase 6** | Regression Testing & Import Graph Verification | ✅ Complete |
| **Phase 7** | Clean Documentation Suite | ✅ Complete |

---

## 📄 License

Proprietary — All rights reserved.
