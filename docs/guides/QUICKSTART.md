# Quickstart Guide

Welcome to **AI Hedge Finance (AHF) v2**. This guide will get you up and running with paper trading, backtesting, and reinforcement learning training in under 5 minutes.

---

## 1. Prerequisites

- **Python 3.11+**
- [**uv**](https://docs.astral.sh/uv/) (Modern fast Python package manager)
- (Optional) **Docker & Docker Compose** for containerized execution

### Install `uv`
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via Homebrew
brew install uv
```

---

## 2. Installation & Setup

Clone the repository and install dependencies using `uv`:

```bash
cd ai-hedge-finance

# Copy environment configuration
cp .env.example .env

# Install project dependencies and dev tools (pytest, ruff, mypy)
uv sync --extra dev
```

Run the automated test suite to ensure your environment is healthy:

```bash
uv run pytest
```
You should see 170+ passing tests.

---

## 3. Configuration via `.env`

AHF v2 is configured entirely via environment variables and `.env` (no complex kickstarter scripts).

Key settings in `.env`:

```ini
# Operating mode: PAPER (simulated orders) | LIVE (real Binance API) | SIMULATION
TRADING_MODE=PAPER

# Trading pair & pipeline config
SYMBOL=BTCUSDT
PIPELINE_CONFIG=configs/pipeline.json

# Initial simulated capital
INITIAL_CAPITAL=1000.0

# Risk guardrails
MAX_DRAWDOWN_PCT=0.15
MAX_LOSS_PCT=0.30
KELLY_FRACTION=0.5

# Pipeline tuning
BUY_THRESHOLD=0.05
SELL_THRESHOLD=-0.05
CONFIDENCE_FLOOR=0.0

# API Keys (Optional for PAPER trading, Required for LIVE or LLM ensemble)
BINANCE_API_KEY=
BINANCE_API_SECRET=
GEMINI_API_KEY=
OPENAI_API_KEY=
```

---

## 4. Running the Trading Bot

### Paper Trading (Default)
Paper trading uses real market feeds with simulated in-memory order execution (`DummyAdapter`), logging signals and audit trails to `data/logs/signal_audit.jsonl`:

```bash
# Using CLI entrypoint
uv run ahf-trade

# Or override parameters inline
AHF_SYMBOL=ETHUSDT AHF_PIPELINE_CONFIG=configs/pipeline.multi_signal.json uv run ahf-trade
```

### Backtesting with Replay Pipeline
Replay recorded signals through the risk engine and order executor:

```bash
AHF_PIPELINE_CONFIG=configs/pipeline.replay.json uv run ahf-backtest
```

### RL Agent Training
Train the PPO reinforcement learning agent with the Kalman Filter environment:

```bash
AHF_SYMBOL=BTCUSDT AHF_STRATEGY=double_kf uv run ahf-train
```

---

## 5. Running with Docker

You can run the entire trading stack without installing Python locally:

```bash
# Start paper trading bot
docker compose up trade

# Run in background
docker compose up -d trade

# Check live trading logs
docker compose logs -f trade

# Stop container
docker compose down
```

---

## 6. Next Steps

- Explore [Pipeline Configuration Guide](PIPELINE_CONFIG.md) to customize signal weights.
- Learn how to build custom alpha signals in [Contributing Signals](CONTRIBUTING_SIGNALS.md).
- Read the [Architecture Overview](../architecture/ARCHITECTURE.md) to understand the 2-layer signal hierarchy and risk gate.
