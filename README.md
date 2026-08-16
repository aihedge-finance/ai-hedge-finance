# AI Hedge Finance (ahf)

> Multi-signal trading system: RL + Technical Indicators + LLM ensemble, built on Option A(b) 2-layer signal architecture.

[![CI](https://github.com/your-org/ai-hedge-finance/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/ai-hedge-finance/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/managed%20by-uv-7C3AED)](https://github.com/astral-sh/uv)

## Architecture

Option A(b): Lightweight 2-Layer Hierarchical Signal Architecture

```
Layer 1 (Signal Producers)         Layer 2 (Aggregator)
┌─────────────────────────┐        ┌──────────────────────┐
│ RLSignalProducer (PPO)  │───────►│                      │
│ TechIndicatorProducer   │───────►│  WeightedVote /      │──► TradeOrchestrator
│ LLMSignalProducer       │───────►│  FixedWeight /       │
│ RuleBasedProducer       │───────►│  MetaLLM             │
└─────────────────────────┘        └──────────────────────┘
```

See `design/pre_upgrade_v2_analysis/` for full architecture documentation.

## Quick Start

```bash
# Install uv (if not already installed)
brew install uv

# Clone and set up
git clone https://github.com/your-org/ai-hedge-finance.git
cd ai-hedge-finance
cp .env.example .env  # fill in your API keys
uv sync --extra dev

# Run tests
uv run pytest

# Start trading (local)
uv run ahf-trade --config configs/pipeline.json

# Train a model
uv run ahf-train --symbol BTCUSDT --interval 15m
```

## Migration from v1 (diewalkure)

See `docs/architecture/MIGRATION_V1_TO_V2.md` _(coming in Phase 7)_.

## Status

| Phase | Status |
|-------|--------|
| 0: Scaffold | ✅ |
| 1: Signal Contracts | 🔲 |
| 2: Aggregators + Stubs | 🔲 |
| 3: Domain Modules | 🔲 |
| 4: RL Engine Port | 🔲 |
| 5: Entrypoints + Docker | 🔲 |
| 6: Regression Testing | 🔲 |
| 7: Documentation | ⏸️ |
