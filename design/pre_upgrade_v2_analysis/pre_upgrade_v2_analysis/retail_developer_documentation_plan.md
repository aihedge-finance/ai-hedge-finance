# Retail Developer Documentation Plan

**Purpose**: Specify every document that must be generated for retail developers (hobbyist algo traders, open-source contributors, students) to productively use the Option A(b) signal pipeline without reading source code.

**Audience**: These docs target users who:
- Follow tutorials step-by-step
- Edit JSON configs, not Python ABCs
- Want working examples before theory
- Expect clear error messages when they make mistakes
- May not have ML/quant backgrounds

---

## Document Inventory

| # | Document | Priority | Format | Location |
|---|----------|----------|--------|----------|
| 1 | Quick Start Guide | 🔴 Critical | Markdown | `docs/signals/QUICKSTART.md` |
| 2 | Pipeline Configuration Reference | 🔴 Critical | Markdown | `docs/signals/PIPELINE_CONFIG.md` |
| 3 | Contributing a New Producer | 🔴 Critical | Markdown | `docs/signals/CONTRIBUTING_SIGNALS.md` |
| 4 | Signal Contract Specification | 🟡 Important | Markdown | `docs/signals/SIGNAL_CONTRACT.md` |
| 5 | Built-in Producers Reference | 🟡 Important | Markdown | `docs/signals/PRODUCERS.md` |
| 6 | Built-in Aggregators Reference | 🟡 Important | Markdown | `docs/signals/AGGREGATORS.md` |
| 7 | Backtesting with Replay Producer | 🟡 Important | Markdown | `docs/signals/BACKTESTING.md` |
| 8 | Troubleshooting & FAQ | 🟢 Nice-to-have | Markdown | `docs/signals/TROUBLESHOOTING.md` |
| 9 | Architecture Overview (Visual) | 🟢 Nice-to-have | Markdown | `docs/signals/ARCHITECTURE.md` |

---

## 1. Quick Start Guide

**File**: `docs/signals/QUICKSTART.md`  
**Priority**: 🔴 Critical  
**Goal**: Get a retail developer from zero to running the signal pipeline in 10 minutes.

### Content Outline

```markdown
# Quick Start: Multi-Signal Trading Pipeline

## Prerequisites
- Python 3.10+
- pydantic >= 2.0
- An existing diewalkure installation (see main README)

## Step 1: Check Default Config
- Show the default `pipeline.json` that ships with the repo (single RL producer)
- Explain that this is backward-compatible with the current system

## Step 2: Run With Default Config
- Command to start the bot with the default pipeline
- Expected startup output (health check logs)
- What the signal audit log looks like

## Step 3: Add a Second Producer
- Copy the Example 1 config from this doc (RL + Tech)
- Show the diff from the default config
- Run again, show 2 producers in the health check output

## Step 4: Try Three LLMs
- Copy the Example 2 config (3 LLMs)
- Explain what API keys/clients are needed
- Show the expected output with 3 LLM signals being aggregated

## Step 5: Understand the Output
- Explain what `signal_audit.jsonl` contains
- Show how to read the audit log to see each producer's vote
- Show a simple Python script to plot signal history

## Next Steps
- Link to PIPELINE_CONFIG.md for all config options
- Link to CONTRIBUTING_SIGNALS.md to build your own producer
- Link to PRODUCERS.md for all built-in producers
```

### Key Rules
- Every step must have a **copy-paste command or JSON block**
- Every step must show **expected output** so the user can verify
- No unexplained jargon (define "producer", "aggregator", "signal" on first use)
- Include a **"What could go wrong"** callout per step

---

## 2. Pipeline Configuration Reference

**File**: `docs/signals/PIPELINE_CONFIG.md`  
**Priority**: 🔴 Critical  
**Goal**: Complete reference for every field in `pipeline.json` with examples and validation error messages.

### Content Outline

```markdown
# Pipeline Configuration Reference

## File Location
- Default: `pipeline.json` in project root
- Override via CLI: `--pipeline-config /path/to/custom.json`

## Full Schema

### Top-Level Structure
- `pipeline.producers[]` — list of signal producers
- `pipeline.aggregator` — the aggregation strategy
- `pipeline.settings` — safety and logging settings

### Producer Fields
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | string | Yes | — | Unique name, no spaces |
| `type` | enum | Yes | — | One of: rl, tech_indicator, llm, rule_based, replay |
| `timeout_seconds` | float | No | 10.0 | Max wait time (0 < x ≤ 300) |
| `config` | object | No | {} | Type-specific settings (see below) |

### Per-Type Config Fields
#### type: "rl"
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_path` | string | Yes | Path to trained model directory |
| `agent` | string | Yes | Agent type: "ppo", "sac", etc. |

#### type: "tech_indicator"
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `strategy` | string | Yes | Strategy name: "double_kf", "rsi", "macd" |
| `params_file` | string | No | Path to strategy params JSON |

#### type: "llm"
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | LLM model identifier |
| `prompt_template` | string | Yes | Prompt template name |
| `client_key` | string | No | Key to look up in runtime_deps |

#### type: "replay"
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `replay_file` | string | Yes | Path to signal_audit.jsonl to replay |

### Aggregator Fields
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `type` | enum | Yes | — | One of: weighted_vote, fixed_weight, majority_vote, meta_llm |
| `config` | object | No | {} | Type-specific settings |

#### type: "fixed_weight"
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `config.weights` | object | Yes | Map of producer_id → weight (must sum meaningfully) |

#### type: "meta_llm"
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `config.model` | string | Yes | LLM model for meta-reasoning |
| `config.prompt_template` | string | Yes | Prompt template name |

### Settings Fields
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `min_valid_signals` | int | No | 1 | Minimum producers that must succeed |
| `audit_log_enabled` | bool | No | true | Enable/disable signal audit log |
| `audit_log_path` | string | No | "logs/signal_audit.jsonl" | Audit log file path |

## Example Configurations
- Example 1: Single RL (backward compatible)
- Example 2: Mixed (RL + Tech + LLM)
- Example 3: Three LLMs only
- Example 4: Two RL models with Meta-LLM aggregator
- Example 5: Backtest with replay

## Common Validation Errors
- Table of error message → what you did wrong → how to fix
```

---

## 3. Contributing a New Producer

**File**: `docs/signals/CONTRIBUTING_SIGNALS.md`  
**Priority**: 🔴 Critical  
**Goal**: Step-by-step guide for a retail developer to add a new signal producer, from copy-paste template to running in the pipeline.

### Content Outline

```markdown
# How to Add a New Signal Producer

## Overview
A signal producer is any Python class that takes market data and returns a 
trading signal (buy/sell/hold + confidence). This guide walks you through 
creating one from scratch.

## Step 1: Copy the Template
```
cp Trade/signals/producers/_template.py Trade/signals/producers/my_producer.py
```

## Step 2: Rename and Implement
- Rename the class
- Implement `produce()` — your core logic
- Implement normalization (explain the [-1, 1] contract)
  - Table of normalization strategies: linear, tanh, clamp
  - Common pitfalls: forgetting to normalize, returning raw RSI values

## Step 3: Implement `from_config()`
- Map your JSON config fields to constructor args
- How to access runtime_deps for things that can't be in JSON

## Step 4: Implement `health_check()`
- What to verify: file exists, API key set, model loads
- Common pattern: try loading the model / making a test API call

## Step 5: Register in Pipeline Loader
- Add one line to PRODUCER_REGISTRY in pipeline_loader.py
- Explain the dotted path format

## Step 6: Add to pipeline.json
- Show a minimal JSON entry
- Show how to set timeout_seconds

## Step 7: Test It
- Start the bot, verify health check passes
- Check signal_audit.jsonl for your producer's output
- Verify action is in [-1, 1] and confidence is in [0, 1]

## Complete Working Example: Sentiment Producer
- A full 40-line producer that reads from a sentiment API
- Shows the complete lifecycle: config → build → health check → produce → audit

## Checklist
- [ ] Class implements SignalProducer ABC
- [ ] produce() returns SignalOutput with action ∈ [-1, 1], confidence ∈ [0, 1]
- [ ] from_config() maps JSON to constructor
- [ ] health_check() verifies dependencies
- [ ] Registered in PRODUCER_REGISTRY
- [ ] Added to pipeline.json
- [ ] Tested: health check passes, audit log shows correct values
```

---

## 4. Signal Contract Specification

**File**: `docs/signals/SIGNAL_CONTRACT.md`  
**Priority**: 🟡 Important  
**Goal**: Formal spec of the `SignalOutput` contract — what it means, why it matters, and how to normalize different signal types.

### Content Outline

```markdown
# Signal Contract: SignalOutput

## What Is It
- The universal data format between Layer 1 (producers) and Layer 2 (aggregator)
- Enforced by Pydantic — invalid values are rejected immediately

## Fields

### signal_name (str)
- Must match the producer's `name` property
- Used for audit logging and debugging

### action (float, [-1.0, +1.0])
- -1.0 = strong sell
- 0.0 = hold / neutral
- +1.0 = strong buy
- Intermediate values = proportional conviction
- Values outside range → ValidationError → producer treated as failed

### confidence (float, [0.0, 1.0])
- 0.0 = no confidence (abstaining)
- 1.0 = maximum confidence
- Used by aggregators for weighting
- Low confidence = signal has less influence on final decision

### metadata (Optional[dict])
- Free-form, not part of the contract
- Standard aggregators never read it
- MetaLLMAggregator reads it as best-effort context
- Recommended keys: model_version, raw_scores, reasoning, latency_ms, error

## Normalization Guide
- Table: source type → native output → recommended normalization → action → confidence
- Common normalization functions: linear, tanh, sigmoid, clamp
- Code examples for each

## What Happens on Violation
- Diagram: ValidationError → caught by orchestrator → None passed to aggregator
- The producer is treated as if it crashed — no special handling needed

## Design Philosophy
- Validate at the boundary, not inside the aggregator
- Each producer owns its translation
- The aggregator trusts blindly
```

---

## 5. Built-in Producers Reference

**File**: `docs/signals/PRODUCERS.md`  
**Priority**: 🟡 Important  
**Goal**: Reference card for every built-in producer type — what it does, its config options, and when to use it.

### Content Outline

```markdown
# Built-in Signal Producers

## RLSignalProducer (type: "rl")
- What it does: Wraps a trained RL agent (PPO, SAC, etc.)
- Config fields: model_path, agent
- How it normalizes: discrete → [-1, 0, +1] map; continuous → passthrough
- Confidence: derived from value head or softmax
- When to use: you have a trained RL model

## TechIndicatorProducer (type: "tech_indicator")
- What it does: Wraps TradingStrategy (RSI, MACD, Kalman, etc.)
- Config fields: strategy, params_file
- How it normalizes: strategy-specific (RSI linear map, MACD tanh, etc.)
- Confidence: distance from neutral
- When to use: you want classical technical analysis signals

## LLMSignalProducer (type: "llm")
- What it does: Asks an LLM to analyze market data and return a signal
- Config fields: model, prompt_template, client_key
- How it normalizes: parses structured JSON output from LLM
- Confidence: parsed from LLM response
- When to use: you want LLM reasoning as a signal source
- Note: requires API key + adds latency

## RuleBasedProducer (type: "rule_based")
- What it does: Evaluates hardcoded rules (e.g., "if RSI < 30, buy")
- Config fields: rules_file or inline rules
- How it normalizes: boolean → [-1, +1] with strength
- Confidence: based on how many sub-rules match
- When to use: you have domain-specific rules that don't need ML

## ReplayProducer (type: "replay")
- What it does: Replays pre-recorded signals from signal_audit.jsonl
- Config fields: replay_file
- When to use: backtesting with deterministic signals, avoiding LLM API costs
- Note: signals are consumed sequentially, returns neutral when exhausted
```

---

## 6. Built-in Aggregators Reference

**File**: `docs/signals/AGGREGATORS.md`  
**Priority**: 🟡 Important  
**Goal**: Reference for every built-in aggregator — behavior, config, and when to choose each.

### Content Outline

```markdown
# Built-in Signal Aggregators

## WeightedVoteAggregator (type: "weighted_vote")
- Behavior: weights each signal by its confidence score
- Formula: action = Σ(action_i × confidence_i) / Σ(confidence_i)
- Config: none (automatic)
- When to use: default, works well when producers have meaningful confidence scores

## FixedWeightAggregator (type: "fixed_weight")
- Behavior: applies fixed weights per producer (ignores confidence for weighting)
- Formula: action = Σ(action_i × weight_i) / Σ(weight_i)
- Config: `weights` map (producer_id → weight)
- When to use: you want explicit control over how much each source matters
- Note: weights for ALL producers must be specified

## MajorityVoteAggregator (type: "majority_vote")
- Behavior: each producer votes buy/sell/hold based on sign of action
- Formula: majority direction wins, magnitude = average of agreeing signals
- Config: none
- When to use: diverse producers where you want a democratic decision

## MetaLLMAggregator (type: "meta_llm")
- Behavior: sends all producer signals + metadata to an LLM for meta-reasoning
- Formula: LLM reads all signals and makes a unified decision
- Config: model, prompt_template
- When to use: you want an LLM to reason about WHY signals agree/disagree
- Note: adds latency and cost, but provides explainable aggregation

## Comparison Matrix
| Aggregator | Speed | Cost | Explainability | Best For |
|---|---|---|---|---|
| weighted_vote | Fast | Free | Low | Default, 2-4 diverse producers |
| fixed_weight | Fast | Free | Medium | When you know relative reliability |
| majority_vote | Fast | Free | Medium | Democratic, diverse producers |
| meta_llm | Slow | API cost | High | Explainable decisions, research |
```

---

## 7. Backtesting with Replay Producer

**File**: `docs/signals/BACKTESTING.md`  
**Priority**: 🟡 Important  
**Goal**: Show how to use the replay producer for deterministic, cost-free backtesting.

### Content Outline

```markdown
# Backtesting the Signal Pipeline

## The Problem
- LLM producers call APIs → expensive and non-deterministic
- You can't reproduce yesterday's results
- Backtesting with live APIs is impractical

## The Solution: Replay Producer
- During live trading, the signal audit log records every producer's output
- During backtesting, the replay producer reads from the audit log
- Same signals, same order, no API calls

## Step 1: Run Live (or paper trading) to Collect Signals
- Enable audit logging in pipeline.json
- Run for N steps to build up signal_audit.jsonl

## Step 2: Create a Backtest Pipeline Config
- Replace LLM producer with replay type
- Keep RL/tech producers live (they're fast and free)
- Show the complete backtest pipeline.json

## Step 3: Run the Backtest
- Use DummyAdapter for exchange
- Command to start backtest
- Expected output

## Step 4: Compare Results
- Show how to diff two audit logs
- Simple Python script to compare signal_audit_live.jsonl vs signal_audit_backtest.jsonl

## Limitations
- Replay is positional (step N replays Nth recorded signal)
- No time alignment — if market data changes, replay signals may be stale
- For rigorous backtesting, also cache market data
```

---

## 8. Troubleshooting & FAQ

**File**: `docs/signals/TROUBLESHOOTING.md`  
**Priority**: 🟢 Nice-to-have  
**Goal**: Common problems, error messages, and solutions.

### Content Outline

```markdown
# Troubleshooting & FAQ

## Startup Errors

### "Pipeline startup failed — N producer(s) unhealthy"
- Cause: health_check() failed for one or more producers
- Check: are API keys set? Model files present? Network available?

### ValidationError on pipeline.json load
- Cause: typo in config, wrong type, missing required field
- Check: compare your config against PIPELINE_CONFIG.md schema

### "Unknown producer type 'xyz'"
- Cause: type string doesn't match any registered type
- Fix: use one of: rl, tech_indicator, llm, rule_based, replay

## Runtime Errors

### "insufficient_valid_signals" — bot keeps returning HOLD
- Cause: too many producers failing, min_valid_signals threshold not met
- Check: signal_audit.jsonl for which producers are returning null
- Fix: lower min_valid_signals, fix failing producers, or increase timeout_seconds

### LLM producer always times out
- Cause: timeout_seconds too low for the LLM model
- Fix: increase timeout_seconds (try 15-30s for large models)

### "action must be in [-1.0, 1.0], got X"
- Cause: your custom producer isn't normalizing its output
- Fix: see normalization guide in SIGNAL_CONTRACT.md

## FAQ

### Can I use 3 of the same type? (e.g., 3 LLMs)
- Yes, each producer entry is independent
- Just give them unique ids

### Can I change the pipeline without restarting?
- Not yet (config hot-reload is deferred to Phase 3)
- Edit pipeline.json, then restart the bot

### What happens if all producers fail?
- min_valid_signals (default 1) means at least 1 must succeed
- If all fail → step returns HOLD, nothing is traded

### Can I use this for backtesting?
- Yes, see BACKTESTING.md for the replay producer pattern
```

---

## 9. Architecture Overview (Visual)

**File**: `docs/signals/ARCHITECTURE.md`  
**Priority**: 🟢 Nice-to-have  
**Goal**: Visual-first explanation of how the 2-layer pipeline works, with diagrams.

### Content Outline

```markdown
# Signal Pipeline Architecture

## The Big Picture
- Mermaid diagram: data flow from market → producers → aggregator → orchestrator → exchange
- One paragraph explanation

## Layer 1: Signal Producers
- What they do
- How many you can have (any N)
- What they return (SignalOutput)
- Diagram: producer internals (native format → normalize → validate → output)

## Layer 2: Aggregator
- What it does
- How it receives signals (list, may contain None for failed producers)
- Diagram: aggregation with re-weighting when producers fail

## The Orchestrator
- How it ties everything together
- Sequence diagram: step-by-step flow of one trading step
- Timeout, min_valid_signals, audit log — where each feature runs

## Data Flow Diagram
- Mermaid: complete flow from JSON config → pipeline_loader → health check
  → runtime loop → audit log

## Config-Driven Design
- How pipeline.json controls the topology
- No code changes needed for different producer/aggregator combinations
```

---

## Generation Order

For maximum impact with minimum effort, generate docs in this order:

```
Phase 1 (Ship with code):
  1. QUICKSTART.md          — gets users running immediately
  2. PIPELINE_CONFIG.md     — reference for the JSON they'll edit most
  3. CONTRIBUTING_SIGNALS.md — enables community contributions

Phase 2 (First week after release):
  4. SIGNAL_CONTRACT.md     — formal spec for advanced users
  5. PRODUCERS.md           — reference card
  6. AGGREGATORS.md         — reference card

Phase 3 (Community maturity):
  7. BACKTESTING.md         — enables reproducible research
  8. TROUBLESHOOTING.md     — reduces support burden
  9. ARCHITECTURE.md        — onboards deep contributors
```

---

## Writing Style Guide (For All Docs)

| Rule | Rationale |
|---|---|
| **Lead with a working example, then explain** | Retail devs learn by doing, not reading |
| **Every code block must be copy-pasteable** | No pseudocode, no "..." abbreviations |
| **Show expected output after every step** | Users need to verify they're on track |
| **Define jargon on first use** | Don't assume ML/quant background |
| **Use callout boxes for "What could go wrong"** | Prevent common mistakes proactively |
| **Link to other docs instead of duplicating** | Keep docs maintainable |
| **Include a TL;DR at the top** | Busy devs want the summary first |
| **Use tables for reference data** | Scannable, not buried in prose |

---

## Estimated Effort

| Document | Estimated Length | Time |
|----------|-----------------|------|
| QUICKSTART.md | ~200 lines | ~2 hours |
| PIPELINE_CONFIG.md | ~300 lines | ~3 hours |
| CONTRIBUTING_SIGNALS.md | ~250 lines | ~3 hours |
| SIGNAL_CONTRACT.md | ~150 lines | ~1.5 hours |
| PRODUCERS.md | ~200 lines | ~2 hours |
| AGGREGATORS.md | ~150 lines | ~1.5 hours |
| BACKTESTING.md | ~150 lines | ~1.5 hours |
| TROUBLESHOOTING.md | ~150 lines | ~1.5 hours |
| ARCHITECTURE.md | ~200 lines | ~2 hours |
| **Total** | **~1,750 lines** | **~18 hours** |
