# Option A(b): Lightweight 2-Layer Hierarchical Signal Architecture

> **This document has been split into focused files for manageability.**  
> The original monolithic file was ~2,000+ lines. Each section now lives in its own file.

---

## Document Map

| # | Document | Lines | Content |
|---|----------|-------|---------|
| 1 | [Architecture](option_ab_architecture.md) | ~1,230 | Core design: philosophy, diagrams, ABCs, concrete implementations, JSON config schema, pipeline loader, wiring, **multi-timeframe signal composition**, comparison tables |
| 2 | [Signal Contract](option_ab_signal_contract.md) | ~200 | Contract enforcement philosophy: Pydantic validation, normalization guide, translation examples, metadata spec |
| 3 | [Production Readiness](option_ab_production_readiness.md) | ~750 | Phase 1/2/3 roadmap + all production implementations: timeout, health check, config validation, logging, audit log, replay producer, template, complete orchestrator |
| 4 | [Audience Analysis](option_ab_audience_analysis.md) | ~120 | Power user vs retail developer gap matrix, verdict, what each audience needs, effort estimates |
| 5 | [Documentation Plan](retail_developer_documentation_plan.md) | ~400 | 9 retail developer docs to generate: QUICKSTART, PIPELINE_CONFIG, CONTRIBUTING_SIGNALS, SIGNAL_CONTRACT, PRODUCERS, AGGREGATORS, BACKTESTING, TROUBLESHOOTING, ARCHITECTURE |

---

## Quick Reference

### Reading Order

- **New to A(b)?** Start with [Architecture](option_ab_architecture.md) → [Signal Contract](option_ab_signal_contract.md)
- **Implementing?** Read [Production Readiness](option_ab_production_readiness.md) for all code
- **Making a release decision?** Read [Audience Analysis](option_ab_audience_analysis.md)
- **Writing docs?** Read [Documentation Plan](retail_developer_documentation_plan.md)

### Key Decisions

| Decision | Answer | Where |
|----------|--------|-------|
| How many layers? | Exactly 2 (Producers → Aggregator) | [Architecture](option_ab_architecture.md) §Design Philosophy |
| How to control signal format? | Pydantic `SignalOutput` with `action ∈ [-1,1]`, `confidence ∈ [0,1]` | [Signal Contract](option_ab_signal_contract.md) |
| How to configure what runs? | `pipeline.json` — edit JSON, not Python | [Architecture](option_ab_architecture.md) §Pipeline Configuration |
| Can I use 3 LLMs only? | Yes — any N producers of any type | [Architecture](option_ab_architecture.md) §Example 2 |
| What happens if a producer fails? | Graceful degradation: treated as None, aggregator re-normalizes | [Architecture](option_ab_architecture.md) §Graceful Degradation |
| How do multi-timeframe signals work? | Orchestrator caches 1D/1H data; producers use `market_data.get("1d")` with staleness checks | [Architecture](option_ab_architecture.md) §Multi-Timeframe Signal Composition |
| Is it production ready? | Yes with Phase 1+2 implemented | [Production Readiness](option_ab_production_readiness.md) §Phase Roadmap |
| Who is it for? | Power users ✅ ready; Retail devs ✅ code ready, ⚠️ docs needed | [Audience Analysis](option_ab_audience_analysis.md) |
