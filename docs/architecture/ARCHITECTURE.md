# Architecture Overview — Option A(b)

AI Hedge Finance (AHF) v2 is engineered around **Option A(b): Lightweight 2-Layer Hierarchical Signal Architecture**. It cleanly decouples signal generation, multi-signal aggregation, risk management, and order execution.

---

## 1. High-Level System Topology

```mermaid
flowchart TD
    subgraph Market ["Market Data Feeds"]
        M1["WebSocket / REST"] --> MD["Multi-Timeframe Market Data Dict"]
    end

    subgraph Layer1 ["Layer 1: Signal Producers (Isolated)"]
        MD --> P1["RLSignalProducer (PPO)"]
        MD --> P2["TechIndicatorProducer (Kalman/RSI)"]
        MD --> P3["LLMSignalProducer (Gemini/Claude)"]
        MD --> P4["RuleBasedProducer"]
    end

    subgraph Layer2 ["Layer 2: Aggregator"]
        P1 -- SignalOutput --> AGG["SignalAggregator<br/>(Weighted / Fixed / MetaLLM)"]
        P2 -- SignalOutput --> AGG
        P3 -- SignalOutput --> AGG
        P4 -- SignalOutput --> AGG
    end

    subgraph Domain ["Domain Core & Execution"]
        AGG -- Consolidated SignalOutput --> SP["SignalProcessor<br/>(Dead-zone + Confidence floor)"]
        SP -- Proposed TradeAction --> RG["RiskManager Gate<br/>(Drawdown, TotalLoss, Kelly)"]
        RG -- Approved / Sized Action --> OE["OrderExecutor"]
        OE --> EX["ExchangeAdapter (Binance / Dummy)"]
        EX --> PT["PositionTracker"]
        PT -. Portfolio State .-> RG
        AGG -. Audit Telemetry .-> AUDIT["SignalAuditLog (JSONL)"]
    end
```

---

## 2. Architectural Layers

### Layer 1: Signal Producers (`ahf.signals.producers`)
- **Isolation**: Each producer executes independently with threading timeout protection (`ahf.signals.timeout`).
- **Contract**: All producers return immutable `SignalOutput` objects with action $\in [-1.0, 1.0]$ and confidence $\in [0.0, 1.0]$.
- **Graceful Degradation**: If a model fails or times out, it produces a neutral `HOLD` signal without halting the engine.

### Layer 2: Signal Aggregator (`ahf.signals.aggregators`)
- Combines $N$ producer outputs into 1 consolidated `SignalOutput`.
- Supports dynamic confidence weighting (`WeightedVote`), static weighting (`FixedWeight`), democratic voting (`MajorityVote`), and LLM arbitration (`MetaLLM`).

### Domain Core (`ahf.domain`)
1. **`SignalProcessor`**: Maps continuous signal action $[-1.0, 1.0]$ to discrete `TradeAction` (`BUY`, `HOLD`, `SELL`) using dead-zone thresholds and confidence floors.
2. **`RiskManager`**: Chain of Responsibility applying deterministic risk guardrails. If any rule vetoes, the trade is cancelled. Rules can also downsize order size (`KellyRule`).
3. **`PositionTracker`**: Tracks balance, open positions, unrealized PnL, and high-water mark drawdown. Compatible with both live adapters and RL environment datastores.
4. **`OrderExecutor`**: Translates approved `TradeAction` into concrete exchange orders (`MARKET`, `LIMIT`, `STOP_LOSS`).
5. **`TradeOrchestrator`**: Master loop coordinating one complete step: market data ingest $\rightarrow$ signal generation $\rightarrow$ aggregation $\rightarrow$ risk gate $\rightarrow$ order execution $\rightarrow$ audit logging.

### Adapters (`ahf.adapters`)
- **`ExchangeAdapter` ABC**: Clean interface for exchange operations (`get_price`, `get_balance`, `place_order`, `cancel_order`).
- **`DummyAdapter`**: In-memory simulation with slippage, partial fills, and balance tracking.
- **`BinanceAdapter`**: Production REST/WebSocket adapter for Binance Spot and Futures.
