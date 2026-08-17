# Architectural Proposal: Parallel Signal Pipeline & Bounded Concurrency

- **Author:** Antigravity / Engineering Team
- **Date:** 2026-08-17
- **Target Component:** `ahf.domain.trade_orchestrator`, `ahf.signals`, `ahf.core.settings`
- **Status:** Proposed

---

## 1. Executive Summary

The current `ai-hedge-finance` signal collection loop executes all Layer 1 signal producers ([`SignalProducer`](file:///Users/jonathan/MyCode/AIHedge/ai-hedge-finance/src/ahf/signals/signal_producer.py)) sequentially within [`TradeOrchestrator.step()`](file:///Users/jonathan/MyCode/AIHedge/ai-hedge-finance/src/ahf/domain/trade_orchestrator.py). This creates **additive latency** ($\sum \text{latency}(P_i)$) and exposes the trading loop to **cascading timeout stalls** when remote endpoints (e.g. LLMs) fail or slow down.

This proposal details the design, concurrency models, and implementation roadmap for:
1. **Parallelizing Layer 1 Signal Producers** (fan-out) while maintaining the synchronous reduction barrier in Layer 2 ([`SignalAggregator`](file:///Users/jonathan/MyCode/AIHedge/ai-hedge-finance/src/ahf/signals/signal_aggregator.py)).
2. **Implementing Bounded Concurrency (`max_workers`)** to protect low-grade hardware nodes (1–2 vCPUs, 1–2 GB RAM) from CPU thrashing, memory spikes, and OS thread contention.
3. **Preserving Clean Domain Contracts** without forcing asynchronous runtime infection onto synchronous RL training environments (`gym.Env.step()`).

---

## 2. Problem Statement & Baseline Analysis

### 2.1 Current Sequential Execution Pattern

In [`TradeOrchestrator.step()`](file:///Users/jonathan/MyCode/AIHedge/ai-hedge-finance/src/ahf/domain/trade_orchestrator.py#L119-L127), producers are called in a blocking serial loop:

```python
# Current serial signal collection
raw_signals: list[Optional[SignalOutput]] = []
producer_pairs: list[tuple[str, Optional[SignalOutput]]] = []

for producer in self._producers:
    t0 = time.monotonic()
    sig, timed_out, error = produce_with_timeout(producer, market_data, ctx, 5.0)
    latency_ms = (time.monotonic() - t0) * 1000
    raw_signals.append(sig)
    producer_pairs.append((producer.name, sig))
    self._signal_logger.log_producer_result(producer.name, sig, latency_ms, error, timed_out)
```

Inside [`produce_with_timeout`](file:///Users/jonathan/MyCode/AIHedge/ai-hedge-finance/src/ahf/signals/timeout.py), each producer spins up a new `threading.Thread` and calls `thread.join(timeout=timeout_seconds)` synchronously.

### 2.2 Bottlenecks Identified

1. **Additive Wall-Clock Latency:**
   $$\text{Total L1 Latency} = \sum_{i=1}^{N} \text{latency}(P_i)$$
   * If a pipeline contains an indicator producer (5ms), an RL model (40ms), and an LLM model (2,000ms), the total step latency is $2,045\text{ms}$.
2. **Worst-Case Cascading Timeouts:**
   * If two network-dependent producers timeout (e.g., 5.0s each), the trading orchestrator blocks for **10.0 seconds** before progressing to risk evaluation and order execution.
3. **Thread Instantiation Overhead:**
   * Spawning and tearing down `threading.Thread` objects on every step for every producer incurs unnecessary OS overhead compared to maintaining a persistent worker pool.

---

## 3. High-Level Architecture: Fan-Out / Fan-In

```
                            ┌── market_data, context ──┐
                            │                          │
                   [ Fan-Out: Parallel Layer 1 ]       │
                  ┌─────────────┼─────────────┐        │
                  ▼             ▼             ▼        │
             [ LLM 1 ]      [ RL PPO ]    [ Tech TA ]  │ Layer 1: Signal Producers (Independent)
              (2.0s)          (40ms)        (5ms)      │ ➔ Bounded by max_workers pool
                  │             │             │        │ ➔ Max latency = 2.0s
                  └─────────────┼─────────────┘        │
                   [ Fan-In: Layer 2 Reduction ]       │
                                ▼                      │
                     [ SignalAggregator ]              │ Layer 2: Aggregator (Consensus)
                    (WeightedVote / MetaLLM)           │ ➔ Evaluates once L1 barrier completes
                                │                      │
                                ▼                      │
           [ SignalProcessor ] ➔ [ RiskGate ] ➔ [ OrderExecutor ]
```

* **Layer 1 (Signal Producers):** Independent, embarrassingly parallel tasks. Running in parallel reduces total latency to $\max_{i=1}^N \text{latency}(P_i)$.
* **Layer 2 (Signal Aggregator):** Acts as a barrier synchronization point (fan-in). Requires all valid or timed-out `SignalOutput` objects before executing consensus aggregation.

---

## 4. Hardware Evaluation: Low-Grade Nodes & Resource Constraints

When deploying trading nodes to cost-efficient, low-grade hardware (e.g., AWS `t3.micro`/`t3.small`, 1–2 vCPUs, 1–2 GB RAM, Raspberry Pi, or edge nodes), unbounded parallelism introduces severe stability risks:

### 4.1 Resource Failure Modes on Constrained Nodes

| Constraint | Unbounded Concurrency Risk | Mitigation via Bounded Concurrency |
| :--- | :--- | :--- |
| **1–2 vCPU Cores** | **Thread Contention & Thrashing:** PyTorch and NumPy/BLAS spawn multiple intra-op threads by default. Running multiple models concurrently causes context switching that degrades latency. | Cap `max_workers` to physical core count ($1 \le N \le 2$). Cap PyTorch intra-op threads (`torch.set_num_threads(1)`). |
| **1–2 GB RAM** | **Out-Of-Memory (OOM) Kills:** Multiple models or large DataFrame copies loaded concurrently trigger kernel OOM kills. | Caps peak in-flight memory by restricting simultaneously active producers. |
| **Burstable CPU Credits** | Bursting to 100% CPU on every tick burns CPU credits, triggering hypervisor throttling down to baseline (e.g. 10–20%). | Smooths CPU usage and prevents CPU credit exhaustion. |

### 4.2 Workload Asymmetry: I/O vs. CPU

* **Network I/O-bound (`LLMSignalProducer`):**
  * Spends $>99\%$ of wall-clock time waiting on network sockets.
  * Consumes negligible CPU ($\approx 0\%$) and RAM. A 1-core machine can easily run 5–10 concurrent HTTP calls.
* **CPU-bound (`RLSignalProducer`, `TechIndicatorProducer`):**
  * Performs dense matrix multiplications or rolling window computations over thousands of candles.
  * Must be strictly gated to prevent core saturation.

---

## 5. Concurrency Model Comparison

| Evaluation Metric | Option A: `ThreadPoolExecutor` (Recommended) | Option B: `asyncio` (`async`/`await`) | Option C: `ProcessPoolExecutor` |
| :--- | :--- | :--- | :--- |
| **Domain Purity** | **High:** Preserves existing synchronous `produce()` & `aggregate()` ABCs. | **Low:** Causes "async contagion"; requires `async def` across domain, adapters, and entrypoints. | **Medium:** Requires pickling all arguments and results across IPC. |
| **RL / Gym Compatibility** | **Native:** Works seamlessly inside synchronous `env.step()`. | **Complex:** Requires running nested event loops (`nest_asyncio` / `asyncio.run()`) inside Gym. | **Complex:** Cannot share in-memory PyTorch models or GPU tensors easily without `torch.multiprocessing`. |
| **Concurrency Control** | **Trivial:** Built-in `max_workers=N` parameter acts as a thread queue. | **Moderate:** Requires explicit `asyncio.Semaphore(N)`. | **Trivial:** Built-in `max_workers=N`. |
| **GIL Behavior** | Releases GIL during I/O (LLMs) and C extensions (PyTorch, NumPy, TA-Lib). | Single-threaded event loop; CPU-bound tasks block loop unless wrapped in `to_thread`. | Full process isolation; bypasses GIL entirely. |
| **Memory Overhead** | **Low:** Shares memory space and loaded models in a single process. | **Very Low:** Single process and thread. | **High:** Duplicates process memory and model weights for each worker. |

---

## 6. Implementation Plan

### 6.1 Configuration Updates

Add `max_producer_concurrency` to [`AHFSettings`](file:///Users/jonathan/MyCode/AIHedge/ai-hedge-finance/src/ahf/core/settings.py) and allow override via environment variable `AHF_MAX_PRODUCER_CONCURRENCY`:

```python
# src/ahf/core/settings.py
class AHFSettings(Settings):
    ...
    # Signal Pipeline Concurrency
    max_producer_concurrency: int = 2  # Default to 2 workers for low-grade hardware safety
```

### 6.2 Persistent Thread Pool in `TradeOrchestrator`

Refactor [`TradeOrchestrator`](file:///Users/jonathan/MyCode/AIHedge/ai-hedge-finance/src/ahf/domain/trade_orchestrator.py) to manage a reusable `ThreadPoolExecutor` and dispatch producers in parallel:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

class TradeOrchestrator:
    def __init__(
        self,
        producers: list[SignalProducer],
        aggregator: SignalAggregator,
        signal_processor: SignalProcessor,
        risk_manager: RiskManager,
        order_executor: OrderExecutor,
        position_tracker: PositionTracker,
        min_valid_signals: int = 1,
        max_concurrency: int = 2,
        audit_log: Optional[SignalAuditLog] = None,
    ) -> None:
        self._producers = producers
        self._aggregator = aggregator
        ...
        self._pool = ThreadPoolExecutor(
            max_workers=max_concurrency,
            thread_name_prefix="ahf-producer"
        )

    def _collect_signals(
        self,
        market_data: dict,
        ctx: dict,
        timeout_seconds: float = 5.0
    ) -> tuple[list[Optional[SignalOutput]], list[tuple[str, Optional[SignalOutput]]]]:
        # Map producer to future
        futures = {
            self._pool.submit(produce_with_timeout, producer, market_data, ctx, timeout_seconds): (idx, producer)
            for idx, producer in enumerate(self._producers)
        }

        raw_signals: list[Optional[SignalOutput]] = [None] * len(self._producers)
        producer_pairs: list[tuple[str, Optional[SignalOutput]]] = [(p.name, None) for p in self._producers]

        for future in as_completed(futures):
            idx, producer = futures[future]
            t0 = time.monotonic()
            sig, timed_out, error = future.result()
            latency_ms = (time.monotonic() - t0) * 1000

            raw_signals[idx] = sig
            producer_pairs[idx] = (producer.name, sig)
            self._signal_logger.log_producer_result(producer.name, sig, latency_ms, error, timed_out)

        return raw_signals, producer_pairs
```

### 6.3 PyTorch Optimization for Single/Dual Core Nodes

In the RL producer wrapper or node initialization:

```python
import torch

# Ensure PyTorch does not monopolize all CPU threads on constrained hardware
torch.set_num_threads(1)
```

---

## 7. Performance & Latency Projection

| Scenario (3 Producers: LLM 2.0s, RL 40ms, TA 5ms) | Sequential Baseline | Parallel (`max_workers=3`) | Bounded Low-End (`max_workers=2`) |
| :--- | :--- | :--- | :--- |
| **Normal Execution** | $\approx 2,045\text{ms}$ | $\approx 2,000\text{ms}$ ($-2.2\%$) | $\approx 2,000\text{ms}$ |
| **Scenario with 2 LLMs (2s each) + 1 RL (40ms)** | $\approx 4,040\text{ms}$ | $\approx 2,000\text{ms}$ (**$-50.5\%$**) | $\approx 2,040\text{ms}$ (**$-49.5\%$**) |
| **Two Timeouts (5.0s timeout each)** | $\ge 10.0\text{s}$ stall | $\le 5.0\text{s}$ stall (**$-50\%$**) | $\le 5.0\text{s}$ stall (**$-50\%$**) |
| **Peak CPU Load (1 vCPU node)** | Low (serialized) | High (thread contention) | **Controlled & stable** |

---

## 8. Verification & Rollout Plan

1. **Unit Tests:**
   * Verify output ordering is deterministic (results match original `producers` index regardless of completion order).
   * Verify partial timeouts and error isolation behave identically to the sequential baseline.
2. **Concurrency Limit Tests:**
   * Verify that with `max_concurrency=1`, behavior matches sequential execution.
   * Verify that with `max_concurrency=2`, active worker count never exceeds 2.
3. **Stress Benchmarks:**
   * Benchmark on a 1 vCPU / 1 GB RAM container with simulated 5s latency producers to verify memory stability and CPU credit health.
