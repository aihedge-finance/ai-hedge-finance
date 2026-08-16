# Contributing Signals & Aggregators

AI Hedge Finance (AHF) v2 is built with a plugin-style signal architecture. Adding a new alpha source or aggregation algorithm requires writing a single class adhering to the `SignalProducer` or `SignalAggregator` interface.

---

## 1. Creating a Custom Signal Producer

To create a new signal producer:
1. Inherit from `SignalProducer` (`ahf.signals.signal_producer`).
2. Implement `name`, `produce()`, `health_check()`, and `from_config()`.
3. Register your producer in `ahf.signals.pipeline_loader.PRODUCER_REGISTRY`.

### Template: `MyCustomProducer`

```python
from __future__ import annotations
from typing import Any
from ahf.signals.signal_producer import SignalProducer
from ahf.signals.signal_types import SignalOutput

class MyCustomProducer(SignalProducer):
    def __init__(self, name: str, threshold: float = 0.02) -> None:
        self._name = name
        self._threshold = threshold

    @property
    def name(self) -> str:
        return self._name

    def produce(self, market_data: dict, context: dict) -> SignalOutput:
        """Compute alpha signal from latest market data."""
        # 1. Extract required feature
        bar = market_data.get("5m", {})
        price_change = bar.get("pct_change", 0.0)

        # 2. Compute direction and action in [-1.0, 1.0]
        if price_change > self._threshold:
            action = 1.0
            confidence = min(1.0, price_change / 0.05)
        elif price_change < -self._threshold:
            action = -1.0
            confidence = min(1.0, abs(price_change) / 0.05)
        else:
            action = 0.0
            confidence = 0.5

        # 3. Return validated SignalOutput
        return SignalOutput(
            signal_name=self._name,
            action=action,
            confidence=confidence,
            metadata={"pct_change": price_change},
        )

    def health_check(self) -> None:
        """Verify data feeds and connections are healthy."""
        pass

    @classmethod
    def from_config(cls, name: str, config: dict, runtime_deps: dict) -> "MyCustomProducer":
        return cls(name=name, threshold=float(config.get("threshold", 0.02)))
```

### Signal Output Rules
- `action` must be a `float` in $[-1.0, 1.0]$. Values $>0$ indicate long, $<0$ indicate short, $0$ indicates hold. (Small floating-point deviations within $10^{-6}$ are automatically clamped).
- `confidence` must be a `float` in $[0.0, 1.0]$.
- When `action == 0.0` (HOLD), `confidence` is capped at $0.5$.

---

## 2. Creating a Custom Aggregator

To create a new aggregator:
1. Inherit from `SignalAggregator` (`ahf.signals.signal_aggregator`).
2. Implement `aggregate()`.
3. Register your aggregator in `ahf.signals.pipeline_loader.AGGREGATOR_REGISTRY`.

### Template: `MyCustomAggregator`

```python
from __future__ import annotations
from ahf.signals.signal_aggregator import SignalAggregator
from ahf.signals.signal_types import SignalOutput

class MyCustomAggregator(SignalAggregator):
    def aggregate(self, signals: list[SignalOutput]) -> SignalOutput:
        if not signals:
            return SignalOutput(signal_name="aggregator", action=0.0, confidence=0.0)

        # Filter out low-confidence signals
        valid = [s for s in signals if s.confidence >= 0.4 and not s.is_hold]
        if not valid:
            return SignalOutput(signal_name="aggregator", action=0.0, confidence=0.5)

        # Average actions of high-confidence signals
        avg_action = sum(s.action for s in valid) / len(valid)
        avg_conf = sum(s.confidence for s in valid) / len(valid)

        return SignalOutput(
            signal_name="my_aggregator",
            action=avg_action,
            confidence=avg_conf,
            metadata={"num_active_signals": len(valid)},
        )
```

---

## 3. Registering Your Components

In `src/ahf/signals/pipeline_loader.py`:

```python
from my_module import MyCustomProducer, MyCustomAggregator

PRODUCER_REGISTRY["my_producer"] = MyCustomProducer
AGGREGATOR_REGISTRY["my_aggregator"] = MyCustomAggregator
```

You can now use `"type": "my_producer"` directly in any `configs/pipeline.json`!

---

## 4. Writing Unit Tests

Every producer and aggregator must be tested for edge cases:
- Extreme values ($+1.0, -1.0, 0.0$).
- Empty market data / missing keys.
- Timeout behavior.
- Invariant guarantees (`SignalOutput` immutability and clamping).

Run your tests using:
```bash
uv run pytest tests/unit/signals/
```
