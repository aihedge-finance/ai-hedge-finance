# Signal Contract Specification

The `SignalOutput` model is the fundamental data contract between Signal Producers (Layer 1), Aggregators (Layer 2), and the Signal Processor.

---

## 1. Schema Definition

```python
class SignalOutput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    signal_name: str
    action: float            # Continuous value in [-1.0, 1.0]
    confidence: float        # Confidence score in [0.0, 1.0]
    rationale: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
```

---

## 2. Invariants & Validation Rules

1. **Action Boundaries & Clamping**:
   - `action` must lie in $[-1.0, 1.0]$.
   - Float precision deviations within $10^{-6}$ (e.g. `1.0000000000000002` from numpy operations) are automatically clamped to $[-1.0, 1.0]$.
   - Any value exceeding tolerance raises a `ValidationError`.

2. **Confidence Boundaries**:
   - `confidence` must lie in $[0.0, 1.0]$ (with $10^{-6}$ clamping).

3. **HOLD Invariant**:
   - If `action == 0.0` (HOLD), confidence is strictly capped at `0.50`. A producer cannot claim high confidence in "doing nothing".

4. **Immutability**:
   - `SignalOutput` is `frozen=True`. Instances cannot be mutated once produced.

---

## 3. Directional Helpers

The model provides convenience boolean properties:

```python
signal.is_bullish  # True if action > 0.0
signal.is_bearish  # True if action < 0.0
signal.is_hold     # True if action == 0.0
signal.weighted_action  # Returns action * confidence
```

---

## 4. Telemetry & Audit Serialization

`SignalOutput.to_audit_dict()` produces a JSON-serializable dictionary formatted for append-only audit logging:

```json
{
  "signal_name": "rl_ppo",
  "action": 1.0,
  "confidence": 0.85,
  "rationale": "Bullish momentum regime detected",
  "timestamp": "2026-08-17T03:40:00.000000Z",
  "metadata": {
    "agent_type": "ppo",
    "raw_action": 1
  }
}
```
