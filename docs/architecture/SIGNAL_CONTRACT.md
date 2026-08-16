# Signal Contract Specification

The `SignalOutput` model is the fundamental data contract between Signal Producers (Layer 1), Aggregators (Layer 2), and the Signal Processor.

---

## 1. Schema Definition

```python
class SignalOutput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    signal_name: str                            # Non-empty string
    action: float                               # Continuous value in [-1.0, 1.0]
    confidence: float                           # Score in [0.0, 1.0]
    rationale: Optional[str] = None             # Optional human-readable explanation
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
```

---

## 2. Invariants & Validation Rules

1. **Action Boundary Clamping** (tolerance `1e-6`):
   - Values within `1e-6` of `±1.0` (e.g. `1.0000000000000002` from numpy) are clamped to `±1.0`.
   - Values far outside `[-1.0, 1.0]` (e.g. `2.0`, `-5.0`) raise a `ValidationError`.

2. **Confidence Boundary Clamping** (tolerance `1e-6`):
   - Same clamping tolerance as `action`.
   - Values outside `[0.0, 1.0]` beyond tolerance raise a `ValidationError`.

3. **HOLD Confidence Cap**:
   - When `action == 0.0`, `confidence` is automatically capped at `0.5` post-validation.
   - Prevents over-confident HOLD signals from inflating aggregator weight.

4. **Immutability**:
   - `SignalOutput` is `frozen=True`. Instances cannot be mutated after creation.

---

## 3. Directional Helpers

```python
signal.is_bullish      # True if action > 0.0
signal.is_bearish      # True if action < 0.0
signal.is_hold         # True if action == 0.0
signal.weighted_action # Returns action * confidence
```

---

## 4. Telemetry & Audit Serialization

`SignalOutput.to_audit_dict()` produces a flat JSON-serializable dictionary. Note: `rationale` is **not** included in the audit dict by default (kept inline in `metadata` if needed):

```json
{
  "signal_name": "rl_ppo",
  "action": 1.0,
  "confidence": 0.85,
  "timestamp": "2026-08-17T03:40:00.000000+00:00",
  "agent_type": "ppo",
  "raw_action": 1
}
```

The `metadata` dict is unpacked flat into the audit dict alongside the core fields.
