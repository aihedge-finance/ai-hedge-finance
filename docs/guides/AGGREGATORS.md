# Signal Aggregators Reference

Signal Aggregators represent **Layer 2** of the Option A(b) signal pipeline. They combine multiple `SignalOutput` objects into a single consolidated decision.

---

## 1. `WeightedVoteAggregator` (`type: "weighted_vote"`)

Calculates a confidence-weighted average action across all valid non-holding signals.

### Formula:
$$W_i = w_i^{config} \times \text{confidence}_i$$
$$\text{Action}_{agg} = \frac{\sum W_i \cdot \text{Action}_i}{\sum W_i}$$
$$\text{Confidence}_{agg} = \frac{\sum W_i \cdot \text{confidence}_i}{\sum W_i}$$

---

## 2. `FixedWeightAggregator` (`type: "fixed_weight"`)

Uses static weights configured in `pipeline.json`, independent of producer confidence. Ideal for benchmark pipelines with known optimal weights.

```json
"aggregator": {
  "type": "fixed_weight",
  "params": {
    "normalize_weights": true
  }
}
```

---

## 3. `MajorityVoteAggregator` (`type: "majority_vote"`)

Directional democratic voting. Quantizes each producer's action into BUY ($>0$), HOLD ($0$), or SELL ($<0$).

- If BUY weight $>50\%$ total valid weight $\rightarrow$ BUY.
- If SELL weight $>50\%$ total valid weight $\rightarrow$ SELL.
- Otherwise $\rightarrow$ HOLD.

---

## 4. `MetaLLMAggregator` (`type: "meta_llm"`)

Uses an LLM as a meta-decision arbiter. When conflicting signals occur (e.g. RL says BUY but Macro LLM says SELL due to CPI release), the MetaLLM evaluates individual rationale strings and issues a synthesis decision.

```json
"aggregator": {
  "type": "meta_llm",
  "params": {
    "model": "claude-3-5-sonnet",
    "temperature": 0.0
  }
}
```
