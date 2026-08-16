"""End-to-end pipeline smoke test.

Loads a real pipeline config (from dict), runs 3 steps, verifies
that the aggregated signal comes back on every step.
"""

from ahf.signals.pipeline_loader import load_pipeline_from_dict
from ahf.signals.signal_types import SignalOutput
from ahf.signals.timeout import produce_with_timeout


def _make_market_data() -> dict:
    return {
        "5m": {"last_close": 67_000.0, "last_volume": 1_234.5, "df": None, "updated_at": None}
    }


def _make_context(step: int, producer_ids: list[str]) -> dict:
    return {"step": step, "producer_ids": producer_ids}


# ---------------------------------------------------------------------------
# Single RL stub pipeline (weighted_vote)
# ---------------------------------------------------------------------------


def test_e2e_single_rl_pipeline(dummy_pipeline_config_dict):
    producers, aggregator = load_pipeline_from_dict(dummy_pipeline_config_dict, {})
    assert len(producers) == 1

    market_data = _make_market_data()
    producer_ids = [p.name for p in producers]

    for step in range(3):
        signals = []
        for p in producers:
            sig, _, _ = produce_with_timeout(p, market_data, {}, timeout_seconds=5.0)
            signals.append(sig)
        result = aggregator.aggregate(signals, _make_context(step, producer_ids))
        assert isinstance(result, SignalOutput)
        assert -1.0 <= result.action <= 1.0
        assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Multi-producer pipeline (fixed_weight, 3 stubs)
# ---------------------------------------------------------------------------


def test_e2e_multi_producer_pipeline(multi_producer_config_dict):
    producers, aggregator = load_pipeline_from_dict(multi_producer_config_dict, {})
    assert len(producers) == 3

    market_data = _make_market_data()
    producer_ids = [p.name for p in producers]

    for step in range(3):
        signals = []
        for p in producers:
            sig, _, _ = produce_with_timeout(p, market_data, {}, timeout_seconds=5.0)
            signals.append(sig)
        result = aggregator.aggregate(signals, _make_context(step, producer_ids))
        assert isinstance(result, SignalOutput)


# ---------------------------------------------------------------------------
# Timeout resilience: one slow producer, min_valid_signals=2 not met
# ---------------------------------------------------------------------------


class _SlowProducer:
    name = "slow"

    def produce(self, market_data, context):
        import time
        time.sleep(10)
        from ahf.signals.signal_types import SignalOutput
        return SignalOutput(signal_name="slow", action=0.5, confidence=0.8)


def test_e2e_timeout_falls_to_valid_signals(dummy_pipeline_config_dict):
    """Even if a producer times out, the aggregator should return HOLD (not crash)."""
    from ahf.signals.aggregators.weighted_vote import WeightedVoteAggregator

    agg = WeightedVoteAggregator()
    slow = _SlowProducer()

    sig, timed_out, _ = produce_with_timeout(slow, {}, {}, timeout_seconds=0.1)
    assert timed_out is True
    assert sig is None

    result = agg.aggregate([sig], {})  # None → HOLD
    assert result.action == 0.0


# ---------------------------------------------------------------------------
# Load from pipeline.json file on disk
# ---------------------------------------------------------------------------


def test_e2e_load_pipeline_json_file():
    from ahf.signals.pipeline_loader import load_pipeline

    producers, aggregator = load_pipeline("configs/pipeline.json", {})
    assert len(producers) == 1
    assert aggregator is not None

    sig, _, _ = produce_with_timeout(producers[0], _make_market_data(), {}, timeout_seconds=5.0)
    result = aggregator.aggregate([sig], {"producer_ids": [producers[0].name]})
    assert isinstance(result, SignalOutput)
