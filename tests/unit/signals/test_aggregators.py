"""Tests for all 4 signal aggregators."""
import pytest

from ahf.signals.aggregators.fixed_weight import FixedWeightAggregator
from ahf.signals.aggregators.majority_vote import MajorityVoteAggregator
from ahf.signals.aggregators.meta_llm import MetaLLMAggregator
from ahf.signals.aggregators.weighted_vote import WeightedVoteAggregator
from ahf.signals.signal_types import SignalOutput


def _sig(name: str, action: float, confidence: float) -> SignalOutput:
    return SignalOutput(signal_name=name, action=action, confidence=confidence)


# ---------------------------------------------------------------------------
# WeightedVoteAggregator
# ---------------------------------------------------------------------------


class TestWeightedVote:
    def setup_method(self):
        self.agg = WeightedVoteAggregator()

    def test_three_signals_weighted_average(self):
        signals = [_sig("a", 0.8, 0.9), _sig("b", 0.4, 0.6), _sig("c", -0.2, 0.3)]
        result = self.agg.aggregate(signals, {})
        # Σ(action*conf) / Σ(conf) = (0.72 + 0.24 - 0.06) / 1.8 = 0.9/1.8 = 0.5
        assert abs(result.action - 0.5) < 1e-6

    def test_all_none_returns_hold(self):
        result = self.agg.aggregate([None, None], {})
        assert result.action == 0.0
        assert result.confidence <= 0.5

    def test_one_none_excluded(self):
        signals = [_sig("a", 1.0, 1.0), None]
        result = self.agg.aggregate(signals, {})
        assert result.action == 1.0

    def test_equal_confidence_signals(self):
        signals = [_sig("a", 0.6, 0.5), _sig("b", -0.6, 0.5)]
        result = self.agg.aggregate(signals, {})
        assert abs(result.action) < 1e-6  # Cancel out

    def test_from_config(self):
        agg = WeightedVoteAggregator.from_config({}, {})
        assert isinstance(agg, WeightedVoteAggregator)


# ---------------------------------------------------------------------------
# FixedWeightAggregator
# ---------------------------------------------------------------------------


class TestFixedWeight:
    def setup_method(self):
        self.weights = {"rl": 0.5, "tech": 0.3, "llm": 0.2}
        self.agg = FixedWeightAggregator(weights=self.weights)
        self.context = {"producer_ids": ["rl", "tech", "llm"]}

    def test_all_valid_weighted_sum(self):
        signals = [_sig("rl", 0.8, 1.0), _sig("tech", 0.4, 1.0), _sig("llm", -0.2, 1.0)]
        result = self.agg.aggregate(signals, self.context)
        # 0.5*0.8 + 0.3*0.4 + 0.2*(-0.2) = 0.40 + 0.12 - 0.04 = 0.48
        assert abs(result.action - 0.48) < 1e-6

    def test_missing_producer_renormalises(self):
        signals = [_sig("rl", 1.0, 1.0), None, _sig("llm", 0.0, 1.0)]
        result = self.agg.aggregate(signals, self.context)
        # rl weight=0.5, llm weight=0.2, total=0.7 → rl:5/7, llm:2/7
        expected = (0.5 / 0.7) * 1.0 + (0.2 / 0.7) * 0.0
        assert abs(result.action - expected) < 1e-6

    def test_all_none_returns_hold(self):
        result = self.agg.aggregate([None, None, None], self.context)
        assert result.action == 0.0

    def test_from_config_valid(self):
        agg = FixedWeightAggregator.from_config({"weights": {"a": 0.6, "b": 0.4}}, {})
        assert isinstance(agg, FixedWeightAggregator)

    def test_from_config_missing_weights_raises(self):
        with pytest.raises(ValueError, match="weights"):
            FixedWeightAggregator.from_config({}, {})


# ---------------------------------------------------------------------------
# MajorityVoteAggregator
# ---------------------------------------------------------------------------


class TestMajorityVote:
    def setup_method(self):
        self.agg = MajorityVoteAggregator()

    def test_two_buy_one_sell_is_buy(self):
        signals = [_sig("a", 0.8, 0.9), _sig("b", 0.6, 0.7), _sig("c", -0.5, 0.8)]
        result = self.agg.aggregate(signals, {})
        assert result.action > 0

    def test_two_sell_one_buy_is_sell(self):
        signals = [_sig("a", -0.8, 0.9), _sig("b", -0.6, 0.7), _sig("c", 0.5, 0.8)]
        result = self.agg.aggregate(signals, {})
        assert result.action < 0

    def test_tie_buy_sell_is_hold(self):
        signals = [_sig("a", 0.8, 0.9), _sig("b", -0.8, 0.9)]
        result = self.agg.aggregate(signals, {})
        assert result.action == 0.0

    def test_all_none_returns_hold(self):
        result = self.agg.aggregate([None, None], {})
        assert result.action == 0.0

    def test_from_config_custom_thresholds(self):
        agg = MajorityVoteAggregator.from_config({"buy_threshold": 0.1, "sell_threshold": -0.1}, {})
        assert agg._buy_threshold == 0.1


# ---------------------------------------------------------------------------
# MetaLLMAggregator
# ---------------------------------------------------------------------------


class _MockLLMClient:
    """Minimal LLM client mock."""
    def __init__(self, response: str):
        self._response = response

    def generate(self, prompt: str, **kwargs) -> str:
        return self._response


class TestMetaLLM:
    def test_valid_llm_response_parsed(self):
        client = _MockLLMClient('{"action": 0.7, "confidence": 0.85, "reasoning": "Bullish trend"}')
        agg = MetaLLMAggregator(client=client)
        signals = [_sig("a", 0.8, 0.9), _sig("b", 0.6, 0.7)]
        result = agg.aggregate(signals, {})
        assert result.signal_name == "meta_llm"
        assert abs(result.action - 0.7) < 1e-6
        assert abs(result.confidence - 0.85) < 1e-6
        assert result.metadata.get("reasoning") == "Bullish trend"

    def test_llm_api_failure_falls_back_to_weighted_vote(self):
        class _FailingClient:
            def generate(self, *args, **kwargs):
                raise RuntimeError("API timeout")

        agg = MetaLLMAggregator(client=_FailingClient())
        signals = [_sig("a", 1.0, 1.0), _sig("b", 1.0, 1.0)]
        result = agg.aggregate(signals, {})
        # Falls back — should still return a valid signal
        assert result is not None
        assert result.signal_name == "meta_llm_fallback"

    def test_no_client_falls_back_to_weighted_vote(self):
        agg = MetaLLMAggregator(client=None)
        signals = [_sig("a", 0.6, 0.8), _sig("b", 0.4, 0.6)]
        result = agg.aggregate(signals, {})
        assert result is not None
        assert result.action > 0  # weighted average of 0.6 and 0.4

    def test_all_none_with_no_client_returns_hold(self):
        agg = MetaLLMAggregator(client=None)
        result = agg.aggregate([None, None], {})
        assert result.action == 0.0

    def test_invalid_json_response_falls_back(self):
        client = _MockLLMClient("Sorry, I cannot provide a signal right now.")
        agg = MetaLLMAggregator(client=client)
        signals = [_sig("a", 0.5, 0.8)]
        result = agg.aggregate(signals, {})
        assert result is not None  # Fallback triggered

    def test_from_config_no_client(self):
        agg = MetaLLMAggregator.from_config({"model": "gemini-2.0-flash"}, {})
        assert agg._client is None  # No runtime_deps provided
