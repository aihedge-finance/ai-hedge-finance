"""Tests for all signal producers (Phase 2 stubs + real producers)."""
import json
from pathlib import Path

import pytest

from ahf.signals.producers.llm_producer import LLMSignalProducer
from ahf.signals.producers.replay_producer import ReplayProducer
from ahf.signals.producers.rl_producer import RLSignalProducer
from ahf.signals.producers.rule_producer import RuleBasedProducer
from ahf.signals.producers.tech_producer import TechIndicatorProducer
from ahf.signals.signal_types import SignalOutput


class _MockLLMClient:
    def generate(self, prompt: str, **kwargs) -> str:
        return '{"action": 0.6, "confidence": 0.8, "reasoning": "Bullish"}'


# ---------------------------------------------------------------------------
# RLSignalProducer (stub)
# ---------------------------------------------------------------------------


class TestRLProducer:
    def test_returns_signal(self):
        p = RLSignalProducer.from_config("rl", {"model_path": "", "agent": "ppo"}, {})
        sig = p.produce({}, {})
        assert isinstance(sig, SignalOutput)
        assert sig.signal_name == "rl"

    def test_name_matches_id(self):
        p = RLSignalProducer.from_config("rl_ppo", {}, {})
        assert p.name == "rl_ppo"

    def test_health_check_passes(self):
        p = RLSignalProducer("test", "")
        p.health_check()  # Should not raise


# ---------------------------------------------------------------------------
# TechIndicatorProducer (stub)
# ---------------------------------------------------------------------------


class TestTechProducer:
    def test_returns_signal(self):
        p = TechIndicatorProducer.from_config("tech_kf", {"strategy": "double_kf"}, {})
        sig = p.produce({}, {})
        assert isinstance(sig, SignalOutput)
        assert sig.signal_name == "tech_kf"


# ---------------------------------------------------------------------------
# LLMSignalProducer
# ---------------------------------------------------------------------------


class TestLLMProducer:
    def test_valid_llm_response(self):
        p = LLMSignalProducer("llm_gemini", client=_MockLLMClient(), model="gemini-2.0-flash")
        sig = p.produce({"5m": {"last_close": 67000.0}}, {})
        assert abs(sig.action - 0.6) < 1e-6
        assert abs(sig.confidence - 0.8) < 1e-6

    def test_no_client_returns_hold_not_raises(self):
        p = LLMSignalProducer("llm_no_client", client=None)
        sig = p.produce({}, {})
        assert sig.action == 0.0
        assert sig.confidence == 0.0

    def test_api_failure_returns_hold_not_raises(self):
        class _FailClient:
            def generate(self, *args, **kwargs):
                raise RuntimeError("Network error")

        p = LLMSignalProducer("llm_fail", client=_FailClient())
        sig = p.produce({}, {})
        assert sig.action == 0.0  # Graceful HOLD

    def test_health_check_raises_without_client(self):
        p = LLMSignalProducer("llm_no_client", client=None)
        with pytest.raises(RuntimeError, match="LLM client not configured"):
            p.health_check()

    def test_health_check_passes_with_client(self):
        p = LLMSignalProducer("llm_ok", client=_MockLLMClient())
        p.health_check()  # Should not raise

    def test_from_config_wires_client(self):
        client = _MockLLMClient()
        p = LLMSignalProducer.from_config(
            "llm_g",
            {"model": "gemini-2.0-flash"},
            {"gemini-2.0-flash": client},
        )
        assert p._client is client


# ---------------------------------------------------------------------------
# RuleBasedProducer
# ---------------------------------------------------------------------------


class TestRuleProducer:
    def test_no_rules_returns_hold(self):
        p = RuleBasedProducer("rules")
        sig = p.produce({}, {})
        assert sig.action == 0.0
        assert sig.confidence == 0.0

    def test_single_buy_rule(self):
        p = RuleBasedProducer("rules")
        p.add_rule("always_buy", lambda md, ctx: 1.0)
        sig = p.produce({}, {})
        assert sig.action == 1.0
        assert sig.confidence == 1.0

    def test_conflicting_rules_averaged(self):
        p = RuleBasedProducer("rules")
        p.add_rule("buy", lambda md, ctx: 1.0)
        p.add_rule("sell", lambda md, ctx: -1.0)
        sig = p.produce({}, {})
        assert abs(sig.action) < 1e-6  # Average of +1 and -1

    def test_failed_rule_skipped(self):
        p = RuleBasedProducer("rules")
        p.add_rule("good", lambda md, ctx: 0.8)
        p.add_rule("bad", lambda md, ctx: 1 / 0)  # Will raise
        sig = p.produce({}, {})
        # Only 'good' contributes
        assert abs(sig.action - 0.8) < 1e-6

    def test_confidence_reflects_agreement(self):
        """3 rules: two buys (0.5, 0.3), one sell (-0.8).
        action = (0.5+0.3-0.8)/3 = 0.0 exactly — averaged to HOLD.
        Since action=0.0, SignalOutput validator caps confidence at 0.5.
        The producer computes agreement as 'fraction of rules == exact 0',
        which is 0/3=0.0, then the HOLD cap makes no further change since
        the input confidence is already 0.0.
        This tests that the aggregated-to-HOLD path doesn't crash.
        """
        p = RuleBasedProducer("rules")
        p.add_rule("r1", lambda md, ctx: 0.5)
        p.add_rule("r2", lambda md, ctx: 0.3)
        p.add_rule("r3", lambda md, ctx: -0.8)
        sig = p.produce({}, {})
        # action cancels to 0.0; confidence is 0 (no rules == 0 exactly)
        assert abs(sig.action) < 1e-9
        assert 0.0 <= sig.confidence <= 0.5  # HOLD cap applies

    def test_confidence_reflects_buy_agreement(self):
        """2 buy rules, 1 sell → action > 0 → confidence = 2/3."""
        p = RuleBasedProducer("rules")
        p.add_rule("r1", lambda md, ctx: 0.5)
        p.add_rule("r2", lambda md, ctx: 0.8)
        p.add_rule("r3", lambda md, ctx: -0.4)
        sig = p.produce({}, {})
        # action = (0.5+0.8-0.4)/3 = 0.3 > 0 → buy direction
        # 2 of 3 rules agree with buy → confidence = 2/3
        assert sig.action > 0
        assert abs(sig.confidence - 2 / 3) < 1e-6

    def test_from_config(self):
        p = RuleBasedProducer.from_config("rules", {}, {})
        assert p.name == "rules"


# ---------------------------------------------------------------------------
# ReplayProducer
# ---------------------------------------------------------------------------


class TestReplayProducer:
    def _make_audit_log(self, tmp_path: Path, entries: list[dict]) -> Path:
        log_path = tmp_path / "signal_audit.jsonl"
        with log_path.open("w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        return log_path

    def test_replay_returns_correct_signal(self, tmp_path):
        log_path = self._make_audit_log(
            tmp_path,
            [
                {
                    "step": 0,
                    "producers": [{"id": "rl_ppo", "action": 0.7, "confidence": 0.85, "failed": False}],
                    "aggregated": {"action": 0.7, "confidence": 0.85},
                }
            ],
        )
        p = ReplayProducer("replay", str(log_path), "rl_ppo")
        sig = p.produce({}, {})
        assert abs(sig.action - 0.7) < 1e-6
        assert abs(sig.confidence - 0.85) < 1e-6

    def test_replay_exhausted_returns_hold(self, tmp_path):
        log_path = self._make_audit_log(
            tmp_path,
            [
                {
                    "step": 0,
                    "producers": [{"id": "rl_ppo", "action": 0.5, "confidence": 0.8, "failed": False}],
                    "aggregated": {},
                }
            ],
        )
        p = ReplayProducer("replay", str(log_path), "rl_ppo")
        p.produce({}, {})  # Consume entry 1
        sig = p.produce({}, {})  # Exhausted
        assert sig.action == 0.0

    def test_health_check_missing_file_raises(self):
        p = ReplayProducer("replay", "/nonexistent/signal_audit.jsonl", "rl_ppo")
        with pytest.raises(RuntimeError, match="Audit log not found"):
            p.health_check()

    def test_health_check_existing_file_ok(self, tmp_path):
        log_path = self._make_audit_log(tmp_path, [])
        p = ReplayProducer("replay", str(log_path), "rl_ppo")
        p.health_check()  # Should not raise

    def test_from_config(self, tmp_path):
        log_path = self._make_audit_log(tmp_path, [])
        p = ReplayProducer.from_config(
            "replay",
            {"audit_log_path": str(log_path), "producer_id": "rl_ppo"},
            {},
        )
        assert p.name == "replay"
        assert p._producer_id == "rl_ppo"
