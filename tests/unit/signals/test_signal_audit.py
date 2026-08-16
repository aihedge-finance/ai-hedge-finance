"""Tests for SignalAuditLog (Issue 12 - was 46% coverage)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ahf.signals.signal_audit import SignalAuditLog, _is_jsonable
from ahf.signals.signal_types import SignalOutput


@pytest.fixture()
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "test_audit.jsonl"


@pytest.fixture()
def buy_signal() -> SignalOutput:
    return SignalOutput(signal_name="rl_ppo", action=1.0, confidence=0.9)


@pytest.fixture()
def sell_signal() -> SignalOutput:
    return SignalOutput(signal_name="rl_ppo", action=-1.0, confidence=0.8)


def test_write_step_creates_file(log_path: Path, buy_signal: SignalOutput) -> None:
    with SignalAuditLog(str(log_path)) as audit:
        audit.write_step(0, [("rl_ppo", buy_signal)], buy_signal, {"cash": 1000.0})
    assert log_path.exists()


def test_write_step_valid_json(log_path: Path, buy_signal: SignalOutput) -> None:
    with SignalAuditLog(str(log_path)) as audit:
        audit.write_step(1, [("rl_ppo", buy_signal)], buy_signal, {})
    entry = json.loads(log_path.read_text().strip())
    assert entry["step"] == 1
    assert entry["aggregated"]["action"] == 1.0
    assert entry["aggregated"]["confidence"] == 0.9
    assert "ts" in entry


def test_write_step_appends_multiple(
    log_path: Path, buy_signal: SignalOutput, sell_signal: SignalOutput
) -> None:
    with SignalAuditLog(str(log_path)) as audit:
        audit.write_step(0, [("rl_ppo", buy_signal)], buy_signal, {})
        audit.write_step(1, [("rl_ppo", sell_signal)], sell_signal, {})
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["aggregated"]["action"] == -1.0


def test_write_step_failed_producer(log_path: Path, buy_signal: SignalOutput) -> None:
    with SignalAuditLog(str(log_path)) as audit:
        audit.write_step(0, [("rl_ppo", buy_signal), ("llm", None)], buy_signal, {})
    entry = json.loads(log_path.read_text().strip())
    producers = entry["producers"]
    assert producers[0]["failed"] is False
    assert producers[0]["action"] == 1.0
    assert producers[1]["failed"] is True
    assert producers[1]["action"] is None


def test_write_step_drops_non_jsonable_context(log_path: Path, buy_signal: SignalOutput) -> None:
    ctx = {"step": 1, "obj": object(), "valid": "hello"}
    with SignalAuditLog(str(log_path)) as audit:
        audit.write_step(0, [("rl_ppo", buy_signal)], buy_signal, ctx)
    entry = json.loads(log_path.read_text().strip())
    assert "valid" in entry["context"]
    assert "obj" not in entry["context"]


def test_write_step_context_included(log_path: Path, buy_signal: SignalOutput) -> None:
    ctx = {"cash": 1000.0, "drawdown": 0.02}
    with SignalAuditLog(str(log_path)) as audit:
        audit.write_step(0, [("rl_ppo", buy_signal)], buy_signal, ctx)
    entry = json.loads(log_path.read_text().strip())
    assert entry["context"]["cash"] == 1000.0
    assert entry["context"]["drawdown"] == pytest.approx(0.02)


def test_context_manager_closes_without_error(log_path: Path, buy_signal: SignalOutput) -> None:
    with SignalAuditLog(str(log_path)) as audit:
        audit.write_step(0, [("rl_ppo", buy_signal)], buy_signal, {})
    assert log_path.stat().st_size > 0


@pytest.mark.parametrize("val,expected", [
    (1, True),
    ("hello", True),
    (1.5, True),
    (None, True),
    ([1, 2], True),
    ({"a": 1}, True),
    (object(), False),
])
def test_is_jsonable(val: object, expected: bool) -> None:
    assert _is_jsonable(val) == expected
