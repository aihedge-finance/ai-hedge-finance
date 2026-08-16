"""ReplayProducer — replays signals from a JSONL audit log.

Used for backtesting: reads pre-recorded signal entries from a
signal_audit.jsonl file and replays them step-by-step.

This enables testing the aggregator + domain logic against historical
signal data without re-running the original producers.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

from ahf.signals.signal_producer import SignalProducer
from ahf.signals.signal_types import SignalOutput

logger = logging.getLogger(__name__)


class ReplayProducer(SignalProducer):
    """Replays signals from a JSONL audit log, one entry per produce() call.

    Config in pipeline.json:
        {
            "id": "replay_source",
            "type": "replay",
            "config": {
                "audit_log_path": "data/logs/signal_audit.jsonl",
                "producer_id":    "rl_ppo"   // which producer's entries to replay
            }
        }
    """

    def __init__(self, name: str, audit_log_path: str, producer_id: str) -> None:
        self._name = name
        self._audit_log_path = Path(audit_log_path)
        self._producer_id = producer_id
        self._iter: Iterator[dict] | None = None

    @property
    def name(self) -> str:
        return self._name

    def _get_iterator(self) -> Iterator[dict]:
        """Lazily open and iterate over matching audit log entries."""
        with self._audit_log_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    # Find the matching producer entry
                    for p in entry.get("producers", []):
                        if p["id"] == self._producer_id and p["action"] is not None:
                            yield p
                except json.JSONDecodeError:
                    continue

    def produce(self, market_data: dict, context: dict) -> SignalOutput:
        if self._iter is None:
            self._iter = self._get_iterator()

        try:
            entry = next(self._iter)
            return SignalOutput(
                signal_name=self.name,
                action=float(entry["action"]),
                confidence=float(entry.get("confidence", 0.5)),
                metadata={"replayed_from": self._producer_id},
            )
        except StopIteration:
            logger.warning(
                "ReplayProducer exhausted audit log — returning HOLD",
                extra={"producer": self.name, "path": str(self._audit_log_path)},
            )
            return SignalOutput(signal_name=self.name, action=0.0, confidence=0.0)

    def health_check(self) -> None:
        if not self._audit_log_path.exists():
            raise RuntimeError(
                f"[{self.name}] Audit log not found: {self._audit_log_path.resolve()}"
            )

    @classmethod
    def from_config(cls, name: str, config: dict, runtime_deps: dict) -> "ReplayProducer":
        return cls(
            name=name,
            audit_log_path=config["audit_log_path"],
            producer_id=config.get("producer_id", name),
        )
