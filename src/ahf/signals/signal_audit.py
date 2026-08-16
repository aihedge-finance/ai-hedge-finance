"""Append-only JSONL audit trail for the signal pipeline.

Each trade step appends one JSON line to a file. This log is the source
of truth for post-trade analysis and replay backtesting (ReplayProducer).

Design reference: design/pre_upgrade_v2_analysis/option_ab_production_readiness.md
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ahf.signals.signal_types import SignalOutput


class SignalAuditLog:
    """Append-only JSONL audit log.

    Thread-safe for single-process use (no concurrent writers).
    File is opened in append mode and kept open for efficiency.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a", encoding="utf-8")

    def write_step(
        self,
        step: int,
        producer_signals: list[tuple[str, Optional[SignalOutput]]],
        aggregated: SignalOutput,
        context: dict[str, Any],
    ) -> None:
        """Write one complete pipeline step to the audit log.

        Args:
            step: Trading step index.
            producer_signals: List of (producer_id, signal_or_None) pairs.
            aggregated: The final aggregated signal.
            context: Runtime context dict (portfolio snapshot, etc.).
        """
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "producers": [
                {
                    "id": pid,
                    "action": sig.action if sig else None,
                    "confidence": sig.confidence if sig else None,
                    "failed": sig is None,
                }
                for pid, sig in producer_signals
            ],
            "aggregated": {
                "action": aggregated.action,
                "confidence": aggregated.confidence,
            },
            "context": {k: v for k, v in context.items() if _is_jsonable(v)},
        }
        self._file.write(json.dumps(entry, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Flush and close the audit log file."""
        self._file.flush()
        self._file.close()

    def __enter__(self) -> "SignalAuditLog":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _is_jsonable(val: Any) -> bool:
    """True if val can be JSON-serialised without custom encoder."""
    try:
        json.dumps(val)
        return True
    except (TypeError, ValueError):
        return False
