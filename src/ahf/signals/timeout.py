"""Threading-based hard timeout for signal producers.

Uses a daemon thread so a hung producer doesn't block the trading loop.
The thread is abandoned (not killed — Python can't kill threads), but
it is daemonised so it won't prevent process exit.

Design reference: design/pre_upgrade_v2_analysis/option_ab_production_readiness.md
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

from ahf.signals.signal_types import SignalOutput

logger = logging.getLogger(__name__)


def produce_with_timeout(
    producer,
    market_data: dict,
    context: dict,
    timeout_seconds: float,
) -> tuple[Optional[SignalOutput], bool, Optional[Exception]]:
    """Run producer.produce() with a hard timeout.

    Args:
        producer: Any SignalProducer instance.
        market_data: Passed through to produce().
        context: Passed through to produce().
        timeout_seconds: Wall-clock seconds before the producer is abandoned.

    Returns:
        (signal, timed_out, error):
          - signal: The SignalOutput, or None if timed out / errored.
          - timed_out: True if the timeout was hit.
          - error: The exception if one was raised, None otherwise.
    """
    result: list[Optional[SignalOutput]] = [None]
    error: list[Optional[Exception]] = [None]

    def _target() -> None:
        try:
            result[0] = producer.produce(market_data, context)
        except Exception as e:  # noqa: BLE001
            error[0] = e

    thread = threading.Thread(target=_target, daemon=True, name=f"producer-{producer.name}")
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        logger.warning(
            "Producer timed out",
            extra={"producer": producer.name, "timeout_seconds": timeout_seconds},
        )
        return None, True, None

    if error[0] is not None:
        logger.error(
            "Producer raised an exception",
            extra={"producer": producer.name, "error": str(error[0])},
        )
        return None, False, error[0]

    return result[0], False, None
