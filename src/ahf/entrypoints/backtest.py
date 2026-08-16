"""ahf-backtest — backtesting entrypoint.

Usage:
    AHF_SYMBOL=BTCUSDT AHF_STRATEGY=double_kf AHF_POD=pod_000000 uv run ahf-backtest

Runs the pipeline with ReplayProducer to replay signal_audit.jsonl.
"""
from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)


def main() -> int:
    """Backtesting entrypoint."""
    try:
        from ahf.core.logging import setup_logging
        from ahf.core.settings import AHFSettings

        settings = AHFSettings()
        setup_logging(settings.log_level)

        logger.info(
            "AHF Backtest starting",
            extra={"symbol": settings.symbol, "pipeline": settings.pipeline_config},
        )

        # Delegate to RL backtest (ported from diewalkure/RL_Backtest.py)
        # TODO(Phase 6): implement full backtest runner
        logger.info("Backtest runner not yet ported — use replay pipeline config for now")
        return 0

    except Exception as e:
        logger.error("Backtest failed", extra={"error": str(e)}, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
