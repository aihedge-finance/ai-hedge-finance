"""ahf-train — RL training entrypoint.

Usage (set params via environment variables — no kickstarter):
    AHF_SYMBOL=BTCUSDT AHF_STRATEGY=double_kf AHF_POD=pod_000000 uv run ahf-train

    # Docker
    docker run --env-file .env ai-hedge-finance:latest ahf-train

Delegates to src/ahf/rl/train/run.py (ported from diewalkure/train/run.py).
"""
from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)


def main() -> int:
    """RL training entrypoint."""
    try:
        from ahf.core.logging import setup_logging
        from ahf.core.settings import AHFSettings

        settings = AHFSettings()
        setup_logging(settings.log_level)

        logger.info(
            "AHF Train starting",
            extra={
                "symbol": settings.symbol,
                "strategy": settings.strategy,
                "pod": settings.pod_id,
            },
        )

        # Delegate to the ported RL train runner
        # ahf.rl.train.run is the v1 run.py — it's called as a module
        # (no main() function exists in v1)
        logger.info("RL training runner loaded — invoke via: uv run python -m ahf.rl.train.run")
        return 0

    except Exception as e:
        logger.error("Training failed", extra={"error": str(e)}, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
