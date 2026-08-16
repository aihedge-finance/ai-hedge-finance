"""ahf-trade — live/paper trading entrypoint.

Usage (set params via environment variables — no kickstarter):
    # Paper trading (default)
    AHF_SYMBOL=BTCUSDT AHF_PIPELINE=configs/pipeline.json uv run ahf-trade

    # Live trading
    AHF_TRADING_MODE=LIVE AHF_SYMBOL=BTCUSDT uv run ahf-trade

    # Docker
    docker run --env-file .env ai-hedge-finance:latest ahf-trade

All configuration is via environment variables or .env file.
See .env.example for full list.
"""
from __future__ import annotations

import logging
import signal
import sys
import time

from ahf.adapters.exchange.dummy_adapter import DummyAdapter
from ahf.core.logging import setup_logging
from ahf.core.settings import AHFSettings
from ahf.domain.order_executor import OrderExecutor
from ahf.domain.position_tracker import PositionTracker
from ahf.domain.risk.drawdown_rule import MaxDrawdownRule
from ahf.domain.risk.kelly_rule import KellyRule
from ahf.domain.risk.risk_manager import RiskManager
from ahf.domain.risk.total_loss_rule import TotalLossRule
from ahf.domain.signal_processor import SignalProcessor
from ahf.domain.trade_orchestrator import TradeOrchestrator
from ahf.signals.pipeline_loader import load_pipeline
from ahf.signals.signal_audit import SignalAuditLog

logger = logging.getLogger(__name__)

_SHUTDOWN = False


def _handle_sigterm(signum, frame):
    global _SHUTDOWN
    logger.info("Received SIGTERM — graceful shutdown initiated")
    _SHUTDOWN = True


def main() -> int:
    """Main trading loop entrypoint. Returns exit code."""
    settings = AHFSettings()
    setup_logging(settings.log_level)

    logger.info(
        "AHF Trade starting",
        extra={
            "mode": settings.trading_mode,
            "symbol": settings.symbol,
            "pipeline": settings.pipeline_config,
        },
    )

    # Build runtime deps (no agent/env yet — stub producers until Phase 6)
    runtime_deps: dict = {}

    # Load pipeline
    producers, aggregator = load_pipeline(settings.pipeline_config, runtime_deps)
    logger.info(f"Loaded {len(producers)} producer(s) from {settings.pipeline_config}")

    # Exchange adapter — real Binance adapter in Phase 6, DummyAdapter for now
    from decimal import Decimal
    exchange = DummyAdapter(initial_balance=Decimal(str(settings.initial_capital)))

    # Domain stack
    risk_manager = (
        RiskManager()
        .add_rule(MaxDrawdownRule(settings.max_drawdown_pct))
        .add_rule(TotalLossRule(settings.max_loss_pct, settings.initial_capital))
        .add_rule(KellyRule(fraction=settings.kelly_fraction))
    )
    signal_processor = SignalProcessor(
        buy_threshold=settings.buy_threshold,
        sell_threshold=settings.sell_threshold,
        confidence_floor=settings.confidence_floor,
    )
    order_executor = OrderExecutor(exchange, settings.symbol)
    position_tracker = PositionTracker(exchange, settings.symbol, Decimal(str(settings.initial_capital)))
    audit_log = SignalAuditLog(settings.audit_log_path) if settings.audit_log_enabled else None

    orchestrator = TradeOrchestrator(
        producers=producers,
        aggregator=aggregator,
        signal_processor=signal_processor,
        risk_manager=risk_manager,
        order_executor=order_executor,
        position_tracker=position_tracker,
        min_valid_signals=settings.min_valid_signals,
        audit_log=audit_log,
    )

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    step = 0
    while not _SHUTDOWN:
        try:
            # In live/paper mode, market_data comes from exchange adapters.
            # For now, provide a minimal placeholder.
            market_data: dict = {}
            result = orchestrator.step(market_data)
            logger.info(
                "Step complete",
                extra={
                    "step": result.step,
                    "action": str(result.action),
                    "risk": result.risk_verdict.value,
                    "elapsed_ms": round(result.elapsed_ms, 2),
                },
            )
            step += 1
            time.sleep(settings.step_interval_seconds)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — shutting down")
            break
        except Exception as e:
            logger.error("Unexpected error in main loop", extra={"error": str(e)}, exc_info=True)
            if settings.halt_on_error:
                return 1
            time.sleep(5)

    logger.info(f"AHF Trade stopped after {step} steps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
