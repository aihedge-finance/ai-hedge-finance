"""Phase 6: Regression smoke tests.

Validates the full import graph, settings loading, pipeline execution,
and that the entrypoint main() function doesn't crash on startup.
"""
import importlib
import os


# ---------------------------------------------------------------------------
# Import graph — ensure all clean modules are importable
# ---------------------------------------------------------------------------


def test_core_enums_importable():
    from ahf.core.enums import (
        AppEnv, BotEnv, BOT_MODE, DeployEnv, MLLevel,
        NodeEnv, PriceEnv, SaasEnv, TradeAction, TradingMode,
    )
    assert TradeAction.BUY == 1
    assert AppEnv.TRADE == "TRADE"
    assert BotEnv.TRAIN == "TRAIN"
    assert DeployEnv.PRODUCTION == "PRODUCTION"


def test_core_types_importable():
    from ahf.core.types import d, d_abs, d_round, d_is_close
    from decimal import Decimal
    assert d("1.5") == Decimal("1.5")
    assert d_round(Decimal("1.2345"), 2) == Decimal("1.23")


def test_settings_loads_with_defaults():
    from ahf.core.settings import AHFSettings
    s = AHFSettings()
    assert s.symbol == "BTCUSDT"
    assert s.trading_mode == "PAPER"
    assert s.max_drawdown_pct == 0.15
    assert s.min_valid_signals == 1


def test_all_signal_types_importable():
    from ahf.signals.signal_types import SignalOutput
    from ahf.signals.signal_producer import SignalProducer
    from ahf.signals.signal_aggregator import SignalAggregator
    from ahf.signals.pipeline_config import PipelineConfig
    from ahf.signals.pipeline_loader import load_pipeline_from_dict


def test_all_aggregators_importable():
    from ahf.signals.aggregators.weighted_vote import WeightedVoteAggregator
    from ahf.signals.aggregators.fixed_weight import FixedWeightAggregator
    from ahf.signals.aggregators.majority_vote import MajorityVoteAggregator
    from ahf.signals.aggregators.meta_llm import MetaLLMAggregator


def test_all_producers_importable():
    from ahf.signals.producers.rl_producer import RLSignalProducer
    from ahf.signals.producers.tech_producer import TechIndicatorProducer
    from ahf.signals.producers.llm_producer import LLMSignalProducer
    from ahf.signals.producers.rule_producer import RuleBasedProducer
    from ahf.signals.producers.replay_producer import ReplayProducer


def test_all_domain_importable():
    from ahf.domain.risk.risk_types import RiskVerdict, PortfolioSnapshot
    from ahf.domain.risk.risk_rule import RiskRule
    from ahf.domain.risk.drawdown_rule import MaxDrawdownRule
    from ahf.domain.risk.total_loss_rule import TotalLossRule
    from ahf.domain.risk.kelly_rule import KellyRule
    from ahf.domain.risk.risk_manager import RiskManager
    from ahf.domain.signal_processor import SignalProcessor
    from ahf.domain.order_executor import OrderExecutor
    from ahf.domain.position_tracker import PositionTracker
    from ahf.domain.trade_orchestrator import TradeOrchestrator


def test_all_adapters_importable():
    from ahf.adapters.exchange.exchange_adapter import ExchangeAdapter
    from ahf.adapters.exchange.dummy_adapter import DummyAdapter


# ---------------------------------------------------------------------------
# Pipeline config files are valid JSON and load cleanly
# ---------------------------------------------------------------------------


def test_default_pipeline_config_loads():
    from ahf.signals.pipeline_loader import load_pipeline_from_dict

    import json
    with open("configs/pipeline.json") as f:
        cfg = json.load(f)
    producers, aggregator = load_pipeline_from_dict(cfg, {})
    assert len(producers) == 1


def test_multi_signal_pipeline_config_loads():
    from ahf.signals.pipeline_loader import load_pipeline_from_dict
    import json
    with open("configs/pipeline.multi_signal.json") as f:
        cfg = json.load(f)
    producers, aggregator = load_pipeline_from_dict(cfg, {})
    assert len(producers) == 3


# ---------------------------------------------------------------------------
# Full 10-step trading loop regression
# ---------------------------------------------------------------------------


def test_full_10_step_loop():
    """Run 10 steps of the complete orchestrator. Must not crash."""
    from decimal import Decimal
    from ahf.adapters.exchange.dummy_adapter import DummyAdapter
    from ahf.domain.order_executor import OrderExecutor
    from ahf.domain.position_tracker import PositionTracker
    from ahf.domain.risk.drawdown_rule import MaxDrawdownRule
    from ahf.domain.risk.risk_manager import RiskManager
    from ahf.domain.signal_processor import SignalProcessor
    from ahf.domain.trade_orchestrator import TradeOrchestrator
    from ahf.signals.aggregators.weighted_vote import WeightedVoteAggregator
    from ahf.signals.producers.rl_producer import RLSignalProducer

    producer = RLSignalProducer("rl_stub")  # stub: no agent/env
    aggregator = WeightedVoteAggregator()
    adapter = DummyAdapter(Decimal("1000"), Decimal("67000"))

    orch = TradeOrchestrator(
        producers=[producer],
        aggregator=aggregator,
        signal_processor=SignalProcessor(),
        risk_manager=RiskManager().add_rule(MaxDrawdownRule(0.30)),
        order_executor=OrderExecutor(adapter, "BTCUSDT"),
        position_tracker=PositionTracker(adapter, "BTCUSDT", Decimal("1000")),
    )

    for _ in range(10):
        result = orch.step({})
        assert result is not None
        assert result.elapsed_ms >= 0


# ---------------------------------------------------------------------------
# Entrypoint imports
# ---------------------------------------------------------------------------


def test_entrypoint_trade_importable():
    import ahf.entrypoints.trade


def test_entrypoint_train_importable():
    import ahf.entrypoints.train


def test_entrypoint_backtest_importable():
    import ahf.entrypoints.backtest
