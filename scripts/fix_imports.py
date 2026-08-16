#!/usr/bin/env python3
"""Fix old diewalkure import paths → new ahf.* namespace."""
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent / "src" / "ahf"

REPLACEMENTS = [
    # envs.*
    (r"from envs\.(BrunhildEnv_v11|BrunhildDatastore_v11|BaseEnv_v11|BrunhildReadWrite|DRLAgent|TradeEnum|plot_helper_v21|BacktestValueNetworkEnv|StockTradingEnv_v21|Datastore_v22|Datastore_v21)\b",
     r"from ahf.rl.envs.\1"),
    (r"from envs import\b", r"from ahf.rl.envs import"),
    # agents.*
    (r"from agents\.AgentPPO\b", r"from ahf.rl.agents.AgentPPO"),
    (r"from agents\.AgentPPO_H\b", r"from ahf.rl.agents.AgentPPO_H"),
    (r"from agents\.AgentPPO_035\b", r"from ahf.rl.agents.AgentPPO_035"),
    (r"from agents\.AgentBase\b", r"from ahf.rl.agents.AgentBase"),
    (r"from agents\.net\b", r"from ahf.rl.agents.net"),
    (r"from agents import\b", r"from ahf.rl.agents import"),
    # train.*
    (r"from train\.config\b", r"from ahf.rl.train.config"),
    (r"from train\.utils\b", r"from ahf.rl.train.utils"),
    (r"from train\.evaluator\b", r"from ahf.rl.train.evaluator"),
    (r"from train\.replay_buffer\b", r"from ahf.rl.train.replay_buffer"),
    (r"from train import\b", r"from ahf.rl.train import"),
    # TradingStrategy.*
    (r"from TradingStrategy\.utils\b", r"from ahf.rl.strategies.utils"),
    (r"from TradingStrategy import\b", r"from ahf.rl.strategies import"),
    # preprocessor.*
    (r"from preprocessor\.kf\.", r"from ahf.preprocessor.kf."),
    (r"from preprocessor\.ta\.", r"from ahf.preprocessor.ta."),
    (r"from preprocessor\.ukf\.", r"from ahf.preprocessor.ukf."),
    (r"from preprocessor\.finrl\.", r"from ahf.preprocessor.finrl."),
    (r"from preprocessor\.tail\.", r"from ahf.preprocessor.tail."),
    (r"from preprocessor\.helpers\b", r"from ahf.preprocessor.helpers"),
    (r"from preprocessor\.preprocessors\b", r"from ahf.preprocessor.preprocessors"),
    (r"from preprocessor import\b", r"from ahf.preprocessor import"),
    # app.* → ahf.*
    (r"from app\.utils\b", r"from ahf.utils.utils"),
    (r"from app\.logger\b", r"from ahf.utils.logger"),
    (r"from app\.helper\b", r"from ahf.utils.helper"),
    (r"from app\.enums\b", r"from ahf.core.enums"),
    (r"from app\.notification\b", r"from ahf.utils.notification"),
    (r"from app\.schema\b", r"from ahf.utils.schema"),
    (r"from app\.scipy_helper\b", r"from ahf.utils.scipy_helper"),
    (r"from app\.dataservice\b", r"from ahf.utils.dataservice"),
    (r"from app\.dask_division_helper\b", r"from ahf.utils.dask_division_helper"),
    (r"from app import utils\b", r"from ahf.utils import utils"),
    (r"from app import\b", r"from ahf.utils import"),
    # TradeBot.* — stub out (not ported in v2 yet)
    (r"from TradeBot\.core\.crud\.[^\n]+", r"# TODO(v2): port TradeBot DB CRUD — see ahf domain layer"),
    (r"from TradeBot\.core\.schema\.[^\n]+", r"# TODO(v2): port TradeBot schema — see ahf domain layer"),
]

SKIP = {"__pycache__", ".git"}


def fix_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text
    for pattern, repl in REPLACEMENTS:
        text = re.sub(pattern, repl, text)
    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    changed, skipped = 0, 0
    for d in [ROOT / "rl", ROOT / "utils", ROOT / "preprocessor"]:
        for f in d.rglob("*.py"):
            if any(s in f.parts for s in SKIP):
                continue
            if fix_file(f):
                changed += 1
                print(f"  Fixed: {f.relative_to(ROOT.parent.parent)}")
            else:
                skipped += 1
    print(f"\nDone — {changed} changed, {skipped} unchanged.")


if __name__ == "__main__":
    main()
