import sys

import torch
import importlib
import numpy as np
from ahf.rl.train.evaluator import Evaluator
from ahf.rl.train.replay_buffer import ReplayBuffer, ReplayBufferList


def load_read_write(env_name, logger):
    """這個被換到 strategy 了，請調整"""

    if env_name == "StockTradingEnv-v2":
        import envs.StockTradingEnvReadWrite_v21 as read_write
    elif env_name == "StockTradingEnv-v21":
        import envs.StockTradingEnvReadWrite_v21 as read_write
    elif env_name == "BrunhildEnv-v11":
        import envs.BrunhildReadWrite as read_write
    elif env_name == "GondulEnv-v1":
        import envs.GondulEnv_v1 as read_write
    else:
        logger.error(f"Unknown env_name:{env_name}")
        sys.exit(-1)

    return read_write


def load_env(env_name, logger):
    if env_name == "StockTradingEnv-v2":
        from ahf.rl.envs.StockTradingEnv_v2 import StockTradingEnv, check_env
    elif env_name == "StockTradingEnv-v21":
        from ahf.rl.envs.StockTradingEnv_v21 import StockTradingEnv, check_env
    elif env_name == "BrunhildEnv-v11":
        from ahf.rl.envs.BrunhildEnv_v11 import BrunhildEnv as StockTradingEnv
        from ahf.rl.envs.BrunhildEnv_v11 import check_env
    elif env_name == "GondulEnv-v1":
        from ahf.rl.envs.GondulEnv_v1 import GondulEnv as StockTradingEnv
        from ahf.rl.envs.GondulEnv_v1 import check_env
    else:
        logger.error(f"Unknown env_name:{env_name}")
        sys.exit(-1)

    return StockTradingEnv, check_env


def load_trade_arg_setup(tech_id: str, logger):
    if tech_id == "double_kf":
        from ahf.rl.strategies.double_kf.Strategy import Trade_Args_Setup
    elif tech_id.lower() == "rsi_macd":
        from ahf.rl.strategies.rsi_macd.Strategy import Trade_Args_Setup
    else:
        logger.error(f"load_trading_strategy Unknown tech_id: {tech_id}")
        sys.exit(-1)

    return Trade_Args_Setup


def load_trading_strategy(tech_id, logger):
    if tech_id == "double_kf":
        from ahf.rl.strategies.double_kf.Strategy import Strategy
    elif tech_id.lower() == "rsi_macd":
        from ahf.rl.strategies.rsi_macd.Strategy import Strategy
    else:
        logger.error(f"load_trading_strategy Unknown tech_id: {tech_id}")
        sys.exit(-1)

    return Strategy


def load_strategy_plotter(tech_id: str):
    if tech_id == "double_kf":
        from ahf.rl.strategies.double_kf.Strategy import plot_sim
    elif tech_id == "double_ukf":
        from ahf.rl.strategies.double_ukf.Strategy import plot_sim
    elif tech_id.lower() == "rsi_macd":
        from ahf.rl.strategies.rsi_macd.Strategy import plot_sim
    else:
        print(f"load_strategy_plotter failed: Unknown tech_id:'{tech_id}'")
        sys.exit(-1)

    return plot_sim


def init_agent(args, gpu_id: int, env=None):
    agent = args.agent(args.net_dim, args.state_dim, args.action_dim, gpu_id=gpu_id, args=args)
    agent.save_or_load_agent(args.cwd, if_save=False)

    if env is not None:
        """assign `agent.states` for exploration"""
        if args.env_num == 1:
            states = [env.reset(), ]
            assert isinstance(states[0], np.ndarray) or isinstance(states[0], torch.Tensor)
            assert states[0].shape in {(args.state_dim,), args.state_dim}, \
                f"expect in {(args.state_dim,)} or {args.state_dim} got states[0].shape: {states[0].shape}"
        else:
            states = env.reset()
            assert isinstance(states, torch.Tensor)
            assert states.shape == (args.env_num, args.state_dim)
        agent.states = states
    return agent


def init_buffer(args, gpu_id: int) -> [ReplayBuffer]:
    if args.if_off_policy:
        buffer = ReplayBuffer(gpu_id=gpu_id,
                              max_capacity=args.max_memo,
                              state_dim=args.state_dim,
                              action_dim=1 if args.if_discrete else args.action_dim, )
        buffer.save_or_load_history(args.cwd, if_save=False)

    else:
        buffer = ReplayBufferList()
    return buffer


def init_evaluator(args, gpu_id: int, logger=None) -> Evaluator:
    evaluator = Evaluator(cwd=args.cwd, agent_id=gpu_id, eval_env=args.env, args=args, logger=logger)
    return evaluator

def init_read_write(tech_id, logger):
    if tech_id == "double_kf":
        # read_write = importlib.import_module(f"TradingStrategy.{tech_id}.read_write")
        from ahf.rl.strategies.double_kf import read_write
    elif tech_id.lower() == "rsi_macd":
        from ahf.rl.strategies.rsi_macd import read_write
    else:
        logger.error(f"load_trading_strategy Unknown tech_id: {tech_id}")
        sys.exit(-1)

    return read_write