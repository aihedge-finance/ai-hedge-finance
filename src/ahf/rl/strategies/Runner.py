import sys
import time
from ahf.utils.utils import setup_logger
from api.PriceFetcher import AppEnv
from ahf.utils.utils import readable_error
from api.Binance.BinanceOrder import BinanceOrder, DummyOrder
from .utils import plot_sim
from .ValueNetworkEnv import BaseEnv


def evaluator(symbol, env_name,
              buy_delta1, sell_delta1,
              buy_delta2, sell_delta2,
              logger=None, plot=False):
    logger = setup_logger(f'{env_name}_RunSim.log', symbol) if logger is None else logger

    config = _init_env()

    assert symbol == config.symbol, f"symbol inconsistent expect {symbol} got {config.symbol}"
    assert env_name == config.env_name, f"env_name inconsistent expect {env_name} got {config.env_name}"

    re = run_sim(config,
                 buy_delta1, sell_delta1,
                 buy_delta2, sell_delta2, logger, verbose=False)

    if plot:
        plot_sim(config.trade_args, re, logger)

    return re['gain']


def run_sim(c,
            buy_delta1, sell_delta1,
            buy_delta2, sell_delta2, logger):
    try:
        c.trade_args['buy_delta1'] = buy_delta1
        c.trade_args['sell_delta1'] = sell_delta1
        c.trade_args['buy_delta2'] = buy_delta2
        c.trade_args['sell_delta2'] = sell_delta2

        env = DummyEnv(c.trade_args,
                       c.tech_list,
                       c.trade_model,
                       logger,
                       price_fetcher=c.price_fetcher,
                       done_enabled=False)

        price_len = c.price_fetcher.price_len()
        _state = env.reset()
        assert _state is not None, "state cannot be None after reset"

        for i in range(price_len):
            ary_action = 1
            ary_state, reward, done, _ = env.step(ary_action)

        gain = env.ds.get_cumulative_realized_pnl()

        return {
            'gain': gain,
            'buy_delta1': buy_delta1,
            'sell_delta1': sell_delta1,
            'buy_delta2': buy_delta2,
            'sell_delta2': sell_delta2,
            'buy_indi': buy_indi,
            'sell_indi': sell_indi,
            'price_actual': price_actual
        }
    except Exception as e:
        logger.error(readable_error(e, __file__))
        time.sleep(3)
        sys.exit()


def _init_env(logger):
    # 虛擬測試
    app_env = AppEnv.TRAIN
    exch_mode = 'SpotTest'
    spot_margin = 'spot'

    msg = "在 虛擬 TRAIN 測試"

    logger.info(f">> test_level: {msg} ==>\n"
                f"app_env: {AppEnv.TRADE}, exch_mode: {exch_mode}, "
                f"spot_margin: {spot_margin}")

    if app_env == AppEnv.TRADE:
        exch_api = BinanceOrder(exch_mode, logger)
    elif app_env == AppEnv.TRAIN:
        exch_api = DummyOrder(exch_mode, logger)
    else:
        raise Exception('app_env can only be TRADE or TRAIN')

    return _StandaloneRunnerConfig(app_env, exch_mode, exch_api, spot_margin, logger)
