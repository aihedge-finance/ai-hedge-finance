import sys
import math
import numpy as np
import datetime as dt
import numpy.typing as npt
import numpy.random as rd

from ahf.utils.utils import convert_to_min, pretty_dict, is_dir_exist, readable_error
from ahf.utils.utils import d, d_round, d_abs

# Price fetcher
from ahf.core.enums import AppEnv, PriceEnv
from api.PriceFetcher import PriceFetcher, PriceFetcherTrain

# Order
from api.Binance.BacktestOrder import BacktestOrder
from api.Binance.BacktestOrderData import BacktestOrderData


class BaseEnv:
    def __init__(self, trade_args, tech_list, trade_model, logger,
                 price_fetcher=None, exch_api=None):

        # 基本設定
        self.symbol = trade_args.get("symbol")
        self.home_asset = trade_args.get("home_asset")
        self.target_asset = trade_args.get("target_asset")

        self.app_env = AppEnv.TRAIN  # DEFAULT
        self.price_env = PriceEnv.TRAIN  # DEFAULT
        self.trade_args = trade_args
        self.tech_list = tech_list
        self.trade_model = trade_model

        self.price_fetcher = price_fetcher
        self.bt_order = None
        self.exch_api = exch_api
        self.ds = None

        self.logger = logger
        self.logger.info(f'[Datastore] {__name__} Class loaded')

    def reset(self):
        try:
            self.ds.reset()
            assert self.ds.get_idx() == 0, f'idx should be 0 but got {self.ds.get_idx()}'

            dt_idx, price_new = self.get_price(0)

            # *******************************************
            # 處理 ds.reset() 和 stacking_lookback init
            # *******************************************
            # 至少跑一次
            # 要先有價格才能 step
            self.ds.set_price(dt_idx=dt_idx,
                              price=price_new,
                              app_env=self.app_env)
            self.ds.set_cash_asset(cash=d(self.ds.init_trade_cash),
                                   borrowed_cash=d(0),
                                   asset=d(0))  # only cash initially
            self.ds.set_target_cash(d(self.ds.init_target_cash))  # only cash initially

            # 注入 tech init
            self.ds.set_tech_ary(self.ds.tech_ary_init[0, :])

            self.ds.step_idx(self.app_env)

            # 假裝 step
            state = None
            for i in range(self.ds.stacking_lookback):
                # 會記錄下所有事
                _ = self.step(np.zeros(self.action_dim, dtype=np.float32))

            return state

        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)
            sys.exit()

    def step(self, action: npt.ArrayLike):
        try:


            return None
        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)
            self.logger.error(f'silo: {self.ds.silo.position} '
                              f'position:{self.ds.get_last_5_position()}')

            sys.exit()

    def get_price(self, idx: int):
        if self.price_env == PriceEnv.TRAIN:
            dt_idx, price_arr = self.price_fetcher.get_price(idx)
            price_new = price_arr[-1]
        else:
            price_se, _ = self.price_fetcher.get_price()
            price_new = price_se[-1]
            dt_idx = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)

        return dt_idx, price_new

    def Trade_Args_Setup(self):
        _env_name = "DummyEnv"

        # get cmd args
        from Trade.Binance.BinanceTrade import load_args
        from Trade.Binance.parse_arguments import trade_parse_arguments

        # 取得替代參數的 cmd args
        self.cmd_args = _cmd_args = vars(trade_parse_arguments(
            ['--env_name', 'DummyEnv',
             '--exch_mode', 'SpotAPI',
             '--agent_id', '0',
             '--init_trade_cash', '0.005',
             '--app_env', 'TRAIN',
             '--trade_args_path', './trade_args/ETHBTC_LONG.json',
             '--reset_trade_cash', '0'
             ]))

        self.exch_mode, self.trade_args, self.tech_list = load_args(_env_name, _cmd_args)

        # 檢查 trade_model 所需 trade_args
        order_trade_args_checker(self.trade_args)

    def DataStore_Trade_Mode_Setup(self):
        # You must have price_fetcher at the level
        self.ds = BacktestOrderData(self.symbol,
                                    self.home_asset,
                                    self.target_asset,
                                    self.exch_api,
                                    3,
                                    self.init_trade_cash,
                                    self.init_target_cash,
                                    self.exch_mode,
                                    self.kelly_cap_args,
                                    self.logger)
        self.ds.reset()

        if self.price_fetcher is None:
            raise Exception("You have to init price_fetcher before setting up BacktestDatastore")

    def PriceFetcher_Setup(self):
        if self.app_env == AppEnv.TRADE:
            self.price_fetcher = PriceFetcher(self.trade_args,
                                              self.price_env,
                                              self.logger,
                                              catchup_price=False)
        else:
            self.price_fetcher = PriceFetcherTrain(self.trade_args,
                                                   self.price_env,
                                                   self.logger,
                                                   catchup_price=False)

    def ExchAPI_Setup(self):
        self.exch_api.validate([self.trade_args.get('symbol')])

    def Trade_Setup(self, spot_margin: str):
        self.bt_order = BacktestOrder(self.ds,
                                      self.exch_api,
                                      self.trade_args,
                                      self.app_env,
                                      self.logger,
                                      spot_margin=spot_margin)

    def populate_init_data(self):
        if self.price_env == PriceEnv.TRAIN:
            dt_idx, price_arr = self.price_fetcher.get_price(self.ds.get_idx())
            price_new = price_arr[-1]
        else:
            price_se, _ = self.price_fetcher.get_price()
            price_new = price_se[-1]
            dt_idx = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)

        self.ds.set_price(dt_idx=dt_idx,
                          price=price_new,
                          app_env=self.app_env)





