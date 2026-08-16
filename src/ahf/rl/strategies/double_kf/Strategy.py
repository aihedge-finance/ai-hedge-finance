import sys
import math
import time
import threading
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime as dt
import simplejson as json
from loguru import logger
import matplotlib
import matplotlib.pyplot as plt

from typing import Deque, List, Optional, Tuple, Dict
from decimal import Decimal
from collections import deque
from ahf.rl.envs.TradeEnum import TradeAction

from ahf.utils.logger import configure_logger
from ahf.core.enums import AppEnv, PriceEnv, BotEnv
from ahf.utils.schema import PlotSimParam
from ahf.utils.utils import d, save_file, DecimalEncoder, get_project_root
from ahf.utils.utils import pretty_dict, readable_error, pct_change, get_project_root

from ahf.preprocessor.kf.Tracer_v2 import Tracer_v2, TracerState
from ahf.preprocessor.kf.TracerMem_v2 import TracerMem_v2
from ahf.preprocessor.kf.TracerSimple import TracerSimple
from ahf.preprocessor.kf.TracerSimpleMem import TracerSimpleMem

from ahf.rl.envs.BrunhildDatastore_v11 import BacktestOrderData
from ahf.rl.envs.BacktestValueNetworkEnv import BacktestValueNetworkEnv
from api.Binance.BinanceOrder import BinanceOrder
from api.Binance.BinanceOrder import DummyOrder

from ahf.rl.strategies.utils import order_trade_args_checker

from ahf.rl.train.config import get_trade_args, get_tech_args


page_counter = 0
project_root = get_project_root()

lock = threading.Lock()

DEFAULT_BUY_OBS_COV = 0.5
DEFAULT_BUY_SD_DELTA = 1e-6
DEFAULT_BUY_SD_OBS_COV = 0.1
DEFAULT_SELL_OBS_COV = 0.5
DEFAULT_SELL_SD_DELTA = 1e-6
DEFAULT_SELL_SD_OBS_COV = 0.1

class Strategy:
    NAME = "double_kf"
    STATE_TECH_COLS = ["buy_tracer", "buy_alfa", "sell_tracer", "sell_alfa", "buy_sd", "sell_delta"]
    STATE_TECH_DIM = len(STATE_TECH_COLS)
    ALL_TECH_COLS = ["buy_tracer", "buy_alfa", "sell_tracer", "sell_alfa", "buy_sd", "sell_delta"]
    ALL_TECH_DIM = len(ALL_TECH_COLS)
    # STATE_TECH_COLS_HIGH = [10, 10, 10]
    # STATE_TECH_COLS_LOW = [-10, -10, -10]

    """
        ### Tuning Space
        The space can be used for boundary or optimization. By finding a good estimate,
        we can fix those value for better/easier hyperparameter tuning again

        CONTROLLING SPACE
        | Num | Boundary              | Min                  | Max                |   example   |
        |-----|-----------------------|----------------------|--------------------|-------------|
        | 0   | long_level            | 0                    | 0.4                |  0.12(=12%) |
        | 1   | long_exit_level       | 0                    | 0.3                |  0.5        |
        | 2   | short_level           | -1                   | 1                  |  12 (=12%)  |
        | 3   | short_exit_level      | -Inf                 | 10                 |  0.5        |
        | 4   | exit_rule             | 0                    | 5                  |  0.5        |
        | 5   | stop_loss_margin      | 0.001                | 0.999              |             |
        | 4   | buy_delta             | 0                    | 5                  |  0.5        |
        | 5   | sell_delta            | 0.001                | 0.999              |             |


        INTERNAL SPACE(set constant):
        | 6   | buy_obs_cov           | 0.0                  | 0.999              |             |
        | 7   | buy_sd_delta          | -1                   | 1                  | 0,1,-1      |
        | 8   | buy_sd_obs_cov        | -1                   | 1                  | 0.01 (=1%)  |
        | 9   | sell_obs_cov          | 0.0                  | 0.999              |             |
        | 10  | sell_sd_delta         | -1                   | 1                  | 0,1,-1      |
        | 11  | sell_sd_obs_cov       | -1                   | 1                  | 0.01 (=1%)  |

    """
    def __init__(self, app_env, trade_args, tech_args, _logger):
        """

        """
        try:
            self.app_env = app_env
            self.price_env = trade_args.get("price_env")
            self.trade_args = trade_args
            self.tech_args = tech_args
            self.tech_list = tech_args.get("tech_list")

            self.logger = _logger

            # assert len(self.ALL_TECH_COLS) == self.ALL_TECH_DIM
            # assert len(self.STATE_TECH_COLS) == self.STATE_TECH_DIM
            # assert set(self.STATE_TECH_COLS).issubset(set(self.ALL_TECH_COLS))

            self.logger.info(f"Init Strategy {self.NAME}")

            # tech normalizer
            # .... pass .....

            self._idx: Optional[int] = None
            self._signal: Optional[Deque[Optional[int]]] = None
            self._position_side: Optional[Deque[Optional[int]]] = None

            # kalman filter configuration
            self.buy_obs_cov: float = tech_args.get("buy_obs_cov", DEFAULT_BUY_OBS_COV) or DEFAULT_BUY_OBS_COV  # 0.3 ~ 1
            self.buy_sd_delta: float = tech_args.get("buy_sd_delta", DEFAULT_BUY_SD_DELTA) or DEFAULT_BUY_SD_DELTA  # 1e-6 ~ 1e-4
            self.buy_sd_obs_cov: float = tech_args.get("buy_sd_obs_cov", DEFAULT_BUY_SD_OBS_COV) or DEFAULT_BUY_SD_OBS_COV   # 0.1 ~ 0.5
            self.sell_obs_cov: float = tech_args.get("sell_obs_cov", DEFAULT_SELL_OBS_COV) or DEFAULT_SELL_OBS_COV  # 0.3 - 1
            self.sell_sd_delta: float = tech_args.get("sell_sd_delta", DEFAULT_SELL_SD_DELTA) or DEFAULT_SELL_SD_DELTA  # 1e-6 ~ 1e-4
            self.sell_sd_obs_cov: float = tech_args.get("sell_sd_obs_cov", DEFAULT_SELL_SD_OBS_COV) or DEFAULT_SELL_SD_OBS_COV  # 0.1 ~ 0.5

            # 注入主要操作
            self.buy_delta = tech_args.get("buy_delta")  # 0.999 ~ 0.0001
            self.sell_delta = tech_args.get("sell_delta")  # 0.9 ~ 0.0001

            # check param
            if self.buy_delta is None or self.sell_delta is None:
                raise Exception("buy_delta and sell_delta cannot be None")

            # ==== indicator signal ====
            self.BuyIndi: Optional[Tracer_v2] = None
            self.SellIndi: Optional[Tracer_v2] = None

            self.BuySimple: Optional[TracerSimple] = None
            self.SellSimple: Optional[TracerSimple] = None

            self.signal = deque([0])
            self.position_side = deque([0])

            # tech_args
            self.long_level = tech_args.get("long_level")
            self.short_level = tech_args.get("short_level")

            assert self.long_level is not None, "long_level is required"
            assert self.short_level is not None, "long_level is required"
            assert self.short_level <= 0 , "short_level must be smaller than zero"

            self.long_exit_level = tech_args.get("long_exit_level")
            self.short_exit_level = tech_args.get("short_exit_level")
            self.exit_rule = tech_args.get("exit_rule")
            assert self.long_exit_level is not None, "long_exit_level is required"
            assert self.short_exit_level is not None, "short_exit_level is required"
            assert self.exit_rule is not None, "short_exit_level is required"

            if tech_args.get("exit_rule") not in ("price_cross", "sell_tracer"):
                raise Exception("Invalid exit rule")

            assert trade_args.get("stop_loss_margin") is not None, "stop_loss_margin is required for strategy double_kf"

            self.stop_loss_margin = float(trade_args.get("stop_loss_margin"))
            assert self.stop_loss_margin < 0, "stop_loss_margin mus be smaller than zero"

            self.reset()
        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(f"Strategy {self.NAME}: {err}")
            sys.exit(-3)

    def warm_up(self, price_hist: pd.DataFrame):
        """
        給實際交易 BinanceTrade (Bot) 起始值
        """
        # 處理 init
        price_hist["open"] = price_hist["open"].apply(lambda x: Decimal(x))
        last_price: Decimal = None
        buy_ind: TracerState = None
        sell_ind: TracerState = None
        buy_simple: float = None
        sell_simple: float = None

        for idx, (dt_idx, price) in tqdm(enumerate(price_hist["open"].items())):
            assert isinstance(dt_idx, dt.datetime)
            # dt_idx = dt_idx.astype(dt.datetime)
            if idx == 0:
                price_list = [price, price]
            else:
                price_list = [last_price, price]

            # 交易規則模型
            buy_ind, sell_ind, buy_simple, sell_simple = self.step_ind(price_list,
                                                                        dt_idx,
                                                                        self.buy_delta,
                                                                        self.sell_delta,
                                                                        self.trade_args.get("long_short"))
            last_price = price

            # 列印 stdout
            if idx % 1000 == 0:
                msg = f"\033[KTraining at {idx}                                                         \r"
                sys.stdout.write(msg)
                sys.stdout.flush()

        # [END for]

        # STATE_TECH_COLS = ["buy_tracer", "buy_alfa", "sell_tracer", "sell_alfa", "buy_sd", "sell_delta"]
        last_tech_ary = buy_ind.tracer[1], buy_ind.alfa_norm_100[1], sell_ind.tracer[1], sell_ind.alfa_norm_100[1], buy_ind.sd[1], sell_ind.delta[1]

        return last_tech_ary


    def render(self):
        return f" | sell_delta:{self.sell_delta:>7,.5f}"

    def reset(self):
        try:
            self._idx = 0
            self._signal = deque([0])
            self._position_side = deque([0])

            self.BuySimple = TracerSimpleMem(self.trade_args, self.NAME, "BuySimple", self.buy_delta, self.buy_obs_cov, self.logger)
            self.SellSimple = TracerSimpleMem(self.trade_args, self.NAME, "SellSimple", self.sell_delta, self.buy_obs_cov, self.logger)

            # delta, self.obs_cov, self.sd_delta, self.sd_obs_cov
            self.BuyIndi = TracerMem_v2(self.trade_args,
                                        self.NAME,
                                        "Buy",
                                        self.buy_delta,
                                        self.buy_obs_cov,
                                        self.buy_sd_delta,
                                        self.buy_sd_obs_cov, self.logger)

            # === sell ====
            self.SellIndi = TracerMem_v2(self.trade_args,
                                         self.NAME,
                                        "Sell",
                                        self.sell_delta,
                                        self.sell_obs_cov,
                                        self.sell_sd_delta,
                                        self.sell_sd_obs_cov, self.logger)

            self.populate_init_data()
        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(f"{self.NAME}.reset(): {err}")
            time.sleep(3)
            sys.exit()

    def populate_init_data(self):
        if self.app_env in (AppEnv.TRADE, AppEnv.SIMULATION):
            # TODO load from file data
            pass
        elif self.app_env == AppEnv.TRAIN:
            # do nothing
            return

        # [unreachable]
        raise Exception(f"Strategy populate_init_data unknown self.app_env got {self.app_env}")

    def step_ind(self, price_list, dt_idx, buy_delta, sell_delta, long_short)-> Tuple[TracerState, TracerState, float, float]:
        """
        這個可以被單獨拿來使用，如果你沒有要拿來看 signal, 只是要看 indicator 的
        改變狀況（如：用來 RL 訓練時 state 的使用），就可以用這個。
        Parameters
        ----------

        price_list
        dt_idx
        sell_delta
        buy_delta
        long_short
        make_pair: return data as pair
        """
        price_pct_list = pct_change(price_list, include_first=True)

        price_pct = price_pct_list[1]
        # update indicators
        buy_state: TracerState = self.BuyIndi.update(float(price_pct), delta=buy_delta)
        sell_state: TracerState = self.SellIndi.update(float(price_pct), delta=sell_delta)

        buy_simple_now: float = self.BuySimple.update(float(price_list[1]), delta=buy_delta)
        sell_simple_now: float = self.SellSimple.update(float(price_list[1]), delta=sell_delta)

        return buy_state, sell_state, buy_simple_now, sell_simple_now

    def __append(self):
        """注意， append data slot 跟 step_idx 不要在一起！這是step 一開始，step_idx 時 step 最後做"""
        self._position_side.append(None)
        self._signal.append(None)

    def __step_idx(self):
        """注意， append data slot 跟 step_idx 不要在一起！這是step 一開始，step_idx 時 step 最後做"""
        self._idx += 1

    def step(self,
             price_list: List[Decimal],
             dt_idx: dt.datetime,
             ds, long_short,
             buy_delta=None,
             sell_delta=None) -> Tuple[TradeAction, Tuple]:
        """
        Step strategy 來取得 signal 和執行的訊號，主要交易邏輯
        在外部自己執行就放 order=None

        Parameters
        ----------
        buy_delta
        sell_delta
        price_list: is the last price and the current price in list format
        dt_idx: datetime index
        ds: Datasource container
        long_short: long or short
        """

        try:
            if not isinstance(price_list, (np.ndarray, list)):
                raise Exception("Invalid prices type, it must be a list of size 2")

            if dt_idx is None:
                raise Exception("TradingStrategy.step => dt_idx cannot be None")

            if not isinstance(dt_idx, dt.datetime):
                raise Exception(f"dt_idx must be dt.datetime but got {type(dt_idx)}")

            self.buy_delta = buy_delta if buy_delta is not None else self.buy_delta
            self.sell_delta = sell_delta if sell_delta is not None else self.sell_delta

            price_pct_list = pct_change(price_list, include_first=True)

            if self._idx == 0:
                self.__append()

                price_pct: Decimal = price_pct_list[0]
                price: Decimal = price_list[0]
                last_position: Decimal = ds.get_last_position()

                # update tracer
                buy_state: TracerState = self.BuyIndi.update(float(price_pct), delta=buy_delta)
                sell_state: TracerState = self.SellIndi.update(float(price_pct), delta=sell_delta)

                buy_simple_now: float = self.BuySimple.update(float(price), delta=buy_delta)
                sell_simple_now: float = self.SellSimple.update(float(price), delta=sell_delta)

                self._signal[self._idx] = 0
                self._position_side[self._idx] = 0
                self.__step_idx()
            # [end if]

            """self._idx>0"""
            self.__append()

            price_pct: Decimal = price_pct_list[1]
            price: Decimal = price_list[1]
            last_position: Decimal = ds.get_last_position()

            # update tracer
            buy_state: TracerState = self.BuyIndi.update(float(price_pct), delta=buy_delta)
            sell_state: TracerState = self.SellIndi.update(float(price_pct), delta=sell_delta)

            buy_simple_now: float = self.BuySimple.update(float(price), delta=buy_delta)
            sell_simple_now: float = self.SellSimple.update(float(price), delta=sell_delta)

            """
            buy_state = {
                "dt_idx": dt_idx,
                "price_pct": price_pct, 
                "tracer": tracer_new, 
                "alfa": alfa_new,
                "alfa_norm_100": alfa_norm_100,
                "sd": sd_new, 
                "delta": delta, 
                "state_cov": state_cov_new, 
                "sd_state_cov": sd_state_cov_new,
            }
            """

            # 檢查太少 => 沒關係，這只是初步給訊號，實際交易還要看 silo, 這是上層的工作， strategy 就是決定基本邏輯
            in_position: int = ds.is_significant_pos()

            signal = self.cal_signal_run(price_list, ds, buy_state.alfa_norm_100, buy_state.tracer, sell_state.alfa_norm_100, last_position)
            self._signal[self._idx] = signal

            position_side = self._position_side[self._idx]  # useless?
            trade_action_new = self.cal_trade_action(signal, in_position)
            if trade_action_new == TradeAction.BUY:
                position_side = 1
            elif trade_action_new == TradeAction.SHORT:
                position_side = -1
            elif trade_action_new in (TradeAction.SELL, TradeAction.COVER):
                position_side = 0
            else:
                position_side = self._position_side[self._idx-1]

            # 如果只能單邊買賣，那就只能改成 HOLD
            if long_short == "long" and trade_action_new == TradeAction.SHORT:
                trade_action_new = TradeAction.HOLD
            if long_short == "short" and trade_action_new == TradeAction.BUY:
                trade_action_new = TradeAction.HOLD

            self._position_side[self._idx] = position_side

            self.__step_idx()

            # return trade_action_new, {"buy_state": buy_state, "sell_state": sell_state, "buy_simple": buy_simple_now, "sell_simple": sell_simple_now}
            tech_ary = buy_state.tracer[1], buy_state.alfa_norm_100[1], sell_state.tracer[1], sell_state.alfa_norm_100[1], buy_state.sd[1], sell_state.delta[1]

            return trade_action_new, tech_ary
        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)
            raise Exception(e)

    def cal_signal_run(self, price_list, ds, buy_alfa_list, buy_tracer_list, sell_alfa_list, last_position) ->  int:
        try:
            in_position = ds.is_significant_pos()

            # Exit position if exceed loss tolerance， drawdown 太多就出場
            _, paper_gain_pct = ds.cal_paper_pnl_run(price_list[1])
            paper_gain_pct = float(paper_gain_pct)

            if self.exit_rule == "price_cross":
                if not in_position and buy_alfa_list[0] < self.long_level <= buy_alfa_list[1]:
                    signal = 1
                elif not in_position and buy_alfa_list[0] > self.short_level >= buy_alfa_list[1]:
                    signal = -1
                elif in_position and price_list[0] > buy_tracer_list[1] >= price_list[1]:
                    signal = 2
                elif in_position and price_list[0] < buy_tracer_list[1] <= price_list[1]:
                    signal = -2
                elif in_position and last_position > 0 and  paper_gain_pct < self.stop_loss_margin:
                    signal = 3
                elif in_position and last_position < 0 and paper_gain_pct < self.stop_loss_margin:
                    signal = -3
                else:
                    signal = 0
            elif self.exit_rule == "sell_tracer":
                if not in_position and buy_alfa_list[0] < self.long_level <= buy_alfa_list[1]:
                    signal = 1  # TradeAction.BUY
                elif not in_position and buy_alfa_list[0] > self.short_level >= buy_alfa_list[1]:
                    signal = -1  # TradeAction.SHORT
                elif in_position and last_position > 0 and sell_alfa_list[0] >= self.long_exit_level:  #  >= sell_alfa_list[1] or :
                    signal = 2  # TradeAction.SELL
                elif in_position and last_position < 0 and sell_alfa_list[0] <= self.short_exit_level:  # <= sell_alfa_list[1]:
                    signal = -2  # TradeAction.COVER
                elif in_position and last_position > 0 and  paper_gain_pct < self.stop_loss_margin:
                    signal = 3  # # TradeAction.SELL
                elif in_position and last_position < 0 and paper_gain_pct < self.stop_loss_margin:
                    signal = -3  # TradeAction.COVER
                else:
                    signal = 0
            else:
                raise Exception(f"Invalid exit_rule:{self.exit_rule}")


            return signal

        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)
            raise Exception(e)

    def cal_trade_action(self, signal: int, in_position: int):
        try:
            if signal == 1 and not in_position == -1:  # position_signal == 0
                trade_action = TradeAction.BUY
            elif signal == 2 and in_position == 1:  # position_signal == 1
                trade_action = TradeAction.SELL
            elif signal == -1 and not in_position == 1:  # position_signal == 0
                trade_action = TradeAction.SHORT
            elif signal == -2 and in_position == -1:  # position_signal == -1
                trade_action = TradeAction.COVER
            elif signal == 3 and in_position == 1:
                trade_action = TradeAction.SELL
            elif signal == -3 and in_position == -1:
                trade_action = TradeAction.COVER
            else:
                trade_action = TradeAction.HOLD

            return trade_action
        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)
            raise Exception(e)

    def cal_reward(self, ds: BacktestOrderData , realized_pnl_pct: Decimal , reward_per_step: float) -> float:
        """
        ML 計算 reward 的 function
        Parameters
        ----------
        :param ds : 資料源
        :param realized_pnl_pct： 已實現利益
        :param reward_per_step： reward for long-running

        Returns
        -------

        """
        try:
            # sample
            # return 1.0 if np.all(self.state == action) else 0.0

            # get reward for nothing because we want it to continue longer
            step_reward = reward_per_step

            if ds.get_buysell() in (TradeAction.SELL, TradeAction.COVER):
                # encourage instead of scolding, because it may earn money and then lose it later
                # therefore we need to capture good process.
                step_reward += float(ds.get_realized_pnl()) if ds.get_realized_pnl() > d("0") else float(ds.get_realized_pnl()) * 0.8

            if ds.get_buysell() == TradeAction.BUY or ds.get_buysell() == TradeAction.SHORT:
                step_reward += 0.01

            # penalize when making delta change when in position
            # if buy_delta_steer == 0 or sell_delta_steer == 0:  # my_obv.position == 0 and
            #     step_reward += 0.1

            # how many profitable trade for the past 10 trades
            # penalize if past 10 trades are bad, it is like hitting a wall
            # if my_obv.position == 0 and ds.get_num_profit_trades() <= 4:
            #     step_reward -= 0.5

            # if ds.get_buysell() == TradeAction.BUY or ds.get_buysell() == TradeAction.SHORT:
            #     step_reward += 0.5

            # Confusion matrix
            # cm = my_obv.confusion_matrix
            # num_trades = cm['tp'] + cm['tn'] + cm['fp'] + cm['fn']
            # accuracy = (cm['tp'] + cm['tn']) / num_trades if num_trades > 0 else 0
            # if accuracy > 0.5:
            #     step_reward += (accuracy - 0.5) * 100
            # elif accuracy <= 0.5:
            #     step_reward -= (0.5 - accuracy)

            # extremely simplistic HIRO value function
            # self.last_reward = self.total_reward

            # IMPORTANT
            '''
            I found that by adding `accuracy * (cash_asset - trade_cash)`, it trades long term(HODL)

            '''
            # self.total_reward = accum_realized_pnl + accuracy * (cash_asset - trade_cash) + (0.75 * step_reward)

            # c = math.sqrt(math.log(num_trades+1)/math.log(self.current_step+1)) if self.current_step > 0 else 0
            # total_reward = (c * ds.get_last_cumulative_realized_pnl()) + (0.75 * step_reward)
            total_reward = step_reward

            return float(total_reward)

        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)
            sys.exit()


def _run_sim(_trade_args, _tech_args, plot_args, _logger,
             verbose=True,
             BinanceOrderInject=None,
             monthly_report=False):

    # logger = setup_logger("double_kf_strategy_py.log", trade_args.get("symbol"))

    try:
        exch_mode = _trade_args.get("spot_margin")
        app_env = AppEnv(_trade_args.get("app_env"))
        price_env = PriceEnv(_trade_args.get("price_env"))
        spot_margin = _trade_args.get("spot_margin")

        # Using update method, not train method
        exch_api = DummyOrder(exch_mode, _logger, BinanceOrderInject=BinanceOrderInject)
        v = BacktestValueNetworkEnv(app_env, price_env, exch_mode, exch_api, spot_margin,
                                    _trade_args, _tech_args, _logger,
                                    verbose=verbose)

        strategy = Strategy(app_env, _trade_args, _tech_args, _logger)

        # 開始
        idx = 0
        # start = timer()
        while True:
            # 拿到新價格
            dt_idx, price_new = v.price_fetcher.get_price(idx)
            if dt_idx is None and price_new is None:
                break

            dt_idx = dt_idx.astype(dt.datetime)
            if idx == 0:
                prices = [price_new[0], price_new[0]]
            else:
                prices = [v.ds.get_last_price()[0], price_new[0]]

            # 環境 step
            v.ds.step_idx(app_env)
            idx = v.ds.get_idx()

            # 交易規則模型
            trade_action_new, tech_ary = strategy.step(prices,
                                                       dt_idx,
                                                       v.ds,
                                                       _trade_args.get("long_short"),
                                                       strategy.buy_delta,
                                                       strategy.sell_delta
                                                       )

            # 交易
            if trade_action_new == TradeAction.BUY and bool(_trade_args.get("kelly_cap_enabled")):
                action = d(max(v.ds.ds_cal_kelly_cap(), float(_trade_args.get("min_kelly_cap"))))
            else:
                action = d(1)

            if trade_action_new == TradeAction.HOLD:
                v.order.hold(idx, dt_idx, prices[1])
            elif trade_action_new == TradeAction.BUY:
                v.order.buy(idx, dt_idx, action, prices[1])
            elif trade_action_new == TradeAction.SELL:
                v.order.sell(idx, dt_idx, action, prices[1])
            elif trade_action_new == TradeAction.SHORT:
                v.order.short(idx, dt_idx, action, prices[1])
            elif trade_action_new == TradeAction.COVER:
                v.order.cover(idx, dt_idx, action, prices[1])
            else:
                raise Exception(f"[TradingStrategy] unknown trade_action_new {trade_action_new}")

            # 列印 stdout
            if idx % 1000 == 0:
                msg = f"\033[KTraining at {idx}                                                         \r"
                sys.stdout.write(msg)
                sys.stdout.flush()

            final_pnl = v.ds.get_cash_asset()
            gain = (final_pnl - v.ds.init_trade_cash) / v.ds.init_trade_cash * 100
            if gain < -30: # -30%
                break

        # RMSE
        # price_sell_tracer_diff = price_train - sell_indi['tracer']
        # sell_buy_tracer_diff = sell_indi['tracer'] - buy_indi['tracer']

        # MSE = mean_squared_error(price_sell_tracer_diff, sell_buy_tracer_diff)
        # RMSE = math.sqrt(MSE)
        # end = timer()
        # print(f"time spent per training {end -start}")

        global page_counter

        final_pnl = v.ds.get_cash_asset()
        gain_pct = (final_pnl - v.ds.init_trade_cash) / v.ds.init_trade_cash * 100
        _logger.info("[Strategy] Double_KF Final Result for step, "
                    "init:${0:>9,.3f}, "
                    "pnl:${1:>9,.3f}, "
                    "gain:{2:>8,.3f}%, "
                    "cal count:{3:>3}"
                    "".format(v.ds.init_trade_cash, final_pnl, gain_pct,
                              page_counter))  # , opts:{5}, vars(opts)

        if gain_pct > 15:
            lock.acquire()
            page_counter += 1

            data = {
                "hash_tag": _tech_args.get("hash_tag"),
                "gain_pct": gain_pct,
                "tech_args": _tech_args,
                "trade_args": _trade_args,
                "trade_count": v.ds.num_trade
            }

            json_line = json.dumps(data) + '\n'
            save_file(f"{project_root}/logs/double_kf_strategy_RunSim_{v.ds.symbol}_{_trade_args.get('trade_interval')}.txt", json_line, 'a')
            lock.release()

        if plot_args.get("plot_show") or plot_args.get("save_img"):
            plot_sim_params = PlotSimParam(**{"ds": v.ds, "strategy": strategy, "tech_args": _tech_args, "trade_args": _trade_args, "plot_args": plot_args, "gain_pct": gain_pct})
            plot_sim(plot_sim_params, _logger)

        if monthly_report:
            daily_json = gen_monthly_report(v.ds._dt_idx, v.ds._cash_asset, v.ds.price_ary, _logger)
            _logger.info(f"monthly_data = {daily_json}")

        return {
            "gain": gain_pct,
            "ds": v.ds
        }
    except Exception as e:
        _logger.error(readable_error(e, __file__))
        sys.exit()

def evaluator(_trade_args, _tech_args, plot_args, _logger,
              BinanceOrderInject=None,
              monthly_report=False):
    """
    跟 run 不同的是，他是給 XGBoost 或其他來使用的，屬於哪來訓練的，
    可以另外自行帶入參數的改變等
    """

    def sigmoid(x):
        sig = 1 / (1 + math.exp(-x))
        return sig

    eval_re = _run_sim(_trade_args, _tech_args, plot_args, _logger,
                       BinanceOrderInject=BinanceOrderInject,
                       monthly_report=monthly_report)
    profit_loss = eval_re.get("gain")
    ds = eval_re.get("ds")

    # loss = 1 - math.tanh(gain / 100)
    loss_fn = 1 - 2 * (sigmoid(profit_loss / 100) - 0.5)

    return {"gain": profit_loss, "loss_fn": loss_fn, "trade_count": ds.num_trade }


def gen_monthly_report(
        dt_index: Deque[dt.datetime],
        cash_asset: Deque[Decimal],
        price_ary: Deque[Decimal],
        _logger
) -> str:
    """
    Convert high-frequency data to end-of-month JSON format with percentage changes,
    including the first date of the series

    Args:
        dt_index (Deque[datetime]): Deque of datetime timestamps
        cash_asset (Deque[Decimal]): Deque of corresponding cash+asset values in Decimal type
        price_ary (Deque[Decimal]): Deque of corresponding price values in Decimal type
        _logger: logger

    Returns:
        str: JSON string with first date and end-of-month data where dates are formatted as YYYY-MM-DD
              Including original values and percentage changes from initial values
    """
    try:
        # Convert deque to lists and ensure we're getting single values, not nested lists
        dates = list(dt_index)
        cash_asset_values = [x[0] if isinstance(x, (list, tuple)) else x for x in cash_asset]
        price_values = [x[0] if isinstance(x, (list, tuple)) else x for x in price_ary]

        # Check if any of the lists are empty
        if not dates or not cash_asset_values or not price_values:
            raise ValueError("One or more input sequences are empty")

        # Check if all lists have the same length
        if not (len(dates) == len(cash_asset_values) == len(price_values)):
            raise ValueError(
                f"Input sequences have different lengths: "
                f"dates={len(dates)}, "
                f"cash_asset={len(cash_asset_values)}, "
                f"price_ary={len(price_values)}"
            )

        # Create dictionary to store monthly data
        monthly_data: Dict[str, Dict] = {}

        # Get initial values for percentage calculations
        initial_cash_asset = float(cash_asset_values[0])
        initial_price = float(price_values[0])

        # Add first date entry
        first_date_str = str(dates[0])[:10]
        monthly_data['0000-00'] = {  # Use special key for first date to ensure it comes first
            'date': first_date_str,
            'cash_asset': initial_cash_asset,
            'cash_asset_pct': 0.0,
            'price': initial_price,
            'price_pct': 0.0
        }

        # Iterate through the data
        for date, cash_asset_val, price_val in zip(dates, cash_asset_values, price_values):
            # Convert datetime to string (YYYY-MM)
            month_str = str(date)[:7]  # Get YYYY-MM
            date_str = str(date)[:10]  # Get YYYY-MM-DD

            # Convert values to float for calculations
            cash_asset_val = float(cash_asset_val)
            price_val = float(price_val)

            # Calculate percentage changes
            if cash_asset_val == initial_cash_asset:
                cash_asset_pct = 0.0
            else:
                cash_asset_pct = ((cash_asset_val - initial_cash_asset) / initial_cash_asset) * 100

            if price_val == initial_price:
                price_pct = 0.0
            else:
                price_pct = ((price_val - initial_price) / initial_price) * 100

            # Update the values for this date - will keep overwriting until last value of month
            monthly_data[month_str] = {
                'date': date_str,
                'cash_asset': cash_asset_val,
                'cash_asset_pct': cash_asset_pct,
                'price': price_val,
                'price_pct': price_pct
            }

        # Convert to list of dictionaries format for better JSON structure
        result = [
            {
                **values
            }
            for month, values in sorted(monthly_data.items())  # Sort by month, '0000-00' will come first
        ]

        # Return JSON string
        return json.dumps(result, indent=2)
    except Exception as e:
        err = readable_error(e, __file__)
        _logger.error(err)
        raise


def plot_sim(params: PlotSimParam, _logger):
    try:
        lock.acquire()

        hash_tag = params.tech_args.get("hash_tag")

        # IMPORTANT 有 hash_tag 就是在跑連續程式
        if hash_tag is not None:
            matplotlib.use("Agg")

        # convert to custom data
        buy_ind = params.strategy.BuyIndi
        sell_ind = params.strategy.SellIndi
        buy_simple = params.strategy.BuySimple.tracer
        sell_simple = params.strategy.SellSimple.tracer
        signal = params.strategy._signal
        position_side = params.strategy._position_side

        # Start plot
        plt.figure(1, figsize=(15, 9))

        """
        ds 資料的第一筆會被直接跳過（但還是有一些空值），
        因為即使是訓練也把他當作假裝 load 資料進來
        但是 strategy 有可能有需要事前計算, 也有可能我們就開始從零開始算，所以長度跟 ds 不同
        所以不管如何，我們去比較少的值
        
        另外一個需要注意的邏輯是，如果時遇到 done, 那 dt_index 最後會是 None, 所以要注意
        以下是處理方法
        
        1. 取得 dt_idx 和 buy_simple 的長度
        2. 因為 buy_simple 是 price[0], price[1] 產生出來
        所以 strategy indicator 會永遠少一個，要減 1 => strategy 做了檢查，把他補上了，不再需要
        3. 如果有 done 發生，要再減一個，因為 有被 step ds 一次然後檢查 done，所以是上一筆資料
        """
        dt_index = list(params.ds._dt_idx)
        dt_index_len = len(dt_index)  # step 1
        buy_simple_len = params.strategy.BuySimple.idx_now

        d_size = min(dt_index_len, buy_simple_len) # -1  # step 2
        if dt_index[-1] is None:  # step 3
            # done 發生，沒有完全執行完
            d_size = d_size - 1


        # 整理資料
        dt_index = dt_index[:d_size]
        price_ary = np.array(list(params.ds.price_ary)[:d_size]).ravel()
        buy_simple = list(buy_simple[:d_size])
        assert buy_simple[-1] is not None, "Strategy buy_simple length does not match dt_index"
        sell_simple = list(sell_simple[:d_size])
        signal = list(signal)[:d_size]
        pos = list(position_side)[:d_size]

        cash_asset = [float(item) for item in list(params.ds._cash_asset)[:d_size]]
        shares = [float(item) for item in list(params.ds._position)[:d_size]]
        buy_ind_alfa = list(buy_ind.alfa)[:d_size]
        sell_ind_alfa = list(sell_ind.alfa)[:d_size]
        buy_ind_deltas = list(buy_ind.deltas)[:d_size]
        sell_ind_deltas = list(sell_ind.deltas)[:d_size]

        # ============================================================
        ax1 = plt.subplot(711)
        # plt.yscale("log")
        plt.plot(dt_index, price_ary, label="price actual", lw=1)
        plt.plot(dt_index, buy_simple, label="Buy Simple", color="green", ls="-.", lw=1)
        plt.plot(dt_index, sell_simple, label="Sell Simple", color="orange", ls="-.", lw=1)

        ax1.set_ylabel("price")
        plt.grid()
        plt.legend(loc="best")

        plt.title(f"{params.trade_args.get('tech_id')} TradingStrategy {params.ds.symbol}, Gain:{params.gain_pct:.2f}%")

        # ============================================================
        ax2 = plt.subplot(712, sharex=ax1)
        # plt.step(signal_pd.index, signal_pd, label="Signal", lw=1)
        plt.step(dt_index, list(signal), label="Signal ref", lw=1)

        plt.axhline(0, color="black", ls="-.", lw=1)
        plt.axhline(1, color="green", ls="-.", lw=1)
        plt.axhline(2, color="green", ls="-.", lw=1)
        plt.axhline(-1, color="green", ls="-.", lw=1)
        plt.axhline(-2, color="green", ls="-.", lw=1)
        plt.axhline(3, color="red", ls="-.", lw=1)
        plt.axhline(-3, color="red", ls="-.", lw=1)
        ax2.set_ylabel("Signal")
        plt.grid()
        plt.legend(loc="upper right")

        # ============================================================
        ax3 = plt.subplot(713, sharex=ax1)
        plt.step(dt_index, pos, label="Position", lw=1)
        max_y = max(pos) * d(1.1)
        min_y = min(pos) * d(1.1)
        plt.axhline(0, color="black", ls="-.", lw=1)
        plt.axhline(max_y, color="green", ls="-.", lw=1)
        plt.axhline(min_y, color="green", ls="-.", lw=1)
        ax3.set_ylabel("position")
        plt.grid()
        plt.legend(loc="upper right")

        # ============================================================
        ax4 = plt.subplot(714, sharex=ax1)

        #plt.yscale("log")
        plt.step(dt_index, cash_asset, label="cash + asset", lw=1)
        plt.axhline(params.trade_args.get("init_trade_cash"), color="black", ls="-.", lw=1)
        ax4.set_ylabel("cash_asset in log")
        plt.grid()
        plt.legend(loc="best")

        # ============================================================
        ax5 = plt.subplot(715, sharex=ax1)
        plt.step(dt_index, shares, label="green", lw=1)
        plt.axhline(0, color="black", ls="-.", lw=1)

        ax5.set_ylabel(f"shares")
        plt.grid()
        plt.legend(loc="upper right")

        # ============================================================
        ax6 = plt.subplot(716, sharex=ax1)

        # Plot buy_ind_alfa on the left y-axis
        ax6.step(dt_index, buy_ind_alfa, label="buy alfa", color="blue", lw=1)
        ax6.set_ylabel("Buy Alfa", color="blue")
        ax6.tick_params(axis="y", labelcolor="blue")
        min_alfa, max_alfa = min(buy_ind_alfa), max(buy_ind_alfa)
        min_alfa, max_alfa = min_alfa-abs(min_alfa*1.1), max_alfa + max_alfa*1.1
        ax6.set_ylim([min_alfa, max_alfa])  # Adjust the range as needed

        # Create a twin axis for sell_ind_alfa on the right side
        ax6_twin = ax6.twinx()
        ax6_twin.step(dt_index, sell_ind_alfa, label="sell alfa", color="red", ls="-.", lw=1)
        ax6_twin.set_ylabel("Sell Alfa", color="red")
        ax6_twin.tick_params(axis="y", labelcolor="red")
        min_alfa, max_alfa = min(sell_ind_alfa), max(sell_ind_alfa)
        min_alfa, max_alfa = min_alfa-abs(min_alfa*1.1), max_alfa + max_alfa*1.1
        ax6_twin.set_ylim([min(sell_ind_alfa), max(sell_ind_alfa)])  # Adjust the range as needed

        # Add a horizontal line at y=0
        ax6.axhline(0, color="black", ls="-.", lw=1)

        # Set grid for both axes
        ax6.grid(True)
        ax6_twin.grid(True)

        # Create a combined legend
        lines1, labels1 = ax6.get_legend_handles_labels()
        lines2, labels2 = ax6_twin.get_legend_handles_labels()
        ax6.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        # ============================================================
        ax7 = plt.subplot(717, sharex=ax1)
        plt.plot(dt_index, buy_ind_deltas , label="buy delta", color="green", ls="-.", lw=1)
        plt.plot(dt_index, sell_ind_deltas, label="sell delta", color="orange", ls="-.", lw=1)
        plt.axhline(0, color="black", ls="-.", lw=1)

        ax7.set_ylabel(f"KF Delta")
        plt.grid()
        plt.legend(loc="upper right")

        if params.plot_args.get("save_img") and params.gain_pct > 30 and hash_tag is not None:
            # logger.info("=================================================")
            img_path = f"img/RunSim_{params.ds.symbol}_gain{params.gain_pct:.0f}_{hash_tag}.png"
            # logger.info(f"hash_tag:{hash_tag}, gain_pct:{params.gain_pct}, img_path:{img_path}, tech_args:{params.tech_args}, trade_args: {params.trade_args}")

            plt.savefig(f"{project_root}/{img_path}", bbox_inches="tight", dpi=300)

        if params.plot_args.get("plot_show") :
            plt.show()

        lock.release()

    except Exception as e:
        err = readable_error(e, __file__)
        _logger.error(err)


def run_ind(form_start, _trade_args, _tech_args, _logger):
    """
    Strategy 標準函式：產生出 RL 訓練用 indicator 資料
    """
    try:
        start_time = time.time()
        _logger.info(f"[Double_KF] Double_KF.Strategy.gen_data.trade_args:{pretty_dict(_trade_args)}")

        app_env = AppEnv(_trade_args.get("app_env"))
        price_env = PriceEnv(_trade_args.get("price_env"))
        symbol = _trade_args.get("symbol")
        _trade_args["form_start"] = form_start
        _trade_args["form_end"] = None
        long_short = _trade_args.get("long_short")

        # Load price
        from api.PriceFetcher import PriceFetcherTrain
        price_fetcher = PriceFetcherTrain(_trade_args, price_env, _logger,
                                          catchup_price=False)

        # Strategy
        strategy = Strategy(app_env, _trade_args, _tech_args, _logger)

        dt_idx_list, prices_list, buy_tracer_list, buy_alfa_list, sell_tracer_list, sell_alfa_list, buy_sd_list, sell_delta_list  = [], [], [], [], [], [], [], []

        # 開始
        idx = 0
        while True:
            # 拿到新價格
            dt_idx, price_new = price_fetcher.get_price(idx)
            if dt_idx is None and price_new is None:
                break

            dt_idx = dt_idx.astype(dt.datetime)
            if idx == 0:
                prices = [price_new[0], price_new[0]]
            else:
                prices = [prices_list[-1], price_new[0]]

            # 交易規則模型
            buy_ind, sell_ind, buy_simple, sell_simple = strategy.step_ind(prices,
                                                                           dt_idx,
                                                                           strategy.buy_delta,
                                                                           strategy.sell_delta,
                                                                           long_short)

            dt_idx_list.append(dt_idx)
            prices_list.append(prices[1])
            buy_tracer_list.append(buy_ind.tracer[1])
            buy_alfa_list.append(buy_ind.alfa_norm_100[1])
            sell_tracer_list.append(sell_ind.tracer[1])
            sell_alfa_list.append(sell_ind.alfa_norm_100[1])
            buy_sd_list.append(buy_ind.sd[1])
            sell_delta_list.append(sell_ind.delta[1])

            idx += 1
            if idx % 1000 == 0:
                msg = f"\033[K[Strategy] Training at {idx}                                                         \r"
                sys.stdout.write(msg)
                sys.stdout.flush()

        # END while
        data_dict = {
            "date": dt_idx_list,
            "open": prices_list,
            "buy_tracer": buy_tracer_list,
            "buy_alfa": buy_alfa_list,
            "sell_tracer": sell_tracer_list,
            "sell_alfa": sell_alfa_list,
            "buy_sd": buy_sd_list,
            "sell_delta": sell_delta_list
        }
        tech_df = pd.DataFrame(data_dict)
        tech_df.set_index("date", inplace=True)
        end_time = time.time() - start_time
        duration_str = str(dt.timedelta(seconds=end_time))

        _logger.info(f"[{symbol}] gen_data took {duration_str[:-5]} to complete")

        # read_write.save_tech(symbol, trade_args, tech_args, tech_df,
        #                      append=is_append, dest_dir=dest_dir)
        return tech_df

    except Exception as e:
        err_str = readable_error(e, __file__)
        _logger.error(err_str)

def Trade_Args_Setup(exch_mode, _logger):
    env_name = "BrunhildEnv-v11"

    target_asset = "BTC"
    home_asset = "USDT"
    symbol = "BTCUSDT"

    # get cmd args
    from Trade.Binance.BinanceTrade import load_args
    # ORIGINAL from Trade.Binance.parse_arguments import train_parse_arguments
    from ahf.utils.init_loader import train_parse_arguments

    # =======================================
    # 取得替代參數的 cmd args
    # =======================================
    cmd_args = vars(train_parse_arguments(
        ["--env_name", "BrunhildEnv-v11",
         "--exch_mode", exch_mode,
         "--agent_id", "0",
         "--init_trade_cash", "10000",
         "--init_target_cash", "0.01",
         "--app_env", "TRAIN",
         "--tech_id", "double_kf",
         "--trade_args_path", "./trade_args/BTCUSDT_DUAL.json",
         "--reset_trade_cash", "0",
         "--job_id", "LOCAL_TEST",
         "--tech_data_path", "./appData/trainData_crypto/diewalkure_v1.parquet",
         "--done_kelly_mode", "false"
         ]))

    exch_mode, hyper_args, _trade_args, _tech_args = load_args(env_name, cmd_args)

    # 檢查 trade_model 所需 trade_args
    order_trade_args_checker(_trade_args)

    # =======================================
    # technical indicators
    # =======================================
    # tech_file_name = f"{_trade_args["tech_id"]}.json"
    # tech_args = get_tech_args(tech_file_name, logger=_logger)

    # 檢查萬一豬腦
    assert symbol == _trade_args.get("symbol"), f"symbol {symbol} 與 trade_args 裡面的 symbol {_trade_args.get('symbol')} 不符合"
    assert env_name == _trade_args.get("env_name"), f"env_name {env_name} 與 trade_args 裡面的 env_name {_trade_args.get('env_name')} 不符合"
    _h = symbol.replace(_trade_args.get("target_asset"), "")
    assert home_asset == _h, f"home_asset {home_asset} 與 trade_args 裡面的 home_asset {_h} 不符合"
    assert target_asset == _trade_args.get("target_asset"), f"target_asset {target_asset} 與 trade_args 裡面的 target_asset {_trade_args.get('target_asset')} 不符合"

    return _trade_args, _tech_args


def manual_load_param():
    _trade_args, _tech_args = Trade_Args_Setup(_exch_mode, logger)
    _trade_args["app_env"] = AppEnv.TRAIN
    _trade_args["long_short"] = "dual"  # long | short | dual
    _trade_args["kelly_cap_enabled"] = False
    _trade_args["min_kelly_cap"] = False
    _tech_args["entry_rule"] = None
    _tech_args["exit_rule"] = "sell_tracer"  # macd_cross | ind_cross | macd_long_rsi_exit
    if (_tech_args["exit_rule"] == "sell_tracer"
            and _trade_args["long_short"] == "dual"):
        """
        result: 
        """
        # ======== trade_args =======
        _trade_args["trade_interval"] = "100T"  # 5T, 10T  # 似乎很吃 trade_interval, 不然過多 noise
        _trade_args["spot_margin"] = "margin"  # spot | margin

        # ========= tech_args =======
        # Controlling parameters
        _tech_args["long_level"] = 0.4
        _tech_args["short_level"] = -0.4
        _tech_args["long_exit_level"] = 0.34
        _tech_args["short_exit_level"] = -0.34

        _tech_args["buy_delta"] = 0.5
        _tech_args["sell_delta"] = 0.1

        # Internal parameters
        _tech_args["buy_obs_cov"] = 0.5
        _tech_args["buy_sd_delta"] = 1e-6
        _tech_args["buy_sd_obs_cov"] = 0.1
        _tech_args["sell_obs_cov"] = 0.5
        _tech_args["sell_sd_delta"] = 1e-6
        _tech_args["sell_sd_obs_cov"] = 0.1

        _tech_args["buy_delta_limit_upper"] = 0.9999
        _tech_args["buy_delta_limit_lower"] = 0.0001
        _tech_args["sell_delta_limit_upper"] = 0.7


    elif (_tech_args["exit_rule"] == "price_cross"
          and _trade_args["long_short"] == "dual"):
        """
        macd_long_rsi_exit + short
        result: ~??%, 2022-01~2022-10
        """
        # ======== trade_args =======
        _trade_args["trade_interval"] = "100T"  # 5T, 10T  # 似乎很吃 trade_interval, 不然過多 noise
        _trade_args["spot_margin"] = "margin"  # spot | margin

        # ========= tech_args =======
        _tech_args["buy_obs_cov"] = 0.5
        _tech_args["buy_sd_delta"] = 1e-6
        _tech_args["buy_sd_obs_cov"] = 0.1
        _tech_args["sell_obs_cov"] = 0.5
        _tech_args["sell_sd_delta"] = 1e-6
        _tech_args["sell_sd_obs_cov"] = 0.1
        _tech_args["buy_delta"] = 0.999
        _tech_args["sell_delta"] = 0.7
    else:
        raise Exception(f"Unknown combination of exit rule({_tech_args.get('exit_rule')})")
    _tech_args = {"tech_list": ["change", "price_alfa_001", "price_alfa_0001"],
                  "long_level": 4.505020586755329, "short_level": -4.7546665545309095,
                  "long_exit_level": 4.001721907380722, "short_exit_level": -4.560265106422815, "buy_obs_cov": 0.5,
                  "buy_sd_delta": 1e-06, "buy_sd_obs_cov": 0.1, "sell_obs_cov": 0.5, "sell_sd_delta": 1e-06,
                  "sell_sd_obs_cov": 0.1, "buy_delta": 0.7385868393707197, "sell_delta": 0.27401750321104446,
                  "exit_rule": "sell_tracer", "buy_delta_limit_upper": 0.9999, "buy_delta_limit_lower": 1e-09,
                  "sell_delta_limit_upper": 0.9999, "sell_delta_limit_lower": 1e-09,
                  "job_id": "8294009908384eb3ac42e9ddd8dc59c0", "entry_rule": None, "stop_loss_margin": -0.11,
                  "hash_tag": None
                  }
    _trade_args = {"exch_mode": "SpotAPI", "tech_id": "double_kf", "symbol": "BTCUSDT", "exchange": "Binance",
                   "stock_crypto": "crypto", "plot": True, "image": True,
                   "price_data_path": "./appData/trainData_crypto/prices_v3.parquet",
                   "tech_data_path": "./appData/trainData_crypto/diewalkure_v1.parquet",
                   "kf_data_path": "./appData/trainData_crypto/KF_theta", "form_start": "2018-03-01",
                   "form_end": "2023-07-01", "trade_interval": "8h", "interval_base": "1T", "interval_eval": "1h",
                   "gamma": 0.95, "decimal_qty": 2, "decimal_price": 2, "target_return": 999999,
                   "done_total_loss": -0.05, "done_monthly_loss": -0.08, "done_drawdown": -0.2,
                   "stop_loss_margin": -0.1, "take_profit_margin": 0.15, "margin_call_level": 1.2,
                   "max_margin_qty": -1, "min_kelly_cap": False, "max_kelly_cap": 1, "done_kelly_mode": "false",
                   "kelly_cap_enabled": False, "max_risk": 100, "must_trade_max": "6w", "render_mode": "console",
                   "is_render": True, "app_env": AppEnv("TRAIN"), "bot_env": BotEnv("SIMULATION"),
                   "price_env": PriceEnv("TRAIN"),
                   "long_short": "dual", "debug": False, "silo_size": 1, "spot_margin": "spot", "stacking_lookback": 1,
                   "buy_fee_pct": 0.001, "sell_fee_pct": 0.001, "short_fee_pct": 0.001, "cover_fee_pct": 0.001,
                   "stop_loss_pct": 0.05, "init_trade_cash": 10000.0, "trade_add": True, "trade_mode": "simulation",
                   "train_mode": "DEV", "orderId": None, "model_dir": "rl_trained_agents",
                   "env_name": "BrunhildEnv-v11", "algo": "PPO", "last_model_file": None, "target_asset": "BTC",
                   "init_target_cash": 0.01, "folder": "rl_trained_agents", "log_path": "rl_trained_agents/ppo",
                   "job_id": "LOCAL_TEST", "agent_id": 0, "trade_args_path": "./trade_args/BTCUSDT_DUAL.json",
                   "execute_now": False, "exit_now": False, "reset_trade_cash": 0.0, "saas_env": None}

    return _trade_args, _tech_args

def load_pod(_pod_dir: str):
    _trade_args = get_trade_args(f"{_pod_dir}/trade_args.json")
    _tech_args = get_tech_args(f"{_pod_dir}/tech_args.json")

    return _trade_args, _tech_args

if __name__ == "__main__":
    _exch_mode = "SpotAPI"  # data only, no trading
    configure_logger("double_kf_strategy_main", False)


    is_load_pod = True
    pod_dir = f"{get_project_root()}/train/pod/gainpct721_20240910_041605_7616aa59521644caaec33bf5e52930eb"

    if is_load_pod:
        trade_args, tech_args = load_pod(pod_dir)
    else:
        trade_args, tech_args = manual_load_param()

    _plot_args = {"plot_show": True,
                 "save_img": False}

    # trade_args["form_start"] = "2022/01/01"
    inject = BinanceOrder(_exch_mode, logger)
    _eval_re = evaluator(trade_args, tech_args, _plot_args,
                         BinanceOrderInject=inject,
                         _logger=logger,
                         monthly_report=True)
    _gain = _eval_re.get("gain")
    _trade_count = _eval_re.get("trade_count")

    print(f"gain: {_gain:.2f}%, trade_count: {_trade_count}")
