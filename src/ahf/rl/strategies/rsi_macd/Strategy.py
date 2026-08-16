import os
import sys
import math
import time
import json
import numpy as np
from tqdm import tqdm
import datetime as dt
from typing import Deque, Dict, List, Union, Any
from decimal import Decimal
from collections import deque

import pandas as pd
from loguru import logger

from ahf.utils.logger import configure_logger
from ahf.core.enums import AppEnv, PriceEnv
from ahf.rl.envs.TradeEnum import TradeAction

from ahf.utils.utils import d, d_abs, d_negate, DecimalEncoder
from ahf.utils.utils import pretty_dict, readable_error

from ahf.preprocessor.ta.RSIMem import RSIMem
from ahf.preprocessor.ta.MACDMem import MACDMem
from ahf.preprocessor.ta.EMA import EMA

from ahf.rl.envs.BacktestValueNetworkEnv import BacktestValueNetworkEnv
from api.Binance.BinanceOrder import BinanceOrder
from api.Binance.BinanceOrder import DummyOrder
from ahf.rl.train.config import get_tech_args
from ahf.rl.strategies.utils import order_trade_args_checker

page_counter = 0

MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
RSI_TIME_PERIOD = 14


class Strategy:
    # State_dim = trade_state_dim + tech_state_dim
    NAME = "RSI_MACD"
    STATE_TECH_COLS = ["ema", "rsi", "macd_bar_norm"]
    STATE_TECH_DIM = len(STATE_TECH_COLS)
    ALL_TECH_COLS = ["ema", "rsi", "max_rsi", "macd_bar_norm", "macd"]
    ALL_TECH_DIM = len(ALL_TECH_COLS)
    STATE_TECH_COLS_HIGH = [10, 10, 10]
    STATE_TECH_COLS_LOW = [-10, -10, -10]


    def __init__(self, app_env, trade_args, tech_args, _logger):
        """
        策略 class,
        price_hist 如果給了的話，會計算出來
        1. 緩存
        2. 然後儲存下來
        在 連續 訓練時建議使用，不然每次都要計算，雖然 RSI 和 MACD 很快
        """
        self.max_deque = sys.maxsize if os.getenv("NODE_ENV") is None else 500

        self.app_env = app_env
        self.price_env = trade_args.get("price_env")
        self.trade_args = trade_args
        self.tech_args = tech_args
        self.tech_list = tech_args.get("tech_list")
        self.logger = _logger

        assert len(self.ALL_TECH_COLS) == self.ALL_TECH_DIM
        assert len(self.STATE_TECH_COLS) == self.STATE_TECH_DIM
        assert set(self.STATE_TECH_COLS).issubset(set(self.ALL_TECH_COLS))

        # tech normalizer
        self.tech_norm_dict = None
        self.tech_shift_dict = None
        self.tech_type_dict = None

        # EMA
        self.ema_time_period = tech_args.get("ema_time_period")

        # 注入主要操作 RSI
        self.rsi_period = tech_args.get("rsi_period")

        # self.macd_fast_period = tech_args.get("macd_fast_period")
        # self.macd_slow_period = tech_args.get("macd_slow_period")
        # self.macd_signal_period = tech_args.get("macd_signal_period")

        # tech_args
        # RSI LONG/SHORT_LEVEL
        self.rsi_long_level = tech_args.get("rsi_long_level")
        self.rsi_short_level = tech_args.get("rsi_short_level")

        self.rsi_long_exit_level = tech_args.get("rsi_long_exit_level")
        self.rsi_short_exit_level = tech_args.get("rsi_short_exit_level")

        self.rsi_long_loss_level = tech_args.get("rsi_long_loss_level")
        self.rsi_short_loss_level = tech_args.get("rsi_short_loss_level")

        # MACD LONG 進入點 3 連採點
        self.macd_price_multi = int(tech_args.get("macd_price_multi"))
        self.macd_bar_long_level = tech_args.get("macd_bar_long_level")
        self.macd_bar_short_level = tech_args.get("macd_bar_short_level")

        self.entry_rule = tech_args.get("entry_rule")
        self.exit_rule = tech_args.get("exit_rule")
        self.loss_rule = tech_args.get("loss_rule")

        if tech_args.get("exit_rule") not in ["ind_cross", "macd_cross", "ema_macd_rsi",
                                              "trailing_stop", "trailing_rsi"]:
            raise Exception("Invalid exit rule")

        # ==== indicator signal ====
        self.EMA_Ind = EMA(self.ema_time_period, logger)

        self.RSI_Ind = RSIMem("RSI_MACD",
                              self.trade_args,
                              self.tech_args,
                              logger)

        self.MACD_Ind = MACDMem("RSI_MACD",
                                self.trade_args,
                                self.tech_args,
                                logger)

        self.MACD_Ind.set_lookback(tech_args.get("macd_lookback"))

        self.signal = deque([0], maxlen=self.max_deque)


    def warm_up(self, price_hist: pd.DataFrame):
        """
        給實際交易 BinanceTrade (Bot) 起始值
        """


        # 處理 init
        price_hist["open"] = price_hist["open"].apply(lambda x: Decimal(x))
        self.logger.info(f"> Strategy->warm_up() with data len={len(price_hist)}")
        last_price = None
        ema, rsi, max_rsi, macd_bar_norm, macd = None, None, None, None, None
        for idx, (dt_idx, price) in tqdm(enumerate(price_hist["open"].items())):
            assert isinstance(dt_idx, dt.datetime)
            # dt_idx = dt_idx.astype(dt.datetime)
            if idx == 0:
                price_list = [price, price]
            else:
                price_list = [last_price, price]

            # 交易規則模型
            ema, rsi, max_rsi, macd_bar_norm, macd = self.step_ind(price_list, dt_idx,
                                                                   self.trade_args.get("long_short"))
            last_price = price

            # 列印 stdout
            if idx % 1000 == 0:
                msg = f"\033[K Training at {idx}                                                         \r"
                sys.stdout.write(msg)
                sys.stdout.flush()

        # [END for]
        last_tech_ary = ema[1], rsi[1], max_rsi, macd_bar_norm[1], macd[1]

        return last_tech_ary

    def get_tech_list(self):
        return self.tech_list

    def reset(self):

        # 訓練時資料先 load 進來
        self.signal = deque([0], maxlen=self.max_deque)

        self.EMA_Ind = EMA(self.ema_time_period, self.logger)

        self.RSI_Ind = RSIMem("RSI_MACD",
                              self.trade_args,
                              self.tech_args,
                              self.logger)

        self.MACD_Ind = MACDMem("RSI_MACD",
                                self.trade_args,
                                self.tech_args,
                                self.logger)
        self.MACD_Ind.set_lookback(self.tech_args.get("macd_lookback"))

        self.populate_int_data()

    def populate_int_data(self):
        if self.app_env == AppEnv.TRAIN:
            pass

    def step_ind(self, price_list: List[Decimal], dt_idx: dt.datetime, long_short: str):
        """
        這個可以被單獨拿來使用，如果你沒有要拿來看 signal, 只是要看 indicator 的
        改變狀況（如：用來 RL 訓練時 state 的使用），就可以用這個。
        Parameters
        ----------
        price_list
        dt_idx
        long_short
        """
        price = price_list[1]
        # update indicators
        ema_list = self.EMA_Ind.add_one(price, dt_idx).get("ema")
        rsi_list = self.RSI_Ind.add_one(price, dt_idx).get("rsi")
        if long_short == "long":
            max_rsi = max([abs(self.RSI_Ind.rsi[i]) for i in range(-1, -13, -1)]) if len(self.RSI_Ind.rsi) > 12 else 0
        else:
            max_rsi = min([abs(self.RSI_Ind.rsi[i]) for i in range(-1, -13, -1)]) if len(self.RSI_Ind.rsi) > 12 else 1.2

        macd_re = self.MACD_Ind.add_one(price * self.macd_price_multi, dt_idx)
        macd_bar = macd_re.get("macd_bar")
        macd_list = macd_re.get("macd")
        macd_bar_norm_list = [v / float(price_list[0]) for v in macd_bar]
        # print(f"rsi: {rsi[1]:.2f}, macd_bar_norm:{macd_bar_norm[1]:.4f}")

        return ema_list, rsi_list, max_rsi, macd_bar_norm_list, macd_list

    def step(self,
             price_list: List[Decimal],
             dt_idx: dt.datetime,
             ds: Any,
             long_short: str):
        """
        Step strategy 來取得 signal 和執行的訊號，主要交易邏輯
        在外部自己執行就放 order=None

        Parameters
        ----------
        price_list: is the last price and the current price in list format
        dt_idx: datetime index
        ds: Datasource container
        long_short: long or short
        order_min_qty: DEPRECATED 最少要買多少 qty => 測試：BacktestOrder， 實戰：BinanceOrder
        """
        try:
            if not isinstance(price_list, (np.ndarray, list)):
                raise Exception("Invalid prices type, it must be a list of size 2")

            if dt_idx is None:
                raise Exception("TradingStrategy.step => dt_idx cannot be None")

            if not isinstance(dt_idx, dt.datetime):
                raise Exception(f"dt_idx must be dt.datetime but got {type(dt_idx)}")

            # last_position = ds.get_last_position()

            # 檢查太少
            # in_position = (last_position != 0)
            # if in_position and d_abs(last_position) < order_min_qty:
            #     in_position = False

            # 檢查太少 => 沒關係，這只是初步給訊號，實際交易還要看 silo, 這是上層的工作， strategy 就是決定基本邏輯
            in_position: int = ds.is_significant_pos()

            idx = ds.get_idx()

            # update indicators
            ema, rsi, max_rsi, macd_bar_norm, macd = self.step_ind(price_list, dt_idx, long_short)
            tech_ary = ema[1], rsi[1], max_rsi, macd_bar_norm[1], macd[1]

            # 一開始會所有的值太大而成交，把他擋住
            signal = self.cal_signal_run(price_list, ema, rsi, max_rsi, macd_bar_norm, macd, in_position, long_short, ds) if idx >= 1 else 0
            self.signal.append(signal)

            trade_action_new = self.cal_trade_action(signal, in_position)


            trade_action_new_name = "-" if trade_action_new.name == "HOLD" else trade_action_new.name

            ema_norm = (price_list[1]- d(ema[1]))/price_list[1]* d(100)
            self.logger.debug(f"[Strategy] step result:\n"
                             f"  ema_norm: {ema_norm:>6,.1f} | ema: {ema[1]:>6,.5f},\n"
                             f"  rsi[]: [{rsi[0]:>6,.5f}, {rsi[1]:>6,.5f}],\n"
                             f"  max RSI: {max_rsi:>6,.5f},\n"
                             f"  macd_bar_norm[]: [{macd_bar_norm[0]:>6,.5f}, {macd_bar_norm[1]:>6,.5f}],\n"
                             f"  macd[]: [{macd[0]:>8,.5f}, {macd[1]:>8,.5f}],\n"
                             f"  signal: {signal:>3},\n"
                             f"  act: {trade_action_new_name}\n")

            return trade_action_new, tech_ary

        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)

    def cal_signal_run(self,
                       price_list: List[Decimal],
                       ema_list: List[float],
                       rsi_list: List[float],
                       max_rsi: float,
                       macd_bar_list: List[float],
                       macd_list: List[float],
                       in_position: int,
                       long_short: str,
                       ds=None):
        """in_position: 1=LONG, -1=SHORTED, 0=NO POSITION"""

        try:
            self.logger.debug(f"[Strategy] cal_signal_run:\n"
                              f"  price_list: {price_list},"
                              f"  ema_list: {ema_list}, \n"
                              f"  rsi_list: {rsi_list}, "
                              f"  max_rsi: {max_rsi}, \n"
                              f"  macd_bar_list: {macd_bar_list}, \n"
                              f"  macd_list:{macd_list}, "
                              f"  in_position:{in_position}, \n"
                              f"  long_short: {long_short}, \n"
                              f"  entry_rule: {self.entry_rule}")
            self.logger.debug(f"[Strategy] cal_signal_run:\n"
                              f"  macd_bar_long_level: {self.macd_bar_long_level}, \n"
                              f"  rsi_long_level: {self.rsi_long_level}\n")

            self.logger.debug(
                f"[Strategy] cal_signal_run:\n"
                f"  price_list: {price_list},"
                f"  ema_list: {ema_list}, \n"
                f"  rsi_list: {rsi_list}, "
                f"  max_rsi: {max_rsi}, \n"
                f"  macd_bar_list: {macd_bar_list}, \n"
                f"  macd_list:{macd_list}, "
                f"  in_position:{in_position}, \n"
                f"  long_short: {long_short}, \n"
                f"  entry_rule: {self.entry_rule}")
            self.logger.debug(
                f"[Strategy] cal_signal_run:\n"
                f"  macd_bar_long_level: {self.macd_bar_long_level}, \n"
                f"  rsi_long_level: {self.rsi_long_level}\n")

            signal = 0


            if long_short in ["long", "dual"]:
                # LOSS EXIT RULE
                if self.loss_rule == "ema_macd_rsi":
                    if in_position == 1 and rsi_list[1] <= self.rsi_long_loss_level:
                        self.logger.debug("[Strategy] Step long C")
                        signal = 3

                # EXIT RULE
                if self.exit_rule == "ema_macd_rsi":
                    if max_rsi > 0.8:
                        a = 1
                    if in_position == 1 and max_rsi * 0.9 >= rsi_list[
                        1]:  # max_rsi * self.rsi_long_exit_level >= rsi_list[1]:
                        signal = 2

                elif self.exit_rule == "trailing_rsi":
                    if signal == 0:
                        if in_position == 1 and rsi_list[1] <= self.rsi_long_exit_level:
                            signal = 2
                            self.logger.debug("[Strategy] step long D "
                                              f"long_short={long_short}"
                                              f"rsi_list[1] ({rsi_list[1]}) <= "
                                              f"self.rsi_long_exit_level ({self.rsi_long_exit_level})")

                elif self.exit_rule == "trailing_stop":
                    if signal == 0:
                        trailing_pnl = float(max(ds.get_range_paper_pnl_pct(20))) if long_short == "long" else float(
                            min(ds.get_range_paper_pnl_pct(20)))
                        current_pnl = float(ds.get_last_paper_pnl_pct())
                        if in_position == 1 and 0 < current_pnl <= trailing_pnl * 0.92:
                            signal = 2

                elif self.exit_rule == "ind_cross":
                    if in_position == 1 and rsi_list[0] > self.rsi_long_loss_level >= rsi_list[1]:
                        signal = 3
                    elif in_position == 1 and macd_list[0] > self.rsi_long_exit_level >= macd_list[1]:
                        signal = 2

                elif self.exit_rule == "macd_cross":
                    if in_position == 1 and rsi_list[0] > self.rsi_long_loss_level >= rsi_list[1]:
                        signal = 3
                    elif in_position == 1 and macd_list[0] > self.rsi_long_exit_level >= macd_list[1]:
                        signal = 2

                else:
                    raise Exception("Invalid exit_rule:{0}".format(self.exit_rule))

                # ENTRY RULE
                if self.entry_rule == "ema_macd_rsi" and in_position == 0:
                    self.logger.debug("[Strategy] step long A")
    
                    if price_list[1] > ema_list[1]:
                        if (macd_bar_list[1] > self.macd_bar_long_level >= macd_bar_list[0] or
                                (self.macd_bar_long_level <= macd_bar_list[1] <= self.macd_bar_long_level * 1.2)):  # 直接過頭
                            self.logger.debug("[Strategy] Step long B")
                            if rsi_list[1] >= self.rsi_long_level >= rsi_list[0]:
                                self.logger.debug("[Strategy] Step long BB")
                                signal = 1
                                return signal


            # [if long_short in ["long", "dual"]


            if long_short in ["short", "dual"] and signal == 0:
                # LOSS EXIT RULE
                if self.loss_rule == "ema_macd_rsi":
                    if long_short == "short" and in_position == -1 and rsi_list[1] >= self.rsi_short_loss_level:
                        self.logger.debug("[Strategy] step short C: "
                                          f"rsi_list[1] ({rsi_list[1]}) <= "
                                          f"self.rsi_short_loss_level ({self.rsi_short_loss_level})")
                        signal = -3

                # EXIT RULE
                if self.exit_rule == "ema_macd_rsi":
                    if max_rsi > 0.8:
                        a = 1
                    if long_short == "long" and in_position and max_rsi * 0.9 >= rsi_list[
                        1]:  # max_rsi * self.rsi_long_exit_level >= rsi_list[1]:
                        signal = 2
                    elif long_short == "short" and in_position and max_rsi * 1.1 <= rsi_list[
                        1]:  # max_rsi * self.rsi_short_exit_level <= rsi_list[1]:
                        signal = -2

                elif self.exit_rule == "trailing_rsi":
                    if signal == 0:
                        if (long_short == "long" and in_position and
                                rsi_list[1] <= self.rsi_long_exit_level):
                            signal = 2
                            self.logger.debug("[Strategy] step short D "
                                              f"long_short={long_short}"
                                              f"rsi_list[1] ({rsi_list[1]}) <= "
                                              f"self.rsi_long_exit_level ({self.rsi_long_exit_level})")

                        elif (long_short == "short" and in_position and
                              rsi_list[1] >= self.rsi_short_exit_level):
                            signal = -2
                            self.logger.debug("[Strategy] step short D "
                                              f"long_short={long_short}"
                                              f"rsi_list[1] ({rsi_list[1]}) <= "
                                              f"self.rsi_long_exit_level ({self.rsi_short_exit_level})")

                elif self.exit_rule == "trailing_stop":
                    if signal == 0:
                        trailing_pnl = float(max(ds.get_range_paper_pnl_pct(20))) if long_short == "long" else float(
                            min(ds.get_range_paper_pnl_pct(20)))
                        current_pnl = float(ds.get_last_paper_pnl_pct())
                        if long_short == "long" and in_position and 0 < current_pnl <= trailing_pnl * 0.92:
                            signal = 2
                        elif (long_short == "short" and in_position and current_pnl > 0 and
                              trailing_pnl * 1.08 <= current_pnl):
                            signal = -2

                elif self.exit_rule == "ind_cross":
                    if long_short == "long" and not in_position and rsi_list[0] < self.rsi_long_level <= rsi_list[1]:
                        if self.macd_bar_long_level >= macd_bar_list[1] > macd_bar_list[0]:
                            signal = 1
                    elif long_short == "short" and not in_position and rsi_list[0] > self.rsi_short_level >= rsi_list[
                        1]:
                        if self.macd_bar_short_level <= macd_bar_list[1] <= macd_bar_list[0]:
                            signal = -1
                    elif long_short == "long" and in_position and rsi_list[0] > self.rsi_long_loss_level >= rsi_list[1]:
                        signal = 3
                    elif long_short == "short" and in_position and rsi_list[0] < self.rsi_short_loss_level <= rsi_list[
                        1]:
                        signal = -3
                    elif long_short == "long" and in_position and macd_list[0] > self.rsi_long_exit_level >= macd_list[
                        1]:
                        signal = 2
                    elif long_short == "short" and in_position and macd_list[0] < self.rsi_short_exit_level <= \
                            macd_list[1]:
                        signal = -2

                elif self.exit_rule == "macd_cross":
                    if long_short == "long" and not in_position and rsi_list[0] < self.rsi_long_level <= rsi_list[1]:
                        if self.macd_bar_long_level >= macd_list[1] > macd_list[0]:
                            signal = 1
                    elif long_short == "short" and not in_position and rsi_list[0] > self.rsi_short_level >= rsi_list[
                        1]:
                        if self.macd_bar_short_level <= macd_list[1] <= macd_list[0]:
                            signal = -1
                    elif long_short == "long" and in_position and rsi_list[0] > self.rsi_long_loss_level >= rsi_list[1]:
                        signal = 3
                    elif long_short == "short" and in_position and rsi_list[0] < self.rsi_short_loss_level <= rsi_list[
                        1]:
                        signal = -3
                    elif long_short == "long" and in_position and macd_list[0] > self.rsi_long_exit_level >= macd_list[
                        1]:
                        signal = 2
                    elif long_short == "short" and in_position and macd_list[0] < self.rsi_short_exit_level <= \
                            macd_list[1]:
                        signal = -2

                else:
                    raise Exception("Invalid exit_rule:{0}".format(self.exit_rule))


                # ENTRY RULE
                if self.entry_rule == "ema_macd_rsi" and in_position == 0 and signal == 0:
                    self.logger.debug("[Strategy] step short A")

                    if price_list[1] < ema_list[1]:
                        if (macd_bar_list[1] < self.macd_bar_short_level <= macd_bar_list[0] or
                                macd_bar_list[1] < self.macd_bar_short_level):  # 直接過頭
                            if rsi_list[1] <= self.rsi_short_level:
                                signal = -1
                                return signal


            # [if long_short in ["short", "dual"]

            return signal
        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)
            raise Exception(e)

    def cal_trade_action(self, signal, in_position):
        try:
            if signal == 1 and not in_position:  # position_signal == 0
                trade_action = TradeAction.BUY
            elif signal == 2 and in_position:  # position_signal == 1
                trade_action = TradeAction.SELL
            elif signal == -1 and not in_position:  # position_signal == 0
                trade_action = TradeAction.SHORT
            elif signal == -2 and in_position:  # position_signal == -1
                trade_action = TradeAction.COVER
            elif signal == 3 and in_position:
                trade_action = TradeAction.SELL
            elif signal == -3 and in_position:
                trade_action = TradeAction.COVER
            else:
                trade_action = TradeAction.HOLD

            return trade_action
        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)
            raise Exception(e)

    def cal_reward(self, ds, realized_pnl_pct, reward_per_step) -> float:
        """
        :param ds: 資料源
        :param realized_pnl_pct:
        :param reward_per_step: reward for long-running
        """
        try:
            idx = ds.get_idx()

            # period = int(convert_to_min("5d") / convert_to_min(ds.trade_args["trade_interval"]))
            # period_start = max(idx - period, 0)
            # period_end = idx + 1

            # reward long-running
            step_reward = reward_per_step

            # kelly cap
            min_kelly_cap = ds.trade_args["min_kelly_cap"]
            kelly_cap = ds.get_kelly_cap()
            if kelly_cap < min_kelly_cap:  # and ds.trade_args["long_short"] == "long":
                step_reward -= 0.1

            # if kelly_cap > -min_kelly_cap and ds.trade_args["long_short"] == "short":
            #     step_reward -= 0.1

            # penalty for trading less than 2 times a week
            # num_trades = ds.get_num_trades(period_start, period_end)
            # if num_trades <= 2 and (period_end - period_start > period):
            #    step_reward -= 0.005

            # reward buy / short
            if ds.get_buysell() == TradeAction.BUY or ds.get_buysell() == TradeAction.SHORT:
                step_reward += 0.3

            # how many profitable trade for the past 10 trades
            # penalize if past 10 trades are bad, it is like hitting a wall
            # if ds.get_position() == d(0) and ds.get_num_profit_trades() <= 4 and idx > 10:
            #    step_reward -= 0.5

            # penalty for big drawdown
            paper_gain_pct = float(ds.get_paper_pnl_pct())
            stop_loss_margin = float(ds.trade_args.get("stop_loss_margin"))

            # if stop_loss_margin > 0:
            #     raise Exception(f"cal_reward.stop_loss_margin must be less than zero got {stop_loss_margin}")

            # drawdown
            if paper_gain_pct < stop_loss_margin and ds.is_significant_pos_now():
                step_reward += paper_gain_pct  # paper_gain_pct is negative, so we use plus sign

            # penalty if paper pnl too high
            # take_profit_margin = float(ds.trade_args.get("take_profit_margin", 0.2))
            # if take_profit_margin < 0:
            #    raise Exception(f"cal_reward.take_profit_margin must be larger than zero got {take_profit_margin}")

            # if paper_gain_pct > take_profit_margin:
            #     # print(f"paper_gain_pct: {paper_gain_pct}, take_profit_margin: {take_profit_margin}")
            #     step_reward -= 0.005  # paper_gain_pct is positive, so we use minus sign

            # realized pnl and cash asset
            # amt_bought = ds.silo.get_total_amt_bought()
            # cash_asset_mean = np.mean(self.cash_asset[period_start:period_end])

            # Realized Percentage
            a = float(realized_pnl_pct) * 100 * 300
            step_reward += a if realized_pnl_pct > 0 else a * 0.8

            c = math.sqrt(math.log(ds.num_trades + 1) / math.log(ds.get_idx() + 1)) if ds.get_idx() > 10 and realized_pnl_pct > 0 else 0
            cum_pct = float(ds.get_cumulative_realized_pnl()/ds.get_cash_asset()) * 100
            if step_reward > 0:
                total_reward = (c * cum_pct) + step_reward
            else:
                total_reward = step_reward

            if float(total_reward) > 3:
                k = 1
            return float(total_reward)
        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)
            time.sleep(3)
            sys.exit()


    # def populate_init(self, dt_idx, price, position, buysell, executed_qty, executed_amt,
    #                   fee1, fee2, paper_pnl, cash, borrowed_cash, target_cash, asset, cash_asset,
    #                   realized_pnl, drawdown, cumulative_realized_pnl,
    #                   total_position, total_amt_bought,
    #                   orderId):
    #     """
    #     取得第一筆資料
    #     """
    #     # trade_data
    #     self._dt_idx[0] = dt_idx
    #
    #     self.price_ary[0] = [d(price)]
    #     self._position[0] = position
    #     self._buysell[0] = buysell
    #     self._asset[0] = asset
    #     self._cash[0] = cash
    #     self._borrowed_cash[0] = borrowed_cash
    #     self._cash_asset[0] = cash_asset
    #     self._realized_pnl[0] = realized_pnl
    #
    #     self._kelly_cap[0] = kelly_cap
    #
    #     self._cumulative_realized_pnl[0] = cumulative_realized_pnl
    #     self._drawdown[0] = drawdown
    #
    #     self._target_cash[0] = target_cash
    #
    #     self._executed_qty[0] = executed_qty
    #     self._executed_amt[0] = executed_amt
    #     self._fee1[0] = fee1
    #     self._fee2[0] = fee2
    #
    #     self._paper_pnl[0] = paper_pnl
    #     self._total_position[0] = total_position
    #     self._total_amt_bought[0] = total_amt_bought
    #
    #     self._orderId[0] = orderId
    #
    #     # ==== Data Source ====
    #     # _dt_idx = dt_idx.replace(tzinfo=dt.timezone.utc)
    #     # self.set_price(0, _dt_idx, price)
    #     self.set_trade(buysell, position, cash, borrowed_cash, asset)
    #     self.set_pnl(d(0.), d(0.), cumulative_realized_pnl)
    #
    #     """
    #     # no need update
    #     self.KellyCls[0] = kelly_cap
    #     self.KellyCls_short[0] = kelly_cap_short
    #     self.silo[0] = silo
    #     self.silo_short[0] = silo_short
    #
    #     self.set_fee1(0, 0)
    #     self.set_fee2(0, 0)
    #     """
    #
    #     # self._paper_pnl[0] = paper_pnl
    #     # self._total_position[0] = total_position
    #     # self._total_amt_bought[0] = total_amt_bought

    def generate_tech_state_normalizer(self, tech_ary_pd):
        """
        IMPORTANT
        技術指標放在 state, 定義在這，但 normalize_tech_list 執行
        因為我們要做成 step 的方式，這個是用在以前 training 時 全部一起算的方式
        """
        tech_norm_vector = {}
        tech_shift_vector = {}
        tech_type_vector = {}

        cols = list(tech_ary_pd.columns.values)

        self.logger.info(f"Tech col: {cols}")

        # setup parquet schema
        for col in cols:
            if "rsi" in col:
                tech_norm_vector["rsi"] = 1
                tech_shift_vector["rsi"] = -0.5
                tech_type_vector["rsi"] = "state"
            elif "ema" in col:
                tech_norm_vector["ema"] = "divide_price"
                tech_shift_vector["ema"] = "minus_price"
                tech_type_vector["ema"] = "state"
            elif "macd_bar_norm" in col:
                tech_norm_vector["macd_bar_norm"] = 1
                tech_shift_vector["macd_bar_norm"] = 0
                tech_type_vector["macd_bar_norm"] = "state"
            else:
                raise Exception(f"tech normalizer for column '{col}' not found")

        self.tech_norm_dict = tech_norm_vector
        self.tech_shift_dict = tech_shift_vector
        self.tech_type_dict = tech_type_vector
        assert len(self.tech_norm_dict) == len(self.tech_shift_dict) == len(self.tech_type_dict)

    def normalize_tech_list(self, tech_ary, price, _logger):
        try:
            tech_new = []

            if  len(self.ALL_TECH_COLS) != len(tech_ary):
                raise Exception("normalize_tech_list.tech_pd length does not match STATE_TECH_COLS "
                                f"expect {len(self.ALL_TECH_COLS)} got {len(tech_ary)}")

            if isinstance(price, Decimal):
                raise Exception("Strategy.normalize_tech_list.price Please convert from Decimal to float")

            if not isinstance(price, float):
                _logger.error(f"Strategy normalize_tech_list.price has to be of type float but got {type(price)}")

            for (i, col) in enumerate(self.STATE_TECH_COLS):
                # 依照要用的排列出來，但是要從主要的 ALL_TECH_COLS 拉出
                icol = self.ALL_TECH_COLS.index(col)
                if "rsi" == col:
                    col_pd = tech_ary[icol] - 0.5
                    tech_new.append(col_pd)
                elif "macd_bar_norm" == col:
                    tech_new.append(tech_ary[icol])
                elif "ema" == col: # normalized
                    col_pd = (price - tech_ary[icol]) / price * 10
                    tech_new.append(col_pd)
                elif "max_rsi" == col or "macd" == col or "ema" == col:
                    # Do No Nothing, just to make sure everything is considered
                    pass
                else:
                    raise Exception(f"tech feature '{col}' does not exist")

            return tech_new
        except Exception as e:
            err = f"[Strategy] {readable_error(e, __file__)}"
            _logger.error(err)
            time.sleep(3)
            sys.exit()


def _run_sim(trade_args, tech_args,
             plot=False,
             monthly_report=False,
             verbose=True,
             BinanceOrderInject=None):

    configure_logger(f"RSI_MACD_{trade_args.get('symbol')}")

    try:
        exch_mode = trade_args.get("spot_margin")
        app_env = trade_args.get("app_env")
        price_env = trade_args.get("price_env")
        spot_margin = trade_args.get("spot_margin")

        # Using update method, not train method
        exch_api = DummyOrder(exch_mode, logger, BinanceOrderInject=BinanceOrderInject)
        v = BacktestValueNetworkEnv(app_env, price_env, exch_mode, exch_api, spot_margin,
                                    trade_args, tech_args, logger,
                                    verbose=verbose)

        strategy = Strategy(app_env, trade_args, tech_args, logger)

        # 開始
        idx = 0
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
            trade_action_new, tech_ary = strategy.step(prices, dt_idx, v.ds,
                                                       trade_args.get("long_short")
                                                       )

            # 交易
            if trade_action_new == TradeAction.BUY and bool(trade_args.get("kelly_cap_enabled")):
                action = d(max(v.ds.ds_cal_kelly_cap(), float(trade_args.get("min_kelly_cap"))))
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
            if gain < -40:
                break

        # RMSE
        # price_sell_tracer_diff = price_train - sell_indi["tracer"]
        # sell_buy_tracer_diff = sell_indi["tracer"] - buy_indi["tracer"]

        # MSE = mean_squared_error(price_sell_tracer_diff, sell_buy_tracer_diff)
        # RMSE = math.sqrt(MSE)

        global page_counter

        final_pnl = v.ds.get_cash_asset()
        gain = (final_pnl - v.ds.init_trade_cash) / v.ds.init_trade_cash * 100
        logger.info("[Strategy] MACD_RSI Final Result for step, "
                    "init:${0:>9,.3f}, "
                    "pnl:${1:>9,.3f}, "
                    "gain:{2:>8,.3f}%, "
                    "cal count:{3:>3}"
                    "".format(v.ds.init_trade_cash, final_pnl, gain,
                              page_counter))  # , opts:{5}, vars(opts)

        page_counter += 1

        if plot:
            plot_sim(v.ds,
                     strategy.EMA_Ind,
                     strategy.RSI_Ind,
                     strategy.MACD_Ind,
                     strategy.signal,
                     tech_args,
                     trade_args,
                     gain,
                     logger)

        if monthly_report:
            daily_json = gen_monthly_report(v.ds._dt_idx, v.ds._cash_asset, logger)
            print(daily_json)
        return {
            "gain": gain,
            "ds": v.ds
        }
    except Exception as e:
        logger.error(readable_error(e, __file__))
        time.sleep(3)
        sys.exit()

def evaluator(trade_args, tech_args,
              plot=False,
              verbose=True,
              BinanceOrderInject=None,
              monthly_report=False):
    """
    跟 run 不同的是，他是給 XGBoost 或其他來使用的，屬於哪來訓練的，
    可以另外自行帶入參數的改變等
    """

    def sigmoid(x):
        sig = 1 / (1 + math.exp(-x))
        return sig

    eval_re = _run_sim(trade_args, tech_args, plot, monthly_report, verbose,
                       BinanceOrderInject=BinanceOrderInject)
    profit_loss = eval_re.get("gain")
    ds = eval_re.get("ds")

    # loss = 1 - math.tanh(gain / 100)
    loss = 1 - 2 * (sigmoid(profit_loss / 100) - 0.5)

    return {"gain": profit_loss, "loss_fn": loss, "trade_count": ds.num_trade }

def gen_monthly_report(dt_index: Deque[dt.datetime], cash_asset: Deque[Decimal], _logger) -> str:
    """
    Convert high-frequency cash+asset data to daily end-of-day JSON format

    Args:
        dt_index (Deque[datetime]): Deque of datetime timestamps
        cash_asset (Deque[Decimal]): Deque of corresponding cash+asset values in Decimal type
        _logger: logger

    Returns:
        str: JSON string with daily data where dates are formatted as YYYY-MM-DD
    """
    try:
        # Convert deques to lists for easier processing
        dates = list(dt_index)
        values = list(cash_asset)  # No need to convert Decimal objects

        # Create dictionary to store daily data
        daily_data: Dict[str, Decimal] = {}

        # Iterate through the data
        for date, value in zip(dates, values):
            # Convert datetime to date string (YYYY-MM-DD)
            date_str = date.strftime('%Y-%m-%d')
            # Update the value for this date - will automatically keep the last value of the day
            daily_data[date_str] = value

        # Convert to list of dictionaries format for better JSON structure
        result = [
            {"date": date, "value": value}
            for date, value in daily_data.items()
        ]

        # Return JSON string using custom encoder for Decimal
        return json.dumps(result, indent=2, cls=DecimalEncoder)
    except Exception as e:
        err = readable_error(e, __file__)
        _logger.error(err)


def plot_sim(ds, ema_ind, rsi_ind, macd_ind, signal, tech_args, trade_args, gain, _logger):
    import matplotlib.pyplot as plt
    try:
        plt.figure(1, figsize=(15, 9))

        dt_index = list(ds._dt_idx)

        ax1 = plt.subplot(711)
        plt.yscale("log")
        plt.plot(dt_index, np.array(list(ds.price_ary)).ravel(), label="price actual", lw=1)
        plt.plot(dt_index[1:], list(ema_ind.ema), label="EMA", color="g", lw=1)

        ax1.set_ylabel("price")
        plt.grid()
        plt.legend(loc="best")

        plt.title(f"{trade_args.get('tech_id')} TradingStrategy {ds.symbol}, Gain:{gain:.2f}%")

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
        ax2.set_ylabel("signal")
        plt.grid()
        plt.legend(loc="upper right")

        ax3 = plt.subplot(713, sharex=ax1)
        pos = list(ds._position)
        plt.bar(dt_index, pos, label="Position", lw=1)
        max_y = max(pos) * d(1.1)
        min_y = min(pos) * d(1.1)
        plt.axhline(0, color="black", ls="-.", lw=1)
        plt.axhline(max_y, color="green", ls="-.", lw=1)
        plt.axhline(min_y, color="green", ls="-.", lw=1)
        ax3.set_ylabel("position")
        plt.grid()
        plt.legend(loc="upper right")

        ax4 = plt.subplot(714, sharex=ax1)

        cash_asset = [float(item) for item in list(ds._cash_asset)]
        plt.yscale("log")
        plt.step(dt_index, cash_asset, label="cash + asset", lw=1)
        plt.axhline(trade_args.get("init_trade_cash"), color="black", ls="-.", lw=1)
        ax4.set_ylabel("cash_asset in log")
        plt.grid()
        plt.legend(loc="best")

        ax5 = plt.subplot(715, sharex=ax1)
        if trade_args.get("long_short") == "long":
            plt.axhline(tech_args.get("macd_bar_long_level"), color="green", ls="-.", lw=1)
            label = f"macd_bar_norm long:{tech_args.get('macd_bar_long_level')}"
        elif trade_args.get("long_short") == "short":
            plt.axhline(tech_args.get("macd_bar_short_level"), color="green", ls="-.", lw=1)
            label = f"macd_bar_norm short:{tech_args.get('macd_bar_short_level')}"
        else:
            raise Exception(f"wrong trade_args.get('long_short')")

        plt.plot(dt_index[1:], list(macd_ind.macd_bar_norm), label=label, lw=1)
        # plt.plot(dt_index[1:], list(macd_ind.macd), label="macd", color="g", lw=1)
        # plt.plot(dt_index[1:], list(macd_ind.macd_signal), label="signal", color="b", lw=1)

        plt.axhline(0, color="black", ls="-.", lw=1)

        ax5.set_ylabel("MACD bar")
        ax5.set_ylim([-10, 10])
        plt.grid()
        plt.legend(loc="upper left")

        ax6 = plt.subplot(716, sharex=ax1)
        if trade_args.get("long_short") == "long":
            plt.axhline(tech_args.get("rsi_long_level"), color="green", ls="-.", lw=1)
            # plt.axhline(tech_args.get("long_exit_level"), color="green", ls="-.", lw=1)
            label = f"RSI long_lvl:{tech_args.get('rsi_long_level')}"
        elif trade_args.get("long_short") == "short":
            plt.axhline(tech_args.get("rsi_short_level"), color="green", ls="-.", lw=1)
            # plt.axhline(tech_args.get("short_exit_level"), color="green", ls="-.", lw=1)
            label = f"RSI short_lvl:{tech_args.get('rsi_short_level')}"

        plt.plot(dt_index[1:], list(rsi_ind.rsi), label=label, lw=1)
        # is_active = np.where(st["buy_indi"].sd < 0.95, 0, st["buy_indi"].sd)
        # plt.step(st["obv"].dt_index, is_active, label="buy_sd", lw=2)

        plt.axhline(0, color="black", ls="-.", lw=1)

        ax6.set_ylabel(f"RSI")
        plt.grid()
        plt.legend(loc="upper right")

        ax7 = plt.subplot(717, sharex=ax1)
        realized_pnl = np.array(ds._realized_pnl_pct)[1:]
        realized_pnl = np.multiply(realized_pnl, d(100))

        plt.bar(dt_index[1:], realized_pnl, label="realized_pnl", color=np.where(realized_pnl > d(0), "b", "r"), width=5)

        plt.axhline(0, color="black", ls="-.", lw=1)

        ax7.set_ylabel("realized_pnl")
        plt.grid()
        plt.legend(loc="upper right")

        plt.show()

    except Exception as e:
        err = readable_error(e, __file__)
        _logger.error(err)


def run_ind(form_start, trade_args, tech_args, _logger):
    """
    Strategy 標準函式：產生出 RL 訓練用 indicator 資料
    """
    try:
        start_time = time.time()
        _logger.info(f"[RSI_MACD] RSI_MACD.Strategy.gen_data.trade_args:{pretty_dict(trade_args)}")

        app_env = AppEnv(trade_args.get("app_env"))
        price_env = PriceEnv(trade_args.get("price_env"))
        symbol = trade_args.get("symbol")
        trade_args["form_start"] = form_start
        trade_args["form_end"] = None
        long_short = trade_args.get("long_short")

        # Load price
        from api.PriceFetcher import PriceFetcherTrain
        price_fetcher = PriceFetcherTrain(trade_args, price_env, _logger,
                                          catchup_price=False)

        # Strategy
        strategy = Strategy(app_env, trade_args, tech_args, _logger)

        prices_list, ema_list, rsi_list, max_rsi_list, macd_bar_norm_list, macd_list, dt_idx_list = [], [], [], [], [], [], []

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
            ema_pair, rsi_pair, max_rsi, macd_bar_norm_pair, macd_pair = strategy.step_ind(prices, dt_idx, long_short)

            dt_idx_list.append(dt_idx)
            prices_list.append(prices[1])
            ema_list.append(ema_pair[1])
            rsi_list.append(rsi_pair[1])
            max_rsi_list.append(max_rsi)
            macd_bar_norm_list.append(macd_bar_norm_pair[1])
            macd_list.append(macd_pair[1])

            idx += 1
            if idx % 1000 == 0:
                msg = f"\033[K[Strategy] Training at {idx}                                                         \r"
                sys.stdout.write(msg)
                sys.stdout.flush()

        # END while
        data_dict = {
            "date": dt_idx_list,
            "open": prices_list,
            "ema": ema_list,
            "rsi": rsi_list,
            "max_rsi": max_rsi_list,
            "macd_bar_norm": macd_bar_norm_list,
            "macd": macd_list
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
         "--init_trade_cash", "1000",
         "--init_target_cash", "0.005",
         "--app_env", "TRAIN",
         "--tech_id", "rsi_macd",
         "--trade_args_path", "./trade_args/BTCUSDT_LONG.json",
         "--reset_trade_cash", "0",
         "--job_id", "LOCAL_TEST",
         "--tech_data_path", "./appData/trainData_crypto/diewalkure_v1.parquet",
         "--done_kelly_mode", "false"
         ]))

    exch_mode, hyper_args, trade_args, tech_args = load_args(env_name, cmd_args)

    # 檢查 trade_model 所需 trade_args
    order_trade_args_checker(trade_args)

    # =======================================
    # technical indicators
    # =======================================
    # tech_file_name = f"tech_args/{_trade_args["tech_id"]}.json"
    # tech_args = get_tech_args(tech_file_name, logger=_logger)

    # 檢查萬一豬腦
    assert symbol == trade_args.get("symbol"), f"symbol {symbol} 與 trade_args 裡面的 symbol {trade_args.get('symbol')} 不符合"
    assert env_name == trade_args.get("env_name"), f"env_name {env_name} 與 trade_args 裡面的 env_name {trade_args.get('env_name')} 不符合"
    _h = symbol.replace(trade_args.get("target_asset"), "")
    assert home_asset ==  _h, f"home_asset {home_asset} 與 trade_args 裡面的 home_asset {_h} 不符合"
    assert target_asset == trade_args.get("target_asset"), f"target_asset {target_asset} 與 trade_args 裡面的 target_asset {trade_args.get('target_asset')} 不符合"

    return trade_args, tech_args


if __name__ == "__main__":
    _exch_mode = "SpotAPI"  # data only, no trading
    configure_logger(f"RSI_MACD_Strategy_main")

    _trade_args, _tech_args = Trade_Args_Setup(_exch_mode, logger)
    _tech_args["app_env"] = AppEnv.TRAIN

    _tech_args["entry_rule"] = "ema_macd_rsi"
    _tech_args["exit_rule"] = "trailing_rsi"  # macd_cross | ind_cross | macd_long_rsi_exit
    _trade_args["long_short"] = "long"  # long | short

    if (_tech_args["entry_rule"] == "ema_macd_rsi"
            and _tech_args["exit_rule"] == "ema_macd_rsi"
            and _trade_args["long_short"] == "long"):
        """
        ema_macd_rsi + long
        result: ~460%, 2022-01~2022-10
        """
        # ======== trade_args =======
        _trade_args["trade_interval"] = "8h"  # 5T, 10T  # 似乎很吃 trade_interval, 不然過多 noise
        _trade_args["spot_margin"] = "spot"  # spot | margin

        # ========= tech_args =======
        _tech_args["ema_time_period"] = 100  # ema_time_period
        _tech_args["rsi_long_level"] = 0.55  # RSI
        _tech_args["rsi_long_exit_level"] = 0.55  # exit: rsi 比前最近幾最高點的多少% cross down exit
        _tech_args["rsi_long_loss_level"] = 0.5  # loss: rsi 比前期掉多少
        _tech_args["macd_bar_long_level"] = 0.6  # entry: 判斷 macd 趨勢
        _tech_args["macd_lookback"] = 3  # 2, 3, 4
        _tech_args["rsi_time_period"] = 14

    elif (_tech_args["entry_rule"] == "ema_macd_rsi"
          and _tech_args["exit_rule"] == "ema_macd_rsi"
          and _trade_args["long_short"] == "short"):
        """
        macd_long_rsi_exit + short
        result: ~??%, 2022-01~2022-10
        """
        # ======== trade_args =======
        _trade_args["trade_interval"] = "8h"  # 5T, 10T  # 似乎很吃 trade_interval, 不然過多 noise
        _trade_args["spot_margin"] = "margin"  # spot | margin

        # ========= tech_args =======
        _tech_args["ema_time_period"] = 200  # ema_time_period
        _tech_args["rsi_short_level"] = 0.5  # RSI 沒用到
        _tech_args["rsi_short_exit_level"] = 0.1  # exit: rsi 比前最近幾最高點的多少% cross down exit
        _tech_args["rsi_short_loss_level"] = 0.65  #
        _tech_args["macd_bar_short_level"] = -0  # entry: 判斷 macd 趨勢
        _tech_args["macd_lookback"] = 3  # 2, 3, 4
        _tech_args["rsi_time_period"] = 14

    elif (_tech_args["entry_rule"] == "ema_macd_rsi" and
          _tech_args["exit_rule"] == "trailing_stop" and
          _trade_args["long_short"] == "long"):
        """
        ema_macd_rsi + long
        result: ~460%, 2022-01~2022-10
        """
        # ======== trade_args =======
        _trade_args["trade_interval"] = "8h"  # 5T, 10T
        _trade_args["long_short"] = "long"  # long | short
        _trade_args["spot_margin"] = "spot"  # spot | margin

        # ========= tech_args =======
        _tech_args["ema_time_period"] = 150  # ema_time_period
        _tech_args["rsi_long_level"] = 0.55  # RSI
        _tech_args["rsi_long_exit_level"] = 0.55  # exit: rsi 比前最近幾最高點的多少% cross down exit
        _tech_args["rsi_long_loss_level"] = 0.5  # loss: rsi 比前期掉多少
        _tech_args["macd_bar_long_level"] = 0.6  # entry: 判斷 macd 趨勢
        _tech_args["macd_lookback"] = 3  # 2, 3, 4
        _tech_args["rsi_time_period"] = 14

    elif (_tech_args["exit_rule"] == "ema_macd_rsi" and
          _trade_args["long_short"] == "short"):
        # ======== trade_args =======
        _trade_args["trade_interval"] = "770T"  # 5T, 10T
        _trade_args["long_short"] = "long"  # long | short
        _trade_args["spot_margin"] = "spot"  # spot | margin

        # ========= tech_args =======
        _tech_args["ema_time_period"] = 200

        _tech_args["rsi_long_level"] = 0.5  #
        _tech_args["rsi_short_level"] = 0.5  #

        _tech_args["rsi_long_exit_level"] = 0.85  #
        _tech_args["rsi_short_exit_level"] = 0.85  #

        _tech_args["rsi_long_loss_level"] = 0.5  #
        _tech_args["rsi_short_loss_level"] = 0.5  #

        _tech_args["macd_bar_long_level"] = 0.
        _tech_args["macd_bar_short_level"] = 0.

        _tech_args["macd_lookback"] = 3  # 2, 3, 4
        _tech_args["rsi_time_period"] = 14

    elif (_tech_args["entry_rule"] == "ema_macd_rsi" and
          _tech_args["exit_rule"] == "trailing_rsi" and
          _trade_args["long_short"] == "long"):
        """
        ema_macd_rsi + long
        result: ~?%, 2022-01~2022-10
        """
        print(f"long_short: long, entry_rule: {_tech_args['entry_rule']}, exit_rule: {_tech_args['exit_rule']}")

        # ======== trade_args =======
        _trade_args["trade_interval"] = "8h"  # 5T, 10T
        _trade_args["long_short"] = "long"  # long | short
        _trade_args["spot_margin"] = "spot"  # spot | margin

        # ========= tech_args =======
        _tech_args["ema_time_period"] = 100  # ema_time_period
        _tech_args["rsi_long_level"] = 0.45  # RSI
        _tech_args["rsi_long_exit_level"] = 0.50  # exit: rsi 比前最近幾最高點的多少% cross down exit
        _tech_args["rsi_long_loss_level"] = 0.4  # loss: rsi 比前期掉多少
        _tech_args["macd_bar_long_level"] = 0  # entry: 判斷 macd 趨勢
        _tech_args["macd_lookback"] = 3  # 2, 3, 4
        _tech_args["rsi_time_period"] = 48
    else:
        raise Exception(f"Unknown combination of entry rule({_tech_args.get('entry_rule')}) "
                        f"and exit rule({_tech_args.get('exit_rule')})")
    # 暫時不用，太多參數
    # _tech_args["macd_fast_period"] = 12
    # _tech_args["macd_slow_period"] = 26
    # _tech_args["macd_signal_period"] = 9


    inject = BinanceOrder(_exch_mode, logger)
    _eval_re = evaluator(_trade_args, _tech_args,
                         plot=True,
                         BinanceOrderInject=inject,
                         monthly_report=False)
    _gain = _eval_re.get("gain")
    _trade_count = _eval_re.get("trade_count")

    print(f"gain: {_gain:.2f}%, trade_count: {_trade_count}")
