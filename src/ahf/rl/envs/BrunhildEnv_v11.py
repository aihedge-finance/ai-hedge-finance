import sys
import os
import time

import numpy as np
import pandas as pd
import datetime as dt
import numpy.typing as npt
import numpy.random as rd
from typing import List, Tuple, Dict, Any
from collections import deque
from decimal import Decimal
from dask.diagnostics import ProgressBar
from ahf.core.enums import AppEnv, PriceEnv

from ahf.utils.utils import d, pretty_dict, readable_error, normalize_decreasing_values_centered
from ahf.rl.envs.TradeEnum import TradeAction


from ahf.rl.envs.BaseEnv_v11 import BaseEnv_v11

# plot
from ahf.rl.envs.plot_helper_v21 import Tech_Plot, Trade_Plot

# import envs.BrunhildReadWrite as read_write
from ahf.rl.train.utils import init_agent, init_read_write


class BrunhildEnv:
    """
        ### Action Space
        The action is a `ndarray` with shape `(2,)` which can take values `{0, 1}` indicating the direction
        of the fixed force the cart is pushed with.

        | Num | Action                                                  |
        |-----|---------------------------------------------------------|
        | 0   | Increase or decrease sell_delta, or stay the same       |

        **Note**: The buy_delta that is reduced or increased by the applied force is not fixed, and it depends on 'something'
        trade_cash_pac is the percentage of trade_cash that we want to use to trade, if there are 9 win out of 10 trades
        then we use only 90% of the trade_cash

        ### Observation Space
        The observation is a `ndarray` with shape `(3,)` with the values corresponding to the following:

        | Num | Observation           | Min                  | Max                |   example   |
        |-----|-----------------------|----------------------|--------------------|-------------|
        | 0   | Buy_Tracer(price_pct) | -1                   | 1                  |  0.12(=12%) |
        | 1   | Buy_Alfa              | -10                  | 10                 |  0.5        |
        | 2   | Sell_Tracer(price_pct)| -1                   | 1                  |  12 (=12%)  |
        | 3   | Sell_Alfa             | -Inf                 | 10                 |  0.5        |
        | 4   | Sd                    | 0                    | 5                  |  0.5        |
        | 5   | Sell_Delta            | 0.0                  | 0.999              |             |
        | 6   | Position              | -1                   | 1                  | 0,1,-1      |

        BOUNDARY:
        | 1   | SELL_DELTA_LIMIT_UPPER| -1                   | 1                  | 0.01 (=1%)  |
        | 2   | SELL_DELTA_LIMIT_LOWER| -1                   | 1                  | 0.01 (=1%)  |

        CONSTANT:
        | 1   | BUY_DELTA             |  1e-7                | to 1^-             | 0.01 (=1%)  |
        | 2   | SELL_DELTA_CHANGE     | -1                   | 1                  | 0.01 (=1%)  |

        ### Rewards

        ### Starting State
        The bot start at a given buy_delta that we had good result from the past

        ### Episode Termination

        The episode terminates if any one of the following occurs:
        1. Lose total money more than pre-set value, e.g. 10%
        2. Lose money more than 10% in one month
        2. No more data

        ### Arguments

    """
    ACTION_NAMES: List[str] = ("sell_delta", )
    SELL_DELTA_LIMIT_UPPER: float = 0.99999
    SELL_DELTA_LIMIT_LOWER: float = 1e-07
    DEFAULT_SELL_DELTA_CHANGE: float = 0.05

    STATE_TRADE_COLS = ["IN_POS"]
    STATE_TRADE_DIM = len(STATE_TRADE_COLS)
    STATE_TRADE_COLS_HIGH = [1]
    STATE_TRADE_COLS_LOW = [0]

    def __init__(self, hyper_args, env_args, trade_args, tech_args, strategy_cls, logger,
                 price_fetcher=None, exch_api=None, done_enabled=True, rl_mode=None):
        """

        """
        self.max_deque = sys.maxsize if os.getenv("NODE_ENV") is None else 500
        self.hyper_args = hyper_args
        self.env_args = env_args
        self.trade_args = trade_args
        self.tech_args = tech_args
        self.strategy_cls = strategy_cls
        self.strategy = None
        self.price_fetcher = price_fetcher
        self.exch_api = exch_api
        self.done_enabled = done_enabled

        assert price_fetcher is not None, "price_fetcher cannot be None"
        assert exch_api is not None, "exch_api cannot be None"
        assert strategy_cls is not None, "strategy_cls cannot be None"

        # rendering
        self.is_render = trade_args.get("is_render", True)
        self.render_mode = trade_args.get("render_mode", "console")

        self.logger = logger
        try:
            assert "DEFAULT_SELL_DELTA_CHANGE" in self.env_args, "DEFAULT_SELL_DELTA_CHANGE is required for Env"
            assert "SELL_DELTA_LIMIT_UPPER" in self.env_args, "SELL_DELTA_LIMIT_UPPER is required for Env"

            self.SELL_DELTA_LIMIT_UPPER = env_args["SELL_DELTA_LIMIT_UPPER"] or self.SELL_DELTA_LIMIT_UPPER
            self.DEFAULT_SELL_DELTA_CHANGE = env_args["DEFAULT_SELL_DELTA_CHANGE"] or self.DEFAULT_SELL_DELTA_CHANGE

            # move 5% each time
            self.SELL_DELTA_CHANGE = (self.SELL_DELTA_LIMIT_UPPER - self.SELL_DELTA_LIMIT_LOWER) * self.DEFAULT_SELL_DELTA_CHANGE  # default 5% change

            self.obs_high = np.array(
                [
                    1,   # Buy_Tracer e.g.
                    10,  # Buy_Alfa
                    1,   # Sell_Tracer(price_pct)
                    10,  # Sell_Alfa
                    100, # sd
                    self.SELL_DELTA_LIMIT_UPPER,  # Sell_Delta
                    1,  # Position
                ],
                dtype=np.float32,
            )

            self.obs_low = np.array(
                [
                    -1,   # Buy_Tracer
                    -10,  # Buy_Alfa
                    -1,   # Sell_Tracer(price_pct)
                    -10,  # Sell_Alfa
                      0,  # sd
                    self.SELL_DELTA_LIMIT_LOWER,  # Sell_Delta
                    -1,  # Position
                ],
                dtype=np.float32,
            )
            assert self.obs_high.size == self.obs_low.size, "obs_high/low size must match"


            self.action_high = np.array([1], dtype=np.float32)
            self.action_low = np.array([-1], dtype=np.float32)

            assert self.action_high.size == self.action_low.size, "action_high/low size must match"

            # Env agent specific parameter, REQUIRED
            self.env_name = "BrunhildEnv-v11"
            self.if_discrete = False
            self.action_dim = self.action_high.size
            self.state_dim =  self.obs_high.size

            self.reward_per_step = 0
            self.total_reward = 0

            self.max_step = 0
            self.exch_env = None
            self.state = None

            self.env_args = {
                "env_name":  self.env_name,
                "if_discrete": self.if_discrete,
                "action_dim": self.action_dim,
                "state_dim": self.state_dim,
                "DEFAULT_SELL_DELTA_CHANGE": self.DEFAULT_SELL_DELTA_CHANGE,
                "SELL_DELTA_CHANGE": self.SELL_DELTA_CHANGE,
                "SELL_DELTA_LIMIT_UPPER": self.SELL_DELTA_LIMIT_UPPER,
                "SELL_DELTA_LIMIT_LOWER": self.SELL_DELTA_LIMIT_LOWER,
                "OBS_HIGH": self.obs_high,
                "OBS_LOW": self.obs_low,
                "ACTION_HIGH": self.action_high,
                "ACTION_LOW": self.action_low
            }
            self.tech_cols = self.STATE_TRADE_COLS + self.strategy_cls.STATE_TECH_COLS

        except Exception as e:
            self.logger.error(readable_error(e, __file__))

    def render(self, mode: str="console"):
        self.exch_env.render(mode)

    def clip_observation(self, observation):
        # Ensure the observation is a numpy array
        obs = np.array(observation, dtype=np.float32)
        # Truncate the observation
        truncated_obs = np.clip(obs, self.obs_low, self.obs_high)

        return truncated_obs

    def clip_action(self, action):
        # Ensure the action is a numpy array
        action = np.array(action, dtype=np.float32)
        # Truncate the observation
        truncated_action = np.clip(action, self.action_low, self.action_high)

        return truncated_action

    def _init_tech_data(self):
        """load and check everything is right"""

        try:
            if self.exch_env.price_env == PriceEnv.TRAIN:
                """
                嘗試把它改成 step 的方式， 不然一件事做很多次很白癡
                # 已移到 super() 去執行
                # self._dt_idx = self._dt_idx_init
                # self.price_ary = self.price_ary_init
    
                """
                dt_idx, price_ary, tech_pd = self.exch_env.read_write.load_data(self.trade_args, self.tech_args, self.logger)

                # make sure columns match
                if set(tech_pd.columns.values.tolist()) != set(self.strategy.ALL_TECH_COLS):
                    raise Exception(f"column expect {self.strategy.ALL_TECH_COLS} got {tech_pd.columns.values.tolist()}")

                # EXAMPLE ONLY
                # self.strategy.generate_tech_state_normalizer(tech_pd)

                # IMPORTANT ORIGINAL generic method
                # tech_list = self.tech_args.get("tech_list")
                # tech_ary_pd = self.build_tech_feature(tech_list, price_ary, tech_pd)

                if len(tech_pd.index) == 0:
                    raise Exception('self.tech_ary cannot be empty')

                # load 好
                self.tech_ary_init = tech_pd.to_numpy()  # REQUIRED IN STEP

                assert self.tech_ary_init.shape[1] == self.strategy.ALL_TECH_DIM

                # self.tech_ary = deque(self.tech_ary_init)
                self.tech_ary = deque([None], maxlen=self.max_deque)  # REQUIRED IN STEP

                # 假裝 load history 的樣子
                self.tech_ary[0] = self.tech_ary_init[0, :]  # REQUIRED IN STEP

                loaded_tech_cols = tech_pd.columns.values.tolist()
                assert set(loaded_tech_cols) == set(self.strategy_cls.STATE_TECH_COLS), f"expect equal but got loaded_tech_cols={set(loaded_tech_cols)} tech_cols={set(self.tech_cols)}"


            elif self.exch_env.price_env == PriceEnv.TRADE:
                self.tech_ary = deque([None], maxlen=self.max_deque)

                # ====== START 更新價格，load 入價格, 設定初始 tech_ary =====
                self.price_fetcher.catchup_price(self.trade_args["exchange"]
                                                 , self.trade_args["symbol"]
                                                 , self.trade_args["price_data_path"]
                                                 , self.trade_args["form_start"]
                                                 , self.trade_args["interval_base"])
                price_ddf, _ = self.price_fetcher.load_from_file()

                # 10 天應該可以把數字過高的地方弭平, 跟 後面 price catchup 搭配使用
                form_start_days = 10
                #form_start = (dt.date.today() - dt.timedelta(days=form_start_days)).strftime("%Y-%m-%d")

                # with ProgressBar():
                #    price_pd = price_ddf.loc[form_start:].compute()
                price_pd = self.process_data_in_chunks(price_ddf, days=form_start_days, chunk_size='1D')

                last_tech_ary = self.strategy.warm_up(price_pd)  # 價格在這放入，讓他的 tech indicator 自己跟上
                print(f"last_tech_ary: {last_tech_ary}")

                # IMPORTANT
                self.set_tech_ary(last_tech_ary)
                # ====== EMD 更新價格，load 入價格, 設定初始 tech_ary =====

                dt_idx, price_ary, tech_pd = None, None, None

            else:
                raise Exception("BrunhildDatastore.init_tech_data Deal with this case")

            return dt_idx, price_ary, tech_pd
        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(f"BinanceTrade: {err}")
            time.sleep(3)
            sys.exit()

    @staticmethod
    def process_data_in_chunks(price_ddf, days=10, chunk_size='1D'):
        """
        Process a Dask DataFrame in chunks to avoid memory issues.
        # Usage
        price_pd = process_data_in_chunks(price_ddf, days=10, chunk_size='1D')

        Parameters:
        -----------
        price_ddf : dask.dataframe.DataFrame
            The Dask DataFrame to process
        days : int
            Number of days to look back
        chunk_size : str
            Size of each chunk as a pandas frequency string (e.g., '1D' for daily chunks)

        Returns:
        --------
        pandas.DataFrame
            Combined result of all chunks
        """
        form_start_date = pd.Timestamp(dt.date.today() - dt.timedelta(days=days))
        print(f"Processing data from {form_start_date} in chunks of {chunk_size}")

        # Create a list of date ranges for chunking
        date_ranges = pd.date_range(start=form_start_date, end=pd.Timestamp.today(), freq=chunk_size)
        if date_ranges[-1].date() < dt.date.today():
            date_ranges = date_ranges.append(pd.DatetimeIndex([pd.Timestamp.today()]))

        results = []
        total_chunks = len(date_ranges) - 1

        for i in range(total_chunks):
            chunk_start = date_ranges[i].strftime("%Y-%m-%d")
            chunk_end = date_ranges[i + 1].strftime("%Y-%m-%d")
            print(f"Processing chunk {i + 1}/{total_chunks}: {chunk_start} to {chunk_end}")

            try:
                # Process one day at a time
                with ProgressBar():
                    chunk_data = price_ddf.loc[chunk_start:chunk_end].compute()

                print(f"Chunk {i + 1} size: {len(chunk_data)} rows")
                results.append(chunk_data)
            except Exception as e:
                print(f"Error processing chunk {i + 1}: {type(e).__name__}: {str(e)}")

        # Combine all chunks
        if results:
            return pd.concat(results)
        else:
            return pd.DataFrame()


    def reset(self):
        self.logger.debug("BrunhildEnv_v11.reset() entered")
        assert self.env_name is not None, "Env env_name cannot be None"
        assert self.action_dim is not None, "Env action_dim cannot be None"
        assert self.state_dim is not None, "Env state_dim cannot be None"

        self.total_reward = 0
        try:
            self.price_fetcher.reset()
            self.strategy = self.strategy_cls(self.trade_args.get("app_env"),
                                              self.trade_args,
                                              self.tech_args,
                                              self.logger)

            self.exch_env = BaseEnv_v11(self.hyper_args, self.trade_args, self.tech_args, self.strategy, self.logger,
                                     price_fetcher=self.price_fetcher, exch_api=self.exch_api, done_enabled=self.done_enabled,
                                     rl_mode=None)
            self.exch_env.reset()

            dt_idx, price_ary, tech_pd = self._init_tech_data()

            # IMPORTANT： reset 要在 max_step 前面，才能計算正確
            if self.exch_env.price_env == PriceEnv.TRAIN:
                self.max_step = len(self.price_fetcher.price_actual) - self.exch_env.ds.stacking_lookback

            if self.exch_env.price_env == PriceEnv.TRAIN and self.exch_env.app_env == AppEnv.TRAIN:
                self.reward_per_step = 0.01  # 5000 / self.price_fetcher.price_len()

                self.env_args["reward_per_step"] = self.reward_per_step
                self.env_args["max_step"] = self.max_step
                self.exch_env.spec["env_args"] = self.env_args
                self.logger.info(f"[BrunhildEnv] properties list:\n{pretty_dict(self.exch_env.spec)}")

            # 假裝 step
            # for i in range(self.exch_env.ds.stacking_lookback):
            #     # 會記錄下所有事
            #     self.state, _, _, _ = self.step(np.zeros(self.action_dim, dtype=np.float32), None)
            self.state = self.get_state(idx=0)

            return self.state
        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(f"{self.env_name}.reset(): {err}")
            time.sleep(3)
            sys.exit()

    def _get_price(self) -> Tuple[dt.datetime, List[Decimal]]:
        """取得價格然後更新 ds"""
        if self.exch_env.ds.price_env == PriceEnv.TRADE:
            price_se, _ = self.price_fetcher.get_price()
            price: Decimal = price_se[-1]
            dt_idx = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
            self.exch_env.ds.set_price(dt_idx, price,
                              app_env=self.exch_env.app_env)
            last_price = np.squeeze(self.exch_env.ds.get_last_price())[()]
            dt_idx = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)

        elif self.exch_env.ds.price_env == PriceEnv.TRAIN:
            dt_idx, price_arr = self.price_fetcher.get_price(self.exch_env.ds.get_idx())
            dt_idx = pd.to_datetime(dt_idx).to_pydatetime()
            price = price_arr[-1]
            self.exch_env.ds.set_price(dt_idx, price,
                                       app_env=self.exch_env.app_env)
            last_price = np.squeeze(self.exch_env.ds.get_last_price())[()]
        else:
            raise Exception(f"BrunhildEnv_v11.cal_tech_ary price_env expect [TRAIN|TRADE] got {self.exch_env.ds.price_env} ")

        price_list: List[Decimal] = [last_price, price]

        #  paper_pnl !!!!!
        paper_pnl, paper_pnl_pct = self.exch_env.ds.cal_paper_pnl_run(price)
        self.exch_env.ds.set_paper_pnl_pct(paper_pnl_pct)

        return dt_idx, price_list


    def step(self, action: npt.NDArray, prior_model: Dict[str, Any]=None):
        """
        Step into the environment.
        :return: A tuple containing the new observation, the reward signal,
        whether the episode is over and additional information.
        """
        try:
            idx = self.exch_env.ds.get_idx()
            power = 1
            self.exch_env.ds.set_buysell_lvl(power)

            # this happens when done half way and have to restart again, but idx become zero
            if idx == 0 and self.exch_env.app_env==AppEnv.TRADE:
                # self.strategy.step()
                self.exch_env.ds.step_idx(self.exch_env.app_env)
                return

            self.tech_ary.append(None)

            # Kelly_cap
            kelly_cap = self.exch_env.ds.ds_cal_kelly_cap()

            done_kelly = (kelly_cap <= self.exch_env.ds.trade_args.get("min_kelly_cap")) if self.exch_env.ds.done_kelly_active else False

            # 正常來說，如果是 PriceEnv.TRADE 時 (BotEnv.SIMULATION 或 BotEnv.TRADE) 時 prior_model 一定要有
            if prior_model is None and (self.exch_env.price_env in [PriceEnv.TRADE, PriceEnv.WS]) and idx==0:
                raise Exception(f"prior_model cannot be None when idx={idx}, bot_env={self.exch_env.bot_env}")

            # 取得價錢和 計算 tech 必須分為 TRAIN & TRADE
            if prior_model is not None:
                assert prior_model.get("dt_idx") is not None, "prior_model.dt_idx cannot be None"
                assert prior_model.get("price_list") is not None, "prior_model.price_list cannot be None"
                assert prior_model.get("trade_action_new") is not None, "prior_model.trade_action_new cannot be None"
                assert prior_model.get("tech_ary") is not None, "prior_model.tech_ary cannot be None"
                assert prior_model.get("user_input") is not None, "prior_model.user_input cannot be None"

                dt_idx, price_list = prior_model.get("dt_idx"), prior_model.get("price_list")
                trade_action_new, tech_ary = prior_model.get("trade_action_new"), prior_model.get("tech_ary")
            else:
                dt_idx, price_list = self._get_price()

                buy_delta: float = self.strategy.buy_delta
                sell_delta: float = self.strategy.sell_delta

                # training purpose
                if idx < 20 and self.exch_env.price_env == PriceEnv.TRAIN:
                    power = 1 # 沒用了？
                    self.exch_env.ds.set_buysell_lvl(power)
                else:
                    power = 1  # it is ok, there is another 0.98 inside BUY
                    self.exch_env.ds.set_buysell_lvl(power)

                    if not self.exch_env.ds.is_significant_pos():
                        sell_delta_steer = action[0] if abs(action[0]) > 0.3 else 0.
                        sell_delta_steer = sell_delta_steer * self.SELL_DELTA_CHANGE

                        # self.logger.info(">> sell_delta_steer:{0:.4f}".format(sell_delta_steer))

                        sell_delta += sell_delta_steer
                    # [end if]
                # [end if]

                sell_delta = np.clip(sell_delta, self.SELL_DELTA_LIMIT_LOWER, self.SELL_DELTA_LIMIT_UPPER)

                # IMPORTANT
                # 交易規則模型
                trade_action_new, tech_ary = self.strategy.step(price_list, dt_idx, self.exch_env.ds,
                                                                self.exch_env.ds.trade_args.get("long_short"),
                                                                buy_delta, sell_delta)

                self.set_tech_ary(tech_ary)
            # [if prior_model is not None:]

            # print(f"trade_args.get('debug'):{self.trade_args.get('debug')}")
            if self.trade_args.get("debug") and False:
                self.logger.debug(f"[{self.env_name}] Replace trade_action_new for DEBUGGING !!!")
                if idx == 2:
                    trade_action_new = TradeAction.BUY
                elif idx == 3:
                    trade_action_new = TradeAction.HOLD
                elif idx == 4:
                    trade_action_new = TradeAction.SELL
                elif idx == 5:
                    trade_action_new = TradeAction.HOLD
                elif idx == 6:
                    trade_action_new = TradeAction.BUY
                elif idx == 7:
                    trade_action_new = TradeAction.BUY
                elif idx == 8:
                    trade_action_new = TradeAction.HOLD
                elif idx == 9:
                    trade_action_new = TradeAction.SELL
                elif idx == 10:
                    trade_action_new = TradeAction.HOLD

            realized_pnl, realized_pnl_pct, cumulative_returns, order_result = self.exch_env.step(dt_idx,
                                                                                                  price_list[1],
                                                                                                  trade_action_new,
                                                                                                  done_kelly,
                                                                                                  power)

            # kelly cap record
            # self.kelly_p[idx] = self.KellyCls.p
            # self.trade_args["min_kelly_cap"]
            # idx, kelly_cap, kelly_b_win, kelly_b_loss
            self.exch_env.ds.set_kelly(kelly_cap,
                              self.exch_env.ds.KellyCls.p,
                              self.exch_env.ds.KellyCls.b_win,
                              self.exch_env.ds.KellyCls.b_loss)

            # ===== IMPORTANT ==========
            state = self.get_state(idx)

            step_reward = self.strategy.cal_reward(self.exch_env.ds, realized_pnl_pct, self.reward_per_step)
            self.exch_env.ds.set_rewards(step_reward)

            # self.total_reward = (accuracy * accum_realized_pnl) + (0.75 * step_reward)

            done = self._check_done(done_kelly) if self.done_enabled else False

            # IMPORTANT, 全部 ds 相關的事情做完後才能執行
            self.exch_env.ds.step_idx(self.exch_env.app_env)  # idx += 1

            done_price = False
            if self.exch_env.ds.app_env == AppEnv.TRAIN:
                dt_idx, price = self.exch_env.get_price(idx+1)
                # done_price 價格都跑完
                if dt_idx is None and price is None:
                    self.logger.info(f"Done price with final asset_cash {self.exch_env.ds.get_last_cash_asset():.0f} "
                                     f"[{cumulative_returns * 100:.1f}%]")
                    done_price = True

            done = done or done_price
            if done or done_price:
                # step_reward += 1 / (1 - self.exch_env.ds.gamma) * self.exch_env.ds.get_mean_reward()
                final_pnl = float((d(1) + cumulative_returns))
                self.total_reward += (final_pnl * 123)
                step_reward = self.total_reward
                self.exch_env.ds.set_last_rewards(step_reward)

                self.exch_env.cumulative_returns = cumulative_returns

            return state, step_reward, done, order_result  # self.exch_env.ds
        except Exception as e:
            self.logger.error(readable_error(e, __file__))
            self.logger.error(f"silo: {self.exch_env.ds.silo.position}\n"
                              f"position:{self.exch_env.ds.get_last_5_position()}")

            time.sleep(3)
            sys.exit()

    def get_state(self, idx):
        # ===== IMPORTANT ==========
        buy_ind = self.strategy.BuyIndi.value(idx)
        sell_ind = self.strategy.SellIndi.value(idx)
        last_position = self.exch_env.ds.get_last_position()
        in_pos = self.exch_env.ds.is_significant_pos()
        state = self.clip_observation(
            [
                buy_ind.tracer / 100,
                buy_ind.alfa,
                sell_ind.tracer / 100,
                sell_ind.alfa,
                buy_ind.sd,
                normalize_decreasing_values_centered(sell_ind.delta, self.SELL_DELTA_LIMIT_LOWER,
                                                     self.SELL_DELTA_LIMIT_UPPER),
                in_pos
            ]
        )
        return state

    def _check_done(self, done_kelly: bool):
        done_max_step = self.exch_env.ds.get_idx() >= self.max_step

        paper_pnl_pct = self.exch_env.ds.get_paper_pnl_pct()

        # DEPRECATING CODE 在每一個 hold/buy/sell/short/cover 都已執行
        """
        self.ds.set_drawdown()
        self.ds.set_drawdown_pct(idx, drawdown_pct)
        """

        # 檢查 loss_at_month
        # self._is_new_month()
        self._is_trailing_30_days()
        starting_asset_cash = self.exch_env.ds.get_starting_asset_cash()
        loss_at_month = (self.exch_env.ds.get_cash_asset() - starting_asset_cash) / starting_asset_cash if starting_asset_cash > 0 else 0
        total_loss = (self.exch_env.ds.get_cash_asset() - self.exch_env.ds.init_trade_cash) / self.exch_env.ds.init_trade_cash if self.exch_env.ds.init_trade_cash >0 else 0

        # 檢查 done
        done_total_loss = (total_loss < self.exch_env.ds.trade_args['done_total_loss'])
        done_monthly_loss = False  # (loss_at_month < self.ds.trade_args['done_monthly_loss'])
        # IMPORTANT 這裡必須是 now, 最新計算出來的
        done_drawdown = (paper_pnl_pct < self.exch_env.ds.trade_args['done_drawdown'] and self.exch_env.ds.is_significant_pos_now())
        # defined above
        # kelly_cap = self.kelly_cap[idx]
        # done_kelly = (kelly_cap < self.trade_args['min_kelly_cap']) if self.done_kelly_active else False
        done = (done_max_step or done_total_loss or done_monthly_loss or done_kelly or done_drawdown)

        if done_total_loss:
            self.logger.warning(f"[BrunhildEnv] Max total loss reached, expect not worse than "
                                f"{float(self.exch_env.ds.trade_args['done_total_loss']) * 100}% "
                                f"but got {total_loss * 100:.3f}%")

        if done_monthly_loss:
            self.logger.warning(f"[BrunhildEnv] Max monthly loss reached, expect not worse than "
                                f"{float(self.exch_env.ds.trade_args['done_monthly_loss']) * 100}% "
                                f"but got {loss_at_month * 100:.3f}%")

        if done_kelly:
            self.logger.warning(f"[BrunhildEnv] kelly_cap expect larger than {self.exch_env.ds.trade_args['min_kelly_cap']} "
                                f"but got {round(self.exch_env.ds.get_kelly_cap(), 5)}")
            self.exch_env.ds.print_last_10_kelly()

        if done_drawdown:
            print(f"[BrunhildEnv] drawdown: {[self.exch_env.ds.get_paper_pnl_pct_at(i) for i in range(-1, -10, -1)]}")
            self.logger.warning(
                f"[BrunhildEnv] Max drawdown reached, expect not worse than {self.exch_env.ds.trade_args['done_drawdown']} "
                f"but got {round(paper_pnl_pct, 2)}")

        return done

    def get_step_per_episode(self):
        return self.max_step

    def _is_new_month(self):
        # 本來是 numpy.datetime64
        # m = self.ds.get_dt_idx().astype(object).month
        current_date = self.exch_env.ds.get_dt_idx().date()
        if current_date.month != self.exch_env.ds.last_recorded_date.month:
            self.exch_env.ds.last_recorded_date = current_date.month
            # self.ds.set_starting_asset_cash(self.ds.get_cumulative_realized_pnl() + self.ds.init_trade_cash)
            self.exch_env.ds.set_starting_asset_cash(self.exch_env.ds.get_cash_asset())
            return True
        return False

    def _is_trailing_30_days(self):
        current_date = self.exch_env.ds.get_dt_idx().date()
        last_recorded_date = getattr(self.exch_env.ds, "_last_recorded_date", None)

        if last_recorded_date is None or (current_date - last_recorded_date) >= dt.timedelta(days=30):
            self.exch_env.ds.last_recorded_date = current_date
            self.exch_env.ds.set_starting_asset_cash(self.exch_env.ds.get_cash_asset())
            return True
        return False

    def draw_cumulative_return(self, args, _torch) -> np.ndarray:
        try:

            env = args.env

            gpu_id = args.learner_gpus[0]

            agent = init_agent(args, gpu_id, env)

            cwd = args.cwd

            # agent.init(net_dim, state_dim, action_dim)
            agent.save_or_load_agent(cwd, if_save=False)
            act = agent.act
            device = agent.device

            state = self.reset()
            episode_returns = np.zeros(self.max_step)  # the cumulative_return / initial_account
            with _torch.no_grad():
                for i in range(self.max_step):
                    s_tensor = _torch.as_tensor((state,), device=device)
                    a_tensor = act(s_tensor)  # action_tanh = act.forward()
                    action = (
                        a_tensor.detach().cpu().numpy()[0]
                    )  # not need detach(), because with torch.no_grad() outside

                    state, reward, done, _ = self.step(action)

                    # total_asset = (self.ds.price_ary[self.ds.get_idx()] * self.ds.get_position(i)).sum() + self.ds.cash
                    # 我們用上一個因為已經 step
                    total_asset = self.exch_env.ds.get_last_cash_asset()

                    episode_return = total_asset / self.exch_env.ds.init_trade_cash
                    episode_returns[i] = episode_return
                    if done:
                        episode_returns = episode_returns[:i]
                        break

                import matplotlib.pyplot as plt

                plt.plot(episode_returns)
                plt.grid()
                plt.title("cumulative return")
                plt.xlabel("interval")
            plt.xlabel("multiple of initial_account")
            plt.savefig(f"{cwd}/cumulative_return.jpg")
            print(f"| draw_cumulative_return: save in {cwd}/cumulative_return.jpg")

            return episode_returns
        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)

    def get_tech_ary(self):
        return self.tech_ary[-1]

    def get_last_tech_ary(self):
        if self.exch_env.ds.get_idx() <= 1:
            return self.tech_ary[-1]
        return self.tech_ary[-2]

    def set_tech_ary(self, tech_ary):
        self.tech_ary[-1] = tech_ary

    def copy_last_tech_ary(self):
        self.tech_ary[-1] = self.tech_ary[-2]


def check_env(hyper_args, env_args, trade_args, tech_args, strategy_cls, logger):

    env = BrunhildEnv(hyper_args, env_args, trade_args, tech_args, strategy_cls, logger)
    env.if_random_reset = False
    evaluate_time = 4
    """
    env = StockTradingEnv(beg_idx=0, end_idx=1113)
    cumulative_returns of random action   :      1.63
    cumulative_returns of buy all share   :      2.80

    env = StockTradingEnv(beg_idx=0, end_idx=834)
    cumulative_returns of random action   :      1.94
    cumulative_returns of buy all share   :      2.51

    env = StockTradingEnv(beg_idx=834, end_idx=1113)
    cumulative_returns of random action   :      1.12
    cumulative_returns of buy all share   :      1.19
    """

    policy_name = "random action"
    state = env.reset()
    for _ in range(env.exch_env.max_step * evaluate_time):  # env.max_step * evaluate_time
        action = rd.uniform(-1, +1, env.action_dim)
        state, reward, done, _ = env.step(action)
        env.exch_env.render(mode="console")

        idx = env.exch_env.ds.get_idx() - 1
        if done:
            # # env.ds.get_cumulative_returns()
            print(f"cumulative_realized_pnl of {policy_name}: {env.exch_env.ds.get_cumulative_realized_pnl():9.2f}%, "
                  f"idx: {idx:9}")
            state = env.reset()
    dir(state)

    print()
    policy_name = "buy all share"
    state = env.reset()
    for _ in range(evaluate_time):  # env.max_step * evaluate_time
        action = np.ones(env.action_dim, dtype=np.float32)
        state, reward, done, _ = env.step(action)
        if done:
            idx = env.exch_env.ds.get_idx() - 1
            print(f"cumulative_realized_pnl of {policy_name}: {env.exch_env.ds.get_cumulative_realized_pnl():9.2f}, "
                  f"idx: {idx:9}")
            state = env.reset()
    dir(state)
    print()

