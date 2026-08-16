import sys

import numpy as np
import datetime as dt
import numpy.typing as npt
import numpy.random as rd
from decimal import Decimal
from ahf.core.enums import AppEnv, PriceEnv

from ahf.utils.utils import readable_error
from ahf.rl.envs.TradeEnum import TradeAction, ErrorCodes
from ahf.rl.envs.BrunhildDatastore_v11 import BrunhildDatastore_v11 as Datastore
from api.Binance.BacktestOrder import BacktestOrder
from ahf.utils.utils import d, d_abs, to_dict

# plot
from ahf.rl.envs.plot_helper_v21 import Tech_Plot, Trade_Plot

# import envs.BrunhildReadWrite as read_write
from ahf.rl.train.utils import init_agent, init_read_write


class BaseEnv_v11:
    """
        Base environment , extend from here
        this provides basic functionality, just like openai.Gym
    """

    def __init__(self, hyper_args, trade_args, tech_args, strategy, logger,
                 price_fetcher=None, exch_api=None, done_enabled=True, rl_mode=None):
        """

        Parameters
        ----------
        hyper_args
            - type: dict
            - Description: hyper_params pre-set for trading related arguments
        trade_args
            - type: dict
            - Description: trading related arguments
        tech_args
            - type: list
            - Description: technical indicators that are used
        strategy
            - type: Strategy Class
            - Description: Trading Strategy to carry out actual logic
        logger
            - type: Logging
        price_fetcher
            - type: PriceFetcher
            - Description:  Provide prices when train or trade
        done_enabled
            - type: bool
            - Description: 遇到 done 條件會停止運行
        """
        try:
            self.done_enabled = done_enabled

            self.logger = logger
            self.hyper_args = hyper_args
            self.trade_args = trade_args
            self.tech_args = tech_args
            self.exch_api = exch_api
            self.strategy = strategy
            self.rl_mode = rl_mode

            self.logger.info(f'[Datastore] {__name__} Class loaded')

            self.env_num = hyper_args.get("env_num", 1) or 1  # for vec environment

            """

            """
            self.app_env = trade_args.get("app_env")
            self.price_env = trade_args.get("price_env")
            self.bot_env = trade_args.get("bot_env")
            self.price_fetcher = price_fetcher
            self.read_write = init_read_write(trade_args.get('tech_id'), logger)

            assert isinstance(self.app_env, AppEnv), "app_env expect type AppEnv got str"
            assert isinstance(self.price_env, PriceEnv), "price_env expect type PriceEnv got str"

            if self.app_env == AppEnv.TRAIN and self.price_env == PriceEnv.TRAIN:
                # Check indicators exist, otherwise, generate it for RL training
                self.read_write.create_tech_if_not_exist(trade_args, tech_args, logger)
            elif self.app_env == AppEnv.TRAIN and self.price_env == PriceEnv.TRADE:
                pass
            elif self.app_env == AppEnv.TRADE:
                pass
            else:
                raise Exception("Please handle it")

            self.ds = Datastore(hyper_args,
                                trade_args,
                                tech_args,
                                self.exch_api,
                                price_fetcher,
                                self.read_write,
                                self.strategy,
                                self.logger)

            # rendering
            self.is_render = trade_args.get("is_render", True)
            self.render_mode = trade_args.get("render_mode", "console")

            self.cumulative_returns = 0.

            self.bt_order = BacktestOrder(self.ds, self.exch_api,
                                          trade_args, self.app_env,
                                          self.logger)

            self.ds.reset()  # IMPORTANT: init_tech_data is done inside

            # no kelly involved for Brunhild
            self.ds.done_kelly_active = False

            # print out params and save
            self.spec = {"ds": to_dict(self.ds), "hyper_args": self.hyper_args, "tech_args": self.tech_args, "trade_args": self.trade_args}

            self.obs_low: np.array = None
            self.obs_high: np.array = None
            self.action_low: np.array = None
            self.action_high: np.array = None

        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)
            sys.exit()

    def get_price(self, idx: int):
        if self.price_env == PriceEnv.TRAIN:
            # 這裡的 price_fetcher 會是 PriceFetcherTrain class
            dt_idx, price_arr = self.price_fetcher.get_price(idx)
            if dt_idx is None and price_arr is None:
                return None, None
            if isinstance(price_arr, list):
                price_new = price_arr[-1]
            else:
                price_new = price_arr
        else:
            price_se, _ = self.price_fetcher.get_price()
            if isinstance(price_se, list):
                price_new = price_se[-1]
            else:
                price_new = price_se
            dt_idx = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)

        # 轉成 dt.datetime
        if not isinstance(dt_idx, dt.datetime):
            dt_idx = dt_idx.astype(dt.datetime)
        return dt_idx, price_new[0]

    def reset(self):
        """reset exch_env 環境和 ds 資料

        *** IMPORTANT， TRAINING USAGE ONLY***
        在 PriceEnv.TRADE (==AppEnv.TRADE or AppEnv.SIMULATION), 所以完全不應該 env.reset() 會發生
        """
        idx = self.ds.get_idx()
        if self.price_env == PriceEnv.TRADE and idx > 0:
            raise Exception(f"Do not do reset on env once started on PriceEnv.TRADE mode when started idx: {idx}")

        try:
            # ===== BEGIN BaseENv =====
            self.ds.reset()
            assert self.ds.get_idx() == 0, f"idx should be 0 but got {self.ds.get_idx()}"

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
            self.ds.set_buysell_lvl(0)

            # IMPORTANT！！！ 跳過第一個，因為在訓練時沒有初始值
            if self.price_env == PriceEnv.TRAIN:
                self.ds.step_idx(self.app_env)

            # ===== END BaseENv =====

            # ===== START Env Level
            self.ds.set_kelly(self.ds.ds_cal_kelly_cap(),
                              self.ds.KellyCls.p,
                              self.ds.KellyCls.b_win,
                              self.ds.KellyCls.b_loss)
            self.ds.set_rewards(0.0)

        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)
            sys.exit()

    def step(self,
             dt_idx: dt.datetime,
             price: Decimal,
             trade_action_new: TradeAction,
             done_kelly: bool,
             power: float):
        try:
            idx = self.ds.get_idx()
            last_position = self.ds.get_last_position()

            if self.trade_args.get("debug") and False:
                self.logger.debug("[BrunhildEnv] Replace trade_action_new for DEBUGGING !!!")
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

            # 考慮 kelly_cap, 買太多不要買
            # if trade_action_new == TradeAction.BUY:
            #     a = self.agent.ds.get_last_asset()
            #     b = self.agent.ds.get_last_cash_asset()
            #     if self.agent.ds.get_last_asset() / self.agent.ds.get_last_cash_asset() > kelly_cap:
            #         trade_action_new = TradeAction.HOLD

            # start buy/sell/short/cover
            if done_kelly:
                trade_action_new = TradeAction.HOLD

            last_asset = self.ds.get_last_asset()
            last_trade_cash = self.ds.get_last_trade_cash()
            realized_pnl, realized_pnl_pct = d(0.0), d(0.0)

            # 已買到超過就不要買
            if trade_action_new == TradeAction.HOLD:
                order_re = self.bt_order.hold(idx, dt_idx, price)

            # if kelly_cap >= 0 and not done_kelly:
            elif ((trade_action_new == TradeAction.BUY and
                   price > 0 and
                  self.ds.silo.can_buy(self.ds.is_significant_pos())) and
                  idx >=  1):
                order_re = self.bt_order.buy(idx, dt_idx, d(power), price)

            elif (trade_action_new == TradeAction.SELL and
                  last_position > 0 and
                  self.ds.silo.can_sell(self.ds.is_significant_pos()) and
                  idx >=  1):  # action[0] < 0
                # Sell only if current asset is > 0
                action_num = d_abs(power)
                if self.ds.silo.get_size() == 1:
                    action_num = d(1)
                order_re = self.bt_order.sell(idx, dt_idx, action_num, price)
                realized_pnl, realized_pnl_pct = order_re.get("realized_pnl"), order_re.get("realized_pnl_pct")

                # elif trade_action_new == TradeAction.HOLD and action[1] < 0 and last_position > 0 and self.agent.ds.silo.can_sell() and idx > self.agent.ds.stacking_lookback + 1:
                #     order_re = self.agent.bt_order.sell(idx, dt_idx, d_abs(action[1]), price, last_position)
                #     realized_pnl, realized_pnl_pct = order_re.get("realized_pnl"), order_re.get("realized_pnl_pct")
                # elif action[1] > 0 and self.agent.ds.silo.can_buy():
                #     pass


            # if kelly_cap < 0 and not done_kelly:
            elif (trade_action_new == TradeAction.COVER and
                  last_position < 0 and
                  self.ds.silo.can_cover(self.ds.is_significant_pos()) and
                  price > 0 and
                  idx >=  1):
                order_re = self.bt_order.cover(idx, dt_idx, d_abs(power), price, last_position)
                realized_pnl = order_re.get("realized_pnl")
                realized_pnl_pct = order_re.get("realized_pnl_pct")

            elif (trade_action_new == TradeAction.SHORT and
                  self.ds.silo.can_short(self.ds.is_significant_pos()) and
                  price > 0 and
                  idx >=  1):  # sell_stock
                # Sell only if current asset is > 0
                order_re = self.bt_order.short(idx, dt_idx, d_abs(power), price, last_position)
            else:
                error_code = ErrorCodes.ERROR_TH000
                # 只能買賣一筆
                if trade_action_new == TradeAction.BUY and not self.ds.silo.can_buy(self.ds.is_significant_pos()):
                    error_code = ErrorCodes.ERROR_TH012

                # 沒部位卻想賣掉
                if trade_action_new == TradeAction.SELL and not self.ds.is_significant_pos():
                    error_code = ErrorCodes.ERROR_TH013

                order_re = self.bt_order.hold(idx, dt_idx, price, trade_hold_code=error_code)
                # raise Exception("No TradeAction finalized")

            # 處理手續費不夠, 這只有在 Training 用到
            if self.ds.get_buysell() == TradeAction.HOLD:
                self.bt_order.do_check_fee(price)

            assert trade_action_new is not None, "[BaseEnv] trade_action_new cannot be None "
            # DEBUG
            # if self.agent.ds.get_buysell() != TradeAction.HOLD:
            #    print(f"                      => silo: {self.agent.ds.silo.position}")
            #    print(f"                      => position:{self.agent.ds._position[max(self.agent.ds.get_idx() - 5, 0):self.agent.ds.get_idx() + 1]}")

            # else:
            #     raise Exception(f"trade_args.stock_crypto must be "
            #                    f""stock" or "crypto" but we got {self.trade_args.stock_crypto}")
            cumulative_returns = self.ds.get_cumulative_realized_pnl() / self.ds.init_trade_cash if self.ds.init_trade_cash > 0 else d("0")

            return realized_pnl, realized_pnl_pct, cumulative_returns, order_re

        except Exception as e:
            self.logger.error(readable_error(e, __file__))
            self.logger.error(f"silo: {self.ds.silo.position}\n"
                              f"position:{self.ds.get_last_5_position()}")

    # def step_sample(self, action: npt.NDArray,
    #          prior_model: dict = None):
    #     """下面都是 demo code 而已，你要自己寫自己的"""
    #
    #     try:
    #         idx = self.ds.get_idx()
    #
    #         # 取得價錢和 計算 tech 必須分為 TRAIN & TRADE
    #         # if self.ds.app_env == AppEnv.TRAIN:
    #         #     dt_idx, price = self.get_price(idx)
    #         #     self.ds.set_tech_ary(self.ds.tech_ary_init[idx, :])
    #         # elif self.ds.app_env == AppEnv.TRADE:
    #         #     dt_idx, price = self.get_price(idx)
    #         # else:
    #         #     raise Exception("Add more cases, do something you want")
    #
    #         if prior_model is None and (self.price_env in [PriceEnv.TRADE, PriceEnv.WS]):
    #             raise Exception(f"prior_model cannot be None when {self.bot_env}")
    #
    #         # this happens when done half way and have to restart again, but idx become zero
    #         if idx == 0:
    #             self.ds.step_idx(self.app_env)
    #             return
    #
    #         done_kelly = False
    #
    #         # action = action.copy()
    #
    #         last_position = self.ds.get_last_position()
    #
    #         # Kelly_cap
    #         kelly_cap = self.ds.ds_cal_kelly_cap()
    #
    #         realized_pnl, realized_pnl_pct = d(0.0), d(0.0)
    #
    #         if self.ds.trade_args['stock_crypto'] == 'stock':
    #             pass
    #             """
    #             action0_int = (action[0] * self.ds.cash).astype(int)
    #             # actions initially is scaled between -1 and 1
    #             # convert into integer because we can't buy fraction of shares for stock
    #
    #             stock_action0 = action0_int
    #
    #             adj_close_price = self.ds.price_ary[self.ds.get_idx()]  # `adjcp` denotes adjusted close price
    #             if stock_action0 > 0:  # buy_stock
    #                 delta_stock = min(self.ds.cash // adj_close_price, stock_action0)
    #                 self.ds.cash -= adj_close_price * delta_stock * (1 + self.ds.buy_cost_rate)
    #                 self.ds.position[idx] = delta_stock + last_position
    #             elif self.ds.position[idx] > 0:  # sell_stock
    #                 delta_stock = min(-stock_action0, self.ds.position[idx])
    #                 realized_pnl = adj_close_price * delta_stock * (1 - self.ds.sell_cost_rate)
    #                 self.ds.cash += realized_pnl
    #                 self.ds.position[idx] = last_position - delta_stock
    #             """
    #
    #         if self.ds.trade_args['stock_crypto'] == 'crypto':
    #
    #             done_kelly = (kelly_cap <= self.ds.trade_args['min_kelly_cap']) if self.ds.done_kelly_active else False
    #
    #             if prior_model is None:
    #                 dt_idx, price = self.get_price(idx)
    #                 self.ds.set_price(dt_idx,
    #                                   price,
    #                                   app_env=self.app_env)
    #                 trade_action_new = TradeAction.HOLD  # 暫時的
    #             else:
    #                 dt_idx, price = prior_model.get("dt_idx"), prior_model.get("price")
    #                 trade_action_new = prior_model.get("trade_action_new")
    #
    #             # training purpose
    #             if idx < 20 and self.price_env == PriceEnv.TRAIN:
    #                 power = 0
    #                 trade_action_new = TradeAction.HOLD
    #                 self.ds.set_buysell_lvl(power)
    #             elif self.rl_mode == "fix":
    #                 power = 1  # it is ok, there is another 0.98 inside BUY
    #                 self.ds.set_buysell_lvl(power)
    #                 if prior_model is None:
    #                     last_price = np.squeeze(self.ds.get_last_price())[()]
    #                     prices = [last_price, price]
    #                     trade_action_new, tech_ary = self.strategy.step(prices, dt_idx, self.ds,
    #                                                                     self.ds.trade_args.get("long_short"
    #                                                                     )
    #
    #             elif self.rl_mode == "power":
    #                 # action[(-0.1 < action) & (action < 0.1)] = 0
    #                 power = np.clip(action[0], 0.05, 1)  # OLD
    #                 self.ds.set_buysell_lvl(power)
    #                 if prior_model is None:
    #                     re = self.cal_tech_ary()
    #                     trade_action_new, tech_ary = re.get("trade_action_new"), re.get("tech_ary")
    #
    #             elif self.rl_mode == "steer":
    #                 action[(-0.5 < action) & (action < 0.5)] = 0
    #                 multi = 0.05 if action[0] > 0 else 0.1  # 增加慢，減少快，有如煞車
    #                 power_steer = action[0] * multi
    #
    #                 last_power = self.ds.get_last_buysell_lvl()
    #                 power = last_power + power_steer
    #                 power = np.clip(power, 0.01, 1)
    #                 self.ds.set_buysell_lvl(power)
    #
    #                 # kelly_cap, active_long_short = self.ds.decision.step(self.ds.trade_args['min_kelly_cap'],
    #                 #                                                      self.ds.trade_args['max_kelly_cap'],
    #                 #                                                      kelly_cap,
    #                 #                                                      self.ds.long_short)
    #                 if prior_model is None:
    #                     re = self.cal_tech_ary()
    #                     trade_action_new, tech_ary = re.get("trade_action_new"), re.get("tech_ary")
    #
    #             elif self.rl_mode == "rl":
    #                 gas = self.step_gas(action[0])
    #                 power = self.step_break(gas, action[1])
    #
    #                 done_kelly = False
    #                 if power > 0 and self.ds.trade_args['long_short'] == 'long':
    #                     trade_action_new = TradeAction.BUY
    #                 elif power < 0 and self.ds.trade_args['long_short'] == 'long':
    #                     trade_action_new = TradeAction.SELL
    #                 elif power > 0 and self.ds.trade_args['long_short'] == 'short':
    #                     trade_action_new = TradeAction.SHORT
    #                 elif power < 0 and self.ds.trade_args['long_short'] == 'short':
    #                     trade_action_new = TradeAction.COVER
    #                 else:
    #                     trade_action_new = TradeAction.HOLD
    #             else:
    #                 raise Exception(f"Unknown rl_mode got {self.rl_mode}")
    #
    #             # [end if]
    #
    #             if self.trade_args.get("debug") and False:
    #                 self.logger.debug("[BaseEnv] Replace trade_action_new for DEBUGGING !!!")
    #                 if idx == 2:
    #                     trade_action_new = TradeAction.BUY
    #                 elif idx == 3:
    #                     trade_action_new = TradeAction.HOLD
    #                 elif idx == 4:
    #                     trade_action_new = TradeAction.SELL
    #                 elif idx == 5:
    #                     trade_action_new = TradeAction.HOLD
    #                 elif idx == 6:
    #                     trade_action_new = TradeAction.BUY
    #                 elif idx == 7:
    #                     trade_action_new = TradeAction.BUY
    #                 elif idx == 8:
    #                     trade_action_new = TradeAction.HOLD
    #                 elif idx == 9:
    #                     trade_action_new = TradeAction.SELL
    #                 elif idx == 10:
    #                     trade_action_new = TradeAction.HOLD
    #
    #             # 考慮 kelly_cap, 買太多不要買
    #             # if trade_action_new == TradeAction.BUY:
    #             #     a = self.ds.get_last_asset()
    #             #     b = self.ds.get_last_cash_asset()
    #             #     if self.ds.get_last_asset() / self.ds.get_last_cash_asset() > kelly_cap:
    #             #         trade_action_new = TradeAction.HOLD
    #
    #             # start buy/sell/short/cover
    #             if done_kelly:
    #                 self.bt_order.hold(idx, dt_idx, price)
    #
    #             if self.ds.trade_args['long_short'] in ('long', 'dual'):
    #
    #                 last_asset = self.ds.get_last_asset()
    #                 last_trade_cash = self.ds.get_last_trade_cash()
    #
    #                 # 已買到超過就不要買
    #                 if trade_action_new == TradeAction.BUY and last_asset * d(1.1) > last_trade_cash:
    #                     self.logger.debug(f"[BaseEnv] asset {last_asset} * 1.1 is larger than"
    #                                       f"trade_cash {last_trade_cash}")
    #                     self.logger.info("[BaseEnv] Changing BUY to HOLD due to trade_cash")
    #                     trade_action_new = TradeAction.HOLD
    #
    #                 # if kelly_cap >= 0 and not done_kelly:
    #                 if (trade_action_new == TradeAction.BUY and price > 0 and power > 0 and
    #                         self.ds.silo.can_buy(self.ds.is_significant_pos())):
    #                     self.bt_order.buy(idx, dt_idx, d(power), price)
    #
    #                 elif trade_action_new == TradeAction.SELL and last_position > 0 and \
    #                         self.ds.silo.can_sell() and price > 0 and idx > self.ds.stacking_lookback + 1:  # action[0] < 0
    #                     # Sell only if current asset is > 0
    #                     action_num = d_abs(power)
    #                     if self.ds.silo.get_size() == 1:
    #                         action_num = d(1)
    #                     order_re = self.bt_order.sell(idx, dt_idx, action_num, price, last_position)
    #                     realized_pnl, realized_pnl_pct = order_re.get('realized_pnl'), order_re.get('realized_pnl_pct')
    #
    #                 # elif trade_action_new == TradeAction.HOLD and action[1] < 0 and last_position > 0 and self.ds.silo.can_sell() and idx > self.ds.stacking_lookback + 1:
    #                 #     order_re = self.bt_order.sell(idx, dt_idx, d_abs(action[1]), price, last_position)
    #                 #     realized_pnl, realized_pnl_pct = order_re.get('realized_pnl'), order_re.get('realized_pnl_pct')
    #                 # elif action[1] > 0 and self.ds.silo.can_buy():
    #                 #     pass
    #                 else:
    #                     self.bt_order.hold(idx, dt_idx, price)
    #
    #             if self.ds.trade_args['long_short']  in ('short', 'dual'):
    #                 # if kelly_cap < 0 and not done_kelly:
    #                 if (trade_action_new == TradeAction.COVER and power > 0 and last_position < 0
    #                         and self.ds.silo.can_cover() and price > 0):
    #                     order_re = self.bt_order.cover(idx, dt_idx, d_abs(power), price, last_position)
    #                     realized_pnl, realized_pnl_pct = order_re.get('realized_pnl'), order_re.get(
    #                         'realized_pnl_pct')
    #
    #                 elif (trade_action_new == TradeAction.SHORT and power < 0 and self.ds.silo.can_short() and
    #                       price > 0 and idx > self.ds.stacking_lookback + 1):  # sell_stock
    #                     # Sell only if current asset is > 0
    #                     self.bt_order.short(idx, dt_idx, d_abs(power), price, last_position)
    #                 else:
    #                     self.bt_order.hold(idx, price)
    #
    #
    #             # 處理手續費不夠, 這只有在 Training 用到
    #             if self.ds.get_buysell() == TradeAction.HOLD:
    #                 self.bt_order.do_check_fee(price)
    #
    #         # DEBUG
    #         # if self.ds.get_buysell() != TradeAction.HOLD:
    #         #    print(f'                      => silo: {self.ds.silo.position}')
    #         #    print(f'                      => position:{self.ds._position[max(self.ds.get_idx() - 5, 0):self.ds.get_idx() + 1]}')
    #
    #         # else:
    #         #     raise Exception(f"trade_args.stock_crypto must be "
    #         #                    f"'stock' or 'crypto' but we got {self.trade_args.stock_crypto}")
    #
    #         # kelly cap record
    #         # self.kelly_p[idx] = self.KellyCls.p
    #         # self.trade_args['min_kelly_cap']
    #         # idx, kelly_cap, kelly_b_win, kelly_b_loss
    #         self.ds.set_kelly(kelly_cap,
    #                           self.ds.KellyCls.p,
    #                           self.ds.KellyCls.b_win,
    #                           self.ds.KellyCls.b_loss)
    #
    #         # ===== IMPORTANT ==========
    #         state = self.ds.get_state()
    #
    #         step_reward = self.strategy.cal_reward(self.ds, realized_pnl_pct, self.reward_per_step)
    #         self.ds.set_rewards(step_reward)
    #
    #         # self.total_reward = (accuracy * accum_realized_pnl) + (0.75 * step_reward)
    #
    #         cumulative_returns = self.ds.get_cumulative_realized_pnl() / self.ds.init_trade_cash
    #
    #         done = self.check_done(done_kelly) if self.done_enabled else False
    #
    #         # IMPORTANT
    #         self.ds.step_idx(self.app_env)  # idx += 1
    #
    #         done_price = False
    #         if self.ds.app_env == AppEnv.TRAIN:
    #             dt_idx, price = self.get_price(idx + 1)
    #             # done_price 價格都跑完
    #             if dt_idx is None and price is None:
    #                 self.logger.info(f"Done price with final asset_cash {self.ds.get_last_cash_asset():.0f} "
    #                                  f"[{cumulative_returns * 100:.1f}%]")
    #                 done_price = True
    #
    #         done = done or done_price
    #         if done or done_price:
    #             # step_reward += 1 / (1 - self.ds.gamma) * self.ds.get_mean_reward()
    #             final_pnl = float((d(1) + cumulative_returns) * d(100 * 50))
    #             self.total_reward += final_pnl
    #             step_reward = self.total_reward
    #             self.ds.set_last_rewards(step_reward)
    #
    #             self.cumulative_returns = cumulative_returns
    #
    #         return state, step_reward, done, self.ds
    #     except Exception as e:
    #         err = readable_error(e, __file__)
    #         self.logger.error(err)
    #         self.logger.error(f'silo: {self.ds.silo.position} '
    #                           f'position:{self.ds.get_last_5_position()}')
    #
    #         sys.exit()


    def draw_tech(self, args):
        tech_plot = Tech_Plot(f'{args.cwd}/tech_ind.jpg')
        tech_plot.plot(self.ds.get_current_dt_idx(), self.ds.price_ary, self.ds.tech_ary,
                       self.strategy.ALL_TECH_COLS, [], [])

    @staticmethod
    def draw_trades(args):
        trade_plot = Trade_Plot(f'{args.cwd}/trade.png')
        trade_plot.plot()

    def render(self, mode="console"):
        if not (mode == 'human' or mode == 'console'):
            raise NotImplementedError()

        # Render the environment to the screen
        try:
            if mode == 'console':

                idx = self.ds.get_idx() - 1
                # dt_idx = self.ds.get_last_dt_idx().astype(dt.datetime).strftime("%Y-%m-%d %H:%M")
                dt_idx = self.ds.get_last_dt_idx().strftime("%Y-%m-%d %H:%M")

                cash_asset = self.ds.get_last_cash_asset()
                # cash = self.ds.get_last_cash()
                # asset = self.ds.get_last_asset()
                realized_pnl = self.ds.get_last_realized_pnl()
                gain = realized_pnl / cash_asset * d(100)
                gain_sign = '+' if gain > 0 else ''
                gain = gain_sign + str(round(gain, 1))
                buysell = self.ds.get_last_buysell().name if self.ds.get_last_buysell().name != 'HOLD' else '-'
                buysell = buysell if buysell != 'BUY_FEE' else 'FEE'
                price = np.squeeze(self.ds.get_last_price())  # [0]
                # turbulence = self.ds.tech_ary[idx].squeeze()  # [0]
                # drawdown_pct = self.ds.get_last_drawdown_pct() * d(100)

                paper_pnl_pct = 0.0
                if self.ds.get_last_position() *  self.ds.get_last_price()[0] > self.ds.min_notional * d(2):
                    paper_pnl_pct = self.ds.get_last_paper_pnl_pct() * d(100)

                position = self.ds.get_last_position()
                kelly_cap_pct = self.ds.get_last_kelly_cap() * 100
                buysell_lvl = self.ds.get_last_buysell_lvl() * 100

                # {cash:>10,.1f}+{asset:>10,.1f}=
                # turb:{turbulence:>4,.1f} |

                position_norm = f"{(position / d(1000)):.4f}k" if position > d(1000) else f"{position:.8f}"
                re = f'{dt_idx} UTC | {idx - 1:>7} | price:{price:>11.6g} | {buysell:>5}' \
                     f' | pos:{position_norm:>12} | ca:{cash_asset:>10,.7g}' \
                     f' | r_pnl:{realized_pnl:>12,.4g}({gain:>5}%) | cap:{kelly_cap_pct:>3,.0f}%' \
                     f' | lvl:{buysell_lvl:>5,.0f}% | rwd:{self.ds.get_last_rewards():>8,.1f}' \
                     f' | p_pnl:{paper_pnl_pct:>6,.3f}%'


                re += self.strategy.render()
                re += f' | cum_pnl:{self.ds.get_last_cumulative_realized_pnl() / self.ds.init_trade_cash * 100:>6,.1f}% | '

                re = re.replace('pos:    0.00', 'pos:        -')
                re = re.replace('rwd:     0.0', 'rwd:       -')
                re = re.replace('p_pnl:  -0.0%', 'p_pnl:      -')
                re = re.replace('p_pnl: 0.000%', 'p_pnl:      -')

                re = re.replace('r_pnl:           0(  0.0%)', 'r_pnl:                   -')
                re = re.replace('r_pnl:         0.0(  0.0%)', 'r_pnl:                   -')

                if idx == int(self.ds.trade_args['stacking_lookback']) + 2:
                    print('=========================================================================================='
                          '=========================================================================================='
                          '=============')
                print(re)

                return re
            elif mode == 'plot':
                # plot_sim(self.opts, obv, self.price_train, self.logger)
                # plt_update()
                pass

        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)
            sys.exit()
