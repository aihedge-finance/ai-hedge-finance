import os
import sys

proj_root = os.path.join(sys.path[0], '../..')
sys.path.insert(1, proj_root)

import time
import math

import tempfile
import traceback
import numpy as np
import pandas as pd
import datetime as dt
import multiprocessing
from copy import deepcopy as cp

import matplotlib.pyplot as plt
from multiprocessing import Process, Pool


from ahf.utils import utils
from ahf.utils.utils import readable_error
from ahf.preprocessor.kf.Tracer_v2 import Tracer_v2
from ahf.preprocessor.kf.TracerSimple import TracerSimple
# from optimizer.model.KF_Estimator import KF_Estimator

one_minus = np.nextafter(np.float32(1), np.float32(0)).item()
zero_plus = np.nextafter(np.float32(0), np.float32(1)).item()


page_counter = 0


class BrunhildNaiveModel_v2_4:
    """

    """
    def __init__(self, opts, logger, verbose=True):
        try:
            self.logger = logger

            if verbose:
                logger.info('[Brunhild] Init BrunhildNaiveModel_v2_4')
            # strategy, exchange, symbol,
            # price_data_path, brunhild_data_path, interval,
            # buy_obs_cov, buy_delta, buy_sd_delta,
            # sell_obs_cov, sell_delta, sell_sd_delta,
            # buy_sd_obs_cov, sell_sd_obs_cov,
            # long_level, short_level, long_exit_level, short_exit_level,
            # exit_rule
            self.verbose = verbose
            self.strategy = opts.strategy
            self.exchange = opts.exchange
            self.symbol = opts.symbol
            self.price_data_path = opts.price_data_path
            self.brunhild_data_path = opts.brunhild_data_path
            self.interval = opts.interval

            self.buy_obs_cov = opts.buy_obs_cov
            self.buy_delta = min(opts.buy_delta, one_minus)
            self.buy_sd_obs_cov = opts.buy_sd_obs_cov
            self.buy_sd_delta = opts.buy_sd_delta

            self.sell_obs_cov = opts.sell_obs_cov
            self.sell_delta = min(opts.sell_delta, one_minus)
            self.sell_sd_obs_cov = opts.sell_sd_obs_cov
            self.sell_sd_delta = opts.sell_sd_delta

            self.indicators = None
            self.long_level = opts.long_level
            self.short_level = opts.short_level

            # self.warm_up_period = opts.warm_up_period
            # self.sd_period = opts.sd_period
            # self.entry_accept_pct = opts.entry_accept_pct

            self.init_trade_cash = self.trade_cash = opts.init_trade_cash
            self.commission = opts.commission

            valid = ['trailing_stop_discrete', 'sell_tracer', 'price_cross']
            if opts.exit_rule not in valid:
                raise Exception(f'Invalid exit rule:{opts.exit_rule}, it must be {valid}')

            # check param
            if self.buy_delta is None:
                raise Exception('buy_delta cannot be None')

            # ==== buy signal ====
            self.BuyIndi = Tracer_v2(self.strategy, 'BuyIndi', self.buy_delta, self.buy_obs_cov,
                                     self.buy_sd_delta, self.buy_sd_obs_cov, self.logger)

            # === sell ====
            self.SellIndi = Tracer_v2(self.strategy, 'SellIndi', self.sell_delta, self.sell_obs_cov,
                                      self.sell_sd_delta, self.sell_sd_obs_cov, self.logger)

            self.obv = OBV()
            self.idx_model = 0
            # print('sss:{0}'.format(vars(opts)))

            self.BuySimple = TracerSimple(self.strategy, 'BuySimple', self.buy_delta, self.buy_obs_cov, self.logger)
            self.SellSimple = TracerSimple(self.strategy, 'SellSimple', self.sell_delta, self.buy_obs_cov, self.logger)

            self.last_10_trades = [True for _ in range(10)]  # True if profitable, False mean lose money
            self.starting_long_asset = 0.
            self.starting_short_asset = 0.

            self.confusion_matrix = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
            self.num_trade = 0

            # Trailing Stop Trade
            if opts.trailing_stop_mode_continuous is None:
                raise Exception('[Brunhild] You have to specify opts.trailing_stop_mode')

            self.trailing_stop_mode_continuous = opts.trailing_stop_mode_continuous
            self.trailing_stop_mode_discrete = opts.trailing_stop_mode_discrete

            if opts.exit_rule == 'trailing_stop_discrete':
                self.trailing_stop_mode_discrete = True

            if self.trailing_stop_mode_discrete:
                opts.exit_rule = 'trailing_stop_discrete'

            if self.trailing_stop_mode_continuous and self.trailing_stop_mode_discrete:
                raise Exception('trailing_stop_mode_continuous and trailing_stop_mode_discrete cannot be both True')


            '''
            BIPS	Percentage	Multiplier
            1       0.01%       0.0001
            10      0.1%        0.001
            100     1%          0.01
            1000	10%         0.1
            e.g. 0.5% => 50
            '''
        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)
            sys.exit()

    def set_trade_cash(self, trade_cash):
        self.init_trade_cash = self.trade_cash = trade_cash

    def build_train_data(self, prices_pd):
        try:

            _, buy_sd_mv, buy_tracer, buy_alfa, buy_alfa_norm_100 = self.BuyIndi.train(prices_pd)
            _, sell_sd_mv, sell_tracer, sell_alfa, sell_alfa_norm_100 = self.SellIndi.train(prices_pd)

            self.idx_model = self.BuyIndi.idx_now

            buy_tracer_pd = pd.Series(buy_tracer, index=prices_pd.index)
            buy_alfa_norm_pd = pd.Series(buy_alfa_norm_100, index=prices_pd.index)
            buy_sd_mv_pd = pd.Series(buy_sd_mv, index=prices_pd.index)
            sell_tracer_pd = pd.Series(sell_tracer, index=prices_pd.index)
            sell_alfa_norm_pd = pd.Series(sell_alfa_norm_100, index=prices_pd.index)
            sell_sd_mv_pd = pd.Series(sell_sd_mv, index=prices_pd.index)

            re_se = [prices_pd, buy_tracer_pd, buy_alfa_norm_pd, buy_sd_mv_pd,
                     sell_tracer_pd, sell_alfa_norm_pd, sell_sd_mv_pd]
            col_names = ['price',
                         'buy_tracer', 'buy_alfa', 'buy_sd_mv',
                         'sell_tracer', 'sell_alfa', 'sell_sd_mv']
            re_pd = pd.concat(re_se, axis=1)
            re_pd.columns = col_names

            re_pd.index.name = 'date'

            self.indicators = re_pd

            return re_pd

        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)

    def sim_trade(self, price_pd, signal_pd, stop_loss_pct, trade_cash, commission, trade_add):
        try:
            n = len(price_pd.index)
            position_pd = pd.Series(np.zeros(n), index=price_pd.index)
            cash = pd.Series(np.zeros(n), index=price_pd.index)
            asset = pd.Series(np.zeros(n), index=price_pd.index)
            share = pd.Series(np.zeros(n), index=price_pd.index)

            starting_long_asset = 0.0
            starting_short_asset = 0.0

            for i, (ind, v) in enumerate(position_pd.iteritems()):
                if i == 0:
                    cash[0] = trade_cash
                    continue

                asset_now = price_pd[i] * share[i - 1]
                # if share is zero, asset will be zero anyway
                pnl = asset_now - starting_long_asset if starting_long_asset > 0 else asset_now - starting_short_asset

                # Exit position if loss
                if position_pd[i - 1] == 1 and pnl <= -trade_cash * stop_loss_pct:
                    signal_pd[i] = 3
                    position_pd[i] = 0
                    starting_long_asset = 0.0
                elif position_pd[i - 1] == -1 and pnl <= -trade_cash * stop_loss_pct:
                    signal_pd[i] = -3
                    position_pd[i] = 0
                    starting_short_asset = 0.0
                else:
                    position_pd[i] = self.cal_pos(signal_pd[i], position_pd[i - 1])

                # calculate share
                if position_pd[i - 1] == 0 and position_pd[i] == 1:
                    share[i] = round(trade_cash / price_pd[i] * 0.98, 2)
                elif position_pd[i - 1] == 0 and position_pd[i] == -1:
                    share[i] = -round(trade_cash / price_pd[i] * 0.98, 2)
                elif position_pd[i - 1] == 1 and position_pd[i] == 0:
                    share[i] = 0.0
                elif position_pd[i - 1] == -1 and position_pd[i] == 0:
                    share[i] = 0.0
                elif position_pd[i] == 0:
                    share[i] = 0.0
                else:
                    share[i] = share[i - 1]

                # Calculate Asset and cash
                asset[i] = price_pd[i] * share[i]

                # Enter position and update status
                if position_pd[i - 1] == 0 and position_pd[i] == 1:
                    starting_long_asset = asset[i]
                    cash[i] = cash[i - 1] - starting_long_asset - (starting_long_asset * commission)
                elif position_pd[i - 1] == 0 and position_pd[i] == -1:
                    starting_short_asset = asset[i]
                    cash[i] = cash[i - 1] - starting_short_asset - (abs(starting_short_asset) * commission)
                elif position_pd[i - 1] == 1 and position_pd[i] == 0:
                    starting_long_asset = 0.0
                    cash[i] = cash[i - 1] + asset_now - (asset_now * commission)

                    trade_cash = cash[i] * 0.98 if trade_add else trade_cash

                elif position_pd[i - 1] == -1 and position_pd[i] == 0:
                    starting_short_asset = 0.0
                    cash[i] = cash[i - 1] + asset_now - (abs(asset_now) * commission)

                    trade_cash = cash[i] * 0.98 if trade_add else trade_cash

                elif position_pd[i] == 0:
                    starting_long_asset = 0.0
                    starting_short_asset = 0.0
                    cash[i] = cash[i - 1]
                else:
                    # starting_long_asset and starting_short_asset remains
                    cash[i] = cash[i - 1]

                sys.stdout.write('\r[Brunhild] Sim Trade Progress: {0:.2f}%'.format(round(i / n * 100, 1)))
                sys.stdout.flush()

            return position_pd, share, cash, asset
        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)
            raise Exception(e)

    def step(self, price_mu_list, price_actual_list, prices_raw, dt_index, last_position_actual=None,
             buy_delta=None, sell_delta=None,
             long_level=None, short_level=None,
             long_exit_level=None, short_exit_level=None, trailing_delta=None):
        try:
            self.buy_delta = buy_delta if buy_delta is not None else self.buy_delta
            self.sell_delta = sell_delta if sell_delta is not None else self.sell_delta
            self.long_level = long_level if long_level is not None else self.long_level
            self.short_level = short_level if short_level is not None else self.short_level
            self.long_exit_level = long_exit_level if long_exit_level is not None else self.long_exit_level
            self.short_exit_level = short_exit_level if short_exit_level is not None else self.short_exit_level
            self.trailing_stop['trailing_delta'] = trailing_delta if trailing_delta is not None else self.trailing_stop[
                'trailing_delta']

            self.buy_delta = min(self.buy_delta, one_minus)
            self.sell_delta = min(self.sell_delta, one_minus)

            price_mu_pct_list = utils.pct_change(price_mu_list, include_first=True)

            if not isinstance(price_mu_pct_list, (np.ndarray, list)):
                raise Exception('Invalid prices type, it must be a list of size 2')

            price_mu_pct = price_mu_pct_list[1]
            price_mu = price_mu_list[1]

            price_actual = price_actual_list[1]

            # update tracer
            _, buy_sd_mv, buy_tracer, _, buy_alfa = self.BuyIndi.update(price_mu_pct, buy_delta)
            _, sell_sd_mv, sell_tracer, _, sell_alfa = self.SellIndi.update(price_mu_pct, sell_delta)

            buy_delta_simple = buy_delta if buy_delta is not None else None  # * 0.1
            sell_delta_simple = sell_delta * 0.1 if sell_delta is not None else None  # * 0.1
            self.BuySimple.update(price_mu, buy_delta_simple)
            self.SellSimple.update(price_mu, sell_delta_simple)

            # signal
            i = self.idx_model
            obv = self.obv
            buy_alfas = self.BuyIndi.alfa_norm_100[i - 1:i + 1]
            buy_tracers = self.BuySimple.tracer[i - 1:i + 1]
            sell_alfas = self.SellIndi.alfa_norm_100[i - 1:i + 1]

            position, share, cash, asset = 0, 0., 0., 0
            realized_pnl = 0.
            buysell = None

            if i == 0:  # self.idx_model <= self.warm_up_period:
                signal = 0
                cash = self.init_trade_cash
                asset = 0.0
                pnl = 0.
                buysell = TradeAction.HOLD

                # starting with full force since we already optimize it off-line
                num_profit_trade = 10
            else:
                last_position = last_position_actual if last_position_actual is not None else self.last_obv_position()
                signal = self.cal_signal_run(buy_alfas, sell_alfas, last_position, price_actual_list, buy_tracers)

                # IMPORTANT
                # we use the last share first because we have not calculated the share now,
                # this will be changed in few steps
                asset_now = price_actual * obv.share[i - 1]
                # if share is zero, asset will be zero anyway
                pnl = asset_now - self.starting_long_asset if self.starting_long_asset > 0 else asset_now - self.starting_short_asset

                # Exit position if loss
                if obv.position[i - 1] == 1 and pnl <= -self.trade_cash * self.stop_loss_pct:
                    signal = 3
                    position = 0
                    # self.starting_long_asset = 0.0
                elif obv.position[i - 1] == -1 and pnl <= -self.trade_cash * self.stop_loss_pct:
                    signal = -3
                    position = 0
                    # self.starting_short_asset = 0.0
                else:
                    position = self.cal_pos(signal, obv.position[i - 1])

                # calculate share
                if obv.position[i - 1] == 0 and position == 1:
                    share = round(self.trade_cash / price_actual * 0.98, 2)
                elif obv.position[i - 1] == 0 and position == -1:
                    share = -round(self.trade_cash / price_actual * 0.98, 2)
                elif obv.position[i - 1] == 1 and position == 0:
                    share = 0.0
                elif obv.position[i - 1] == -1 and position == 0:
                    share = 0.0
                elif position == 0:
                    share = 0.0
                else:
                    share = obv.share[i - 1]

                # Calculate Asset and cash
                asset = price_actual * share
                num_profit_trade = sum(self.last_10_trades)

                # Enter position and update status
                if obv.position[i - 1] == 0 and position == 1:
                    buysell = TradeAction.BUY

                    # Execute Buy

                    # === Market Order ===
                    # print(vars(self.last_value()))

                    asset = entry_market_sim(share, buysell, price_actual, self.logger)

                    # === Trailing Stop ===
                    if self.trailing_stop_mode_discrete:
                        # Init
                        self.trailing_stop['active'] = True
                        self.trailing_stop['price_trailing'] = price_actual

                    if self.trailing_stop_mode_continuous:
                        trailing_stop_share = share
                        self.trailing_stop['order']['buysell'] = TradeAction.SELL

                        executed, asset_exit, \
                        pnl = self.exit_stop_loss_continuous(trailing_stop_share, obv.position[i - 1],
                                                             self.trailing_stop['order']['buysell'],
                                                             prices_raw, price_actual,
                                                             self.trailing_stop['trailing_delta'],
                                                             self.starting_long_asset, self.starting_short_asset,
                                                             obv.cash[i - 1], self.logger)

                        if executed:
                            # append trade history, shrink trade_cash if fail many times
                            self._update_num_profit_trade(TradeAction.SELL, realized_pnl, obv.cash[i - 1])

                    self.starting_long_asset = asset
                    self.entry_commission = (self.starting_long_asset * self.commission)
                    cash = obv.cash[i - 1] - self.starting_long_asset - self.entry_commission

                elif obv.position[i - 1] == 0 and position == -1:
                    buysell = TradeAction.SHORT

                    # Execute Short
                    if self.trailing_stop_mode_discrete:
                        # Init
                        self.trailing_stop['active'] = True
                        self.trailing_stop['price_trailing'] = price_actual

                    # === Market Order ===
                    asset = entry_market_sim(share, buysell, price_actual, self.logger)

                    # === Trailing Stop ===
                    if self.trailing_stop_mode_continuous:
                        trailing_stop_share = share
                        self.trailing_stop['order']['buysell'] = TradeAction.COVER

                        executed, asset_exit, \
                        pnl = self.exit_stop_loss_continuous(trailing_stop_share, obv.position[i - 1],
                                                             self.trailing_stop['order']['buysell'],
                                                             prices_raw, price_actual,
                                                             self.trailing_stop['trailing_delta'],
                                                             self.starting_long_asset, self.starting_short_asset,
                                                             obv.cash[i - 1], self.logger)

                        if executed:
                            # append trade history, shrink trade_cash if fail many times
                            self._update_num_profit_trade(TradeAction.COVER, realized_pnl, obv.cash[i - 1])

                    self.starting_short_asset = asset
                    self.entry_commission = abs(self.starting_short_asset) * self.commission
                    cash = obv.cash[i - 1] - self.starting_short_asset - self.entry_commission

                elif obv.position[i - 1] == 1 and position == 0:
                    buysell = TradeAction.SELL

                    # Execute sell

                    # last_share, target_price, buysell, prices_raw
                    # DO IT ONLY WHEN NOT EXISTING WITH LOSS
                    if signal == 3 or signal == -3:
                        asset_exit = asset_now
                    else:
                        # === Limit Order ===
                        # default, market price exit, if following exit plan not specified
                        asset_exit = asset_now

                        last_share = obv.share[i - 1]
                        '''
                        # === Market Order ===
                        asset_exit, pnl = exit_market_sim(last_share, buysell, price_actual, 
                                                     self.starting_long_asset, self.starting_short_asset, self.logger)
                        
                        # === Limit Order ===
                        good_price_range, asset_exit, \
                        pnl = exit_market_sim(last_share, price_mu, buysell,
                                                      prices_raw, price_actual,
                                                      self.starting_long_asset, self.starting_short_asset, self.logger)
                        # === Stop Loss ===
                        executed, asset_exit, \
                        pnl = self.exit_stop_loss_continuous(last_share, buysell,
                                                  prices_raw, price_actual, self.trailing_stop['trailing_delta'],
                                                  self.starting_long_asset, self.starting_short_asset, 
                                                  obv.cash[i - 1], self.logger)
                        '''

                    self.starting_long_asset = 0.0
                    cash = obv.cash[i - 1] + asset_exit - (asset_exit * self.commission)

                    self.trade_cash = cash  # if self.trade_add else self.trade_cash

                    # realized pnl
                    realized_pnl = pnl - (asset_exit * self.commission) - self.entry_commission
                    self.entry_commission = 0

                    # append trade history, shrink trade_cash if fail many times
                    self._update_num_profit_trade(TradeAction.SELL, realized_pnl, cash)

                elif obv.position[i - 1] == -1 and position == 0:
                    buysell = TradeAction.COVER

                    # Execute Cover
                    # last_share, target_price, buysell, prices_raw
                    # DO IT ONLY WHEN NOT EXISTING WITH LOSS
                    if signal == 3 or signal == -3:
                        asset_exit = asset_now
                    else:
                        # Execute Cover
                        # default, market price exit, if following exit plan not specified
                        asset_exit = asset_now

                        last_share = obv.share[i - 1]

                    self.starting_short_asset = 0.0

                    cash = obv.cash[i - 1] + asset_exit - (abs(asset_exit) * self.commission)

                    # realized pnl, how about fee at entry??
                    realized_pnl = pnl - (abs(asset_exit) * self.commission) - self.entry_commission
                    self.entry_commission = 0.

                    self.trade_cash = cash  # if self.trade_add else self.trade_cash

                    # append trade history, shrink trade_cash if fail many times
                    self._update_num_profit_trade(TradeAction.COVER, realized_pnl, cash)

                elif position == 0:
                    buysell = TradeAction.HOLD
                    self.starting_long_asset = 0.0
                    self.starting_short_asset = 0.0
                    cash = obv.cash[i - 1]

                else:
                    if self.trailing_stop_mode_discrete:
                        # Init
                        if last_position == 1 and price_actual > self.trailing_stop['price_trailing']:
                            self.trailing_stop['price_trailing'] = price_actual

                        if last_position == -1 and price_actual < self.trailing_stop['price_trailing']:
                            self.trailing_stop['price_trailing'] = price_actual

                    if self.trailing_stop_mode_continuous and self.trailing_stop['active']:
                        # === Trailing Stop ===
                        trailing_stop_share = last_share = obv.share[i - 1]

                        executed, asset_exit, \
                        pnl = self.exit_stop_loss_continuous(trailing_stop_share, obv.position[i - 1],
                                                             self.trailing_stop['order']['buysell'],
                                                             prices_raw, price_actual,
                                                             self.trailing_stop['trailing_delta'],
                                                             self.starting_long_asset, self.starting_short_asset,
                                                             obv.cash[i - 1], self.logger)

                        if executed:
                            buysell = None
                            realized_pnl = pnl
                            # append trade history, shrink trade_cash if fail many times
                            self._update_num_profit_trade(TradeAction.SELL if last_position == 1 else TradeAction.COVER,
                                                          realized_pnl, self.trailing_stop['order']['cash'])
                        # obv.starting_long_asset and obv.starting_short_asset remains
                    else:
                        cash = obv.cash[i - 1]
                        buysell = TradeAction.HOLD

            my_obv = None
            # if buysell == TradeAction.BUY or buysell == TradeAction.SHORT or buysell == TradeAction.HOLD:
            if buysell is not None:
                obv.append(price_mu, price_actual, dt_index, signal, position, share, cash, asset, buysell, pnl,
                           self.starting_long_asset, self.starting_short_asset, realized_pnl, num_profit_trade,
                           self.long_level, self.short_level, self.long_exit_level, self.short_exit_level,
                           self.num_trade, self.logger)

                self.idx_model += 1
                my_obv = self.last_value()

            my_obv_trailing_stop = None
            # and self.trailing_stop['order']['price_executed'] > 0:
            if self.trailing_stop_mode_continuous and self.trailing_stop['active'] and buysell is None:
                buysell = self.trailing_stop['order']['buysell']
                signal = self.trailing_stop['order']['signal']
                price_actual = max(self.trailing_stop['order']['price_executed'], price_actual)
                realized_pnl = pnl = self.trailing_stop['order']['realized_pnl']
                asset = self.trailing_stop['order']['asset']
                cash = self.trailing_stop['order']['cash']
                share = self.trailing_stop['order']['share']
                position = self.trailing_stop['order']['position']

                if self.trailing_stop['order']['price_executed'] > 0:
                    self._clear_trailing_stop_order()

                obv.append(price_mu, price_actual, dt_index, signal, position, share, cash, asset, buysell, pnl,
                           self.starting_long_asset, self.starting_short_asset, realized_pnl, num_profit_trade,
                           self.long_level, self.short_level, self.long_exit_level, self.short_exit_level,
                           self.num_trade, self.logger)

                self.idx_model += 1
                my_obv_trailing_stop = self.last_value()

            if my_obv is None and my_obv_trailing_stop is None:
                raise Exception('my_obv and my_obv_trailing_stop cannot both be not happening')

            return my_obv, my_obv_trailing_stop

        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)
            sys.exit()

    def _update_num_profit_trade(self, executed_trade_action, realized_pnl, cash):
        try:
            # append trade history, shrink trade_cash if fail many times
            self.last_10_trades.append(True if realized_pnl > 0 else False)
            if len(self.last_10_trades) > 10:
                self.last_10_trades.pop(0)

            num_profit_trade = sum(self.last_10_trades)
            if num_profit_trade < 5:
                self.trade_cash = round(cash * float(num_profit_trade + 0.1) / 10.5, 1)
            else:
                self.trade_cash = cash

            if executed_trade_action == TradeAction.SELL:
                self.num_trade += 1
                if realized_pnl > 0:
                    self.confusion_matrix['tp'] += 1
                else:
                    self.confusion_matrix['fp'] += 1
            elif executed_trade_action == TradeAction.COVER:
                self.num_trade += 1
                if realized_pnl > 0:
                    self.confusion_matrix['tn'] += 1
                else:
                    self.confusion_matrix['fn'] += 1
        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)
            sys.exit()

    def _update_trailing_stop_order(self, buysell, signal, position, share, price_executed, realized_pnl,
                                    asset_exit, cash):
        self.trailing_stop['order'] = {
            'buysell': buysell,
            'signal': signal,
            'position': position,
            'share': share,
            'price_executed': price_executed,
            'realized_pnl': realized_pnl,
            'asset': asset_exit,
            'cash': cash,
        }

    def _clear_trailing_stop_order(self):
        self.trailing_stop['order'] = {
            'buysell': TradeAction.HOLD,
            'signal': 0,
            'position': 0,
            'share': 0,
            'price_executed': 0.,
            'realized_pnl': 0,
            'asset': 0,
            'cash': 0,
        }
        self.trailing_stop['active'] = False

    def exit_stop_loss_continuous(self, last_share, last_position, stop_loss_buysell, prices_raw, price_actual,
                                  trailing_delta, starting_long_asset, starting_short_asset, last_cash, logger):
        try:
            # Init
            self.trailing_stop['active'] = True
            self.trailing_stop['price_trailing'] = price_actual
            executed = False

            # assert
            if stop_loss_buysell != TradeAction.SELL and stop_loss_buysell != TradeAction.COVER:
                raise Exception('Stop_loss buysell can only be SELL or COVER while existing')

            # pre-cal required result
            # asset_now = price_actual * last_share
            # pnl = asset_now - starting_long_asset if starting_long_asset > 0 else asset_now - starting_short_asset

            prices_raw_len = len(prices_raw)

            # default value
            price_executed = price_actual

            if prices_raw_len == 0:
                # sometimes there is no price due to Binance system problem, we just go around it
                price_executed = price_actual
                self.logger.warning('[Brunhild] prices_raw_len is zero at {0}'.format(self.idx_model))

            for i, current_price in enumerate(prices_raw):
                # deal with last round
                if self.tstop_exit_max_round == 1 and i >= prices_raw_len - 1:  # only one interval
                    price_executed = current_price
                    executed = True
                    break
                elif i >= prices_raw_len - 1 and self.tstop_exit_max_round > 1:
                    # till next round
                    price_executed = price_actual
                    executed = False
                    break

                # continue if not last round
                # if not last round
                self._update_trailing_stop_price(current_price, stop_loss_buysell)
                if stop_loss_buysell == TradeAction.SELL:
                    price_cutoff = self.trailing_stop['price_trailing'] * (1 - trailing_delta / 10000)

                    if price_cutoff >= current_price:
                        price_executed = current_price
                        executed = True
                        break
                    else:
                        continue
                else:  # buysell == TradeAction.COVER:
                    price_cutoff = self.trailing_stop['price_trailing'] * (1 + trailing_delta / 10000)

                    if price_cutoff <= current_price:
                        price_executed = current_price
                        executed = True
                        break
                    else:
                        continue

            # we use the last share first because we have not calculated the share now
            asset_now = price_executed * last_share
            # if share is zero, asset will be zero anyway
            if executed:
                # self.trailing_stop['order']['buysell'] = TradeAction.HOLD
                if stop_loss_buysell == TradeAction.SELL:
                    signal = 4
                    position = 0
                    share = 0
                    self.starting_long_asset = 0.0
                else:  # buysell == TradeAction.COVER:
                    signal = -4
                    position = 0
                    share = 0
                    self.starting_short_asset = 0.0

                cash = last_cash + asset_now - (asset_now * self.commission)

                pnl = asset_now - starting_long_asset if starting_long_asset > 0 else asset_now - starting_short_asset

                # asset become zero after updated cash and pnl
                asset = 0.

                self._end_trailing_stop()

            else:
                stop_loss_buysell = stop_loss_buysell
                price_executed = 0.0
                signal = 0
                position = last_position
                cash = last_cash
                share = last_share
                pnl = 0.
                asset = asset_now

            # update stop_loss order
            self._update_trailing_stop_order(stop_loss_buysell, signal, position, share, price_executed, pnl, asset,
                                             cash)

            return executed, asset_now, pnl
        except Exception as e:
            err = readable_error(e, __file__)
            logger.error(err)
            sys.exit()

    def _update_trailing_stop_price(self, current_price, trade_action):
        self.trailing_stop['active'] = True

        if trade_action != TradeAction.SELL and trade_action != TradeAction.COVER:
            raise Exception('buysell can only be SELL or COVER while existing')

        if trade_action == TradeAction.SELL:
            if self.trailing_stop['price_trailing'] <= current_price:
                self.trailing_stop['price_trailing'] = current_price
        else:
            if self.trailing_stop['price_trailing'] >= current_price:
                self.trailing_stop['price_trailing'] = current_price

        self.trailing_stop['order']['buysell'] = trade_action

    def _end_trailing_stop(self):
        self.trailing_stop['price_trailing'] = 0.
        self.trailing_stop['order']['buysell'] = TradeAction.HOLD

    def get_obv(self):
        try:
            i = self.obv.idx_obv
            obv = lambda: None

            obv.idx_obv = i
            obv.price_mu = self.obv.price_mu[:i]
            obv.price_actual = self.obv.price_actual[:i]
            obv.dt_index = self.obv.dt_index[:i]
            obv.signal = self.obv.signal[:i]
            obv.position = self.obv.position[:i]
            obv.cash = self.obv.cash[:i]
            obv.asset = self.obv.asset[:i]
            obv.cash_asset = self.obv.cash_asset[:i]
            obv.share = self.obv.share[:i]
            obv.pnl = self.obv.pnl[:i]
            obv.realized_pnl = self.obv.realized_pnl[:i]
            obv.accum_realized_pnl = self.obv.accum_realized_pnl[:i]
            obv.buysell = self.obv.buysell[:i]
            obv.long_level = self.obv.long_level[:i]
            obv.short_level = self.obv.short_level[:i]
            obv.long_exit_level = self.obv.long_exit_level[:i]
            obv.short_exit_level = self.obv.short_exit_level[:i]

            buy_indi = self.buy_indi()

            obv.buy_tracer = buy_indi.tracer[:i]
            obv.buy_alfa = buy_indi.alfa[:i]
            obv.buy_sd = buy_indi.sd[:i]
            obv.buy_delta = buy_indi.delta[:i]

            sell_indi = self.sell_indi()
            obv.sell_tracer = sell_indi.tracer[:i]
            obv.sell_alfa = sell_indi.alfa[:i]
            obv.sell_sd = sell_indi.sd[:i]
            obv.sell_delta = sell_indi.delta[:i]

            buy_simple = self.buy_simple()
            obv.buy_simple = buy_simple.tracer[:i]

            sell_simple = self.sell_simple()
            obv.sell_simple = sell_simple.tracer[:i]

            obv.num_profit_trade = sum(self.last_10_trades)

            return obv
        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)
            sys.exit()

    def last_obv_position(self):
        i = self.idx_model
        return self.obv.position[i - 1]

    def last_2values(self):
        i = self.idx_model
        obv = lambda: None

        obv.idx_obv = i
        obv.num_trade = self.num_trade
        obv.price_mu = self.obv.price_mu[i - 2:i]
        obv.price_actual = self.obv.price_actual[i - 2:i]
        obv.dt_index = self.obv.dt_index[i - 2:i]
        obv.signal = self.obv.signal[i - 2:i]
        obv.position = self.obv.position[i - 2:i]
        obv.cash = self.obv.cash[i - 2:i]
        obv.asset = self.obv.asset[i - 2:i]
        obv.cash_asset = self.obv.cash_asset[i - 2:i]
        obv.share = self.obv.share[i - 2:i]
        obv.pnl = self.obv.pnl[i - 2:i]
        obv.realized_pnl = self.obv.realized_pnl[i - 2:i]
        obv.accum_realized_pnl = self.obv.accum_realized_pnl[i - 2:i]
        obv.buysell = self.obv.buysell[i - 2:i]
        obv.long_level = self.obv.long_level[i - 2:i]
        obv.short_level = self.obv.short_level[i - 2:i]
        obv.long_exit_level = self.obv.long_exit_level[i - 2:i]
        obv.short_exit_level = self.obv.short_exit_level[i - 2:i]

        buy_indi = self.buy_indi()

        obv.buy_tracer = buy_indi.tracer[i - 2:i]
        obv.buy_alfa = buy_indi.alfa[i - 2:i]
        obv.buy_sd = buy_indi.sd[i - 2:i]
        obv.buy_delta = buy_indi.delta[i - 2:i]

        sell_indi = self.sell_indi()
        obv.sell_tracer = sell_indi.tracer[i - 2:i]
        obv.sell_alfa = sell_indi.alfa[i - 2:i]
        obv.sell_sd = sell_indi.sd[i - 2:i]
        obv.sell_delta = sell_indi.delta[i - 2:i]

        buy_simple = self.buy_simple()
        obv.buy_simple = buy_simple.tracer[i - 2:i]

        sell_simple = self.sell_simple()
        obv.sell_simple = sell_simple.tracer[i - 2:i]

        obv.num_profit_trade = sum(self.last_10_trades)

        return obv

    def last_value(self):
        try:
            i = self.idx_model
            obv = lambda: None

            obv.idx_obv = i
            obv.num_trade = self.num_trade
            obv.price_mu = self.obv.price_mu[i - 1]
            obv.price_actual = self.obv.price_actual[i - 1]
            obv.dt_index = self.obv.dt_index[i - 1]
            obv.signal = self.obv.signal[i - 1]
            obv.position = self.obv.position[i - 1]
            obv.cash = self.obv.cash[i - 1]
            obv.asset = self.obv.asset[i - 1]
            obv.cash_asset = self.obv.cash_asset[i - 1]
            obv.share = self.obv.share[i - 1]
            obv.pnl = self.obv.pnl[i - 1]
            obv.realized_pnl = self.obv.realized_pnl[i - 1]
            obv.accum_realized_pnl = self.obv.accum_realized_pnl[i - 1]
            obv.buysell = self.obv.buysell[i - 1]
            obv.long_level = self.obv.long_level[i - 1]
            obv.short_level = self.obv.short_level[i - 1]
            obv.long_exit_level = self.obv.long_exit_level[i - 1]
            obv.short_exit_level = self.obv.short_exit_level[i - 1]
            obv.long_level = self.obv.long_level[i - 1]
            obv.short_level = self.obv.short_level[i - 1]
            obv.long_exit_level = self.obv.long_exit_level[i - 1]
            obv.short_exit_level = self.obv.short_exit_level[i - 1]

            buy_indi = self.buy_indi()

            obv.buy_tracer = buy_indi.tracer[i - 1]
            obv.buy_alfa = buy_indi.alfa[i - 1]
            obv.buy_sd = buy_indi.sd[i - 1]
            obv.buy_delta = buy_indi.delta[i - 1]

            sell_indi = self.sell_indi()
            obv.sell_tracer = sell_indi.tracer[i - 1]
            obv.sell_alfa = sell_indi.alfa[i - 1]
            obv.sell_sd = sell_indi.sd[i - 1]
            obv.sell_delta = sell_indi.delta[i - 1]

            buy_simple = self.buy_simple()
            obv.buy_simple = buy_simple.tracer[i - 1]

            sell_simple = self.sell_simple()
            obv.sell_simple = sell_simple.tracer[i - 1]

            obv.num_profit_trade = sum(self.last_10_trades)

            obv.confusion_matrix = self.confusion_matrix

            return obv
        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)
            sys.exit()

    def buy_indi(self):
        return self.BuyIndi.value()

    def sell_indi(self):
        return self.SellIndi.value()

    def buy_simple(self):
        return self.BuySimple.value()

    def sell_simple(self):
        return self.SellSimple.value()


class OBV(object):
    def __init__(self):
        self.idx_obv = 0  # for controlling data size
        self.num_trade = 0

        n = 10000

        self.price_mu = np.zeros(n)
        self.price_actual = np.zeros(n)
        self.dt_index = np.zeros(n, dtype='datetime64[ms]')
        self.signal = np.zeros(n, dtype='int8')
        self.position = np.zeros(n, dtype='int8')
        self.cash = np.zeros(n)
        self.asset = np.zeros(n)
        self.cash_asset = np.zeros(n)
        self.share = np.zeros(n)
        self.buysell = np.zeros(n, dtype=TradeAction)
        self.pnl = np.zeros(n)
        self.starting_long_asset = np.zeros(n)
        self.starting_short_asset = np.zeros(n)
        self.realized_pnl = np.zeros(n)
        self.accum_realized_pnl = np.zeros(n)
        self.num_profit_trade = np.zeros(n)
        self.confusion_matrix = np.zeros((2, 2))
        self.long_level = np.zeros(n)
        self.short_level = np.zeros(n)
        self.long_exit_level = np.zeros(n)
        self.short_exit_level = np.zeros(n)

    def append(self, price_mu, price_actual, dt_index, signal, position, share, cash, asset, buysell, pnl,
               starting_long_asset, starting_short_asset, realized_pnl, num_profit_trade,
               long_level, short_level, long_exit_level, short_exit_level, num_trade, logger):
        try:
            i = self.idx_obv
            self.dt_index[i] = dt_index
            self.price_mu[i] = price_mu
            self.price_actual[i] = price_actual
            self.signal[i] = signal
            self.position[i] = position
            self.share[i] = share
            self.cash[i] = cash
            self.asset[i] = asset
            self.cash_asset[i] = cash + asset
            self.buysell[i] = buysell
            self.pnl[i] = pnl
            self.starting_long_asset = starting_long_asset
            self.starting_short_asset = starting_short_asset
            self.realized_pnl[i] = realized_pnl
            self.accum_realized_pnl[i] = self.accum_realized_pnl[i - 1] + realized_pnl
            self.num_profit_trade[i] = num_profit_trade
            self.long_level[i] = long_level
            self.short_level[i] = short_level
            self.long_exit_level[i] = long_exit_level
            self.short_exit_level[i] = short_exit_level

            self.idx_obv += 1
            self.num_trade = num_trade

            if self.idx_obv >= len(self.price_mu):
                n = 10000
                self.price_mu = np.append(self.price_mu, np.zeros(n))
                self.price_actual = np.append(self.price_actual, np.zeros(n))
                self.dt_index = np.append(self.dt_index, np.zeros(n, dtype='datetime64[ms]'))
                self.signal = np.append(self.signal, np.zeros(n, dtype='int8'))
                self.position = np.append(self.position, np.zeros(n, dtype='int8'))
                self.cash = np.append(self.cash, np.zeros(n))
                self.asset = np.append(self.asset, np.zeros(n))
                self.cash_asset = np.append(self.cash_asset, np.zeros(n))
                self.share = np.append(self.share, np.zeros(n))
                self.buysell = np.append(self.buysell, np.zeros(n, dtype=TradeAction))
                self.pnl = np.append(self.pnl, np.zeros(n))
                self.starting_long_asset = np.append(self.starting_long_asset, np.zeros(n))
                self.starting_short_asset = np.append(self.starting_short_asset, np.zeros(n))
                self.realized_pnl = np.append(self.realized_pnl, np.zeros(n))
                self.accum_realized_pnl = np.append(self.accum_realized_pnl, np.zeros(n))
                self.num_profit_trade = np.append(self.num_profit_trade, np.zeros(n))
                self.long_level = np.append(self.long_level, np.zeros(n))
                self.short_level = np.append(self.short_level, np.zeros(n))
                self.long_exit_level = np.append(self.long_exit_level, np.zeros(n))
                self.short_exit_level = np.append(self.short_exit_level, np.zeros(n))

        except Exception as e:
            err = readable_error(e, __file__)
            logger.error(err)
            sys.exit()

    def get_last(self):
        pass


class OPTS(object):
    def __init__(self):
        self.init_obs_cov = None
        self.strategy = None
        self.exchange = None
        self.symbol = None
        self.price_data_path = None
        self.brunhild_data_path = None
        self.interval = None
        self.form_start = None
        self.buy_obs_cov = None
        self.buy_delta = None
        self.buy_sd_delta = None
        self.buy_ref_sd_delta = None
        self.sell_obs_cov = None
        self.sell_delta = None
        self.sell_sd_delta = None
        self.buy_sd_obs_cov = None
        self.sell_sd_obs_cov = None
        self.long_level = None
        self.short_level = None
        self.long_exit_level = None
        self.short_exit_level = None
        self.stop_loss_pct = None
        self.exit_rule = None
        self.trade_add = None

        self.tuning = None
        self.num_cpu = None
        self.confusion_matrix = None
        self.cash_asset = None


def default_data_init(symbol=None, long_exit_level=None, short_exit_level=None):
    opts = OPTS()

    # symbol
    opts.symbol = symbol if symbol is not None else opts.symbol
    opts.long_exit_level = long_exit_level if long_exit_level is not None else opts.long_exit_level
    opts.short_exit_level = short_exit_level if short_exit_level is not None else opts.short_exit_level

    # Hyper-parameters
    opts.dynamic_sd = False
    opts.stop_loss_pct = 0.04
    opts.buy_obs_cov = 5
    opts.sell_obs_cov = 5

    opts.buy_delta = 0.6  # 0.01
    opts.sell_delta = 0.08  # 0.0001

    opts.buy_sd_delta = 1e-6
    opts.sell_sd_delta = 1e-6

    # opts.buy_sell_delta_diff = -0.2  # -0.8
    # a = math.log(opts.buy_delta, 0.5)
    # b = 0.5 ** (a + opts.buy_sell_delta_diff)  # 0.57456
    # opts.sell_delta = b

    window_size = 11
    opts.stop_loss_pct = 0.05

    opts.buy_sd_obs_cov, opts.sell_sd_obs_cov = 5., 5.  # DO NOT CHANGE

    # App specific Paramters
    opts.strategy = 'brunhild_naive_v2_4'
    opts.init_trade_cash = 10000
    opts.exchange = 'Binance'
    opts.symbol = 'SOLUSDT'
    opts.interval = '30T'
    opts.long_short_dual = 'long'

    opts.commission = 1 / 1000
    opts.sd_period = 100  # we estimate around 100 days to 180 to become more stable

    opts.price_data_path = './appData/trainData_crypto/prices_v3.parquet'
    opts.brunhild_data_path = './appData/trainData_crypto/Brunhild_naive.parquet'

    opts.trade_add = True
    opts.exit_rule = 'sell_tracer'  # sell_tracer, 'price_cross, trailing_stop_discrete
    opts.form_start = '2021-03-01'
    opts.form_end = '2022-06-04'
    opts.buy_delta, opts.sell_delta = 0.05, 0.01
    opts.long_level, opts.short_level = 0.15, -0.
    opts.long_exit_level, opts.short_exit_level = .1, -0.

    opts.trailing_stop_mode_continuous = False
    opts.trailing_stop_mode_discrete = False

    if opts.exit_rule == 'trailing_stop_discrete':
        opts.trailing_delta = 300
        opts.stop_loss_pct = 0.01
    else:
        opts.trailing_delta = 600

    # KF_Estimator
    opts.tuning = True
    opts.WINDOW_SIZE = 30
    opts.POLY_ORDER = 3
    opts.output_dir = 'output/SMAC3/'
    opts.temp_dir = tempfile.mkdtemp()

    opts.interval_raw = '1T'
    opts.num_interval_eval = 1
    opts.interval_eval = '{}T'.format(int(utils.convert_to_min(opts.interval) * opts.num_interval_eval))  # '1h'

    # self.look_backs = ['12h', '3h', '1h']
    opts.kf_est_data_path = f'./appData/trainData_crypto/kf_delta_est_{opts.interval_eval}_v1.parquet'

    opts.period_look_back = '1d'
    opts.n_look_back = int(utils.convert_to_min(opts.period_look_back) / utils.convert_to_min(opts.interval_eval))

    opts.init_delta = 0.1  # 0.0035866597
    opts.init_obs_cov = 5
    opts.num_cpu = multiprocessing.cpu_count() * 5 // 6

    opts.kf_est_is_train = True

    # Deep Learning params
    net_name = 'fc'  # 'lenet5'
    epochs = 80
    batch_size = 32

    ml_params = {"input_w": 15, "input_h": 15, "num_classes": 3, "batch_size": batch_size, "epochs": epochs,
                 'net_name': net_name}

    return opts, window_size, ml_params


'''
def evaluator(opts, buy_delta, buy_sell_delta_diff):
    # USAGE, get your default from default_data_init and then change to whatever you want and plug
    # into evaluator function
    # buy_delta, buy_sell_delta_diff are to replace the original opts data
    # Example: use below outside evaluator
    # opts, window_size, ml_params = default_data_init()

    opts.buy_delta = buy_delta
    opts.buy_sell_delta_diff = buy_sell_delta_diff
    a = math.log(opts.buy_delta, 0.5)
    b = 0.5 ** (a + opts.buy_sell_delta_diff)  # 0.57456
    opts.sell_delta = b
'''


def evaluator(opts, buy_delta, sell_delta):
    opts.buy_delta = buy_delta
    opts.sell_delta = sell_delta

    # opts.symbol = symbol
    # opts.long_exit_level, opts.short_exit_level = long_exit_level, short_exit_level

    re = run_sim(opts, verbose=False)

    return re['gain']


def _render(obv):
    idx_obv = obv.idx_obv

    cash_asset = obv.cash_asset
    realized_pnl = obv.realized_pnl
    gain = realized_pnl / cash_asset * 100
    gain_sign = '+' if gain > 0 else ''

    d = round((obv.price_actual - obv.price_mu) / obv.price_mu * 100, 2) if obv.signal == 1 else ''
    d = round(-(obv.price_actual - obv.price_mu) / obv.price_mu * 100, 2) if obv.signal == -1 else d

    re = '>> {0} UTC | step:{1:>7} | prc:{2:>12.4f} | sig:{3:>2} | pos:{4:>5} ' \
         '| ca:{5:>10,.2f} | realized:{6:>9.2f}({7:>4}%) ' \
         '| b_del:{8:>5,.5f} | s_del:{9:>5,.5f} | long:{10:>7.3f} | long:{11:>7.3f}' \
         ''.format(obv.dt_index.astype(dt.datetime).strftime("%Y-%m-%d %H:%M"),
                   idx_obv - 1, obv.price_actual, int(obv.signal),
                   obv.position, cash_asset, realized_pnl, gain_sign + str(round(gain, 1)),
                   obv.buy_delta, obv.sell_delta, obv.long_level, obv.long_exit_level)

    re = re.replace('signal: 0', 'signal: -')
    re = re.replace('pos:  0', 'pos:  -')
    re = re.replace('diff:       %', 'diff:      -')

    re = re.replace('realized_pnl:      0.00( 0.0%)', 'realized_pnl:              - ')
    re = re.replace(' | s_delta:0.999', ' |             -')
    re = re.replace(' | b_delta:0.999', ' |             -')

    print(re)

    return re


def run_sim(opts,
            prices_mu_train=None,
            prices_actual_train=None,
            prices_raw_train=None,
            logger=None,
            verbose=True,
            csv=None,
            parquet_file=None):
    try:
        if not (opts.strategy == 'brunhild_ppo' or opts.strategy == 'brunhild_naive_v2_4'):
            logger.error('Wrong TradingStrategy ,expect brunhild_ppo but got {}'.format(opts.strategy))
            sys.exit()

        logger = utils.setup_logger('brunhild_train.log', opts.symbol) if logger is None else logger

        if verbose:
            logger.info('=================================================')
            logger.info('Start BrunhildNaiveModel_v2_3 Unit Test:\n{0}'.format(vars(opts)))

        if prices_mu_train is None or prices_actual_train is None or prices_raw_train is None:
            prices_mu_train, prices_actual_train, prices_raw_train, \
            price_pd_interval_half, _ = helper.load_price(opts.exchange, opts.symbol,
                                                                               opts.price_data_path,
                                                                               opts.interval,
                                                                               opts.interval_raw,
                                                                               opts.interval_eval, logger)

            if 'form_end' in vars(opts):
                prices_mu_train = prices_mu_train[opts.form_start:opts.form_end]
                prices_actual_train = prices_actual_train[opts.form_start:opts.form_end]
                prices_raw_train = prices_raw_train[opts.form_start:opts.form_end]
            else:
                prices_mu_train = prices_mu_train[opts.form_start:]
                prices_actual_train = prices_actual_train[opts.form_start:]
                prices_raw_train = prices_raw_train[opts.form_start:]

        if len(prices_mu_train.index) == 0:
            logger.error('[Brunhild] Price is empty')
            sys.exit()

            # Using update method, not train method
        actor = BrunhildNaiveModel_v2_4(opts, logger, verbose)
        # est = KF_Estimator(opts, opts.kf_est_is_train, logger, False)

        # delta_est = np.zeros(len(prices_mu_train.index))

        for i, (dt_idx, price_mu) in enumerate(prices_mu_train.iteritems()):
            if i == 0:
                prices_mu = [price_mu, price_mu]
                prices_actual = [prices_actual_train[0], prices_actual_train[0]]

                dt_idx_start = dt_idx
                dt_idx_end = dt_idx_start + pd.Timedelta(opts.interval)
                prices_raw = prices_raw_train[dt_idx_start:dt_idx_end]
            else:
                prices_mu = [prices_mu_train[i - 1], price_mu]
                prices_actual = [prices_actual_train[i - 1], prices_actual_train[i]]

                dt_idx_start = dt_idx
                dt_idx_end = dt_idx_start + pd.Timedelta(opts.interval)
                prices_raw = prices_raw_train[dt_idx_start:dt_idx_end]

            '''
            if opts.tuning:
                # re = est.step(dt_idx)
                # delta_est[i] = re['delta'][0] if re is not None else None
                # buy_delta_new = delta_est[i] * 0.1
                # sell_delta_new = 0.1 if buy_delta_new is not None else None

                my_obv, my_obv_trailing_stop = actor.step(prices_mu, prices_actual, prices_raw, dt_idx)  # buy_delta=buy_delta_new, sell_delta=sell_delta_new)
            else:
            '''
            my_obv, my_obv_trailing_stop = actor.step(prices_mu, prices_actual, prices_raw, dt_idx)

            if verbose:
                if my_obv is not None:
                    _render(my_obv)
                if my_obv_trailing_stop is not None:
                    _render(my_obv_trailing_stop)

        obv = actor.get_obv()

        buy_indi = actor.buy_indi()
        sell_indi = actor.sell_indi()

        buy_simple = actor.buy_simple()
        sell_simple = actor.sell_simple()

        # confusion matrix
        cm = my_obv.confusion_matrix
        b = cm['tp'] + cm['tn'] + cm['fp'] + cm['fn']
        accuracy = (cm['tp'] + cm['tn']) / b * 100 if b > 0 else 0

        global page_counter

        # final_realized_pnl = obv.cash_asset[-1]
        # gain = (final_realized_pnl - opts.init_trade_cash) / opts.init_trade_cash * 100
        final_realized_pnl = obv.accum_realized_pnl[-1] + opts.init_trade_cash
        gain = obv.accum_realized_pnl[-1] / opts.init_trade_cash * 100
        if page_counter == 0:
            logger.info(' {0:>12} | {1:>8}  | {2:>10} | {3:>10} | {4:>10} | {5:>10} | {6:>9} | {7:>10} | {8:>7} | '
                        '{9:>5} | {10:>10} |'
                        ''.format('pnl', 'gain', 'buy delta', 'sell delta',
                                  'long entry', 'short entry', 'long exit', 'short exit', 'acc pct', 'num trade',
                                  'cal count'))
        logger.info(f'${final_realized_pnl:>12,.0f} | {gain:>8,.2f}% | '
                    f'{opts.buy_delta:>10.5f} | {opts.sell_delta:>10.5f} | '
                    f'{opts.long_level:>10.3f} | {opts.short_level:>11.3f} | '
                    f'{opts.long_exit_level:>9.3f} | {opts.short_exit_level:>10.3f} | '
                    f'{accuracy:>6.1f}% | {my_obv.num_trade:>9} | {page_counter:>5}, {opts.interval:>3} |')
        # , opts:{5}, vars(opts)

        # save data for log or late processing
        if 'form_end' in vars(opts):
            form_end = opts.form_end
        else:
            form_end = prices_mu_train.index[-1]

        if csv is not None:
            row = {
                'form_start': opts.form_start,
                'form_end': form_end,
                'gain': round(gain, 1),
                'buy_delta': opts.buy_delta,
                'sell_delta': opts.sell_delta,
                'long_level': round(opts.long_level, 8),
                'long_exit_level': round(opts.long_exit_level, 8),
                'short_level': round(opts.short_level, 8),
                'short_exit_level': round(opts.short_exit_level, 8),
                'buy_sd_delta': f'{opts.buy_sd_delta:.1e}',
                'sell_sd_delta': f'{opts.sell_sd_delta:.1e}',
                'num_trade': my_obv.num_trade,
                'accuracy': round(accuracy, 1)
            }
            df = pd.DataFrame([row])
            if os.path.isfile(csv):
                df.to_csv(csv, mode='a', index=False, header=False, na_rep='None')
            else:
                df.to_csv(csv, header=row.keys(), index=False, na_rep='None')

        if parquet_file is not None:
            # TODO
            helper.save_pdfo_result(data, self.parquet_file, append=True, logger=self.logger)

        page_counter += 1

        return {'gain': gain, 'actor': actor, 'buy_indi': buy_indi, 'sell_indi': sell_indi,
                'buy_simple': buy_simple, 'sell_simple': sell_simple,
                'prices_mu': prices_mu_train, 'prices_actual': prices_actual_train,
                'obv': obv, 'my_obv': my_obv, 'opts': opts}

    except Exception as e:
        logger.error(readable_error(e, __file__))
        sys.exit()


def on_error(e):
    print(traceback.print_exception(type(e), e, e.__traceback__))


def unit_test():
    opts, window_size, ml_params = default_data_init()

    logger = utils.setup_logger('brunhild_train.log', opts.symbol)

    print('===================================================')
    print(opts)

    re = run_sim(opts, logger=logger, verbose=True)

    plot_sim(opts, re, logger)


def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    return sig


def plot_sim(opts, re, logger):
    try:
        plt.figure(1, figsize=(15, 8))

        # RMSE
        price_sell_tracer_diff = re['prices_mu'] - re['sell_indi'].tracer
        sell_buy_tracer_diff = re['sell_indi'].tracer - re['buy_indi'].tracer

        dt_index = re['obv'].dt_index

        ax1 = plt.subplot(711)
        plt.plot(re['prices_actual'], label="price actual", lw=1)

        ax1.set_ylabel('price')
        plt.grid()
        plt.legend(loc="best")

        plt.title('{0} TradingStrategy {1}, Gain:{2:.2f}%'
                  ''.format(opts.strategy, opts.symbol, re['gain']))

        ax2 = plt.subplot(712, sharex=ax1)
        # plt.step(signal_pd.index, signal_pd, label="Signal", lw=1)
        plt.step(dt_index, re['obv'].signal, label="Signal ref", lw=1)

        plt.axhline(0, color='black', ls='-.', lw=1)
        plt.axhline(1, color='green', ls='-.', lw=1)
        plt.axhline(-1, color='green', ls='-.', lw=1)
        ax2.set_ylabel('signal')
        plt.grid()
        plt.legend(loc="upper right")

        ax3 = plt.subplot(713, sharex=ax1)
        plt.step(dt_index, re['obv'].position, label="Position", lw=1)
        plt.axhline(0, color='black', ls='-.', lw=1)
        plt.axhline(1, color='green', ls='-.', lw=1)
        plt.axhline(-1, color='green', ls='-.', lw=1)
        ax3.set_ylabel('position')
        plt.grid()
        plt.legend(loc="upper right")

        ax4 = plt.subplot(714, sharex=ax1)
        plt.step(dt_index, re['obv'].share, label="share", lw=1)
        plt.axhline(0, color='black', ls='-.', lw=1)
        ax4.set_ylabel('share')
        plt.grid()
        plt.legend(loc="best")

        ax5 = plt.subplot(715, sharex=ax1)
        plt.plot(dt_index, re['obv'].cash_asset, label="cash + asset", lw=1)
        ax5.set_ylabel('cash+asset')
        plt.grid()
        plt.legend(loc="upper left")

        ax6 = plt.subplot(716, sharex=ax1)
        plt.plot(re['obv'].dt_index, re['buy_indi'].alfa, label="buy_alfa", lw=1)
        # plt.plot(obv.dt_index, buy_indi['price_pct'], label="price_pct", lw=1)
        is_active = np.where(re['buy_indi'].sd < 0.95, None, re['buy_indi'].sd)
        plt.step(re['obv'].dt_index, re['buy_indi'].sd, label="buy_sd", lw=1)
        plt.step(re['obv'].dt_index, is_active, label="buy_sd active", lw=1)

        plt.axhline(0, color='black', ls='-.', lw=1)
        plt.axhline(opts.long_level, color='green', ls='-.', lw=1)
        plt.axhline(opts.short_level, color='green', ls='-.', lw=1)

        ax6.set_ylabel('buy_tracer')
        plt.grid()
        plt.legend(loc="upper right")

        ax7 = plt.subplot(717, sharex=ax1)
        plt.plot(dt_index, re['sell_indi'].alfa, label="sell_alfa", lw=1)
        plt.axhline(0, color='black', ls='-.', lw=1)
        plt.axhline(opts.long_exit_level, color='green', ls='-.', lw=1)
        plt.axhline(opts.short_exit_level, color='green', ls='-.', lw=1)
        ax7.set_ylabel('sell_tracer')
        plt.grid()
        plt.legend(loc="upper right")

        # ===== Figure 2 ========
        plt.figure(2, figsize=(15, 8))

        ax21 = plt.subplot(611)
        plt.plot(re['prices_actual'], label="price actual")
        plt.plot(dt_index, re['buy_simple'].tracer, label="BuySimple, delta:{0}".format(re['buy_simple'].delta),
                 ls='-.', lw=1)
        plt.plot(dt_index, re['sell_simple'].tracer, label="SellSimple, delta:{0}".format(re['sell_simple'].delta),
                 ls='--', lw=1)

        ax21.set_ylabel('price')
        plt.grid()
        plt.legend(loc="best")
        plt.title('{0} TradingStrategy {1} Train'.format(opts.strategy, opts.symbol))

        dt_index = re['obv'].dt_index
        ax22 = plt.subplot(612, sharex=ax21)

        plt.plot(dt_index, re['sell_indi'].alfa, label="sell_alfa (price_pct - its' tracer)", lw=1)
        plt.plot(dt_index, re['buy_indi'].alfa, ls='-.', label="buy_alfa (price_pct - its' tracer)", lw=1)

        plt.axhline(0, color='black', ls='-.', lw=1)
        plt.axhline(opts.long_level, color='green', ls='-.', lw=1)
        plt.axhline(opts.short_level, color='green', ls='-.', lw=1)
        ax22.set_ylabel('buy sell alfa')
        plt.grid()
        plt.legend(loc="upper right")

        ax23 = plt.subplot(613, sharex=ax21)
        plt.plot(dt_index, re['buy_indi'].tracer - re['sell_indi'].tracer, label="buy tracer - sell tracer", lw=1)

        plt.axhline(0, color='black', ls='-.', lw=1)
        plt.axhline(opts.long_level, color='green', ls='-.', lw=1)
        plt.axhline(opts.short_level, color='green', ls='-.', lw=1)
        plt.grid()
        plt.legend(loc="upper right")

        ax24 = plt.subplot(614, sharex=ax21)
        ax24.set_ylabel('buy_delta')
        plt.plot(dt_index, re['buy_indi'].alfa - re['sell_indi'].alfa, label="buy alfa - sell alfa", lw=1)

        plt.axhline(0, color='black', ls='-.', lw=1)
        plt.grid()
        plt.legend(loc="upper right")

        ax25 = plt.subplot(615, sharex=ax21)
        ax25.set_ylabel('buy_delta')
        plt.plot(dt_index, re['obv'].buy_delta, label="buy_delta", lw=1)

        plt.axhline(0, color='black', ls='-.', lw=1)
        plt.grid()
        plt.legend(loc="upper right")

        ax26 = plt.subplot(616, sharex=ax21)
        ax26.set_ylabel('sell_delta')
        plt.plot(dt_index, re['obv'].sell_delta, label="sell_delta", lw=1)

        plt.axhline(0, color='black', ls='-.', lw=1)

        plt.grid()
        plt.legend(loc="upper right")

        plt.show()

    except Exception as e:
        err = readable_error(e, __file__)
        logger.error(err)


# DEBUG
if __name__ == '__main__':
    unit_test()
