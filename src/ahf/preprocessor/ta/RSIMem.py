import gc
import sys
import math
import numpy as np
import pandas as pd
from os import path
import datetime as dt
from ahf.utils.utils import is_dt_offset_aware, d
from ahf.utils.utils import readable_error, datefmt, datesffmt, pct_change
from .RSI import RSI


class RSIMem(RSI):
    def __init__(self, strategy, trade_args, tech_args, logger,
                 is_unittest=False, verbose=True):
        """
        RSI 的 有記憶的 indicator
        exchange, symbol, trade_interval,
        """

        super().__init__(tech_args.get('rsi_time_period'), logger)

        self.name = 'RSI'
        self.exchange = trade_args.get('exchange')
        self.symbol = trade_args.get('symbol')
        self.trade_interval = trade_args.get('trade_interval')
        self.job_id = trade_args.get('job_id', "RSI_job")

        self.rsi_time_period = tech_args.get('rsi_time_period')
        kf_data_path = trade_args.get('kf_data_path')

        self.data_hist_pd = None

        self.verbose = True if is_unittest else verbose

        self.sig_code = f"rsi:period={self.rsi_time_period}"

        # 識別用
        self.theta_id = f"{self.exchange}_" \
                        f"{self.symbol}_" \
                        f"{self.trade_interval}_" \
                        f"{strategy}_" \
                        f"{self.job_id}_" \
                        f"{self.sig_code}"

        if is_unittest:
            unittest_str = '_UNITTEST'
        else:
            unittest_str = ''

        self.file_dir = f"{kf_data_path}/{self.name}_{self.theta_id}{unittest_str}.csv"

    def catchup(self, priceA):
        # IMPORTANT+ CAREFUL: This is used with side-line (Jobber cron job),
        # don't mix parameters with online (running program)
        try:
            if self.verbose:
                self.logger.info(f'[RSI] catching up for RSI time_period {self.time_period}')

            if not isinstance(priceA, pd.DataFrame):
                raise Exception("RSIMem catchup must provide pd.DataFrame")

            if len(priceA.index) == 0:
                raise Exception("price is empty")

            if 'date' not in priceA and priceA.index.name != 'date':
                raise Exception('date index column does not exist')

            if priceA.index.name != 'date':
                price_pd = priceA.set_index('date', drop=False, inplace=False)
            else:
                price_pd = priceA

            # self.logger.info(f'[RSI] {self.name} data_history does not exist, building one up now .....')
            last_price_dt_index = last_save_dt_index = price_pd.index[0].to_pydatetime()  # .strftime(datefmt)

            # capture only since from last save part for training
            t_delta = last_price_dt_index - last_save_dt_index
            if t_delta.total_seconds() < 0:
                raise Exception(f'[RSI] price did not catch up with history data, '
                                f'price:{last_price_dt_index}, history:{last_save_dt_index}')

            price_pd = price_pd[last_save_dt_index:]

            if self.verbose:
                self.logger.info(f'[RSI] Start training new RSI prices with time_period {self.time_period}')

            price_np = price_pd['close'].to_numpy()

            # Setup variables
            dt_idx = price_pd.index.to_pydatetime()
            total_len = len(price_pd.index)

            for _price, _dt_idx in zip(price_np, dt_idx):
                idx_now = self.idx_now
                self.add_one(d(_price), _dt_idx)
                self._stdout(idx_now, total_len)

            # only save when it is larger than 1, otherwise there are lots of zeros
            # if self.idx_now > 1:
            #     self._save_data(self.file_dir,
            #                     self.idx,
            #                     self.dt_idx,
            #                     price_np,
            #                     self.rsi,
            #                     need_header=True)

            new = {
                'dt_idx': dt_idx,
                'price': price_np,
                'rsi': self.rsi,
            }

            new_pd = pd.DataFrame(new)
            new_pd.set_index('dt_idx', drop=True, inplace=True)

            self.reset()
            return True, new_pd

        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)

            return False, {'err': err}

    def _stdout(self, idx_now, total_len):
        if idx_now % 1000 == 0:
            msg = f'\r{dt.datetime.now().strftime(datesffmt)}, ' \
                  f'INFO {self.exchange}_{self.symbol} [RSI] ' \
                  f'Training at {idx_now / total_len * 100:.2f}%\r'
            sys.stdout.write(msg)
            sys.stdout.flush()
