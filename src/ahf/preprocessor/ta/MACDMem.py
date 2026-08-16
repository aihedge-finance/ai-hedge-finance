import gc
import sys
import math
import numpy as np
import pandas as pd
import datetime as dt
from collections import deque
from ahf.utils.utils import is_dt_offset_aware, d
from ahf.utils.utils import readable_error, datefmt, datesffmt, pct_change
from .MACD import MACD


class MACDMem(MACD):
    def __init__(self, strategy, trade_args, tech_args, logger,
                 is_unittest=False, verbose=True):

        super().__init__(tech_args.get("macd_fast_period"),
                         tech_args.get("macd_slow_period"),
                         tech_args.get("macd_signal_period"), logger)

        self.name = 'MACD'
        self.exchange = trade_args.get('exchange')
        self.symbol = trade_args.get('symbol')
        self.trade_interval = trade_args.get('trade_interval')

        self.job_id = trade_args.get('job_id', "MACD_job")

        self.trade_args = trade_args
        self.data_hist_pd = None

        self.verbose = True if is_unittest else verbose

        self.sig_code = f"macd:fast_period={self.fast_period}||" \
                        f"slow_period={self.slow_period}||"\
                        f"signal_period={self.signal_period}"

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

        self.file_dir = f"{self.trade_args['kf_data_path']}/{self.name}_{self.theta_id}{unittest_str}.csv"

        # Child layer data fields
        self.macd_bar_norm = deque(maxlen=self.max_deque)

    def reset(self):
        super().reset()
        self.macd_bar_norm = deque(maxlen=self.max_deque)

    def populate_init(self, dt_idx, price, macd_signal, macd):
        """
        給 populate init data ONLY from last run
        """
        super().populate_init(dt_idx, price, macd_signal, macd)
        self.macd_bar_norm.append(macd - macd_signal)

    def add_one(self, price, dt_idx):
        re = super().add_one(price, dt_idx)
        macd_bar_norm = [v / float(price) * 1000 for v in re.get('macd_bar')]
        self.macd_bar_norm.append(macd_bar_norm[1])
        re["macd_bar_norm"] = macd_bar_norm

        return re

    def catchup(self, priceA, data_hist_pd=None):
        # IMPORTANT+ CAREFUL: This is used with side-line (Jobber cron job),
        # don't mix parameters with online (running program)
        try:
            if self.verbose:
                self.logger.info(f'[MACD] catching up for MACD time_period {self.theta_id}')

            if not isinstance(priceA, pd.DataFrame):
                raise Exception("MACDMem catchup must provide pd.DataFrame")

            if len(priceA.index) == 0:
                raise Exception("price is empty")

            if 'date' not in priceA and priceA.index.name != 'date':
                raise Exception('date index column does not exist')

            if priceA.index.name != 'date':
                price_pd = priceA.set_index('date', drop=False, inplace=False)
            else:
                price_pd = priceA

            # self.logger.info(f'[MACD] {self.name} data_history does not exist, building one up now .....')
            last_price_dt_index = last_save_dt_index = price_pd.index[0].to_pydatetime()  # .strftime(datefmt)

            # capture only since from last save part for training
            t_delta = last_price_dt_index - last_save_dt_index
            if t_delta.total_seconds() < 0:
                raise Exception(f'[MACD] price did not catch up with history data, '
                                f'price:{last_price_dt_index}, history:{last_save_dt_index}')

            price_pd = price_pd[last_save_dt_index:]

            if self.verbose:
                self.logger.info(f'[MACD] Start training new MACD prices with fast_period {self.fast_period}')

            price_np = price_pd['open'].to_numpy()

            # Setup variables
            dt_idx = price_pd.index.to_pydatetime()
            total_len = len(price_pd.index)

            for _price, _dt_idx in zip(price_np, price_pd.index):
                idx_now = self.idx_now
                self.add_one(d(_price), _dt_idx)
                dt_idx[idx_now] = _dt_idx.to_pydatetime()
                self._stdout(idx_now, total_len)

            # only save when it is larger than 1, otherwise there are lots of zeros
            # if self.idx_now > 1:
            #     self._save_data(self.file_dir,
            #                     self.idx,
            #                     self.dt_idx,
            #                     price_np,
            #                     self.macd_signal,
            #                     self.macd,
            #                     need_header=True)

            new = {
                'dt_idx': dt_idx,
                'price': price_np,
                'macd_signal': self.macd_signal,
                'macd': self.macd,
                'bar': [a - b for a, b in zip(self.macd, self.macd_signal)]
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
                  f'INFO {self.exchange}_{self.symbol} [MACD] ' \
                  f'Training at {idx_now / total_len * 100:.2f}%\r'
            sys.stdout.write(msg)
            sys.stdout.flush()
