import os
import sys
import pandas as pd
import numpy as np
import datetime as dt
from decimal import Decimal
from collections import deque
from ahf.utils.utils import readable_error, datefmt, create_dir_if_non_exist

import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
from ahf.preprocessor.tail.tail_C import MACD as MACD_tail


class MACD:
    cols_name = ['idx', 'dt_idx', 'price', 'macd_signal', 'macd']
    cols_types = {'idx': int,
                  'dt_idx': str,
                  'price': float,
                  'macd_signal': float,
                  'macd': float
                  }

    def __init__(self, fast_period=12, slow_period=26, signal_period=9, logger=None, verbose=True):
        """
        REFERENCE: https://indzara.com/2021/04/rsi-technical-indicator-excel-template/
        """
        self.max_deque = sys.maxsize if os.getenv("NODE_ENV") is None else 500

        self.idx_now = 0

        # fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period
        self.MACD_tail = MACD_tail()

        self.price_pd = None
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

        self.idx = None
        self.dt_idx = None
        self.price = None
        self.macd = None
        self.macd_signal = None

        self.lookback = 2  # 預設，-1是最新，-2是前一個

        self.logger = logger
        self.verbose = verbose

        self.reset()

    def reset(self):
        self.idx_now = 0
        self.price_pd = None

        self.idx = deque(maxlen=self.max_deque)
        self.dt_idx = deque(maxlen=self.max_deque)  # dtype='datetime64[us]'
        self.price = deque(maxlen=self.max_deque)
        self.macd_signal = deque(maxlen=self.max_deque)
        self.macd = deque(maxlen=self.max_deque)

    def set_lookback(self, lookback):
        if lookback is None:
            self.logger.warning("MACD set_lookback param is None, are you sure?")
        if lookback is not None:
            self.lookback = lookback

    def populate_init(self, dt_idx, price, macd_signal, macd):
        """
        給 populate init data ONLY from last run
        """
        self.idx.append(0)
        self.dt_idx.append(dt_idx)
        self.price.append(price)
        self.macd_signal.append(macd_signal)
        self.macd.append(macd)
        self.idx_now = 1
        if len(self.idx) != 1:
            raise Exception("MACD populate_init is called after initialized")

    def add_one(self, price, dt_idx):
        try:
            if not isinstance(price, Decimal):
                raise Exception('price must be of type Decimal')

            if dt_idx is None:
                raise Exception(f'dt_idx is required for calculation since you chose this in the first place')

            if not isinstance(dt_idx, dt.datetime):
                raise Exception(f'dt_idx must be dt.datetime but got {type(dt_idx)}')

            # record price
            self.price.append(price)
            tmp = self.MACD_tail.add_one(price)

            # register dt_idx if we decide to record it
            self.idx.append(self.idx_now)
            self.dt_idx.append(dt_idx)
            self.macd_signal.append(tmp[1])
            self.macd.append(tmp[0])

            self.idx_now += 1

            l = len(self.macd)

            lookback_pos = -min(l, self.lookback)

            return {
                "macd": [self.macd[lookback_pos], self.macd[-1]],
                "macd_signal": [self.macd_signal[lookback_pos], self.macd_signal[-1]],
                "macd_bar": [self.macd[lookback_pos] - self.macd_signal[lookback_pos],
                             self.macd[-1] - self.macd_signal[-1]]
            }


        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)

    def _save_data(self, file_dir: str,
                   idx: np.ndarray,
                   dt_idx: np.ndarray,
                   price_np: np.ndarray,
                   macd_signal: np.ndarray,
                   macd: np.ndarray,
                   need_header=False):
        """
        result from parent update()=tracer_new

        :param file_dir: usage => f"{self.trade_args['data_path']}/TA_theta/{self.name}_{self.theta_id}.csv"
        :param dt_idx:
        :param price_np:
        :param macd_signal:
        :param macd:
        :return:
        """

        try:
            # 日期
            dt_idx_str = [x.strftime(datefmt) for x in dt_idx]
            data = np.column_stack(
                [
                    idx,
                    dt_idx_str,
                    price_np,
                    macd_signal,
                    macd
                ])
            r = pd.DataFrame(data, index=idx,
                             columns=self.cols_name)

            create_dir_if_non_exist(file_dir)
            # file_dir = f"{self.trade_args['data_path']}/KF_theta/{self.name}_{self.theta_id}.csv"
            r.to_csv(file_dir, mode='a', header=need_header, index=False, na_rep='None', index_label='idx')
            if self.verbose:
                self.logger.info('[MACD] MACD saved to \n{0}'.format(file_dir))
        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)
            sys.exit()
