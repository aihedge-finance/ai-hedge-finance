import os
import sys
import pandas as pd
import numpy as np
import datetime as dt
from decimal import Decimal
from collections import deque
from ahf.utils.utils import readable_error, datefmt, create_dir_if_non_exist
from ahf.preprocessor.tail.tail_C import EMA as EMA_tail

class EMA:
    cols_name = ['idx', 'dt_idx', 'price', 'ema']
    cols_types = {'idx': int,
                  'dt_idx': str,
                  'price': float,
                  'ema': float
                  }

    def __init__(self, time_period, logger, verbose=True):
        """
        REFERENCE: https://indzara.com/2021/04/rsi-technical-indicator-excel-template/

        :param time_period:
        :param logger:
        :param verbose:
        """
        self.max_deque = sys.maxsize if os.getenv("NODE_ENV") is None else 500

        self.idx_now = 0

        self.EMA_tail = EMA_tail(timeperiod=time_period)

        self.price_pd = None
        self.time_period = time_period

        self.idx = None
        self.dt_idx = None
        self.price = None
        self.ema = None

        self.logger = logger
        self.verbose = verbose

        self.reset()

    def reset(self):
        self.idx_now = 0
        self.price_pd = None

        self.idx = deque(maxlen=self.max_deque)
        self.dt_idx = deque(maxlen=self.max_deque)  # dtype='datetime64[us]'
        self.price = deque(maxlen=self.max_deque)
        self.ema = deque(maxlen=self.max_deque)

    def populate_init(self, dt_idx, price, ema):
        """
        給 populate init data ONLY from last run
        """
        self.idx.append(0)
        self.dt_idx.append(dt_idx)
        self.price.append(price)
        self.ema.append(ema)
        self.idx_now = 1
        if len(self.idx) != 1:
            raise Exception("EMA populate_init is called after initialized")

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
            tmp = self.EMA_tail.add_one(price)

            # register dt_idx if we decide to record it
            self.idx.append(self.idx_now)
            self.dt_idx.append(dt_idx)
            self.ema.append(tmp)

            self.idx_now += 1

            if len(self.ema) > 1:
                return {"ema": [self.ema[-2], self.ema[-1]]}
            else:
                return {"ema": [self.ema[-1], self.ema[-1]]}

        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)


    def _save_data(self, file_dir: str,
                   idx: np.ndarray,
                   dt_idx: np.ndarray,
                   price_np: np.ndarray,
                   ema: np.ndarray,
                   need_header=False,
                   write_mode="w"):
        """
        這是單機版訓練用在 catchup 的的儲存，如果是 socket 版，
        那就是算好儲存並且同時打資料給我們，策略就會是 stateless

        或許這方式以後可以不用？？

        result from parent update()=tracer_new

        :param file_dir: usage => f"{self.trade_args['data_path']}/TA_theta/{self.name}_{self.theta_id}.csv"
        """
        try:
            dt_idx_str = [x.strftime(datefmt) for x in dt_idx]
            data = np.column_stack(
                [
                    idx,
                    dt_idx_str,
                    price_np,
                    ema
                ])
            r = pd.DataFrame(data, index=idx,
                             columns=self.cols_name)

            create_dir_if_non_exist(file_dir)
            r.to_csv(file_dir, mode=write_mode, header=need_header, index=False, na_rep='None', index_label='idx')
            if self.verbose:
                self.logger.info('[EMA] EMA saved to \n{0}'.format(file_dir))
        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)
            sys.exit()

