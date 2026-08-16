import sys
import time
import datetime as dt

import pandas as pd
from typing import Dict, Any
from joblib import Parallel, delayed
from ahf.preprocessor.kf.TracerSimpleMem import TracerSimpleMem
from ahf.utils.utils import extract_num, readable_error, convert_to_min, datefmt


class Market_Alfa_Processor:
    def __init__(self,
                 market_pd: pd.DataFrame,
                 kf_list: list,
                 market_args: Dict[str, Any],
                 logger,
                 catchup: bool = True,
                 use_thread: bool = True,
                 verbose: bool = True):

        def extract(d):
            # price_alfa_001 -> 001 -> 0.01
            _delta = float(f"0.{d.split('_')[2][1:]}")
            return _delta

        self.market_pd = market_pd
        self.use_thread = use_thread
        self.market_args = market_args

        # kalman filter configuration
        self.obs_cov = 0.5
        self.sd_delta = 1e-6
        self.sd_obs_cov = 0.5

        self.kf_list = kf_list
        self.delta_list = delta_list = tuple(extract(v) for v in self.kf_list)
        self.delta_list_set = delta_list_set = set(delta_list)

        self.last_tracers = [0, 0]
        self.current_tracers = [0, 0]

        self.items = []
        self.idx = 0

        for i, delta in enumerate(delta_list_set):
            """
            market_alfa_05', 'market_alfa_05_slope',
            'market_alfa_01', 'market_alfa_01_slope'
            """
            # trade_args, strategy, job_id, delta, obs_cov, logger, initial_state_mean, initial_state_cov
            self.items.append(TracerSimpleMem(self.market_args, 'Alfa', f'Market_Alfa_{delta}', delta,
                                              self.obs_cov, logger, verbose=verbose))

            # self.items.append(TracerSimple('FeatureEngineer', 'market_kf_ema', delta, self.obs_cov, logger))

        self.logger = logger

        if catchup and market_pd is not None:
            self.catchup(market_pd)

    def step(self, se):
        self.current_tracers = [0, 0]
        re = []
        for i, delta in enumerate(self.delta_list):
            if 'slope' in self.kf_list[i]:
                v = self.items[i].update(self.idx, self.current_tracers[int(i // 2)], self.last_tracers[int(i // 2)])
            else:
                v = self.items[i].update(se)
                self.current_tracers[int(i // 2)] = v

            re.append(v)

        self.last_tracers = self.current_tracers

        return re

    def value(self):
        results = []
        for i, delta in enumerate(self.delta_list):
            re = self.items[i].value()
            results.append(re.tracer)

        return results

    def catchup(self, market_pd: pd.DataFrame):
        try:
            start_time = time.time()

            status_re = []
            history_re = []
            for i, v in enumerate(self.delta_list_set):
                status, history = self.items[i].catchup(market_pd)
                status_re.append(status)
                history_re.append(history)

            history_arr = []
            for status_row, history_row in zip(status_re, history_re):
                if not status_row:
                    raise Exception(history_row['err'])

                # already embed it in the catchup, no longer needed
                # history_row.set_index('dt_idx', inplace=True)
                # del history_row['idx']
                # history_row = history_row[~history_row.index.duplicated(keep='first')]

                history_arr.append(history_row)

            duration_str = str(dt.timedelta(seconds=(time.time() - start_time)))
            self.logger.info(f'[Market_Alfa_Processor] catchup took {duration_str[:-5]} to complete')

            return history_arr
        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)
            sys.exit()

    def load_hist(self, use_thread=None):
        try:
            def job_func(item):
                return item.load_tracer()

            start_time = time.time()
            use_thread = self.use_thread if use_thread is None is None else use_thread

            # catchup alfa
            n_jobs = len(self.delta_list_set) if self.use_thread else 1
            _re = Parallel(n_jobs=n_jobs, )(
                delayed(job_func)(self.items[i]) for i, v in enumerate(self.delta_list_set))

            # show error
            a, re = zip(*_re)
            for re_status, re_data in zip(a, re):
                if not re_status:
                    raise Exception(re_data['err'])

            duration_str = str(dt.timedelta(seconds=(time.time() - start_time)))
            self.logger.info(f'[Market_Alfa_Processor] load_hist catchup took {duration_str[:-5]} to complete')

            return re

        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)
            sys.exit()