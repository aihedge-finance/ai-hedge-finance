import sys
import time
import datetime as dt

import pandas as pd
from typing import Dict, Any
from joblib import Parallel, delayed
from ahf.utils.utils import readable_error
from .KDJMem import KDJMem
from .RSIMem import RSIMem
from .ATRMem import ATRMem
from .STOCHMem import STOCHMem


class TA_Processor:
    def __init__(self,
                 price_pd: pd.DataFrame,
                 tech_list: list,
                 trade_args: Dict[str, Any],
                 logger,
                 catchup=True,
                 use_thread=True,
                 verbose=True):

        self.use_thread = use_thread
        self.logger = logger

        def extract(d):
            # rsi_30
            params = d.split('_')
            return params

        self.trade_args = trade_args
        self.tech_list = tech_list
        self.tech_tuple = tuple(extract(v) for v in self.tech_list)

        self.items = []
        for tech in self.tech_tuple:
            # trade_args, strategy, job, delta, obs_cov, sd_delta, sd_obs_cov, logger
            if tech[0] == 'rsi':
                period = int(tech[1])
                self.items.append(RSIMem(self.trade_args,
                                         'TA',
                                         f'{tech[0]}_{tech[1]}', period,
                                         self.logger,
                                         is_unittest=False))
            elif tech[0] == 'stoch':
                self.items.append(STOCHMem())
            elif tech[0] == 'kdj':
                self.items.append(KDJMem())
            elif tech[0] == 'atr':
                self.items.append(ATRMem())
            else:
                raise Exception(f'technical indicator {tech[0]} is not functional')

        self.logger = logger

        if catchup and price_pd is not None:
            self.catchup(price_pd)

    def step(self, se):
        re = []
        for i, delta in enumerate(self.tech_tuple):
            self.items[i].update(se)
            v = self.items[i].value()

            re.append(v.alfa[-1])

        return re

    def catchup(self, data):
        """
        main method to do the feature engineering
        @:param config: source dataframe
        @:return: a DataMatrices object
        """
        try:
            start_time = time.perf_counter()

            price_pd = data.copy()

            if thread_mode == 'thread':
                def job_func(price_df, item):
                    return item.catchup(price_df)

                n_jobs = len(self.tech_tuple) if self.use_thread else 1

                jobs_re = Parallel(n_jobs=n_jobs, )(
                    delayed(job_func)(price_pd, self.items[i]) for i, v in enumerate(self.tech_tuple))

                # show error
                status_re, history_re = zip(*jobs_re)
            elif thread_mode == 'single':
                status_re = []
                history_re = []
                for i, v in enumerate(self.tech_tuple):
                    status, history = self.items[i].catchup(price_pd)
                    status_re.append(status)
                    history_re.append(history)

            elif thread_mode == 'process':
                def job_func(price_df, item):
                    return item.catchup(price_df)

                n_jobs = len(self.tech_tuple) if self.use_thread else 1

                jobs_re = Parallel(n_jobs=n_jobs, )(
                    delayed(job_func)(price_pd, self.items[i]) for i, v in enumerate(self.tech_tuple))

                # show error
                status_re, history_re = zip(*jobs_re)
            else:
                raise Exception(f'Unknown run_mode {thread_mode}')

            history_arr = []
            for status_row, history_row in zip(status_re, history_re):
                if not status_row:
                    raise Exception(history_row['err'])

                # already embed it in the catchup, no longer needed
                # history_row.set_index('dt_idx', inplace=True)
                # del history_row['idx']
                # history_row = history_row[~history_row.index.duplicated(keep='first')]

                history_arr.append(history_row)

            duration_str = str(dt.timedelta(seconds=(time.perf_counter() - start_time)))
            self.logger.info(f'[TA_Processor] catchup took {duration_str[:-5]} to complete')

            return history_arr
        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)

    def value(self):
        results = []
        for i, delta in enumerate(self.tech_tuple):
            re = self.items[i].value()
            results.append(re.alfa)

        return results

    def load_hist(self, use_thread=None):
        def job_func(item):
            return item.load_tracer()

        start_time = time.perf_counter()
        use_thread = self.use_thread if use_thread is None is None else use_thread

        n_jobs = len(self.tech_tuple) if use_thread else 1

        _re = Parallel(n_jobs=n_jobs, )(
            delayed(job_func)(self.items[i]) for i, v in enumerate(self.tech_tuple))

        # show error
        a, re = zip(*_re)
        for re_status, re_data in zip(a, re):
            if not re_status:
                raise Exception(re_data['err'])

        duration_str = str(dt.timedelta(seconds=(time.perf_counter() - start_time)))
        self.logger.info(f'[TA_Processor] Price_Alfa loading history took {duration_str[:-5]} to complete')

        return re
