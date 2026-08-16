import time
import numpy as np
import datetime as dt

import multiprocessing as mp
import pandas as pd
from typing import Dict, Any
from joblib import Parallel, delayed
from ahf.utils.utils import readable_error, pretty_dict

from ahf.preprocessor.kf.TracerMem_v2 import TracerMem_v2

from ahf.preprocessor.helpers import preprocessor_load_data, price_ta_job

# from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from ahf.preprocessor.finrl.FeatureEngineer import FeatureEngineer
from ahf.preprocessor.preprocessors import FeatureEngineer_KF, FeatureEngineer_Market_Capital, \
    FeatureEngineer_Market_KF  # , FeatureEngineer_TracerSimple

# threading
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from deprecated import deprecated

# Preprocessing
from ahf.preprocessor.kf.Market_Alfa_Processor import Market_Alfa_Processor
from ahf.preprocessor.ta.TA_Processor import TA_Processor


@deprecated(reason="Demo purpose", action="ignore")
class Price_Alfa_Processor:
    def __init__(self,
                 price_pd: pd.DataFrame,
                 kf_list: list,
                 trade_args: Dict[str, Any],
                 logger,
                 catchup=True,
                 verbose=True):

        def extract(d):
            # price_alfa_001 -> 001 -> 0.01
            _delta = float(f"0.{d.split('_', 2)[2][1:]}")
            return _delta

        self.trade_args = trade_args

        # kalman filter configuration
        self.obs_cov = 0.5
        self.sd_delta = 1e-6
        self.sd_obs_cov = 0.5

        self.kf_list = kf_list
        self.delta_list = delta_list = tuple(extract(v) for v in self.kf_list)

        self.items = []
        for delta in delta_list:
            # trade_args, strategy, job, delta, obs_cov, sd_delta, sd_obs_cov, logger
            self.items.append(TracerMem_v2(self.trade_args,
                                           'Alfa',
                                           f'Price_Alfa_{delta}',
                                           delta,
                                           self.obs_cov,
                                           self.sd_delta,
                                           self.sd_obs_cov, logger, verbose=verbose))

        self.logger = logger

        if catchup:
            self.catchup(price_pd)

    def step(self, se):
        re = []
        for i, delta in enumerate(self.delta_list):
            self.items[i].update(se)
            v = self.items[i].value()

            re.append(v.alfa[-1])

        return re

    def catchup(self, price_pd, multi_process=False):
        try:
            start_time = time.time()

            if multi_process:
                def job_func(price_df, item):
                    return item.catchup(price_df)

                n_jobs = max(1, mp.cpu_count() - 2)  # len(self.delta_list)

                jobs_re = Parallel(n_jobs=n_jobs, )(
                    delayed(job_func)(price_pd, self.items[i]) for i, v in enumerate(self.delta_list))

                # show error
                status_re, history_re = zip(*jobs_re)
            else:
                status_re = []
                history_re = []
                for i, v in enumerate(self.delta_list):
                    status, history = self.items[i].catchup(price_pd)
                    status_re.append(status)
                    history_re.append(history)

            # status_re = []
            # history_re = []
            # for i, v in enumerate(self.delta_list):
            #     status, history = self.items[i].catchup(price_pd)
            #     status_re.append(status)
            #     history_re.append(history)

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
            self.logger.info(f'[Price_Alfa_Processor] catchup took {duration_str[:-5]} to complete')

            return history_arr
        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)

    def value(self):
        results = []
        for i, delta in enumerate(self.delta_list):
            re = self.items[i].value()
            results.append(re.alfa)

        return results

    def load_hist(self):
        def job_func(item):
            return item.load_tracer()

        start_time = time.time()
        # use_thread = self.use_thread if use_thread is None is None else use_thread

        n_jobs = len(self.delta_list)  #  if use_thread else 1

        _re = Parallel(n_jobs=n_jobs, )(
            delayed(job_func)(self.items[i]) for i, v in enumerate(self.delta_list))

        # show error
        a, re = zip(*_re)
        for re_status, re_data in zip(a, re):
            if not re_status:
                raise Exception(re_data['err'])

        duration_str = str(dt.timedelta(seconds=(time.time() - start_time)))
        self.logger.info(f'[Price_Alfa_Processor] loading history took {duration_str[:-5]} to complete')

        return re


def price_kf_job(price_pd, kf_list, trade_args, logger, multi_process=False):
    """
    給外面的 caller 一個使用 multiprocess 的方式
    使用方式：
        with ProcessPoolExecutor(max_workers=2) as pool:
        price_kf_re = list(pool.map(price_kf_job, self.price_pd, kf_list, repeat(self.logger)))
    """
    try:
        price_kf_arr = []

        price_kf_processor = Price_Alfa_Processor(price_pd,
                                                  kf_list,
                                                  trade_args,
                                                  logger,
                                                  catchup=False)

        history_arr = price_kf_processor.catchup(price_pd, multi_process=multi_process)

        # pick first one as role model
        rows_price = len(history_arr[0].index)

        for i, row in enumerate(history_arr):
            assert rows_price == len(row.index), 'length of history price_kf_job record must match'
            tmp = row['alfa']
            price_kf_arr.append(tmp)

        price_kf_np = np.array(price_kf_arr)
        price_kf_arr_T = np.transpose(price_kf_np)
        price_kf_pd = pd.DataFrame(price_kf_arr_T)

        price_kf_pd.columns = kf_list
        price_kf_pd = price_kf_pd.set_index(history_arr[0].index)

        # no need, we use history_arr's index
        # _price_kf_pd = price_kf_pd[~price_kf_pd.index.duplicated(keep='first')]

        return price_kf_pd
    except Exception as e:
        err_str = readable_error(e, __file__)
        logger.error(err_str)


def slope_kf_job(price_market_pd, market_kf_list, trade_args, trade_interval, logger):
    """
    給外面的 caller 一個使用 multiprocess 的方式
    使用方式：
        with ProcessPoolExecutor(max_workers=2) as pool:
        price_kf_re = list(pool.map(price_kf_job, self.price_pd, kf_list, repeat(self.logger)))
    """
    try:
        market_kf_arr = []
        market_kf_processor = Price_Alfa_Processor(price_market_pd,
                                                   market_kf_list,
                                                   trade_args,
                                                   logger,
                                                   # use_thread=use_thread,
                                                   catchup=False)
        history_arr = market_kf_processor.catchup(price_market_pd)

        # dt_idx = history_arr[0].index WRONG!
        rows = len(history_arr[0].index)

        for row in history_arr:
            assert rows == len(row.index), 'length of history slope_kf_job record must match'

            market_kf_arr.append(row['tracer'])
            market_kf_arr.append(row['slope'])

        market_kf_arr_T = np.transpose(market_kf_arr)
        market_kf_pd = pd.DataFrame(market_kf_arr_T)

        market_kf_pd.columns = market_kf_list
        # you have to use history_arr, cannot use price_market_pd, because it includes history
        market_kf_pd.set_index(history_arr[0].index, inplace=True)

        # trade_interval = trade_args['trade_interval']
        market_kf_pd = market_kf_pd.resample(rule=trade_interval).first().ffill()
        # _market_kf_pd = market_kf_pd[~market_kf_pd.index.duplicated(keep='first')]

        return market_kf_pd

    except Exception as e:
        err_str = readable_error(e, __file__)
        logger.error(err_str)


def gen_data_v2(form_start, form_end, trade_args, tech_list, append,
                save_result, read_write, logger, dest_dir=None):
    try:
        start_time = time.time()

        logger.info(f'[gen_data_v2] trade_args:{pretty_dict(trade_args)}')

        # load price
        prices_dict = preprocessor_load_data(trade_args, form_start, form_end, logger)

        # == Perform market Capitalization Indicator ==
        for key in prices_dict:
            price_pd = prices_dict[key]

            # =================================

            # Perform Custom Alpha Indicators
            kf_list = []
            for item in tech_list:
                if 'price_alfa_' in item:
                    kf_list.append(item)

            # KF part
            # 移到 price_kf_job 處理
            # price_kf_processor = Price_Alfa_Processor(price_pd.copy(), kf_list, trade_args, logger,
            #                                          use_thread=True, catchup=False)

            price_kf_re = price_kf_job(price_pd, kf_list, trade_args, logger)

            # ==================================
            # Perform Feature Engineering:
            logger.info(f'[gen_data_v2] preparing technical indicators')

            # we separate RSI for our own
            tech_list_default = []
            tech_list_custom = []
            use_turbulence = False
            for item in tech_list:
                if 'price_alfa_' not in item and item != 'turbulence':
                    if 'rsi_' not in item:
                        tech_list_default.append(item)
                    if 'rsi_' in item:  # custom indicator by us
                        tech_list_custom.append(item)

                if item == 'turbulence':
                    use_turbulence = True

            if len(tech_list_default) == 0:
                raise Exception("You need to have at least one tech_list_default such as change")

            tech_df = FeatureEngineer(use_technical_indicator=True,
                                      tech_indicator_list=tech_list_default,
                                      use_turbulence=use_turbulence,
                                      use_vix=False,
                                      user_defined_feature=False).preprocess_data(
                price_pd.copy().reset_index(drop=False))

            # add covariance matrix as states
            tech_df = tech_df.sort_values(['date', 'tic'], ignore_index=True)
            # tech_df.index = tech_df.date.factorize()[0]
            tech_df.set_index('date', inplace=True)

            # ==================================
            # if len(tech_list_custom) == 0:
            #    raise Exception('RSI not found, I personally suggest adding RSI')
            tech_custom_re = None
            if len(tech_list_custom) > 0:
                tech_custom_processor = TA_Processor(price_pd, tech_list_custom, trade_args, logger,
                                                     catchup=False,
                                                     use_thread=True, verbose=True)  # .catchup(price_pd)

                tech_custom_re = price_ta_job(tech_custom_processor, price_pd, tech_list_custom, logger)
                tech_custom_re.index.names = ['date']

            # _concat_df = pd.concat([tech_df, market_pf], axis=1)
            # concat_df = _concat_df.dropna()
            # concat_df.set_index('date', inplace=True)

            # ===================================

            # merge everything
            if price_kf_re is None or len(price_kf_re.index) == 0:
                raise Exception('KF_list result cannot be returned empty')

            if tech_df is not None and tech_df.index[-1] != price_kf_re.index[-1]:
                raise Exception(
                    f'tech_df len {len(tech_df.index)} must match price_kf_re data len {len(price_kf_re.index)}')

            if tech_custom_re is not None and tech_custom_re.index[-1] != price_kf_re.index[-1]:
                raise Exception(
                    f'tech_custom_pd len {len(tech_custom_re.index)} must '
                    f'match price_kf_re data len {len(price_kf_re.index)}')

            # IMPORTANT, WRONG! market data lags price, it has to be concated and
            # if len(market_kf_re.index) != len(price_kf_re.index):
            #    raise Exception(f'price_kf_re len {len(price_kf_re.index)} must match market data len '
            #                    f'{len(market_kf_re.index)}')

            cols_pd = [re for re in [tech_df, price_kf_re, tech_custom_re] if re is not None]

            _concat_df = pd.concat(cols_pd, axis=1)
            concat_df = _concat_df.resample(trade_args['trade_interval']).first().ffill()  # .reset_index()
            concat_df = concat_df.dropna()
            concat_df.index.names = ['date']
            if len(_concat_df.index) < len(concat_df.index) * 0.9:
                logger.warning(f'Before dropna() there is {len(concat_df.index)} row, after {len(_concat_df.index)}')

            end_time = time.time() - start_time
            duration_str = str(dt.timedelta(seconds=end_time))

            print(f'[{key}] Feature Engineering took {duration_str[:-5]} to complete')

            # del concat_df['date']  # already in index

            if save_result:
                read_write.save_tech(key, trade_args, concat_df,
                                     append=append, dest_dir=dest_dir)

            return concat_df
    except Exception as e:
        err_str = readable_error(e, __file__)
        logger.error(err_str)
        return False, {'err': err_str}


def gen_data_v1(form_start, form_end, trade_args, market_args, tech_list, market_list, market_kf_list, append,
                save_result, read_write, logger):
    start_time = time.time()

    logger.info(f'trade_args:{pretty_dict(trade_args)}')
    logger.info(f'market_args:{pretty_dict(market_args)}')

    prices_dict = preprocessor_load_data(trade_args, form_start, form_end, logger)

    for key in prices_dict:

        # Perform Feature Engineering:
        print(f'[{key}] preparing technical indicators')

        tech_list_default = []
        use_turbulence = False
        for item in tech_list:
            if 'price_alfa_' not in item and item != 'turbulence':
                tech_list_default.append(item)
            if item == 'turbulence':
                use_turbulence = True

        df = FeatureEngineer(use_technical_indicator=True,
                             tech_indicator_list=tech_list_default,
                             use_turbulence=use_turbulence,
                             use_vix=False,
                             user_defined_feature=False).preprocess_data(prices_dict[key])

        # add covariance matrix as states
        df = df.sort_values(['date', 'tic'], ignore_index=True)
        df.index = df.date.factorize()[0]

        # Perform Custom Alpha Indicators
        kf_list = []
        for item in tech_list:
            if 'price_alfa_' in item:
                kf_list.append(item)
        df_kf = FeatureEngineer_KF(kf_list, logger).preprocess_data(df['change'])

        # merge
        if df_kf is None or len(df_kf.index) == 0:
            raise Exception('KF_list result cannot be returned empty')

        df_concat = pd.concat([df, df_kf], axis=1)

        df_concat = df_concat.set_index(prices_dict[key]['date'])

        # == Perform market Capitalization Indicator ==
        market_symbol = market_args["symbols"][0]
        print(f'[{key}] preparing market indicators {market_symbol}')

        if len(df.index) != len(df_kf.index):
            raise Exception(f'price len {len(df.index)} must match market data len {len(df_kf.index)}')

        price_market = preprocessor_load_data(market_args, form_start, form_end, logger)

        df_cap = FeatureEngineer_Market_Capital(market_list, logger).preprocess_data(price_market[market_symbol])
        df_cap = df_cap.set_index(price_market[market_symbol]['date'])

        df_cap = df_cap.resample(trade_args['trade_interval']).ffill()  # .reset_index()

        df_concat = pd.concat([df_concat, df_cap], axis=1)
        df_concat = df_concat.dropna()

        # Perform tracerSimple for price
        """
        df_tracer_simple = FeatureEngineer_TracerSimple(market_kf_list, logger).preprocess_data(price_market[key]['open'])
        df_tracer_simple = df_tracer_simple.set_index(price_market[key]['date'])

        df_tracer_simple = df_tracer_simple.resample(trade_args['interval']).bfill()  # .reset_index()
        df_concat = pd.concat([df_concat, df_tracer_simple], axis=1)
        df_concat = df_concat.dropna()
        """

        # Perform market Capitalization KF Indicator
        df_market_kf = FeatureEngineer_Market_KF(market_kf_list, logger).preprocess_data(
            price_market[market_symbol]['open'])
        df_market_kf = df_market_kf.set_index(price_market[market_symbol]['date'])

        df_market_kf = df_market_kf.resample(trade_args['trade_interval']).ffill()  # .reset_index()
        df_concat = pd.concat([df_concat, df_market_kf], axis=1)
        df_concat = df_concat.dropna()

        end_time = time.time() - start_time
        duration_str = str(dt.timedelta(seconds=end_time))

        print(f'[{key}] Feature Engineering took {duration_str[:-5]} to complete')

        del df_concat['date']  # already in index

        if save_result:
            read_write.save_tech(key, trade_args, df_concat,  append,
                                 dest_dir=None, logger=logger)

        return df_concat
