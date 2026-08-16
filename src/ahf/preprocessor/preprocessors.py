import math
import talib
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from ahf.preprocessor.kf.Tracer_v2 import Tracer_v2
from ahf.utils.utils import readable_error, safe_div
from ahf.preprocessor.kf.TracerSimple import TracerSimple


class FeatureEngineer_KF:
    """Provides methods for preprocessing the stock price data

        Attributes
        ----------
            obs_cov : float
                observation covariance
            sd_delta : float
                delta for std
            sd_obs_cov : float
                observation covariance for std

        Methods
        -------
        preprocess_data()
            main method to do the feature engineering

        """

    def __init__(self, kf_list, logger):
        self.kf_list = kf_list

        # kalman filter configuration
        self.obs_cov = 0.5
        self.sd_delta = 1e-6
        self.sd_obs_cov = 0.5

        self.logger = logger

    def preprocess_data(self, df):
        """main method to do the feature engineering
        @:param config: source dataframe
        @:return: a DataMatrices object
        """
        try:
            n_jobs = np.clip(len(self.kf_list), 2, 5)

            def extract(d):
                # price_alfa_001 -> 001 -> 0.01
                delta = float(f"0.{d.split('_', 2)[2][1:]}")
                return delta

            delta_list = [extract(v) for v in self.kf_list]
            re_concat = Parallel(n_jobs=n_jobs, )(
                delayed(job_func_Tracer_v2)(df, delta, self.obs_cov, self.sd_delta, self.sd_obs_cov, self.logger) for delta in
                delta_list)

            re_concat_pd = pd.DataFrame(np.array(re_concat).T, columns=self.kf_list)

            return re_concat_pd
        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)


def job_func_Tracer_v2(df, delta, obs_cov, sd_delta, sd_obs_cov, logger, verbose=True):
    alfa_tracer = Tracer_v2('Alfa', delta, delta, obs_cov, sd_delta, sd_obs_cov, logger, verbose=verbose)

    for i, (index, row) in enumerate(df.iteritems()):
        alfa_tracer.update(row)

    re = alfa_tracer.value()

    return re.alfa


def job_func_Tracer_Simple(df, delta, obs_cov, logger, verbose=True):
    simple_tracer = TracerSimple('FeatureEngineer', 'market_kf_ema', delta, obs_cov, logger, verbose=verbose)

    for i, (index, row) in enumerate(df.iteritems()):
        simple_tracer.update(row)

    re = simple_tracer.value()

    return re.tracer


class FeatureEngineer_Market_Capital:
    """Provides methods for preprocessing the stock price data

        Attributes
        ----------
            opts : boolean
                we technical indicator or not
            tech_indicator_list : list
                a list of technical indicator names (modified from neofinrl_config.py)
            use_turbulence : boolean
                use turbulence index or not
            user_defined_feature:boolean
                user user defined features or not

        Methods
        -------
        preprocess_data()
            main method to do the feature engineering

        """

    def __init__(self, market_list, logger):
        self.market_list = market_list
        self.logger = logger

        ema_periods = []
        for i, v in enumerate(market_list):
            # e.g. 'market_open_60_ema', 'market_open_60_ema_slope',
            period = int(v.split('_')[2])
            if period not in ema_periods:
                ema_periods.append(period)
        self.ema_periods = ema_periods  # [60, 30, 15]

        self.logger.info(f'[FeatureEngineer_Market_Capital] ema_periods: {self.ema_periods}')

    def preprocess_data(self, market):
        """main method to do the feature engineering
        @:param config: source dataframe
        @:return: a DataMatrices object
        """
        try:
            re_concat_pd = pd.DataFrame(market['open'])

            if len(self.ema_periods) == 0:
                re_concat_pd.columns = ['market']
            else:
                for period in self.ema_periods:
                    market_ema = talib.EMA(market['open'], period)

                    market_ema_slope = np.zeros(len(market_ema.index))

                    for i, (index, value) in enumerate(market_ema.iteritems()):
                        if i > 0 and not math.isnan(value) and not math.isnan(market_ema[i-1]):
                            market_ema_slope[i] = (value - market_ema[i-1]) / market_ema[i-1] * 100

                    re_pd = pd.DataFrame(np.array([market_ema, market_ema_slope]).T)
                    re_pd.set_index(re_concat_pd.index, inplace=True)

                    re_concat_pd = pd.concat([re_concat_pd, re_pd], axis=1)

                re_concat_pd.columns = ['market'] + self.market_list

            return re_concat_pd
        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)


class FeatureEngineer_TracerSimple:
    def __init__(self, tracer_simple_list, logger):
        self.tracer_simple_list = tracer_simple_list
        self.logger = logger

        # kalman filter configuration
        self.obs_cov = 1
        self.sd_delta = 1e-5
        self.sd_obs_cov = 1

        def extract(d):
            # price_alfa_001 -> 001 -> 0.01
            _delta = float(f"0.{d.split('_')[2][1:]}")
            return _delta

        delta_list = []
        for i, v in enumerate(tracer_simple_list):
            # e.g. ''market_alfa_05', 'market_alfa_05_slope',
            delta = extract(v)
            if delta not in delta_list:
                delta_list.append(delta)

        self.delta_list = delta_list  # [60, 30, 15]

        self.logger.info(f'[FeatureEngineer_Market_KF] delta_list: {self.delta_list}')

    def preprocess_data(self, price):
        try:
            try:
                re_concat_pd = pd.DataFrame([])

                # generating alfa
                n_jobs = np.clip(len(self.delta_list), 2, 5)

                re_concat = Parallel(n_jobs=n_jobs, )(
                    delayed(job_func_Tracer_Simple)(price, delta, self.obs_cov, self.logger) for delta in
                    self.delta_list)

                re_kf_pd = pd.DataFrame(np.array(re_concat).T)  #, columns=self.market_kf_list)

                # generating slope
                for col in re_kf_pd.columns.values:
                    market_kf = re_kf_pd[col]

                    market_kf_slope = np.zeros(len(market_kf.index))

                    for i, (index, value) in enumerate(market_kf.iteritems()):
                        if i > 0 and not math.isnan(value) and not math.isnan(market_kf[i - 1]):
                            market_kf_slope[i] = (value - market_kf[i - 1]) / market_kf[i - 1] * 100 if market_kf[i - 1] > 0 else 0

                    re_pd = pd.DataFrame(np.array([market_kf, market_kf_slope]).T)
                    re_concat_pd = pd.concat([re_concat_pd, re_pd], axis=1)

                re_concat_pd.columns = self.tracer_simple_list

                return re_concat_pd
            except Exception as e:
                err = readable_error(e, __file__)
                self.logger.error(err)
        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)


class FeatureEngineer_Market_KF:
    def __init__(self, market_kf_list, logger):
        self.market_kf_list = market_kf_list
        self.logger = logger

        # kalman filter configuration
        self.obs_cov = 1
        self.sd_delta = 1e-5
        self.sd_obs_cov = 1

        def extract(d):
            # price_alfa_001 -> 001 -> 0.01
            _delta = float(f"0.{d.split('_')[2][1:]}")
            return _delta

        delta_list = []
        for i, v in enumerate(market_kf_list):
            # e.g. ''market_alfa_05', 'market_alfa_05_slope',
            delta = extract(v)
            if delta not in delta_list:
                delta_list.append(delta)

        self.delta_list = delta_list  # [60, 30, 15]

        self.logger.info(f'[FeatureEngineer_Market_KF] delta_list: {self.delta_list}')

    def preprocess_data(self, market):
        try:
            try:
                re_concat_pd = pd.DataFrame([])

                # generating alfa
                n_jobs = np.clip(len(self.delta_list), 2, 5)

                re_concat = Parallel(n_jobs=n_jobs, )(
                    delayed(job_func_Tracer_Simple)(market, delta, self.obs_cov, self.logger) for delta in
                    self.delta_list)

                re_kf_pd = pd.DataFrame(np.array(re_concat).T)  #, columns=self.market_kf_list)

                # generating slope
                for col in re_kf_pd.columns.values:
                    market_kf = re_kf_pd[col]

                    market_kf_slope = np.zeros(len(market_kf.index))

                    for i, (index, value) in enumerate(market_kf.iteritems()):
                        if i > 0 and not math.isnan(value) and not math.isnan(market_kf[i - 1]):
                            market_kf_slope[i] = (value - market_kf[i - 1]) / market_kf[i - 1] * 100 if market_kf[i - 1] > 0 else 0

                    re_pd = pd.DataFrame(np.array([market_kf, market_kf_slope]).T)
                    re_concat_pd = pd.concat([re_concat_pd, re_pd], axis=1)

                re_concat_pd.columns = self.market_kf_list

                return re_concat_pd
            except Exception as e:
                err = readable_error(e, __file__)
                self.logger.error(err)
        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)
