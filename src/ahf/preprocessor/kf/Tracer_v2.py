import sys
import numpy as np
import pandas as pd

import app.utils as utils
import matplotlib.pyplot as plt
from collections import namedtuple
from ahf.utils.utils import readable_error, safe_div, create_dir_if_non_exist
# from ahf.preprocessor.kf.KalmanMovingAverage import KalmanMovingAverage
from ahf.preprocessor.kf.kalman_moving_average import KalmanMovingAverage
from ahf.preprocessor.kf.TracerSimple import TracerSimple

TracerState = namedtuple("TracerState",
                         ["idx",
                          "dt_idx",
                          "price_pct",
                          "tracer",
                          "alfa",
                          "alfa_norm_100",
                          "sd",
                          "delta",
                          "state_cov",
                          "sd_state_cov"])

class Tracer_v2(object):
    def __init__(self, strategy, job, delta, obs_cov, sd_delta, sd_obs_cov, logger,
                 initial_state_mean=None, initial_state_cov=None,
                 initial_price_pct=None,
                 initial_sd=None, initial_sd_state_cov=None, verbose=True):

        self.delta = delta
        self.obs_cov = obs_cov
        self.strategy = strategy
        self.job = job

        self.logger = logger
        self.verbose = verbose

        self.kf = None

        self.idx_now = 0

        # Series
        self.dt_idx = np.zeros(10000)
        self.price = np.zeros(10000)
        self.price_pct = np.zeros(10000)
        self.tracer = np.zeros(10000)
        self.alfa = np.zeros(10000)
        self.alfa_norm = np.zeros(10000)
        self.alfa_norm_100 = np.zeros(10000)
        # self.mu = np.zeros(10000)
        self.sd = np.zeros(10000)
        self.deltas = np.zeros(10000)
        self.state_cov = np.zeros(10000)
        self.sd_state_cov = np.zeros(10000)

        self.look_back = 14

        self.sd_delta = sd_delta
        self.sd_obs_cov = sd_obs_cov

        self.initial_price_pct = initial_price_pct
        self.initial_state_mean = initial_state_mean
        self.initial_state_cov = initial_state_cov

        self.initial_sd = initial_sd
        self.initial_sd_state_cov = initial_sd_state_cov

        sd_tracer_period = 0  # start accumulating data
        # self.sd_tracer = Tracer_v2(self.strategy, '{0}_sd_tracer'.format(job), self.sd_delta, self.sd_obs_cov,
        #                            None, None, sd_tracer_period, logger) if sd_delta is not None else None
        self.sd_tracer = None

    def train(self, price_pct, proc_num=None, return_dict=None):
        try:
            if self.strategy == 'brunhild_naive':
                n = len(price_pct)

                # Setup variables
                idx_now = 0
                tracer = np.zeros(n)
                alfa = np.zeros(n)

                # mu = np.zeros(n)
                sd = np.zeros(n)

                print('[Tracer] Building {0} {1} ...'.format(self.strategy, self.job))

                # load previous stored kf params
                # price_pd.fillna(0, inplace=True)
                price_pct = np.nan_to_num(price_pct)

                initial_state_mean = price_pct[0]
                initial_state_cov = 1.0
                initial_sd_state_cov = 1.0
                trans_cov = self.delta / (1 - self.delta) * np.ones(1)

                for i, v in enumerate(price_pct):
                    if self.idx_now == 0:  # Initialize the Kalman filter
                        self.kf = KalmanMovingAverage(self.obs_cov, initial_state_mean, initial_state_cov, trans_cov)
                        # kf.state_means[i], kf.state_covs[i] = initial_state_mean, initial_state_cov

                        tracer[idx_now] = self.kf.state_means[idx_now]
                        alfa[idx_now] = price_pct[idx_now] - tracer[idx_now]
                    else:
                        if np.isnan(v):
                            v = None
                        # start
                        next_mean, next_covariance = self.kf.update(idx_now, v)
                        tracer[idx_now] = self.kf.state_means[idx_now]
                        alfa[idx_now] = price_pct[idx_now] - tracer[idx_now]

                        sys.stdout.write('\rProgress: {0}%'.format(round(idx_now / n * 100, 1)))
                        sys.stdout.flush()

                    idx_now += 1
                    self.idx_now = idx_now

                # calculate mu and std, finally
                if self.sd_tracer is None:
                    # mu[:] = np.mean(tracer)
                    sd[:] = np.std(tracer)
                else:
                    # mu[:] = np.mean(tracer)
                    price_pct_sqr = abs(price_pct)  # np.sqrt(alfa ** 2)
                    sd = self.sd_tracer.train(price_pct_sqr)

                # calculate alfa_norm
                alfa_norm = safe_div(alfa, sd)  # sd
                # max_buy_alfa_norm = np.max(alfa_norm)
                alfa_norm_100 = safe_div(tracer, sd) / 2  # assume 2 std is high, normalize to 0-1

                self.price_pct = price_pct
                self.tracer = tracer
                self.alfa = alfa
                self.alfa_norm = alfa_norm
                self.alfa_norm_100 = alfa_norm_100
                # self.mu = mu
                self.sd = sd
                self.deltas[:] = self.delta

                mu = None

                # for multiprocess
                if proc_num is not None:
                    return_dict[proc_num] = mu, sd, tracer, alfa, alfa_norm_100

                return mu, sd, tracer, alfa, alfa_norm_100
            else:
                return None, None, None, None, None

        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)

    def values(self, start=None, end=None):
        re = lambda: None
        if start is not None and end is not None:
            if start < 0:
                start = 0

            re.idx_dt = self.dt_idx[start:end]
            re.price_pct = self.price_pct[start:end]
            re.tracer = self.tracer[start:end]
            re.alfa = self.alfa_norm_100[start:end]
            re.sd = self.sd[start:end]
            re.delta = self.deltas[start:end]

            return re
        else:
            i = self.idx_now
            re.idx_dt = self.dt_idx[:i]
            re.price_pct = self.price_pct[:i]
            re.tracer = self.tracer[:i]
            re.alfa = self.alfa_norm_100[:i]
            re.sd = self.sd[:i]
            re.delta = self.deltas[:i]

            return re

        # return self.mu[:self.idx_now], self.sd[:self.idx_now], \
        #       self.tracer[:self.idx_now], self.alfa[:self.idx_now], self.alfa_norm_100[:self.idx_now]

    def value(self, pos):
        re = lambda: None

        re.idx_dt = self.dt_idx[pos]
        re.price_pct = self.price_pct[pos]
        re.tracer = self.tracer[pos]
        re.alfa = self.alfa_norm_100[pos]
        re.sd = self.sd[pos]
        re.delta = self.deltas[pos]

        return re

    def update(self, price_pct: float, delta=None, dt_idx=None) -> TracerState:
        """
        This is an updated version from version 1, which trace price directly.
        Version 2 modify and trace price_pct instead so that it is more generic
        but it is less intuitive for human. Since it is tracing price_pct, so
        for backward compatibility, '' is useless now.

        self.tracer is tracing price_pct
        self.alfa is price_pct - self.tracer
        self.sd is independent simpleTraer that track std of the price_pct
        self.mu is deprecated, but we keep it for backward spec compatibility
        self.alfa_norm is self.alfa
        self.alfa_norm_100 is tracer_new / sd_new / 2 (assume 2 std is high, we try to normalize to 0-1)

        seriously i don't know which is more useful, but in human view, alfa_norm_100 seems to be
        more useful. But we can use ML to decide which variables are useful in future version.

        Right now, use alfa_norm_100 as final BENCHMARK

        :param price_pct: Percentage change in target price
        :param delta: delta of your choice for Kalman Filter
        :param dt_idx: [Optional] dt_idx
        :return: mu_new, sd_new, tracer_new, alfa_new, alfa_norm_100_new


        """
        try:
            assert isinstance(price_pct, float), "price_pct has to be of type float for calculation"
            idx_now = self.idx_now

            if idx_now + 1 >= len(self.tracer):
                self.dt_idx = np.append(self.dt_idx, np.zeros(10000))
                self.price_pct = np.append(self.price_pct, np.zeros(10000))
                self.tracer = np.append(self.tracer, np.zeros(10000))
                self.alfa = np.append(self.alfa, np.zeros(10000))
                self.alfa_norm = np.append(self.alfa_norm, np.zeros(10000))
                self.alfa_norm_100 = np.append(self.alfa_norm_100, np.zeros(10000))
                # self.mu = np.append(self.mu, np.zeros(10000))
                self.sd = np.append(self.sd, np.zeros(10000))
                self.deltas = np.append(self.deltas, np.zeros(10000))
                self.state_cov = np.append(self.state_cov, np.zeros(10000))
                self.sd_state_cov = np.append(self.sd_state_cov, np.zeros(10000))


            # start
            # ==== IMPORTANT =======
            # if there is a change then delta will not be None
            delta = self.delta if delta is None else delta

            if idx_now == 0:  # Initialize the Kalman filter
                price_pct = np.nan_to_num(price_pct)

                initial_state_mean = price_pct if self.initial_state_mean is None else self.initial_state_mean
                initial_state_cov = state_cov_new = 0.1 if self.initial_state_cov is None else self.initial_state_cov

                initial_sd = 0. if self.initial_sd is None else self.initial_sd
                initial_sd_state_cov = 0.1 if self.initial_sd_state_cov is None else self.initial_sd_state_cov
                initial_price_pct = price_pct if self.initial_price_pct is None else self.initial_price_pct

                # first loop, use original self.delta
                trans_cov = self.delta / (1 - self.delta) * np.ones(1)

                self.kf = KalmanMovingAverage(self.obs_cov, initial_state_mean, initial_state_cov, trans_cov)

                tracer_new = self.kf.state_means[idx_now]
                alfa_new = initial_price_pct - tracer_new

                if self.sd_delta is not None:
                    self.sd_tracer = TracerSimple(self.strategy, "{0}_sd_tracer".format(self.job), self.sd_delta,
                                                  self.sd_obs_cov, self.logger,
                                                  initial_sd, initial_sd_state_cov)

            else:
                if np.isnan(price_pct):
                    price_pct = None

                trans_cov = delta / (1 - delta) * np.ones(1)

                next_mean, state_cov_new = self.kf.update(idx_now, price_pct, trans_cov)
                tracer_new = self.kf.state_means[idx_now]
                alfa_new = price_pct - tracer_new

            self.dt_idx[idx_now] = dt_idx
            self.price_pct[idx_now] = price_pct
            self.tracer[idx_now] = tracer_new
            self.alfa[idx_now] = alfa_new
            self.deltas[idx_now] = delta
            self.state_cov[idx_now] = state_cov_new

            # calculate mu and std, finally
            # mu_new = np.mean(self.kf.state_means)
            if self.sd_tracer is None:
                sd_new = np.std(self.kf.state_means)
                self.sd[idx_now] = sd_new
                # self.mu[idx_now] = mu_new
                sd_state_cov_new = 0.
                self.sd_state_cov[idx_now] = sd_state_cov_new
            else:
                # self.mu[idx_now] = mu_new
                price_pct_new = abs(price_pct)  # np.sqrt(alfa_new ** 2)
                sd_new = self.sd_tracer.update(price_pct_new)
                self.sd[idx_now] = sd_new  # sd moving average

                sd_state_cov_new = self.sd_tracer.kf.state_covs[idx_now]
                self.sd_state_cov[idx_now] = sd_state_cov_new

            # calculate alfa_norm
            if idx_now == 0:
                # this is to consider we have history record
                # DEPRECATED alfa_norm_new, alfa_norm_100_new = 0., 0.
                alfa_norm_new = alfa_new if alfa_new != 0 else 0.
                alfa_norm_100_new = safe_div(tracer_new, sd_new) / 2 if tracer_new != 0 and sd_new != 0 else 0.
            else:
                alfa_norm_new = alfa_new  # safe_div(alfa_new, sd_new)
                # max_alfa_norm = np.max(self.alfa_norm)

                # This period is for getting an estimate of std
                if self.sd_delta is None:
                    alfa_norm_100_new = tracer_new / 2  # * sd_new
                else:
                    alfa_norm_100_new = safe_div(tracer_new, sd_new) / 2  # assume 2 std is high, normalize to 0-1

            self.alfa_norm[idx_now] = alfa_norm_new
            self.alfa_norm_100[idx_now] = alfa_norm_100_new

            # sys.stdout.write(f"\rProgress delta {self.job:>6}: {idx_now:>6.0f} ")
            # sys.stdout.flush()

            self.idx_now += 1

            """
            re.price_pct = self.price_pct[:i]
            re.tracer = self.tracer[:i]
            re.alfa = self.alfa_norm_100[:i]
            re.sd = self.sd[:i]
            re.delta = self.deltas[:i]
            """
            idx_prior = max(0, idx_now - 1)
            return TracerState(**{
                "idx": idx_now,
                "dt_idx": [self.dt_idx[idx_prior], dt_idx],
                "price_pct": [self.price_pct[idx_prior], price_pct],
                "tracer": [self.tracer[idx_prior], tracer_new],
                "alfa": [self.alfa[idx_prior], alfa_new],
                "alfa_norm_100": [self.alfa_norm_100[idx_prior], alfa_norm_100_new],
                "sd": [self.sd[idx_prior], sd_new],
                "delta": [self.deltas[idx_prior], delta],
                "state_cov": [self.state_cov[idx_prior], state_cov_new],
                "sd_state_cov": [self.sd_state_cov[idx_prior], sd_state_cov_new],
            })

            # return mu_new, sd_new, tracer_new, alfa_new, alfa_norm_100_new

        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)
            sys.exit()

    def _save_tracer(self, file_dir, idx_run_np, dt_idx, price_np, change_np, tracer, alfa, sd, delta,
                     state_covs, sd_state_covs,
                     need_header=False):
        """
        result from parent update()
        {'price_pct': price_pct, 'tracer': tracer_new, 'alfa':alfa_new, 'sd': sd_new, 'delta': delta}

        :param file_dir: usage => f"{self.trade_args['data_path']}/KF_theta/{self.name}_{self.theta_id}.csv"
        :param idx_run_np:
        :param dt_idx:
        :param price_np:
        :param change_np:
        :param tracer:
        :param alfa:
        :param sd:
        :param delta:
        :param state_covs:
        :param sd_state_covs:
        :param need_header:
        :return:
        """

        try:
            state_covs00 = state_covs.flatten()
            sd_state_covs00 = sd_state_covs.flatten()
            # state_covs01 = state_covs[:, 1].flatten()

            data = np.column_stack(
                [
                    dt_idx,
                    price_np,
                    change_np,
                    tracer,
                    alfa,
                    sd,
                    delta,
                    state_covs00,
                    sd_state_covs00
                ])
            r = pd.DataFrame(data, index=idx_run_np,
                             columns=["dt_idx", "price", "change", "tracer", "alfa", "sd", "delta",
                                      "state_cov00", "sd_state_cov00"])

            # r["idx"] = r["idx"].astype(int)

            create_dir_if_non_exist(file_dir)
            # file_dir = f"{self.trade_args["data_path"]}/KF_theta/{self.name}_{self.theta_id}.csv"
            r.to_csv(file_dir, mode="a", header=need_header, index=True, na_rep="None", index_label="idx")
            if self.verbose:
                self.logger.info("[KF] New KF_theta tracer, alfa, sd are saved to \n{0}".format(file_dir))
        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)
            sys.exit()


def load_price():
    exchange = 'Binance'
    symbol = 'SOLUSDT'
    data_path = './appData/trainData_crypto/prices_v3.parquet'
    days = 50
    # form_start = (dt.date.today() - dt.timedelta(days=days)).strftime('%Y-%m-%d')
    form_start = '2021-01-01'
    form_end = '2021-03-01'
    klines = utils.read_data(exchange, symbol, data_path, ['open', 'close']).compute()
    price = pd.DataFrame(klines, columns=utils.price_columns, index=klines.index)
    # price.index = pd.to_datetime(priceA.date)
    price_pd = price.close
    price_pd = price_pd[form_start:form_end]

    price_pd = price_pd.resample(rule='2h', label='left', closed='right').mean().interpolate()
    # price_pct = price_pd.pct_change()[1:]

    logger = utils.setup_logger('brunhild_naive_tracer_unit_test.log', symbol)

    strategy = 'brunhild_naive'

    buy_delta = 0.6
    buy_sd_delta = 1e-4

    return exchange, symbol, days, form_start, price_pd, logger, strategy, buy_delta, buy_sd_delta


# def unit_test(buy_tracer_diff, buy_alfa_diff, buy_alfa_norm_diff, buy_alfa_norm_100_diff, buy_sd_diff):
def unit_test():
    exchange, symbol, days, form_start, price_pd, logger, strategy, buy_delta, buy_sd_delta = load_price()

    price_np = price_pd.to_numpy()

    # buy_ref_delta = 1e-1
    # sell_delta = 1e-2
    buy_obs_cov = buy_sd_obs_cov = sell_sd_obs_cov = 1
    # sell_obs_cov = 0.3

    sd_period = 100  # means no lookback

    # ==== buy signal ====
    BuyTracer = Tracer_v2(strategy, 'buy_tracer', buy_delta, buy_obs_cov,
                          buy_sd_delta, buy_sd_obs_cov, sd_period, logger)
    _, buy_sd_mv, buy_tracer, buy_alfa, buy_alfa_norm_100 = BuyTracer.train(price_np)

    # BuyRefTracer = Tracer(strategy, 'buy_ref_tracer', buy_ref_delta, buy_obs_cov,
    #                      buy_ref_sd_delta, buy_obs_cov, sd_period, logger)
    # _, _, buy_ref_tracer, buy_ref_alfa, buy_ref_alfa_norm_100 = BuyRefTracer.train(price_np)

    mu = np.mean(price_np)
    sd = np.std(price_np)
    tracer_entry_level = 0.8

    # REFERENCE: divide by zero
    # https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero
    # >>> a = np.array([-1, 0, 1, 2, 3], dtype=float)
    # >>> b = np.array([ 0, 0, 0, 2, 2], dtype=float)
    #
    # # If you don't pass `out` the indices where (b == 0) will be uninitialized!
    # >>> c = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    # >>> print(c)
    # [ 0.   0.   0.   1.   1.5]
    # In this case, it does the divide calculation anywhere 'where' b does not equal zero. When b does equal zero,
    # then it remains unchanged from whatever value you originally gave it in the 'out' argument.

    # === sell ====
    # buy_alfa_norm_dynamic.dropna(how='any', inplace=True)
    # SellTracer = Tracer(strategy, 'sell_tracer', sell_delta, sell_obs_cov,
    #                    sell_sd_delta, sell_sd_obs_cov, sd_period, logger)
    # _, sell_sd_mv, sell_tracer, sell_alfa, sell_alfa_norm = SellTracer.train(buy_alfa_norm_100)

    # ==== moving update: compare the difference =====
    # ==== buy signal ====

    mvt_price_pd_train = price_pd[:sd_period + 1].copy()
    mvt_price_pd_test = price_pd[sd_period + 1:].copy()
    mvt_price_np_train = mvt_price_pd_train.to_numpy()
    mvt_price_np_test = mvt_price_pd_test.to_numpy()

    # BuyTracer
    BuyTracerRun = Tracer_v2(strategy, 'buy_tracer', buy_delta, buy_obs_cov,
                             buy_sd_delta, buy_sd_obs_cov, sd_period, logger)
    _, mvt_buy_sd_mv, mvt_buy_tracer, mvt_buy_alfa, mvt_buy_alfa_norm_100 = BuyTracerRun.train(mvt_price_np_train)

    for idx, v in enumerate(mvt_price_np_test):
        _, sd_new, tracer_new, alfa_new, alfa_norm_new = BuyTracerRun.update(v)

    buy_tracer_running = BuyTracerRun.tracer[:BuyTracerRun.idx_now]
    buy_alfa_running = BuyTracerRun.alfa[:BuyTracerRun.idx_now]
    buy_alfa_norm_running = BuyTracerRun.alfa_norm[:BuyTracerRun.idx_now]
    buy_alfa_norm_100_running = BuyTracerRun.alfa_norm_100[:BuyTracerRun.idx_now]

    value = BuyTracerRun.value()
    buy_alfa_norm_x100_running = value['alfa']
    buy_sd_running = value['sd']
    buy_sd_running_fixed = np.std(BuyTracerRun.kf.state_means)
    buy_state_cov = BuyTracerRun.kf.state_covs

    plt.figure(1, figsize=(15, 8))
    ax1 = plt.subplot(311)
    plt.plot(price_pd[10:], label="price")
    plt.plot(price_pd.index[10:], buy_tracer[10:], label="buy tracer", ls='--', lw=1)
    plt.plot(price_pd.index[10:], buy_tracer_running[10:], label="buy tracer running", ls='-.', lw=1)

    plt.grid()
    plt.legend(loc="best")

    ax2 = plt.subplot(312, sharex=ax1)
    # plt.plot(buy_alfa_norm_static_pd[10:], label="buy signal static alfa", ls='-', lw=1)
    plt.plot(price_pd.index[10:], buy_alfa_norm_100[10:], label="buy alfa norm 100", ls='-.', lw=1)
    # plt.plot(price_pd.index[10:], buy_tracer_diff[10:]/buy_sd_diff[10:]/2, label="buy tracer/sd", ls='-.', lw=1)

    # enable for checking
    # plt.plot(price_pd.index[10:], buy_alfa_norm_running[10:], label="buy_alfa_norm_running", ls='-.', lw=1)
    plt.plot(price_pd.index[10:], buy_alfa_norm_100_running[10:], label="buy_alfa_norm_100_running", ls='-.', lw=1)
    # plt.plot(price_pd.index[10:], buy_alfa_norm_running[10:] / buy_sd_running_fixed,
    #         label="buy_alfa_norm_running/sd_fixed", ls='-.', lw=1)
    # useless
    # plt.plot(price_pd.index[10:], buy_state_cov[10:], label="buy state cov", ls='-.', lw=1)

    # plt.plot(price_pd.index[10:], sell_tracer[10:], label="sell tracer", ls='-.', lw=1)
    plt.axhline(0, color='black', ls='-.', lw=1)
    plt.axhline(tracer_entry_level, color='green', ls='-.', lw=1)
    plt.axhline(-tracer_entry_level, color='green', ls='-.', lw=1)
    # plt.plot(-alfa_pd2[10:]/sd2[0], label="alfa 2")

    plt.grid()
    plt.legend(loc="lower right")

    ax3 = plt.subplot(313, sharex=ax1)
    plt.plot(price_pd.index[10:], buy_alfa_running[10:], label="buy_alfa_running", ls='-.', lw=1)
    plt.plot(price_pd.index[10:], buy_sd_running[10:], label="buy sd", ls='-.', lw=1)

    # plt.axhline(buy_sd_running_fixed, label="buy_sd_running_fixed", ls='-.', lw=1)
    plt.axhline(0, color='black', ls='-.', lw=1)
    ## plt.plot(-alfa_pd2[10:]/sd2[0], label="alfa 2")
    plt.grid()
    plt.legend(loc="lower right")

    return buy_tracer_running, buy_alfa_running, buy_alfa_norm_running, buy_alfa_norm_100_running, buy_sd_running


def unit_test_running(buy_tracer_v1=None, buy_alfa_v1=None, buy_alfa_norm_v1=None, buy_alfa_norm_100_v1=None,
                      buy_sd_v1=None):
    exchange, symbol, days, form_start, price_pd, logger, strategy, buy_delta, buy_sd_delta = load_price()

    price_np = price_pd.to_numpy()
    price_pct = utils.pct_change(price_np, include_first=True)

    buy_obs_cov = 1.
    buy_sd_obs_cov = 0.1

    tracer_entry_level = 0.2

    period = 0  # means no lookback

    # ==== buy signal ====
    BuyTracerRun = Tracer_v2(strategy, 'buy_tracer', buy_delta, buy_obs_cov,
                             buy_sd_delta, buy_sd_obs_cov, logger)

    for idx, v in enumerate(price_pct):
        BuyTracerRun.update(v)

    buy_tracer = BuyTracerRun.tracer[:BuyTracerRun.idx_now]
    buy_sd = BuyTracerRun.sd[:BuyTracerRun.idx_now]
    buy_alfa = BuyTracerRun.alfa[:BuyTracerRun.idx_now]
    buy_alfa_norm = BuyTracerRun.alfa_norm[:BuyTracerRun.idx_now]
    buy_alfa_norm_100 = BuyTracerRun.alfa_norm_100[:BuyTracerRun.idx_now]

    dt_index = price_pd.index[10:]

    plt.figure(2, figsize=(15, 8))
    ax1 = plt.subplot(411)
    plt.plot(price_pd[10:], label="price", lw=0.8)
    plt.grid()
    plt.legend(loc="lower right")

    # AX2
    ax2 = plt.subplot(412, sharex=ax1)
    # plt.plot(dt_index, price_pct[10:], label="price", ls='-.', lw=1)
    plt.plot(dt_index, buy_tracer[10:], label="buy tracer", lw=0.8)
    plt.plot(dt_index, buy_sd[10:], label="buy sd", ls='-.', lw=1)
    plt.axhline(0, color='black', ls='--', lw=1)
    plt.axhline(0.2, color='green', ls='-.', lw=1)
    plt.axhline(-0.2, color='green', ls='-.', lw=1)
    plt.grid()
    plt.legend(loc="lower right")

    # AX3
    ax3 = plt.subplot(413, sharex=ax1)
    plt.plot(dt_index, (buy_alfa_norm_100[10:]), label="buy tracer norm 100(BENCHMARK)", lw=0.8)
    # plt.plot(dt_index, buy_alfa_norm[10:], label="buy_alfa_norm", lw=1)
    if buy_alfa_norm_100_v1 is not None:
        plt.plot(dt_index, buy_alfa_norm_100_v1[10:], label="buy_alfa_norm_100_v1", lw=1)

    # plt.plot(price_pd.index[10:], sell_tracer[10:], label="sell tracer", ls='-.', lw=1)
    plt.axhline(0, color='black', ls='-.', lw=1)
    plt.axhline(tracer_entry_level, color='green', ls='-', lw=0.5)
    plt.axhline(-tracer_entry_level, color='green', ls='-', lw=0.5)

    plt.axhline(1, color='green', ls='--', lw=0.5)
    plt.axhline(-1, color='green', ls='--', lw=0.5)
    plt.axhline(2, color='green', ls='--', lw=0.5)
    plt.axhline(-2, color='green', ls='--', lw=0.5)

    plt.grid()
    plt.legend(loc="lower right")

    # AX4
    ax4 = plt.subplot(414, sharex=ax1)
    plt.plot(dt_index, buy_alfa[10:], label="buy_alfa", lw=0.8)

    plt.axhline(0, color='black', ls='-.', lw=1)
    plt.axhline(1, color='green', ls='--', lw=0.5)
    plt.axhline(-1, color='green', ls='--', lw=0.5)
    plt.axhline(2, color='green', ls='--', lw=0.5)
    plt.axhline(-2, color='green', ls='--', lw=0.5)

    plt.grid()
    plt.legend(loc="lower right")

    plt.show()

    # plt.show()
    return buy_tracer, buy_alfa, buy_alfa_norm, buy_alfa_norm_100, buy_sd


# DEBUG
if __name__ == '__main__':
    # buy_tracer, buy_alfa, buy_alfa_norm, buy_alfa_norm_100, buy_sd = unit_test()

    # unit_test_running(buy_tracer, buy_alfa, buy_alfa_norm, buy_alfa_norm_100, buy_sd)
    unit_test_running()
