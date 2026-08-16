import os
import gc
import sys
import numpy as np
import pandas as pd
from os import path
import datetime as dt
from ahf.utils.utils import readable_error, create_dir_if_non_exist, datesffmt, pct_change

from ahf.preprocessor.kf.Tracer_v2 import Tracer_v2


class TracerMem_v2(Tracer_v2):
    def __init__(self, trade_args, strategy, job_id, delta, obs_cov, sd_delta, sd_obs_cov, logger,
                 initial_state_mean=None, initial_state_cov=None,
                 initial_price_pct=None,
                 initial_sd=None, initial_sd_state_cov=None,
                 is_unittest=False, verbose=True):

        self.name = 'TracerMem_v2'
        self.strategy = strategy
        self.job_id = job_id
        self.trade_args = trade_args
        self.delta = delta
        self.verbose = True if is_unittest else verbose

        self.theta_id = f"{self.trade_args['exchange']}_{self.trade_args['symbol']}_" \
                        f"{self.trade_args['trade_interval']}_{job_id}"

        if is_unittest:
            self.file_dir = f"{self.trade_args['kf_data_path']}/{self.name}_{self.theta_id}_UNITTEST.csv"
        else:
            self.file_dir = f"{self.trade_args['kf_data_path']}/{self.name}_{self.theta_id}.csv"

        self.file_dir = os.path.expanduser(self.file_dir)
        create_dir_if_non_exist(self.file_dir)

        Tracer_v2.__init__(self, strategy, job_id, delta, obs_cov, sd_delta, sd_obs_cov, logger,
                           initial_state_mean, initial_state_cov,
                           initial_price_pct,
                           initial_sd, initial_sd_state_cov)

    def catchup(self, priceA, tracer_hist_pd=None, min_elapsed=0):
        # IMPORTANT+ CAREFUL: This is used with side-line (Jobber cron job),
        # don't mix parameters with online (running program)
        try:
            if self.verbose:
                self.logger.info(f'[KF] catching up for TracerMem_v2 delta:{self.delta}')
            if priceA.index.name != 'date':
                raise Exception('date index column does not exist')

            if priceA.index.name != 'date':
                price_pd = priceA.copy().set_index('date', drop=True, inplace=False)
            else:
                price_pd = priceA.copy()

            if tracer_hist_pd is None:
                tracer_hist_pd = self.load_tracer()

            if tracer_hist_pd is None:
                self.logger.info('[KF] tracer_history does not exist, building up')
                last_price_dt_index = last_save_dt_index = price_pd.index[0].to_pydatetime()   # .strftime(datefmt)
                last_save_idx = 0

                self.initial_state_mean = initial_state_mean = 0.
                self.initial_state_cov = initial_state_cov = 0.

                self.initial_price_pct = 0.
                self.initial_sd, self.initial_sd_state_cov = initial_sd, initial_sd_state_cov = 0., 0.
            else:
                last_price_dt_index = price_pd.index[-1].to_pydatetime()
                last_save_dt_index = tracer_hist_pd['dt_idx'].iloc[-1]
                last_save_idx = tracer_hist_pd.index[-1]
                # last_save_idx = int(tracer_hist_pd['idx'][-1])  # record it to truncate data for saving

                # load previous stored kf params
                self.initial_state_mean = initial_state_mean = tracer_hist_pd['tracer'].iloc[-1]
                self.initial_state_cov = initial_state_cov = tracer_hist_pd['state_cov00'].iloc[-1]
                self.initial_price_pct = tracer_hist_pd['change'].iloc[-1]

                self.initial_sd = initial_sd = tracer_hist_pd['sd'].iloc[-1]
                self.initial_sd_state_cov = initial_sd_state_cov = tracer_hist_pd['sd_state_cov00'].iloc[-1]

            # capture only since from last save part for training
            t_delta = last_price_dt_index - last_save_dt_index
            if t_delta.total_seconds() < 0:
                raise Exception(f'[KF] price did not catch up with history data, '
                                f'price:{last_price_dt_index}, history:{last_save_dt_index}')

            priceA_train = price_pd[last_save_dt_index:]
            # priceA_train = priceA_train[1:]

            n = len(priceA_train.index)

            """
            # right now, we don't need it, we may need that later

            # do nothing when it is below trade_interval
            if n <= min_elapsed:  # 1 hour
                self.logger.info('[KF] too few new prices, no need to train new mu and sd.')
                # too few data, not need to do anything
                return tracer_hist_pd['mu'][-1], tracer_hist_pd['sd'][-1]
            """

            # Setup variables
            dt_idx = np.zeros(n, dtype=dt.datetime)  # datetime_index
            # TODO disable temporally
            """
            state_means = np.zeros(n)
            state_covs = np.zeros(n)  # np.zeros((n, 2))
            sd_state_means = np.zeros(n)
            sd_state_covs = np.zeros(n)
            """

            if 0 <= n <= 1:
                self.logger.info(f'[KF] up to date till {last_save_dt_index}, no catchup done')

                tracer_hist_pd.set_index('dt_idx', drop=True, inplace=True)
                del tracer_hist_pd['idx']
                tracer_hist_pd = tracer_hist_pd[~tracer_hist_pd.index.duplicated(keep='first')]
                return True, tracer_hist_pd

            # disable gc to speed up calculation in case there are too many data
            gc.disable()
            if self.verbose:
                self.logger.info(f'[KF] Start training new kf prices with delta {self.delta}')

            price_np = priceA_train['open'].to_numpy()
            change_np = pct_change(price_np, include_first=True)
            change_np[0] = self.initial_price_pct

            total = len(change_np)
            for i, v in enumerate(change_np):
                idx_now = self.idx_now

                self.update(v)

                index = priceA_train.index[i]
                dt_idx[idx_now] = index.to_pydatetime()

                self._stdout(idx_now, total)

                # TODO disable temporally
                """
                if len(state_means) >= idx_now:
                    dt_idx = np.append(dt_idx, np.zeros(n, dtype=dt.datetime))
                    state_means = np.append(state_means, np.zeros(n))
                    state_covs = np.append(state_covs, np.zeros(n))
                    sd_state_means = np.append(sd_state_means, np.zeros(n))
                    sd_state_covs = np.append(sd_state_covs, np.zeros(n))
                """

            # save slope and intercept
            need_header = True if tracer_hist_pd is None else False

            # add idx from last saved
            # idx_last = self.idx_now + last_save_idx
            # idx_np = np.arange(start=last_save_idx, stop=idx_last, step=1)
            idx_np = np.arange(start=0, stop=self.idx_now, step=1)

            tracer_np = self.tracer[:self.idx_now]
            alfa_np = self.alfa_norm_100[:self.idx_now]
            sd_np = self.sd[:self.idx_now]
            delta_np = self.deltas[:self.idx_now]
            state_cov_np = self.state_cov[:self.idx_now]
            sd_state_cov_np = self.sd_state_cov[:self.idx_now]

            # only save when it is larger than 1, otherwise there are lots of zeros
            if self.idx_now > 1:
                self._save_tracer(self.file_dir, idx_np, dt_idx, price_np, change_np, tracer_np,
                                  alfa_np, sd_np, delta_np, state_cov_np, sd_state_cov_np, need_header)

            # save
            gc.enable()

            new = {
                'idx': idx_np, 'dt_idx': dt_idx, 'price': price_np, 'change': change_np, 'tracer': tracer_np,
                'alfa': alfa_np, 'sd': sd_np,
                'delta': delta_np, 'state_cov00': state_cov_np, 'sd_state_cov00': sd_state_cov_np
            }

            new_pd = pd.DataFrame(new)
            if len(new_pd.index) > 0:
                concat_pd = pd.concat([tracer_hist_pd, new_pd], axis=0) if tracer_hist_pd is not None else new_pd
            else:
                concat_pd = tracer_hist_pd

            concat_pd.set_index('dt_idx', drop=True, inplace=True)
            del concat_pd['idx']
            concat_pd = concat_pd[~concat_pd.index.duplicated(keep='first')]
            # concat_pd.drop_duplicates(subset=['dt_idx'], keep='first', inplace=True)
            # concat_pd.set_index('dt_idx', keep='first', drop=True, inplace=True)

            return True, concat_pd

        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)
            return False, {'err': err}

    def load_tracer(self):
        """
        result from parent update()
        {'price_pct': price_pct, 'tracer': tracer_new, 'alfa':alfa_new,
            'sd': sd_new, 'delta': delta}
        :return:
        """
        try:
            if not os.path.exists(self.file_dir):
                return None
                # r = pd.DataFrame([], columns=['idx', 'slope', 'intercept', 'tracer', 'mu', 'sd',
                #                              'state_cov00', 'state_cov01', 'state_cov10', 'state_cov11'])
                # r.to_csv(file_dir, mode='w', header=True, index=True, na_rep='NA', index_label='dt_idx')

            tracer_hist = pd.read_csv(self.file_dir,
                                      usecols=[
                                          'idx', 'dt_idx', 'price', 'change', 'tracer', 'alfa', 'sd',
                                          'delta', 'state_cov00', 'sd_state_cov00'
                                      ],
                                      # index_col=0, parse_dates=True,
                                      dtype={
                                          'idx': int, 'dt_idx': str, 'price': float, 'change': float,
                                          'tracer': float, 'alfa': float, 'sd': float, 'delta': float,
                                          'state_cov00': float, 'sd_state_cov00': float
                                      })

            tracer_hist['dt_idx'] = pd.to_datetime(tracer_hist['dt_idx'])
            return tracer_hist
        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)
            return None

    def _stdout(self, idx_now:int, total: int):
        # TODO disable temporally
        """
        if idx_now == 0:  # Initialize the Kalman filter
            state_means[idx_now], state_covs[idx_now] = initial_state_mean, initial_state_cov
            sd_state_means[idx_now], sd_state_covs[idx_now] = initial_sd, initial_sd_state_cov
        else:
            # start
            state_means[idx_now], state_covs[idx_now] = self.kf.state_means[idx_now], self.kf.state_covs[idx_now]
            sd_state_means[idx_now], sd_state_covs[idx_now] = self.kf.sd.kf.state_means[idx_now], self.kf.sd.kf.state_covs[idx_now]
        """

        n = 10000
        if idx_now % 1000 == 0:
            msg = '\r{0}, INFO {1} [KF] ' \
                  'Training at {2:.2f}%\r'.format(dt.datetime.now().strftime(datesffmt),
                                                  f"{self.trade_args['exchange']}_"
                                                  f"{self.trade_args['symbol']}",
                                                  idx_now / total * 100)
            sys.stdout.write(msg)
            sys.stdout.flush()
