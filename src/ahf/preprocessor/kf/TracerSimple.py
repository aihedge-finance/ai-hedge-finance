import sys
import pandas as pd
import numpy as np
from ahf.utils.utils import readable_error, datefmt, create_dir_if_non_exist
# from ahf.preprocessor.kf.KalmanMovingAverage import KalmanMovingAverage
from ahf.preprocessor.kf.kalman_moving_average import KalmanMovingAverage

class TracerSimple(object):
    def __init__(self, strategy, job, delta, obs_cov, logger,
                 initial_state_mean: float=None,
                 initial_state_cov: float=None, verbose=True):

        self.delta = delta
        self.obs_cov = obs_cov
        self.strategy = strategy
        self.job = job

        self.kf = None
        self.idx_now = 0
        self.verbose = verbose

        self.initial_state_mean: float = initial_state_mean
        self.initial_state_cov: float = initial_state_cov

        # Series
        self.tracer = np.zeros(10000)
        self.state_cov = np.zeros(10000)
        self.logger = logger

    def update(self, price: float, delta=None, obs_cov=None, is_save=False):
        """
        一步步的去做計算
        """
        try:
            idx_now = self.idx_now

            if idx_now >= len(self.tracer):
                self.tracer = np.append(self.tracer, np.zeros(10000))
                self.state_cov = np.append(self.state_cov, np.zeros(10000))

            price = np.nan_to_num(price)

            # start
            if idx_now == 0:  # Initialize the Kalman filter
                initial_state_mean: float = price if self.initial_state_mean is None else self.initial_state_mean
                initial_state_cov: float = 1.0 if self.initial_state_cov is None else self.initial_state_cov
                # first loop, use original self.delta
                trans_cov = self.delta / (1 - self.delta) * np.ones(1)

                self.kf = KalmanMovingAverage(self.obs_cov, initial_state_mean, initial_state_cov, trans_cov)

                tracer_new = self.kf.state_means[idx_now]
                state_cov_new = self.kf.state_covs[idx_now]
                alfa_new = price - tracer_new
                # TODO: change 1, add next_mean
                next_mean = tracer_new
            else:
                # ==== IMPORTANT =======
                # if there is a change then delta will not be None
                trans_cov = delta / (1 - delta) * np.ones(1) if delta is not None else None
                obs_cov = obs_cov if obs_cov is not None else self.obs_cov
                # TODO: change 1, add next_mean
                if trans_cov is None:
                    next_mean, next_covariance = self.kf.update(idx_now, price)
                else:
                    next_mean, next_covariance = self.kf.update(idx_now, price, trans_cov)
                tracer_new = self.kf.state_means[idx_now]
                state_cov_new = self.kf.state_covs[idx_now]

                assert next_mean == tracer_new, 'Wrong next_mean at TracerSimple'
                assert next_covariance == next_covariance, 'next_covariance at TracerSimple'

            self.tracer[idx_now] = tracer_new
            self.state_cov[idx_now] = state_cov_new

            # sys.stdout.write('Progress sd {0}: {1}\r'.format(self.job, round(idx_now, 1)))
            # sys.stdout.flush()

            self.idx_now += 1

            return tracer_new

        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)

    def value(self, start=None, end=None):
        re = lambda: None
        if start is not None and end is not None:
            if start < 0:
                start = 0
            re.tracer = self.tracer[start:end]
            re.delta = self.delta

            return re
        else:
            i = self.idx_now
            re.tracer = self.tracer[:i]
            re.delta = self.delta

            return re

    def _save_tracer(self, file_dir, idx_run_np, dt_idx, price_np, tracer, slope, delta, state_covs,
                     need_header=False):
        """
        result from parent update()=tracer_new

        :param file_dir: usage => f"{self.trade_args['data_path']}/KF_theta/{self.name}_{self.theta_id}.csv"
        :param idx_run_np:
        :param dt_idx:
        :param price_np:
        :param tracer:
        :param delta:
        :param state_covs:
        :param need_header:
        :return:
        """

        try:
            state_covs00 = state_covs.flatten()
            # state_covs01 = state_covs[:, 1].flatten()

            dt_idx_str = [x.strftime(datefmt) for x in dt_idx]
            data = np.column_stack(
                [
                    dt_idx_str,
                    price_np,
                    tracer,
                    slope,
                    delta,
                    state_covs00
                ])
            r = pd.DataFrame(data, index=idx_run_np,
                             columns=['dt_idx', 'price', 'tracer', 'slope', 'delta', 'state_cov00'])

            create_dir_if_non_exist(file_dir)
            # r['idx'] = r['idx'].astype(int)

            # file_dir = f"{self.trade_args['data_path']}/KF_theta/{self.name}_{self.theta_id}.csv"
            r.to_csv(file_dir, mode='a', header=need_header, index=True, na_rep='None', index_label='idx')
            if self.verbose:
                self.logger.info(f'[KF] New KF_theta tracer, alfa, sd are saved to \n{file_dir}')
        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(err)
            sys.exit()

