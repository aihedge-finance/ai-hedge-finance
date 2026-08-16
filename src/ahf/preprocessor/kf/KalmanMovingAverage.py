import pandas as pd
import datetime as dt
from typing import Optional, Union
from pykalman import KalmanFilter


class KalmanMovingAverage(object):
    """
    Estimates the moving average of a price process via Kalman Filtering, using pykalman
    P: state_covariance
    Q: transition_covariance
    R: observation_covariance
    """

    def __init__(self, observation_covariance: float = 1.0,
                 initial_state_mean: float = 0,
                 initial_state_covariance: float = 1.0,
                 transition_covariance: float = 0.05,
                 initial_dt: Union[int, dt.datetime] = 0):
        """

        Parameters
        ----------
        observation_covariance: R, measurement error
        initial_state_mean: initial price
        initial_state_covariance: P, state_covariance
        transition_covariance: Q, system error
        initial_dt: you can input datetime or number
        """
        # self.asset = asset

        self.kf = KalmanFilter(transition_matrices=[1],
                               observation_matrices=[1],
                               initial_state_mean=initial_state_mean,
                               initial_state_covariance=initial_state_covariance,
                               observation_covariance=observation_covariance,
                               transition_covariance=transition_covariance)

        self.state_means = pd.Series([self.kf.initial_state_mean], index=[initial_dt])  # , name=self.asset
        self.state_covs = pd.Series([self.kf.initial_state_covariance], index=[initial_dt])  # , name=self.asset

    # def update(self, observations):
    #    for dt, observation in observations[self.asset].iterkv():
    #        self._update(dt, observation)

    def update(self,
               idx: Union[int, dt.datetime],
               observation: float,
               trans_cov: float = None):
        mu, cov = self.kf.filter_update(self.state_means.iloc[-1],
                                        self.state_covs.iloc[-1],
                                        observation,
                                        transition_covariance=trans_cov)  # observation_covariance=[obs_cov])
        # TODO: replace the initial value at def __init__ because we got signal coming in?

        self.state_means[idx] = mu[0][0]
        self.state_covs[idx] = cov[0][0]

        return self.state_means[idx], self.state_covs[idx]
