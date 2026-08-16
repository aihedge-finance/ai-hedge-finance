import pandas as pd
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints as MSSP

# Set P, Q, and R relative to the initial price
initial_price = 0.065
relative_p = initial_price * 1e-1  # covariance matrix
relative_q = initial_price * 0.25  # process noise, 25%
relative_r = initial_price * 0.002  # sensor noise, ～千分之2


class AdaptiveUKF_MA:
    """
    An adaptive Unscented Kalman Filter moving average estimation using the
    Kalman-and-Bayesian-Filters-in-Python library.
    """

    def __init__(self, observation_covariance=relative_r,
                 initial_state_mean=initial_price,
                 initial_state_covariance=relative_p,
                 process_noise_covariance=relative_q,
                 alpha=0.001, beta=2.0, kappa=0):
        self.sigma_points = MSSP(n=1, alpha=alpha, beta=beta, kappa=kappa)

        self.ukf = UKF(dim_x=1, dim_z=1,
                       dt=1,
                       hx=lambda x: x,
                       fx=lambda x, dt: x,
                       points=self.sigma_points)

        self.ukf.x = np.array([initial_state_mean])
        self.ukf.P = np.array([[initial_state_covariance]])
        self.ukf.R = np.array([[observation_covariance]])
        self.ukf.Q = np.array([[process_noise_covariance]])

        self.state_means = pd.Series()
        self.state_covs = pd.Series()

        self.adaptive_factor = initial_price * 1e-4

    def update(self, dt, observation):
        self.ukf.predict()
        innovation = observation - self.ukf.x_prior[0]

        # Adaptive update of R and Q
        self.ukf.R *= 1 + np.sign(innovation) * self.adaptive_factor
        self.ukf.Q *= 1 + np.sign(innovation) * self.adaptive_factor

        self.ukf.update(np.array([observation]))

        self.state_means[dt] = self.ukf.x[0]
        self.state_covs[dt] = self.ukf.P[0][0]

        return self.state_means[dt], self.state_covs[dt]