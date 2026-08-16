# cython: language_level=3

import numpy as np
cimport numpy as cnp
ctypedef cnp.double_t DTYPE_t
import pandas as pd
import datetime as dt
from typing import Optional, Union
from libc.math cimport isnan

cdef class KalmanFilter:
    cdef public double[:] transition_matrices
    cdef public double[:] observation_matrices
    cdef public double initial_state_mean
    cdef public double initial_state_covariance
    cdef public double observation_covariance
    cdef public double transition_covariance

    def __init__(self, transition_matrices, observation_matrices, initial_state_mean,
                 initial_state_covariance, observation_covariance, transition_covariance):
        self.transition_matrices = np.array(transition_matrices, dtype=np.float64)
        self.observation_matrices = np.array(observation_matrices, dtype=np.float64)
        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance
        self.observation_covariance = observation_covariance
        self.transition_covariance = transition_covariance

    cdef filter_update(self, double predicted_state, double predicted_state_covariance, double observation, double custom_transition_covariance, bint use_custom_transition_covariance):
        cdef double kalman_gain, updated_state, updated_state_covariance
        cdef double actual_transition_covariance

        if use_custom_transition_covariance:
            actual_transition_covariance = custom_transition_covariance
        else:
            actual_transition_covariance = self.transition_covariance

        # Prediction
        predicted_state = self.transition_matrices[0] * predicted_state
        predicted_state_covariance = (self.transition_matrices[0] ** 2 * predicted_state_covariance) + actual_transition_covariance

        # Update
        kalman_gain = predicted_state_covariance * self.observation_matrices[0] / (
                (self.observation_matrices[0] ** 2 * predicted_state_covariance) + self.observation_covariance)

        updated_state = predicted_state + kalman_gain * (observation - self.observation_matrices[0] * predicted_state)
        updated_state_covariance = (1 - kalman_gain * self.observation_matrices[0]) * predicted_state_covariance

        return np.array([[updated_state]]), np.array([[updated_state_covariance]])


cdef class KalmanMovingAverage:
    cdef public KalmanFilter kf
    cdef public object state_means
    cdef public object state_covs

    def __init__(self, double observation_covariance=1.0,
                 double initial_state_mean=0,
                 double initial_state_covariance=1.0,
                 double transition_covariance=0.05,
                 initial_dt=0):

        self.kf = KalmanFilter(transition_matrices=[1],
                               observation_matrices=[1],
                               initial_state_mean=initial_state_mean,
                               initial_state_covariance=initial_state_covariance,
                               observation_covariance=observation_covariance,
                               transition_covariance=transition_covariance)

        self.state_means = pd.Series([self.kf.initial_state_mean], index=[initial_dt])
        self.state_covs = pd.Series([self.kf.initial_state_covariance], index=[initial_dt])

    cpdef update(self, idx, double observation, double trans_cov=float('nan')):
        cdef cnp.ndarray[double, ndim=2] mu, cov
        cdef bint use_custom_trans_cov = not isnan(trans_cov)

        mu, cov = self.kf.filter_update(self.state_means.iloc[-1],
                                        self.state_covs.iloc[-1],
                                        observation,
                                        trans_cov,
                                        use_custom_trans_cov)

        self.state_means[idx] = mu[0][0]
        self.state_covs[idx] = cov[0][0]

        return self.state_means[idx], self.state_covs[idx]

# Example of how to inherit from KalmanMovingAverage
"""cdef class CustomKalmanMovingAverage(KalmanMovingAverage):

    cdef public double custom_attribute

    def __init__(self, double custom_attribute, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_attribute = custom_attribute

    cpdef custom_method(self):
        print(f"Custom attribute: {self.custom_attribute}")
"""
