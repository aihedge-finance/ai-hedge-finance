import numpy as np
from collections import deque
from typing import Tuple, List, Optional, Union, Deque
from numpy.typing import NDArray
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints as MSSP
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


"""
這個完整的顯示了基本的使用
可以用這個再去加入想要的邏輯
比如
3 個 measurement 的話
_initial_state_covariance = np.diag([0.001, 0.001, 0.001])  # covariance matrix
_process_noise_covariance = np.diag([0.06, 0.06, 0.06])  # process noise
_observation_covariance = np.diag([0.001, 0.001, 0.001])  # sensor noise
"""

class RollingVolatility:
    def __init__(self, window_size: int):
        self.window: Deque = deque(maxlen=window_size)

    def update(self, log_return):
        self.window.append(log_return)
        if len(self.window) < 2:
            return 0  # Not enough data to compute volatility
        return np.std(list(self.window))  # Rolling standard deviation


def compute_thresholds(volatility_series: List[float], k_low: float=0.5, k_high: float=1.5):
    """
    Compute Adaptive Thresholds (low_var, high_var)

    Use the rolling statistics (mean and standard deviation) of volatility to compute thresholds:
    Parameters
    ----------
    volatility_series: list of recent value e.g. [0.012, 0.018, 0.011, 0.019, 0.022]
    k_low: std scaling factors
    k_high: std scaling factors

    # Example rolling volatility series (e.g., from past 30 days)
    recent_volatility = [0.012, 0.018, 0.011, 0.019, 0.022]  # Some dummy volatilities
    low_var, high_var = compute_thresholds(recent_volatility)
    print(f"Low: {low_var}, High: {high_var}")


    """
    mean_vol = np.mean(volatility_series)
    std_vol = np.std(volatility_series)
    low_var = mean_vol - k_low * std_vol
    high_var = mean_vol + k_high * std_vol
    return low_var, high_var


def recalibrate_thresholds(long_term_data: List[float], k_low: float = 0.5, k_high: float = 1.5):
    """
    Periodically Recalibrate With Long-Term Data

    At regular intervals (e.g., every 6 months):

    Use a broader dataset (e.g., 2 years of past volatility data).
    Recompute thresholds globally based on this longer-term data.
    Smoothly transition to the recalibrated values by interpolating them over a short period to avoid sudden cutoff effects.

    Parameters
    ----------
    long_term_data
    k_low
    k_high

    # Example: Recalibration every 6 months using 2 years of data
    long_term_volatility = [0.01, 0.015, 0.02, 0.018, 0.014, 0.013]  # Dummy 2-year volatilities
    recal_low_var, recal_high_var = compute_thresholds(long_term_volatility)
    print(f"Recalibrated Low: {recal_low_var}, Recalibrated High: {recal_high_var}")

    """
    mean_vol = np.mean(long_term_data)
    std_vol = np.std(long_term_data)
    low_var = mean_vol - k_low * std_vol
    high_var = mean_vol + k_high * std_vol
    return low_var, high_var


class HybridVolatility:
    def __init__(self, rolling_window_size: int, recalibration_period: int, long_term_window_size: int):
        self.recalibrated_high_var = None
        self.recalibrated_low_var = None
        self.rolling_vol = RollingVolatility(window_size=rolling_window_size)
        self.recalibration_period = recalibration_period  # Frequency of recalibration (e.g., days)
        self.long_term_window = deque(maxlen=long_term_window_size)
        self.current_low_var = None
        self.current_high_var = None
        self.days_since_recalibration = 0

    def update(self, log_return: float):
        # Update rolling volatility
        rolling_volatility = self.rolling_vol.update(log_return)
        if rolling_volatility > 0:
            self.long_term_window.append(rolling_volatility)
        self.days_since_recalibration += 1

        # Periodic recalibration
        if self.days_since_recalibration >= self.recalibration_period:
            self.recalibrate()
            self.days_since_recalibration = 0

        # Set current thresholds (multi-source hybrid)
        if rolling_volatility > 0:
            short_term_vol_series = list(self.rolling_vol.window)
            low_r, high_r = compute_thresholds(short_term_vol_series)
            # Blend recalibrated values (e.g., weighted 70/30 or immediate switch)
            self.current_low_var = 0.7 * low_r + 0.3 * (self.recalibrated_low_var or low_r)
            self.current_high_var = 0.7 * high_r + 0.3 * (self.recalibrated_high_var or high_r)

        return rolling_volatility, self.current_low_var, self.current_high_var

    def recalibrate(self):
        # Recalibration using long-term volatility
        if len(self.long_term_window) > 1:
            self.recalibrated_low_var, self.recalibrated_high_var = recalibrate_thresholds(list(self.long_term_window))


class UKF_Polynomial:
    INIT_STATE_MEAN = np.array([0.0, 0.0, 0.0])   # pos, vel, acc
    INIT_STATE_COV = np.diag([0.001, 0.001, 0.001])  # covariance matrix
    PROC_NOISE_COV = np.diag([0.01, 0.1, 0.1])  # process noise
    OBV_COV = np.array([[0.0]])  # sensor noise

    def __init__(
            self,
            dt: float,
            observation_covariance: NDArray[np.float64] = OBV_COV,
            initial_state_mean: NDArray[np.float64] = INIT_STATE_MEAN, # pos, vel, acc
            initial_state_covariance: NDArray[np.float64] = INIT_STATE_COV,
            process_noise_covariance: NDArray[np.float64] = PROC_NOISE_COV,
            alpha: float = 1e-3,
            beta: float = 2.0,
            kappa: float = 0.0,
            high_var: float = 0.0,
            low_var: float = 0.0,
            rolling_window_size: int = 30,
            recalibration_period: int = 180,
            long_term_window_size: int = 730
    ) -> None:
        """
        Initialize the UKF polynomial tracker.

        Args:
            dt: Time step size
            alpha: Spread of sigma points around mean. Usually small (e.g., 1e-3)
            beta: Prior knowledge of state distribution (2 for Gaussian)
            kappa: Secondary scaling parameter (usually 0)
            observation_covariance: Measurement noise covariance matrix
            initial_state_mean: Initial state vector [position, velocity, acceleration]
            initial_state_covariance: Initial state covariance matrix
            process_noise_covariance: Process noise covariance matrix
            high_var: value when acceleration is high
            low_var: value when acceleration is low
            rolling_window_size: to smooth out state estimates with last few values
            recalibration_period: to recalibrate var when this number is reached
            long_term_window_size: long term var window
        """

        # type checking
        if initial_state_mean is not None:
            if not isinstance(initial_state_mean, np.ndarray) or initial_state_mean.dtype != np.float64:
                raise TypeError(
                    f"initial_state_mean must be of type NDArray[np.float64], "
                    f"but got {type(initial_state_mean)} with dtype {getattr(initial_state_mean, 'dtype', None)}"
                )

        if initial_state_covariance is not None:
            if not isinstance(initial_state_covariance, np.ndarray) or initial_state_covariance.dtype != np.float64:
                raise TypeError(
                    f"initial_state_covariance must be of type NDArray[np.float64], "
                    f"but got {type(initial_state_covariance)} with dtype {getattr(initial_state_covariance, 'dtype', None)}"
                )

        if process_noise_covariance is not None:
            if not isinstance(process_noise_covariance, np.ndarray) or process_noise_covariance.dtype != np.float64:
                raise TypeError(
                    f"process_noise_covariance must be of type NDArray[np.float64], "
                    f"but got {type(process_noise_covariance)} with dtype {getattr(process_noise_covariance, 'dtype', None)}"
                )

        self.sigma_points = MSSP(n=3, alpha=alpha, beta=beta, kappa=kappa)  # For 3 state variables
        self.dt = dt

        self.ukf = UKF(
            dim_x=3, dim_z=1, dt=dt,
            hx=self._measurement_function,
            fx=self._polynomial_motion_model,
            points=self.sigma_points
        )

        self.ukf.x = initial_state_mean
        self.ukf.P = initial_state_covariance
        self.ukf.R = observation_covariance
        self.ukf.Q = process_noise_covariance

        self.state_means: List[float] = []
        self.state_velocities: List[float] = []
        self.state_accelerations: List[float] = []

        # dynamically adjusting alpha
        # Initialize hybrid volatility tracker
        self.hybrid_vol_tracker = HybridVolatility(rolling_window_size=rolling_window_size,
                                                   recalibration_period=recalibration_period,  # Recalibrate every 6 months
                                                   long_term_window_size=long_term_window_size)  # Store 2 years of volatility

        self.high_var = high_var
        self.low_var = low_var

    @staticmethod
    def _measurement_function(x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Measurement function that extracts position from state vector.

        Args:
            x: State vector [position, velocity, acceleration]

        Returns:
            Position measurement
        """
        return np.array([x[0]])

    @staticmethod
    def _polynomial_motion_model(
            x: NDArray[np.float64],
            dt: float
    ) -> NDArray[np.float64]:
        """
        Second-order polynomial motion model.

        Args:
            x: State vector [position, velocity, acceleration]
            dt: Time step

        Returns:
            Updated state vector
        """
        pos, vel, acc = x
        new_pos = pos + vel * dt + 0.5 * acc * dt**2
        new_vel = vel + acc * dt
        new_acc = acc  # Assume constant acceleration
        return np.array([new_pos, new_vel, new_acc])

    def update(
            self,
            dt: float,
            observation: NDArray[np.float64],
            process_noise_cov: NDArray[np.float64] = None
    ) -> Tuple[float, float, float, float]:
        """
        Update the filter with a new measurement.

        Args:
            dt: time step, e.g. 8.0
            observation: New measurement value, numpy.array of shape (dim_z)
            process_noise_cov: new dynamic process_noise_cov

        Returns:
            Tuple of (estimated_position, position_covariance, estimated_velocity, estimated_acceleration)

        """
        if process_noise_cov is not None:
            self.ukf.Q = np.diag(process_noise_cov)

        self.ukf.predict(dt)

        # getting rolling volatility
        rolling_volatility, self.low_var, self.high_var = self.hybrid_vol_tracker.update(abs(self.ukf.x.copy()))

        # Adjust alpha dynamically based on thresholds
        if rolling_volatility > 0:
            if rolling_volatility > self.high_var:
                self.ukf.alpha = 1e-2  # High variance, less adaptive
            elif rolling_volatility < self.low_var:
                self.ukf.alpha = 1e-3  # Low variance, more adaptive
            else:
                # Scale alpha based on rolling_volatility in the medium range
                ratio = (rolling_volatility - self.low_var) / (self.high_var - self.low_var)
                self.ukf.alpha = 1e-3 + ratio * (1e-2 - 1e-3)

        # update
        self.ukf.update(observation)

        self.state_means.append(self.ukf.x[0])
        self.state_velocities.append(self.ukf.x[1])
        self.state_accelerations.append(self.ukf.x[2])

        return self.ukf.x[0], self.ukf.P[0][0], self.ukf.x[1], self.ukf.x[2]




def generate_complex_signal(t):
    # Base signal
    signal = 2 * np.sin(2 * np.pi * 0.2 * t) + np.cos(2 * np.pi * 0.5 * t)

    # Add sudden jumps
    jumps = np.zeros_like(t)
    jump_points = [25, 50, 75]
    for jp in jump_points:
        jumps[jp:] += 1.5

    # Add exponential decay
    decay = 0.5 * np.exp(-0.2 * t)

    # Add polynomial trend
    trend = 0.01 * t ** 2 - 0.1 * t

    # Combine all components
    return signal + jumps + decay + trend


def main_complex():
    dt = 0.1
    proc_cov = np.diag([0.7, 1., 1.])
    ukf_poly = UKF_Polynomial(dt=dt,
                              process_noise_covariance=proc_cov)

    num_points = 200
    time = np.linspace(0, 20, num_points)

    # Generate complex true signal
    true_position = generate_complex_signal(time)

    # Calculate true velocity and acceleration using finite differences
    true_velocity = np.gradient(true_position, time)
    true_acceleration = np.gradient(true_velocity, time)

    # Add non-uniform noise
    base_noise = 0.2
    varying_noise = base_noise * (1 + 0.5 * np.sin(2 * np.pi * 0.1 * time))
    measurements = true_position + np.random.normal(0, varying_noise, size=num_points)

    estimated_position = []
    estimated_velocity = []
    estimated_acceleration = []

    for z in measurements:
        pos, _, vel, acc = ukf_poly.update(dt, np.array([z]))
        estimated_position.append(pos)
        estimated_velocity.append(vel)
        estimated_acceleration.append(acc)

    # Calculate RMSEs
    position_rmse = np.sqrt(np.mean((true_position - estimated_position) ** 2))
    velocity_rmse = np.sqrt(np.mean((true_velocity - estimated_velocity) ** 2))
    acceleration_rmse = np.sqrt(np.mean((true_acceleration - estimated_acceleration) ** 2))

    print(f"Position RMSE: {position_rmse:.4f}")
    print(f"Velocity RMSE: {velocity_rmse:.4f}")
    print(f"Acceleration RMSE: {acceleration_rmse:.4f}")

    # Enhanced visualization
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)

    # Position plot
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time, true_position, 'b-', label='True Position', alpha=0.7)
    ax1.plot(time, measurements, 'g.', label='Measurements', alpha=0.8, markersize=4)  # Magenta circles for measurements
    ax1.plot(time, estimated_position, 'r-', label='UKF Estimate', linewidth=2)
    ax1.set_title('Position Tracking')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Position')
    ax1.grid(True)
    ax1.legend()

    # Velocity plot
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(time, true_velocity, 'b-', label='True Velocity', alpha=0.7)
    ax2.plot(time, estimated_velocity, 'r-', label='UKF Estimate', linewidth=2)
    ax2.set_title('Velocity Tracking')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Velocity')
    ax2.grid(True)
    ax2.legend()

    # Acceleration plot
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(time, true_acceleration, 'b-', label='True Acceleration', alpha=0.7)
    ax3.plot(time, estimated_acceleration, 'r-', label='UKF Estimate', linewidth=2)
    ax3.set_title('Acceleration Tracking')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Acceleration')
    ax3.grid(True)
    ax3.legend()

    plt.suptitle('UKF State Estimation of Complex Motion', fontsize=16)
    plt.tight_layout()
    plt.show()



def main_simple():
    from scipy.signal import savgol_filter

    dt = 0.1
    ukf_poly = UKF_Polynomial(dt=dt)

    num_points = 100
    time = np.linspace(0, 10, num_points)

    # Generate a random sine + cosine curve
    amp1 = np.random.uniform(0.5, 1.5)  # Random amplitude
    freq1 = np.random.uniform(0.1, 0.3)  # Random frequency
    phase1 = np.random.uniform(0, 2 * np.pi)  # Random phase

    amp2 = np.random.uniform(0.5, 1.5)
    freq2 = np.random.uniform(0.1, 0.3)
    phase2 = np.random.uniform(0, 2 * np.pi)

    true_position = amp1 * np.sin(2 * np.pi * freq1 * time + phase1) + amp2 * np.cos(2 * np.pi * freq2 * time + phase2)
    true_velocity = 2 * np.pi * freq1 * amp1 * np.cos(
        2 * np.pi * freq1 * time + phase1) - 2 * np.pi * freq2 * amp2 * np.sin(2 * np.pi * freq2 * time + phase2)

    measurement_noise_std = 0.2
    measurements = true_position + np.random.normal(0, measurement_noise_std, size=num_points)

    # Savitzky-Golay Filtering for velocity estimation
    window_length = 11  # Adjust this (odd number)
    polyorder = 3  # Adjust this (less than window_length)
    velocity_sg = savgol_filter(measurements, window_length, polyorder, deriv=1, delta=dt)

    estimated_position = []
    estimated_velocity = []  # For the UKF velocity
    estimated_acceleration = []

    for z in measurements:
        pos, _, vel, acc = ukf_poly.update(dt, np.array([z]))
        estimated_position.append(pos)
        estimated_velocity.append(vel)
        estimated_acceleration.append(acc)

        # Calculate RMSE.  Use velocity_sg for the Savitzky-Golay velocity.
    position_rmse = np.sqrt(np.mean((np.array(true_position) - np.array(estimated_position)) ** 2))
    velocity_rmse_sg = np.sqrt(
        np.mean((np.array(true_velocity) - np.array(velocity_sg)) ** 2))  # RMSE for Savitzky-Golay
    velocity_rmse_ukf = np.sqrt(np.mean((np.array(true_velocity) - np.array(estimated_velocity)) ** 2))  # RMSE for UKF

    print(f"Position RMSE: {position_rmse}")
    print(f"Velocity RMSE (Savitzky-Golay): {velocity_rmse_sg}")
    print(f"Velocity RMSE (UKF): {velocity_rmse_ukf}")


    plt.figure(figsize=(10, 6))
    plt.plot(time, true_position, label='True Position', linestyle='--', color='blue')
    plt.plot(time, measurements, label='Measured Position (with noise)', linestyle=':', color='gray')
    plt.plot(time, estimated_position, label='Estimated Position (UKF)', color='red')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Position')
    plt.title('Position: True vs Measured vs Estimated')
    plt.grid()
    plt.show()

    # Plot Savitzky-Golay Velocity
    plt.figure(figsize=(10, 6))
    plt.plot(time, true_velocity, label='True Velocity', linestyle='--', color='blue')
    plt.plot(time, velocity_sg, label='Estimated Velocity (Savitzky-Golay)', color='green')
    plt.plot(time, estimated_velocity, label='Estimated Velocity (UKF)', color='red')  # UKF Velocity

    plt.xlabel('Time [s]')
    plt.ylabel('Velocity')
    plt.title('Velocity: True vs Estimated (Savitzky-Golay)')
    plt.legend()
    plt.grid()
    plt.show()


    plt.show()


if __name__ == "__main__":
    # main_simple()
    main_complex()


