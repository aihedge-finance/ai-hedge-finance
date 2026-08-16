import sys
import pandas as pd
import numpy as np

from typing import Tuple, List, Optional, Union
from numpy.typing import NDArray
from .UKF_Polynomial import UKF_Polynomial
from ahf.utils.utils import readable_error, datefmt, create_dir_if_non_exist

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


"""
3 個 measurement 的話
_initial_state_covariance = np.diag([0.001, 0.001, 0.001])  # covariance matrix
_process_noise_covariance = np.diag([0.06, 0.06, 0.06])  # process noise
_observation_covariance = np.diag([0.001, 0.001, 0.001])  # sensor noise
"""

class TracerSimple:

    def __init__(
            self,
            strategy: str,
            job: str,
            dt: float,
            proc_cov: float,
            logger,
            initial_state_mean: float = None,
            verbose=True
    ) -> None:
        """
        Initialize the UKF polynomial tracker.

        Args:
            strategy: strategy name
            job: job_name
            dt: Time step size
            proc_cov: process_noise_covariance 但是只設定 pos
            logger: logger
            initial_state_mean: initial_state_mean 但是只設定 pos
            verbose: for printing

        """

        self.strategy = strategy
        self.job = job

        # most important controlling param
        self.proc_cov = proc_cov


        # unscented Kalman filter
        self.ukf: UKF_Polynomial = UKF_Polynomial(dt=dt,
                                                  initial_state_mean=np.array([initial_state_mean, 0.0, 0.0], dtype=np.float64),
                                                  process_noise_covariance=np.array([proc_cov, proc_cov * 10, proc_cov * 10], dtype=np.float64))

        self.idx_now = 0
        self.verbose = verbose
        self.logger = logger


    def update(
            self,
            dt: float,
            observation: float,
            process_noise_cov: np.ndarray = None
    ) -> Tuple[float, float, float, float]:
        """
        Update the filter with a new measurement.

        Args:
            dt: time step
            observation: New measurement value
            process_noise_cov: new dynamic process_noise_cov
                e.g. [0.01, 0.1, 0.1] pos, velocity, acceleration

        Returns:
            Tuple of (estimated_position, position_covariance, estimated_velocity, estimated_acceleration)


        """
        # Update process noise covariance if provided
        if process_noise_cov is not None:
            self.ukf.Q = np.diag(process_noise_cov)

        self.idx_now += 1

        return self.ukf.update(dt, np.array([observation]))  # dim_z is 1, only can observe position


    def _save_tracer(self, file_dir, idx_run_np, dt_idx, price_np, tracer, slope, proc_cov, state_covs,
                     need_header=False):
        """
        result from parent update()=tracer_new

        :param file_dir: usage => f"{self.trade_args['data_path']}/KF_theta/{self.name}_{self.theta_id}.csv"
        :param idx_run_np:
        :param dt_idx:
        :param price_np:
        :param tracer:
        :param proc_cov:
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
                    proc_cov,
                    state_covs00
                ])
            r = pd.DataFrame(data, index=idx_run_np,
                             columns=['dt_idx', 'price', 'tracer', 'slope', 'proc_cov', 'state_cov00'])

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
    ukf_poly = TracerSimple(dt=dt)

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
        pos, _, vel, acc = ukf_poly.update(np.array([z]))
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
    ax1.plot(time, measurements, 'mo', label='Measurements', alpha=0.8, markersize=12)  # Magenta circles for measurements
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
    ukf_poly = TracerSimple(dt=dt)

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
        pos, _, vel, acc = ukf_poly.update(np.array([z]))
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
    plt.plot(time, measurements, 'mo', label='Measurements', alpha=0.8, markersize=12)  # Magenta circles for measurements
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


