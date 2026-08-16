import pandas as pd
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import JulierSigmaPoints
import matplotlib.pyplot as plt

initial_price = 1
relative_p = 0
relative_q = max(initial_price * 0.1, 1e-6)
relative_r = 0

class DualUKF_MA(object):
    def __init__(self, observation_covariance=relative_r,
                 initial_state_mean=initial_price,
                 initial_state_covariance=relative_p,
                 process_noise_covariance=relative_q,
                 kappa=0):
        # State UKF setup
        self.sigma_points = JulierSigmaPoints(n=1, kappa=kappa)
        self.ukf = UKF(dim_x=1, dim_z=1,
                       dt=1,
                       hx=lambda x: np.array([float(x[0])]),  # Modified measurement function
                       fx=lambda x, dt: np.array([float(x[0])]),  # Modified state transition
                       points=self.sigma_points)

        self.ukf.x = np.array([initial_state_mean])
        self.ukf.P = np.array([[max(initial_state_covariance, 1e-6)]])
        self.ukf.R = np.array([[max(observation_covariance, 1e-6)]])
        self.ukf.Q = np.array([[max(process_noise_covariance, 1e-6)]])

        # Adaptive UKF setup
        self.sigma_points_adaptive = JulierSigmaPoints(n=2, kappa=kappa)
        self.adaptive_ukf = UKF(dim_x=2, dim_z=1,
                                dt=1,
                                hx=lambda x: np.array([float(x[0])]),  # Modified measurement function
                                fx=lambda x, dt: np.array([float(x[0]), float(x[1])]),  # Modified state transition
                                points=self.sigma_points_adaptive)

        self.adaptive_ukf.x = np.array([self.ukf.R[0][0], self.ukf.Q[0][0]])
        self.adaptive_ukf.P = np.eye(2) * 1e-4
        self.adaptive_ukf.R = np.array([[1e-4]])
        self.adaptive_ukf.Q = np.eye(2) * 1e-4

        self.state_means = pd.Series(dtype=float)
        self.state_covs = pd.Series(dtype=float)

    def update(self, dt, observation):
        # Ensure positive definiteness
        self.ukf.P = np.maximum(self.ukf.P, np.eye(self.ukf.P.shape[0]) * 1e-6)
        self.adaptive_ukf.P = np.maximum(self.adaptive_ukf.P, np.eye(self.adaptive_ukf.P.shape[0]) * 1e-6)

        # Predict step
        self.ukf.predict()
        self.adaptive_ukf.predict()

        # Calculate innovation
        innovation = np.array([observation - float(self.ukf.x_prior[0])])

        # Update adaptive filter
        self.adaptive_ukf.update(innovation)

        # Update noise parameters
        self.ukf.R[0][0] = max(float(self.adaptive_ukf.x[0]), 1e-6)
        self.ukf.Q[0][0] = max(float(self.adaptive_ukf.x[1]), 1e-6)

        # Update state filter
        self.ukf.update(np.array([observation]))

        # Store results
        self.state_means[dt] = float(self.ukf.x[0])
        self.state_covs[dt] = float(self.ukf.P[0][0])

        return self.state_means[dt], self.state_covs[dt]


def main():
    dt = 0.1
    dual_ukf = DualUKF_MA()

    num_points = 100
    time = np.linspace(0, 10, num_points)

    # Generate synthetic data
    amp1 = np.random.uniform(0.5, 1.5)
    freq1 = np.random.uniform(0.1, 0.3)
    phase1 = np.random.uniform(0, 2 * np.pi)

    amp2 = np.random.uniform(0.5, 1.5)
    freq2 = np.random.uniform(0.1, 0.3)
    phase2 = np.random.uniform(0, 2 * np.pi)

    true_position = amp1 * np.sin(2 * np.pi * freq1 * time + phase1) + amp2 * np.cos(2 * np.pi * freq2 * time + phase2)
    true_velocity = 2 * np.pi * freq1 * amp1 * np.cos(
        2 * np.pi * freq1 * time + phase1) - 2 * np.pi * freq2 * amp2 * np.sin(2 * np.pi * freq2 * time + phase2)

    measurement_noise_std = 0.2
    measurements = true_position + np.random.normal(0, measurement_noise_std, size=num_points)

    estimated_position = []

    for i in range(num_points):
        pos, _ = dual_ukf.update(time[i], measurements[i])
        estimated_position.append(pos)

    # Calculate RMSE
    position_rmse = np.sqrt(np.mean((np.array(true_position) - np.array(estimated_position)) ** 2))
    print(f"Position RMSE (Dual UKF): {position_rmse}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(time, true_position, label='True Position', linestyle='--', color='blue')
    plt.plot(time, measurements, label='Measured Position (with noise)', linestyle=':', color='gray')
    plt.plot(time, estimated_position, label='Estimated Position (Dual UKF)', color='purple')

    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Position')
    plt.title('Position: True vs Measured vs Estimated')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
