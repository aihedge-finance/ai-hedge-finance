import numpy as np
import matplotlib.pyplot as plt

"""
用這個來了解 UKF, 他裡面計算簡單
"""

class UnscentedKalmanFilter:
    def __init__(self, process_noise, measurement_noise, initial_state, initial_covariance):
        self.process_noise = process_noise  # Process noise covariance
        self.measurement_noise = measurement_noise  # Measurement noise covariance
        self.state = initial_state  # Initial state estimate
        self.covariance = initial_covariance  # Initial covariance estimate

        # UKF parameters
        self.alpha = 1e-3  # Spread of the sigma points
        self.beta = 2  # Optimal for Gaussian distributions
        self.kappa = 0  # Secondary scaling parameter
        self.n = 1  # State dimension

        # Calculate lambda
        self.lmbda = self.alpha ** 2 * (self.n + self.kappa) - self.n

        # Calculate weights
        self.weights_mean = np.zeros(2 * self.n + 1)
        self.weights_cov = np.zeros(2 * self.n + 1)
        self.weights_mean[0] = self.lmbda / (self.n + self.lmbda)
        self.weights_cov[0] = self.weights_mean[0] + (1 - self.alpha ** 2 + self.beta)

        for i in range(1, 2 * self.n + 1):
            self.weights_mean[i] = 1 / (2 * (self.n + self.lmbda))
            self.weights_cov[i] = self.weights_mean[i]

    def predict(self, dt):
        # Generate sigma points
        sigma_points = self._generate_sigma_points(self.state, self.covariance)

        # Predict state and covariance
        predicted_sigma_points = np.array([self._state_transition(sp, dt) for sp in sigma_points])
        self.state = np.dot(self.weights_mean, predicted_sigma_points)

        # Predict covariance
        self.covariance = self.process_noise + np.dot(self.weights_cov * (predicted_sigma_points - self.state).T,
                                                       (predicted_sigma_points - self.state))

    def update(self, measurement):
        # Generate sigma points
        sigma_points = self._generate_sigma_points(self.state, self.covariance)

        # Predict measurements
        predicted_measurements = np.array([self._measurement_function(sp) for sp in sigma_points])
        predicted_measurement_mean = np.dot(self.weights_mean, predicted_measurements)

        # Calculate measurement covariance
        measurement_covariance = self.measurement_noise + np.dot(self.weights_cov * (predicted_measurements - predicted_measurement_mean).T,
                                                                   (predicted_measurements - predicted_measurement_mean))

        # Calculate cross covariance
        cross_covariance = np.zeros((self.n, self.n))
        for i in range(2 * self.n + 1):
            diff_state = sigma_points[i] - self.state
            diff_measurement = predicted_measurements[i] - predicted_measurement_mean
            cross_covariance += self.weights_cov[i] * np.outer(diff_state, diff_measurement)

        # Calculate Kalman gain
        kalman_gain = np.dot(cross_covariance, np.linalg.inv(measurement_covariance))

        # Update state and covariance
        self.state += np.dot(kalman_gain, (measurement - predicted_measurement_mean))
        self.covariance -= np.dot(kalman_gain, measurement_covariance).dot(kalman_gain.T)

    def _generate_sigma_points(self, state, covariance):
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        sigma_points[0] = state
        sqrt_cov = np.linalg.cholesky((self.n + self.lmbda) * covariance)

        for i in range(self.n):
            sigma_points[i + 1] = state + sqrt_cov[i]
            sigma_points[i + 1 + self.n] = state - sqrt_cov[i]

        return sigma_points

    def _state_transition(self, state, dt):
        # Simple motion model: state remains the same
        return state

    def _measurement_function(self, state):
        # Measurement function: we only measure the position
        return state

# Example usage
if __name__ == "__main__":
    # Initial parameters
    process_noise = np.array([[0.001]])  # Process noise covariance
    measurement_noise = np.array([[0.001]])  # Measurement noise covariance
    initial_state = np.array([0.])  # Initial state
    initial_covariance = np.array([[0.1]])  # Initial covariance

    # Create UKF instance
    ukf = UnscentedKalmanFilter(process_noise, measurement_noise, initial_state, initial_covariance)

    # Simulate particle movement in a sine wave and measurements
    num_steps = 150
    true_positions = []
    measurements = []
    estimates = []

    for t in range(num_steps):
        # True position following a sine wave
        true_position = np.sin(t * 0.2)  # Sine wave with frequency 0.2
        true_positions.append(true_position)

        # Simulate measurement with noise
        measurement = true_position + np.random.normal(-0.2, 0.2)  # Add measurement noise
        measurements.append(measurement)

        # Predict and update UKF
        ukf.predict(dt=1)  # Assume dt=1 for simplicity
        ukf.update(measurement)

        estimates.append(ukf.state[0])

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(true_positions, label='True Position (Sine Wave)', linestyle='--', color='g')
    plt.scatter(range(num_steps), measurements, label='Measurements (Noisy)', color='r', marker='x')
    plt.plot(estimates, label='Estimated Position (UKF)', color='b')
    plt.title('Unscented Kalman Filter for 1D Sine Wave Tracking')
    plt.xlabel('Time Step')
    plt.ylabel('Position')
    plt.legend()
    plt.grid()
    plt.show()
