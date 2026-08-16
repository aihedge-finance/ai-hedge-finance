from functools import partial
import jax.numpy as jnp
from jax import jit
from typing import Tuple, NamedTuple


class UKFState(NamedTuple):
    """State of the UKF filter that needs to be maintained between calls."""
    mean: jnp.ndarray
    covariance: jnp.ndarray


class JAXUKF_MA:
    """
    Immutable implementation of Unscented Kalman Filter (UKF) using JAX for motion analysis.
    Designed for external sequential calling of predict and update methods.
    """

    def __init__(self,
                 dim_state: int = 3,
                 observation_covariance: jnp.ndarray = None,
                 process_noise_covariance: jnp.ndarray = None,
                 alpha: float = 0.1,
                 beta: float = 2.0,
                 kappa: float = 0.0):
        """
        Initialize the UKF object with immutable parameters.

        Args:
            dim_state: Dimension of state vector (default 3 for position, velocity, acceleration)
            observation_covariance: Sensor noise covariance matrix (R)
            process_noise_covariance: Process noise covariance matrix (Q)
            alpha: Scaling factor for sigma points (usually small, e.g., 1e-3)
            beta: Scaling factor for prior knowledge of distribution (2.0 for Gaussian)
            kappa: Secondary scaling factor for sigma points (usually 0 or 3-n for n states)
        """
        self.dim_state = dim_state

        # Default covariance matrices if none provided
        if observation_covariance is None:
            observation_covariance = jnp.eye(dim_state) * 0.001
        if process_noise_covariance is None:
            process_noise_covariance = jnp.eye(dim_state) * 0.06

        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.R = observation_covariance
        self.Q = process_noise_covariance

        # Pre-compute lambda
        self.lambd = self.alpha ** 2 * (self.dim_state + self.kappa) - self.dim_state

    def create_initial_state(self,
                             mean: jnp.ndarray = None,
                             covariance: jnp.ndarray = None
                             ) -> UKFState:
        """
        Create initial UKF state.

        Args:
            mean: Initial state mean. If None, zeros will be used.
            covariance: Initial state covariance. If None, identity matrix will be used.

        Returns:
            UKFState object containing initial mean and covariance
        """
        if mean is None:
            mean = jnp.zeros(self.dim_state)
        if covariance is None:
            covariance = jnp.eye(self.dim_state) * 0.001

        return UKFState(mean=mean, covariance=covariance)

    @staticmethod
    @jit
    def fx(state: jnp.ndarray, dt: float) -> jnp.ndarray:
        """
        State transition function for constant acceleration model.

        Args:
            state: Current state [position, velocity, acceleration]
            dt: Time step

        Returns:
            Predicted next state
        """
        return jnp.array([
            state[0] + state[1] * dt + 0.5 * state[2] * (dt ** 2),  # Position
            state[1] + state[2] * dt,  # Velocity
            state[2]  # Acceleration
        ])

    @staticmethod
    @jit
    def hx(state: jnp.ndarray) -> jnp.ndarray:
        """
        Measurement function. Assumes only position is observed.

        Args:
            state: Current state [position, velocity, acceleration]

        Returns:
            Predicted measurement (position only)
        """
        return state[:1]  # Only return position

    @partial(jit, static_argnames=["self"])
    def compute_sigma_points(self, mean: jnp.ndarray, covariance: jnp.ndarray) -> Tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Compute sigma points and weights using the scaled unscented transformation.

        Args:
            mean: Current state mean
            covariance: Current state covariance

        Returns:
            Tuple of (sigma_points, mean_weights, covariance_weights)
        """
        n = self.dim_state
        scaled_covariance = (n + self.lambd) * covariance

        # Add small value to diagonal for numerical stability
        scaled_covariance = scaled_covariance + jnp.eye(n) * 1e-8

        # Compute square root using Cholesky decomposition
        sqrt_covariance = jnp.linalg.cholesky(scaled_covariance)

        # Create sigma points
        sigma_points = jnp.vstack([
            mean,
            jnp.array([mean + sqrt_covariance[:, i] for i in range(n)]),
            jnp.array([mean - sqrt_covariance[:, i] for i in range(n)])
        ])

        # Compute weights
        weights_mean = jnp.full(2 * n + 1, 1.0 / (2 * (n + self.lambd)))
        weights_mean = weights_mean.at[0].set(self.lambd / (n + self.lambd))

        weights_cov = weights_mean.copy()
        weights_cov = weights_cov.at[0].set(
            self.lambd / (n + self.lambd) + (1.0 - self.alpha ** 2 + self.beta)
        )

        return sigma_points, weights_mean, weights_cov

    @partial(jit, static_argnames=["self"])
    def predict(self,
                state_mean: jnp.ndarray,
                state_cov: jnp.ndarray,
                dt: float) -> UKFState:
        """
        Prediction step of the UKF.

        Args:
            state: Current UKFState (mean and covariance)
            dt: Time step

        Returns:
            New UKFState containing predicted mean and covariance
        """
        sigma_points, weights_mean, weights_cov = self.compute_sigma_points(state_mean, state_cov)

        # Propagate sigma points through state transition function
        sigma_points_pred = jnp.array([self.fx(sigma, dt) for sigma in sigma_points])

        # Compute predicted mean and covariance
        predicted_mean = jnp.sum(weights_mean[:, None] * sigma_points_pred, axis=0)

        predicted_cov = jnp.sum(
            weights_cov[:, None, None] *
            (sigma_points_pred - predicted_mean[None, :])[:, :, None] *
            (sigma_points_pred - predicted_mean[None, :])[:, None, :],
            axis=0
        ) + self.Q

        return UKFState(mean=predicted_mean, covariance=predicted_cov)

    @partial(jit, static_argnames=["self"])
    def update(self, state: UKFState, measurement: jnp.ndarray) -> UKFState:
        """
        Update step of the UKF.

        Args:
            state: Current UKFState (mean and covariance)
            measurement: Current measurement

        Returns:
            New UKFState containing updated mean and covariance
        """
        sigma_points, weights_mean, weights_cov = self.compute_sigma_points(state.mean, state.covariance)

        # Propagate sigma points through measurement function
        sigma_points_meas = jnp.array([self.hx(sigma) for sigma in sigma_points])

        # Predicted measurement mean
        meas_mean = jnp.sum(weights_mean[:, None] * sigma_points_meas, axis=0)

        # Measurement covariance
        S = jnp.sum(
            weights_cov[:, None, None] *
            (sigma_points_meas - meas_mean[None, :])[:, :, None] *
            (sigma_points_meas - meas_mean[None, :])[:, None, :],
            axis=0
        ) + self.R

        # Cross covariance
        Pxz = jnp.sum(
            weights_cov[:, None, None] *
            (sigma_points - state.mean[None, :])[:, :, None] *
            (sigma_points_meas - meas_mean[None, :])[:, None, :],
            axis=0
        )

        # Ensure numerical stability
        S = S + jnp.eye(S.shape[0]) * 1e-8

        # Kalman gain
        K = jnp.linalg.solve(S, Pxz.T).T

        # Update state mean and covariance
        updated_mean = state.mean + K @ (measurement - meas_mean)
        updated_cov = state.covariance - K @ S @ K.T

        return UKFState(mean=updated_mean, covariance=updated_cov)


# Example usage
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from time import time

    # Simulation parameters
    dt = 0.1  # Time step
    t_max = 10.0  # Total simulation time
    t = np.arange(0, t_max, dt)
    n_steps = len(t)

    # Generate true signal (sinusoidal motion)
    frequency = 0.5  # Hz
    amplitude = 2.0
    true_position = amplitude * np.sin(2 * np.pi * frequency * t)
    true_velocity = 2 * np.pi * frequency * amplitude * np.cos(2 * np.pi * frequency * t)
    true_acceleration = -(2 * np.pi * frequency) ** 2 * amplitude * np.sin(2 * np.pi * frequency * t)

    # Add noise to create measurements
    measurement_std = 0.1
    noisy_measurements = true_position + np.random.normal(0, measurement_std, n_steps)

    # Initialize UKF
    ukf = JAXUKF_MA(
        dim_state=3,
        observation_covariance=jnp.eye(3) * 0.01,  # R matrix
        process_noise_covariance=jnp.eye(3) * 0.01,  # Q matrix
        alpha=0.1,
        beta=2.0,
        kappa=0.0
    )

    # Initialize state
    ukf_state = ukf.create_initial_state(
        mean=jnp.array([0.0, 0.0, 0.0]),
        covariance=jnp.eye(3) * 0.1
    )

    # Storage for results
    filtered_states = np.zeros((n_steps, 3))

    # Run UKF
    start_time = time()
    for i in range(n_steps):
        # Create measurement (position, velocity, acceleration)
        measurement = jnp.array([
            noisy_measurements[i],
            0.0,  # We don't measure velocity directly
            0.0  # We don't measure acceleration directly
        ])

        # Predict
        ukf_state = ukf.predict(ukf_state.mean, ukf_state.covariance, dt)

        # Update
        ukf_state = ukf.update(ukf_state, measurement)

        # Store results
        filtered_states[i] = np.array(ukf_state.mean)

    end_time = time()
    print(f"Processing time: {end_time - start_time:.3f} seconds")

    # Plotting
    plt.figure(figsize=(15, 10))

    # Position plot
    plt.subplot(3, 1, 1)
    plt.plot(t, true_position, 'g-', label='True Position')
    plt.plot(t, noisy_measurements, 'r.', alpha=0.5, label='Noisy Measurements')
    plt.plot(t, filtered_states[:, 0], 'b-', label='UKF Estimated Position')
    plt.ylabel('Position')
    plt.legend()
    plt.grid(True)

    # Velocity plot
    plt.subplot(3, 1, 2)
    plt.plot(t, true_velocity, 'g-', label='True Velocity')
    plt.plot(t, filtered_states[:, 1], 'b-', label='UKF Estimated Velocity')
    plt.ylabel('Velocity')
    plt.legend()
    plt.grid(True)

    # Acceleration plot
    plt.subplot(3, 1, 3)
    plt.plot(t, true_acceleration, 'g-', label='True Acceleration')
    plt.plot(t, filtered_states[:, 2], 'b-', label='UKF Estimated Acceleration')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Calculate and print RMS errors
    position_rmse = np.sqrt(np.mean((filtered_states[:, 0] - true_position) ** 2))
    velocity_rmse = np.sqrt(np.mean((filtered_states[:, 1] - true_velocity) ** 2))
    acceleration_rmse = np.sqrt(np.mean((filtered_states[:, 2] - true_acceleration) ** 2))

    print(f"RMS Errors: ")
    print(f"Position: {position_rmse:.4f}")
    print(f"Velocity: {velocity_rmse:.4f}")
    print(f"Acceleration: {acceleration_rmse:.4f}")
