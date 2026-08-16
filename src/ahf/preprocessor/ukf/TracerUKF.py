import sys
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints as MSSP

from ahf.utils.utils import readable_error, create_dir_if_non_exist, read_data, pct_change, setup_logger, price_columns

from collections import namedtuple
import jax.numpy as jnp
import numpy as np


def polynomial_motion_model(x: np.array, dt: float):
    """Second-order polynomial motion model."""
    pos = x[0]
    vel = x[1]
    acc = x[2]  # Now include acceleration in the state
    new_pos = pos + vel * dt + 0.5 * acc * dt**2  # Polynomial model
    new_vel = vel + acc * dt
    new_acc = acc  # Assume constant acceleration (or add a noise term)
    return np.array([new_pos, new_vel, new_acc])


TracerState = namedtuple("TracerState",
                         ["idx",
                          "dt_idx",
                          "price_pct",
                          "tracer",
                          "alfa",
                          "ukf_state",
                          "delta",
                          "state_cov"])


class TracerUKF:
    def __init__(self,
                 strategy: str,
                 job: str,
                 delta: float,
                 obs_cov: float,
                 logger: Optional[object],
                 alpha: float = 1e-3,
                 beta: float = 2,
                 kappa: float = 0,
                 initial_state_mean: Optional[jnp.ndarray] = None,
                 initial_state_cov: Optional[jnp.ndarray] = None,
                 verbose: bool = True,
                 gc: bool = False):
        """
        UKF-based Tracer with Extended State Vector.

        Args:
            strategy: Strategy name
            job: Job ID
            delta: Process noise parameter
            obs_cov: Observation noise parameter
            logger: Logger instance
            alpha: UKF alpha parameter
            beta: UKF beta parameter
            kappa: UKF kappa parameter
            initial_state_mean: Initial state mean vector [price_pct, tracer, alfa]
            initial_state_cov: Initial state covariance matrix
            verbose: Enable verbose logging
            gc: Enable garbage collection
        """
        self.strategy = strategy
        self.job = job
        self.delta = delta
        self.obs_cov = obs_cov

        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        self.logger = logger
        self.verbose = verbose
        self.gc = gc  # Enable or disable garbage collection for old data

        # Store initial state parameters
        self.initial_state_mean = initial_state_mean or jnp.array([0.0, 0.0, 0.0])
        self.initial_state_cov = initial_state_cov  or jnp.array([[1.0, 0.0, 0.0],
                                                                 [0.0, 1.0, 0.0],
                                                                 [0.0, 0.0, 1.0]])
        # Initialize UKF
        self.ukf = JAXUKF_MA(
            dim_state=3,
            observation_covariance=jnp.diag(jnp.array([self.obs_cov] * 3)),
            process_noise_covariance=jnp.diag(jnp.array([self.delta] * 3)),
            alpha=alpha,
            beta=beta,
            kappa=kappa
        )

    def create_initial_state(self, idx: int, dt_idx: float, init_value: float = 0.0) -> TracerState:
        """
        Create initial TracerState.

        Args:
            idx: Initial index
            dt_idx: Initial time index
            init_value: Initial position of the data you want to trace

        Returns:
            Initial TracerState

        """
        if init_value > 0:
            self.initial_state_mean = self.initial_state_mean.at[0].set(init_value)

        # Create UKF initial state
        ukf_state = self.ukf.create_initial_state(
            mean=self.initial_state_mean,
            covariance=self.initial_state_cov
        )

        # Convert to TracerState
        return TracerState(
            idx=idx,
            dt_idx=dt_idx,
            price_pct=ukf_state.mean[0],
            tracer=ukf_state.mean[1],
            alfa=ukf_state.mean[2],
            ukf_state=ukf_state,
            delta=self.delta,
            state_cov=ukf_state.covariance
        )

    def _tracer_to_ukf_state(self, tracer_state: TracerState) -> UKFState:
        """
        Convert TracerState to UKFState.

        Args:
            tracer_state: Current TracerState

        Returns:
            Equivalent UKFState
        """
        return UKFState(
            mean=jnp.array([tracer_state.price_pct, tracer_state.tracer, tracer_state.alfa]),
            covariance=tracer_state.state_cov
        )

    def _ukf_to_tracer_state(self,
                             ukf_state: UKFState,
                             prev_state: TracerState,
                             dt: float) -> TracerState:
        """
        Convert UKFState to TracerState.

        Args:
            ukf_state: Current UKFState
            prev_state: Previous TracerState
            dt: Time step

        Returns:
            Updated TracerState
        """
        return TracerState(
            idx=prev_state.idx + 1,
            dt_idx=prev_state.dt_idx + dt,
            price_pct=ukf_state.mean[0],
            tracer=tracer,
            alfa=alfa,
            ukf_state=ukf_state,
            delta=self.delta,
            state_cov=ukf_state.covariance
        )

    def _predict(self, current_state: TracerState, dt: float) -> TracerState:
        """
        Perform prediction step.

        Args:
            current_state: Current TracerState
            dt: Time step

        Returns:
            Predicted TracerState
        """
        # Convert to UKF state
        ukf_state = self._tracer_to_ukf_state(current_state)

        # Predict
        predicted_ukf_state = self.ukf.predict(ukf_state.mean, ukf_state.covariance, dt)

        # Convert back to TracerState
        return self._ukf_to_tracer_state(predicted_ukf_state, current_state, dt)

    def _update(self, current_state: TracerState, measurement: jnp.ndarray) -> TracerState:
        """
        Perform update step.

        Args:
            current_state: Current TracerState
            measurement: Measurement array [price_pct, tracer, alfa]

        Returns:
            Updated TracerState
        """
        # Convert to UKF state
        ukf_state = self._tracer_to_ukf_state(current_state)

        # Update
        updated_ukf_state = self.ukf.update(ukf_state, measurement)

        # Convert back to TracerState (dt=0 since this is just update)
        return self._ukf_to_tracer_state(updated_ukf_state, current_state, dt=0.0)

    def predict_and_update(self,
                           current_state: TracerState,
                           measurement: jnp.ndarray,
                           dt: float) -> TracerState:
        """
        Perform both prediction and update steps.

        Args:
            current_state: Current TracerState
            measurement: Measurement array [price_pct, tracer, alfa]
            dt: Time step

        Returns:
            Updated TracerState
        """
        # Predict
        predicted_state = self._predict(current_state, dt)

        # Update
        updated_state = self._update(predicted_state, measurement)

        # Log if enabled
        if self.verbose:
            self.log_state(updated_state)

        return updated_state

    def log_state(self, state: TracerState):
        """
        Log current state information.

        Args:
            state: Current TracerState
        """
        if self.logger is not None:
            self.logger.info(f"Strategy: {self.strategy}, Job: {self.job}")
            self.logger.info(f"State: idx={state.idx}, dt_idx={state.dt_idx}")
            self.logger.info(f"price_pct={state.price_pct:.6f}, "
                             f"tracer={state.tracer:.6f}, "
                             f"alfa={state.alfa:.6f}")
            self.logger.info(f"Delta: {state.delta}")
            self.logger.info(f"Covariance: {state.state_cov}")

    def _save_tracer(self, file_dir, idx_run_np, dt_idx, price_np, change_np, tracer, alfa, delta, state_covs,
                     need_header=False):
        """
        Save the tracer data to a CSV file.

        This function saves the state from the UKF's update results to a CSV file, including the price, tracer, alfa, delta,
        and state covariance.

        Args:
            file_dir: Path to the output file (e.g., `f"{self.trade_args['data_path']}/KF_theta/{self.name}_{self.theta_id}.csv"`).
            idx_run_np: Index array for data saving.
            dt_idx: Datetime array.
            price_np: Price array.
            change_np: Change array (e.g., percentage price change).
            tracer: Tracer values (state estimates).
            alfa: Alpha values (deviation from observed price).
            delta: Process noise parameter values.
            state_covs: Covariances of the state estimates.
            need_header: Boolean indicating whether column headers are needed.
        """
        try:
            # Flatten state covariance for saving
            state_covs00 = state_covs.flatten()

            # Organize data for saving
            data = np.column_stack(
                [
                    dt_idx,
                    price_np,
                    change_np,
                    tracer,
                    alfa,
                    delta,
                    state_covs00,
                ]
            )

            # Create a DataFrame for the CSV
            r = pd.DataFrame(data, index=idx_run_np,
                             columns=["dt_idx", "price", "change", "tracer", "alfa", "delta", "state_cov00"])

            # Ensure directory exists for saving
            create_dir_if_non_exist(file_dir)

            # Save to CSV
            r.to_csv(file_dir, mode="a", header=need_header, index=True, na_rep="None", index_label="idx")

            if self.verbose:
                self.logger.info(f"[UKF] New tracer data saved to {file_dir}")

        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(f"[UKF] Error while saving tracer data: {err}")
            sys.exit()


def load_price():
    exchange = 'Binance'
    symbol = 'SOLUSDT'
    data_path = './appData/trainData_crypto/prices_v3.parquet'
    days = 50
    # form_start = (dt.date.today() - dt.timedelta(days=days)).strftime('%Y-%m-%d')
    form_start = '2021-01-01'
    form_end = '2021-03-01'
    klines = read_data(exchange, symbol, data_path, ['open', 'close']).compute()
    price = pd.DataFrame(klines, columns=price_columns, index=klines.index)
    # price.index = pd.to_datetime(priceA.date)
    price_pd = price.close
    price_pd = price_pd[form_start:form_end]

    price_pd = price_pd.resample(rule='2h', label='left', closed='right').mean().interpolate()
    # price_pct = price_pd.pct_change()[1:]

    logger = setup_logger('brunhild_naive_tracer_unit_test.log', symbol)

    strategy = 'brunhild_naive'

    buy_delta = 0.6
    buy_sd_delta = 1e-4

    return exchange, symbol, days, form_start, price_pd, logger, strategy, buy_delta, buy_sd_delta

# # Initialize
# tracer_ukf = TracerUKF(
#     strategy="example",
#     job="test",
#     delta=0.1,
#     obs_cov=0.01,
#     logger=your_logger,
#     alpha=1e-3,
#     beta=2,
#     kappa=0
# )
#
# # Create initial state
# state = tracer_ukf.create_initial_state(idx=0, dt_idx=0.0)
#
# # Option 1: Separate predict and update
# predicted_state = tracer_ukf.predict(state, dt=1.0)
# measurement = jnp.array([0.1, 0.2, 0.3])
# updated_state = tracer_ukf.update(predicted_state, measurement)
#
# # Option 2: Combined predict and update
# state = tracer_ukf.predict_and_update(state, measurement, dt=1.0)

def unit_test_running():
    # Load price data and configuration
    exchange, symbol, days, form_start, price_pd, logger, strategy, buy_delta, _ = load_price()

    # Convert price data to numpy and calculate percentage changes
    price_np = price_pd.to_numpy()
    price_pct = price_np  # pct_change(price_np, include_first=True)

    # Parameters for UKF
    buy_obs_cov = 1.0
    alpha = 1e-3
    beta = 2.0
    kappa = 0.0

    # Initialize storage for results
    buy_tracer = []
    buy_state_cov = []
    buy_alfa = []

    # Initialize UKF-Based Buy Signal Tracker
    tracer_ukf = TracerUKF(
        strategy=strategy,
        job="buy_tracer",
        delta=buy_delta,
        obs_cov=buy_obs_cov,
        logger=logger,
        alpha=alpha,
        beta=beta,
        kappa=kappa,
        verbose=True,
        gc=False
    )

    # Initialize state
    current_state = tracer_ukf.create_initial_state(idx=0, dt_idx=0.0, init_value=price_pct[0])

    # Run TracerUKF
    dt = 1.0  # Fixed time step
    for idx, price_change in enumerate(price_pct):
        # Create measurement array [price_pct, tracer, alfa]
        measurement = jnp.array([price_change, 0.0, 0.0])

        # Predict and update
        current_state = tracer_ukf.predict_and_update(
            current_state=current_state,
            measurement=measurement,
            dt=dt
        )

        # Store results
        buy_tracer.append(current_state.tracer)
        buy_state_cov.append(current_state.state_cov[0, 0])  # Using diagonal element
        buy_alfa.append(current_state.alfa)

    # Convert lists to numpy arrays
    buy_tracer = np.array(buy_tracer)
    buy_state_cov = np.array(buy_state_cov)
    buy_alfa = np.array(buy_alfa)

    # Format datetime index for plotting
    dt_index = price_pd.index[10:]

    # Plot data and results
    plt.figure(2, figsize=(15, 8))

    # Plot 1: Price
    ax1 = plt.subplot(411)
    plt.plot(price_pd[10:], label="price", lw=0.8)
    plt.grid()
    plt.legend(loc="lower right")

    # Plot 2: Buy Tracer and State Covariance
    ax2 = plt.subplot(412, sharex=ax1)
    plt.plot(dt_index, buy_tracer[10:], label="buy tracer", lw=0.8)
    plt.plot(dt_index, buy_state_cov[10:], label="buy state covariance", ls="--", lw=1)
    plt.axhline(0, color="black", ls="--", lw=1)
    plt.axhline(0.2, color="green", ls="-.", lw=1)
    plt.axhline(-0.2, color="green", ls="-.", lw=1)
    plt.grid()
    plt.legend(loc="lower right")

    # Plot 3: Alfa (Deviation of Observations from Predicted States)
    ax3 = plt.subplot(413, sharex=ax1)
    plt.plot(dt_index, buy_alfa[10:], label="buy alfa", lw=0.8)
    plt.axhline(0, color="black", ls="--", lw=1)
    plt.axhline(1, color="green", ls="--", lw=0.5)
    plt.axhline(-1, color="green", ls="--", lw=0.5)
    plt.grid()
    plt.legend(loc="lower right")

    # Plot 4: Example Tracer Levels
    ax4 = plt.subplot(414, sharex=ax1)
    plt.plot(dt_index, buy_alfa[10:], label="buy alfa (repeated)", lw=0.8)
    plt.axhline(0, color="black", ls="--", lw=1)
    plt.axhline(1, color="green", ls="--", lw=0.5)
    plt.axhline(-1, color="green", ls="--", lw=0.5)
    plt.axhline(2, color="green", ls="--", lw=0.5)
    plt.axhline(-2, color="green", ls="--", lw=0.5)
    plt.grid()
    plt.legend(loc="lower right")

    # Display the plots
    plt.show()

    # Return results
    return buy_tracer, buy_alfa, buy_state_cov


if __name__ == '__main__':
    unit_test_running()
