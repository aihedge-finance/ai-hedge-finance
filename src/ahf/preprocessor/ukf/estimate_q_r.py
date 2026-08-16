import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def estimate_noise_covariances(data, method="sma", window=10, scale_q=1.0):
    """
    Estimate process noise covariance components [Q1, Q2, Q3]
    and observation noise covariance (R) from position (price) data.

    Parameters:
        data (array-like): Time-series data (e.g., noisy commodity prices).
        method (str): Smoothing method to use ("sma", "ema", or "regression").
        window (int): Window size for smoothing (applies to SMA and EMA).
        scale_q (float): Scaling factor for Q components to allow flexibility.

    Returns:
        Q (list): [Q1 (position), Q2 (velocity), Q3 (acceleration)].
        R (float): Estimated observation noise covariance.
    """
    # Convert data to a NumPy array
    data = np.array(data)

    # Step 1: Smooth the position data using the chosen method
    if method == "sma":
        smoothed = pd.Series(data).rolling(window=window).mean().to_numpy()
    elif method == "ema":
        smoothed = pd.Series(data).ewm(span=window, adjust=False).mean().to_numpy()
    elif method == "regression":
        x = np.arange(len(data)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, data)
        smoothed = model.predict(x)
    else:
        raise ValueError("Invalid smoothing method. Choose 'sma', 'ema', or 'regression'.")

    # Remove NaN values caused by smoothing (due to SMA or EMA windows)
    smoothed = smoothed[~np.isnan(smoothed)]

    # Step 2: Compute residuals between raw and smoothed data (used for R and Q1)
    residuals = data[:len(smoothed)] - smoothed
    R = np.var(residuals)  # Variance of residuals -> Observation noise (R)

    # Step 3: Estimate velocity (first difference of smoothed position)
    velocity = np.diff(smoothed, prepend=smoothed[0])  # Simple finite difference
    Q2 = np.var(velocity) * scale_q  # Variance of velocity changes -> Q2

    # Step 4: Estimate acceleration (second difference of smoothed position)
    acceleration = np.diff(velocity, prepend=velocity[0])  # Second-order finite difference
    Q3 = np.var(acceleration) * scale_q  # Variance of accelerations -> Q3

    # Step 5: Position noise (Q1)
    Q1 = np.var(smoothed - data[:len(smoothed)]) * scale_q  # Variance of position residuals -> Q1

    # Combine into process noise covariance components
    Q = [Q1, Q2, Q3]
    return Q, R  # Return Q as a list and R as a single scalar

# Example Usage
if __name__ == "__main__":
    # Simulated noisy commodity price data
    np.random.seed(42)
    n_points = 150
    true_prices = np.linspace(50, 55, n_points)  # Gradual trend
    noisy_prices = true_prices + np.random.normal(0, 2, n_points)  # Add noise

    # Estimate Q and R using SMA
    Q_sma, R_sma, residuals_sma = estimate_q_r(noisy_prices, method="sma", window=10)
    print(f"SMA Method -> Q: {Q_sma}, R: {R_sma}")

    # Estimate Q and R using EMA
    Q_ema, R_ema, residuals_ema = estimate_q_r(noisy_prices, method="ema", window=10)
    print(f"EMA Method -> Q: {Q_ema}, R: {R_ema}")

    # Estimate Q and R using Regression
    Q_reg, R_reg, residuals_reg = estimate_q_r(noisy_prices, method="regression")
    print(f"Regression Method -> Q: {Q_reg}, R: {R_reg}")

    # Step 4: Compare accuracy based on residual variance
    residual_variances = {
        "SMA": np.var(residuals_sma),
        "EMA": np.var(residuals_ema),
        "Regression": np.var(residuals_reg),
    }

    # Identify the most accurate method
    best_method = min(residual_variances, key=residual_variances.get)
    print("\nAccuracy Comparison (Residual Variance):")
    for method, variance in residual_variances.items():
        print(f"{method}: Residual Variance = {variance}")
    print(f"\nThe most accurate method is: {best_method}")