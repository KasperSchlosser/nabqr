import numpy as np
import pandas as pd

def set_n_smallest_to_zero(arr, n):
    if n <= 0:
        return arr
    
    if n >= len(arr):
        return [0] * len(arr)
    
    # Find the nth smallest element
    nth_smallest = sorted(arr)[n-1]
    print(nth_smallest)
    
    # Set elements smaller than or equal to nth_smallest to zero
    modified_arr = [0 if x <= nth_smallest else x for x in arr]
    modified_arr = np.array(modified_arr)
    return modified_arr

def set_n_closest_to_zero(arr, n):
    if n <= 0:
        return arr
    
    if n >= len(arr):
        return [0] * len(arr)
    
    # Find the absolute values of the elements
    abs_arr = np.abs(arr)
    
    # Find the indices of the n elements closest to zero
    closest_indices = np.argpartition(abs_arr, n)[:n]
    
    # Set the elements closest to zero to zero
    modified_arr = arr.copy()
    modified_arr[closest_indices] = 0
    
    return modified_arr

def quantile_score(p, z, q):
    """
    Calculate the Quantile Score (QS) for a given probability and set of observations and quantiles.

    Parameters:
    p (float): The probability level.
    z (np.array): The observed values.
    q (np.array): The predicted quantiles.

    Returns:
    float: The Quantile Score (QS).

    From "Flexible and consistent quantile estimation for
            intensity–duration–frequency curves"
            by
            Felix S. Fauer, Jana Ulrich, Oscar E. Jurado, and Henning W. Rust, 2021
    We implemented this directly into the network...
    """
    u = z - q
    rho = np.where(u > 0, p * u, (p - 1) * u)
    return np.sum(rho)  

def simulate_correlated_ar1_process(n, phi, sigma, m, corr_matrix=None, offset=None, smooth="no"):
    if offset is None:
        offset = np.zeros(m)
    elif len(offset) != m:
        raise ValueError("Length of offset array must be equal to m")
    
    if corr_matrix is None:
        corr_matrix = np.eye(m)  # Default to no correlation (identity matrix)
    elif corr_matrix.shape != (m, m):
        raise ValueError("Correlation matrix must be of shape (m, m)")

    # Ensure the covariance matrix is positive semi-definite
    cov_matrix = sigma**2 * corr_matrix
    L = np.linalg.cholesky(cov_matrix)  # Cholesky decomposition

    if isinstance(smooth, int):
        ensembles = np.zeros((n + smooth, m))
        ensembles[0] = np.random.multivariate_normal(np.zeros(m), cov_matrix)

        for t in range(1, n + smooth):
            noise = np.random.multivariate_normal(np.zeros(m), cov_matrix)
            ensembles[t] = phi * ensembles[t-1] + noise

        # Extract the smoothed part of the ensembles
        smoothed_ensembles = ensembles[smooth:]

        return smoothed_ensembles + offset, np.median(smoothed_ensembles + offset, axis=1) + np.random.normal(0, sigma/2, n)

    else:
        ensembles = np.zeros((n, m))
        ensembles[0] = np.random.multivariate_normal(np.zeros(m), cov_matrix)

        for t in range(1, n):
            noise = np.random.multivariate_normal(np.zeros(m), cov_matrix)
            ensembles[t] = phi * ensembles[t-1] + noise
        return ensembles + offset, np.median(ensembles+ offset, axis=1) + np.random.normal(0, sigma/2, n)

# Example usage
# offset = np.arange(10, 110, 10)
# corr_matrix = 0.8 * np.ones((10, 10)) + 0.2 * np.eye(10)  # Example correlation structure
# simulated_data, actuals = simulate_correlated_ar1_process(2500, 0.995, 5, 10, corr_matrix, offset, smooth=5)

# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(simulated_data, color = "grey", alpha = 0.2)
# plt.plot(actuals, color = "black")
# plt.show()

 