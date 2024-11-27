import numpy as np
import pandas as pd

def set_n_smallest_to_zero(arr, n):
    """
    Sets the n smallest elements in an array to zero.
    
    Parameters:
    arr (array-like): Input array of numbers
    n (int): Number of smallest elements to set to zero
    
    Returns:
    numpy.ndarray: Modified array with n smallest elements set to zero
    """
    if n <= 0:
        return arr
    
    if n >= len(arr):
        return [0] * len(arr)
    
    # Find the nth smallest element
    nth_smallest = sorted(arr)[n-1]
    
    # Set elements smaller than or equal to nth_smallest to zero
    modified_arr = [0 if x <= nth_smallest else x for x in arr]
    modified_arr = np.array(modified_arr)
    return modified_arr

def set_n_closest_to_zero(arr, n):
    """
    Sets the n elements closest to zero in an array to zero.
    
    Parameters:
    arr (array-like): Input array of numbers
    n (int): Number of elements closest to zero to set to zero
    
    Returns:
    numpy.ndarray: Modified array with n elements closest to zero set to zero
    """
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
    Implementation based on Fauer et al. (2021): "Flexible and consistent quantile estimation for
    intensity–duration–frequency curves"

    Parameters:
    p (float): The probability level (between 0 and 1)
    z (numpy.ndarray): The observed values
    q (numpy.ndarray): The predicted quantiles

    Returns:
    float: The Quantile Score (QS)
    """
    u = z - q
    rho = np.where(u > 0, p * u, (p - 1) * u)
    return np.sum(rho)  

def simulate_correlated_ar1_process(n, phi, sigma, m, corr_matrix=None, offset=None, smooth="no"):
    """
    Simulate a correlated AR(1) process with multiple dimensions.
    
    Parameters:
    n (int): Number of time steps to simulate
    phi (float): AR(1) coefficient (persistence parameter)
    sigma (float): Standard deviation of the noise
    m (int): Number of dimensions/variables
    corr_matrix (numpy.ndarray, optional): Correlation matrix between dimensions. Defaults to identity matrix
    offset (numpy.ndarray, optional): Offset vector for each dimension. Defaults to zero vector
    smooth (int or str, optional): Number of initial time steps to discard for smoothing. Defaults to "no"
    
    Returns:
    tuple: (simulated_ensembles, actuals) where simulated_ensembles is the AR(1) process 
           and actuals is the median of ensembles with added noise
    """
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

 