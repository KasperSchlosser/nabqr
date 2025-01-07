import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from typing import Dict, List, Tuple
import time

def set_n_smallest_to_zero(arr, n):
    """Set the n smallest elements in an array to zero.

    Parameters
    ----------
    arr : array-like
        Input array of numbers
    n : int
        Number of smallest elements to set to zero

    Returns
    -------
    numpy.ndarray
        Modified array with n smallest elements set to zero
    """
    if n <= 0:
        return arr

    if n >= len(arr):
        return [0] * len(arr)

    # Find the nth smallest element
    nth_smallest = sorted(arr)[n - 1]

    # Set elements smaller than or equal to nth_smallest to zero
    modified_arr = [0 if x <= nth_smallest else x for x in arr]
    modified_arr = np.array(modified_arr)
    return modified_arr


def set_n_closest_to_zero(arr, n):
    """Set the n elements closest to zero in an array to zero.

    Parameters
    ----------
    arr : array-like
        Input array of numbers
    n : int
        Number of elements closest to zero to set to zero

    Returns
    -------
    numpy.ndarray
        Modified array with n elements closest to zero set to zero
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
    """Calculate the Quantile Score (QS) for a given probability and set of observations and quantiles.

    Implementation based on Fauer et al. (2021): "Flexible and consistent quantile estimation for
    intensity–duration–frequency curves"

    Parameters
    ----------
    p : float
        The probability level (between 0 and 1)
    z : numpy.ndarray
        The observed values
    q : numpy.ndarray
        The predicted quantiles

    Returns
    -------
    float
        The Quantile Score (QS)
    """
    u = z - q
    rho = np.where(u > 0, p * u, (p - 1) * u)
    return np.sum(rho)


# def simulate_correlated_ar1_process(
#     n, phi, sigma, m, corr_matrix=None, offset=None, smooth="no"
# ):
#     """Simulate a correlated AR(1) process with multiple dimensions.

#     Parameters
#     ----------
#     n : int
#         Number of time steps to simulate
#     phi : float
#         AR(1) coefficient (persistence parameter)
#     sigma : float
#         Standard deviation of the noise
#     m : int
#         Number of dimensions/variables
#     corr_matrix : numpy.ndarray, optional
#         Correlation matrix between dimensions. Defaults to identity matrix
#     offset : numpy.ndarray, optional
#         Offset vector for each dimension. Defaults to zero vector
#     smooth : int or str, optional
#         Number of initial time steps to discard for smoothing. Defaults to "no"

#     Returns
#     -------
#     tuple
#         (simulated_ensembles, actuals) where simulated_ensembles is the AR(1) process
#         and actuals is the median of ensembles with added noise
#     """
#     if offset is None:
#         offset = np.zeros(m)
#     elif len(offset) != m:
#         raise ValueError("Length of offset array must be equal to m")

#     if corr_matrix is None:
#         corr_matrix = np.eye(m)  # Default to no correlation (identity matrix)
#     elif corr_matrix.shape != (m, m):
#         raise ValueError("Correlation matrix must be of shape (m, m)")

#     # Ensure the covariance matrix is positive semi-definite
#     cov_matrix = sigma**2 * corr_matrix
#     L = np.linalg.cholesky(cov_matrix)  # Cholesky decomposition

#     if isinstance(smooth, int):
#         ensembles = np.zeros((n + smooth, m))
#         ensembles[0] = np.random.multivariate_normal(np.zeros(m), cov_matrix)

#         for t in range(1, n + smooth):
#             noise = np.random.multivariate_normal(np.zeros(m), cov_matrix)
#             ensembles[t] = phi * ensembles[t - 1] + noise

#         # Extract the smoothed part of the ensembles
#         smoothed_ensembles = ensembles[smooth:]

#         return smoothed_ensembles + offset, np.median(
#             smoothed_ensembles + offset, axis=1
#         ) + np.random.normal(0, sigma / 2, n)

#     else:
#         ensembles = np.zeros((n, m))
#         ensembles[0] = np.random.multivariate_normal(np.zeros(m), cov_matrix)

#         for t in range(1, n):
#             noise = np.random.multivariate_normal(np.zeros(m), cov_matrix)
#             ensembles[t] = phi * ensembles[t - 1] + noise
#         return ensembles + offset, np.median(
#             ensembles + offset, axis=1
#         ) + np.random.normal(0, sigma / 2, n)


import numpy as np
import pandas as pd


def build_ar1_covariance(n, rho, sigma=1.0):
    """
    Build the AR(1) covariance matrix for an n-dimensional process.

    Parameters
    ----------
    n : int
        Dimension of the covariance matrix.
    rho : float
        AR(1) correlation parameter (the AR coefficient).
    sigma : float, optional
        Standard deviation of the noise (innovation), defaults to 1.0.

    Returns
    -------
    numpy.ndarray
        The AR(1) covariance matrix of shape (n, n), with elements sigma^2 * rho^(|i-j|).
    """
    indices = np.arange(n)
    abs_diff = np.abs(np.subtract.outer(indices, indices))
    cov_matrix = (sigma**2) * (rho**abs_diff)
    return cov_matrix


def simulate_correlated_ar1_process(
    n, phi, sigma, m, corr_matrix=None, offset=None, smooth="no"
):
    """Simulate a correlated AR(1) process with multiple dimensions.

    Parameters
    ----------
    n : int
        Number of time steps to simulate
    phi : float
        AR(1) coefficient (persistence parameter, often denoted rho)
    sigma : float
        Standard deviation of the noise
    m : int
        Number of dimensions/variables
    corr_matrix : numpy.ndarray, optional
        Correlation (or covariance) matrix between dimensions. If None, an AR(1) covariance
        structure will be generated.
    offset : numpy.ndarray, optional
        Offset vector for each dimension. Defaults to zero vector
    smooth : int or str, optional
        Number of initial time steps to discard for smoothing. Defaults to "no"

    Returns
    -------
    tuple
        (simulated_ensembles, actuals) where simulated_ensembles is the AR(1) process
        and actuals is the median of ensembles with added noise
    """
    if offset is None:
        offset = np.zeros(m)
    elif len(offset) != m:
        raise ValueError("Length of offset array must be equal to m")

    # If no correlation matrix is provided, build the AR(1) covariance matrix
    if corr_matrix is None:
        # Here we assume phi is the AR(1) correlation parameter
        corr_matrix = build_ar1_covariance(m, phi, sigma)
    elif corr_matrix.shape != (m, m):
        raise ValueError("Correlation matrix must be of shape (m, m)")

    # cov_matrix now is already constructed (AR(1) type if corr_matrix was None)
    cov_matrix = corr_matrix

    if isinstance(smooth, int):
        ensembles = np.zeros((n + smooth, m))
        ensembles[0] = np.random.multivariate_normal(np.zeros(m), cov_matrix)

        for t in range(1, n + smooth):
            noise = np.random.multivariate_normal(np.zeros(m), cov_matrix)
            ensembles[t] = phi * ensembles[t - 1] + noise

        # Extract the smoothed part of the ensembles
        smoothed_ensembles = ensembles[smooth:]

        return (
            smoothed_ensembles + offset,
            np.median(smoothed_ensembles + offset, axis=1)
            + np.random.normal(0, sigma / 2, n),
        )
    else:
        ensembles = np.zeros((n, m))
        ensembles[0] = np.random.multivariate_normal(np.zeros(m), cov_matrix)

        for t in range(1, n):
            noise = np.random.multivariate_normal(np.zeros(m), cov_matrix)
            ensembles[t] = phi * ensembles[t - 1] + noise

        return (
            ensembles + offset,
            np.median(ensembles + offset, axis=1) + np.random.normal(0, sigma / 2, n),
        )

def get_parameter_bounds() -> Dict[str, Tuple[float, float]]:
    """Define bounds for all parameters for SDE simulation"""
    return {
        'X0': (0.0, 1.0),
        'theta': (0.2, 0.8),       # Lowered upper bound for mean level
        'kappa': (0.05, 0.5),      # Reduced mean reversion speed
        'sigma_base': (0.01, 2),  # Reduced base volatility
        'alpha': (0.01, 0.8),      # Reduced ARCH effect
        'beta': (0.7, 1.2),       # Increased persistence
        'lambda_jump': (0.005, 0.05), # Fewer jumps
        'jump_mu': (-0.2, 0.2),    # Allowing negative jumps
        'jump_sigma': (0.05, 0.2)   # More consistent jump sizes
    }

import numpy as np

def generate_ou_ensembles(
    X: np.ndarray,
    kappa: float,
    sigma: float,
    chunk_size: int = 24,
    n_ensembles: int = 50
) -> np.ndarray:
    """
    Generate continuous Ornstein-Uhlenbeck (OU) ensemble paths that revert
    to the given reference series X[t]. Each path is simulated in chunks
    of 'chunk_size' (e.g. 24 timesteps). New chunks start exactly
    where the previous chunk ended, ensuring continuity for each path.

    The paths are clipped to remain within [0,1].

    Parameters
    ----------
    X : np.ndarray
        Reference series of length T that serves as the time-varying mean
        for each OU path.
    kappa : float
        Mean-reversion speed for the OU process.
    sigma : float
        Diffusion (volatility) parameter.
    chunk_size : int, optional
        Size of each chunk in timesteps. Defaults to 24.
    n_ensembles : int, optional
        Number of ensemble paths to generate. Defaults to 50.

    Returns
    -------
    Y : np.ndarray, shape (T, n_ensembles)
        Array of OU ensemble paths. Each column is one continuous path of length T,
        re-initialized at chunk boundaries but joined at the previous chunk's endpoint.

    Notes
    -----
    - The time step `dt` is assumed to be 1.0 for simplicity.
    - The OU update at each step t -> t+1 is:
         Y[t+1] = Y[t] + kappa*( X[t] - Y[t] ) + sigma*randn()
      where randn() is a draw from a standard normal distribution.
    - The final chunk may be shorter than `chunk_size` if the length of X is
      not a multiple of chunk_size.
    - The function returns a matrix Y with shape (T, n_ensembles). You can
      plot Y[:, i] over time to see the i-th ensemble path.

    Examples
    --------
    >>> # Suppose we have a reference X of length 100
    >>> X = np.linspace(0.2, 0.8, 100)
    >>> # Parameters
    >>> kappa = 0.3
    >>> sigma = 0.05
    >>> # Generate ensemble
    >>> Y = generate_ou_ensembles(X, kappa, sigma, chunk_size=24, n_ensembles=50)
    >>> # You can now visualize these 50 continuous OU paths
    >>> import matplotlib.pyplot as plt
    >>> for i in range(Y.shape[1]):
    ...     plt.plot(Y[:, i], alpha=0.3)
    >>> plt.plot(X, 'k--', label='Reference X')
    >>> plt.legend()
    >>> plt.show()
    """
    T = len(X)
    # Allocate array for ensembles: shape (T, n_ensembles)
    Y = np.zeros((T, n_ensembles))
    
    # For the very first chunk, initialize all ensembles at X[0]
    Y[0, :] = X[0]
    
    # Precompute chunk boundaries
    # e.g., if T=90 and chunk_size=24, chunk_starts = [0, 24, 48, 72]
    # the last chunk will run from 72 -> 90 (18 steps)
    chunk_starts = list(range(0, T, chunk_size))
    
    # Main simulation loop: each chunk is integrated from chunk_start to chunk_end - 1
    for idx_chunk in range(len(chunk_starts)):
        start = chunk_starts[idx_chunk]
        end = min(start + chunk_size, T)  # end might be T if remainder < chunk_size
        
        # If start==0, we already have Y[start, :] = X[start], so just proceed
        # If start!=0, we need to set Y[start, :] to the "inherited" value
        # from Y[start-1, :], because the chunk should start seamlessly
        # from the previous chunk's last step.
        if start > 0:
            Y[start, :] = Y[start - 1, :]
        
        # Step through the chunk:
        # from 'start' to 'end-1' so that we compute Y[i+1] up to i=end-1
        for i in range(start, end - 1):
            # OU update:
            # Y[i+1] = Y[i] + kappa*( X[i] - Y[i] ) + sigma * randn
            drift = kappa * (X[i] - Y[i, :])
            diffusion = sigma * np.random.randn(n_ensembles)
            Y[i + 1, :] = Y[i, :] + drift + diffusion
            
            # Enforce [0, 1] clipping
            Y[i + 1, :] = np.clip(Y[i + 1, :], 0.0, 1.0)
    
    return Y

def simulate_wind_power_sde(params: Dict[str, float], T: float = 500, dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate wind power production using an Ornstein-Uhlenbeck process with GARCH volatility
    and jumps of normally distributed sizes driven by a Poisson process. The mean reversion
    is state-dependent with a repelling mechanism near 1.0 (upper boundary), and the diffusion
    term vanishes at the boundaries to avoid unphysical values outside [0, 1].

    A few additional tweaks include:
    - GARCH volatility that captures 'vol_shock' from recent values.
    - Repellent forces that strengthen near 1.0, reducing both the drift and diffusion.
    - Jumps that can persist over multiple steps, and become more negative if values are near 1.0.

    Parameters
    ----------
    params : Dict[str, float]
        A dictionary containing all model parameters:
        
        - X0 : float
            Initial wind power production level in [0, 1].
        - theta : float
            Long-term mean level; typically in [0, 1].
        - kappa : float
            Mean reversion speed (absolute value is used).
        - sigma_base : float
            Base volatility level (absolute value is used).
        - alpha : float
            ARCH parameter (absolute value is used).
        - beta : float
            GARCH parameter; must be in [0, 1].
        - lambda_jump : float
            Intensity of jump arrivals in the Poisson process (absolute value is used).
        - jump_mu : float
            Mean jump size (can be positive or negative).
        - jump_sigma : float
            Standard deviation of jump sizes (absolute value is used).
    T : float, optional
        The end time of the simulation (total number of steps is T/dt). Default is 500.
    dt : float, optional
        The size of each time step. Default is 1.0.

    Returns
    -------
    t : np.ndarray
        Array of time points of length N = int(T/dt).
    X : np.ndarray
        Simulated wind power production values of length N, clipped to the interval [0, 1].

    Notes
    -----
    - The drift term implements a state-dependent mean reversion that weakens
      near 1.0 and introduces a strong downward force very close to 1.0.
    - The diffusion term is modified as
      (X_t * (1 - X_t)) * (X_t / (X_t + 0.5)) dB_t,
      ensuring it decreases to zero when X_t is near 0 or 1.
    - GARCH effects are included to model changing volatility based on recent
      shocks in the process.
    - Jumps arrive according to a Poisson process with random normal magnitudes,
      and can persist over multiple time steps with some decay.

    Examples
    --------
    >>> params = {
    ...     'X0': 0.5, 'theta': 0.7, 'kappa': 1.0, 'sigma_base': 0.1,
    ...     'alpha': 0.2, 'beta': 0.5, 'lambda_jump': 0.05,
    ...     'jump_mu': 0.0, 'jump_sigma': 0.02
    ... }
    >>> t, X = simulate_wind_power_sde(params, T=100, dt=1.0)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(t, X)
    >>> plt.show()
    """
    # Unpack parameters and ensure they're valid
    X0 = np.clip(params['X0'], 0, 1)
    theta = np.clip(params['theta'], 0, 1)  # Mean level
    kappa = abs(params['kappa'])           # Mean reversion speed
    sigma_base = abs(params['sigma_base']) # Base volatility
    
    # GARCH parameters
    alpha = abs(params['alpha'])           # ARCH parameter
    beta = np.clip(params['beta'], 0, 1)   # GARCH parameter
    
    # Jump parameters
    lambda_jump = abs(params['lambda_jump'])  # Jump intensity
    jump_mu = params['jump_mu']               # Jump mean
    jump_sigma = abs(params['jump_sigma'])    # Jump size volatility
    
    # Time grid
    t = np.linspace(0, T, int(T/dt))
    N = len(t)
    
    # Initialize arrays
    X = np.zeros(N)
    X[0] = X0
    
    # Initialize GARCH volatility
    sigma = np.zeros(N)
    sigma[0] = sigma_base
    
    # Initialize jump state variables
    jump_state = 0
    previous_jump = 0
    
    # Generate jumps
    jump_times = np.random.poisson(max(1e-10, lambda_jump * dt), N)
    jump_sizes = np.random.normal(jump_mu, max(1e-10, jump_sigma), N)
    
    # Simulation
    for i in range(1, N):
        # Update GARCH volatility
        vol_shock = np.abs(X[i-1] - X[max(0, i-2)]) / dt
        sigma[i] = np.sqrt(alpha * vol_shock**2 + beta * sigma[i-1]**2)
        
        # Get previous value and recent average
        X_prev = X[i-1]
        recent_avg = np.mean(X[max(0, i-5):i]) if i > 5 else X_prev
        
        # State-dependent mean reversion with repellent near 1.0
        if X_prev > 0.99:
            # Strong repellent force very close to 1.0
            drift = -kappa * 2.0 * dt
        elif X_prev > 0.7:
            drift = kappa * (theta - X_prev) * dt * 0.5
        else:
            drift = kappa * (theta - X_prev) * dt
        
        # Modified diffusion term with memory
        diff_term = (X_prev * (1 - X_prev)) * 0.4*(X_prev / (X_prev + 0.1))
        if X_prev > 0.75:
            # Reduce volatility if we've been stable at high values
            if abs(X_prev - recent_avg) < 0.1:
                diff_term *= 0.5
        
        # Additional repellent in diffusion term near 1.0
        if X_prev > 0.95:
            diff_term *= 0.3
            
        diffusion = sigma[i] * diff_term * np.sqrt(dt) * np.random.normal()
        
        # Persistent jumps with stronger downward bias near 1.0
        if jump_state > 0:
            # Continue the previous jump with decay
            jump = previous_jump * 0.9
            jump_state -= 1
        elif jump_times[i] > 0:
            # Start new jump
            jump = jump_sizes[i]
            if X_prev > 0.98:
                jump = -abs(jump) * 1.5
            jump_state = np.random.geometric(0.2)
            previous_jump = jump
            # Make downward jumps more persistent at high values
            if X_prev > 0.8 and jump < 0:
                jump_state = int(jump_state * 1.5)
        else:
            jump = 0
        
        # Update process
        X[i] = X[i-1] + drift + diffusion + jump
        
        # Ensure bounds
        X[i] = np.clip(X[i], 0, 1)

    # Apply moving average filter
    X = np.convolve(X, np.ones((4,))/4, mode='valid')
    
    # The following line is referencing an undefined 'initial_params'.
    # It seems intended to reuse X0. We'll just assume it was meant as X0 in params.
    X = np.concatenate(([X0] * 3, X))  # Append initial value to keep the same length
    
    # Final adjustment and bounds
    X = np.clip(X*1.05, 0, 1)

    # Now we will generate ensembles from X. 
    # We will use an OU process to generate ensembles

    ensembles = generate_ou_ensembles(
        X,
        kappa = 0.3,
        sigma = 0.05,
        chunk_size=24, # 24 hours
        n_ensembles=50
    )
    return t, X, ensembles