from nabqr.nabqr.functions import pipeline
import numpy as np
from nabqr.nabqr.helper_functions import simulate_correlated_ar1_process

# Example usage
offset = np.arange(10, 500, 15)
m = len(offset)
corr_matrix = 0.8 * np.ones((m, m)) + 0.2 * np.eye(m)  # Example correlation structure
simulated_data, actuals = simulate_correlated_ar1_process(500, 0.995, 8, m, corr_matrix, offset, smooth=5)

# Optional kwargs
quantiles_taqr = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]

pipeline(simulated_data, actuals, "NABQR-TEST", training_size = 0.7, epochs = 10, timesteps_for_lstm = [0,1,2,6,12,24], quantiles_taqr = quantiles_taqr)

