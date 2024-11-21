from functions import *
from helper_functions import simulate_correlated_ar1_process, set_n_closest_to_zero
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['no-latex'])
from visualization import visualize_results 
import datetime as dt
# Example usage. Inputs:
offset = np.arange(10, 500, 15)
m = len(offset)
corr_matrix = 0.8 * np.ones((m, m)) + 0.2 * np.eye(m)  # Example correlation structure
data_source = "NABQR-TEST"
today = dt.datetime.today().strftime('%Y-%m-%d')

simulated_data, actuals = simulate_correlated_ar1_process(5000, 0.995, 8, m, corr_matrix, offset, smooth=5)

# Optional kwargs
quantiles_taqr = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]

pipeline(simulated_data, actuals, data_source, training_size = 0.7, epochs = 100, timesteps_for_lstm = [0,1,2,6,12,24], quantiles_taqr = quantiles_taqr)

# Import old results
CE = pd.read_csv(f"results_{today}_{data_source}_corrected_ensembles.csv")
y_hat = np.load(f"results_{today}_{data_source}_actuals_out_of_sample.npy")
q_hat = np.load(f"results_{today}_{data_source}_taqr_results.npy")

# Call the visualization function
visualize_results(y_hat, q_hat, "NABQR-TEST example")