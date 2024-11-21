import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import properscoring as ps
import tensorflow as tf
import tensorflow_probability as tfp
import datetime as dt
from helper_functions import simulate_correlated_ar1_process, set_n_closest_to_zero
from functions_for_TAQR import *

def variogram_score_single_observation(x, y, p=0.5):
    """
    Calculate the Variogram score for a given observation.
    Translated to Python from the R code in the paper Energy and AI, >> An introduction to multivariate probabilistic forecast evaluation <<.

    Parameters:
    x : array
        Ensemble forecast (m x k), where m is the size of the ensemble, and k is the maximal forecast horizon.
    y : array
        Actual observations (k,)
    p : float
        p-parameter for the variogram score.
    """

    m, k = x.shape  # Size of ensemble, Maximal forecast horizon
    score = 0

    # Iterate through all pairs
    for i in range(k - 1):
        for j in range(i + 1, k):
            Ediff = (1 / m) * np.sum(np.abs(x[:, i] - x[:, j])**p)
            score += (1/np.abs(i-j))*(np.abs(y[i] - y[j])**p - Ediff)**2

    # Variogram score
    return score / k # this k can be omitted... 

def variogram_score_R_multivariate(x, y, p=0.5, t1=12, t2=36):
    """
    Calculate the Variogram score for all observations for the time horizon t1 to t2.
    Assumes that x and y starts from day 0, 00:00.
    
    Parameters:
    x : array
        Ensemble forecast (m x k), where m is the size of the ensemble, and k is the maximal forecast horizon.
    y : array
        Actual observations (k,)
    p : float
        Power parameter for the variogram score.
    t1 : int
        Start of the hour range for comparison (inclusive).
    t2 : int
        End of the hour range for comparison (exclusive).
    """

    m, k = x.shape  # Size of ensemble, Maximal forecast horizon
    score = 0
    if m > k:
        x = x.T
        m,k = k,m
    else:
        print("m,k: ", m, k)
    
    score_list = []
    # Iterate through every 24-hour block
    for start in range(0, k, 24):
        # Ensure we don't exceed the forecast horizon
        if start + t2 <= k:
            for i in range(start + t1, start + t2 - 1):
                for j in range(i + 1, start + t2):
                    Ediff = (1 / m) * np.sum(np.abs(x[:, i] - x[:, j])**p)
                    score += (1 / np.abs(i - j)) * (np.abs(y[i] - y[j])**p - Ediff)**2
                score_list.append(score)

    # Variogram score
    return score/(100_000), score_list

def calculate_crps(actuals, corrected_ensembles):
    try:
        crps = ps.crps_ensemble(actuals, corrected_ensembles)
        return np.mean(crps)
    except:
        print(f"CRPS failed") # for {actuals.name}, transposing and trying again...")
        crps = np.mean(ps.crps_ensemble(actuals, corrected_ensembles.T))
        return crps

def calculate_qss(actuals, taqr_results, quantiles):
    """
    Calculate the Quantile Skill Score (QSS) for multiple quantile forecasts.
    See multi_quantile_skill_score for more details.
    """
    qss_scores = multi_quantile_skill_score(actuals, taqr_results, quantiles)
    return np.mean(qss_scores)

def multi_quantile_skill_score(y_true, y_pred, quantiles):
    """
    Calculate the Quantile Skill Score (QSS) for multiple quantile forecasts.

    y_true: This is a 1D numpy array or list of the true observed values.
    y_pred: This is a 2D numpy array or list of lists of the predicted quantile values. The outer dimension should be the same length as y_true, and the inner dimension should be the number of quantiles.
    quantiles: This is a 1D numpy array or list of the quantile levels. It should be the same length as the inner dimension of y_pred.

    Parameters:
    y_true (numpy.array or list): True observed values. 1D array.
    y_pred (numpy.array or list of lists): Predicted quantile values. 2D array.
    quantiles (numpy.array or list): Quantile levels, between 0 and 1. 1D array.

    Returns:
    numpy.array: The QSS for each quantile forecast. 1D array.
    """

    # Convert y_pred to a numpy array
    y_pred = np.array(y_pred)

    if y_pred.shape[0] > y_pred.shape[1]:
        y_pred = y_pred.T

    assert all(0 <= q <= 1 for q in quantiles), "All quantiles must be between 0 and 1"
    assert len(quantiles) == len(y_pred), "Number of quantiles must match inner dimension of y_pred"

    N = len(y_true)
    scores = np.zeros(len(quantiles))

    for i, q in enumerate(quantiles):
        E = y_true - y_pred[i]
        scores[i] = np.sum(np.where(E > 0, q * E, (1 - q) * -E))

    return scores / N

def run_r_script(X_filename, Y_filename, tau):
    """
    Run the R script for quantile regression.
    """
    import subprocess
    process = subprocess.Popen(["R", "--vanilla"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    
    r_script = f"""
    
    options(warn = -1)
    library(onlineforecast) 
    library(quantreg) 
    library(readr) 
    X_full <- read_csv("{X_filename}", col_names = FALSE, show_col_types = FALSE) 
    y <- read_csv("{Y_filename}", col_names = "y", show_col_types = FALSE) 
    X_full <- X_full[1:500,] # [1,  3,  5,  7,  9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49]  
    data <- cbind(X_full, y[1:500,1]) 
    predictor_cols <- colnames(X_full) 
    formula_string <- paste("y ~ 0+", paste(predictor_cols, collapse = " + ")) 
    formula <- as.formula(formula_string) 
    rq_fit <- rq(formula, tau = {tau}, data = data ) 
    write.csv(rq_fit$coefficients, "rq_fit_coefficients.csv") 
    write.csv(rq_fit$residuals, "rq_fit_residuals.csv") 
    """
    
    for line in r_script.strip().split('\n'):
        process.stdin.write(line.encode('utf-8') + b"\n")

    process.stdin.close()

    output = process.stdout.read()
    # print(output.decode())

    process.terminate()

def remove_zero_columns(df):
    return df.loc[:, (df != 0).any(axis=0)]

def remove_zero_columns_numpy(arr):
    return arr[:, (arr != 0).any(axis=0) & (arr != arr[0]).any(axis=0)]

def create_dataset_for_lstm(X, Y, time_steps):
    '''
    This function takes in multidimensional X and array Y with equal length.
    Let us assume X takes the shape (50,10), then the output Xs would have shape (50, len(time_steps), 10)
    '''
    #Ensure X and Y are numpy arrays
    X = np.array(X)
    Y = np.array(Y)

    Xs, Ys = [], []
    for i in range(len(X)):
        X_entry = []
        for ts in time_steps:
            if i - ts >= 0:
                X_entry.append(X[i - ts,:])
            else:
                X_entry.append(np.zeros_like(X[0,:]))  # Padding with zeros for initial entries
        Xs.append(np.array(X_entry))
        Ys.append(Y[i])  # Current day's value
    return np.array(Xs), np.array(Ys)

class QuantileRegressionLSTM(tf.keras.Model):
    def __init__(self, n_quantiles, units, n_timesteps,**kwargs):
        super().__init__(**kwargs)
        self.lstm = tf.keras.layers.LSTM(units, input_shape=(None, n_quantiles, n_timesteps), return_sequences=False)
        # self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dense = tf.keras.layers.Dense(n_quantiles, activation='sigmoid')
        self.dense2 = tf.keras.layers.Dense(n_quantiles, activation='relu') 
        self.n_quantiles = n_quantiles
        self.n_timesteps = n_timesteps

    def call(self, inputs, training=None):
        x = self.lstm(inputs, training=training)
        # x = self.layer_norm(x)
        x = self.dense(x)
        x = self.dense2(x)
        return x
    
    def get_config(self):
        config = super(QuantileRegressionLSTM, self).get_config()
        config.update({
            'n_quantiles': self.n_quantiles,
            'units': self.lstm.units,
            'n_timesteps': self.n_timesteps,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def quantile_loss_3(q, y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tfp.stats.percentile(y_true, 100*q, axis = 1)
    error = y_true - y_pred
    return tf.maximum(q * error, (q - 1) * error) 

def quantile_loss_func(quantiles):
    def loss(y_true, y_pred):
        losses = []
        for i, q in enumerate(quantiles):
            loss = quantile_loss_3(q, y_true, y_pred[:,  i]) 
            losses.append(loss)
        return losses
    return loss

def map_range(values, input_start, input_end, output_start, output_end):

    mapped_values = []
    for value in values:
        # Calculate the proportion of value in the input range
        proportion = (value - input_start) / (input_end - input_start)
        
        # Map the proportion to the output range
        mapped_value = output_start + (proportion * (output_end - output_start))
        mapped_values.append(int(mapped_value))
    
    return np.array(mapped_values)

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

import numpy as np

def remove_straight_line_outliers(ensembles):
    """
    Removes ensemble members that are perfectly straight lines (constant slope).
    
    Parameters:
        ensembles (numpy.ndarray): 2D array where rows are time steps and columns are ensemble members.
    
    Returns:
        numpy.ndarray: Filtered ensemble data without straight-line outliers.
    """
    # Calculate differences along the time axis
    differences = np.diff(ensembles, axis=0)
    
    # Identify columns where all differences are the same (perfectly straight lines)
    straight_line_mask = np.all(differences == differences[0, :], axis=0)
    
    # Remove the columns with perfectly straight lines
    return ensembles[:, ~straight_line_mask]

def train_model_lstm(quantiles, epochs: int, lr: float, batch_size: int, x, y, x_val, y_val, n_timesteps, data_name):
 
    model = QuantileRegressionLSTM(n_quantiles=len(quantiles), units=256, n_timesteps=n_timesteps)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    @tf.function
    def train_step(x_batch, y_batch):

        with tf.GradientTape() as tape:
            y_pred = model(x_batch, training=True)
            losses = quantile_loss_func(quantiles)(y_batch, y_pred)
            total_loss = tf.reduce_mean(losses)
        
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss
    
    @tf.function
    def val_step(x_batch, y_batch):
        y_pred = model(x_batch, training=False)
        losses = quantile_loss_func(quantiles)(y_batch, y_pred)
        total_loss = tf.reduce_mean(losses)
        return total_loss

    train_loss_history = []
    val_loss_history = []
    y_preds = []
    y_true = []

    for epoch in range(epochs):
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        num_batches = 0
        
        # Training loop
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            
            batch_train_loss = train_step(x_batch, y_batch)
            epoch_train_loss += batch_train_loss
            num_batches += 1
            
            y_preds.append(model(x_batch, training=False))
            y_true.append(y_batch)
        
        epoch_train_loss /= num_batches
        train_loss_history.append(epoch_train_loss)
        
        # Validation loop
        num_val_batches = 0
        for i in range(0, len(x_val), batch_size):
            x_val_batch = x_val[i:i+batch_size]
            y_val_batch = y_val[i:i+batch_size]
            
            batch_val_loss = val_step(x_val_batch, y_val_batch)
            epoch_val_loss += batch_val_loss
            num_val_batches += 1
        
        epoch_val_loss /= num_val_batches
        val_loss_history.append(epoch_val_loss)

        print(f"Epoch {epoch+1} Train Loss: {epoch_train_loss:.4f} Validation Loss: {epoch_val_loss:.4f}")

    y_preds_concat = tf.concat(y_preds, axis=0).numpy()
    y_true_concat = tf.concat(y_true, axis=0).numpy()
    
    # print("shape y_preds_concat: ", y_preds_concat.shape)
    # print("shape y_true_concat: ", y_true_concat.shape)

    # Plotting in a 1x2 grid
    # fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    # fig.suptitle('Model Training Analysis', fontsize=16)

    # # Training and Validation Loss Curve
    # axs[0].plot(range(1, epochs+1), train_loss_history, label='Training Loss', color="black")
    # axs[0].plot(range(1, epochs+1), val_loss_history, label='Validation Loss', color="blue")
    # axs[0].scatter(range(1, epochs+1), train_loss_history, color="black")
    # axs[0].scatter(range(1, epochs+1), val_loss_history, color="blue")
    # axs[0].set_xlabel('Epochs')
    # axs[0].set_ylabel('Loss')
    # axs[0].set_title('Loss Curve')
    # axs[0].grid()
    # axs[0].legend()

    # # Predicted vs Actuals with RMSE
    # vals_each_side = 200
    # n = len(y_true_concat)
    # vals_per_epoch = int(n / epochs)

    # rows = np.concatenate((np.arange(int(vals_per_epoch/2), int(vals_per_epoch/2)+vals_each_side),  np.arange(n-vals_each_side, n) ))
    # cols_true = np.array([15,25,35])
    # cols_preds = map_range(cols_true, 0, 53, 0, 20)

    # y_true_plot = y_true_concat[np.ix_(rows, cols_true)]
    # y_preds_plot = y_preds_concat[np.ix_(rows, cols_preds)]

    # axs[1].plot(y_true_plot, label='True', color='black', alpha=0.6)
    # axs[1].plot(y_preds_plot, label='Predictions', color='blue', alpha=0.6)
    # axs[1].axvline(x=vals_each_side, color='r', linestyle='--', linewidth=2)
    # axs[1].set_xlabel('Time')
    # axs[1].set_ylabel('Values')
    # axs[1].set_title(f'Predicted vs Actuals')
    # axs[1].text(0.1, 0.05, f'Epoch 1', horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes, bbox=dict(facecolor='white', edgecolor='white'))
    # axs[1].text(0.9, 0.05, f'Last Epoch', horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes, bbox=dict(facecolor='white', edgecolor='white'))
    # legend_without_duplicate_labels(axs[1])

    # # Adjust layout to prevent overlap
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.savefig(f"Training_NN_{data_name}.pdf")
    # plt.show()

    return model

def one_step_quantile_prediction(X_input, Y_input, n_init, n_full, quantile = 0.5, already_correct_size = False, n_in_X = 5000):
    
    '''
    As input, this function should take the entire training set, and based on the last n_init observations,
    calculate residuals and coefficients for the quantile regression.

    '''

    assert n_init <= n_full - 2, "n_init must be less than n_full" # should it be equal == ? Only if we want one step prediction... 11/6-24.

    if type(X_input) == pd.DataFrame:
        X_input = X_input.to_numpy()

    if type(Y_input) == pd.Series or type(Y_input) == pd.DataFrame:
        Y_input = Y_input.to_numpy()

    n,m = X_input.shape
    
    print("X_input shape: ", X_input.shape)
    # X_input = X_input[:, [1,  3,  5,  7,  9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49] ] 


    
        
    # get shapes
    full_length, p = X_input.shape

    # if not already_correct_size:
    # get input ready
    X = X_input[:n_full, :].copy()
    Y = Y_input[:n_full]
    # else:
    #     X = X_input
    #     Y = Y_input
    #     n_full = len(Y)

    X_for_residuals = X[:n_init, :]
    Y_for_residuals = Y[:n_init]

    # plt.figure()
    # plt.plot(X_for_residuals, color = "grey", alpha = 0.2)
    # plt.plot(Y_for_residuals, color = "black")
    # plt.show()

    # save them for to be used in rq... X_for_residuals and Y_for_residuals
    # X_for_residuals.to_csv("X_for_residuals.csv") #only use if pd dataframe
    np.savetxt("X_for_residuals.csv", X_for_residuals, delimiter=",")
    np.savetxt("Y_for_residuals.csv", Y_for_residuals, delimiter=",")
    # Y_for_residuals.to_csv("Y_for_residuals.csv") # --||--

    # calculate residuals
    run_r_script("X_for_residuals.csv", "Y_for_residuals.csv", tau = quantile)
    # quantile_fit = model_QR.fit(X_for_residuals, Y_for_residuals)
    # Y_predict = quantile_fit.predict(X_for_residuals)
    # residuals = Y_for_residuals - Y_predict

    # Define a converter function to ignore the first column
    def ignore_first_column(s):
        return float(s)

    # Read the CSV file
    residuals = np.genfromtxt(
        'rq_fit_residuals.csv', 
        delimiter=',', 
        skip_header=1, 
        usecols=(1,),  # Only read the second column
        converters={0: ignore_first_column}  # Ignore the first column
    )



    # residuals = np.loadtxt("rq_fit_residuals.csv", delimiter=",", skip_rows = 1)

    beta_init = np.genfromtxt(
        'rq_fit_coefficients.csv', 
        delimiter=',', 
        skip_header=1, 
        usecols=(1,),  # Only read the second column
        converters={0: ignore_first_column}  # Ignore the first column
    )

    # beta_init = np.loadtxt("rq_fit_coef.csv", delimiter=",", skip_header = 1)

    print("len of beta_init: ", len(beta_init))
    # print(beta_init)
    print("There is: ", sum(residuals == 0), "zeros in residuals", "and", sum(abs(residuals) < 1e-8), "close to zeroes")
    print("p: ", p)

    # add 1s to beta_init to match length of p
    beta_init = np.append(beta_init, np.ones(p-len(beta_init)))
    
    r_init = set_n_closest_to_zero(arr = residuals , n = len(beta_init))

    print(sum(r_init==0), "r_init zeros")

    # get the data ready
    # print("X shape: ", X.shape, "Y shape: ", Y.shape, "random choice shape: ", np.random.choice([1,1], size=n_full).shape)
    X_full = np.column_stack((X, Y, np.random.choice([1,1], size=n_full)))
    IX = np.arange(p)
    Iy = p
    Iex = p + 1
    bins = np.array([-np.inf , np.inf]) # rolling, Currently not active, since n_in_bin = full length...
    # beta_init = quantile_fit.coef_
    tau = quantile
    n_in_bin = int(1.0*full_length)
    print("n_in_bin", n_in_bin)


    # call the function
    n_input = n_in_X
    N, BETA, GAIN, Ld, Rny, Mx, Re, CON1, T = rq_simplex_final(X_full, IX, Iy, Iex, r_init, beta_init, n_input, tau, bins , n_in_bin ) # here we set n_init to 5000, to see what happens...
    # find the actual prediction
    # print(BETA.shape, "BETA SHAPE")
    # print(X_input.shape, "X_input full shape")
    # print(X_input[(n_init+1):(n_full), :].shape, "X_input shape") 
    y_pred = np.sum((X_input[(n_input+2):(n_full), :] * BETA[1:,:]), axis = 1) # TODO WHETHER IT IS +1, or 2 here or minus, should def. be checked
    y_actual = Y_input[(n_input):(n_full-2)]
    print(y_pred.shape, "y_pred shape")
    print(y_actual.shape, "y_actual shape")
    # plt.figure()
    # plt.plot(y_pred)
    # plt.plot(y_actual)
    # plt.show()
    y_actual_quantile = np.quantile(y_actual, quantile)
    #print("Quantile: ", quantile, "y_actual_quantile: ", y_actual_quantile)
    # return the prediction, the actual value and the coefficients
    return y_pred, y_actual, BETA

def run_taqr(corrected_ensembles, actuals, quantiles, n_init, n_full, n_in_X):

    '''
    Run TAQR on corrected_ensembles, X, based on the actual values, y and the given quantiles.

    Parameters:
    corrected_ensembles: np.array, shape (n_timesteps, n_ensembles)
        The corrected ensembles to run TAQR on.
    actuals: np.array, shape (n_timesteps,)
        The actual values to run TAQR on.
    quantiles: list
        The quantiles to run TAQR for.
    n_init: int
        The number of initial timesteps to use for warm start.
    n_full: int
        The total number of timesteps to run TAQR for.
    n_in_X: int
        The number of timesteps to include in the design matrix.
    '''



    # Clean for NaNs
    actuals.iloc[np.isnan(actuals)] = 0

    taqr_results = []
    for q in quantiles:
        print("running TAQR for quantile: ", q)
        y_pred, _, _ = one_step_quantile_prediction(corrected_ensembles, actuals, n_init=n_init, n_full=n_full, quantile=q, already_correct_size = True, n_in_X = n_in_X)
        taqr_results.append(y_pred)

    return taqr_results

def pipeline(X, y, 
             name = "TEST",
             training_size = 0.8, 
             epochs = 100,
             timesteps_for_lstm = [0,1,2,6,12,24,48],
             **kwargs):
    '''
    The pipeline trains a LSTM network to correct the provided ensembles.
    It then runs the TAQR algorithm on the corrected ensembles to predict the observations, y, on the test set.

    Parameters:
    X: pd.DataFrame or np.array, shape (n_samples, n_features)
        The ensemble data to be corrected.
    y: pd.Series or np.array, shape (n_samples,)
        The observations to be predicted.
    name: str
        The name of the dataset.
    training_size: float
        The proportion of the data to be used for training.
    epochs: int
        The number of epochs to train the LSTM.
    timesteps_for_lstm: list
        The timesteps to use for the LSTM.
    '''

    # Get the data in the right format
    actuals = y
    ensembles = X
    X_y = np.concatenate((X, y.reshape(-1,1)), axis=1)
    if isinstance(y, pd.Series):
        idx = y.index
    elif isinstance(X, pd.DataFrame):
        idx = X.index
    else:
        idx = pd.RangeIndex(start=0, stop=len(y), step=1)

    train_size = int(training_size * len(actuals))
    ensembles = pd.DataFrame(ensembles, index=idx)
    ensembles.index = pd.to_datetime(ensembles.index, utc = False).tz_localize(None)
    actuals = pd.DataFrame(actuals, index=idx)
    actuals.index = pd.to_datetime(actuals.index, utc = False).tz_localize(None)
    common_index = ensembles.index.intersection(actuals.index)
    X_y = pd.DataFrame(X_y, index=idx)
    X_y.index = pd.to_datetime(X_y.index, utc = False).tz_localize(None)
    ensembles = ensembles.loc[common_index]
    actuals = actuals.loc[common_index]
    X_y = X_y.loc[common_index]
    

    print(ensembles)
    timesteps = timesteps_for_lstm
    Xs, X_Ys = create_dataset_for_lstm(ensembles, X_y, timesteps_for_lstm)

    if np.isnan(Xs).any():
        print("Xs has NaNs")
        Xs[np.isnan(Xs).any(axis=(1,2))] = 0
    if np.isnan(X_Ys).any():
        print("X_Ys has NaNs")
        X_Ys[np.isnan(X_Ys).any(axis=1)] = 0

    # Standardize Xs and X_Ys between 0 and 1
    XY_s_max_train = np.max(X_Ys[:train_size])
    XY_s_min_train = np.min(X_Ys[:train_size])

    X_Ys_scaled_train = (X_Ys[:train_size] - XY_s_min_train) / (XY_s_max_train - XY_s_min_train)
    Xs_scaled_train = (Xs[:train_size] - XY_s_min_train) / (XY_s_max_train - XY_s_min_train)

    # print("SHAPE OF X_Ys_scaled_train: ", X_Ys_scaled_train.shape)
    # print("SHAPE OF Xs_scaled_train: ", Xs_scaled_train.shape)
    validation_size = 100
    X_Ys_scaled_validation = (X_Ys[train_size:(train_size+validation_size)] - XY_s_min_train) / (XY_s_max_train - XY_s_min_train)
    Xs_scaled_validation = (Xs[train_size:(train_size+validation_size)] - XY_s_min_train) / (XY_s_max_train - XY_s_min_train)

    # print("SHAPE OF X_Ys_scaled_validation: ", X_Ys_scaled_validation.shape)
    # print("SHAPE OF Xs_scaled_validation: ", Xs_scaled_validation.shape)

    # print("min and max of X_Ys: ", np.min(X_Ys), np.max(X_Ys))
    # print("min and max of Xs: ", np.min(Xs), np.max(Xs))
    quantiles_lstm = np.linspace(0.05, 0.95,20)
    model = train_model_lstm(quantiles=quantiles_lstm, epochs=epochs, 
                                lr=1e-3, batch_size=50, 
                                x=tf.convert_to_tensor(Xs_scaled_train), 
                                y=tf.convert_to_tensor(X_Ys_scaled_train),
                                x_val = tf.convert_to_tensor(Xs_scaled_validation),
                                y_val = tf.convert_to_tensor(X_Ys_scaled_validation),
                                n_timesteps=timesteps,
                                data_name=f"{name}_LSTM_epochs_{epochs}")
        
    # save the model from tensorflow
    try:    
        import datetime as dt
        today = dt.datetime.today().strftime('%Y-%m-%d')
        model.save(f"Model_{name}_{epochs}_{today}.keras")
    except:
        model.save(f"Models_{name}_{epochs}.keras")

    
    # Transform data back and fourth
    min_Xs_test = np.min(Xs[:train_size])
    max_Xs_test = np.max(Xs[:train_size])
    Xs_scaled_test = (Xs[train_size:] - XY_s_min_train) / (XY_s_max_train - XY_s_min_train)
    corrected_ensembles = model(Xs_scaled_test)
    # Reverse transform back from 0 to 1
    corrected_ensembles = corrected_ensembles * (XY_s_max_train - XY_s_min_train) + XY_s_min_train
    actuals_out_of_sample = actuals[train_size:]
    actuals_out_of_sample = (actuals_out_of_sample) # - min_XY_s) / (max_XY_s - min_XY_s)
    test_idx = idx[train_size:]

    

    # Run the TAQR algorithm with the corrected ensembles for quantiles_taqr
    if kwargs.get("quantiles_taqr", None) is None:
        quantiles_taqr = [0.1, 0.3, 0.5, 0.7, 0.9]
    else:
        quantiles_taqr = kwargs.get("quantiles_taqr")
    
    n_full = len(actuals_out_of_sample)
    n_init = int(0.25*n_full)
    print("n_init, n_full: ",  n_init, n_full)

    corrected_ensembles = corrected_ensembles.numpy()
    corrected_ensembles = remove_zero_columns_numpy(corrected_ensembles)
    corrected_ensembles = remove_straight_line_outliers(corrected_ensembles)

    # Run the TAQR algorithm with the corrected ensembles LSTM
    n_in_X = n_init
    taqr_results = run_taqr(corrected_ensembles, actuals_out_of_sample, quantiles_taqr, n_init, n_full, n_in_X)
    actuals_out_of_sample = actuals_out_of_sample[(n_init+1):(n_full-1)]

    corrected_ensembles = corrected_ensembles[(n_init+1):(n_full-1)]
    idx_to_save = test_idx[(n_init+1):(n_full-1)]

    # Calculate scores (QSS, CRPS, Variogram) for the TAQR results
    # qss_scores = calculate_qss(actuals_out_of_sample, corrected_ensembles, quantiles_taqr)
    # crps_score =  calculate_crps(actuals_out_of_sample, corrected_ensembles)
    # variogram_score = variogram_score_R_multivariate(corrected_ensembles, actuals_out_of_sample)

    data_source = f"{name}"
    # Save the scores, corrected ensembles, TAQR results, and trained model
    today = dt.datetime.today().strftime('%Y-%m-%d')

    # np.save(f"results_{today}_{data_source}_qss_scores.npy", qss_scores)
    # np.save(f"results_{today}_{data_source}_crps_score.npy", crps_score)
    # np.save(f"results_{today}_{data_source}_variogram_score.npy", variogram_score)

    np.save(f"results_{today}_{data_source}_actuals_out_of_sample.npy", actuals_out_of_sample)

    # Save the corrected ensembles
    
    df_corrected_ensembles = pd.DataFrame(corrected_ensembles, index=idx_to_save)
    df_corrected_ensembles.to_csv(f"results_{today}_{data_source}_corrected_ensembles.csv")

    # Save the TAQR results
    np.save(f"results_{today}_{data_source}_taqr_results.npy", taqr_results)
