# NABQR

[![PyPI Version](https://img.shields.io/pypi/v/nabqr.svg)](https://pypi.python.org/pypi/nabqr)
[![Documentation Status](https://readthedocs.org/projects/nabqr/badge/?version=latest)](https://nabqr.readthedocs.io/en/latest/?version=latest)

- **Free software**: MIT license  
- **Documentation**: [NABQR Documentation](https://nabqr.readthedocs.io)

README for nabqr package
=======================

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Main functions](#main-functions)
  - [Pipeline](#pipeline)
  - [Time-Adaptive Quantile Regression](#time-adaptive-quantile-regression)
  - [LSTM Network](#lstm-network)
- [Demonstration: Test file](#test-file)
- [Credits/Copyright](#credits)
---

## Introduction

This section provides an overview of the project. Discuss the goals, purpose, and high-level summary here.


NABQR is a method for sequential error-corrections tailored for wind power forecast in Denmark.

The method is based on the paper: *Sequential methods for Error Corrections in Wind Power Forecasts*, with the following abstract:
> Wind power is a rapidly expanding renewable energy source and is set for continued growth in the future. This leads to parts of the world relying on an inherently volatile energy source.
> Efficient operation of such systems requires reliable probabilistic forecasts of future wind power production to better manage the uncertainty that wind power bring. These forecasts provide critical insights, enabling wind power producers and system operators to maximize the economic benefits of renewable energy while minimizing its potential adverse effects on grid stability.
> This study introduces sequential methods to correct errors in power production forecasts derived from numerical weather predictions. 
> We introduce Neural Adaptive Basis for (Time-Adaptive) Quantile Regression (NABQR), a novel approach that combines neural networks with Time-Adaptive Quantile Regression (TAQR) to enhance the accuracy of wind power production forecasts. 
> First, NABQR corrects power production ensembles using neural networks.
> Our study identifies Long Short-Term Memory networks as the most effective architecture for this purpose.
> Second, TAQR is applied to the corrected ensembles to obtain optimal median predictions along with quantile descriptions of the forecast density. 
> The method achieves substantial improvements upwards of 40% in mean absolute terms. Additionally, we explore the potential of this methodology for applications in energy trading.
> The method is available as an open-source Python package to support further research and applications in renewable energy forecasting.


- **Free software**: MIT license  
- **Documentation**: [NABQR Documentation](https://nabqr.readthedocs.io)
  - [API Documentation](https://nabqr.readthedocs.io/en/latest/api.html#main-pipeline)
---

## Getting Started

### Installation
`pip install nabqr`

Then see the [Test file](#test-file) section for an example of how to use the package.

## Main functions
### Pipeline
```python
import nabqr as nq
```

```python
nq.pipeline(X, y, 
             name = "TEST",
             training_size = 0.8, 
             epochs = 100,
             timesteps_for_lstm = [0,1,2,6,12,24,48],
             **kwargs)
```

The pipeline trains a LSTM network to correct the provided ensembles.
It then runs the TAQR algorithm on the corrected ensembles to predict the observations, y, on the test set.

**Parameters:**

- **X**: `pd.DataFrame` or `np.array`, shape `(n_timesteps, n_ensembles)`
  - The ensemble data to be corrected.
- **y**: `pd.Series` or `np.array`, shape `(n_timesteps,)`
  - The observations to be predicted.
- **name**: `str`
  - The name of the dataset.
- **training_size**: `float`
  - The proportion of the data to be used for training.
- **epochs**: `int`
  - The number of epochs to train the LSTM.
- **timesteps_for_lstm**: `list`
  - The timesteps to use for the LSTM.

**Output:**
The pipeline saves the following outputs and also returns them:

- **Corrected Ensembles**: 
  - File: `results_<today>_<data_source>_corrected_ensembles.csv`
  - A CSV file containing the corrected ensemble data.

- **TAQR Results**: 
  - File: `results_<today>_<data_source>_taqr_results.npy`
  - Contains the results from the Time-Adaptive Quantile Regression (TAQR).

- **Actuals Out of Sample**: 
  - File: `results_<today>_<data_source>_actuals_out_of_sample.npy`
  - Contains the actual observations that are out of the sample.

- **BETA Parameters**: 
  - File: `results_<today>_<data_source>_BETA_output.npy`
  - Contains the BETA parameters from the TAQR.

- **Ensembles**: 
  - Contains the original ensembles.


Note: `<today>` is the current date in the format `YYYY-MM-DD`, and `<data_source>` is the name of the dataset.

The pipeline trains a LSTM network to correct the provided ensembles and then runs the TAQR algorithm on the corrected ensembles to predict the observations, y, on the test set.

### Time-Adaptive Quantile Regression
nabqr also include a time-adaptive quantile regression model, which can be used independently of the pipeline.
```python
import nabqr as nq
```
```python
nq.run_taqr(corrected_ensembles, actuals, quantiles, n_init, n_full, n_in_X)
```

Run TAQR on `corrected_ensembles`, `X`, based on the actual values, `y`, and the given quantiles.

**Parameters:**

- **corrected_ensembles**: `np.array`, shape `(n_timesteps, n_ensembles)`
  - The corrected ensembles to run TAQR on.
- **actuals**: `np.array`, shape `(n_timesteps,)`
  - The actual values to run TAQR on.
- **quantiles**: `list`
  - The quantiles to run TAQR for.
- **n_init**: `int`
  - The number of initial timesteps to use for warm start.
- **n_full**: `int`
  - The total number of timesteps to run TAQR for.
- **n_in_X**: `int`
  - The number of timesteps to include in the design matrix.

### LSTM Network
The LSTM Network is trained on the ensembles and the actual values.

The function `train_model_lstm` is used to train the LSTM Network and can be called directly from the `functions.py` module. 

To build the LSTM network we use the `QuantileRegressionLSTM` class.

For more information, please refer to the [documentation on ReadTheDocs](https://nabqr.readthedocs.io/en/latest/).


## Test file 
Here we introduce the function `simulate_correlated_ar1_process`, which can be used to simulate multivariate AR data. The functions uses the `build_ar1_covariance` function to build the covariance matrix for the AR(1) process. The entire file can be run by 
```python
import nabqr as nq
nq.run_nabqr_pipeline(...)
# or
from nabqr import run_nabqr_pipeline
run_nabqr_pipeline(...)
```
The entire `run_nabqr_pipeline` function is provided below:
```python
def run_nabqr_pipeline(
    n_samples=2000,
    phi=0.995,
    sigma=8,
    offset_start=10,
    offset_end=500,
    offset_step=15,
    correlation=0.8,
    data_source="NABQR-TEST",
    training_size=0.7,
    epochs=20,
    timesteps=[0, 1, 2, 6, 12, 24],
    quantiles=[0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99],
    X=None,
    actuals=None,
):
    """
    Run the complete NABQR pipeline, which may include data simulation, model training,
    and visualization. The user can either provide pre-computed inputs (X, actuals)
    or opt to simulate data if both are not provided.

    Parameters
    ----------
    n_samples : int, optional
        Number of time steps to simulate if no data provided, by default 5000.
    phi : float, optional
        AR(1) coefficient for simulation, by default 0.995.
    sigma : float, optional
        Standard deviation of noise for simulation, by default 8.
    offset_start : int, optional
        Start value for offset range, by default 10.
    offset_end : int, optional
        End value for offset range, by default 500.
    offset_step : int, optional
        Step size for offset range, by default 15.
    correlation : float, optional
        Base correlation between dimensions, by default 0.8.
    data_source : str, optional
        Identifier for the data source, by default "NABQR-TEST".
    training_size : float, optional
        Proportion of data to use for training, by default 0.7.
    epochs : int, optional
        Number of epochs for model training, by default 100.
    timesteps : list, optional
        List of timesteps to use for LSTM, by default [0, 1, 2, 6, 12, 24].
    quantiles : list, optional
        List of quantiles to predict, by default [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99].
    X : array-like, optional
        Pre-computed input features. If not provided along with `actuals`, the function
        will prompt to simulate data.
    actuals : array-like, optional
        Pre-computed actual target values. If not provided along with `X`, the function
        will prompt to simulate data.
    simulation_type : str, optional
        Type of simulation to use, by default "ar1". "sde" is more advanced and uses a SDE model and realistic.
    visualize : bool, optional
        Determines if any visual elements will be plotted to the screen or saved as figures.
    taqr_limit : int, optional
        The lookback limit for the TAQR model, by default 5000.
    save_files : bool, optional
        Determines if any files will be saved, by default True. Note: the R-file needs to save some .csv files to run properly.

    Returns
    -------
    tuple
        A tuple containing:
        - corrected_ensembles: pd.DataFrame
            The corrected ensemble predictions.
        - taqr_results: list of numpy.ndarray
            The TAQR results.
        - actuals_output: list of numpy.ndarray
            The actual output values.
        - BETA_output: list of numpy.ndarray
            The BETA parameters.
        - scores: pd.DataFrame
            The scores for the predictions and original/corrected ensembles.

    Raises
    ------
    ValueError
        If user opts not to simulate data when both X and actuals are missing.
    """
    # If both X and actuals are not provided, ask user if they want to simulate
    if X is None or actuals is None:
        if X is not None or actuals is not None:
            raise ValueError("Either provide both X and actuals, or none at all.")
        choice = (
            input(
                "X and actuals are not provided. Do you want to simulate data? (y/n): "
            )
            .strip()
            .lower()
        )
        if choice != "y":
            raise ValueError(
                "Data was not provided and simulation not approved. Terminating function."
            )

        # Generate offset and correlation matrix for simulation
        offset = np.arange(offset_start, offset_end, offset_step)
        m = len(offset)
        corr_matrix = correlation * np.ones((m, m)) + (1 - correlation) * np.eye(m)

        # Generate simulated data
        # Check if simulation_type is valid
        if simulation_type not in ["ar1", "sde"]:
            raise ValueError("Invalid simulation type. Please choose 'ar1' or 'sde'.")
        if simulation_type == "ar1":    
            X, actuals = simulate_correlated_ar1_process(
                n_samples, phi, sigma, m, corr_matrix, offset, smooth=5
            )
        elif simulation_type == "sde":
            initial_params = {
                    'X0': 0.6,
                    'theta': 0.77,
                    'kappa': 0.12,        # Slower mean reversion
                    'sigma_base': 1.05,  # Lower base volatility
                    'alpha': 0.57,       # Lower ARCH effect
                    'beta': 1.2,        # High persistence
                    'lambda_jump': 0.045, # Fewer jumps
                    'jump_mu': 0.0,     # Negative jumps
                    'jump_sigma': 0.1    # Moderate jump size variation
                }
            # Check that initial parameters are within bounds
            bounds = get_parameter_bounds()
            for param, value in initial_params.items():
                lower_bound, upper_bound = bounds[param]
                if not (lower_bound <= value <= upper_bound):
                    print(f"Initial parameter {param}={value} is out of bounds ({lower_bound}, {upper_bound})")
                    if value < lower_bound:
                        initial_params[param] = lower_bound
                    else:
                        initial_params[param] = upper_bound
            
            t, actuals, X = simulate_wind_power_sde(
                initial_params, T=n_samples, dt=1.0
            )



        # Plot the simulated data with X in shades of blue and actuals in bold black
        plt.figure(figsize=(10, 6))
        cmap = plt.cm.Blues
        num_series = X.shape[1] if X.ndim > 1 else 1
        colors = [cmap(i) for i in np.linspace(0.3, 1, num_series)]  # Shades of blue
        if num_series > 1:
            for i in range(num_series):
                plt.plot(X[:, i], color=colors[i], alpha=0.7)
        else:
            plt.plot(X, color=colors[0], alpha=0.7)
        plt.plot(actuals, color="black", linewidth=2, label="Actuals")
        plt.title("Simulated Data")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

    # Run the pipeline
    corrected_ensembles, taqr_results, actuals_output, BETA_output, X_ensembles = pipeline(
        X,
        actuals,
        data_source,
        training_size=training_size,
        epochs=epochs,
        timesteps_for_lstm=timesteps,
        quantiles_taqr=quantiles,
        limit=taqr_limit,
        save_files = save_files
    )

    # Get today's date for file naming
    today = dt.datetime.today().strftime("%Y-%m-%d")

    # Visualize results
    if visualize:
        visualize_results(actuals_output, taqr_results, f"{data_source} example")

    # Calculate scores
    scores = calculate_scores(
        actuals_output,
        taqr_results,
        X_ensembles,
        corrected_ensembles,
        quantiles,
        data_source,
        plot_reliability=True,
        visualize = visualize
    )

    return corrected_ensembles, taqr_results, actuals_output, BETA_output, scores

```

We provide an overview of the shapes for this test file:
```python
actuals.shape: (n_samples,) # 2000
m: 1 + (offset_end - offset_start) // offset_step # 33
simulated_data.shape: (n_samples, m) # (2000, 33)
len(quantiles_taqr): 7
```

## Requirements

- Python 3.10 or later
- icecream, matplotlib, numpy<2.0.0, pandas, properscoring, rich, SciencePlots, scikit_learn, scipy, tensorflow, tensorflow_probability, torch, typer, sphinx_rtd_theme, myst_parser, tf_keras
- R with the following packages: quantreg, readr, SparseM (implicitly called)

## Credits/Copyright
Copyright © 2024 Technical University of Denmark

This version of the software was developed by Bastian Schmidt Jørgensen as a Research Assistant at the Department of Dynamical Systems, DTU Compute.


This package was partially created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [`audreyr/cookiecutter-pypackage`](https://github.com/audreyr/cookiecutter-pypackage) project template.