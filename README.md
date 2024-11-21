# NABQR

[![PyPI Version](https://img.shields.io/pypi/v/nabqr.svg)](https://pypi.python.org/pypi/nabqr)
[![Documentation Status](https://readthedocs.org/projects/nabqr/badge/?version=latest)](https://nabqr.readthedocs.io/en/latest/?version=latest)

- **Free software**: MIT license  
- **Documentation**: [NABQR Documentation](https://nabqr.readthedocs.io)

nabqr documentation
=======================

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Main functions](#main-functions)
- [Test file](#test-file)
- [Notes](#notes)
- [Credits](#credits)
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
---

## Getting Started

### Installation
`pip install nabqr`

Then see the [Test file](#test-file) section for an example of how to use the package.

## Main functions
### Pipeline
```python
from nabqr.src.functions import pipeline
```

```python
pipeline(X, y, 
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
The pipeline saves the following outputs:

- **Actuals Out of Sample**: 
  - File: `results_<today>_<data_source>_actuals_out_of_sample.npy`
  - Description: Contains the actual observations that are out of the sample.

- **Corrected Ensembles**: 
  - File: `results_<today>_<data_source>_corrected_ensembles.csv`
  - Description: A CSV file containing the corrected ensemble data.

- **TAQR Results**: 
  - File: `results_<today>_<data_source>_taqr_results.npy`
  - Description: Contains the results from the Time-Adaptive Quantile Regression (TAQR).

Note: `<today>` is the current date in the format `YYYY-MM-DD`, and `<data_source>` is the name of the dataset.


The pipeline trains a LSTM network to correct the provided ensembles and then runs the TAQR algorithm on the corrected ensembles to predict the observations, y, on the test set.

### Time-Adaptive Quantile Regression
nabqr also include a time-adaptive quantile regression model, which can be used independently of the pipeline.
```python
from nabqr.src.functions import run_taqr
```
```python
run_taqr(corrected_ensembles, actuals, quantiles, n_init, n_full, n_in_X)
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


## Notes

- TODO
- - Project description
- - Installation instructions


## Test file 
Here we introduce the function `simulate_correlated_ar1_process`, which can be used to simulate multivariate AR data. The entire file can be run by `from nabqr import nabqr`.


```python
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
```

We provide an overview of the shapes for this test file:
```python
simulated_data.shape: (5000, 33)
actuals.shape: (5000,)
m: 33
len(quantiles_taqr): 7
```


## Credits

This package was partially created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [`audreyr/cookiecutter-pypackage`](https://github.com/audreyr/cookiecutter-pypackage) project template.