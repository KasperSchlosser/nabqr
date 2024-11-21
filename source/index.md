<!-- .. nabqr-RTD documentation master file, created by
   sphinx-quickstart on Wed Nov 20 09:07:39 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive. -->

nabqr documentation
=======================

NABQR is a method for sequential error-corrections tailored for wind power forecast in Denmark.

- **Free software**: MIT license  
- **Documentation**: [NABQR Documentation](https://nabqr.readthedocs.io)

## Getting Started
See `test_file.py` for an example of how to use the package.

## Main functions
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
- - Documentation

<!-- .. toctree::
   :maxdepth: 2
   :caption: Contents:
-->
