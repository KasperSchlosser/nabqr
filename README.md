# NABQR

[![PyPI Version](https://img.shields.io/pypi/v/nabqr.svg)](https://pypi.python.org/pypi/nabqr)
[![Documentation Status](https://readthedocs.org/projects/nabqr/badge/?version=latest)](https://nabqr.readthedocs.io/en/latest/?version=latest)

NABQR is a method for sequential error-corrections tailored for wind power forecast in Denmark.

- **Free software**: MIT license  
- **Documentation**: [NABQR Documentation](https://nabqr.readthedocs.io)

## Getting Started
See `test_file.py` for an example of how to use the package.

### Main functions
```python
from nabqr.src.functions import pipeline
```

```python
pipeline(X, y, 
             name = "TEST",
             training_size = 0.8, 
             epochs = 100,
             timesteps_for_lstm = [0,1,2,6,12,24,48],
             **kwargs):
```

The pipeline trains a LSTM network to correct the provided ensembles.
It then runs the TAQR algorithm on the corrected ensembles to predict the observations, y, on the test set.

**Parameters:**

- **X**: `pd.DataFrame` or `np.array`, shape `(n_samples, n_features)`
  - The ensemble data to be corrected.
- **y**: `pd.Series` or `np.array`, shape `(n_samples,)`
  - The observations to be predicted.
- **name**: `str`
  - The name of the dataset.
- **training_size**: `float`
  - The proportion of the data to be used for training.
- **epochs**: `int`
  - The number of epochs to train the LSTM.
- **timesteps_for_lstm**: `list`
  - The timesteps to use for the LSTM.


The pipeline trains a LSTM network to correct the provided ensembles



## Features

- TODO
- - Project description
- - Installation instructions
- - Package requirements
- - Documentation


## Credits

This package was partially created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [`audreyr/cookiecutter-pypackage`](https://github.com/audreyr/cookiecutter-pypackage) project template.