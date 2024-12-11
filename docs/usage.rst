=====
Usage
=====

To use NABQR in a project::

    import nabqr as nq


Minimal example::

    nq.pipeline(
        X_data,
        y_actuals,
        quantiles_taqr=[0.1, 0.25, 0.5, 0.75, 0.9],
    )
