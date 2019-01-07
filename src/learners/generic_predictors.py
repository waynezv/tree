#!/usr/bin/env python
# encoding: utf-8

import pdb


def predictor(X, model, args=None, logger=None):
    '''
    Parameters
    ----------
    X: np.array[float], shape (N, D)
    model: sklearn model
    args: dict
    logger: logger object

    Returns
    -------
    Y_pred: np.array[float], shape (N,)
    '''
    if X.ndim < 2:
        X = X.reshape(-1, 1)

    Y_pred = model.predict(X)

    return Y_pred
