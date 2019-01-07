#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pdb


def trainer(data, model, args=None, logger=None):
    '''
    Parameters
    ----------
    data: (np.array, np.array), shape ((N, D) (N,))
        X, Y
    model: sklearn model
    args: dict
    logger: logger object

    Returns
    -------
    model: trained model
    params: dict
    '''
    X, Y = data
    if X.ndim < 2:
        X = X.reshape(-1, 1)

    model.fit(X, Y)

    params = model.get_params()

    return model, params
