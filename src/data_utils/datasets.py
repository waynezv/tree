#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pdb


def get_3lines_2_data(file, num_load=None):
    '''
    Get 3lines_2 data.

    Parameters
    ----------
    file: str
        path to data file
    num_load: int
        number of samples to load

    Returns
    -------
    X: np.array[float]
    Y: np.array[float]
    '''
    data = np.loadtxt(file)

    if num_load is not None:
        np.random.seed(1111)
        N = data.shape[0]
        idx = np.random.permutation(N)
        data = np.take(data, idx[:num_load], axis=0)

    return data[:, 0], data[:, 1]
