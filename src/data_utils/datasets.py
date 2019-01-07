#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pdb


def get_3lines_2_data(file):
    '''
    Get 3lines_2 data.

    Parameters
    ----------
    file: str
        path to data file

    Returns
    -------
    X: np.array[float]
    Y: np.array[float]
    '''
    data = np.loadtxt(file)
    return data[:, 0], data[:, 1]
