#!/usr/bin/env python
# encoding: utf-8

import sklearn
import pdb


def get_svr(params):
    '''
    Parameters
    ----------
    params: dict

    Returns
    -------
    model: sklearn.svm.SVR
    '''
    from sklearn.svm import SVR

    model = SVR(**params)

    return model
