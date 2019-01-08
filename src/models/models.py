#!/usr/bin/env python
# encoding: utf-8

import sklearn
import pdb


def get_linear(params):
    '''
    Parameters
    ----------
    params: dict

    Returns
    -------
    model: sklearn model
    '''
    from sklearn.linear_model import LinearRegression

    model = LinearRegression(**params)

    return model


def get_svr(params):
    '''
    Parameters
    ----------
    params: dict

    Returns
    -------
    model: sklearn model
    '''
    from sklearn.svm import SVR

    model = SVR(**params)

    return model


def get_decision_tree(params):
    '''
    Parameters
    ----------
    params: dict

    Returns
    -------
    model: sklearn model
    '''
    from sklearn.tree import DecisionTreeRegressor

    model = DecisionTreeRegressor(**params)

    return model
