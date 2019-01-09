#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
import logging.config
import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt
import pdb
for path in [
        'data_utils',
        'models',
        'learners'
]:
    sys.path.append(
        os.path.join(
            os.path.dirname(os.path.dirname(
                os.path.realpath(__file__))),
            path))
from datasets import get_3lines_2_data
from models import get_svr
from generic_trainers import trainer
from generic_predictors import predictor


def params_tune(C, Gamma,
                X_train, Y_train,
                X_valid, Y_valid,
                args, logger):
    '''
    Tune parameters (C, gamma).

    Parameters
    ----------
    C: List[float]
    Gamma: List[float]
    X_*: np.array[float]
    Y_*: np.array[float]
    args: dict
    logger: logger object
    '''
    opts = args['svr']

    mae_best = 1e9

    i_best = 0
    j_best = 0
    i = 0
    for c in C:
        j = 0
        for gm in Gamma:
            logger.info('*' * 50)
            logger.info('C = {:.4f}  Gamma = {:.4f}'.
                        format(c, gm))

            model_params = {"kernel": opts['kernel'],
                            "gamma": gm,
                            "C": c}
            model = get_svr(model_params)

            model, params = trainer((X_train, Y_train), model)

            Yp_train = predictor(X_train, model)
            Yp_valid = predictor(X_valid, model)

            mae_train = metrics.mean_absolute_error(Y_train, Yp_train)
            mae_valid = metrics.mean_absolute_error(Y_valid, Yp_valid)
            rmse_train = np.sqrt(metrics.mean_squared_error(Y_train, Yp_train))
            rmse_valid = np.sqrt(metrics.mean_squared_error(Y_valid, Yp_valid))

            if mae_valid < mae_best:
                mae_best = mae_valid
                i_best = i
                j_best = j

            logger.info('Train MAE = {:.4f}  RMSE = {:.4f}'.
                        format(mae_train, rmse_train))
            logger.info('Valid MAE = {:.4f}  RMSE = {:.4f}'.
                        format(mae_valid, rmse_valid))
            j += 1

        i += 1

    logger.info('=' * 50)
    c_best = C[i_best]
    gm_best = Gamma[j_best]
    logger.info('Best Valid MAE = {:.4f}  with C = {:.4f} Gamma = {:.4f}'.
                format(mae_best, c_best, gm_best))


# Parse arguments
if len(sys.argv) < 2:
    print('python {} configure.json'.format(sys.argv[0]))
    sys.exit(-1)

with open(sys.argv[1], 'r') as f:
    args = json.load(f)

np.random.seed(args['random_seed'])

# Log
if not os.path.exists(args['log_dir']):
    os.makedirs(args['log_dir'])

log_file = os.path.join(args['log_dir'], args['log_file'])
if os.path.isfile(log_file):  # remove existing log
    os.remove(log_file)

logging.config.dictConfig(args['log'])  # setup logger
logger = logging.getLogger('main')

# Data
logger.info('Loading & preparing data')
opts = args['3lines_2']
X_train, Y_train = get_3lines_2_data(
    os.path.join(opts['path'], opts['train']['file'])
)
X_test, Y_test = get_3lines_2_data(
    os.path.join(opts['path'], opts['test']['file'])
)

train_opts = args['train']['svr']
if train_opts['parameter_tuning'] is True:  # tuning

    N = X_train.shape[0]
    num_train = int(np.floor(N * train_opts['valid_split']))
    idx = list(range(N))
    X_train_new = np.take(X_train, idx[:num_train], axis=0)
    Y_train_new = np.take(Y_train, idx[:num_train], axis=0)
    X_valid = np.take(X_train, idx[num_train:], axis=0)
    Y_valid = np.take(Y_train, idx[num_train:], axis=0)

    params_tune(train_opts['C'], train_opts['Gamma'],
                X_train_new, Y_train_new,
                X_valid, Y_valid,
                args, logger)

else:  # normal training

    # Model
    logger.info('Building model')
    model = get_svr(args['svr'])

    # Train
    logger.info('Training')
    model, params = trainer((X_train, Y_train), model)

    # Test
    logger.info('Testing')
    Yp_train = predictor(X_train, model)
    Yp_test = predictor(X_test, model)

    # Metrics
    mae_train = metrics.mean_absolute_error(Y_train, Yp_train)
    mae_test = metrics.mean_absolute_error(Y_test, Yp_test)
    rmse_train = np.sqrt(metrics.mean_squared_error(Y_train, Yp_train))
    rmse_test = np.sqrt(metrics.mean_squared_error(Y_test, Yp_test))

    # Log
    logger.info('*' * 50)
    logger.info('Train MAE = {:.4f}  RMSE = {:.4f}'.
                format(mae_train, rmse_train))
    logger.info('Test MAE = {:.4f}  RMSE = {:.4f}'.
                format(mae_test, rmse_test))

    # Plot data
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(111)
    ax1.scatter(X_train, Y_train, s=2, c='b', marker='.')
    ax1.scatter(X_train, Yp_train, s=2, c='r', marker='+')
    ax1.axis('tight')
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.legend(('original', 'pred'), loc='best', fontsize=11)
    plt.tight_layout()
    #  plt.savefig('3lines_2_svr_train.pdf')

    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(111)
    ax1.scatter(X_test, Y_test, s=2, c='b', marker='.')
    ax1.scatter(X_test, Yp_test, s=2, c='r', marker='+')
    ax1.axis('tight')
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.legend(('original', 'pred'), loc='best', fontsize=11)
    plt.tight_layout()
    #  plt.savefig('3lines_2_svr_test.pdf')

logger.info('=' * 50)
logger.info('Parameters & settings')
logger.info(args)
