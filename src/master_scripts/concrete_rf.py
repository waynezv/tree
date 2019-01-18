#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
import logging.config
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
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
from datasets import get_concrete_data
from models import get_random_forest
from generic_trainers import trainer
from generic_predictors import predictor


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
opts = args['concrete']
X_data, Y_data = get_concrete_data(os.path.join(opts['path'], opts['file']))

X_train, X_test, Y_train, Y_test = train_test_split(
    X_data, Y_data, test_size=opts['test_size'],
    random_state=opts['random_state'])

train_opts = args['train']['random_forest']
if train_opts['parameter_tuning'] is True:  # tuning
    from sklearn.model_selection import GridSearchCV

    if X_train.ndim < 2:
        X_train = X_train.reshape(-1, 1)

    model = get_random_forest(train_opts['preset_model_params'])

    grid = GridSearchCV(model, **train_opts['tuning_settings'])
    grid.fit(X_train, Y_train)

    logger.info('=' * 50)
    logger.info('Best parameters: ')
    logger.info(grid.best_params_)

else:  # normal training

    # Model
    logger.info('Building model')
    model = get_random_forest(args['random_forest'])

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

logger.info('=' * 50)
logger.info('Parameters & settings')
logger.info(args)
