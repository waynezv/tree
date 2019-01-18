#!/usr/bin/env python
# encoding: utf-8

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
from tree import Tree

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
opts = args['CCPP']
X_data, Y_data = get_concrete_data(os.path.join(opts['path'], opts['file']))

X_train, X_test, Y_train, Y_test = train_test_split(
    X_data, Y_data, test_size=opts['test_size'],
    random_state=opts['random_state'])

if X_train.ndim < 2:
    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)

# Model
logger.info('Initializing model')
tree_opts = args['beta']
tree = Tree(logger=logger, args=tree_opts)

# Train
logger.info('Training')
tree.train(X_train, Y_train)
logger.info('=' * 50)
logger.info('t_best:')
logger.info(tree.t_best_list)
logger.info('Q_best:')
logger.info(tree.Q_best_history)

# Test
logger.info('Testing')
if tree_opts['prediction'] == 'soft':
    Yp_train = tree.predict(X_train)
    Yp_test = tree.predict(X_test)

elif tree_opts['prediction'] == 'hard':
    Yp_train = tree.predict_hard(X_train)
    Yp_test = tree.predict_hard(X_test)

# Metrics
mae_train = metrics.mean_absolute_error(Y_train, Yp_train)
rmse_train = np.sqrt(metrics.mean_squared_error(Y_train, Yp_train))
mae_test = metrics.mean_absolute_error(Y_test, Yp_test)
rmse_test = np.sqrt(metrics.mean_squared_error(Y_test, Yp_test))

logger.info('*' * 50)
logger.info('Train MAE = {:.4f}  RMSE = {:.4f}'.
            format(mae_train, rmse_train))
logger.info('Test MAE = {:.4f}  RMSE = {:.4f}'.
            format(mae_test, rmse_test))

logger.info('=' * 50)
logger.info('Parameters & settings')
logger.info(args)
