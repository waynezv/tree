#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import json
import logging
import logging.config
import numpy as np
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
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
opts = args['3lines_2']
X_train, Y_train = get_3lines_2_data(
    os.path.join(opts['path'], opts['train']['file'])
)
X_test, Y_test = get_3lines_2_data(
    os.path.join(opts['path'], opts['test']['file'])
)

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
    #  Yp_train = tree.predict(X_train)
    #  Yp_test = tree.predict(X_test)

    Yp_train, yh_xz_train, qz_x_train = tree.predict(X_train)
    Yp_test, yh_xz_test, qz_x_test = tree.predict(X_test)

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

# Plot data
fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(111)
# ax1.scatter(X_train, Y_train, s=2, c='b', marker='.')
# ax1.scatter(X_train, Yp_train, s=2, c='r', marker='+')

# shaded background
ax1.scatter(X_train, Y_train, s=2, c='b', marker='.', alpha=0.3)

# plot predictions of each leaf predictor
# with color shaded probability confidence
for yh, qz in zip(yh_xz_train, qz_x_train):
    # scale proba for best color
    cc = (qz - np.min(qz)) / (np.max(qz) - np.min(qz))
    cc += 0.1
    cc /= 0.1
    cc = 1 - cc
    cc[0] = 1
    ax1.scatter(X_train, yh, s=0.5, c=cc, marker='.', cmap='inferno')

ax1.axis('tight')
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
# ax1.legend(('original', 'pred'), loc='best', fontsize=11)
plt.tight_layout()
plt.savefig('3lines_2_tree_train.pdf')

fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(111)
# ax1.scatter(X_test, Y_test, s=2, c='b', marker='.')
# ax1.scatter(X_test, Yp_test, s=2, c='r', marker='+')

# shaded background
ax1.scatter(X_test, Y_test, s=2, c='b', marker='.', alpha=0.3)

# plot predictions of each leaf predictor
# with color shaded probability confidence
for yh, qz in zip(yh_xz_test, qz_x_test):
    # scale proba for best color
    cc = (qz - np.min(qz))/(np.max(qz) - np.min(qz))
    cc += 0.1
    cc /= 0.1
    cc = 1 - cc
    cc[0] = 1
    ax1.scatter(X_test, yh, s=0.5, c=cc, marker='.', cmap='inferno')

ax1.axis('tight')
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
# ax1.legend(('original', 'pred'), loc='best', fontsize=11)
plt.tight_layout()
plt.savefig('3lines_2_tree_test.pdf')

logger.info('=' * 50)
logger.info('Parameters & settings')
logger.info(args)
