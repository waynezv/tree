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

logger.info('Initializing model')
tree_opts = args['beta']
tree = Tree(logger=logger, args=tree_opts)

logger.info('Training')
tree.train(X_train, Y_train)
logger.info('=' * 50)
logger.info('t_best:')
logger.info(tree.t_best_list)
logger.info('Q_best:')
logger.info(tree.Q_best_history)
logger.info('=' * 50)

Yp_train = tree.predict(X_train)
Yp_test = tree.predict(X_test)
mae_train = metrics.mean_absolute_error(Y_train, Yp_train)
rmse_train = np.sqrt(metrics.mean_squared_error(Y_train, Yp_train))
mae_test = metrics.mean_absolute_error(Y_test, Yp_test)
rmse_test = np.sqrt(metrics.mean_squared_error(Y_test, Yp_test))
print('Train: ', mae_train, rmse_train)
print('Test: ', mae_test, rmse_test)

pdb.set_trace()
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
plt.savefig('3lines_2_tree_train.pdf')

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
plt.savefig('3lines_2_tree_test.pdf')

pdb.set_trace()
