#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
import logging.config
import numpy as np
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(
            os.path.realpath(__file__))),
        'data_utils'))
import pdb


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
if os.path.isfile(log_file):
    os.remove(log_file)

logging.config.dictConfig(args['log'])
logger = logging.getLogger('main')


# Main

# Data
logger.info('Loading & preparing data')


# Model
logger.info('Building model')

# Train
logger.info('Training')
train_losses, test_losses, train_errors, test_errors = \
    trainer(train_loader, test_loader, model, filnet,
            args, logger, run_id=i)

# Log
logger.info('*' * 50)
logger.info('Fold = {}'.format(i))

logger.info('Train loss = {:.4f}  error = {:.4f}'.
            format(loss_best, err_best))

logger.info('Test loss = {:.4f}  error = {:.4f}'.
            format(loss_best, err_best))
logger.info('*' * 50)


logger.info('Train loss = {:.4f}  error = {:.4f}'.
        format(loss_avg, err_avg))

logger.info('Test loss = {:.4f}  error = {:.4f}'.
        format(loss_avg, err_avg))

logger.info('=' * 50)
logger.info('Parameters & settings')
logger.info(args)
