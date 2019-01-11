#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import json
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

tree_opts = args['beta']
tree = Tree(tree_opts)

tree.grow_subtree(tree.root)
tree.nodes[1].convert_leaf_node()
tree.grow_subtree(tree.nodes[1])
pdb.set_trace()
