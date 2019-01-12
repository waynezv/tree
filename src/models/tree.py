#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import scipy
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pdb


class Node():
    '''
    Binary node.
    '''

    def __init__(self, parent=None, is_root=False, is_leaf=False, args=None):
        assert args is not None, "No arguments provided!"
        self.args = args

        np.random.seed(self.args['random_seed'])

        self.feature_dim = self.args['feature_dim']

        self.is_root = is_root
        self.is_leaf = is_leaf

        # Build hierarchy
        if self.is_root:  # root node
            self.parent = None
            self.children = []
            self.children_count = 0

        else:
            if not self.is_leaf:  # non-root non-leaf node
                self.parent = parent
                self.children = []
                self.children_count = 0

                parent.children.append(self)
                parent.children_count += 1

            else:  # leaf node:
                self.parent = parent
                self.children = None

                parent.children.append(self)
                parent.children_count += 1

        # Initialize node-expert depending on is_leaf or not
        if not self.is_leaf:  # non-leaf node
            #  self.classifier = SVC(**self.args['svm'])
            self.classifier = SVC(**self.args['preset_model_params'])

            self.classifier_best = None
            self.t_best = None

        else:  # leaf node
            self.w = np.random.standard_normal((self.feature_dim + 1,))  # NOTE: include bias
            self.sigma = np.random.uniform(0, 1)  # NOTE: no-correlation

            self.w_best = None
            self.sigma_best = None

        # Initialize node conditional and criterion
        if self.is_root:  # root
            self.qz_x_best = 1.  # q(z->current node | x)
            self.Qk_best = -1e9
        else:  # non-root
            self.qz_x_best = None
            self.Qk_best = None

    def convert_leaf2nonleaf(self):
        '''
        Convert leaf node to non-leaf node.
        '''
        self.is_leaf = False
        self.children = []
        self.children_count = 0

        # Create non-leaf data
        #  self.classifier = SVC(**self.args['svm'])
        self.classifier = SVC(**self.args['preset_model_params'])
        self.classifier_best = None
        self.t_best = None

    def convert_nonleaf2leaf(self):
        '''
        Convert non-leaf node to leaf node.
        '''
        assert self.classifier_best is None, "Attempting to convert a splitter node to leaf node"
        self.is_leaf = True
        self.children = None

        # Null non-leaf data
        self.classifier = None

    def get_parent(self):
        assert self.parent is not None, "No parent! check if it is a root node"
        return self.parent

    def get_children(self):
        assert self.children is not None, "No children! check if it is a leaf node"
        return self.children

    def get_child_left(self):
        assert self.children is not None, "No children! check if it is a leaf node"
        assert self.children_count == 2, "Only 1 child!"
        return self.children[0]

    def get_child_right(self):
        assert self.children is not None, "No children! check if it is a leaf node"
        assert self.children_count == 2, "Only 1 child!"
        return self.children[1]


class Tree():
    '''
    Binary tree.
    '''

    def __init__(self, logger=None, args=None):
        assert args is not None, "No arguments provided!"
        self.args = args
        self.logger = logger

        np.random.seed(self.args['random_seed'])

        self.data = None  # store original data
        self.num_samples = None  # total number of samples

        self.nodes = []  # store nodes on the tree

        self.root = self.create_root()  # create root
        self.nodes.append(self.root)

        self.t_best_list = []  # collection of best node thresholds

    def create_root(self):
        root = Node(is_root=True, args=self.args)
        return root

    def add_node(self, parent=None, is_root=False, is_leaf=True):
        assert parent is not None, "No parent provided!"

        new_node = Node(parent=parent,
                        is_leaf=is_leaf,  # NOTE: first assume leaf node
                        args=self.args)

        return new_node

    def grow_subtree(self, sub_root):
        '''
        Add two nodes to grow a 3-node tree.
        '''
        node_left = self.add_node(sub_root)
        node_right = self.add_node(sub_root)
        return node_left, node_right

    # ========================================
    def train(self, X, Y):
        '''
        Train the tree model.
        '''
        self.data = (X, Y)
        self.num_samples = len(Y)

        self.grow([X], [Y], [self.root])

    def grow(self, X, Y, nodes_per_level):
        '''
        Recursively grow the tree.

        Parameters
        ----------
        X: Tuple(np.array), shape (N, D)
        Y: Tuple(np.array), shape (N,)
        nodes_per_level: List[Node]
        '''
        for i, node in enumerate(nodes_per_level):

            # Get data
            Xi, Yi = X[i], Y[i]

            #  if len(Yi) < self.args['min_samples_split']:
                #  # node has insufficient data to split
                #  continue  # do not split

            if node.is_root is not True:
                # convert leaf to non-leaf node
                node.convert_leaf2nonleaf()

            # Grow subtree
            node_left, node_right = self.grow_subtree(node)
            self.nodes.append(node_left)
            self.nodes.append(node_right)
            self.logger.debug('=' * 50)
            self.logger.debug('grown a subtree')

            # Optimize subtree
            Q_best, best_collection = self.M_step(Xi, Yi, node, node_left, node_right)

            if Q_best > node.Qk_best:  # if improvement, keep growing
            #  if Q_best > node.Qk_best and Q_best - node.Qk_best < self.args['min_Q_increase']:

                self.t_best_list.append(best_collection['t'])

                X_new = (best_collection['data_left'][0], best_collection['data_right'][0])
                Y_new = (best_collection['data_left'][1], best_collection['data_right'][1])
                nodes_per_level_new = [best_collection['node_left'],
                                       best_collection['node_right']]

                self.logger.debug('*' * 50)
                self.logger.debug('subtree saw improvement')
                self.logger.debug('t_best: {:.4f}  Q_best: {:.4f}'.format(best_collection['t'], Q_best))
                self.grow(X_new, Y_new, nodes_per_level_new)

            else:  # no improvement with this subtree
                node.convert_nonleaf2leaf()  # convert back to leaf
                del self.nodes[-1]
                del self.nodes[-1]
                self.logger.debug('*' * 50)
                self.logger.debug('subtree has no improvement, deleted')
                continue

    # ========================================
    def evaluate_Q(self, node, node_left, node_right, X, Y, t):

        Xo, Yo = self.data  # get all original data

        # q(z->left | x) q(z->right | x)
        if node.is_root is True:
            qz_x_prev = 1.
        else:  # get previous conditional
            qz_x_prev = node.qz_x_best

        qz_x_left_current, qz_x_right_current = self._infer_Qz_x(node, Xo)
        qz_x_left = qz_x_prev * qz_x_left_current
        qz_x_right = qz_x_prev * qz_x_right_current

        # p(y | x, z->left) p(y | x, z->right)
        py_xz_left = self._infer_Py_xz(node_left, Xo, Yo)
        py_xz_right = self._infer_Py_xz(node_right, Xo, Yo)

        # p(z->left | x) p(z->right | x)
        pz_x_left, pz_x_right = self._infer_Pz_x(Y, t)

        # Sum
        Q_left = np.sum(qz_x_left * (np.log(pz_x_left * py_xz_left + 1e-9) - np.log(qz_x_left + 1e-9)))
        Q_right = np.sum(qz_x_right * (np.log(pz_x_right * py_xz_right + 1e-9) - np.log(qz_x_right + 1e-9)))
        Q = Q_left + Q_right

        return Q, Q_left, Q_right, qz_x_left, qz_x_right

    def _infer_Pz_x(self, Y, t):
        '''
        Compute p(z=1 | x, t) and p(z=0 | x, t).

        NOTE: using global approximation
        '''
        n_le = np.count_nonzero(Y <= t)
        n_gt = np.count_nonzero(Y > t)

        pz_x_left = n_le / float(self.num_samples)
        pz_x_right = n_gt / float(self.num_samples)

        return pz_x_left, pz_x_right

    def _infer_Py_xz(self, node, X, Y):
        '''
        Compute p(y | x, z).
        '''
        X = self._add_ones(X)

        mu = np.dot(X, node.w)
        std = np.ones(mu.shape) * np.sqrt(node.sigma)

        py_xz = scipy.stats.norm(mu, std).pdf(Y)
        return py_xz

    def _infer_Qz_x(self, node, X):
        '''
        Compute q(z=0 | x) and q(z=1 | x).
        '''
        probs = node.classifier.predict_proba(X)
        qz_x_left = probs[:, 0]
        qz_x_right = probs[:, 1]
        return qz_x_left, qz_x_right

    # ========================================
    def M_step(self, X, Y, node, node_left, node_right):

        # Initialize best settings
        Q_best = node.Qk_best  # init with current sub_root's best Q

        best_collection = {
            "t": None,
            "Q": None,
            "data_left": None,
            "data_right": None,
            "node": None,
            "node_left": None,
            "node_right": None
        }

        # Get t
        t_step = self.args['t_step']
        t_low = np.min(Y) + t_step
        t_high = np.max(Y) - t_step

        #  if t_high - t_low <= 0:
            #  return Q_best, best_collection

        t_grid = np.arange(t_low, t_high, t_step)

        for t in t_grid:
            # Process data
            data_left, data_right = self._split_data(X, Y, t)

            #  if len(data_left[1]) < self.args['min_samples_leaf'] or \
               #  len(data_right[1]) < self.args['min_samples_leaf']:
                #  # leaf has insufficient data to split
                #  continue  # proceed to next t

            self.logger.debug('t: {:.4f}'.format(t))
            Y_new = self._create_new_label(Y, t)

            # Train classifier
            self.train_svm(node, X, Y_new)

            # Train regressor
            self.train_leaf(node_left, *data_left)
            self.train_leaf(node_right, *data_right)

            # Eveluate Q
            Q, Q_left, Q_right, qz_x_left, qz_x_right =\
                self.evaluate_Q(node, node_left, node_right, X, Y, t)
            self.logger.debug('Q: {:.4f}  Q_left: {:.4f}  Q_right: {:.4f}'.format(Q, Q_left, Q_right))

            if Q > Q_best:
                Q_best = Q

                # store current best settings
                best_collection['t'] = t
                best_collection['Q'] = Q
                best_collection['data_left'] = data_left
                best_collection['data_right'] = data_right
                best_collection['node'] = node
                best_collection['node_left'] = node_left
                best_collection['node_right'] = node_right

                # store best node classifier
                node.classifier_best = node.classifier
                node.t_best = t

                # store best leaf regressor
                node_left.w_best = node_left.w
                node_left.sigma_best = node_left.sigma
                node_right.w_best = node_right.w
                node_right.sigma_best = node_right.sigma

                # store best node conditional
                node_left.qz_x_best = qz_x_left
                node_right.qz_x_best = qz_x_right

                # store best node criterion
                node_left.Qk_best = Q_left
                node_right.Qk_best = Q_right

        return Q_best, best_collection

    def _create_new_label(self, Y, t):
        '''
        Create new labels of {0, 1} w.r.t. t.

        Parameters
        ----------
        Y: np.array, shape (N,)
        t: float

        Returns
        -------
        Y_new: np.array[{0, 1}], shape (N,)
        '''
        Y_new = np.array(list(map(lambda y, x: 0 if y <= x else 1,
                                  Y, np.ones(Y.shape) * t)),
                         dtype=float)
        return Y_new

    def _split_data(self, X, Y, t):
        '''
        Split data w.r.t. Y <= t or Y > t.

        Parameters
        ----------
        X: np.array, shape (N, D)
        Y: np.array, shape (N,)
        t: float

        Returns
        -------
        (X_, Y_): (np.array, np.array)
        '''
        idx_le = np.nonzero(Y <= t)[0]
        idx_gt = np.nonzero(Y > t)[0]

        X_left = np.take(X, idx_le, axis=0)
        Y_left = np.take(Y, idx_le, axis=0)

        X_right = np.take(X, idx_gt, axis=0)
        Y_right = np.take(Y, idx_gt, axis=0)

        return (X_left, Y_left), (X_right, Y_right)

    def _add_ones(self, X):
        '''
        Add one column of 1s before X.
        '''
        N = X.shape[0]
        return np.concatenate((np.ones((N, 1)), X), axis=1)

    def train_svm(self, node, X, Y):
        '''
        Train SVM.
        '''
        assert node.is_leaf is not True, "Error: leaf node reached!"
        # node.classifier.fit(X, Y)

        svm = node.classifier
        grid = GridSearchCV(svm, **self.args['tuning_settings'])
        grid.fit(X, Y)

        self.logger.info('-' * 50)
        self.logger.info('Best parameters: ')
        self.logger.info(grid.best_params_)
        pdb.set_trace()
        grid.predict()
        grid.best_estimator_

    def train_leaf(self, node, X, Y):
        '''
        Train leaf.
        '''
        # TODO: N >> D, non-invertible case?
        assert node.is_leaf is True, "Error: non-leaf node!"

        X = self._add_ones(X)

        W = np.dot(
            np.dot(np.linalg.inv(np.dot(X.transpose(), X)),
                   X.transpose()),
            Y)
        node.w = W

        N = len(Y)
        sigma = np.sum(np.square(Y - X.dot(W))) / (N - 2)
        node.sigma = sigma

    # ========================================
    def predict(self, X):
        Yh = []
        for x in X:
            x = x.reshape(-1, 1)

            # Trace down to leaf node
            node = self.root
            while not node.is_leaf:
                cls = node.classifier_best.predict(x)
                if cls <= 0:  # -> left node
                    node = node.children[0]
                else:  # -> right node
                    node = node.children[1]

            x = self._add_ones(x)
            yh = np.dot(x, node.w_best)
            Yh.append(yh)

        return np.array(Yh, dtype=float)

    # ========================================
    def get_path(self):
        pass

    def get_params(self):
        pass

    def export(self):
        pass
