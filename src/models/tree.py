#!/usr/bin/env python
# encoding: utf-8

from functools import reduce
import numpy as np
import scipy
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
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
            if self.args['node_expert']['expert'] == 'svm':
                if self.args['node_expert']['tune'] is True:
                    self.classifier = SVC(**self.args['svm']['preset_model_params'])
                else:
                    self.classifier = SVC(**self.args['svm']['default'])
            else:
                self.logger.error('Non-SVM node expert not implemented!')
                raise NotImplementedError("Non-SVM node expert not implemented!")

            self.classifier_best = None
            self.t_best = None

        else:  # leaf node
            if self.args['leaf_expert']['expert'] == 'linear':
                self.regressor = LinearRegression()
            elif self.args['leaf_expert']['expert'] == 'svr':
                if self.args['leaf_expert']['tune'] is True:
                    self.regressor = SVR(**self.args['svr']['preset_model_params'])
                else:
                    self.regressor = SVR(**self.args['svr']['default'])

            self.regressor_best = None

        # Initialize node conditional and criterion
        if self.is_root:  # root
            self.qz_x_best = 1.  # q(z->current node | x), best
            self.qz_x = None  # current

        else:  # non-root
            self.qz_x_best = None
            self.qz_x = None

    def convert_leaf2nonleaf(self):
        '''
        Convert leaf node to non-leaf node.
        '''
        self.is_leaf = False
        self.children = []
        self.children_count = 0

        # Create non-leaf data
        if self.args['node_expert']['expert'] == 'svm':
            if self.args['node_expert']['tune'] is True:
                self.classifier = SVC(**self.args['svm']['preset_model_params'])
            else:
                self.classifier = SVC(**self.args['svm']['default'])
        else:
            self.logger.error('Non-SVM node expert not implemented!')
            raise NotImplementedError("Non-SVM node expert not implemented!")

        self.classifier_best = None
        self.t_best = None

    def convert_nonleaf2leaf(self):
        '''
        Convert non-leaf node to leaf node.
        '''
        assert self.classifier_best is None,\
            "Attempting to convert a splitter node to leaf node"
        self.is_leaf = True
        self.children = None
        self.children_count = 0

        # Null non-leaf data
        self.classifier = None

    def get_parent(self):
        assert self.parent is not None,\
            "No parent! check if it is a root node"
        return self.parent

    def get_children(self):
        assert self.children is not None,\
            "No children! check if it is a leaf node"
        return self.children

    def get_child_left(self):
        assert self.children is not None,\
            "No children! check if it is a leaf node"
        assert self.children_count == 2,\
            "Only 1 child!"
        return self.children[0]

    def get_child_right(self):
        assert self.children is not None,\
            "No children! check if it is a leaf node"
        assert self.children_count == 2,\
            "Only 1 child!"
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

        self.feature_dim = self.args['feature_dim']

        self.data = None  # store original data
        self.num_samples = None  # total number of samples

        self.nodes = []  # store nodes on the tree

        self.root = self.create_root()  # create root
        self.nodes.append(self.root)

        self.t_best_list = []  # collection of best node thresholds
        self.Q_best = -1e9  # best Q of current tree
        self.Q_best_history = []  # collection of best Qs

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

    def get_leaves(self, node):
        '''
        Get all leaves of the current tree.
        '''
        leaves = set()

        if node.is_leaf:
            assert node.children is None, "Non-leaf node reached!"
            leaves.add(node)

        else:
            for child in node.children:
                leaves.update(self.get_leaves(child))  # NOTE: set union

        return leaves

    def get_leaves_tree(self):
        '''
        Yet another version of getting all leaves of the current tree.
        '''
        leaves = []

        def _get_leaves_node(node):
            if node is not None:
                if node.is_leaf:
                    assert node.children is None, "Non-leaf node reached!"
                    leaves.append(node)
                    return

                for child in node.children:
                    _get_leaves_node(child)

        _get_leaves_node(self.root)

        return leaves

    # ========================================
    def train(self, X, Y):
        '''
        Train the tree model.

        Parameters
        ----------
        X: Tuple(np.array), shape (N, D)
        Y: Tuple(np.array), shape (N,)
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

            # Grow subtree
            if node.is_root is not True:
                # convert leaf to non-leaf node
                node.convert_leaf2nonleaf()

            node_left, node_right = self.grow_subtree(node)
            self.nodes.append(node_left)
            self.nodes.append(node_right)

            self.logger.debug('=' * 50)
            self.logger.debug('grown a subtree')

            # Optimize subtree
            Q_best, best_collection = self.EM_step(Xi, Yi,
                                                   node, node_left, node_right)

            if Q_best > self.Q_best:  # if improvement, keep growing
            #  if Q_best > self.Qk_best and Q_best - self.Qk_best < self.args['min_Q_increase']:

                # Retrain
                self.logger.debug('-' * 25)
                self.logger.debug('Re-train with best t: {:.4f}'.
                                  format(best_collection['t']))

                Q = self._retrain_t(best_collection['t'], Xi, Yi,
                                    node, node_left, node_right)

                self.logger.debug('Q: {:.4f}  Q_best {:.4f}'.format(Q, Q_best))
                self.logger.debug('*' * 50)
                self.logger.debug('subtree saw improvement')
                self.logger.debug('t_best: {:.4f}  Q_best: {:.4f}'.
                                  format(best_collection['t'], Q_best))

                # Store t_best
                self.t_best_list.append(best_collection['t'])

                # Store Q_best
                self.Q_best = Q_best
                self.Q_best_history.append(Q_best)

                # Store best node classifier
                node.t_best = best_collection['t']
                node.classifier_best = node.classifier

                # Store best leaf regressor
                node_left.regressor_best = node_left.regressor
                node_right.regressor_best = node_right.regressor

                # Store best node conditional
                node_left.qz_x_best = node_left.qz_x
                node_right.qz_x_best = node_right.qz_x

                # Recursive grow
                X_new = (best_collection['data_left'][0],
                         best_collection['data_right'][0])
                Y_new = (best_collection['data_left'][1],
                         best_collection['data_right'][1])
                nodes_per_level_new = [node_left, node_right]
                self.grow(X_new, Y_new, nodes_per_level_new)

            else:  # no improvement with this subtree
                node.convert_nonleaf2leaf()  # convert back to leaf

                del self.nodes[-1]  # delete subtree
                del self.nodes[-1]

                self.logger.debug('*' * 50)
                self.logger.debug('subtree has no improvement, deleted')
                continue

    # ========================================
    def EM_step(self, X, Y, node, node_left, node_right):
        '''
        EM method for optimizing the tree.
        '''
        # Initialize best settings
        Q_best = self.Q_best  # init with best Q of current tree

        best_collection = {
            "t": None,
            "Q": None,
            "data_left": None,
            "data_right": None
        }

        # Get t
        t_step = self.args['t_step']
        t_low = np.min(Y) + t_step
        t_high = np.max(Y) - t_step

        if t_high - t_low <= 0:  # no room for further split
            return Q_best, best_collection

        t_grid = np.arange(t_low, t_high, t_step)

        for t in t_grid:
            self.logger.debug('-' * 50)
            self.logger.debug('t: {:.4f}'.format(t))

            # Process data
            data_left, data_right = self._split_data(X, Y, t)
            if len(data_left[1]) < self.args['min_samples_leaf'] or \
                    len(data_right[1]) < self.args['min_samples_leaf']:
                # leaf has insufficient data to split
                self.logger.debug('leaf has insufficient data to split')
                continue  # proceed to next t
            Y_new = self._create_new_label(Y, t)

            # EM
            Q = self._EM_step_t(X, Y, Y_new, data_left, data_right,
                                node, node_left, node_right)

            self.logger.debug('-' * 25)
            self.logger.debug('Q: {:.4f}  Q_best {:.4f}'.
                              format(Q, Q_best))

            if Q > Q_best:  # store current best settings
                Q_best = Q

                best_collection['t'] = t
                best_collection['Q'] = Q
                best_collection['data_left'] = data_left
                best_collection['data_right'] = data_right

        return Q_best, best_collection

    def _EM_step_t(self, X, Y, Y_new, data_left, data_right,
                   node, node_left, node_right):
        '''
        EM for one step t.
        '''
        # Train classifier
        self.train_svm(node, X, Y_new)

        # Populate node conditionals
        qz_x_prev = node.qz_x_best  # get previous conditional

        qz_x_left_current, qz_x_right_current =\
            self._infer_Qz_x(node, self.data[0])

        qz_x_left = qz_x_prev * qz_x_left_current
        qz_x_right = qz_x_prev * qz_x_right_current
        node_left.qz_x = qz_x_left
        node_right.qz_x = qz_x_right

        # Train regressor
        self.train_leaf(node_left, *data_left)
        self.train_leaf(node_right, *data_right)

        # Eveluate Q
        Q = self.evaluate_Q()

        return Q

    def train_svm(self, node, X, Y):
        '''
        Train SVM.
        '''
        assert node.is_leaf is not True, "Error: leaf node reached!"
        assert len(np.unique(Y)) == 2, "Error: non-binary labels!"

        if self.args['node_expert']['tune'] is True:
            svm = node.classifier
            grid = GridSearchCV(svm, **self.args['svm']['tuning_settings'])
            grid.fit(X, Y)
            self.logger.debug('-' * 25)
            self.logger.debug('Best SVC parameters: ')
            self.logger.debug(grid.best_params_)
            node.classifier = grid.best_estimator_  # NOTE: copy trained classifier

        else:
            node.classifier.fit(X, Y)

    def train_leaf(self, node, X, Y):
        '''
        Train leaf.
        '''
        assert node.is_leaf is True, "Error: non-leaf node!"

        if self.args['leaf_expert']['tune'] is True:
            svr = node.regressor
            grid = GridSearchCV(svr, **self.args['svr']['tuning_settings'])
            grid.fit(X, Y)
            self.logger.debug('-' * 25)
            self.logger.debug('Best SVR parameters: ')
            self.logger.debug(grid.best_params_)
            node.regressor = grid.best_estimator_

        else:
            node.regressor.fit(X, Y)

    def evaluate_Q(self):
        # Get all original data
        Xo, Yo = self.data

        # Get all leaves
        leaves = list(self.get_leaves(self.root))

        # Compute E_q(z | x) [y | z, x]
        #  = sum_k {(y_i | z_k, x_i) q(z_k | x_i)} for all i
        Ey_xz = 0
        for lk in leaves:
            # q(z->lk | x)
            if lk.qz_x_best is not None:  # use optimized conditional
                assert lk.regressor_best is not None, "No best regressor!"
                qz_x_lk = lk.qz_x_best
            else:
                qz_x_lk = lk.qz_x

            # (y_pred | x, z->lk)
            if lk.regressor_best is not None:  # use optimized regressor
                yh_xz_lk = lk.regressor_best.predict(Xo)
            else:
                yh_xz_lk = lk.regressor.predict(Xo)

            # prediction
            fh_x_lk = yh_xz_lk * qz_x_lk

            Ey_xz += fh_x_lk

        # Sum  NOTE: negative to get score
        if self.args['Q'] == 'neg_mae':
            Q = -np.mean(np.abs(Yo - Ey_xz))

        elif self.args['Q'] == 'neg_mse':
            Q = -np.mean(np.square(Yo - Ey_xz))

        return Q

    def _infer_Qz_x(self, node, X):
        '''
        Compute q(z=0 | x) and q(z=1 | x).
        '''
        probs = node.classifier.predict_proba(X)

        qz_x_left = probs[:, 0]
        qz_x_right = probs[:, 1]

        return qz_x_left, qz_x_right

    def _retrain_t(self, t, X, Y,
                   node, node_left, node_right):
        '''
        Retrain with best t to get best model.
        '''
        data_left, data_right = self._split_data(X, Y, t)
        Y_new = self._create_new_label(Y, t)

        Q = self._EM_step_t(X, Y, Y_new, data_left, data_right,
                            node, node_left, node_right)

        return Q

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

    def _add_ones(self, X):
        '''
        Add one column of 1s before X.
        '''
        N = X.shape[0]
        return np.concatenate((np.ones((N, 1)), X), axis=1)

    # ========================================
    def predict_hard(self, X):
        Yh = []
        for x in X:
            x = x.reshape(-1, self.feature_dim)

            # Trace down to leaf node
            node = self.root
            while not node.is_leaf:
                cls = node.classifier_best.predict(x)

                if cls <= 0:  # -> left node
                    node = node.children[0]
                else:  # -> right node
                    node = node.children[1]

            # Predict
            yh = node.regressor_best.predict(x)
            Yh.append(yh)

        return np.array(Yh, dtype=float)

    def predict(self, X):
        '''
        Compute E_q(z | x) [y | z, x] recursively.
        '''
        Ey_xz_list = []

        def _compute_Ey_xz(node, qz_x):
            if node is not None:
                if node.is_leaf:
                    yh_xz_k = node.regressor_best.predict(X)

                    Ey_xz_list.append(yh_xz_k * qz_x)
                    return

                assert node.children_count == 2, "Incomplete children!"

                proba = node.classifier_best.predict_proba(X)
                qz_x_l = proba[:, 0]
                qz_x_r = proba[:, 1]

                _compute_Ey_xz(node.get_child_left(), qz_x * qz_x_l)
                _compute_Ey_xz(node.get_child_right(), qz_x * qz_x_r)

        _compute_Ey_xz(self.root, self.root.qz_x_best)

        Ey_xz = reduce(lambda x, y: x + y, Ey_xz_list)

        return Ey_xz

    # ========================================
    def get_path(self):
        pass

    def get_params(self):
        pass

    def export(self):
        pass
