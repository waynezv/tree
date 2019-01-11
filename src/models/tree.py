#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import scipy
from sklearn.svm import SVC
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
            self.child_left = None
            self.child_right = None
        else:
            if not self.is_leaf:  # non-root non-leaf node
                self.parent = parent
                self.children = []

                parent.children.append(self)
                parent.children_count += 1

            else:  # leaf node:
                self.parent = parent
                self.children = None

                parent.children.append(self)
                parent.children_count += 1

        # Initialize node-expert depending on is_leaf or not
        if not self.is_leaf:  # non-leaf node
            self.classifier = SVC(self.args['svm'])
        else:  # leaf node
            self.w = np.random.standard_normal((self.feature_dim + 1,))  # include bias
            self.sigma = np.random.uniform(0, 1)  # TODO: no-correlation?

        # Initialize likelihood depending on is_leaf or not
        if not self.is_leaf:  # non-leaf node
            self.qz_l = None  # q(z->left | x)
            self.qz_r = None  # q(z->right | x)
        else:  # leaf node
            self.py_xz = None  # P(y | x, z)

        self.classifier_best = None  # TODO: storing?
        self.w_best = None
        self.sigma_best = None

    def convert_leaf_node(self):
        '''
        Convert leaf node to non-leaf node.
        '''
        self.is_leaf = False
        self.children = []
        self.children_count = 0

        self.classifier = SVC(self.args['svm'])
        self.qz_l = None  # q(z->left | x)
        self.qz_r = None  # q(z->right | x)

        self.w = None
        self.sigma = None
        self.py_xz = None  # P(y | x, z)

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

    def __init__(self, args=None):
        assert args is not None, "No arguments provided!"
        self.args = args

        np.random.seed(self.args['random_seed'])

        self.nodes = []
        self.nodes_per_level = []

        self.root = self.create_root()
        self.nodes.append(self.root)
        self.nodes_per_level.append(self.root)

        self.t_best_list = []  # collection of best node thresholds
        self.t_best = None
        self.t_current = np.random.uniform(args['t_low'], args['t_high'])

        self.Q_best = -1e9  # expectation of complete data log-likelihood
        self.Q_best_list = []

        self.nodes_best = []

        self.tol = 0.1

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

        self.grow([X], [Y], [self.root])

    def grow(self, X, Y, nodes_per_level):
        for i, node in enumerate(nodes_per_level):

            Xi, Yi = X[i], Y[i]

            node.convert_leaf_node()  # TODO: is this correct???
            node_left, node_right = self.grow_subtree(node)
            self.nodes.append(node_left)
            self.nodes.append(node_right)

            Q_best, best_collection = self.M_step(Xi, Yi, node, node_left, node_right)

            if Q_best > self.Q_best:

                self.Q_best = Q_best
                node_best = best_collection['node']
                node_left_best = best_collection['node_left']
                node_right_best = best_collection['node_right']

                self.Q_best_list.append(Q_best)
                self.t_best_list.append(best_collection['t'])
                self.nodes_best.append(node_best)
                self.nodes_best.append(node_left_best)
                self.nodes_best.append(node_right_best)

                node_best.classifier_best = node_best.classifier
                node_left_best.w_best = node_left_best.w
                node_right_best.w_best = node_right_best.w
                # TODO: store node proba?

                Q_increase = Q_best - self.Q_best

                if Q_increase < self.tol:
                    break
                #  return best_collection  # TODO: ??

                nodes_per_level_new = [node_left_best, node_right_best]
                X_new = (best_collection['data_left'][0], best_collection['data_right'][0])
                Y_new = (best_collection['data_left'][1], best_collection['data_right'][1])

                self.grow(X_new, Y_new, nodes_per_level_new)

            else:
                continue

    # ========================================
    def E_step(self):
        self.evaluate_Q()

    def evaluate_Q(self, node, node_left, node_right, X, Y, t):

        qz_x_left, qz_x_right = self._infer_Qz_x(node, X)
        py_xz_left = self._infer_Py_xz(node_left, X, Y)
        py_xz_right = self._infer_Py_xz(node_right, X, Y)
        pz_x_left, pz_x_right = self._infer_Pz_x(Y, t)

        # SUM
        Q = (qz_x_left * (np.log(pz_x_left * py_xz_left) -
                        np.log(qz_x_left)) +
             qz_x_right * (np.log(pz_x_right * py_xz_right) -
                        np.log(qz_x_right)))

        return Q

    def _infer_Pz_x(self, Y, t):  # TODO: local or global?
        '''
        Compute p(z=1 | x, t) and p(z=0 | x, t).
        '''
        N = len(Y)
        n_le = np.count_nonzero(Y <= t)

        pz_x_left = n_le / float(N)
        pz_x_right = 1 - pz_x_left

        return pz_x_left, pz_x_right

    def _infer_Py_xz(self, node, X, Y):
        N = X.shape[0]
        mu = np.dot(X, node.w)
        std = np.eye(N) * np.sqrt(node.sigma)
        py_xz = scipy.stats.norm(mu, std).pdf(Y)
        return py_xz

    def _infer_Qz_x(self, node, X):
        qz_x_left, qz_x_right = node.svm.predict_proba(X)
        return qz_x_left, qz_x_right

    # ========================================
    def M_step(self, X, Y, node, node_left, node_right):  # TODO: data per node is different
        #  t_grid = np.arange(
            #  self.args['t_low'],
            #  self.args['t_high'] + self.args['t_step'],
            #  self.args['t_step'])

        t_step = self.args['t_step']
        t_grid = np.arange(np.min(Y) + t_step, np.min(Y) - t_step, t_step)

        Q_best = self.Q_best
        best_collection = {
            "t": None,
            "Q": None,
            "data_left": None,
            "data_right": None,
            "node": None,
            "node_left": None,
            "node_right": None
        }

        for t in t_grid:
            Y_new = self._create_new_label(Y, t)
            self._train_svm(node, X, Y_new)

            data_left, data_right = self.split_data(X, Y, t)

            self._train_leaf(node_left, *data_left)
            self._train_leaf(node_right, *data_right)

            Q = self.evaluate_Q(node, node_left, node_right,
                            X, Y, t)

            if Q > Q_best:
                Q_best = Q

                best_collection['t'] = t
                best_collection['Q'] = Q
                best_collection['data_left'] = data_left
                best_collection['data_right'] = data_right
                best_collection['node'] = node
                best_collection['node_left'] = node_left
                best_collection['node_right'] = node_right

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
                                  Y, np.ones(Y.shape) * t)))
        return Y_new

    def split_data(self, X, Y, t):
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
        idx_le = np.nonzero(Y <= t)
        idx_gt = np.nonzero(Y > t)

        X_left = np.take(X, idx_le, axis=0)
        Y_left = np.take(Y, idx_le, axis=0)

        X_right = np.take(X, idx_gt, axis=0)
        Y_right = np.take(Y, idx_gt, axis=0)

        return (X_left, Y_left), (X_right, Y_right)

    def _train_svm(self, node, X, Y):
        assert node.is_leaf is not True, "Error: leaf node reached!"

        node.classifier.fit(X, Y)

    def _add_ones(self, X):
        N = X.shape(0)
        return np.concatenate((np.ones((N, 1)), X), axis=1)

    def _train_leaf(self, node, X, Y):
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
    def predict(self):
        pass

    def get_path(self):
        pass

    def get_params(self):
        pass

    def export(self):
        pass
