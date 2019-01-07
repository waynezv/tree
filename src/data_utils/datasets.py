# -*- coding: utf-8 -*-
'''
Data utilities for spoken digit dataset.
'''
import os
import numpy as np
from torch.utils.data import Dataset
import pdb


def prepare_data(data_list, num_folds=5, seed=1234):
    '''
    Prepare train and test data with N folds.

    Parameters
    ----------
    data_list: List[string]
        List of files.
    num_folds: Int
    seed: Int

    Returns
    -------
    yield_train_test: Function
        A closure function to yield train and test lists.
    '''
    np.random.seed(seed)

    files = [l.rstrip('\n') for l in open(data_list)]

    def split_folds(N):
        '''
        Split data_list to N folds.
        '''
        num_files = len(files)
        num_per_fold = num_files // N

        file_idx = np.random.permutation(range(num_files))
        folds = [file_idx[i:i + num_per_fold]
                 for i in range(0, num_files, num_per_fold)]

        return folds

    folds = split_folds(num_folds)
    tot_idx = (0, 1, 2, 3, 4)
    train_idx = [(0, 1, 2),
                 (0, 1, 3),
                 (0, 1, 4),
                 (0, 2, 3),
                 (0, 2, 4),
                 (0, 3, 4),
                 (1, 2, 3),
                 (1, 2, 4),
                 (1, 3, 4),
                 (2, 3, 4)]

    def yield_train_test(i):
        '''
        Yield train and test lists.

        Parameters
        ----------
        i: Int
            The i-th run in total C(num_folds, 3) runs.

        Returns
        -------
        train_files, test_files: List[string]
        '''
        train_i = train_idx[i]
        test_i = list(set(tot_idx) - set(train_i))

        train_set = np.concatenate((folds[train_i[0]],
                                    folds[train_i[1]],
                                    folds[train_i[2]]),
                                   axis=0)
        test_set = np.concatenate((folds[test_i[0]],
                                   folds[test_i[1]]),
                                  axis=0)

        train_files = np.take(files, train_set, axis=0)
        test_files = np.take(files, test_set, axis=0)

        return train_files, test_files

    return yield_train_test


class SpokenDigits(Dataset):
    '''
    Wrapper for SpokenDigits dataset.
    '''
    def __init__(self, file_list, root='.'):
        self.root = root
        self.files = file_list
        self.num_files = len(self.files)

    def __len__(self):
        return self.num_files

    def __getitem__(self, i):
        X = np.load(os.path.join(self.root, self.files[i]))[np.newaxis, :]
        attr = self.files[i].split('_')  # e.g. "0_theo_7.npy"
        Y = int(attr[0])  # torch.LongTensor
        return X, Y


class SpokenDigitsSpecCeps(Dataset):
    def __init__(self, file_list, root_spec='.', root_ceps='.'):
        self.root_spec = root_spec
        self.root_ceps = root_ceps
        self.files = file_list
        self.num_files = len(self.files)

    def __len__(self):
        return self.num_files

    def __getitem__(self, i):
        spec = np.load(os.path.join(
            self.root_spec, self.files[i]))[np.newaxis, :].astype('float32')
        ceps = np.load(os.path.join(
            self.root_ceps, self.files[i]))[np.newaxis, :].astype('float32')
        attr = self.files[i].split('_')  # e.g. "0_theo_7.npy"
        Y = int(attr[0])  # torch.LongTensor
        return spec, ceps, Y


class FEMH(Dataset):
    def __init__(self, file_list, root1='.', root2='.'):
        self.root1 = root1
        self.root2 = root2
        self.files = file_list
        self.num_files = len(self.files)

    def __len__(self):
        return self.num_files

    def __getitem__(self, i):
        f, Y = self.files[i].split()
        f = os.path.splitext(f)[0] + '.npy'
        X1 = np.load(os.path.join(self.root1, f)).astype('float32')
        X2 = np.load(os.path.join(self.root2, f)).astype('float32')
        return X1, X2, int(Y)
