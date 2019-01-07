#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from matplotlib import pyplot as plt
import pdb


def line_samples(x, a, b, n=0, seed=1111):
    '''
    Generate samples along line y = ax + b
    with Gaussian noise variance n^2.


    Parameters
    ----------
    x: np.array[float]
    a: float
        slop
    b: float
        intercept
    n: float
        standard deviation
    seed: int
        random seed

    Returns
    -------
    y: np.array[float]
    '''
    np.random.seed(seed)

    N = x.shape[0]

    noise = np.random.normal(0, n, (N,))

    y = a * x + b + noise

    return y


# Generate data
x1 = np.linspace(-8, 10, 900)
y1 = line_samples(x1, 0.8, -1, 0.2)
y1o = line_samples(x1, 0.8, -1, 0)

x2 = np.linspace(-10, 2, 800)
y2 = line_samples(x2, 1.2, 6, 0.15)
y2o = line_samples(x2, 1.2, 6, 0)

x3 = np.linspace(-2, 6, 800)
y3 = line_samples(x3, -2, 5, 0.25)
y3o = line_samples(x3, -2, 5, 0)

# Plot data
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
ax.scatter(x1, y1, s=2, c='r', marker='o')
ax.scatter(x2, y2, s=2, c='b', marker='.')
ax.scatter(x3, y3, s=2, c='k', marker='+')
l1 = ax.plot(x1, y1o, 'r-', linewidth=1.5)
l2 = ax.plot(x2, y2o, 'b-', linewidth=1.5)
l3 = ax.plot(x3, y3o, 'k-', linewidth=1.5)
ax.axis('tight')
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.legend(('y=0.8x-1, noise=0.2', 'y=1.2x+6, noise=0.15', 'y=-2x+5, noise=0.25'),
          loc='best', fontsize=11)
plt.tight_layout()
# plt.show()
# plt.savefig('data_3lines_2.pdf')

# Save data
X = np.concatenate((x1, x2, x3), axis=0)
Y = np.concatenate((y1, y2, y3), axis=0)
data = np.concatenate((X[:, np.newaxis], Y[:, np.newaxis]), axis=1)
# np.savetxt('3lines_2.txt', data)

# Split data
N = data.shape[0]
num_train = int(np.floor(N * 0.7))
num_test = N - num_train
sample_idx = np.random.permutation(N)
train_idx = sample_idx[:num_train]
test_idx = sample_idx[num_train:]
data_train = np.take(data, train_idx, axis=0)
data_test = np.take(data, test_idx, axis=0)
# np.savetxt('3lines_2_train.txt', data_train)
# np.savetxt('3lines_2_test.txt', data_test)

print('Total {} samples, train {}, test {}'.format(N, num_train, num_test))
