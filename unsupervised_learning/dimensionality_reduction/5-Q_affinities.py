#!/usr/bin/env python3
"""Calculate Q affinities for t-SNE."""

import numpy as np


def Q_affinities(Y):
    """
    Calculates the Q affinities.

    Args:
        Y (numpy.ndarray): Low dimensional dataset of shape (n, ndim).

    Returns:
        tuple: Q affinities matrix and numerator matrix.
    """
    sum_Y = np.sum(np.square(Y), axis=1)

    D = sum_Y[:, np.newaxis] + sum_Y[np.newaxis, :] - 2 * np.matmul(Y, Y.T)
    D = np.maximum(D, 0)

    num = 1 / (1 + D)
    np.fill_diagonal(num, 0)

    Q = num / np.sum(num)

    return Q, num
