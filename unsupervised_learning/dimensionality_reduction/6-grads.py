#!/usr/bin/env python3
"""Calculate gradients for t-SNE."""

import numpy as np

Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    Calculates the gradients of Y.

    Args:
        Y (numpy.ndarray): Low dimensional dataset of shape (n, ndim).
        P (numpy.ndarray): P affinities matrix of shape (n, n).

    Returns:
        tuple: dY gradients and Q affinities.
    """
    n, ndim = Y.shape
    Q, num = Q_affinities(Y)

    dY = np.zeros((n, ndim))

    for i in range(n):
        diff = Y[i] - Y
        coeff = (P[i] - Q[i]) * num[i]
        dY[i] = np.sum(coeff[:, np.newaxis] * diff, axis=0)

    return dY, Q
