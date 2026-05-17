#!/usr/bin/env python3
"""Initialize variables for t-SNE P affinities."""

import numpy as np


def P_init(X, perplexity):
    """
    Initializes variables required to calculate P affinities in t-SNE.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        perplexity (float): Desired perplexity.

    Returns:
        tuple: (D, P, betas, H)
    """
    n = X.shape[0]

    sum_X = np.sum(np.square(X), axis=1)

    D = sum_X[:, np.newaxis] + sum_X[np.newaxis, :] - 2 * np.matmul(X, X.T)

    D = np.maximum(D, 0)
    np.fill_diagonal(D, 0)

    P = np.zeros((n, n))

    betas = np.ones((n, 1))

    H = np.log2(perplexity)

    return D, P, betas, H
