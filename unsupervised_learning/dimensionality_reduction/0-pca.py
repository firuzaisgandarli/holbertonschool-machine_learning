#!/usr/bin/env python3
"""Module for Principal Component Analysis."""

import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d), centered around 0.
        var (float): Fraction of variance to maintain.

    Returns:
        numpy.ndarray: Weights matrix W of shape (d, nd).
    """
    U, S, Vt = np.linalg.svd(X)

    variances = S ** 2
    cumulative_variance = np.cumsum(variances) / np.sum(variances)

    nd = np.argmax(cumulative_variance >= var) + 1

    W = Vt[:nd].T

    return W
