#!/usr/bin/env python3
"""PCA module."""

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
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    eig_vals = S ** 2
    total = np.sum(eig_vals)
    cumulative = np.cumsum(eig_vals) / total

    nd = np.where(cumulative >= var)[0][0] + 1

    W = Vt[:nd].T

    return W
