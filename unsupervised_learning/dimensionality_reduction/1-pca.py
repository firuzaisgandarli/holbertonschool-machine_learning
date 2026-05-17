#!/usr/bin/env python3
"""PCA v2 module"""

import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset and reduces its dimensionality.

    Args:
        X (numpy.ndarray): shape (n, d)
        ndim (int): new dimensionality

    Returns:
        T (numpy.ndarray): shape (n, ndim)
    """
    # 1. Center the data
    X_centered = X - np.mean(X, axis=0)

    # 2. Perform SVD
    U, S, Vt = np.linalg.svd(X_centered)

    # 3. Select top ndim components
    W = Vt[:ndim].T  # shape (d, ndim)

    # 4. Transform data
    T = np.matmul(X_centered, W)

    return T
