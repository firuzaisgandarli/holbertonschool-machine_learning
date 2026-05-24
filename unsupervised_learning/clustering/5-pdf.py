#!/usr/bin/env python3
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution.
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None

    n, d = X.shape

    if m.shape[0] != d or S.shape != (d, d):
        return None

    # determinant and inverse of covariance matrix
    det = np.linalg.det(S)
    if det <= 0:
        return None

    inv = np.linalg.inv(S)

    diff = X - m  # (n, d)

    # exponent term: (x - m)^T S^-1 (x - m)
    exponent = np.sum((diff @ inv) * diff, axis=1)

    norm_const = 1 / (np.sqrt(((2 * np.pi) ** d) * det))

    P = norm_const * np.exp(-0.5 * exponent)

    # enforce minimum value
    P = np.clip(P, 1e-300, None)

    return P
