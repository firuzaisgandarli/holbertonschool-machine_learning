#!/usr/bin/env python3
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    E-step of GMM (fully vectorized).
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape != (k, d) or S.shape != (k, d, d):
        return None, None

    # compute pdf for each cluster (k, n)
    P = np.array([pdf(X, m[j], S[j]) for j in range(k)])

    # multiply by priors
    P = P * pi[:, np.newaxis]

    # total probability per point
    total = np.sum(P, axis=0)

    total = np.clip(total, 1e-300, None)

    # responsibilities
    g = P / total

    # log likelihood
    l = np.sum(np.log(total))

    return g, l
