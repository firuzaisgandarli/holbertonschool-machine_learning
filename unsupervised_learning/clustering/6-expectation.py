#!/usr/bin/env python3
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Performs the expectation step in the EM algorithm.
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

    # Compute likelihoods P(x | cluster j) for all clusters
    P = np.zeros((k, n))

    for j in range(k):  # allowed only loop
        P[j] = pdf(X, m[j], S[j]) * pi[j]

    # Total likelihood per data point
    total = np.sum(P, axis=0)

    # Avoid division by zero
    total = np.clip(total, 1e-300, None)

    # Responsibilities (posterior probabilities)
    g = P / total

    # log-likelihood
    l = np.sum(np.log(total))

    return g, l
