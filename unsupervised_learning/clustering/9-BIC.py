#!/usr/bin/env python3
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds best number of clusters using BIC for a GMM.
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n

    if kmin <= 0 or kmax <= 0 or kmax < kmin:
        return None, None, None, None

    ks = list(range(kmin, kmax + 1))

    results = []
    l_vals = []
    b_vals = []

    best_k = None
    best_result = None
    best_bic = np.inf

    for k in ks:
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations=iterations, tol=tol, verbose=verbose
        )

        # number of parameters:
        # pi: k-1 (since sum to 1)
        # m: k*d
        # S: k*(d(d+1)/2)
        p = (k - 1) + (k * d) + (k * d * (d + 1)) // 2

        bic = p * np.log(n) - 2 * log_likelihood

        results.append((pi, m, S))
        l_vals.append(log_likelihood)
        b_vals.append(bic)

        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, np.array(l_vals), np.array(b_vals)
