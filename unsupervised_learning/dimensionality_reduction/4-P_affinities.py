#!/usr/bin/env python3
"""Calculate symmetric P affinities for t-SNE."""

import numpy as np

P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Calculates the symmetric P affinities of a dataset.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        tol (float): Maximum tolerance for Shannon entropy difference.
        perplexity (float): Desired perplexity.

    Returns:
        numpy.ndarray: Symmetric P affinities of shape (n, n).
    """
    D, P, betas, H = P_init(X, perplexity)
    n = X.shape[0]

    for i in range(n):
        Di = np.concatenate((D[i, :i], D[i, i + 1:]))

        beta_min = None
        beta_max = None

        Hi, Pi = HP(Di, betas[i])
        diff = Hi - H

        while np.abs(diff) > tol:
            if diff > 0:
                beta_min = betas[i].copy()
                if beta_max is None:
                    betas[i] *= 2
                else:
                    betas[i] = (betas[i] + beta_max) / 2
            else:
                beta_max = betas[i].copy()
                if beta_min is None:
                    betas[i] /= 2
                else:
                    betas[i] = (betas[i] + beta_min) / 2

            Hi, Pi = HP(Di, betas[i])
            diff = Hi - H

        P[i, :i] = Pi[:i]
        P[i, i + 1:] = Pi[i:]

    P = (P + P.T) / (2 * n)

    return P
