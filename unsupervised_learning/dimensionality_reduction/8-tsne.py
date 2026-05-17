#!/usr/bin/env python3
"""Performs t-SNE transformation."""

import numpy as np

pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    Performs a t-SNE transformation.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        ndims (int): New dimensional representation.
        idims (int): Intermediate dimensional representation after PCA.
        perplexity (float): Perplexity value.
        iterations (int): Number of iterations.
        lr (float): Learning rate.

    Returns:
        numpy.ndarray: Optimized low dimensional transformation of X.
    """
    n = X.shape[0]

    X = pca(X, idims)

    P = P_affinities(X, perplexity=perplexity)
    P *= 4

    Y = np.random.randn(n, ndims)

    dY_prev = np.zeros((n, ndims))
    Y_prev = np.zeros((n, ndims))

    for i in range(iterations):
        iteration = i + 1

        if iteration <= 20:
            a = 0.5
        else:
            a = 0.8

        dY, Q = grads(Y, P)

        Y_new = Y - lr * dY + a * (Y - Y_prev)

        Y_prev = Y.copy()
        Y = Y_new

        Y = Y - np.mean(Y, axis=0)

        if iteration == 100:
            P /= 4

        if iteration % 100 == 0:
            C = cost(P, Q)
            print("Cost at iteration {}: {}".format(iteration, C))

    return Y
