#!/usr/bin/env python3
"""Likelihood module for binomial distribution"""


import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining the data (x successes
    out of n trials) for various hypothetical probabilities.

    Parameters:
    x (int): number of observed successes
    n (int): total number of trials
    P (numpy.ndarray): 1D array of hypothetical probabilities

    Returns:
    numpy.ndarray: likelihood of the data for each probability in P
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    comb = np.math.factorial(n) / (
        np.math.factorial(x) * np.math.factorial(n - x)
    )

    return comb * (P ** x) * ((1 - P) ** (n - x))
