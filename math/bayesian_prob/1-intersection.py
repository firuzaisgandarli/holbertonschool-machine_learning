#!/usr/bin/env python3
"""Intersection module for Bayesian probability"""

import numpy as np
from 0-likelihood import likelihood


def intersection(x, n, P, Pr):
    """
    Calculates the intersection of the data with various hypothetical
    probabilities and their prior beliefs.

    Parameters:
    x (int): number of patients with severe side effects
    n (int): total number of patients observed
    P (numpy.ndarray): 1D array of hypothetical probabilities
    Pr (numpy.ndarray): 1D array of prior beliefs of P

    Returns:
    numpy.ndarray: intersection of likelihood and prior for each P
    """
    # Check n
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # Check x
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    # x cannot be greater than n
    if x > n:
        raise ValueError("x cannot be greater than n")

    # Check P
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    # Check Pr
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    # Check P values
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Check Pr values
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    # Check Pr sums to 1
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Compute likelihood
    L = likelihood(x, n, P)

    # Intersection = likelihood * prior
    return L * Pr
