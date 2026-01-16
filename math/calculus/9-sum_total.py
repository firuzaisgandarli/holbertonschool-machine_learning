#!/usr/bin/env python3
"""
Module that provides summation of squares from 1 to n
"""


def summation_i_squared(n):
    """
    Calculates the sum of squares from 1 to n.
    Returns integer value if n is valid, otherwise None.
    """
    if not isinstance(n, int) or n < 1:
        return None
    return n * (n + 1) * (2 * n + 1) // 6
