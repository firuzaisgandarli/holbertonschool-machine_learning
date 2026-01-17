#!/usr/bin/env python3
"""
Module 11-integral
Provides a function to compute the indefinite integral of a polynomial
represented as a list of coefficients.
"""

def poly_integral(poly):
    """
    Calculate the indefinite integral of a polynomial.

    Args:
        poly (list): list of coefficients, where poly[i] is the coefficient
                     for x^i

    Returns:
        list: coefficients of the integral polynomial
              (constant of integration is represented as +C, not included)
              or None if input is invalid
    """
    # Validate input
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not all(isinstance(c, (int, float)) for c in poly):
        return None

    # Compute integral: new coefficient = poly[i] / (i+1)
    integral = [0]  # constant of integration term
    for i in range(len(poly)):
        integral.append(poly[i] / (i + 1))

    return integral
