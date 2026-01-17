#!/usr/bin/env python3
"""
Module that computes the integral of a polynomial
"""

def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial.

    poly: list of coefficients
    C: integration constant
    """
    # Validation
    if not isinstance(poly, list) or not isinstance(C, int):
        return None
    if len(poly) == 0:
        return None

    result = [C]

    for power, coef in enumerate(poly):
        if not isinstance(coef, (int, float)):
            return None
        new_coef = coef / (power + 1)
        # Convert whole numbers to int
        if new_coef.is_integer():
            new_coef = int(new_coef)
        result.append(new_coef)

    # Remove trailing zeros to keep list as small as possible
    while len(result) > 1 and result[-1] == 0:
        result.pop()

    return result
