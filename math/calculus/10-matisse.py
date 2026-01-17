#!/usr/bin/env python3
def poly_derivative(poly):
    """
    Calculate the derivative of a polynomial.

    Args:
        poly (list): list of coefficients, where poly[i] is the coefficient for x^i

    Returns:
        list: coefficients of the derivative polynomial
              or None if input is invalid
    """
    # Validate input
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not all(isinstance(c, (int, float)) for c in poly):
        return None

    # Constant polynomial
    if len(poly) == 1:
        return [0]

    # Compute derivative
    derivative = [i * poly[i] for i in range(1, len(poly))]

    # If derivative is all zeros
    if all(c == 0 for c in derivative):
        return [0]

    return derivative
