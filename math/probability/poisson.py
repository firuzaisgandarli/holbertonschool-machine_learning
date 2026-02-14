#!/usr/bin/env python3
"""
Poisson distribution module
"""

e = 2.7182818285


class Poisson:
    """
    Class that represents a Poisson distribution
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize Poisson distribution

        Parameters:
        - data: list of data points to estimate lambtha
        - lambtha: expected number of occurrences in a given time frame
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the PMF for a given number of successes

        Parameters:
        - k: number of successes

        Returns:
        - PMF value for k
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        # factorial calculation without imports
        fact = 1
        for i in range(1, k + 1):
            fact *= i

        return ((e ** (-self.lambtha)) * (self.lambtha ** k)) / fact

    def cdf(self, k):
        """
        Calculates the CDF for a given number of successes

        Parameters:
        - k: number of successes

        Returns:
        - CDF value for k
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        cumulative = 0
        for i in range(0, k + 1):
            # factorial calculation
            fact = 1
            for j in range(1, i + 1):
                fact *= j
            cumulative += ((e ** (-self.lambtha)) * (self.lambtha ** i)) / fact

        return cumulative
