#!/usr/bin/env python3
"""Binomial distribution module"""


class Binomial:
    """Class that represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """Initialize Binomial distribution"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError(
                    "p must be greater than 0 and less than 1"
                )
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            # Estimate n and p from mean and variance
            n_est = round(mean ** 2 / (mean - variance)
                          if mean != variance else 1)
            p_est = mean / n_est if n_est != 0 else 0

            self.n = n_est
            self.p = p_est

    def pmf(self, k):
        """Calculates the PMF for a given number of successes"""
        if k < 0 or k > self.n:
            return 0
        k = int(k)

        def fact(x):
            result = 1
            for i in range(1, x + 1):
                result *= i
            return result

        comb = fact(self.n) / (fact(k) * fact(self.n - k))
        return (comb * (self.p ** k) *
                ((1 - self.p) ** (self.n - k)))

    def cdf(self, k):
        """Calculates the CDF for a given number of successes"""
        if k < 0:
            return 0
        k = int(k)
        cdf_sum = 0
        for i in range(k + 1):
            cdf_sum += self.pmf(i)
        return cdf_sum
