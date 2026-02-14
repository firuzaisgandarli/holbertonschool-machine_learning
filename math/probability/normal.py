#!/usr/bin/env python3
"""Normal distribution module"""


class Normal:
    """Class that represents a normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize Normal distribution"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            n = len(data)
            mean_value = sum(data) / n

            variance = 0
            for x in data:
                variance += (x - mean_value) ** 2
            variance /= n

            self.mean = float(mean_value)
            self.stddev = float(variance ** 0.5)
