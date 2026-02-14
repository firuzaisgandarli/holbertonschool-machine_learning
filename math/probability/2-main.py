#!/usr/bin/env python3

Poisson = __import__('poisson').Poisson

# Example dataset to estimate lambtha
data = [2, 4, 6, 5, 7, 3, 5, 4, 6, 5]
p1 = Poisson(data)
print('F(9):', p1.cdf(9))

# Using a fixed lambtha
p2 = Poisson(lambtha=5)
print('F(9):', p2.cdf(9))
