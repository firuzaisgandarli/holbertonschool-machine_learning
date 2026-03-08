#!/usr/bin/env python3
"""Deep Neural Network performing binary classification"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network with multiple layers"""

    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for l in range(self.L):
            if l == 0:
                self.weights["W1"] = (np.random.randn(layers[l], nx) *
                                      np.sqrt(2 / nx))
            else:
                self.weights["W{}".format(l + 1)] = (
                    np.random.randn(layers[l], layers[l - 1]) *
                    np.sqrt(2 / layers[l - 1])
                )
            self.weights["b{}".format(l + 1)] = np.zeros((layers[l], 1))
