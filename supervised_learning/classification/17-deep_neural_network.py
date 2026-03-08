#!/usr/bin/env python3
"""DeepNeuralNetwork module"""

import numpy as np


class DeepNeuralNetwork:
    """Deep Neural Network performing binary classification"""

    def __init__(self, nx, layers):
        """Constructor for DeepNeuralNetwork
        nx: number of input features
        layers: list containing number of nodes in each layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for l in range(self.__L):
            layer_size = layers[l]
            prev_size = nx if l == 0 else layers[l - 1]
            self.__weights["W{}".format(l + 1)] = (
                np.random.randn(layer_size, prev_size) * np.sqrt(2 / prev_size)
            )
            self.__weights["b{}".format(l + 1)] = np.zeros((layer_size, 1))

    @property
    def L(self):
        """Number of layers in the network"""
        return self.__L

    @property
    def cache(self):
        """Holds all intermediary values of the network"""
        return self.__cache

    @property
    def weights(self):
        """Holds all weights and biases of the network"""
        return self.__weights
