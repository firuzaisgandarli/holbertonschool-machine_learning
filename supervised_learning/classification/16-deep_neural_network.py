#!/usr/bin/env python3
"""
Module defines a DeepNeuralNetwork for binary classification
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Class that defines a deep neural network
    """

    def __init__(self, nx, layers):
        """
        Class constructor
        Args:
            nx: number of input features
            layers: list representing the number of nodes in each layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(self.L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            # Layer indexing starts at 1
            curr_layer = i + 1
            # Previous size is nx for the first layer, else layers[i-1]
            prev_size = nx if i == 0 else layers[i - 1]

            # He et al. initialization: randn * sqrt(2 / size_of_previous_layer)
            self.weights["W" + str(curr_layer)] = (
                np.random.randn(layers[i], prev_size) * np.sqrt(2 / prev_size)
            )
            # Biases initialized to zeros
            self.weights["b" + str(curr_layer)] = np.zeros((layers[i], 1))
