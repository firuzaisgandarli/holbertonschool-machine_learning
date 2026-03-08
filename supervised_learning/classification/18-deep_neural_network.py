#!/usr/bin/env python3
"""
Module that defines a deep neural network performing binary classification
with forward propagation.
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Deep neural network performing binary classification.

    Attributes:
        __L (int): Number of layers in the network.
        __cache (dict): Dictionary to store intermediate values.
        __weights (dict): Dictionary to store weights and biases.
    """

    def __init__(self, nx, layers):
        """
        Initialize the deep neural network.

        Args:
            nx (int): Number of input features.
            layers (list): Number of nodes in each layer.

        Raises:
            TypeError: If nx is not an integer or layers not a list of positive integers.
            ValueError: If nx < 1.
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
            self.__weights[f"W{l + 1}"] = (np.random.randn(layer_size, prev_size) *
                                           np.sqrt(2 / prev_size))
            self.__weights[f"b{l + 1}"] = np.zeros((layer_size, 1))

    @property
    def L(self):
        """Getter for the number of layers."""
        return self.__L

    @property
    def cache(self):
        """Getter for the cache dictionary."""
        return self.__cache

    @property
    def weights(self):
        """Getter for the weights dictionary."""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m)

        Returns:
            A (numpy.ndarray): Output of the neural network
            cache (dict): Dictionary with all intermediary activated outputs
        """
        self.__cache['A0'] = X
        for l in range(1, self.__L + 1):
            W = self.__weights[f"W{l}"]
            b = self.__weights[f"b{l}"]
            A_prev = self.__cache[f"A{l - 1}"]
            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))  # Sigmoid activation
            self.__cache[f"A{l}"] = A
        return A, self.__cache
