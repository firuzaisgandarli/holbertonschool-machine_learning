#!/usr/bin/env python3
"""NeuralNetwork class with one hidden layer performing binary classification
with private attributes, getters, and forward propagation.
"""

import numpy as np


class NeuralNetwork:
    """Defines a neural network with one hidden layer for binary classification"""

    def __init__(self, nx, nodes):
        """
        Initialize the neural network.

        Parameters:
        nx (int): number of input features
        nodes (int): number of nodes in the hidden layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Private attributes
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    # Getter methods
    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """
        Performs forward propagation of the neural network

        Parameters:
        X (numpy.ndarray): input data of shape (nx, m)

        Returns:
        tuple: activated outputs of the hidden layer (__A1) and output neuron (__A2)
        """
        # Sigmoid activation for hidden layer
        Z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        # Sigmoid activation for output neuron
        Z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2
