#!/usr/bin/env python3
"""
12-neural_network.py

This module defines a NeuralNetwork class with one hidden layer
for binary classification using logistic regression.
Includes methods for forward propagation, cost calculation, and evaluation.
"""
import numpy as np


class NeuralNetwork:
    """Neural network with one hidden layer for binary classification."""

    def __init__(self, nx, nodes):
        """Initialize the neural network.

        Args:
            nx (int): Number of input features.
            nodes (int): Number of nodes in the hidden layer.
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
        """Calculate forward propagation of the neural network.

        Args:
            X (np.ndarray): Input data of shape (nx, m).

        Returns:
            tuple: Activated outputs for hidden layer and output neuron.
        """
        Z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))  # Sigmoid activation

        Z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid activation

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculate cost using logistic regression.

        Args:
            Y (np.ndarray): Correct labels of shape (1, m).
            A (np.ndarray): Activated output of shape (1, m).

        Returns:
            float: Cost.
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """Evaluate the neural network's predictions.

        Args:
            X (np.ndarray): Input data of shape (nx, m).
            Y (np.ndarray): Correct labels of shape (1, m).

        Returns:
            tuple: Predicted labels (0 or 1) and cost.
        """
        self.forward_prop(X)
        predictions = np.where(self.__A2 >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A2)
        return predictions, cost
