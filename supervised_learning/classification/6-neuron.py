#!/usr/bin/env python3
"""Neuron class performing binary classification and training."""

import numpy as np


class Neuron:
    """Class that defines a single neuron performing binary classification."""

    def __init__(self, nx):
        """
        Initialize the neuron.

        Parameters:
        nx (int): number of input features
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Weights vector for the neuron."""
        return self.__W

    @property
    def b(self):
        """Bias for the neuron."""
        return self.__b

    @property
    def A(self):
        """Activated output of the neuron (prediction)."""
        return self.__A

    def forward_prop(self, X):
        """
        Calculate forward propagation of the neuron.

        Parameters:
        X (numpy.ndarray): input data, shape (nx, m)

        Returns:
        numpy.ndarray: activated output
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Compute logistic regression cost.

        Parameters:
        Y (numpy.ndarray): true labels
        A (numpy.ndarray): activated output

        Returns:
        float: cost
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A + 1e-8) + (1 - Y) * np.log(1 - A + 1e-8)
        )
        return cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Perform one pass of gradient descent.

        Parameters:
        X (numpy.ndarray): input data
        Y (numpy.ndarray): true labels
        A (numpy.ndarray): activated output
        alpha (float): learning rate
        """
        m = Y.shape[1]
        dZ = A - Y
        dW = np.matmul(dZ, X.T) / m
        db = np.sum(dZ) / m
        self.__W -= alpha * dW
        self.__b -= alpha * db

    def evaluate(self, X, Y):
        """
        Evaluate neuron predictions and cost.

        Parameters:
        X (numpy.ndarray): input data
        Y (numpy.ndarray): true labels

        Returns:
        tuple: (predictions, cost)
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = np.where(A >= 0.5, 1, 0)
        return np.round(predictions, decimals=10), np.round(cost, decimals=10)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Train the neuron.

        Parameters:
        X (numpy.ndarray): input data
        Y (numpy.ndarray): true labels
        iterations (int): number of iterations
        alpha (float): learning rate

        Returns:
        tuple: (predictions, cost) after training
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for _ in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

        return self.evaluate(X, Y)
