#!/usr/bin/env python3
"""Neuron class performing binary classification with gradient descent."""

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
        Calculate the forward propagation of the neuron.

        Parameters:
        X (numpy.ndarray): shape (nx, m) input data

        Returns:
        numpy.ndarray: activated output
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Perform one pass of gradient descent on the neuron.

        Parameters:
        X (numpy.ndarray): shape (nx, m) input data
        Y (numpy.ndarray): shape (1, m) correct labels
        A (numpy.ndarray): shape (1, m) activated output
        alpha (float): learning rate
        """
        m = Y.shape[1]
        dZ = A - Y
        dW = np.matmul(dZ, X.T) / m
        db = np.sum(dZ) / m

        self.__W -= alpha * dW
        self.__b -= alpha * db
