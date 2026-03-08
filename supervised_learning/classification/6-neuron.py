#!/usr/bin/env python3
'''module documented'''
import numpy as np


class Neuron:
    '''class documented'''

    def __init__(self, nx):
        '''init documented'''
        if not (isinstance(nx, int)):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    def gradient_descent(self, X, Y, A, alpha=0.05):
        '''update W and b'''
        m = Y.shape[1]
        dZ = A - Y
        dW = 1 / m * np.dot(dZ, X.T)
        db = (1/m) * np.sum(dZ)
        self.__W = self.__W - (alpha * dW)
        self.__b = self.__b - (alpha * db)

    @property
    def W(self):
        '''getter for W'''
        return self.__W

    @property
    def b(self):
        '''getter for b'''
        return self.__b

    @property
    def A(self):
        '''getter for A'''
        return self.__A

    def forward_prop(self, X):
        '''forward propogation'''
        s = np.matmul(self.__W, X) + self.__b
        self.__A = self.sigmoid(s)
        return self.__A

    def cost(self, Y, A):
        '''cost function'''
        x = 1.0000001 - A
        cost_f = -np.mean((Y * np.log(A)) + (1 - Y) * np.log(x))
        return cost_f

    def evaluate(self, X, Y):
        '''evaulation'''
        pred = self.forward_prop(X)
        pred = np.where(pred >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A)
        return pred, cost

    def train(self, X, Y, iterations=5000, alpha=0.05):
        '''train and evaluate return'''
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        # Validate alpha
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
        return self.evaluate(X, Y)

    @staticmethod
    def sigmoid(X):
        '''sigmoid function'''
        return 1 / (1 + np.e ** (-X))
