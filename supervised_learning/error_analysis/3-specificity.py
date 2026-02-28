#!/usr/bin/env python3
"""
Module that calculates specificity for each class
"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix

    Parameters:
    confusion (numpy.ndarray): matrix of shape (classes, classes)
                               where rows are correct labels and
                               columns are predicted labels

    Returns:
    numpy.ndarray: shape (classes,) containing specificity of each class
    """
    true_positives = np.diag(confusion)
    actual_positives = np.sum(confusion, axis=1)
    predicted_positives = np.sum(confusion, axis=0)

    false_negatives = actual_positives - true_positives
    false_positives = predicted_positives - true_positives
    total = np.sum(confusion)

    true_negatives = total - (true_positives +
                              false_negatives +
                              false_positives)

    actual_negatives = true_negatives + false_positives

    return true_negatives / actual_negatives
