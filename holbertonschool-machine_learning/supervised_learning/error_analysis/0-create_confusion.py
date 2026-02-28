#!/usr/bin/env python3
"""
Module that creates a confusion matrix
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix

    Parameters:
    labels (numpy.ndarray): one-hot array of shape (m, classes)
                            with correct labels
    logits (numpy.ndarray): one-hot array of shape (m, classes)
                            with predicted labels

    Returns:
    numpy.ndarray: confusion matrix of shape (classes, classes)
                   where rows represent correct labels and
                   columns represent predicted labels
    """
    true_classes = np.argmax(labels, axis=1)
    pred_classes = np.argmax(logits, axis=1)

    classes = labels.shape[1]
    confusion = np.zeros((classes, classes))

    for i in range(len(true_classes)):
        confusion[true_classes[i], pred_classes[i]] += 1

    return confusion
