#!/usr/bin/env python3
"""
Builds a modified version of the LeNet-5 archite"""
from tensorflow import keras as K


def lenet5(X):
    """
    a function that builds a modified LeNet-5 using tensorflow
    :param X: tf.placeholder of shape (m, 28, 28, 1) containing the input
    images for the network
        m is the number of images
    :return:
        a tensor for the softmax activated output
        a training operation that utilizes Adam optimization (with default
        hyperparameters)
        a tensor for the loss of the network
        a tensor for the accuracy of the network
    """
    he_init = K.initializers.HeNormal(seed=0)

    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding="same",
        activation="relu",
        kernel_initializer=he_init,
    )(X)

    pool1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
    )(conv1)

    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding="valid",
        activation="relu",
        kernel_initializer=he_init,
    )(pool1)

    pool2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
    )(conv2)

    flat = K.layers.Flatten()(pool2)

    fc1 = K.layers.Dense(
        units=120,
        activation="relu",
        kernel_initializer=he_init,
    )(flat)

    fc2 = K.layers.Dense(
        units=84,
        activation="relu",
        kernel_initializer=he_init,
    )(fc1)

    y = K.layers.Dense(
        units=10,
        activation="softmax",
        kernel_initializer=he_init,
    )(fc2)

    model = K.Model(inputs=X, outputs=y)
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model
