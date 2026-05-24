#!/usr/bin/env python3
"""Neural Style Transfer module."""

import numpy as np
import tensorflow as tf


class NST:
    """Performs tasks for neural style transfer."""

    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image,
                 alpha=1e4, beta=1):
        """
        Initialize NST instance.

        Args:
            style_image (np.ndarray): Style reference image.
            content_image (np.ndarray): Content reference image.
            alpha (float): Weight for content cost.
            beta (float): Weight for style cost.
        """
        if (not isinstance(style_image, np.ndarray) or
                len(style_image.shape) != 3 or
                style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (not isinstance(content_image, np.ndarray) or
                len(content_image.shape) != 3 or
                content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (not isinstance(alpha, (int, float)) or alpha < 0):
            raise TypeError(
                "alpha must be a non-negative number"
            )

        if (not isinstance(beta, (int, float)) or beta < 0):
            raise TypeError(
                "beta must be a non-negative number"
            )

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)

        self.alpha = alpha
        self.beta = beta

        self.model = self.load_model()

    @staticmethod
    def scale_image(image):
        """
        Rescale an image.

        Args:
            image (np.ndarray): Image of shape (h, w, 3).

        Returns:
            tf.Tensor: Scaled image tensor.
        """
        if (not isinstance(image, np.ndarray) or
                len(image.shape) != 3 or
                image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape
        max_dim = 512

        if h > w:
            new_h = max_dim
            new_w = int(w * max_dim / h)
        else:
            new_w = max_dim
            new_h = int(h * max_dim / w)

        image = tf.convert_to_tensor(image, dtype=tf.float32)

        image = tf.image.resize(
            image,
            (new_h, new_w),
            method=tf.image.ResizeMethod.BICUBIC
        )

        image = image / 255.0

        image = tf.clip_by_value(image, 0.0, 1.0)

        image = tf.expand_dims(image, axis=0)

        return image

    def load_model(self):
        """
        Create the model used to calculate cost.

        Returns:
            tf.keras.Model: The NST model.
        """
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )

        vgg.trainable = False

        outputs = []

        for layer_name in self.style_layers:
            outputs.append(vgg.get_layer(layer_name).output)

        outputs.append(
            vgg.get_layer(self.content_layer).output
        )

        model = tf.keras.models.Model(
            inputs=vgg.input,
            outputs=outputs
        )

        model.trainable = False

        return model

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculate the gram matrix of an input layer.

        Args:
            input_layer (tf.Tensor or tf.Variable):
                Tensor of shape (1, h, w, c).

        Returns:
            tf.Tensor:
                Gram matrix of shape (1, c, c).
        """
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable)) or
                len(input_layer.shape) != 4):
            raise TypeError(
                "input_layer must be a tensor of rank 4"
            )

        gram = tf.linalg.einsum(
            'bijc,bijd->bcd',
            input_layer,
            input_layer
        )

        h = input_layer.shape[1]
        w = input_layer.shape[2]

        return gram / (h * w)
