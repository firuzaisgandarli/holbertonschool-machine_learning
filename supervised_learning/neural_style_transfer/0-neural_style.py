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
        Initialize the NST instance.

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

        if (not isinstance(alpha, (int, float)) or
                alpha < 0):
            raise TypeError(
                "alpha must be a non-negative number"
            )

        if (not isinstance(beta, (int, float)) or
                beta < 0):
            raise TypeError(
                "beta must be a non-negative number"
            )

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)

        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        Rescales an image.

        The image is resized so that its largest side is 512 pixels
        while maintaining aspect ratio. Pixel values are scaled
        to the range [0, 1].

        Args:
            image (np.ndarray): Image of shape (h, w, 3).

        Returns:
            tf.Tensor: Scaled image tensor of shape
            (1, h_new, w_new, 3).
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
