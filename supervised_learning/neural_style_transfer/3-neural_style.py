#!/usr/bin/env python3
import numpy as np
import tensorflow as tf


class NST:
    """
    Neural Style Transfer class
    """

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Constructor
        """
        if not isinstance(style_image, np.ndarray) or style_image.shape[-1] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(content_image, np.ndarray) or content_image.shape[-1] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.alpha = alpha
        self.beta = beta

        # Preprocess images (important: float conversion)
        style_image = style_image.astype("float32")
        content_image = content_image.astype("float32")

        self.style_image = tf.keras.applications.vgg19.preprocess_input(
            style_image * 255
        )
        self.content_image = tf.keras.applications.vgg19.preprocess_input(
            content_image * 255
        )

        # Load VGG19 model
        vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")

        vgg.trainable = False

        # Style layers (as required by Holberton project)
        style_layers = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ]

        # Content layer
        content_layer = "block5_conv2"

        outputs = [
            vgg.get_layer(name).output for name in style_layers + [content_layer]
        ]

        self.model = tf.keras.Model([vgg.input], outputs)

        self.style_layers = style_layers
        self.content_layer = content_layer

        # Extract features immediately
        self.generate_features()

    def gram_matrix(self, input_tensor):
        """Computes Gram matrix"""
        result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(
            input_shape[1] * input_shape[2], tf.float32
        )
        return result / num_locations

    def generate_features(self):
        """
        Extract style and content features
        """
        outputs = self.model(
            tf.convert_to_tensor(self.style_image[tf.newaxis, ...])
        )

        style_outputs = outputs[:-1]
        content_output = outputs[-1]

        self.gram_style_features = [
            self.gram_matrix(style_output) for style_output in style_outputs
        ]

        self.content_feature = content_output

        return self.gram_style_features, self.content_feature
