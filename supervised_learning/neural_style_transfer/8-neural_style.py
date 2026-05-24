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

        self.model = self.load_model()

        self.generate_features()

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

    def generate_features(self):
        """
        Extract style and content features.
        """
        style_input = self.style_image * 255.0
        content_input = self.content_image * 255.0

        style_input = tf.keras.applications.vgg19.preprocess_input(
            style_input
        )

        content_input = tf.keras.applications.vgg19.preprocess_input(
            content_input
        )

        style_outputs = self.model(style_input)
        content_outputs = self.model(content_input)

        style_features = style_outputs[:-1]

        self.gram_style_features = [
            self.gram_matrix(style_feature)
            for style_feature in style_features
        ]

        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculate style cost for a single layer.

        Args:
            style_output (tf.Tensor or tf.Variable):
                Style layer output of generated image.
            gram_target (tf.Tensor or tf.Variable):
                Gram matrix target.

        Returns:
            tf.Tensor: Layer style cost.
        """
        if (not isinstance(style_output, (tf.Tensor, tf.Variable)) or
                len(style_output.shape) != 4):
            raise TypeError(
                "style_output must be a tensor of rank 4"
            )

        c = style_output.shape[-1]

        expected_shape = [1, c, c]

        if (not isinstance(gram_target, (tf.Tensor, tf.Variable)) or
                gram_target.shape != expected_shape):
            raise TypeError(
                "gram_target must be a tensor of shape "
                "[1, {}, {}] where {} is the number "
                "of channels in style_output".format(c, c, c)
            )

        gram_style = self.gram_matrix(style_output)

        style_cost = tf.reduce_sum(
            tf.square(gram_style - gram_target)
        ) / (c ** 2)

        return style_cost

    def style_cost(self, style_outputs):
        """
        Calculate the total style cost.

        Args:
            style_outputs (list): List of style output tensors.

        Returns:
            tf.Tensor: Total style cost.
        """
        if (not isinstance(style_outputs, list) or
                len(style_outputs) != len(self.style_layers)):
            raise TypeError(
                "style_outputs must be a list with a length "
                "of {}".format(len(self.style_layers))
            )

        weight = 1 / len(self.style_layers)

        style_cost = 0

        for style_output, gram_target in zip(
                style_outputs,
                self.gram_style_features):

            layer_cost = self.layer_style_cost(
                style_output,
                gram_target
            )

            style_cost += weight * layer_cost

        return style_cost

    def content_cost(self, content_output):
        """
        Calculate the content cost.

        Args:
            content_output (tf.Tensor or tf.Variable):
                Content output tensor for generated image.

        Returns:
            tf.Tensor: Content cost.
        """
        expected_shape = self.content_feature.shape

        if (not isinstance(content_output, (tf.Tensor, tf.Variable)) or
                content_output.shape != expected_shape):
            raise TypeError(
                "content_output must be a tensor of shape {}".format(
                    expected_shape
                )
            )

        h = expected_shape[1]
        w = expected_shape[2]
        c = expected_shape[3]

        content_cost = tf.reduce_sum(
            tf.square(content_output - self.content_feature)
        ) / (h * w * c)

        return content_cost

    def total_cost(self, generated_image):
        """
        Calculate total cost.

        Args:
            generated_image (tf.Tensor or tf.Variable):
                Generated image tensor.

        Returns:
            tuple: (total_cost, content_cost, style_cost)
        """
        generated_input = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255.0
        )

        outputs = self.model(generated_input)

        style_outputs = outputs[:-1]
        content_output = outputs[-1]

        J_style = self.style_cost(style_outputs)
        J_content = self.content_cost(content_output)

        J_total = (self.alpha * J_content) + (
            self.beta * J_style
        )

        return J_total, J_content, J_style

    def compute_grads(self, generated_image):
        """
        Compute gradients for generated image.

        Args:
            generated_image (tf.Tensor or tf.Variable):
                Generated image tensor.

        Returns:
            tuple:
                gradients, J_total, J_content, J_style
        """
        expected_shape = self.content_image.shape

        if (not isinstance(generated_image,
                           (tf.Tensor, tf.Variable)) or
                generated_image.shape != expected_shape):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(
                    expected_shape
                )
            )

        with tf.GradientTape() as tape:
            J_total, J_content, J_style = self.total_cost(
                generated_image
            )

        gradients = tape.gradient(
            J_total,
            generated_image
        )

        return gradients, J_total, J_content, J_style
