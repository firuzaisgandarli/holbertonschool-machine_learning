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
        Create the NST model.
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
        Calculate gram matrix.
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
        style_input = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255.0
        )

        content_input = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255.0
        )

        style_outputs = self.model(style_input)
        content_outputs = self.model(content_input)

        self.gram_style_features = [
            self.gram_matrix(style_output)
            for style_output in style_outputs[:-1]
        ]

        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculate style cost for one layer.
        """
        if (not isinstance(style_output, (tf.Tensor, tf.Variable)) or
                len(style_output.shape) != 4):
            raise TypeError(
                "style_output must be a tensor of rank 4"
            )

        c = style_output.shape[-1]

        if (not isinstance(gram_target, (tf.Tensor, tf.Variable)) or
                gram_target.shape != (1, c, c)):
            raise TypeError(
                "gram_target must be a tensor of shape "
                "[1, {}, {}] where {} is the number "
                "of channels in style_output".format(c, c, c)
            )

        gram_style = self.gram_matrix(style_output)

        return tf.reduce_sum(
            tf.square(gram_style - gram_target)
        ) / (c ** 2)

    def style_cost(self, style_outputs):
        """
        Calculate total style cost.
        """
        if (not isinstance(style_outputs, list) or
                len(style_outputs) != len(self.style_layers)):
            raise TypeError(
                "style_outputs must be a list with a length "
                "of {}".format(len(self.style_layers))
            )

        weight = 1 / len(self.style_layers)

        cost = 0

        for style_output, gram_target in zip(
                style_outputs,
                self.gram_style_features):

            cost += weight * self.layer_style_cost(
                style_output,
                gram_target
            )

        return cost

    def content_cost(self, content_output):
        """
        Calculate content cost.
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

        return tf.reduce_sum(
            tf.square(content_output - self.content_feature)
        ) / (h * w * c)

    def total_cost(self, generated_image):
        """
        Calculate total cost.
        """
        preprocessed = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255.0
        )

        outputs = self.model(preprocessed)

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
        Compute gradients.
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

    def generate_image(self, iterations=1000, step=None,
                       lr=0.01, beta1=0.9, beta2=0.99):
        """
        Generate the neural style transferred image.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")

        if iterations <= 0:
            raise ValueError("iterations must be positive")

        if step is not None:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")

            if step <= 0 or step > iterations:
                raise ValueError(
                    "step must be positive and less than iterations"
                )

        if not isinstance(lr, (float, int)):
            raise TypeError("lr must be a number")

        if lr <= 0:
            raise ValueError("lr must be positive")

        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")

        if beta1 < 0 or beta1 > 1:
            raise ValueError(
                "beta1 must be in the range [0, 1]"
            )

        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")

        if beta2 < 0 or beta2 > 1:
            raise ValueError(
                "beta2 must be in the range [0, 1]"
            )

        generated_image = tf.Variable(self.content_image)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr,
            beta_1=beta1,
            beta_2=beta2
        )

        best_cost = float('inf')
        best_image = None

        for i in range(iterations + 1):

            grads, J_total, J_content, J_style = (
                self.compute_grads(generated_image)
            )

            optimizer.apply_gradients(
                [(grads, generated_image)]
            )

            clipped = tf.clip_by_value(
                generated_image,
                0.0,
                1.0
            )

            generated_image.assign(clipped)

            if J_total < best_cost:
                best_cost = J_total
                best_image = generated_image.numpy()

            if step is not None:
                if i % step == 0 or i == iterations:
                    print(
                        "Cost at iteration {}: {}, content {}, "
                        "style {}".format(
                            i,
                            J_total,
                            J_content,
                            J_style
                        )
                    )

        return best_image[0], best_cost
