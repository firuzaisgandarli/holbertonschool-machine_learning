#!/usr/bin/env python3
"""Neural Style Transfer with Variational Cost."""

import numpy as np
import tensorflow as tf


class NST:
    """Neural Style Transfer class."""

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]

    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image,
                 alpha=1e4, beta=1, var=10):
        """Initialize NST."""

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
            raise TypeError("alpha must be a non-negative number")

        if (not isinstance(beta, (int, float)) or beta < 0):
            raise TypeError("beta must be a non-negative number")

        if (not isinstance(var, (int, float)) or var < 0):
            raise TypeError("var must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)

        self.alpha = alpha
        self.beta = beta
        self.var = var

        self.model = self.load_model()

        self.generate_features()

    @staticmethod
    def scale_image(image):
        """Resize image to max side 512 and scale to [0,1]."""
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

        return tf.expand_dims(image, axis=0)

    def load_model(self):
        """Load VGG19 model for feature extraction."""
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )
        vgg.trainable = False

        outputs = [vgg.get_layer(l).output for l in self.style_layers]
        outputs.append(vgg.get_layer(self.content_layer).output)

        return tf.keras.Model(inputs=vgg.input, outputs=outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """Compute Gram matrix."""
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable)) or
                len(input_layer.shape) != 4):
            raise TypeError("input_layer must be a tensor of rank 4")

        gram = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)
        h = tf.cast(input_layer.shape[1], tf.float32)
        w = tf.cast(input_layer.shape[2], tf.float32)

        return gram / (h * w)

    def generate_features(self):
        """Extract style and content features."""
        style = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255
        )
        content = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255
        )

        style_outputs = self.model(style)
        content_outputs = self.model(content)

        self.gram_style_features = [
            self.gram_matrix(s) for s in style_outputs[:-1]
        ]

        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """Compute style cost for one layer."""
        if (not isinstance(style_output, (tf.Tensor, tf.Variable)) or
                len(style_output.shape) != 4):
            raise TypeError("style_output must be a tensor of rank 4")

        c = style_output.shape[-1]

        if (not isinstance(gram_target, (tf.Tensor, tf.Variable)) or
                gram_target.shape != (1, c, c)):
            raise TypeError(
                f"gram_target must be a tensor of shape [1, {c}, {c}] "
                f"where {c} is the number of channels in style_output"
            )

        gram_style = self.gram_matrix(style_output)

        return tf.reduce_sum(tf.square(gram_style - gram_target)) / (c ** 2)

    def style_cost(self, style_outputs):
        """Compute total style cost."""
        if (not isinstance(style_outputs, list) or
                len(style_outputs) != len(self.style_layers)):
            raise TypeError(
                f"style_outputs must be a list with a length of "
                f"{len(self.style_layers)}"
            )

        total = 0
        w = 1 / len(self.style_layers)

        for s, g in zip(style_outputs, self.gram_style_features):
            total += w * self.layer_style_cost(s, g)

        return total

    def content_cost(self, content_output):
        """Compute content cost."""
        if (not isinstance(content_output, (tf.Tensor, tf.Variable)) or
                content_output.shape != self.content_feature.shape):
            raise TypeError(
                f"content_output must be a tensor of shape "
                f"{self.content_feature.shape}"
            )

        h, w, c = self.content_feature.shape[1:]

        return tf.reduce_sum(
            tf.square(content_output - self.content_feature)
        ) / (h * w * c)

    @staticmethod
    def variational_cost(generated_image):
        """Compute total variation loss (FIXED)."""

        # ✅ IMPORTANT FIX: handle rank properly
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                len(generated_image.shape) not in (3, 4)):
            raise TypeError(
                "image must be a tensor of rank 3 or 4"
            )

        if len(generated_image.shape) == 3:
            generated_image = tf.expand_dims(generated_image, axis=0)

        x_diff = generated_image[:, :, 1:, :] - generated_image[:, :, :-1, :]
        y_diff = generated_image[:, 1:, :, :] - generated_image[:, :-1, :, :]

        return tf.reduce_sum(tf.abs(x_diff)) + tf.reduce_sum(tf.abs(y_diff))

    def total_cost(self, generated_image):
        """Compute full loss."""
        gen = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255
        )

        outputs = self.model(gen)

        style_outputs = outputs[:-1]
        content_output = outputs[-1]

        J_style = self.style_cost(style_outputs)
        J_content = self.content_cost(content_output)
        J_var = self.variational_cost(generated_image)

        J = self.alpha * J_content + self.beta * J_style + self.var * J_var

        return J, J_content, J_style, J_var

    def compute_grads(self, generated_image):
        """Compute gradients."""
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                generated_image.shape != self.content_image.shape):
            raise TypeError(
                f"generated_image must be a tensor of shape "
                f"{self.content_image.shape}"
            )

        with tf.GradientTape() as tape:
            J, Jc, Js, Jv = self.total_cost(generated_image)

        grads = tape.gradient(J, generated_image)

        return grads, J, Jc, Js, Jv

    def generate_image(self, iterations=1000, step=None,
                       lr=0.01, beta1=0.9, beta2=0.99):
        """Run Adam optimization."""

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")

        if step is not None:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError(
                    "iterations must be positive and less than iterations"
                )

        if not isinstance(lr, (int, float)):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")

        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if not 0 <= beta1 <= 1:
            raise ValueError("beta1 must be in the range [0, 1]")

        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if not 0 <= beta2 <= 1:
            raise ValueError("beta2 must be in the range [0, 1]")

        generated = tf.Variable(self.content_image)
        opt = tf.keras.optimizers.Adam(lr, beta1, beta2)

        best_cost = float('inf')
        best_img = generated.numpy()

        for i in range(iterations + 1):

            grads, J, Jc, Js, Jv = self.compute_grads(generated)
            opt.apply_gradients([(grads, generated)])

            generated.assign(tf.clip_by_value(generated, 0, 1))

            if J < best_cost:
                best_cost = J
                best_img = generated.numpy()

            if step is not None and (i % step == 0 or i == iterations):
                print(
                    f"Cost at iteration {i}: {J}, "
                    f"content {Jc}, style {Js}, var {Jv}"
                )

        return best_img[0], best_cost
