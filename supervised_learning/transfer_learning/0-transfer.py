#!/usr/bin/env python3
import tensorflow.keras as K
import numpy as np


def preprocess_data(X, Y):
    """
    Preprocess CIFAR-10 data
    """
    # Normalize images
    X_p = X.astype('float32') / 255.0

    # One-hot encode labels
    Y = Y.reshape(-1)
    Y_p = K.utils.to_categorical(Y, 10)

    return X_p, Y_p


def build_base_model(input_shape=(32, 32, 3)):
    """
    Build base model using MobileNetV2
    """
    base_model = K.applications.MobileNetV2(
        input_shape=(96, 96, 3),  # upscale target
        include_top=False,
        weights='imagenet'
    )

    # Freeze most layers
    for layer in base_model.layers:
        layer.trainable = False

    inputs = K.Input(shape=input_shape)

    # Resize CIFAR-10 images to 96x96
    x = K.layers.Lambda(
        lambda img: K.backend.resize_images(img, 3, 3, "channels_last")
    )(inputs)

    # Preprocess for MobileNetV2
    x = K.applications.mobilenet_v2.preprocess_input(x)

    x = base_model(x, training=False)
    x = K.layers.GlobalAveragePooling2D()(x)

    return inputs, x, base_model


def train_model():
    """
    Train the transfer learning model
    """
    # Load CIFAR-10
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    inputs, features, base_model = build_base_model()

    # Add classifier on top
    x = K.layers.Dense(256, activation='relu')(features)
    x = K.layers.Dropout(0.5)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    model = K.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train only top layers first
    model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=10,
        batch_size=128,
        verbose=1
    )

    # Optional fine-tuning (unfreeze top layers)
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=10,
        batch_size=128,
        verbose=1
    )

    # Save compiled model
    model.save('cifar10.h5')


if __name__ == "__main__":
    train_model()
