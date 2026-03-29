#!/usr/bin/env python3
'''same convolution on grayscale images'''
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    '''
    images is a numpy.ndarray with shape (m, h, w) containing multiple
    grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for th
    convolution
        kh is the height of the kernel
        kw is the width of the kernel
    Returns: a numpy.ndarray containing the convolved images'''
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    images = np.pad(images,
                    ((0, 0), (padding[0], padding[0]),
                        (padding[1], padding[1])),
                    constant_values=0)
    conv = np.zeros((m, images.shape[1] - kh + 1, images.shape[2] - kw + 1))
    for row in range(images.shape[1] - kh + 1):
        for column in range(images.shape[2] - kw + 1):
            part = images[:, row:row + kh, column:column + kw] * kernel
            conv[:, row, column] = np.sum(part, axis=(1, 2))
    return conv
