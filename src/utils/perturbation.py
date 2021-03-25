import tensorflow as tf
import numpy as np


def blur_perturbation(image, kernel_size=32, sigma=5):
    kernel_list = gaussian_kernel_2d_separable(sigma=sigma, kernel_size=kernel_size)
    return convolve_volume_separable(image, kernel_list)


def zero_perturbation(image):
    return tf.zeros(image.shape, dtype=tf.float32)


def convolve_volume_separable(volume, kernel_list):
    res = tf.nn.conv2d(input=volume,
                       filters=kernel_list[0],
                       strides=[1, 1, 1, 1],
                       padding="SAME",
                       data_format="NCHW")
    res = tf.nn.conv2d(input=res,
                       filters=kernel_list[1],
                       strides=[1, 1, 1, 1],
                       padding="SAME",
                       data_format="NCHW")
    return res


def gaussian_kernel_2d_separable(sigma, kernel_size):
    x = np.arange(-(kernel_size // 2), (kernel_size // 2) + 1, 1)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
    return tf.constant(kernel.reshape(kernel.shape[0], 1, 1, 1), dtype=tf.float32), \
           tf.constant(kernel.reshape(1, kernel.shape[0], 1, 1), dtype=tf.float32)