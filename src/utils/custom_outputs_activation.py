from functools import partial
import tensorflow as tf


class CustomOutputActivation:

    def __init__(
            self,
            type: str = "clip",
            parameters: dict = None
    ):

        self.type = type
        if parameters is None:
            self._parameters = {}
        else:
            self._parameters = parameters
        self._set_activation()

    def _set_activation(self):
        if self.type is None:
            self._activation = None
        elif self.type == "clip":
            self._activation = partial(clip, **self._parameters)
        elif self.type == "sigmoid":
            self._activation = partial(sigmoid, **self._parameters)
        elif self.type == "sigmoid_shifted":
            self._activation = partial(sigmoid_shifted, **self._parameters)
        elif self.type == "tanh":
            self._activation = partial(tanh, **self._parameters)
        else:
            raise NotImplementedError

    def activation(self, x):
        if self._activation is None:
            return x
        return self._activation(x)


def sigmoid(x, a=1.0):
    if a == 1.0:
        return tf.sigmoid(x)
    return sigmoid_shifted(x, a=a)


def tanh(x):
    return tf.tanh(x)


def sigmoid_shifted(x, shift=0.0, a=1.0):
    return 1 / (1 + tf.exp(-a * (x - shift)))


def clip(x, cmin=0, cmax=1):
    return tf.clip_by_value(x, cmin, cmax)


def normalization_min_max(x, axis=None):
    return (x - tf.reduce_min(x, axis=axis)) / \
           (tf.reduce_max(x, axis=axis) - tf.reduce_min(x, axis=axis))


def scaling_01_m11(x):
    return (x - 0.5) / 0.5


def scaling_m11_01(x):
    return x * 0.5 + 0.5
