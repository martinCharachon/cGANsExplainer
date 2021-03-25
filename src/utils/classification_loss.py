from functools import partial
from typing import Union, List
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


class ClassificationLossOperator:

    def __init__(
            self,
            loss_type: str = "bce_loss",
            coefficient: int = 1.0,
            parameters: dict = None
    ):

        self._coefficient = coefficient
        self._loss_type = loss_type
        if parameters is None:
            self._parameters = {}
        else:
            self._parameters = parameters
        self._set_classification_loss()

    def _set_classification_loss(self):
        if self._loss_type is None:
            self._classification_loss = lambda yt, yp: 0
        elif self._loss_type == "bce_loss":
            self._classification_loss = partial(bce_loss, **self._parameters)
        elif self._loss_type == "bce_with_logits_loss":
            self._classification_loss = partial(bce_with_logits_loss, **self._parameters)
        elif self._loss_type == "negative bce_loss":
            self._classification_loss = partial(neg_bce_loss, **self._parameters)
        elif self._loss_type == "adversarial bce_loss":
            self._classification_loss = partial(adversarial_bce_loss, **self._parameters)
        elif self._loss_type == "weighted_bce_loss":
            self._classification_loss = partial(
                weighted_bce_loss, **self._parameters)
        else:
            raise NotImplementedError

    def classification_loss(self, y1, y2):
        if self._coefficient == 0:
            return tf.reduce_mean(tf.zeros(1, dtype=tf.float32))
        return tf.reduce_mean(self._coefficient * self._classification_loss(y1, y2))


class ClassificationMetricsOperator:

    def __init__(
            self,
            metrics_type: str = "binary_accuracy",
            parameters: dict = None
    ):

        self._metrics_type = metrics_type
        if parameters is None:
            self._parameters = {}
        else:
            self._parameters = parameters
        self._set_classification_metrics()

    def _set_classification_metrics(self):
        if self._metrics_type is None:
            self._classification_metrics = lambda yt, yp: 0
        elif self._metrics_type == "binary_accuracy":
            self._classification_metrics = partial(
                tf.keras.metrics.binary_accuracy, **self._parameters)
        elif self._metrics_type == "custom_binary_accuracy":
            self._classification_metrics = partial(
                custom_binary_accuracy, **self._parameters)
        elif self._metrics_type == "custom_binary_accuracy_with_logits":
            self._classification_metrics = partial(
                custom_binary_accuracy_with_logits, **self._parameters)
        elif self._metrics_type == "mae":
            self._classification_metrics = partial(
                tf.keras.losses.MeanAbsoluteError(), **self._parameters)
        else:
            raise NotImplementedError

    def classification_metrics(self, y1, y2):
        return self._classification_metrics(y1, y2)


def bce_loss(y_true, y_pred):
    return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)


def adversarial_bce_loss(y_true, y_pred):
    y_true = 1 - y_true
    return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)


def neg_bce_loss(y_true, y_pred):
    return - tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)


def weighted_bce_loss(y_true, y_pred, w_class=None):
    if w_class is None:
        w_class = [1, 1]
    return WeightedBinaryCrossEntropy(w_class=w_class)(y_true, y_pred)


def bce_with_logits_loss(y_true, y_pred):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true)


def custom_binary_accuracy(y_true, y_pred, threshold=0.5):
    y_true = tf.where(y_true > threshold, 1.0, 0.0)
    y_pred = tf.where(y_pred > threshold, 1.0, 0.0)
    return tf.reduce_mean(y_true * y_pred + (1 - y_true) * (1 - y_pred))


def custom_binary_accuracy_with_logits(y_true, logits_pred, threshold=0.5):
    y_pred = tf.sigmoid(logits_pred)
    y_true = tf.where(y_true > threshold, 1.0, 0.0)
    y_pred = tf.where(y_pred > threshold, 1.0, 0.0)
    return tf.reduce_mean(y_true * y_pred + (1 - y_true) * (1 - y_pred))


class WeightedBinaryCrossEntropy:
    def __init__(self, w_class: Union[np.ndarray, List], backend=K):
        self._backend = backend
        if self._backend == np:
            self.w_class_0 = np.array(w_class[0])
            self.w_class_1 = np.array(w_class[1])
        else:
            self.w_class_0 = K.variable(w_class[0])
            self.w_class_1 = K.variable(w_class[1])

    def __call__(self, y_true, y_pred):
        y_true = self._backend.clip(y_true, self._backend.epsilon(), 1 - self._backend.epsilon())
        y_pred = self._backend.clip(y_pred, self._backend.epsilon(), 1 - self._backend.epsilon())
        logloss = -(y_true * self._backend.log(y_pred) * self.w_class_1 + (
                1 - y_true) * self._backend.log(
            1 - y_pred) * self.w_class_0)
        loss = self._backend.mean(logloss, axis=-1)
        return self._backend.mean(loss, axis=0)



