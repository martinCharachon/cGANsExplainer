from functools import partial
import tensorflow as tf


class ReconstructionLoss:

    def __init__(
            self,
            coefficient=1,
            loss_type="mse_loss",
            parameters=None
    ):
        self._coef = coefficient
        self._loss_type = loss_type
        if parameters is None:
            self._params = {}
        else:
            self._params = parameters
        self._set_reconstruction_loss()

    def _set_reconstruction_loss(self):
        if self._loss_type == "mse_loss":
            self._reconstruction_loss = mse_loss
        elif self._loss_type == "mae_loss":
            self._reconstruction_loss = mae_loss
        elif self._loss_type == "bce_loss":
            self._reconstruction_loss = bce_loss
        elif self._loss_type == "mse_mae_loss":
            self._reconstruction_loss = partial(
                mse_mae_loss, **self._params)
        elif self._loss_type is None:
            self._reconstruction_loss = lambda x, y: tf.zeros(1, dtype=tf.float32)
        else:
            raise KeyError

    def reconstruction_loss(self, x1, x2):
        return self._coef * tf.reduce_mean(self._reconstruction_loss(x1, x2))


class LatentSeparationLoss:
    def __init__(self,
                 coefficient=1.0,
                 loss_type="kl_divergence",
                 parameters={}):
        self._coef = coefficient
        self._loss_type = loss_type
        self._params = parameters
        self._set_loss()

    def _set_loss(self):
        if self._loss_type == "kl_divergence":
            self._loss = kl_divergence_loss
        elif self._loss_type == "mse_loss":
            self._loss = mse_loss
        else:
            raise NotImplementedError

    def loss(self, inp1, inp2):
        return self._coef * tf.reduce_mean(self._loss(inp1, inp2))


class ReconstructionMetrics:
    def __init__(
            self,
            metrics_type_list=None
    ):
        self._metrics_type_list = metrics_type_list
        self._set_reconstruction_metrics_list()

    def _set_reconstruction_metrics_list(self):
        self._reconstruction_metrics_list = []
        if self._metrics_type_list is None:
            self._reconstruction_metrics_list.append(lambda x, xd: 0)
        elif len(self._metrics_type_list) == 0:
            self._reconstruction_metrics_list.append(lambda x, xd: 0)
        else:
            if "mse" in self._metrics_type_list:
                self._reconstruction_metrics_list.append(mse_loss)
            if "mae" in self._metrics_type_list:
                self._reconstruction_metrics_list.append(mae_loss)

    def reconstruction_metrics(self, x, x_decoded):
        results = []
        for metrics in self._reconstruction_metrics_list:
            results.append(metrics(x, x_decoded))
        return results


def mse_loss(y_true, y_pred):
    return tf.keras.losses.MeanSquaredError()(y_true, y_pred)


def mae_loss(y_true, y_pred):
    return tf.reduce_mean(tf.keras.losses.MAE(y_true, y_pred))


def mse_mae_loss(y_true, y_pred, coef_mse=1.0, coef_mae=1.0):
    return coef_mse * tf.keras.losses.MeanSquaredError()(y_true, y_pred) + \
           coef_mae * tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)


def bce_loss(y_true, y_pred):
    return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)


def kl_divergence_loss(mean, logvar, coefficient=1.0):
    latent_loss = -0.5 * (1 + logvar - tf.square(mean) - tf.exp(logvar))
    latent_loss = tf.reduce_sum(latent_loss, axis=1)
    latent_loss = tf.reduce_mean(latent_loss, axis=0)
    return coefficient * latent_loss

