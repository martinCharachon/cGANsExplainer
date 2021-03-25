import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from ..utils.data_generator import DataGeneratorFromH5File
from .optimization_base import OptimizationBase
from ..utils.reconstruction_loss import ReconstructionMetrics, ReconstructionLoss
from ..utils.custom_outputs_activation import CustomOutputActivation


class OptimizationInterpreterBase(OptimizationBase):
    """
    Model operator
    """

    def __init__(self,
                 reference_model: keras.models.Model = None,
                 model_config: dict = {},
                 training_config: dict = {},
                 training_manager: DataGeneratorFromH5File = None,
                 validation_manager: DataGeneratorFromH5File = None
                 ):
        super().__init__(model_config, training_config, training_manager, validation_manager)
        if "ground_truth_type" not in list(self.training_config.keys()):
            self.training_config["ground_truth_type"] = 2
        self._ground_truth_type = self.training_config["ground_truth_type"]
        self.reference_model = reference_model
        self._tr_step, self._val_step = 0, 0
        self._training_loss = []
        self._validation_loss = []
        self._validation_reconstruction_metrics = []
        self._validation_classification_metrics = []
        self._logs_file = "logs.json"
        self._wait = 0
        self._wait_plateau = 0
        self._patience_plateau = np.Inf
        self._best_criteria = np.Inf
        self.model_threshold = None
        if "model_threshold" in list(self.training_config["loss"].keys()):
            self.model_threshold = self.training_config["loss"]["model_threshold"]
        self._set_patience()
        self._set_tools()
        configuration = {
            "training": training_config,
            "model": model_config
        }
        with open(os.path.join(self.training_config["outputs_directory"],
                               "configuration.json"), "w") as f:
            json.dump(configuration, f, indent=2)

    def _set_patience(self):
        if "ReduceLROnPlateau" in self.training_config["call_backs"]:
            self._patience_plateau = \
                self.training_config["call_backs"]["ReduceLROnPlateau"]["patience"]
            self.min_lr = self.training_config["call_backs"]["ReduceLROnPlateau"]["min_lr"]
            self.factor = self.training_config["call_backs"]["ReduceLROnPlateau"]["factor"]

    def _set_tools(self):
        super()._set_tools()
        self._set_reconstruction_loss()
        self._set_reconstruction_metrics()
        self._set_image_activation()

    def _set_ground_truth(self, X_batch, y_batch):
        if self._ground_truth_type == 0:
            return y_batch
        elif self._ground_truth_type == 1:
            return self.reference_model(X_batch)
        elif self._ground_truth_type == 2:
            y = self.reference_model(X_batch)
            if self.model_threshold is not None:
                return tf.cast(tf.where(y >= self.model_threshold, 1.0, 0.0),
                               dtype=tf.float32)
            else:
                return tf.round(y)

    def _fit_generator(self, initial_epoch=0):
        self._epoch = initial_epoch
        while self._epoch < self.training_config["epochs"]:
            print(f"Epoch {self._epoch + 1}/{self.training_config['epochs']}")
            if self._stop_training():
                break
            self._epoch_step()
            self._epoch += 1

    def _epoch_initialization(self):
        self._tr_step, self._val_step = 0, 0
        self._training_loss = []
        self._validation_loss = []
        self._validation_classification_metrics = []
        self._validation_reconstruction_metrics = []

    def _epoch_closing(self):
        self._set_callbacks_on_epoch_end()

    def _get_next_batch(self, generator, batch_size=None):
        if batch_size is None:
            batch_size = self.training_config["batch_size"]
        X_batch, y_batch = next(generator)
        if X_batch.shape[0] != batch_size:
            while X_batch.shape[0] != batch_size:
                X_batch, y_batch = next(generator)
        X_batch = normalize_data(X_batch)
        return X_batch, y_batch

    def _set_callbacks_on_epoch_end(self):
        loss = self._get_mean_validation_criteria()
        self._check_reduce_lr_on_plateau(loss)
        self._set_model_checkpoint(loss)
        self._write_logs_to_json()

    def build_model(self):
        pass

    def _set_model_checkpoint(self, curent_criteria):
        pass

    def _get_mean_validation_criteria(self):
        mean_loss = np.mean(self._validation_loss, axis=0)[0]
        return mean_loss

    def _check_reduce_lr_on_plateau(self, current_criteria):
        pass

    def _stop_training(self):
        pass

    def _save_model_weights(self, directory):
        pass

    def _write_logs_to_json(self):
        pass

    def _set_reconstruction_loss(self):
        self._reconstruction_loss_operator = ReconstructionLoss(
            **self.training_config["loss"]["reconstruction"]["parameters"])
        self._reconstruction_loss = self._reconstruction_loss_operator.reconstruction_loss

    def _set_reconstruction_metrics(self):
        self._representation_metrics_operator = ReconstructionMetrics(["mse"])
        self._reconstruction_metrics = self._representation_metrics_operator.reconstruction_metrics

    def _set_image_activation(self):
        if "image_activation" in list(self.model_config):
            self._image_activation_operator = \
                CustomOutputActivation(**self.model_config["image_activation"])
        else:
            self._image_activation_operator = CustomOutputActivation(type="clip")
        self._image_activation = self._image_activation_operator.activation


def normalize_data(X_batch, eps=1e-10):
    if np.min(X_batch) + eps < np.max(X_batch):
        X_batch = (X_batch - np.min(X_batch)) / \
                  (np.max(X_batch) - np.min(X_batch))
    return X_batch


def wait_incrementation(wait, best_criteria, current_criteria):
    if current_criteria < best_criteria:
        wait = 0
    else:
        wait += 1
    return wait


def disable_training(model):
    for layer in model.layers:
        layer.trainable = False
    model.trainable = False
