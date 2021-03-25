import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import json

from ..utils.instantiate_model import instantiate_model
from .optimization_interpreter_base import OptimizationInterpreterBase, wait_incrementation
from ..vae.decoder import Decoder
from ..vae.vae_encoder import VAEEncoder
from ..vae.vae_inception_encoder import VAEInceptionEncoder
from ..utils.reconstruction_loss import LatentSeparationLoss
from ..utils.data_generator import DataGeneratorFromH5File


class OptimizationVAE(OptimizationInterpreterBase):

    def __init__(self,
                 reference_model: keras.models.Model = None,
                 model_config: dict = {},
                 training_config: dict = {},
                 training_manager: DataGeneratorFromH5File = None,
                 validation_manager: DataGeneratorFromH5File = None
                 ):
        super().__init__(
            reference_model=reference_model,
            model_config=model_config, training_config=training_config,
            training_manager=training_manager, validation_manager=validation_manager)

    def _set_tools(self):
        super()._set_tools()
        self._set_specific_kl_loss()

    def _set_specific_kl_loss(self):
        if "kl" in list(self.training_config["loss"].keys()):
            self._kl_loss_operator = LatentSeparationLoss(
                **self.training_config["loss"]["kl"]["parameters"])
        else:
            self._kl_loss_operator = LatentSeparationLoss(
                coefficient=0.0)
        self._kl_loss = self._kl_loss_operator.loss

    @staticmethod
    def re_parametrize(mean, log_var, mean_ref=0.0, var_ref=1.0):
        epsilon = tf.random.normal(
            shape=tf.shape(mean),
            mean=mean_ref,
            stddev=var_ref)
        return mean + tf.exp(0.5 * log_var) * epsilon

    def _build_encoded_classifier(self):
        enc_classifier = None
        if "enc_classifier" in list(self.model_config["model_path"].keys()):
            enc_classifier = instantiate_model(
                model_path=self.model_config["model_path"]["enc_classifier"])
        if enc_classifier is None:
            inputs = tf.keras.layers.Input(shape=self.encoded_shape)
            layer = tf.keras.layers.Dense(units=64, activation="relu")(inputs)
            outputs = tf.keras.layers.Dense(units=1, activation="sigmoid")(layer)
            enc_classifier = tf.keras.Model(inputs=inputs, outputs=outputs)

        self.enc_classifier = enc_classifier
        self.enc_classifier.stop_training = False
        self._write_model_to_json_and_summary(self.enc_classifier, "enc_classifier")

    def _build_encoder(self):
        input_shape = tuple(self.model_config["input_shape"])

        encoder = None
        if "encoder" in list(self.model_config["model_path"].keys()):
            encoder = instantiate_model(
                model_path=self.model_config["model_path"]["encoder"])
        if encoder is None:
            if "vae-encoder" in list(self.model_config["config"].keys()):
                encoder_op = VAEEncoder(
                    input_shape=input_shape,
                    **self.model_config["config"]["vae-encoder"]["parameters"])
                encoder = encoder_op.model
                self.encoded_shape = encoder_op.encoded_layer_shape
                # self.encoding_dimension = encoder_op._encoded_dimension
                self.target_size = encoder_op.target_size
            elif "vae-inception-encoder" in list(self.model_config["config"].keys()):
                encoder_op = VAEInceptionEncoder(
                    input_shape=input_shape,
                    **self.model_config["config"]["vae-inception-encoder"]["parameters"])
                encoder = encoder_op.model
                self.encoded_shape = encoder_op.encoded_layer_shape
                # self.encoding_dimension = encoder_op._output_size
                self.target_size = encoder_op.target_size
            else:
                raise AttributeError

        self.encoder = encoder
        self.encoder.stop_training = False
        self._write_model_to_json_and_summary(self.encoder, "encoder")

    def _build_decoder(self):
        input_shape = tuple(self.model_config["input_shape"])

        decoder = None
        if "decoder" in list(self.model_config["model_path"].keys()):
            decoder = instantiate_model(
                model_path=self.model_config["model_path"]["decoder"])
        if decoder is None:
            if "decoder" in list(self.model_config["config"].keys()):
                decoder_op = Decoder(
                    input_shape=input_shape,
                    target_size=self.target_size,
                    encoding_dimension=self.encoded_shape,
                    **self.model_config["config"]["decoder"]["parameters"])
                decoder = decoder_op.model
            else:
                raise AttributeError

        self.decoder = decoder
        self.decoder.stop_training = False
        self._write_model_to_json_and_summary(self.decoder, "decoder")

    def build_model(self):
        self._build_encoder()
        self._build_decoder()
        self._build_encoded_classifier()
        try:
            self._optimizer = getattr(tf.keras.optimizers,
                                      self.training_config["optimizer"]["generator"]["name"])(
                lr=self.training_config["optimizer"]["generator"]["learning_rate"])
        except:
            self._optimizer = getattr(tf.keras.optimizers,
                                      self.training_config["optimizer"]["name"])(
                lr=self.training_config["optimizer"]["learning_rate"])

    def _epoch_step(self):
        self._epoch_initialization()
        while self._tr_step < self.tr_steps:
            X_batch, y_batch = self._get_next_batch(self.tr_gen)
            y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)
            self._optimize([X_batch, y_batch])
            self._tr_step += 1

            if self._tr_step >= self.tr_steps:
                self._epoch_validation()
        self._epoch_closing()

    def _epoch_validation(self):
        while self._val_step < self.val_steps:
            X_batch, y_batch = self._get_next_batch(self.val_gen)
            y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)
            self._validate([X_batch, y_batch])
            self._val_step += 1

    def _optimize(self, inputs):
        with tf.GradientTape() as t:
            x_batch, y_batch = inputs
            y_batch = self._set_ground_truth(x_batch, y_batch)
            encoded_outs = self.encoder(x_batch)
            mean, log_var = encoded_outs
            z = self.re_parametrize(mean, log_var, 0.0, 1.0)
            x_rec = self.decoder(z)
            x_rec = self._image_activation(x_rec)
            z_cls = self.enc_classifier(mean)

            c_loss = self._classification_loss(y_batch, z_cls)
            r_loss = self._reconstruction_loss(x_batch, x_rec)
            kl_loss = self._kl_loss(mean, log_var)
            tot_loss = r_loss + kl_loss + c_loss
            gradient = t.gradient(
                tot_loss,
                self.encoder.trainable_variables + self.decoder.trainable_variables +
                self.enc_classifier.trainable_variables)

        self._optimizer.apply_gradients(zip(
            gradient,
            self.encoder.trainable_variables + self.decoder.trainable_variables +
            self.enc_classifier.trainable_variables))

        losses = np.array([tot_loss.numpy(),
                           r_loss.numpy(), kl_loss.numpy(), c_loss.numpy()])

        self._training_loss.append(losses)

    def _validate(self, inputs):
        x_batch, y_batch = inputs
        y_batch = self._set_ground_truth(x_batch, y_batch)
        encoded_outs = self.encoder(x_batch)
        mean, log_var = encoded_outs
        z = self.re_parametrize(mean, log_var, 0.0, 1.0)
        x_rec = self.decoder(z)
        x_rec = self._image_activation(x_rec)
        z_cls = self.enc_classifier(mean)

        c_loss = self._classification_loss(y_batch, z_cls)
        r_loss = self._reconstruction_loss(x_batch, x_rec)
        kl_loss = self._kl_loss(mean, log_var)
        tot_loss = r_loss + kl_loss + c_loss

        losses = np.array([tot_loss.numpy(),
                           r_loss.numpy(), kl_loss.numpy(), c_loss.numpy()])
        self._validation_loss.append(losses)
        cls = self._classification_metrics(y_batch, z_cls)
        metrics = np.array([np.mean(cls.numpy())])
        self._validation_classification_metrics.append(metrics)
        self._validation_reconstruction_metrics.append(0)

    def _save_model_weights(self, directory):
        encoder_path = os.path.join(
            directory, f"encoder_{self._epoch + 1}.hdf5")
        self.encoder.save(encoder_path, overwrite=True)
        decoder_path = os.path.join(
            directory, f"decoder_{self._epoch + 1}.hdf5")
        self.decoder.save(decoder_path, overwrite=True)
        enc_cls_path = os.path.join(
            directory, f"encoded_classifier_{self._epoch + 1}.hdf5")
        self.enc_classifier.save(enc_cls_path, overwrite=True)

    def _custom_checkpoint(self):
        if (self._epoch + 1) % 3 == 0:
            directory = self._weights_directory
            model_0_path = os.path.join(
                directory, f"encoder_{self._epoch + 1}_cst_chkpt.hdf5")
            self.encoder.save(model_0_path, overwrite=True)
            model_1_path = os.path.join(
                directory, f"decoder_{self._epoch + 1}_cst_chkpt.hdf5")
            self.decoder.save(model_1_path, overwrite=True)
            enc_cls_path = os.path.join(
                directory, f"encoded_classifier_{self._epoch + 1}_cst_chkpt.hdf5")
            self.enc_classifier.save(enc_cls_path, overwrite=True)

    def _set_callbacks_on_epoch_end(self):
        super()._set_callbacks_on_epoch_end()
        self._custom_checkpoint()

    def _set_model_checkpoint(self, curent_criteria):
        if self._best_criteria is None:
            self._best_criteria = curent_criteria
            self._save_model_weights(self._weights_directory)
        else:
            if curent_criteria < self._best_criteria:
                self._best_criteria = curent_criteria
                self._save_model_weights(self._weights_directory)

    def _check_reduce_lr_on_plateau(self, current_criteria):
        self._wait_plateau = wait_incrementation(
            self._wait_plateau, self._best_criteria, current_criteria)
        if self._wait_plateau >= self._patience_plateau:
            old_lr_gen = float(keras.backend.get_value(self._optimizer.lr))
            if old_lr_gen > self.min_lr:
                new_lr_gen = old_lr_gen * self.factor
                new_lr_gen = max(new_lr_gen, self.min_lr)
                keras.backend.set_value(self._optimizer.lr, new_lr_gen)
                print(f"Reduce learning rate to {new_lr_gen} for generator")
                self._wait_plateau = 0
            else:
                self.encoder.stop_training = True
                self.decoder.stop_training = True

    def _stop_training(self):
        stop = False
        if self.encoder.stop_training:
            stop = True
        return stop

    def _write_logs_to_json(self):
        log_filepath = os.path.join(self._logs_directory, self._logs_file)
        if self._epoch > 0:
            with open(log_filepath, "r") as f:
                out_dict = json.load(f)
        else:
            out_dict = {"validation_logs": {}, "training_logs": {}, "learning rate": {}}

        out_dict["validation_logs"][f"epoch_{self._epoch + 1}"] = {
            "generator loss": np.mean(self._validation_loss, axis=0).astype(float).tolist(),
            "classification metrics":
                np.mean(self._validation_classification_metrics, axis=0).astype(float).tolist(),
            "reconstruction metrics":
                np.mean(self._validation_reconstruction_metrics, axis=0).astype(float).tolist(),
        }
        out_dict["training_logs"][f"epoch_{self._epoch + 1}"] = {
            "generator loss": np.mean(self._training_loss, axis=0).astype(float).tolist(),
        }
        out_dict["learning rate"][f"epoch_{self._epoch + 1}"] = {
            "generator value": float(keras.backend.get_value(self._optimizer.lr))
        }
        with open(log_filepath, "w") as f:
            json.dump(out_dict, f, indent=2)

