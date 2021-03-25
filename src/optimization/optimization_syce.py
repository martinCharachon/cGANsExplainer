import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import os

from ..interpreters.discriminator import Discriminator
from ..interpreters.unet_generator import UNetGenerator
from ..utils.instantiate_model import instantiate_model
from .optimization_interpreter_base import OptimizationInterpreterBase, wait_incrementation, \
    disable_training
from ..utils.adversarial_loss import AdversarialLoss
from ..utils.classification_loss import ClassificationLossOperator
from ..utils.reconstruction_loss import ReconstructionLoss
from ..utils.data_generator import DataGeneratorFromH5File


class OptimizationSyCE(OptimizationInterpreterBase):
    """
    Model operator
    """

    def __init__(self,
                 reference_model: keras.models.Model = None,
                 model_config: dict = {},
                 training_config: dict = {},
                 training_manager_0: DataGeneratorFromH5File = None,
                 validation_manager_0: DataGeneratorFromH5File = None,
                 training_manager_1: DataGeneratorFromH5File = None,
                 validation_manager_1: DataGeneratorFromH5File = None
                 ):
        self.training_manager_1 = training_manager_1
        self.validation_manager_1 = validation_manager_1
        super().__init__(
            reference_model=reference_model,
            model_config=model_config,
            training_config=training_config,
            training_manager=training_manager_0,
            validation_manager=validation_manager_0
        )
        if "discriminator_steps" not in list(self.training_config.keys()):
            self.training_config["discriminator_steps"] = 1
        self._discriminator_steps = self.training_config["discriminator_steps"]

    def _set_data_generator(self):
        self.tr_steps = np.minimum(
            self.training_manager.num_index, self.training_manager_1.num_index) // \
                        self.training_config["batch_size"]
        self.val_steps = np.minimum(
            self.validation_manager.num_index, self.validation_manager_1.num_index) // \
                         self.training_config["batch_size"]
        self.tr_gen_0 = self.training_manager()
        self.val_gen_0 = self.validation_manager()
        self.tr_gen_1 = self.training_manager_1()
        self.val_gen_1 = self.validation_manager_1()

    def _set_tools(self):
        super()._set_tools()
        self._discriminator_training_loss = []
        self._discriminator_validation_loss = []
        self._discriminator_validation_metrics = []
        self._set_adversarial_losses()
        self._set_classification_cyclic_loss()

    def _optimize_generators(self, inputs):
        loss_0 = self._optimize_generator_0(inputs)
        loss_1 = self._optimize_generator_1(inputs)
        self._training_loss.append(np.concatenate((loss_0, loss_1), axis=0))

    def _optimize_discriminators(self, inputs):
        loss_0 = self._optimize_discriminator_0(inputs)
        loss_1 = self._optimize_discriminator_1(inputs)
        self._discriminator_training_loss.append(np.concatenate((loss_0, loss_1), axis=0))

    def _optimize_generator_0(self, inputs):
        with tf.GradientTape() as t:
            disable_training(self.reference_model)
            x_0_batch, y_0_gt_batch, x_1_batch, y_1_gt_batch = inputs
            y_0_batch = self._set_ground_truth(x_0_batch, y_0_gt_batch)
            _, x_adv_01 = self.generator_0(x_0_batch)
            x_adv_01 = self._image_activation(x_adv_01)
            _, x_cy_0110 = self.generator_1(x_adv_01)
            x_cy_0110 = self._image_activation(x_cy_0110)
            _, x_sy_0110 = self.generator_0(x_adv_01)
            x_sy_0110 = self._image_activation(x_sy_0110)
            classifier_adv_01_score = self.reference_model(x_adv_01)
            classifier_cy_0110_score = self.reference_model(x_cy_0110)
            classifier_sy_0110_score = self.reference_model(x_sy_0110)
            discriminator_01_score = self.discriminator_1(x_adv_01)

            r_sy_0110_loss = self._reconstruction_loss_sy(x_0_batch, x_sy_0110)
            r_cy_0110_loss = self._reconstruction_loss_cy(x_0_batch, x_cy_0110)
            c_adv_01_loss = self._classification_loss(y_0_batch, classifier_adv_01_score)
            c_cy_0110_loss = self._classification_cyc_loss(y_0_batch,
                                                           classifier_cy_0110_score)
            c_sy_0110_loss = self._classification_sym_loss(y_0_batch, classifier_sy_0110_score)
            disc_1_loss = self._adversarial_generator_loss(discriminator_01_score)
            tot_loss = r_sy_0110_loss + c_adv_01_loss + c_sy_0110_loss + disc_1_loss + \
                       r_cy_0110_loss + c_cy_0110_loss
            gradient = t.gradient(
                tot_loss,
                self.generator_0.trainable_variables + self.generator_1.trainable_variables)

        self._optimizer.apply_gradients(zip(
            gradient, self.generator_0.trainable_variables + self.generator_1.trainable_variables))

        losses = np.array([tot_loss.numpy(), r_sy_0110_loss.numpy(), c_adv_01_loss.numpy(),
                           c_sy_0110_loss.numpy(), disc_1_loss.numpy(),
                           r_cy_0110_loss.numpy(), c_cy_0110_loss.numpy()])

        return losses

    def _optimize_generator_1(self, inputs):
        with tf.GradientTape() as t:
            disable_training(self.reference_model)
            x_0_batch, y_0_gt_batch, x_1_batch, y_1_gt_batch = inputs
            y_1_batch = self._set_ground_truth(x_1_batch, y_1_gt_batch)
            _, x_adv_10 = self.generator_1(x_1_batch)
            x_adv_10 = self._image_activation(x_adv_10)
            _, x_cy_1001 = self.generator_0(x_adv_10)
            x_cy_1001 = self._image_activation(x_cy_1001)
            _, x_sy_1001 = self.generator_1(x_adv_10)
            x_sy_1001 = self._image_activation(x_sy_1001)
            classifier_adv_10_score = self.reference_model(x_adv_10)
            classifier_cy_1001_score = self.reference_model(x_cy_1001)
            classifier_sy_1001_score = self.reference_model(x_sy_1001)
            discriminator_10_score = self.discriminator_0(x_adv_10)

            r_sy_1001_loss = self._reconstruction_loss_sy(x_1_batch, x_sy_1001)
            r_cy_1001_loss = self._reconstruction_loss_cy(x_1_batch, x_cy_1001)
            c_adv_10_loss = self._classification_loss(y_1_batch, classifier_adv_10_score)
            c_cy_1001_loss = self._classification_cyc_loss(y_1_batch,
                                                           classifier_cy_1001_score)
            c_sy_1001_loss = self._classification_sym_loss(y_1_batch, classifier_sy_1001_score)
            disc_0_loss = self._adversarial_generator_loss(discriminator_10_score)
            tot_loss = r_sy_1001_loss + c_adv_10_loss + c_sy_1001_loss + disc_0_loss + \
                       r_cy_1001_loss + c_cy_1001_loss
            gradient = t.gradient(
                tot_loss,
                self.generator_0.trainable_variables + self.generator_1.trainable_variables)

        self._optimizer.apply_gradients(zip(
            gradient, self.generator_0.trainable_variables + self.generator_1.trainable_variables))

        losses = np.array([tot_loss.numpy(), r_sy_1001_loss.numpy(), c_adv_10_loss.numpy(),
                           c_sy_1001_loss.numpy(), disc_0_loss.numpy(),
                           r_cy_1001_loss.numpy(), c_cy_1001_loss.numpy()])

        return losses

    def _optimize_discriminator_0(self, inputs):
        with tf.GradientTape() as t:
            x_0_batch, y_0_batch, x_1_batch, y_1_batch = inputs
            _, x_10 = self.generator_1(x_1_batch)
            x_10 = self._image_activation(x_10)
            real_discriminator_score_0 = self.discriminator_0(x_0_batch)
            fake_discriminator_score_0 = self.discriminator_0(x_10)
            adv_disc_loss_0 = self._adversarial_discriminator_loss(
                real_discriminator_score_0, fake_discriminator_score_0)
            adv_gradient_penalty_0 = self._adversarial_discriminator_gp_loss(
                discriminator=self.discriminator_0,
                real_input=x_0_batch,
                fake_input=x_10)
            tot_loss = adv_disc_loss_0 + adv_gradient_penalty_0
            gradient = t.gradient(tot_loss, self.discriminator_0.trainable_variables)
        self._disc_optimizer.apply_gradients(zip(
            gradient, self.discriminator_0.trainable_variables))
        losses = np.array([tot_loss.numpy(),
                           adv_disc_loss_0.numpy(), adv_gradient_penalty_0.numpy()])
        return losses

    def _optimize_discriminator_1(self, inputs):
        with tf.GradientTape() as t:
            x_0_batch, y_0_batch, x_1_batch, y_1_batch = inputs
            _, x_01 = self.generator_0(x_0_batch)
            x_01 = self._image_activation(x_01)
            real_discriminator_score_1 = self.discriminator_1(x_1_batch)
            fake_discriminator_score_1 = self.discriminator_1(x_01)
            adv_disc_loss_1 = self._adversarial_discriminator_loss(
                real_discriminator_score_1, fake_discriminator_score_1)
            adv_gradient_penalty_1 = self._adversarial_discriminator_gp_loss(
                discriminator=self.discriminator_1,
                real_input=x_1_batch,
                fake_input=x_01)
            tot_loss = adv_disc_loss_1 + adv_gradient_penalty_1
            gradient = t.gradient(tot_loss, self.discriminator_1.trainable_variables)
        self._disc_optimizer.apply_gradients(zip(
            gradient, self.discriminator_1.trainable_variables))
        losses = np.array([tot_loss.numpy(),
                           adv_disc_loss_1.numpy(), adv_gradient_penalty_1.numpy()])
        return losses

    def _validate(self, inputs):
        x_0_batch, y_0_gt_batch, x_1_batch, y_1_gt_batch = inputs
        y_0_batch = self._set_ground_truth(x_0_batch, y_0_gt_batch)
        y_1_batch = self._set_ground_truth(x_1_batch, y_1_gt_batch)
        _, x_adv_01 = self.generator_0(x_0_batch)
        _, x_adv_10 = self.generator_1(x_1_batch)
        x_adv_01 = self._image_activation(x_adv_01)
        x_adv_10 = self._image_activation(x_adv_10)
        _, x_cy_0110 = self.generator_1(x_adv_01)
        _, x_cy_1001 = self.generator_0(x_adv_10)
        x_cy_0110 = self._image_activation(x_cy_0110)
        x_cy_1001 = self._image_activation(x_cy_1001)
        _, x_sy_0110 = self.generator_0(x_adv_01)
        _, x_sy_1001 = self.generator_1(x_adv_10)
        x_sy_0110 = self._image_activation(x_sy_0110)
        x_sy_1001 = self._image_activation(x_sy_1001)

        classifier_adv_01_score = self.reference_model(x_adv_01)
        classifier_adv_10_score = self.reference_model(x_adv_10)
        classifier_cy_0110_score = self.reference_model(x_cy_0110)
        classifier_cy_1001_score = self.reference_model(x_cy_1001)
        classifier_sy_0110_score = self.reference_model(x_sy_0110)
        classifier_sy_1001_score = self.reference_model(x_sy_1001)
        discriminator_01_score = self.discriminator_1(x_adv_01)
        discriminator_10_score = self.discriminator_0(x_adv_10)
        discriminator_1_score = self.discriminator_1(x_1_batch)
        discriminator_0_score = self.discriminator_0(x_0_batch)

        r_sy_0110_loss = self._reconstruction_loss_sy(x_0_batch, x_sy_0110)
        r_sy_1001_loss = self._reconstruction_loss_sy(x_1_batch, x_sy_1001)
        r_cy_0110_loss = self._reconstruction_loss_cy(x_0_batch, x_cy_0110)
        r_cy_1001_loss = self._reconstruction_loss_cy(x_1_batch, x_cy_1001)
        c_adv_01_loss = self._classification_loss(y_0_batch, classifier_adv_01_score)
        c_adv_10_loss = self._classification_loss(y_1_batch, classifier_adv_10_score)
        c_sy_0110_loss = self._classification_sym_loss(y_0_batch, classifier_sy_0110_score)
        c_sy_1001_loss = self._classification_sym_loss(y_1_batch, classifier_sy_1001_score)
        c_cy_0110_loss = self._classification_cyc_loss(y_0_batch, classifier_cy_0110_score)
        c_cy_1001_loss = self._classification_cyc_loss(y_1_batch, classifier_cy_1001_score)
        disc_gen_0_loss = self._adversarial_generator_loss(discriminator_10_score)
        disc_gen_1_loss = self._adversarial_generator_loss(discriminator_01_score)
        disc_disc_1_loss = self._adversarial_discriminator_loss(
            discriminator_1_score, discriminator_01_score)
        disc_disc_0_loss = self._adversarial_discriminator_loss(
            discriminator_0_score, discriminator_10_score)
        tot_loss_0 = r_sy_0110_loss + c_adv_01_loss + c_sy_0110_loss + disc_gen_1_loss + \
                     r_cy_0110_loss + c_cy_0110_loss
        tot_loss_1 = r_sy_1001_loss + c_adv_10_loss + c_sy_1001_loss + disc_gen_0_loss + \
                     r_cy_1001_loss + c_cy_1001_loss
        tot_loss = tot_loss_0 + tot_loss_1

        losses = np.array([tot_loss.numpy(), tot_loss_0.numpy(), tot_loss_1.numpy(),
                           r_sy_0110_loss.numpy(), r_sy_1001_loss.numpy(),
                           r_cy_0110_loss.numpy(), r_cy_1001_loss.numpy(),
                           c_adv_01_loss.numpy(), c_adv_10_loss.numpy(),
                           c_sy_0110_loss.numpy(), c_sy_1001_loss.numpy(),
                           c_cy_0110_loss.numpy(), c_cy_1001_loss.numpy(),
                           disc_gen_0_loss.numpy(), disc_gen_1_loss.numpy()])

        c_adv_10_metrics = self._classification_metrics(y_1_batch, classifier_adv_10_score)
        c_adv_01_metrics = self._classification_metrics(y_0_batch, classifier_adv_01_score)
        c_sy_1001_metrics = self._classification_metrics(y_1_batch, classifier_sy_1001_score)
        c_sy_0110_metrics = self._classification_metrics(y_0_batch, classifier_sy_0110_score)
        c_cy_1001_metrics = self._classification_metrics(y_1_batch, classifier_cy_1001_score)
        c_cy_0110_metrics = self._classification_metrics(y_0_batch, classifier_cy_0110_score)
        metrics = \
            np.array([np.mean(c_adv_10_metrics.numpy()), np.mean(c_adv_01_metrics.numpy()),
                      np.mean(c_sy_1001_metrics.numpy()), np.mean(c_sy_0110_metrics.numpy()),
                      np.mean(c_cy_1001_metrics.numpy()), np.mean(c_cy_0110_metrics.numpy())])
        r_metrics_sy_0110 = np.mean(self._reconstruction_metrics(x_0_batch, x_sy_0110)[0])
        r_metrics_sy_1001 = np.mean(self._reconstruction_metrics(x_1_batch, x_sy_1001)[0])
        r_metrics_adv_0_01 = np.mean(self._reconstruction_metrics(x_0_batch, x_adv_01)[0])
        r_metrics_adv_1_10 = np.mean(self._reconstruction_metrics(x_1_batch, x_adv_10)[0])
        r_metrics_cy_0110 = np.mean(self._reconstruction_metrics(x_0_batch, x_cy_0110)[0])
        r_metrics_cy_1001 = np.mean(self._reconstruction_metrics(x_1_batch, x_cy_1001)[0])
        reconstruction_metrics = np.array(
            [r_metrics_sy_0110, r_metrics_sy_1001,
             r_metrics_adv_0_01, r_metrics_adv_1_10,
             r_metrics_cy_0110, r_metrics_cy_1001])
        disc_losses = np.array([disc_disc_0_loss.numpy(), disc_disc_1_loss.numpy()])
        c_disc_0_metrics = self._adversarial_discriminator_metrics(
            discriminator_1_score, discriminator_01_score)
        c_disc_1_metrics = self._adversarial_discriminator_metrics(
            discriminator_0_score, discriminator_10_score)
        disc_metrics = np.array(
            [np.mean(c_disc_0_metrics, 0), np.mean(c_disc_1_metrics, 0)])

        self._validation_loss.append(losses)
        self._validation_classification_metrics.append(metrics)

        self._validation_reconstruction_metrics.append(reconstruction_metrics)
        self._discriminator_validation_loss.append(disc_losses)
        self._discriminator_validation_metrics.append(disc_metrics)

    def _epoch_step(self):
        self._epoch_initialization()
        while self._tr_step < self.tr_steps:
            if self._tr_step % 10 == 0:
                print(f"Step: {self._tr_step} / {self.tr_steps}")
            x_0_batch, y_0_gt_batch = self._get_next_batch(self.tr_gen_0)
            x_1_batch, y_1_gt_batch = self._get_next_batch(self.tr_gen_1)
            y_0_gt_batch = tf.convert_to_tensor(y_0_gt_batch, dtype=tf.float32)
            y_1_gt_batch = tf.convert_to_tensor(y_1_gt_batch, dtype=tf.float32)
            self._optimize_generators([x_0_batch, y_0_gt_batch, x_1_batch, y_1_gt_batch])
            for i in range(self._discriminator_steps):
                self._optimize_discriminators([x_0_batch, y_0_gt_batch, x_1_batch, y_1_gt_batch])
            self._tr_step += 1

            if self._tr_step >= self.tr_steps:
                self._epoch_validation()
        self._epoch_closing()

    def _epoch_initialization(self):
        super()._epoch_initialization()
        self._discriminator_training_loss = []
        self._discriminator_validation_loss = []
        self._discriminator_validation_metrics = []

    def _epoch_validation(self):
        while self._val_step < self.val_steps:
            x_0_batch, y_0_gt_batch = self._get_next_batch(self.val_gen_0)
            x_1_batch, y_1_gt_batch = self._get_next_batch(self.val_gen_1)
            y_0_gt_batch = tf.convert_to_tensor(y_0_gt_batch, dtype=tf.float32)
            y_1_gt_batch = tf.convert_to_tensor(y_1_gt_batch, dtype=tf.float32)
            self._validate([x_0_batch, y_0_gt_batch, x_1_batch, y_1_gt_batch])
            self._val_step += 1

    def _set_adversarial_losses(self):
        self._adversarial_operator = AdversarialLoss(
            **self.training_config["loss"]["adversarial"]["parameters"])
        self._adversarial_generator_loss = self._adversarial_operator.generator_loss
        self._adversarial_discriminator_loss = self._adversarial_operator.discriminator_loss
        self._adversarial_discriminator_gp_loss = \
            self._adversarial_operator.gradient_penalty_loss
        self._adversarial_discriminator_metrics = \
            self._adversarial_operator.discriminator_metrics

    def _set_reconstruction_loss(self):
        self._representation_loss_operator_sy = ReconstructionLoss(
            **self.training_config["loss"]["reconstruction"][
                "parameters_sym"])
        self._reconstruction_loss_sy = \
            self._representation_loss_operator_sy.reconstruction_loss
        self._representation_loss_operator_cy = ReconstructionLoss(
            **self.training_config["loss"]["reconstruction"][
                "parameters_cyc"])
        self._reconstruction_loss_cy = \
            self._representation_loss_operator_cy.reconstruction_loss

    def _set_classification_cyclic_loss(self):
        self._classification_sym_loss = ClassificationLossOperator(
            **self.training_config["loss"]["classification_sym"][
                "parameters"]).classification_loss
        if "classification_cyc" in list(self.training_config["loss"].keys()):
            self._classification_cyc_loss = ClassificationLossOperator(
                **self.training_config["loss"]["classification_cyc"]["parameters"]). \
                classification_loss
        else:
            self._classification_cyc_loss = ClassificationLossOperator(
                coefficient=0).classification_loss

    def _build_generators(self):
        input_shape = tuple(self.model_config["input_shape"])

        generator_0 = None
        if "generator_0" in list(self.model_config["model_path"].keys()):
            generator_0 = instantiate_model(
                model_path=self.model_config["model_path"]["generator_0"])
        if generator_0 is None:
            if "unet" in list(self.model_config["config"].keys()):
                generator_0 = UNetGenerator(
                    input_shape=input_shape,
                    **self.model_config["config"]["unet"]["parameters"]).model
            else:
                raise AttributeError

        self.generator_0 = generator_0
        self.generator_0.stop_training = False
        self._write_model_to_json_and_summary(self.generator_0, "generator_0")

        generator_1 = None
        if "generator_1" in list(self.model_config["model_path"].keys()):
            generator_1 = instantiate_model(
                model_path=self.model_config["model_path"]["generator_1"])
        if generator_1 is None:
            if "unet" in list(self.model_config["config"].keys()):
                generator_1 = UNetGenerator(
                    input_shape=input_shape,
                    **self.model_config["config"]["unet"]["parameters"]).model
            else:
                raise AttributeError

        self.generator_1 = generator_1
        self.generator_1.stop_training = False
        self._write_model_to_json_and_summary(self.generator_1, "generator_1")

    def _build_discriminators(self):
        input_shape = tuple(self.model_config["input_shape"])
        discriminator_0 = None
        discriminator_1 = None
        if "discriminator_0" in list(self.model_config["model_path"].keys()):
            discriminator_0 = instantiate_model(
                model_path=self.model_config["model_path"]["discriminator_0"])
        if discriminator_0 is None:
            discriminator_0 = \
                Discriminator(input_shape=input_shape,
                      **self.model_config["config"][
                          "discriminator"]["parameters"]).model
        self.discriminator_0 = discriminator_0
        self.discriminator_0.stop_training = False
        self._write_model_to_json_and_summary(self.discriminator_0,
                                              "discriminator_0")

        if "discriminator_1" in list(self.model_config["model_path"].keys()):
            discriminator_1 = instantiate_model(
                model_path=self.model_config["model_path"]["discriminator_1"])
        if discriminator_1 is None:
            discriminator_1 = \
                Discriminator(input_shape=input_shape,
                      **self.model_config["config"][
                          "discriminator"]["parameters"]).model
        self.discriminator_1 = discriminator_1
        self.discriminator_1.stop_training = False
        self._write_model_to_json_and_summary(self.discriminator_1, "discriminator_1")

    def build_model(self):
        self._build_generators()
        self._build_discriminators()
        self._optimizer = getattr(tf.keras.optimizers,
                                  self.training_config["optimizer"]["generator"]["name"])(
            lr=self.training_config["optimizer"]["generator"]["learning_rate"])
        self._disc_optimizer = getattr(tf.keras.optimizers,
                                       self.training_config["optimizer"][
                                           "discriminator"][
                                           "name"])(
            lr=self.training_config["optimizer"]["discriminator"]["learning_rate"])

    def _stop_training(self):
        stop = False
        if self.generator_0.stop_training:
            stop = True
        return stop

    def _close_training(self):
        self.training_manager.close()
        self.validation_manager.close()
        self.training_manager_1.close()
        self.validation_manager_1.close()

    def _set_model_checkpoint(self, curent_criteria):
        if self._best_criteria is None:
            self._best_criteria = curent_criteria
            self._save_model_0_weights(self._weights_directory)
            self._save_model_1_weights(self._weights_directory)
        else:
            if curent_criteria < self._best_criteria:
                self._best_criteria = curent_criteria
                self._save_model_0_weights(self._weights_directory)
                self._save_model_1_weights(self._weights_directory)

    def _save_model_0_weights(self, directory):
        generator_0_path = os.path.join(
            directory, f"generator_0_{self._epoch + 1}.hdf5")
        self.generator_0.save(generator_0_path, overwrite=True)
        discriminator_1_path = os.path.join(
            directory, f"discriminator_1_{self._epoch + 1}.hdf5")
        self.discriminator_1.save(discriminator_1_path, overwrite=True)

    def _save_model_1_weights(self, directory):
        generator_1_path = os.path.join(
            directory, f"generator_1_{self._epoch + 1}.hdf5")
        self.generator_1.save(generator_1_path, overwrite=True)
        discriminator_0_path = os.path.join(
            directory, f"discriminator_0_{self._epoch + 1}.hdf5")
        self.discriminator_0.save(discriminator_0_path, overwrite=True)

    def _check_reduce_lr_on_plateau(self, current_criteria):
        self._wait_plateau = wait_incrementation(
            self._wait_plateau, self._best_criteria, current_criteria)
        if self._wait_plateau >= self._patience_plateau:
            old_lr_gen = float(keras.backend.get_value(self._optimizer.lr))
            old_lr_disc = float(keras.backend.get_value(self._disc_optimizer.lr))
            if old_lr_gen > self.min_lr:
                new_lr_gen = old_lr_gen * self.factor
                new_lr_gen = max(new_lr_gen, self.min_lr)
                keras.backend.set_value(self._optimizer.lr, new_lr_gen)
                print(f"Reduce learning rate to {new_lr_gen} for generators")
                new_lr_disc = old_lr_disc * self.factor
                new_lr_disc = max(new_lr_disc, self.min_lr)
                keras.backend.set_value(self._disc_optimizer.lr, new_lr_disc)
                print(f"Reduce learning rate to {new_lr_disc} for discriminators")
                self._wait_plateau = 0
            else:
                self.generator_0.stop_training = True
                self.generator_1.stop_training = True

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
            "discriminator loss": np.mean(self._discriminator_validation_loss, axis=0).astype(
                float).tolist(),
            "discriminator metrics": np.mean(self._discriminator_validation_metrics, axis=0).astype(
                float).tolist()
        }
        out_dict["training_logs"][f"epoch_{self._epoch + 1}"] = {
            "generator loss": np.mean(self._training_loss, axis=0).astype(float).tolist(),
            "discriminator loss": np.mean(self._discriminator_training_loss, axis=0).astype(
                float).tolist()
        }
        out_dict["learning rate"][f"epoch_{self._epoch + 1}"] = {
            "generator value": float(keras.backend.get_value(self._optimizer.lr)),
            "discriminator value": float(keras.backend.get_value(self._disc_optimizer.lr))
        }
        with open(log_filepath, "w") as f:
            json.dump(out_dict, f, indent=2)
