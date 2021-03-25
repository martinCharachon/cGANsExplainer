import os
import tensorflow as tf
from ..utils.data_generator import DataGeneratorFromH5File
from .optimization_base import OptimizationBase
from ..classifiers.lenet import LeNet
from ..classifiers.resnet import ResNet
from ..classifiers.densenet import DenseNet


class OptimizationClassifier(OptimizationBase):
    """
    Model operator
    """

    def __init__(self,
                 model_config: dict = {},
                 training_config: dict = {},
                 training_manager: DataGeneratorFromH5File = None,
                 validation_manager: DataGeneratorFromH5File = None,
                 ):
        super().__init__(model_config, training_config, training_manager, validation_manager)

    def _set_tools(self):
        super()._set_tools()
        self._set_call_backs()

    def _build_model(self):
        input_shape = tuple(self.model_config["input_shape"])
        if self.model_config["name"] == "densenet":
            self.model = DenseNet(
                input_shape=input_shape, **self.model_config["config"]).model
        elif self.model_config["name"] == "resnet":
            self.model = ResNet(
                input_shape=input_shape, **self.model_config["config"]).model
        elif self.model_config["name"] == "lenet":
            self.model = LeNet(
                input_shape=input_shape, **self.model_config["config"]).model
        else:
            NotImplementedError

    def build_model(self):
        self._build_model()
        self._optimizer = getattr(tf.keras.optimizers,
                                  self.training_config["optimizer"]["name"])(
            lr=self.training_config["optimizer"]["learning_rate"])
        self.model.compile(
            optimizer=self._optimizer,
            loss=self._classification_loss,
            metrics=[self._classification_metrics])

    def _fit_generator(self, initial_epoch=0):
        self.model.fit(
            self.tr_gen,
            steps_per_epoch=self.tr_steps,
            validation_data=self.val_gen,
            validation_steps=self.val_steps,
            initial_epoch=initial_epoch,
            epochs=self.training_config["epochs"],
            callbacks=list(self.call_backs.values()),
            verbose=self.training_config["verbose"])

    def _set_call_backs(self):
        self.call_backs = {}
        call_backs_dict = self.training_config["call_backs"]
        for fct, params in call_backs_dict.items():
            if fct not in dir(tf.keras.callbacks):
                Warning("Did not find {} in keras.callbacks.".
                        format(fct))
                continue
            if fct == "ModelCheckpoint":
                params["filepath"] = os.path.join(
                    self._weights_directory, 'weights_{epoch:02d}.hdf5' )

            if fct == "CSVLogger":
                params["filename"] = os.path.join(self._logs_directory, "log.csv")
            self.call_backs[fct] = getattr(tf.keras.callbacks, fct)(**params)



