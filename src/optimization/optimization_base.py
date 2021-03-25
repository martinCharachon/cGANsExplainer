import os
import json
from ..utils.data_generator import DataGeneratorFromH5File
from ..utils.classification_loss import ClassificationLossOperator, ClassificationMetricsOperator


class OptimizationBase:
    """
    Model operator
    """

    def __init__(self,
                 model_config: dict = {},
                 training_config: dict = {},
                 training_manager: DataGeneratorFromH5File = None,
                 validation_manager: DataGeneratorFromH5File = None,
                 ):
        self.model_config = model_config
        self.training_config = training_config
        self.training_manager = training_manager
        self.validation_manager = validation_manager
        self._logs_directory = self.training_config["outputs_directory"]
        if "weights" not in os.listdir(self.training_config["outputs_directory"]):
            os.mkdir(os.path.join(self.training_config["outputs_directory"], "weights"))
        self._weights_directory = os.path.join(self.training_config["outputs_directory"], "weights")
        self._set_tools()
        configuration = {
            "training": training_config,
            "model": model_config
        }
        with open(os.path.join(self.training_config["outputs_directory"],
                               "configuration.json"), "w") as f:
            json.dump(configuration, f, indent=2)

    def _set_data_generator(self):
        self.tr_steps = self.training_manager.num_index // self.training_config["batch_size"]
        self.val_steps = self.validation_manager.num_index // self.training_config["batch_size"]
        self.tr_gen = self.training_manager()
        self.val_gen = self.validation_manager()

    def _set_tools(self):
        self._set_data_generator()
        self._set_classification_loss()
        self._set_classification_metrics()

    def _fit_generator(self, initial_epoch=0):
        pass

    def build_model(self):
        pass

    def _close_training(self):
        self.training_manager.close()
        self.validation_manager.close()

    def train(self, initial_epoch=0):
        self.build_model()
        self._fit_generator(initial_epoch)
        self._close_training()

    def _set_classification_loss(self):
        self._classification_op = ClassificationLossOperator(
            **self.training_config["loss"]["classification"]["parameters"])
        self._classification_loss = self._classification_op.classification_loss

    def _set_classification_metrics(self):
        self._classification_metrics_op = \
                ClassificationMetricsOperator("binary_accuracy")
        self._classification_metrics = self._classification_metrics_op.classification_metrics

    def _write_model_to_json_and_summary(self, model, model_name=""):
        model_json = model.to_json()
        with open(os.path.join(self._logs_directory,
                               str(model_name) + ".json"), "w") as json_file:
            json_file.write(model_json)
        write_path = os.path.join(
            self._logs_directory,
            str(model_name) + "_summary.txt")
        with open(write_path, "w") as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

