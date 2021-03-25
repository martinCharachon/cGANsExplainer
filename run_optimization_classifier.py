import json
import configargparse
from src.utils.data_generator import set_data_generator
from src.optimization.optimization_classifier import OptimizationClassifier


def main():
    p = configargparse.ArgParser()
    p.add_argument('--config-file', required=False, default="./mnist/config_classifier.json")
    args = p.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)
    training_config = config["training_configuration"]
    model_config = config["model_configuration"]
    db_path = config["db_path"]
    with open(config["split_path"], "r") as f:
        split = json.load(f)
    train = split["train_indexes"]
    val = split["val_indexes"]

    training_manager = set_data_generator(
        input_shape=model_config["input_shape"], db_path=db_path, reference_list=train,
        batch_size=training_config["batch_size"],
        preprocessor_config=training_config["geometric_preprocessor_training"],
        seed=training_config["generator_seed"]
    )
    validation_manager = set_data_generator(
        input_shape=model_config["input_shape"], db_path=db_path, reference_list=val,
        batch_size=training_config["batch_size"],
        preprocessor_config=training_config["geometric_preprocessor_validation"],
        seed=training_config["generator_seed"]
    )

    operator = OptimizationClassifier(
        training_config=training_config,
        model_config=model_config,
        training_manager=training_manager,
        validation_manager=validation_manager,
    )

    operator.train()


if __name__ == "__main__":
    main()
