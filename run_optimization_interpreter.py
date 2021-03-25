import json
import configargparse
from src.utils.instantiate_model import instantiate_model
from src.utils.data_generator import set_data_generator
from src.utils.set_split_prediction_specific import set_prediction_specific_split, \
    write_specific_split_to_json_file
from src.optimization.optimization_syce import OptimizationSyCE
from src.optimization.optimization_cyce import OptimizationCyCE


def main():
    p = configargparse.ArgParser()
    p.add_argument('--config-file', required=True)
    p.add_argument('--ref-model-settings-json-path', required=True)
    p.add_argument('--split-specific-path', required=False,
                   default="./mnist/split/split_prediction_specific.json")
    args = p.parse_args()

    json_file = open(args.ref_model_settings_json_path, 'r', encoding="utf8", errors='ignore')
    settings = json.load(json_file)
    json_file.close()
    reference_model = instantiate_model(
        model_json_path=settings["model_json_path"],
        weights_path=settings["weights_h5_path"],
        model_path=settings["model_h5_path"]
    )

    with open(args.config_file, "r") as f:
        config = json.load(f)
    if "version" in list(config.keys()):
        version = config["version"]
    else:
        version = 1
    training_config = config["training_configuration"]
    model_config = config["model_configuration"]
    db_path = config["db_path"]
    with open(config["split_path"], "r") as f:
        split = json.load(f)
    if "train_indexes_0" and "train_indexes_1" in list(split.keys()):
        train_0, train_1 = split["train_indexes_0"], split["train_indexes_1"]
        val_0, val_1 = split["val_indexes_0"], split["val_indexes_1"]

    else:
        train_0, train_1, val_0, val_1 = set_prediction_specific_split(
            classifier=reference_model, db_path=db_path, split=split)
        write_specific_split_to_json_file(
            args.split_specific_path, train_0, train_1, val_0, val_1)

    training_manager_0 = set_data_generator(
        input_shape=model_config["input_shape"], db_path=db_path, reference_list=train_0,
        batch_size=training_config["batch_size"],
        preprocessor_config=training_config["geometric_preprocessor_training"],
        seed=training_config["generator_seed"]
    )
    training_manager_1 = set_data_generator(
        input_shape=model_config["input_shape"], db_path=db_path, reference_list=train_1,
        batch_size=training_config["batch_size"],
        preprocessor_config=training_config["geometric_preprocessor_training"],
        seed=training_config["generator_seed"]
    )
    validation_manager_0 = set_data_generator(
        input_shape=model_config["input_shape"], db_path=db_path, reference_list=val_0,
        batch_size=training_config["batch_size"],
        preprocessor_config=training_config["geometric_preprocessor_validation"],
        seed=training_config["generator_seed"]
    )
    validation_manager_1 = set_data_generator(
        input_shape=model_config["input_shape"], db_path=db_path, reference_list=val_1,
        batch_size=training_config["batch_size"],
        preprocessor_config=training_config["geometric_preprocessor_validation"],
        seed=training_config["generator_seed"]
    )

    if version == 1:
        explainer = OptimizationSyCE(
            reference_model=reference_model,
            training_config=training_config,
            model_config=model_config,
            training_manager_0=training_manager_0,
            training_manager_1=training_manager_1,
            validation_manager_0=validation_manager_0,
            validation_manager_1=validation_manager_1
        )
    elif version == 2:
        explainer = OptimizationCyCE(
            reference_model=reference_model,
            training_config=training_config,
            model_config=model_config,
            training_manager_0=training_manager_0,
            training_manager_1=training_manager_1,
            validation_manager_0=validation_manager_0,
            validation_manager_1=validation_manager_1
        )

    else:
        raise NotImplementedError

    explainer.train()


if __name__ == "__main__":
    main()
