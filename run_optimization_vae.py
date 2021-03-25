import json
import configargparse
from src.utils.instantiate_model import instantiate_model
from src.utils.data_generator import set_data_generator
from src.optimization.optimization_vae import OptimizationVAE


def main():
    p = configargparse.ArgParser()
    p.add_argument('--config-file', required=True)
    p.add_argument('--ref-model-settings-json-path', required=True)
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

    vae = OptimizationVAE(
        reference_model=reference_model,
        training_config=training_config,
        model_config=model_config,
        training_manager=training_manager,
        validation_manager=validation_manager
    )

    vae.train()


if __name__ == "__main__":
    main()
