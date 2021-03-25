import h5py
from tqdm import tqdm
import configargparse
import numpy as np
import json
from src.utils.instantiate_model import instantiate_model


def main():
    p = configargparse.ArgParser()
    p.add_argument('--config-file', required=True)
    p.add_argument('--indexes-name', required=False, default="test_indexes")
    args = p.parse_args()
    with open(args.config_file, "r") as f:
        config = json.load(f)

    encoder = instantiate_model(
        model_path=config["vae_model_path"]
    )

    with open(config["split_path"], "r") as f:
        split = json.load(f)
        reference_list = split[args.indexes_name]

    if "label_access" not in list(config.keys()):
        label_access = "x_real/label"
    else:
        label_access = config["label_access"]

    if "pred_access" not in list(config.keys()):
        pred_access = "x_real/prediction"
    else:
        pred_access = config["pred_access"]

    if "name_list" not in list(config.keys()):
        name_list = ["x_real", "x_ours_a", "x_ours_st", "x_mg_a", "x_sa_a", "x_sa_st"]
    else:
        name_list = config["name_list"]

    explanation_file = h5py.File(config["interpretation_path"], 'r')
    results = h5py.File(config["embeddings_path"], "w")
    for ref in tqdm(reference_list):
        group = f"data/{ref[0]}/{ref[1]}"
        label = explanation_file[group + label_access][()]
        prediction = explanation_file[group + pred_access][()]
        results.create_dataset(group + "/x_real/label", data=label)
        results.create_dataset(group + "/x_real/prediction", data=prediction)
        for name in name_list:
            x = explanation_file[group + f"/{name}/data"][()]
            if encoder.input_shape[1] == 3:
                x = np.array([[x, x, x]])
            elif encoder.input_shape[1] == 1:
                x = np.array([[x]])
            mu, _ = encoder(x)
            results.create_dataset(group + f"/{name}/mu", data=mu.numpy()[0])

    explanation_file.close()
    results.close()


if __name__ == '__main__':
    main()
